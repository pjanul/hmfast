import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici, erf 
from hmfast.base_tracer import BaseTracer, HankelTransform
from hmfast.emulator_eval import Emulator
from hmfast.halo_model import HaloModel
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path
import os

import numpy as np # it may be a good idea to eventually remove numpy dependence altogether
_eps = 1e-30

jax.config.update("jax_enable_x64", True)



class CMBLensingTracer(BaseTracer):
    """
    CMB lensing tracer. Implements the formalism described in Kusiak et al (2023)
    Link to paper: https://arxiv.org/pdf/2203.12583

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
     x : array
        The x array used to define the radial profile over which the tracer will be evaluated
    """

    def __init__(self, cosmo_model=0, x=None):        
        
        if x is None:
            x = jnp.logspace(jnp.log10(1e-4), jnp.log10(20.0), 512)
            
        self.hankel = HankelTransform(x, nu=0.5)
        self.x = x
        self.emulator = Emulator(cosmo_model=cosmo_model)
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")
        self.halo_model = HaloModel(cosmo_model=cosmo_model)  # Eventually want to allow the user to pass hmf prescription (e.g. T08)


    def get_W_kappa_cmb(self, z, params=None):
        """
        Compute the CMB lensing kernel W_kappa_cmb at redshift z.
        """
        # Merge default parameters with input
        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params=params)
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))  # Ensure z is an array

        c_km_s = 299792.458  # Speed of light in km/s
        c_m_s = 299792458 

        
        # Cosmological constants
        H0 = params["H0"]  # Hubble constant in km/s/Mpc
        Omega_m = cparams["Omega0_m"]  # Matter density parameter
        #print("Rho_crit_0", cparams["Rho_crit_0"])
        h = H0 / 100
        # Compute comoving distance and Hubble parameter
        chi_z = self.emulator.get_angular_distance_at_z(zq, params=params) * (1 + zq) * h # Comoving distance in Mpc/h
        H_z = self.emulator.get_hubble_at_z(zq, params=params)   # Hubble parameter in km/s/Mpc
    
        # Placeholder for chi(z*) (CMB distance)
        chi_z_cmb = 13860.0 * h # Approximate comoving distance to the last scattering surface (z ~ 1090) in Mpc/h
    
        # Compute the CMB lensing kernel
        W_kappa_cmb =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            ((chi_z_cmb - chi_z) / chi_z_cmb)
        )

       
        return W_kappa_cmb 
        
            

    def c_Duffy2008(self, z, m, A=5.71, B=-0.084, C=-0.47, M_pivot=2e12):
        """
        Duffy et al. 2008 mass-concentration relation.
        A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
        """
        return A * (m / M_pivot)**B * (1 + z)**C
     
    def get_u_m_ell(self, z, m, params = None):
        """
        This function calculates u_ell^m(z, M) via the analytic method described in Kusiak et al (2023).
        """
        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params)
        #jax.debug.print("M / Rho_crit_0: {}",  m / cparams["Rho_crit_0"])
        m = jnp.atleast_1d(m) 
        h = params["H0"]/100
        x_min = 1e-8
        x = jnp.clip(self.x, a_min=x_min)

        c_200c = self.c_Duffy2008(z, m)
        r_200c = self.emulator.get_r_delta_of_m_delta_at_z(200, m, z, params=params) 
        lambda_val = params.get("lambda_HOD", 1.0) 

       
        dummy_profile = jnp.ones_like(x)
        k, _ = self.hankel.transform(dummy_profile)

        rho_crit_0 = cparams["Rho_crit_0"] 

        m_over_rho_crit = jnp.broadcast_to((m / rho_crit_0)[:, None], (m.shape[0], k.shape[0])) 
        
        chi = self.emulator.get_angular_distance_at_z(z, params=params) * (1.0 + z) * h
        ell = k * chi - 0.5
       
        ell = jnp.broadcast_to(ell[None, :], (m.shape[0], k.shape[0]))    # (N_m, N_k)
        
  
        k_mat = k[None, :]                            # (1, N_k)
        r_mat = r_200c[:, None]                       # (N_m, 1)
        c_mat = jnp.atleast_1d(c_200c)[:, None]       # (N_m, 1)
                
        q = k_mat * r_mat / c_mat * (1+z)            # (N_m, N_k)
        q_scaled = (1 + lambda_val * c_mat) * q

        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x/(1 + x))
        f_nfw_val = f_nfw(lambda_val * c_mat)

        
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        
        u_ell_m =  (   jnp.cos(q) * (Ci_q_scaled - Ci_q) 
                    +  jnp.sin(q) * (Si_q_scaled - Si_q) 
                    -  jnp.sin(lambda_val * c_mat * q) / q_scaled ) * f_nfw_val * m_over_rho_crit 
        
     
        return ell, u_ell_m



    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CMB lensing tracer u_ell.
        For CMB lensing:, 
            First moment:     W_k_cmb * u_ell_m
            Second moment:    W_k_cmb^2 * u_ell_m^2 
        """

        params = merge_with_defaults(params)
        W = self.get_W_kappa_cmb(z, params=params) 
        ell, u_m = self.get_u_m_ell(z, m, params=params)

        h = params["H0"]/100
    
        moment_funcs = [
            lambda _:  W[:, None] * u_m ,
            lambda _: (W**2)[:, None] * u_m**2,
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell
  
  