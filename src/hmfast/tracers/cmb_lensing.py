import os
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici, erf 

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path
from hmfast.literature import c_D08

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

    def __init__(self, halo_model=HaloModel()):        
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        self.halo_model.emulator._load_emulator("DER")

    def get_W_kappa_cmb(self, z, params=None):
        """
        Compute the CMB lensing kernel W_kappa_cmb at redshift z.
        """
        # Merge default parameters with input
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params=params)
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))  # Ensure z is an array
        
        # Cosmological constants
        H0 = params["H0"]  # Hubble constant in km/s/Mpc
        Omega_m = cparams["Omega0_m"]  # Matter density parameter
        c_km_s = 299792.458  # Speed of light in km/s        
        h = H0 / 100
        
        # Compute comoving distance and Hubble parameter
        chi_z = self.halo_model.emulator.angular_diameter_distance(zq, params=params) * (1 + zq) * h # Comoving distance in Mpc/h
        H_z = self.halo_model.emulator.hubble_parameter(zq, params=params)   # Hubble parameter in km/s/Mpc
        
        # Comoving distance to the last scattering surface (z ~ 1090) in Mpc/h
        chi_z_cmb = self.halo_model.emulator.derived_parameters(params=params)["chi_star"] * h  
        
        # Compute the CMB lensing kernel
        W_kappa_cmb =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            ((chi_z_cmb - chi_z) / chi_z_cmb)
        )    

       
        return W_kappa_cmb 


        
    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CMB lensing tracer u_ell.
        For CMB lensing:, 
            First moment:     W_k_cmb * u_ell_m
            Second moment:    W_k_cmb^2 * u_ell_m^2 
        """

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        W = self.get_W_kappa_cmb(z, params=params) 

        # Compute u_m_ell from BaseTracer
        ell, u_m = self.u_ell_analytic(z, m, params=params)

        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"] / cparams["h"]**2  
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _:  W[:, None] * u_m ,
            lambda _: (W**2)[:, None] * u_m**2,
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell
  
  