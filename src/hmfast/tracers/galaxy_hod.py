import os
import numpy as np # it may be a good idea to eventually remove numpy dependence altogether, but now we need it for np.loadtxt
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici, erf 

from hmfast.tracers.base_tracer import BaseTracer
from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path

jax.config.update("jax_enable_x64", True)



class GalaxyHODTracer(BaseTracer):
    """
    Galaxy HOD tracer implementing central + satellite occupation and
    NFW satellites. Implements the formalism described in Kusiak et al (2023)
    Link to paper: https://arxiv.org/pdf/2203.12583

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
    x : array
        The x array used to define the radial profile over which the tracer will be evaluated
    """

    def __init__(self, halo_model, dndz = None):        
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")

        if dndz is None:
            dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "normalised_dndz_cosmos_0.txt")
            dndz = self.load_file_data(dndz_path)

        self.dndz = dndz

    @property
    def dndz(self):
        """Access the normalized dndz: tracer.dndz"""
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        """
        Intercepts any attempt to set dndz (initially or later) and forces it to be a normalized JAX array tuple.
        """
        z_raw = jnp.atleast_1d(jnp.array(value[0]))
        phi_raw = jnp.atleast_1d(jnp.array(value[1]))
        norm = jnp.trapezoid(phi_raw, x=z_raw)
        self._dndz_data = (z_raw, phi_raw / norm)
            

    def load_file_data(self, dndz_path):
        data = np.loadtxt(dndz_path)
        x = data[:, 0]
        y = data[:, 1]
        return (jnp.array(x), jnp.array(y))
        
        
    def n_cen(self, m, params = None):
        """Mean central occupation: shape = M.shape"""

        params = merge_with_defaults(params)
        M_min = params["M_min_HOD"]
        sigma = params["sigma_log10M_HOD"]

        # Set up the input x of the error function and evaluate it
        x = (jnp.log10(m) - jnp.log10(M_min)) / sigma
        return 0.5 * (1.0 + erf(x))

    
    def n_sat(self, m, params = None):
        """Mean satellite occupation: shape = M.shape"""
        params = merge_with_defaults(params)
        M0 = params["M0_HOD"]
        M1p = params["M1_prime_HOD"]
        alpha = params["alpha_s_HOD"]

        # power law only above M0 and use jnp.where to keep differentiability
        pow_term = jnp.maximum((m - M0) / M1p, 0.0)**alpha
        N_c = self.n_cen(m, params = params)
        return  N_c * pow_term
        

    def ng_bar(self, m, z, params = None):
        """
        Compute comoving galaxy number density ng(z) = ∫ dlnM [dn/dlnM] [Nc+Ns].
        halo_model: HaloModel instance
        tracer: GalaxyHODTracer instance (provides HOD via helper funcs)
        z: scalar redshift
        params: parameter dict 
        """
        
        params = merge_with_defaults(params)
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Nc = self.n_cen(m, params=params)
        Ns = self.n_sat(m, params=params)
        Ntot = Nc + Ns

        # shape (N_m, N_z)
        dndlnm = self.halo_model.halo_mass_function(m, z, params=params)
        integrand = dndlnm * Ntot[:, None]

        return jnp.trapezoid(integrand, x=logm, axis=0)
        

    def kernel(self, z, params=None):
        """
        Return Wg_grid at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        """
        
        params = merge_with_defaults(params)
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
    
        # Extract precomputed phi_prime
        z_data, phi_prime_data = self.dndz
    
        # Interpolate phi_prime to requested z
        phi_prime_at_z = jnp.interp(zq, z_data, phi_prime_data, left=0.0, right=0.0)
    
        H_grid = self.halo_model.emulator.hubble_parameter(zq, params=params)  # 1/Mpc
        chi_grid = self.halo_model.emulator.angular_diameter_distance(zq, params=params) * (1.0 + zq)  # Mpc comov

        # Assemble W_g on the grid
        W_g = H_grid * (phi_prime_at_z / chi_grid**2)
        return W_g



    def u_k(self, k, m, z, moment=1, params=None):
        """ 
        Compute either the first or second moment of the galaxy HOD tracer u_ell.
        For galaxy HOD:, 
            First moment:     W_g / ng_bar * [Nc + Ns * u_ell_m]
            Second moment:    W_g^2 / ng_bar^2 * [Ns^2 * u_ell_m^2 + 2 * Ns * u_ell_m]
        You cannot simply take u_ell_g**2.
        """

        params = merge_with_defaults(params)
        Ns = self.n_sat(m, params=params)
        Nc = self.n_cen(m, params=params)
        ng = self.ng_bar(m, z, params=params) * (params["H0"]/100)**3

        _, u_m = self.u_k_matter(k, m, z, params=params)  
    
        moment_funcs = [
            lambda _: (1/ng) * (Nc[None, :, None] + Ns[None, :, None] * u_m),
            lambda _: (1/ng**2) * (Ns[None, :, None]**2 * u_m**2 + 2*Ns[None, :, None] * u_m),
        ]
    
        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k
  
        