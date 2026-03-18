import os
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path
from hmfast.utils import Const

jax.config.update("jax_enable_x64", True)



class GalaxyLensingTracer(BaseTracer):
    """
    Galaxy lensing tracer. Implements the formalism described in Kusiak et al (2023)
    Link to paper: https://arxiv.org/pdf/2203.12583

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
     x : array
        The x array used to define the radial profile over which the tracer will be evaluated
    """

    
    def __init__(self, halo_model, dndz_source=None, dndz_lens=None):        

        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")

        if dndz_source is None:
            # Call _load_dndz_data from BaseTracer
            dndz_source_path = os.path.join(get_default_data_path(), "auxiliary_files", "nz_source_normalized_bin4.txt")
            self.dndz_source = self._load_dndz_data(dndz_source_path)
        else:
            self.dndz_source = dndz_source

        if dndz_lens is None:
            # Call _load_dndz_data from BaseTracer
            dndz_lens_path = os.path.join(get_default_data_path(), "auxiliary_files", "nz_lens_bin1.txt")
            self.dndz_lens = self._load_dndz_data(dndz_lens_path)
        else: 
            self.dndz_lens = dndz_lens

    @property
    def dndz_source(self):
        return self._dndz_source_data

    @dndz_source.setter
    def dndz_source(self, value):
        self._dndz_source_data = self._normalize_dndz(value)

    @property
    def dndz_lens(self):
        return self._dndz_lens_data

    @dndz_lens.setter
    def dndz_lens(self, value):
        self._dndz_lens_data = self._normalize_dndz(value)

    
    def I_g(self, z, params=None):
        """
        Return I_g at requested z.
        Uses pre-loaded dndz_source_data = [z, phi_prime].
        Integrates only over sources behind the lens (z_s > z).
        """
        
        params = merge_with_defaults(params)
        z = jnp.atleast_1d(z)
        h = params["H0"] / 100
        
        # Load source distribution and lens redshift distribution
        z_data, phi_prime_data = self.dndz_source
        z_s_data, n_s_data = self.dndz_lens
    
        # Interpolate phi_prime to z_s grid
        phi_prime_at_z_s = jnp.interp(z_s_data, z_data, phi_prime_data, left=0.0, right=0.0)
    
        # Angular distances
        chi_z_s = self.halo_model.emulator.angular_diameter_distance(z_s_data, params=params) * (1 + z_s_data) * h
        chi_z = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) * h
    
        # Reshape for broadcasting
        chi_z_s = chi_z_s[:, None]  # (N_s, 1)
        chi_z = chi_z[None, :]      # (1, N_z)
    
        # Lensing factor
        chi_diff = (chi_z_s - chi_z) / chi_z_s
    
        # Mask: only include sources behind the lens
        mask = (z_s_data[:, None] > z[None, :])  # (N_s, N_z)
        chi_diff_masked = chi_diff * mask
    
        # Integrate over z_s using trapezoid
        I_g = jnp.trapezoid(phi_prime_at_z_s[:, None] * chi_diff_masked, x=z_s_data, axis=0)
    
        return I_g



    def kernel(self, z, params=None):
        """
        Compute the galaxy lensing kernel W_kappa_g at redshift z.
        """
        # Merge default parameters with input
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params=params)
        z = jnp.atleast_1d(z) # Ensure z is an array

        c_km_s = Const._c_ / 1e3  # Speed of light in km/s
       
        # Cosmological constants
        H0 = params["H0"]  # Hubble constant in km/s/Mpc
        h = H0 / 100
        Omega_m = cparams["Omega0_m"]  # Matter density parameter

        # Compute comoving distance and Hubble parameter
        chi_z = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) * h # Comoving distance in Mpc/h
        H_z = self.halo_model.emulator.hubble_parameter(z, params=params)   # Hubble parameter in km/s/Mpc
    
        I_g = self.I_g(z, params=params) 
    
        # Compute the CMB lensing kernel
        W_kappa_g =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            I_g 
        ) 
    
        return W_kappa_g 


    def u_k(self, k, m, z, moment=1, params=None):
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)

        k, u_m = self.u_k_matter(k, m, z, params=params)
    
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)
        u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _: u_m,
            lambda _: u_m**2,
        ]
        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
        return k, u_k
      
  


       
