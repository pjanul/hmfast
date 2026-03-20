import os
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
    halo_model : 
        Halo model used to compute relevant quantities
    dndz :
        The redshift distribution of the galaxy population. This distribution will be normalized if it is not already done.
    """

    def __init__(self, halo_model, dndz = None):        
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")

        if dndz is None:
            # Call _load_dndz_data from BaseTracer
            dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "normalised_dndz_cosmos_0.txt")
            dndz = self._load_dndz_data(dndz_path)  

        self.dndz = dndz

    @property
    def has_central_contribution(self):
        return True

    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._normalize_dndz(value)
        
        
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
        z: redshift
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
        ng_bar = jnp.trapezoid(integrand, x=logm, axis=0)

        # Add the halo model consistency counter terms if hm_consistency is set to True, otherwise do not change anything
        ng_bar = jax.lax.cond(self.halo_model.hm_consistency, lambda x: x + self.halo_model.counter_terms(m, z, params=params)[0] * Ntot[0], lambda x: x, ng_bar)


        return ng_bar
        

    def kernel(self, z, params=None):
        """
        Return Wg_grid at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        """
        
        params = merge_with_defaults(params)
        z = jnp.atleast_1d(z)
    
        # Extract precomputed phi_prime
        z_g, phi_prime_g = self.dndz
    
        # Interpolate phi_prime to requested z
        phi_prime_g_at_z = jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)
    
        H_grid = self.halo_model.emulator.hubble_parameter(z, params=params)  # 1/Mpc
        chi_grid = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1.0 + z)  # Mpc comov

        # Assemble W_g on the grid
        W_g = H_grid * (phi_prime_g_at_z / chi_grid**2)
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



    def sat_and_cen_contribution(self, k, m, z, params=None):
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

        sat_term = (1/ng) * (Ns[None, :, None] * u_m)
        cen_term = (1/ng) * (Nc[None, :, None]**0)
    
        return sat_term, cen_term
  
        