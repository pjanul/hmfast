import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const

jax.config.update("jax_enable_x64", True)

class tSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, halo_model, x=None):

        # Set tracer parameters
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-4), jnp.log10(20.0), 512)
        self.hankel = HankelTransform(self.x, nu=0.5)
        self.profile = self.gnfw_pressure_profile
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        

    def gnfw_pressure_profile(self, z, m, x, params = None):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        """ 
        params = merge_with_defaults(params)
        
    
        # Pull needed parameters
        H0, P0, alpha, beta, gamma, B = (params[k] for k in ("H0", "P0GNFW", "alphaGNFW", "betaGNFW", "gammaGNFW", "B")) 
        c_delta = self.halo_model.c_delta(z, m, params=params) 
        
        # Compute helper variables and the final value of Pe
        h = H0 / 100.0 
        c_km_s = Const._c_ / 1e3
        H = self.halo_model.emulator.hubble_parameter(z, params=params) * c_km_s  # multiply by speed of light in km/s 
        m_delta_tilde = (m / B) # convert to M_sun 
        C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * (m_delta_tilde / (0.7 * 3e14))**(2 / 3 + 0.12) * (0.7/h)**1.5 # eV cm^-3
        scaled_x = c_delta * x
        Pe = C * P0 * scaled_x**(-gamma) * (1 + scaled_x**alpha)**((gamma - beta) / alpha)
       
        return Pe
        

    def prefactor(self, z, m, params=None):
        """
        Compute tSZ prefactor.
        """
        params = merge_with_defaults(params)
        h, B = params['H0']/100, params['B']
        delta = self.halo_model.delta
        d_A = self.halo_model.emulator.angular_diameter_distance(z, params=params) * h
        r_delta = self.halo_model.r_delta(z, m, delta, params=params) / B**(1/3)
        ell_delta = d_A / r_delta

        # Get electon mass in eV, Thomson cross section in cm^2, and Mpc/h in cm
        m_e = Const._m_e_ * Const._c_**2 / Const._eV_
        sigma_T = Const._sigma_T_ * 1e6
        mpc_per_h_to_cm =  Const._Mpc_over_m_ / h

        # Define prefactor and return it
        prefactor = (1 + z) * 4 * jnp.pi * r_delta * mpc_per_h_to_cm / (ell_delta**2)
        return prefactor

    def kernel(self, z, params=None):
        params = merge_with_defaults(params)
        h = params['H0']/100
        
        # Get electon mass in eV, Thomson cross section in cm^2, and Mpc/h in cm
        m_e = Const._m_e_ * Const._c_**2 / Const._eV_
        sigma_T = Const._sigma_T_ * 1e6
        mpc_per_h_to_cm =  Const._Mpc_over_m_ / h
        return (sigma_T / m_e) / (1+z) # Check this
    
    def get_u_ell(self, z, m, ell, moment=1, params=None):
        """
        Compute either the first or second moment of the tSZ power spectrum tracer u_ell.
        For tSZ:
            1st moment:  u_ell
            2nd moment:  u_ell^2
        """
        params = merge_with_defaults(params)
        prefactor = self.prefactor(z, m, params=params)
        k_native, u_k_native = self.u_k_hankel(z, m, params=params)

        delta = self.halo_model.delta 
        d_A = self.halo_model.emulator.angular_diameter_distance(z, params=params) * params['H0'] / 100
        r_delta = self.halo_model.r_delta(z, m, delta, params=params) 
        ell_delta = d_A / r_delta


        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[None, :]))
        ell_native = k_native[None, :] * ell_delta[:, None] 
        
    
        ell_native *= params['B']**(1/3)
        u_ell_base = prefactor[:, None] * u_ell_native
    
        moment_funcs = [
            lambda _: u_ell_base,
            lambda _: u_ell_base**2,
        ]
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        # Interpolate onto input ell for all masses
        def interpolate_single(ell_row, u_ell_row):
            interpolator = jscipy.interpolate.RegularGridInterpolator((ell_row,), u_ell_row, method='linear', bounds_error=False, fill_value=None)
            return interpolator(ell)
        u_ell_interp = jax.vmap(interpolate_single)(ell_native, u_ell)
    
        return ell, u_ell_interp

