import jax
import jax.numpy as jnp

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.tools.constants import Const



class TSZTracer(BaseTracer):
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
        

    def gnfw_pressure_profile(self, z, m, params = None):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        """ 
        params = merge_with_defaults(params)
        x = self.x
    
        # Pull needed parameters
        H0, P0, alpha, beta, gamma, B = (params[k] for k in ("H0", "P0GNFW", "alphaGNFW", "betaGNFW", "gammaGNFW", "B")) 
        c_delta = self.halo_model.c_delta(z, m, params=params) 
        
        # Compute helper variables and the final value of Pe
        h = H0 / 100.0 
        H = self.halo_model.emulator.hubble_parameter(z, params=params) * 299792.458  # multiply by speed of light in km/s 
        m_delta_tilde = (m / B) # convert to M_sun 
        C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * (m_delta_tilde / (0.7 * 3e14))**(2 / 3 + 0.12) * (0.7/h)**1.5 # eV cm^-3
        scaled_x = c_delta * x
        Pe = C * P0 * scaled_x**(-gamma) * (1 + scaled_x**alpha)**((gamma - beta) / alpha)
       
        return Pe
        

    def get_prefactor(self, z, m, params=None):
        """
        Compute tSZ prefactor.
        """
        params = merge_with_defaults(params)
        h, B = params['H0']/100, params['B']
        delta = self.halo_model.delta
        d_A = self.halo_model.emulator.angular_diameter_distance(z, params=params) * h
        r_delta = self.halo_model.r_delta(z, m, delta, params=params) / B**(1/3)
        ell_delta = d_A / r_delta

        m_e, sigma_T, mpc_per_h_to_cm = 510998.95, 6.6524587321e-25, 3.085677581e24 / h

        prefactor = (sigma_T / m_e) * 4 * jnp.pi * r_delta * mpc_per_h_to_cm / (ell_delta**2)
        return prefactor

  
    def get_u_ell(self, z, m, moment=1, params=None):
        """
        Compute either the first or second moment of the tSZ power spectrum tracer u_ell.
        For tSZ:
            1st moment:  u_ell
            2nd moment:  u_ell^2
        """
        
        params = merge_with_defaults(params)
       
         # Get prefactor and perform Hankel transform from BaseTracer
        prefactor = self.get_prefactor(z, m, params=params)
        ell, u_ell = self.u_ell_hankel(z, m, self.x, params=params)

        # For tSZ, we need to correct with the hydrostatic mass bias factor B 
        ell *= params['B']**(1/3)
        u_ell_base = prefactor[:, None] * u_ell 
    
        # Select moment using JAX-safe branching
        moment_funcs = [
            lambda _: u_ell_base,          # moment = 1
            lambda _: u_ell_base**2,       # moment = 2
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell
