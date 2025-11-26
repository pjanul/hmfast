import jax
import jax.numpy as jnp
from hmfast.halo_model import HaloModel
from hmfast.emulator_eval import CosmoEmulator
from functools import partial
from hmfast.base_tracer import BaseTracer


#@jax.jit
def gnfw_pressure_profile(x, z, m, emulator, params = None):
    """
    GNFW pressure profile as a function of dimensionless scaled radius x = r/r500.
    """ 
    # Pull needed parameters
    H0, P0, c500, alpha, beta, gamma, B = (params[k] for k in ("H0", "P0GNFW", "c500", "alphaGNFW", "betaGNFW", "gammaGNFW", "B"))

    # Compute helper variables and the final value of Pe
    h = H0 / 100.0 

    H = emulator.get_hubble_at_z(z, params=params) * 299792.458  # multiply by speed of light in km/s 
    m_delta_tilde = (m / B) # convert to M_sun 
    C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * (m_delta_tilde / (0.7 * 3e14))**(2 / 3 + 0.12) * (0.7/h)**1.5 # eV cm^-3
    scaled_x = c500 * x
    Pe = C * P0 * scaled_x**(-gamma) * (1 + scaled_x**alpha)**((gamma - beta) / alpha)
   
    return Pe


class TSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    Fully continuous tracer: local_contribution = 0, profile_contribution = 1
    """
    def __init__(self, emulator, params):
        super().__init__(params)
        self.emulator = emulator

    def _compute_r_and_ell(self, z, m):
        """
        Helper to compute r_delta and ell_delta for each halo.
        """
        h, B, delta = self.params['H0']/100, self.params['B'], self.params['delta']
        d_A = self.emulator.get_angular_distance_at_z(z, params=self.params) * h
        r_delta = self.emulator.get_r_delta_of_m_delta_at_z(delta, m, z, params=self.params) / B**(1/3)
        ell_delta = d_A / r_delta
        return r_delta, ell_delta


    def get_contributions(self, z, m):
        """
        tSZ is a continuous tracer: local_contribution = 0, profile_contribution = 1
        """
        N = jnp.atleast_1d(m)
        local_contribution = jnp.zeros_like(N)
        profile_contribution = jnp.ones_like(N)
        return local_contribution, profile_contribution

    def get_prefactor(self, z, m):
        """
        Compute tSZ prefactor.
        """
        r_delta, ell_delta = self._compute_r_and_ell(z, m)
        h = self.params['H0'] / 100
        m_e, sigma_T, mpc_per_h_to_cm = 510998.95, 6.6524587321e-25, 3.085677581e24 / h

        prefactor = (sigma_T / m_e) * 4 * jnp.pi * r_delta * mpc_per_h_to_cm / (ell_delta**2)
        return prefactor, ell_delta

    def get_hankel_integrand(self, x, z, m):
        x_min, x_max = self.params['x_min'], self.params['x_max']
        W_x = jnp.where((x >= x_min) & (x <= x_max), 1.0, 0.0)

        def single_m(m_val):
            Pe = gnfw_pressure_profile(x, z, m_val, self.emulator, params=self.params)
            return x**0.5 * Pe * W_x

        return jax.vmap(single_m)(m)
