import jax
import jax.numpy as jnp
from hmfast.tools.hankel import HankelTransform
from hmfast.halo_model import HaloModel
from hmfast.ede_emulator import EDEEmulator
from functools import partial


#@jax.jit
def gnfw_pressure_profile(x, z, m, emulator, params = None):
    """
    GNFW pressure profile as a function of dimensionless scaled radius x = r/r500.
    """ 
    # Pull needed parameters
    H0, P0, c500, alpha, beta, gamma, B = (params[k] for k in ("H0", "P0GNFW", "c500", "alphaGNFW", "betaGNFW", "gammaGNFW", "B"))

    # Compute helper variables and the final value of Pe
    h = H0 / 100.0 

    H = emulator.get_hubble_at_z(z, params_values_dict=params) * 299792.458  # multiply by speed of light in km/s 
    m_delta_tilde = (m / B) # convert to M_sun 
    C = 1.65 * (h / 0.7)**2 * (H / H0)**(8 / 3) * (m_delta_tilde / (0.7 * 3e14))**(2 / 3 + 0.12) * (0.7/h)**1.5 # eV cm^-3
    scaled_x = c500 * x
    Pe = C * P0 * scaled_x**(-gamma) * (1 + scaled_x**alpha)**((gamma - beta) / alpha)
   
    return Pe



class TSZTracer:
    def __init__(self, emulator, params):
        self.params = params
        x_min, x_max, x_npoints = self.params["x_min"], self.params["x_max"], self.params["x_npoints"]
        self.x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), num=x_npoints)
        self.hankel = HankelTransform(x_min=x_min, x_max=x_max, x_npoints=x_npoints, nu=0.5)
        self.emulator = emulator
        
    #@partial(jax.jit, static_argnums=(0,))
    def _compute_r_and_ell(self, z, m):
        h, B, delta = self.params['H0']/100, self.params['B'], self.params['delta']

        d_A = self.emulator.get_angular_distance_at_z(z, params_values_dict=self.params) * h
        r_delta = self.emulator.get_r_delta_of_m_delta_at_z(delta, m, z, params_values_dict=self.params) / (B**(1/3))  

        ell_delta = d_A / r_delta
        return r_delta, ell_delta
        

    #@partial(jax.jit, static_argnums=(0,))
    def hankel_integrand(self, x, z, m):
        """
        Compute x^0.5 * Pe(x) * W(x).
        Handles x and m as arrays using vmap for vectorization.
        """
        x_min, x_max = self.params['x_min'], self.params['x_max']
        W_x = jnp.where((x >= x_min) & (x <= x_max), 1.0, 0.0) # Window function

        def single_m(m_val):
            Pe = gnfw_pressure_profile(x, z, m_val, self.emulator, params=self.params)
            return x**0.5 * Pe * W_x

        return jax.vmap(single_m)(m)  # Shape: (len(m), len(x))



    #@partial(jax.jit, static_argnums=(0,))
    def compute_y_ell(self, z, m):

        # Define: h, electron mass in eV/c^2; Thompson cross-section in cm^2;  1 Mpc/h in cm
        h, x_min, x_max, delta = self.params['H0'] / 100, self.params['x_min'], self.params['x_max'], self.params['delta']
        m_e, sigma_T, mpc_per_h_to_cm = 510998.95, 6.6524587321e-25, 3.085677581e24 / h  

        # Compute prefactor and Hankel integrand
        r_delta, ell_delta = self._compute_r_and_ell(z, m)
        prefactor = sigma_T / m_e  * 4 * jnp.pi * r_delta * mpc_per_h_to_cm / (ell_delta**2)
        integrand = self.hankel_integrand(self.x_grid, z, m)
        
        # Use the pre-compiled jitted Hankel transform.
        k, y_k = self.hankel._hankel_jit(integrand)  # k = ell/ell_delta
    
        # Compute ell and combine to get y_ell
        ell = k[None, :] * ell_delta[:, None]
        y_ell = prefactor[:, None] * y_k * jnp.sqrt(jnp.pi / (2 * k[None, :]))

        return ell, y_ell

    


    
