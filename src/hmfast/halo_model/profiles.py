import jax
import jax.numpy as jnp
import mcfit
import functools
from jax.scipy.special import sici
from jax.tree_util import register_pytree_node_class

from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const
from hmfast.halo_model.mass_definition import MassDefinition


class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """
    def __init__(self, x, nu=0.5):
        
        self._hankel = mcfit.Hankel(x, nu=nu, lowring=True, backend='jax')
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


class HaloProfile:
    def u_k_hankel(self, halo_model, x, m, z, params=None):
        """
        Hankel-transform a 3D halo/tracer profile to u_ell for halo model use.
    
        Parameters
        ----------
        x : arrat like
            Radius r scaled by the scale radius x = r / r_s
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        k : array_like, optional
            k values over which the hankel transform will be evaluated. 
            If None, the transform's natural k grid will be output.
            If not None, the transform will be inteprolated to match this k
        params : dict, optional
            Parameter dictionary

        Returns ell, u_ell_m
    
        """

        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params=params)
        h = params['H0']/100
       
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m_z(m_val, z_val):
            profile = jnp.squeeze(self.profile(halo_model, x, m_val, z_val, params=params))  # remove extra axes
            return profile * x**0.5 * W_x  # shape (Nx,)

        hankel_integrand = jax.vmap(jax.vmap(single_m_z, in_axes=(None, 0)), in_axes=(0, None) )(m, z)
            
        # We need u_k_native to have shape (Nx, Nm, Nz)
        k_native, u_k_native = self._hankel.transform(hankel_integrand)
        u_k_native = jnp.swapaxes(u_k_native, 2, 0)
        u_k_native = jnp.swapaxes(u_k_native, 2, 1)
 
        return k_native, u_k_native


class MatterProfile(HaloProfile):
    pass

class DensityProfile(HaloProfile):
    pass

class PressureProfile(HaloProfile):
    pass



class B16DensityProfile(DensityProfile):
    def __init__(self, x=None, model="agn"):
        # Grid initialization (triggers the x.setter)
        self.x = x if x is not None else jnp.logspace(-4, 2, 256)
        # Model initialization (triggers the model.setter)
        self.model = model

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        """
        Updates physics attributes by switching between 'agn' and 'shock'.
        Strictly case-insensitive and validates input.
        """
        val_lower = value.lower()
        if val_lower not in ["agn", "shock"]:
            raise ValueError(f"Invalid model '{value}'. Must be 'agn' or 'shock'.")
            
        self._model = val_lower
        b16_configs = self._get_model_configs(self._model)
        
        # Physics parameters reset in throuples
        self.A_rho0, self.A_alpha, self.A_beta = b16_configs['A_rho0'], b16_configs['A_alpha'], b16_configs['A_beta']
        self.alpha_m_rho0, self.alpha_m_alpha, self.alpha_m_beta = b16_configs['alpha_m_rho0'], b16_configs['alpha_m_alpha'], b16_configs['alpha_m_beta']
        self.alpha_z_rho0, self.alpha_z_alpha, self.alpha_z_beta = b16_configs['alpha_z_rho0'], b16_configs['alpha_z_alpha'], b16_configs['alpha_z_beta']

    def _get_model_configs(self, model_key):
        """Internal lookup for Battaglia 2016 Table 2 parameters."""
        AGN = {
            'A_rho0': 4000.0, 'A_alpha': 0.88, 'A_beta': 3.83,
            'alpha_m_rho0': 0.29, 'alpha_m_alpha': -0.03, 'alpha_m_beta': 0.04,
            'alpha_z_rho0': -0.66, 'alpha_z_alpha': 0.19, 'alpha_z_beta': -0.025
        }
        SHOCK = {
            'A_rho0': 1.9e4, 'A_alpha': 0.70, 'A_beta': 4.43,
            'alpha_m_rho0': 0.09, 'alpha_m_alpha': -0.017, 'alpha_m_beta': 0.005,
            'alpha_z_rho0': -0.95, 'alpha_z_alpha': 0.27, 'alpha_z_beta': 0.037
        }
        return SHOCK if model_key == "shock" else AGN


    def profile(self, halo_model, x, m, z, params=None):
        """
        Battaglia et al. 2016 gas density profile (AGN feedback model).
        Fully vectorized to support:
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        h = cparams["h"]

        gamma = -0.2
        xc = 0.5
        
        # Ensure 1D and setup broadcasting shapes
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m),  jnp.atleast_1d(z)  # (Nx,)
        x_b, m_b, z_b = x[:, None, None], m[None, :, None], z[None, None, :]      # (Nx, 1, 1), (1, Nm, 1), (1, 1, Nz)
        
        # Critical density broadcast to (1, 1, Nz)
        rho_crit_z = jnp.atleast_1d(halo_model.emulator.critical_density(z, params=params))[None, None, :]
        
        # Mass scaling logic
        m_200c_msun = m_b / h
        mass_ratio = m_200c_msun / 1e14 
       
        # Compute Shape Parameters (Equations A1, A2 from B16)
        rho0 = self.A_rho0 * mass_ratio**self.alpha_m_rho0 * (1 + z_b)**self.alpha_z_rho0 
        alpha = self.A_alpha * mass_ratio**self.alpha_m_alpha * (1 + z_b)**self.alpha_z_alpha 
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta 
        
        # Profile Shape Function (Nx, Nm, Nz)
        p_x = (x_b / xc)**gamma * (1 + (x_b / xc)**alpha)**(-(beta + gamma) / alpha)
        
        # Final result: M_sun h^2 / Mpc^3 
        rho_gas = rho0 * rho_crit_z * f_b * p_x 
        
        return rho_gas

        

class NFWDensityProfile(DensityProfile):
    def __init__(self, x=None):
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-4), jnp.log10(1.0), 256)
    

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        """
        Whenever x is modified, immediately rebuild the hankel transform object
        """
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)
        

    def profile(self, halo_model, x, m, z, params=None):
        params = merge_with_defaults(params)
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
       
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Get scale radius r_s
        r_delta = halo_model.r_delta(m, z, params=params)
        c_delta = halo_model.c_delta(m, z, params=params)
        r_s = r_delta / c_delta # (Nm, Nz)
        
        # Calculate rho_s
        m_nfw = jnp.log(1 + c_delta) - c_delta / (1 + c_delta) # (Nm, Nz)
        rho_s = m[:, None] / (4 * jnp.pi * r_s**3 * m_nfw)    # (Nm, Nz)
        
        # Final broadcast to (Nx, Nm, Nz)
        # x needs to be (Nx, 1, 1) and rho_s (1, Nm, Nz)
        rho_gas = f_b * rho_s[None, :, :] / (x[:, None, None] * (1 + x[:, None, None])**2)
        
        return rho_gas
   

class GNFWPressureProfile(PressureProfile):
    def __init__(self, x=None):
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-5), jnp.log10(4.0), 256) 


    @property
    def x(self):
       return self._x

    @x.setter
    def x(self, value):
        """
        Whenever x is modified, immediately rebuild the hankel transform object
        """
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)

    def profile(self, halo_model, x, m, z, params=None):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        
        Fully vectorized: supports
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
       
    
        # Retrieve all required parameters and ensure all inputs are 1D  
        params = merge_with_defaults(params)
        H0, P0, alpha, beta, gamma, B = (params[k] for k in ("H0", "P0_GNFW", "alpha_GNFW", "beta_GNFW", "gamma_GNFW", "B"))
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z) 
       
        # Helper variables for normalization
        h = H0 / 100.0
        c_km_s = Const._c_ / 1e3
        H = halo_model.emulator.hubble_parameter(z, params=params) * c_km_s  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)

        # Corrected mass given the hydrostatic mass bias
        m_delta_tilde = (m / B)[None, :, None]  # (1, Nm, 1)
    
        P_500c = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m_delta_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # Scaled radius and GNFW formula
        c_delta = halo_model.c_delta(m, z, params=params)  # (Nm, Nz)
        scaled_x = c_delta[None, :, :] * x[:, None, None]   # (Nx, Nm, Nz)
        Pe = P_500c * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return Pe  # shape: (Nx, Nm, Nz)




class NFWMatterProfile(MatterProfile):
    def __init__(self):
        pass

    def u_k_matter(self, halo_model, k, m, z, params=None):
        """
        Calculate u^m(k, M, z) supporting independent dimensions for k, m, and z.
        
        Returns u_k_m with shape (N_k, N_m, N_z).
        """
        params = merge_with_defaults(params)
        
        # Ensure all inputs are 1D arrays
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Get c_delta and r_delta
        c_delta = halo_model.c_delta(m, z, params=params)
        r_delta = halo_model.r_delta(m, z, params=params)
        lambda_val = 1.0 
        
        # Compute analytical profile q terms with shape: (N_k, N_m, N_z)
        q = k[:, None, None] * r_delta[None, :, :] / c_delta[None, :, :] * (1 + z[None, None, :])
        q_scaled = (1 + lambda_val * c_delta[None, :, :]) * q
        
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        
        # NFW normalization
        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x / (1 + x))
        f_nfw_val = f_nfw(lambda_val * c_delta)
        f_nfw_val = f_nfw_val[None, :, :]  
        
        # Fourier-space profile calculation
        u_k_m = (jnp.cos(q) * (Ci_q_scaled - Ci_q)
                   + jnp.sin(q) * (Si_q_scaled - Si_q)
                   - jnp.sin(lambda_val * c_delta[None,:,:] * q) / q_scaled) * f_nfw_val 
    
        return k, u_k_m
    
