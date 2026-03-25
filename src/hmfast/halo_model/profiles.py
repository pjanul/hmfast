import jax
import jax.numpy as jnp
import mcfit
import functools
from jax.scipy.special import sici

from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const


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
        """
        Battaglia et al. 2016 gas density profile (AGN feedback model).
        Fully vectorized to support:
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
        params = merge_with_defaults(params)
        
        # Ensure 1D and setup broadcasting shapes
        x = jnp.atleast_1d(x)  # (Nx,)
        m = jnp.atleast_1d(m)  # (Nm,)
        z = jnp.atleast_1d(z)  # (Nz,)
        
        x_b = x[:, None, None]      # (Nx, 1, 1)
        m_b = m[None, :, None]      # (1, Nm, 1)
        z_b = z[None, None, :]      # (1, 1, Nz)
        
        h = params["H0"] / 100.0
        cparams = halo_model.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Critical density and Concentration
        rho_crit_z = jnp.atleast_1d(halo_model.emulator.critical_density(z, params=params))
        rho_crit_z = rho_crit_z[None, None, :]  # (1, 1, Nz)
        
        c_delta = halo_model.c_delta(m, z, params=params) # (Nm, Nz)
        c_delta = c_delta[None, :, :]  # (1, Nm, Nz)
        
        # Battaglia+16 parameters (Table 2: AGN feedback model)
        # Scaling parameters for rho0, alpha, beta
        A_rho0, A_alpha, A_beta = 4000.0, 0.88, 3.83
        
        # Mass scaling for M > 1e14
        alpha_m_rho0, alpha_m_alpha, alpha_m_beta = 0.29, -0.03, 0.04
        # Redshift scaling
        alpha_z_rho0, alpha_z_alpha, alpha_z_beta = -0.66, 0.19, -0.025
        # Concentration scaling (usually 0 in B16, but kept for completeness)
        alpha_c_rho0, alpha_c_alpha, alpha_c_beta = 0.0, 0.0, 0.0
        
        # Low-mass scaling (M < 1e14)
        alphap_m_rho0, alphap_m_alpha, alphap_m_beta = 0.29, -0.03, 0.04
        
        # Mass scaling logic
        mcut = 1e14  # M_sun
        m_200c_msun = m_b / h
        mass_ratio = m_200c_msun / mcut
        
        # Use jnp.where to choose the mass scaling exponent
        am_rho0 = jnp.where(m_200c_msun > mcut, alpha_m_rho0, alphap_m_rho0)
        am_alpha = jnp.where(m_200c_msun > mcut, alpha_m_alpha, alphap_m_alpha)
        am_beta = jnp.where(m_200c_msun > mcut, alpha_m_beta, alphap_m_beta)
        
        # Compute Shape Parameters (Equations A1, A2 from B16)
        # These result in shape (1, Nm, Nz)
        rho0 = A_rho0 * mass_ratio**am_rho0 * (1 + z_b)**alpha_z_rho0 * (1 + c_delta)**alpha_c_rho0
        alpha = A_alpha * mass_ratio**am_alpha * (1 + z_b)**alpha_z_alpha * (1 + c_delta)**alpha_c_alpha
        beta = A_beta * mass_ratio**am_beta * (1 + z_b)**alpha_z_beta * (1 + c_delta)**alpha_c_beta
        
        gamma = -0.2
        xc = 0.5
        
        # Profile Shape Function (Nx, Nm, Nz)
        # x_b: (Nx,1,1) / rho0: (1,Nm,Nz) -> auto-broadcasts
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
    
        C = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m_delta_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # Scaled radius and GNFW formula
        c_delta = halo_model.c_delta(m, z, params=params)  # (Nm, Nz)
        scaled_x = c_delta[None, :, :] * x[:, None, None]   # (Nx, Nm, Nz)
        Pe = C * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
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
        delta = halo_model.mass_definition.delta
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
    
