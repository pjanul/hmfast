import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
import functools

from hmfast.download import get_default_data_path
from hmfast.utils import lambertw, Const
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import HaloProfile, HankelTransform


class DensityProfile(HaloProfile):
    
    def u_k(self, halo_model, k, m, z, moment=1):
        """
        Compute the kSZ tracer u_ell (Nk, Nm, Nz).
        Supports arbitrary input shapes for k, m, and z.
        """
        
        h = halo_model.cosmology.H0 / 100
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute r_delta and ell_delta
        delta = halo_model.mass_definition.delta
        r_delta = halo_model.r_delta(m, z)
        d_A_z = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z)) * h
        ell_delta = d_A_z[None, :] / r_delta
        
        # chi: (Nz,) -> Target ell grid: (Nk, Nz)
        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
    
        # Calculate kSZ Prefactor as (Nm, Nz)
        vrms = jnp.sqrt(halo_model.cosmology.v_rms_squared(z))
        mu_e = 1.14
        f_free = 1.0
        prefactor = (4 * jnp.pi * r_delta**3 * f_free / mu_e * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :])
    
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        k_native, u_k_native = self.u_k_hankel(halo_model, self.x, m, z)   # New way
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None]))
        ell_native = k_native[:, None, None] * ell_delta[None, :, :]
        
        # Apply prefactor and moment logic
        u_ell_base = prefactor[None, :, :] * u_ell_native
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # 5. Vectorized Interpolation (Double vmap)
        def interp_single_column(target_x, native_x, native_y):
            return jnp.interp(target_x, native_x, native_y)
    
        # Map over Redshift (Nz) then Mass (Nm)
        vmapped_interp = jax.vmap(
            jax.vmap(interp_single_column, in_axes=(None, 1, 1), out_axes=1),
            in_axes=(1, 2, 2), out_axes=2
        )
        
        u_ell_interp = vmapped_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp


class B16DensityProfile(DensityProfile):
    """
    Electron density profile from `Battaglia et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016JCAP...08..058B/abstract>`_.
    """
    def __init__(self, x=None, 
                 A_rho0=4000.0, A_alpha=0.88, A_beta=3.83,
                 alpha_m_rho0=0.29, alpha_m_alpha=-0.03, alpha_m_beta=0.04,
                 alpha_z_rho0=-0.66, alpha_z_alpha=0.19, alpha_z_beta=-0.025,
                ):
        
        # Grid initialization (triggers the x.setter)
        self.x = x if x is not None else jnp.logspace(-4, 1, 256)

        self.A_rho0, self.A_alpha, self.A_beta = A_rho0, A_alpha, A_beta
        self.alpha_m_rho0, self.alpha_m_alpha, self.alpha_m_beta = alpha_m_rho0, alpha_m_alpha, alpha_m_beta
        self.alpha_z_rho0, self.alpha_z_alpha, self.alpha_z_beta = alpha_z_rho0, alpha_z_alpha, alpha_z_beta

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)


    def _tree_flatten(self):
        # Dynamic calibration parameters
        leaves = (
            self.A_rho0, self.A_alpha, self.A_beta,
            self.alpha_m_rho0, self.alpha_m_alpha, self.alpha_m_beta,
            self.alpha_z_rho0, self.alpha_z_alpha, self.alpha_z_beta
        )
        # Static metadata
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        obj = cls.__new__(cls)
        
        # Unpack leaves back into attributes
        (obj.A_rho0, obj.A_alpha, obj.A_beta,
         obj.alpha_m_rho0, obj.alpha_m_alpha, obj.alpha_m_beta,
         obj.alpha_z_rho0, obj.alpha_z_alpha, obj.alpha_z_beta) = leaves
        
        obj._x = x
        obj._hankel = hankel
        return obj


    # def update(self, **kwargs):
    #     """
    #     Return a new profile instance with updated calibration parameters.

    #     Parameters
    #     ----------
    #     A_rho0 : float, optional
    #     A_alpha : float, optional
    #     A_beta : float, optional
    #     alpha_m_rho0 : float, optional
    #     alpha_m_alpha : float, optional
    #     alpha_m_beta : float, optional
    #     alpha_z_rho0 : float, optional
    #     alpha_z_alpha : float, optional
    #     alpha_z_beta : float, optional

    #     Returns
    #     -------
    #     B16DensityProfile
    #         New profile instance with updated parameters.
    #     """
    #     names = [
    #         "A_rho0", "A_alpha", "A_beta",
    #         "alpha_m_rho0", "alpha_m_alpha", "alpha_m_beta",
    #         "alpha_z_rho0", "alpha_z_alpha", "alpha_z_beta"
    #     ]
        
    #     # Strict Check: Block typos immediately
    #     if not set(kwargs).issubset(names):
    #         invalid = set(kwargs) - set(names)
    #         raise ValueError(f"Invalid parameter(s): {invalid}. Expected: {names}")

    #     leaves, treedef = self._tree_flatten()
    #     # Create new leaf list by replacing values from kwargs if they exist
    #     new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
    #     return self._tree_unflatten(treedef, new_leaves)

    def update(self, A_rho0=None, A_alpha=None, A_beta=None,
               alpha_m_rho0=None, alpha_m_alpha=None, alpha_m_beta=None,
               alpha_z_rho0=None, alpha_z_alpha=None, alpha_z_beta=None):
        """
        Return a new profile instance with updated calibration parameters.

        Parameters
        ----------
        A_rho0 : float, optional
        A_alpha : float, optional
        A_beta : float, optional
        alpha_m_rho0 : float, optional
        alpha_m_alpha : float, optional
        alpha_m_beta : float, optional
        alpha_z_rho0 : float, optional
        alpha_z_alpha : float, optional
        alpha_z_beta : float, optional

        Returns
        -------
        B16DensityProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        new_leaves = (
            A_rho0 if A_rho0 is not None else self.A_rho0,
            A_alpha if A_alpha is not None else self.A_alpha,
            A_beta if A_beta is not None else self.A_beta,
            alpha_m_rho0 if alpha_m_rho0 is not None else self.alpha_m_rho0,
            alpha_m_alpha if alpha_m_alpha is not None else self.alpha_m_alpha,
            alpha_m_beta if alpha_m_beta is not None else self.alpha_m_beta,
            alpha_z_rho0 if alpha_z_rho0 is not None else self.alpha_z_rho0,
            alpha_z_alpha if alpha_z_alpha is not None else self.alpha_z_alpha,
            alpha_z_beta if alpha_z_beta is not None else self.alpha_z_beta,
        )
        
        return self._tree_unflatten(treedef, new_leaves)

    @staticmethod
    def get_params(model_key="agn"):
        """Static helper to grab Table 2 values."""
        presets = {
            "agn": {
                'A_rho0': 4000.0, 'A_alpha': 0.88, 'A_beta': 3.83,
                'alpha_m_rho0': 0.29, 'alpha_m_alpha': -0.03, 'alpha_m_beta': 0.04,
                'alpha_z_rho0': -0.66, 'alpha_z_alpha': 0.19, 'alpha_z_beta': -0.025
            },
            "shock": {
                'A_rho0': 1.9e4, 'A_alpha': 0.70, 'A_beta': 4.43,
                'alpha_m_rho0': 0.09, 'alpha_m_alpha': -0.017, 'alpha_m_beta': 0.005,
                'alpha_z_rho0': -0.95, 'alpha_z_alpha': 0.27, 'alpha_z_beta': 0.037
            }
        }
        key = model_key.lower()
        if key not in presets:
            raise ValueError(f"Model {model_key} not recognized. Choose 'agn' or 'shock'.")
        return presets[key]
        

    def profile(self, halo_model, x, m, z):
        """
        Compute the BCM gas density profile.

        Parameters
        ----------
        halo_model : HaloModel
            The parent halo model instance.
        x : array-like
            Dimensionless radius r/R_vir (Nx,).
        m : array-like
            Halo mass M_vir [M_sun/h] (Nm,).
        z : array-like
            Redshift (Nz,).

        Returns
        -------
        rho_gas : array-like
            Gas density 
        """
        cparams = halo_model.cosmology._cosmo_params()
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        h = cparams["h"]

        gamma = -0.2
        xc = 0.5
        
        # Ensure 1D and setup broadcasting shapes
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m),  jnp.atleast_1d(z)  # (Nx,)
        x_b, m_b, z_b = x[:, None, None], m[None, :, None], z[None, None, :]      # (Nx, 1, 1), (1, Nm, 1), (1, 1, Nz)
        
        # Critical density broadcast to (1, 1, Nz)
        rho_crit_z = jnp.atleast_1d(halo_model.cosmology.critical_density(z))[None, None, :]
        
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


jax.tree_util.register_pytree_node(
    B16DensityProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: B16DensityProfile._tree_unflatten(aux_data, children)
)

        

class NFWDensityProfile(DensityProfile):
    """
    Matter density profile from `Navarro, Frenk & White (1997) <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_.
    """
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
        

    def profile(self, halo_model, x, m, z):
        cparams = halo_model.cosmology._cosmo_params()
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
       
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Get scale radius r_s
        r_delta = halo_model.r_delta(m, z)
        c_delta = halo_model.concentration.c_delta(halo_model, m, z)
        r_s = r_delta / c_delta # (Nm, Nz)
        
        # Calculate rho_s
        m_nfw = jnp.log(1 + c_delta) - c_delta / (1 + c_delta) # (Nm, Nz)
        rho_s = m[:, None] / (4 * jnp.pi * r_s**3 * m_nfw)    # (Nm, Nz)
        
        # Final broadcast to (Nx, Nm, Nz)
        # x needs to be (Nx, 1, 1) and rho_s (1, Nm, Nz)
        rho_gas = f_b * rho_s[None, :, :] / (x[:, None, None] * (1 + x[:, None, None])**2)
        
        return rho_gas





class BCMDensityProfile(DensityProfile):
    """
    Electron density profile from `Schneider et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019JCAP...03..020S/abstract>`_, 
    also known as the Baryon Correction Model (BCM).
    """
    def __init__(self, x=None, 
                 log10Mc=13.25, theta_ej = 4.711, eta_star = 0.2, 
                 delta = 7.0, gamma = 2.5, mu = 1.0, nu_log10Mc = -0.038,
                ):
        
        # Grid initialization (triggers the x.setter)
        self.x = x if x is not None else jnp.logspace(-4, 1, 256)

        self.log10Mc, self.theta_ej, self.eta_star = log10Mc, theta_ej, eta_star
        self.delta, self.gamma, self.mu, self.nu_log10Mc = delta, gamma, mu, nu_log10Mc
        

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)


    def _tree_flatten(self):
        # Dynamic calibration parameters
        leaves = (
            self.log10Mc, self.theta_ej, self.eta_star,
            self.delta, self.gamma, self.mu, self.nu_log10Mc
        )
        # Static metadata
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        obj = cls.__new__(cls)
        
        # Unpack leaves back into attributes
        (obj.log10Mc, obj.theta_ej, obj.eta_star,
         obj.delta, obj.gamma, obj.mu, obj.nu_log10Mc) = leaves
        
        obj._x = x
        obj._hankel = hankel
        return obj


    def update(self, log10Mc=None, theta_ej=None, eta_star=None, 
               delta=None, gamma=None, mu=None, nu_log10Mc=None):
        """
        Return a new profile instance with updated calibration parameters.

        Parameters
        ----------
        log10Mc : float, optional
        theta_ej : float, optional
        eta_star : float, optional
        delta : float, optional
        gamma : float, optional
        mu : float, optional
        nu_log10Mc : float, optional

        Returns
        -------
        BCMDensityProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        new_leaves = (
            log10Mc if log10Mc is not None else self.log10Mc,
            theta_ej if theta_ej is not None else self.theta_ej,
            eta_star if eta_star is not None else self.eta_star,
            delta if delta is not None else self.delta,
            gamma if gamma is not None else self.gamma,
            mu if mu is not None else self.mu,
            nu_log10Mc if nu_log10Mc is not None else self.nu_log10Mc,
        )
        
        return self._tree_unflatten(treedef, new_leaves)


    def profile(self, halo_model, x, m, z):
        """
        BCM gas density profile based.
        
        Args:
            x: Dimensionless radius r/R200c (Nx,)
            m: Halo mass M200c [M_sun/h] (Nm,)
            z: Redshift (Nz,)
        Returns:
            rho_gas: Gas density in [M_sun h^2 / Mpc^3] (Nx, Nm, Nz)
        """
       
        cparams = halo_model.cosmology._cosmo_params()
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Broadcasting shapes: (Nx, 1, 1), (1, Nm, 1), (1, 1, Nz)
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
        xb, mb, zb = x[:, None, None], m[None, :, None], z[None, None, :]
        
        # This model is calibrated for the virial radius 
        r_vir = halo_model.r_delta(m, z, mass_definition=MassDefinition("vir", "critical"))
        r_asked = xb * r_vir
        
        # Redshift Dependent Mc (Matching your C logic)
        mc_z_log = self.log10Mc * (1. + zb)**self.nu_log10Mc
        mc = 10.**mc_z_log
        
        # Profile Components
        ms = 2.5e11  # M_sun/h, fixed value
        fstar_ms = 0.055 # Fixed value
        f_star = fstar_ms * (m / ms)**(-self.eta_star)
        num = f_b - f_star
        
        # beta_m scaling (Mass dependent slope)
        m_ratio_mu = (mb / mc)**self.mu
        beta_m = 3. * m_ratio_mu / (1. + m_ratio_mu)
        
        # Denominator 1: Large scale bound gas
        denom1 = (1. + 10. * r_asked / r_vir)**beta_m
        
        # Denominator 2: Ejected gas / transition,
        scaled_r = r_asked / (self.theta_ej * r_vir)
        denom2 = (1. + (scaled_r)**self.gamma)**((self.delta - beta_m) / self.gamma)
    
        
        return num / (denom1 * denom2) 


jax.tree_util.register_pytree_node(
    BCMDensityProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: BCMDensityProfile._tree_unflatten(aux_data, children)
)

    
   