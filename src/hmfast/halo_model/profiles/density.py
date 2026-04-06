import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import mcfit
import functools
from jax.scipy.special import sici, erf
from jax.tree_util import register_pytree_node_class

from hmfast.download import get_default_data_path
from hmfast.defaults import merge_with_defaults
from hmfast.utils import lambertw, Const
from hmfast.halo_model.mass_definition import MassDefinition
from hmfast.halo_model.profiles import HaloProfile, HankelTransform


class DensityProfile(HaloProfile):
    
    def u_k(self, halo_model, k, m, z, moment=1, params=None):
        """
        Compute the kSZ tracer u_ell (Nk, Nm, Nz).
        Supports arbitrary input shapes for k, m, and z.
        """
        
        params = merge_with_defaults(params)
        h = params['H0'] / 100.0
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute r_delta and ell_delta
        delta = halo_model.mass_definition.delta
        r_delta = halo_model.r_delta(m, z, params=params)
        d_A_z = jnp.atleast_1d(halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        ell_delta = d_A_z[None, :] / r_delta
        
        # chi: (Nz,) -> Target ell grid: (Nk, Nz)
        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
    
        # Calculate kSZ Prefactor as (Nm, Nz)
        vrms = jnp.sqrt(halo_model.emulator.v_rms_squared(z, params=params))
        mu_e = 1.14
        f_free = 1.0
        prefactor = (4 * jnp.pi * r_delta**3 * f_free / mu_e * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :])
    
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        #k_native, u_k_native = self.u_k_hankel(m, z, params=params) 
        k_native, u_k_native = self.u_k_hankel(halo_model, self.x, m, z, params=params)   # New way
        
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


@register_pytree_node_class
class B16DensityProfile(DensityProfile):
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


    def tree_flatten(self):
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
    def tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        obj = cls.__new__(cls)
        
        # Unpack leaves back into attributes
        (obj.A_rho0, obj.A_alpha, obj.A_beta,
         obj.alpha_m_rho0, obj.alpha_m_alpha, obj.alpha_m_beta,
         obj.alpha_z_rho0, obj.alpha_z_alpha, obj.alpha_z_beta) = leaves
        
        obj._x = x
        obj._hankel = hankel
        return obj


    def update_params(self, **kwargs):
        """Helper to return a NEW profile with updated leaf values."""
        names = [
            "A_rho0", "A_alpha", "A_beta",
            "alpha_m_rho0", "alpha_m_alpha", "alpha_m_beta",
            "alpha_z_rho0", "alpha_z_alpha", "alpha_z_beta"
        ]
        
        # Strict Check: Block typos immediately
        if not set(kwargs).issubset(names):
            invalid = set(kwargs) - set(names)
            raise ValueError(f"Invalid parameter(s): {invalid}. Expected: {names}")

        leaves, treedef = jax.tree_util.tree_flatten(self)
        # Create new leaf list by replacing values from kwargs if they exist
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

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
   