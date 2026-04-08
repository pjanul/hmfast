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



class PressureProfile(HaloProfile):
     def u_k(self, halo_model, k, m, z, moment=1):
        
        
        h = halo_model.emulator.H0 / 100 
        B = self.B
        delta = halo_model.mass_definition.delta
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        r_delta = halo_model.r_delta(m, z) / B**(1/3) # (Nm, Nz)
        d_A = jnp.atleast_1d(halo_model.emulator.angular_diameter_distance(z)) * h
        ell_delta = d_A[None, :] / r_delta  # (Nm, Nz)
        
        Mpc_per_h_to_cm = Const._Mpc_over_m_ / h # This is actually Mpc_per_h_to_m, but the math is currently working
        prefactor = (1 + z)[None, :] * 4 * jnp.pi * r_delta * Mpc_per_h_to_cm / (ell_delta**2)  # (Nm, Nz)
        
        # Target ell grid for interpolation: (Nk, Nz)
        chi = d_A * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
        
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        k_native, u_k_native = self.u_k_hankel(halo_model, self.x, m, z)  
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None])) 
        ell_native = k_native[:, None, None] * ell_delta[None, :, :] # (Nk_native, Nm, Nz)
        
        # Apply prefactor and moment
        u_ell_base = prefactor[None, :, :] * u_ell_native # (Nk_native, Nm, Nz)
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # Interpolate over the native k-axis (axis 0) for every combination of m and z    
        def interp_at_z(ell_t, ell_n, u_n):
            return jnp.interp(ell_t, ell_n, u_n)
       
        vmap_interp = jax.vmap(
            jax.vmap(interp_at_z, in_axes=(None, 1, 1), out_axes=1), 
            in_axes=(1, 2, 2), out_axes=2
        )
        
        # Resulting shape: (Nk, Nm, Nz)
        u_ell_interp = vmap_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp






@register_pytree_node_class
class GNFWPressureProfile(PressureProfile):
    def __init__(self, x=None, P0_GNFW=8.130, alpha_GNFW=1.0620, beta_GNFW=5.4807, gamma_GNFW=0.3292, B=1.4):

        self.P0_GNFW = P0_GNFW
        self.alpha_GNFW = alpha_GNFW
        self.beta_GNFW = beta_GNFW
        self.gamma_GNFW = gamma_GNFW
        self.B = B

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


    def tree_flatten(self):
        # The dynamic parameters JAX should track
        leaves = (self.P0_GNFW, self.alpha_GNFW, self.beta_GNFW, self.gamma_GNFW, self.B)
        # Static metadata: the grid and the Hankel object
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        # Create object without calling __init__ to avoid rebuilding Hankel
        obj = cls.__new__(cls)
        obj.P0_GNFW, obj.alpha_GNFW, obj.beta_GNFW, obj.gamma_GNFW, obj.B = leaves
        obj._x = x
        obj._hankel = hankel
        return obj

    def update_params(self, **kwargs):
        """Helper to return a NEW profile with updated leaf values."""
        names = ["P0_GNFW", "alpha_GNFW", "beta_GNFW", "gamma_GNFW", "B"]
        
        # STRICT CHECK: Block typos immediately
        if not set(kwargs).issubset(names):
            invalid = set(kwargs) - set(names)
            raise ValueError(f"Invalid GNFW parameter(s): {invalid}. Expected: {names}")

        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
        

    def profile(self, halo_model, x, m, z):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        
        Fully vectorized: supports
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
       
    
        # Retrieve all required parameters and ensure all inputs are 1D  
        
        H0 = halo_model.emulator.H0
        
        P0, alpha, beta, gamma, B = self.P0_GNFW, self.alpha_GNFW, self.beta_GNFW, self.gamma_GNFW, self.B 
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z) 
       
        # Helper variables for normalization
        h = H0 / 100.0
        c_km_s = Const._c_ / 1e3
        H = halo_model.emulator.hubble_parameter(z) * c_km_s  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)

        # Corrected mass given the hydrostatic mass bias
        m_delta_tilde = (m / B)[None, :, None]  # (1, Nm, 1)
    
        P_500c = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m_delta_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # Scaled radius and GNFW formula
        c_delta = halo_model.c_delta(m, z)  # (Nm, Nz)
        scaled_x = c_delta[None, :, :] * x[:, None, None]   # (Nx, Nm, Nz)
        Pe = P_500c * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return Pe  # shape: (Nx, Nm, Nz)



@register_pytree_node_class
class B12PressureProfile(PressureProfile):
    def __init__(self, x=None, 
                 A_P0=18.1, A_xc=0.497, A_beta=4.35,
                 alpha_m_P0=0.154, alpha_m_xc=-0.00865, alpha_m_beta=0.0393,
                 alpha_z_P0=-0.758, alpha_z_xc=0.731, alpha_z_beta=0.415):
        
        # Physics Parameters (The Leaves)
        self.A_P0, self.A_xc, self.A_beta = A_P0, A_xc, A_beta
        self.alpha_m_P0, self.alpha_m_xc, self.alpha_m_beta = alpha_m_P0, alpha_m_xc, alpha_m_beta
        self.alpha_z_P0, self.alpha_z_xc, self.alpha_z_beta = alpha_z_P0, alpha_z_xc, alpha_z_beta

        self.B = 1.0  # Need to get rid of this eventually
        # Grid initialization
        self.x = x if x is not None else jnp.logspace(-4, 1, 256)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)

    def tree_flatten(self):
        leaves = (
            self.A_P0, self.A_xc, self.A_beta,
            self.alpha_m_P0, self.alpha_m_xc, self.alpha_m_beta,
            self.alpha_z_P0, self.alpha_z_xc, self.alpha_z_beta, 
        )
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        obj = cls.__new__(cls)
        
        (obj.A_P0, obj.A_xc, obj.A_beta,
         obj.alpha_m_P0, obj.alpha_m_xc, obj.alpha_m_beta,
         obj.alpha_z_P0, obj.alpha_z_xc, obj.alpha_z_beta, 
         ) = leaves
        
        obj._x = x
        obj._hankel = hankel
        return obj

    def update_params(self, **kwargs):
        """Helper to return a NEW profile with updated leaf values."""
        names = [
            "A_P0", "A_xc", "A_beta",
            "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
            "alpha_z_P0", "alpha_z_xc", "alpha_z_beta", 
        ]
        
        if not set(kwargs).issubset(names):
            invalid = set(kwargs) - set(names)
            raise ValueError(f"Invalid parameter(s): {invalid}. Expected: {names}")

        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    def profile(self, halo_model, x, m, z):
       
        cparams = halo_model.emulator.get_all_cosmo_params()
        h = cparams["h"]   
        
        # B12 fixed slopes
        alpha_gnfw, gamma_gnfw = 1.0, -0.3
        
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
        x_b, m_b, z_b = x[:, None, None], m[None, :, None], z[None, None, :]
        
        # M200c mass scaling, making sure to apply hydrostatic bias factor
        m_delta_tilde =  m[None, :, None]  # (m / self.B)[None, :, None]  # (1, Nm, 1)
        mass_ratio = (m_delta_tilde / h) / 1e14
        
        # Compute Shape Parameters via consistent scaling naming
        P0 = self.A_P0 * mass_ratio**self.alpha_m_P0 * (1 + z_b)**self.alpha_z_P0 
        xc = self.A_xc * mass_ratio**self.alpha_m_xc * (1 + z_b)**self.alpha_z_xc 
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta 
        
        # Normalized GNFW shape
        scaled_x = x_b / xc
        p_x = (scaled_x)**gamma_gnfw * (1 + scaled_x**alpha_gnfw)**(-beta)
        
        # Thermal Pressure Normalization (P200c)
        # Usually follows P200c = 200 * G * M200c * rho_crit * f_b / (2 * R200c)
        rho_crit = jnp.atleast_1d(halo_model.emulator.critical_density(z))
        H = jnp.atleast_1d(halo_model.emulator.hubble_parameter(z)) * (Const._c_ / 1e3)
        r_delta = halo_model.r_delta(m, z)


        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        P_200c = ((m_b / r_delta[None, :, :]) *  f_b *  2.61051e-18 * (H[None, None, :])**2 )   #2.61051e-18 should be written with proper constants
                
        return P_200c * P0 * p_x