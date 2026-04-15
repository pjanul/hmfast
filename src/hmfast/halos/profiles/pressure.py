import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
import functools

from hmfast.download import get_default_data_path
from hmfast.utils import Const
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import HaloProfile, HankelTransform



class PressureProfile(HaloProfile):
     def u_k(self, halo_model, k, m, z, moment=1):
        
        
        h = halo_model.cosmology.H0 / 100 
        B = 1.0 #self.B
        delta = halo_model.mass_definition.delta
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)

        
        r_delta = halo_model.r_delta(m, z) / B**(1/3) # (Nm, Nz)
        d_A = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z)) * h
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





class GNFWPressureProfile(PressureProfile):
    """
    Electron pressure profile from `Nagai, Kravtsov & Vikhlinin (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract>`_.
    """
    
    def __init__(self, x=None, P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292, B=1.4):

        self.P0 = P0
        self.c500 = c500
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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


    def _tree_flatten(self):
        # The dynamic parameters JAX should track
        leaves = (self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B)
        # Static metadata: the grid and the Hankel object
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        # Create object without calling __init__ to avoid rebuilding Hankel
        obj = cls.__new__(cls)
        obj.P0, obj.c500, obj.alpha, obj.beta, obj.gamma, obj.B = leaves
        obj._x = x
        obj._hankel = hankel
        return obj

    def update(self, P0=None, c500=None, alpha=None, beta=None, gamma=None, B=None):
        """
        Return a new profile instance with updated GNFW pressure profile parameters.
    
        Parameters
        ----------
        P0 : float, optional
        c500 : float, optional
        alpha : float, optional
        beta : float, optional
        gamma : float, optional
        B : float, optional
    
        Returns
        -------
        GNFWPressureProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        new_leaves = (
            P0 if P0 is not None else self.P0,
            c500 if c500 is not None else self.c500,
            alpha if alpha is not None else self.alpha,
            beta if beta is not None else self.beta,
            gamma if gamma is not None else self.gamma,
            B if B is not None else self.B,
        )
        
        return self._tree_unflatten(treedef, new_leaves)
    

    def profile(self, halo_model, x, m, z):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        Fully vectorized: supports
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
        H0 = halo_model.cosmology.H0
        P0, c500, alpha, beta, gamma, B = self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Convert input mass to M500c for normalization, since this profile was calibrated for 500c
        mass_def_old = halo_model.mass_definition
        mass_def_500c = MassDefinition(500, "critical")
        m500c = halo_model.convert_m_delta(m, z, mass_def_old, mass_def_500c)
    
        # Compute r_delta (input) and r_500c (for GNFW scaling)
        r_delta = halo_model.r_delta(m, z)  # (Nm, Nz)
        r_500c = halo_model.r_delta(m500c, z, mass_definition=mass_def_500c)  # (Nm, Nz)
    
        # Convert input x = r/r_delta to x_500c = r/r_500c
        x_500c = x[:, None, None] * (r_delta[None, :, :] / r_500c[None, :, :])  # (Nx, Nm, Nz)
    
        # Compute normalization P_500c (with hydrostatic bias)
        h = H0 / 100.0
        c_km_s = Const._c_ / 1e3
        H = halo_model.cosmology.hubble_parameter(z) * c_km_s  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)
        m500c_tilde = (m500c / B)[None, :, None]  # (1, Nm, 1)
        P_500c = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m500c_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # GNFW profile
        scaled_x = c500 * x_500c  # (Nx, Nm, Nz)
        Pe = P_500c * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return Pe  # shape: (Nx, Nm, Nz)


jax.tree_util.register_pytree_node(
    GNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GNFWPressureProfile._tree_unflatten(aux_data, children)
)


class B12PressureProfile(PressureProfile):
    """
    Electron pressure profile from `Battaglia et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...758...74B/abstract>`_.
    """
    def __init__(self, x=None, 
                 A_P0=18.1, A_xc=0.497, A_beta=4.35,
                 alpha_m_P0=0.154, alpha_m_xc=-0.00865, alpha_m_beta=0.0393,
                 alpha_z_P0=-0.758, alpha_z_xc=0.731, alpha_z_beta=0.415):
        
        # Physics Parameters (The Leaves)
        self.A_P0, self.A_xc, self.A_beta = A_P0, A_xc, A_beta
        self.alpha_m_P0, self.alpha_m_xc, self.alpha_m_beta = alpha_m_P0, alpha_m_xc, alpha_m_beta
        self.alpha_z_P0, self.alpha_z_xc, self.alpha_z_beta = alpha_z_P0, alpha_z_xc, alpha_z_beta

        # Grid initialization
        self.x = x if x is not None else jnp.logspace(-4, 1, 256)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value
        self._hankel = HankelTransform(self._x, nu=0.5)

    def _tree_flatten(self):
        leaves = (
            self.A_P0, self.A_xc, self.A_beta,
            self.alpha_m_P0, self.alpha_m_xc, self.alpha_m_beta,
            self.alpha_z_P0, self.alpha_z_xc, self.alpha_z_beta, 
        )
        aux_data = (self._x, self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x, hankel = aux_data
        obj = cls.__new__(cls)
        
        (obj.A_P0, obj.A_xc, obj.A_beta,
         obj.alpha_m_P0, obj.alpha_m_xc, obj.alpha_m_beta,
         obj.alpha_z_P0, obj.alpha_z_xc, obj.alpha_z_beta, 
         ) = leaves
        
        obj._x = x
        obj._hankel = hankel
        return obj

    def update(self, A_P0=None, A_xc=None, A_beta=None,
               alpha_m_P0=None, alpha_m_xc=None, alpha_m_beta=None,
               alpha_z_P0=None, alpha_z_xc=None, alpha_z_beta=None):
        """
        Return a new profile instance with updated B12 pressure profile parameters.
    
        Parameters
        ----------
        A_P0 : float, optional
        A_xc : float, optional
        A_beta : float, optional
        alpha_m_P0 : float, optional
        alpha_m_xc : float, optional
        alpha_m_beta : float, optional
        alpha_z_P0 : float, optional
        alpha_z_xc : float, optional
        alpha_z_beta : float, optional
    
        Returns
        -------
        B12PressureProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        new_leaves = (
            A_P0 if A_P0 is not None else self.A_P0,
            A_xc if A_xc is not None else self.A_xc,
            A_beta if A_beta is not None else self.A_beta,
            alpha_m_P0 if alpha_m_P0 is not None else self.alpha_m_P0,
            alpha_m_xc if alpha_m_xc is not None else self.alpha_m_xc,
            alpha_m_beta if alpha_m_beta is not None else self.alpha_m_beta,
            alpha_z_P0 if alpha_z_P0 is not None else self.alpha_z_P0,
            alpha_z_xc if alpha_z_xc is not None else self.alpha_z_xc,
            alpha_z_beta if alpha_z_beta is not None else self.alpha_z_beta,
        )
        
        return self._tree_unflatten(treedef, new_leaves)

    def profile(self, halo_model, x, m, z):
        """
        Battaglia 2012 pressure profile generalized for arbitrary mass definition.
        Always normalizes and scales using the native 200c definition.
        """
        cparams = halo_model.cosmology.get_all_cosmo_params()
        h = cparams["h"]
        alpha, gamma = 1.0, -0.3
        x, m, z = jnp.atleast_1d(x), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Convert input mass to M200c for normalization
        mass_def_old = halo_model.mass_definition
        mass_def_200c = MassDefinition(200, "critical")
        m200c = halo_model.convert_m_delta(m, z, mass_def_old, mass_def_200c)
    
        # Compute r_delta (input) and r_200c (for B12 scaling)
        r_delta = halo_model.r_delta(m, z)  # (Nm, Nz)
        r_200c = halo_model.r_delta(m200c, z, mass_definition=mass_def_200c)  # (Nm, Nz)
    
        # Rescale x: x_200c = x * (r_delta / r_200c)
        x_200c = x[:, None, None] * (r_delta[None, :, :] / r_200c[None, :, :])  # (Nx, Nm, Nz)
        m200c_b = m200c[None, :, None]
        z_b = z[None, None, :]
        mass_ratio = (m200c_b / h) / 1e14
    
        # Compute shape parameters using M200c
        P0 = self.A_P0 * mass_ratio**self.alpha_m_P0 * (1 + z_b)**self.alpha_z_P0
        xc = self.A_xc * mass_ratio**self.alpha_m_xc * (1 + z_b)**self.alpha_z_xc
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta
    
        # Normalized GNFW shape
        scaled_x = x_200c / xc
        p_x = (scaled_x)**gamma * (1 + scaled_x**alpha)**(-beta)
    
        # Thermal Pressure Normalization (P200c)
        rho_crit = jnp.atleast_1d(halo_model.cosmology.critical_density(z))
        H = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z)) * (Const._c_ / 1e3)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        # Use M200c and r_200c for normalization
        P_200c = ((m200c_b / r_200c[None, :, :]) * f_b * 2.61051e-18 * (H[None, None, :])**2)
    
        return P_200c * P0 * p_x


jax.tree_util.register_pytree_node(
    B12PressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: B12PressureProfile._tree_unflatten(aux_data, children)
)