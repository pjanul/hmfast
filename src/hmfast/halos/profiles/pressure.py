import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
from functools import partial

from hmfast.download import get_default_data_path
from hmfast.utils import Const
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.profiles import HaloProfile, HankelTransform



class PressureProfile(HaloProfile):
    @partial(jax.jit, static_argnums=(0,))
    def u_k(self, halo_model, k, m, z):
        """
        Compute the projected Fourier-space pressure profile for halo-model calculations.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and halo-radius relation.
        k : float or jnp.ndarray
            Comoving wavenumber(s) in :math:`\\mathrm{Mpc}^{-1}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Transformed profile with shape :math:`(N_k, N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        B = getattr(self, "B", 1.0)

        r_delta_native = jnp.reshape(halo_model.mass_definition.r_delta(halo_model.cosmology, m, z), (len(m), len(z)))
        r_delta = r_delta_native / B**(1 / 3)
        r = self.x[:, None, None] * r_delta_native[None, :, :] * (1.0 + z[None, None, :])
        d_A = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))
        ell_delta = d_A[None, :] / r_delta  # (Nm, Nz)
        
        mpc_to_m = Const._Mpc_over_m_
        prefactor = (1 + z)[None, :] * 4 * jnp.pi * r_delta * mpc_to_m / (ell_delta**2)  # (Nm, Nz)
        
        # Target ell grid for interpolation: (Nk, Nz)
        chi = d_A * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
        
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        k_native, u_k_native = self._u_k_hankel(halo_model, self.x, r, m, z)
        u_k_native = jnp.reshape(u_k_native, (len(k_native), len(m), len(z)))
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None])) 
        ell_native = k_native[:, None, None] * ell_delta[None, :, :] # (Nk_native, Nm, Nz)
        
        # Apply prefactor
        u_ell_base = prefactor[None, :, :] * u_ell_native # (Nk_native, Nm, Nz)
        u_ell_val = u_ell_base
    
        # Interpolate over the native k-axis (axis 0) for every combination of m and z    
        def interp_at_z(ell_t, ell_n, u_n):
            return jnp.interp(ell_t, ell_n, u_n)
       
        vmap_interp = jax.vmap(
            jax.vmap(interp_at_z, in_axes=(None, 1, 1), out_axes=1), 
            in_axes=(1, 2, 2), out_axes=2
        )
        
        # Resulting shape: (Nk, Nm, Nz)
        u_ell_interp = vmap_interp(ell_target, ell_native, u_ell_val)
        
        return jnp.squeeze(u_ell_interp)






class GNFWPressureProfile(PressureProfile):
    """
    Electron pressure profile from `Nagai, Kravtsov & Vikhlinin (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract>`_.

    The profile is evaluated as a function of the comoving radius
    :math:`r`, but its normalization and shape are defined using the native
    :math:`500c` calibration mass and radius:

    .. math::

        P_e(r, M, z) = P_{500c} \\, P_0
        \\left(c_{500} x_{500c}\\right)^{-\\gamma}
        \\left[1 + \\left(c_{500} x_{500c}\\right)^\\alpha\\right]^{(\\gamma-\\beta)/\\alpha}
        \\tag{1}

    where :math:`x_{500c} = r / r_{500c}` and :math:`r_{500c}` has the same
    units as :math:`r`, and

    .. math::

        P_{500c} =
        1.65
        \\left(\\frac{h}{0.7}\\right)^2
        E(z)^{8/3}
        \\left(\\frac{M_{500c} / B}{0.7 \\times 3 \\times 10^{14} \\, M_\\odot}\\right)^{2/3 + 0.12}
        \\left(\\frac{0.7}{h}\\right)^{3/2}
        \\tag{2}

    with :math:`E(z) = H(z) / H_0`. In the implementation, the input halo mass
    is first converted from the halo model's mass definition to
    :math:`M_{500c}`.

    The projected Fourier-space pressure profile is evaluated as

    .. math::

        u_\\ell(\\ell, M, z) =
        \\frac{4 \\pi (1+z) r_\\Delta}{\\ell_\\Delta^2}
        \\int dx \\, x^2 \\, P_e(x, M, z)
        \\, \\frac{\\sin\\!\\left[(\\ell / \\ell_\\Delta) x\\right]}
        {(\\ell / \\ell_\\Delta) x}
        \\tag{3}

    where :math:`\\ell_\\Delta(M, z) = d_A(z) / r_\\Delta(M, z)` and
    :math:`\\chi(z) = (1+z) d_A(z)`.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_\\Delta` used to tabulate the profile and define the Hankel transform, with :math:`r_\\Delta` expressed in the same units as :math:`r`.
    P0 : float
        Dimensionless gNFW normalization :math:`P_0`.
    c500 : float
        Concentration parameter :math:`c_{500}` of the :math:`500c` pressure profile.
    alpha : float
        Intermediate-slope parameter :math:`\\alpha` of the gNFW profile.
    beta : float
        Outer-slope parameter :math:`\\beta` of the gNFW profile.
    gamma : float
        Inner-slope parameter :math:`\\gamma` of the gNFW profile.
    B : float
        Hydrostatic mass bias factor :math:`B` used in the :math:`M_{500c}` normalization.
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

    @partial(jax.jit, static_argnums=(0,))
    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-pressure profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, mass-definition conversion, and halo
            radius.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron pressure profile with shape :math:`(N_r, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """
        H0 = halo_model.cosmology.H0
        P0, c500, alpha, beta, gamma, B = self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = H0 / 100.0
    
        # Convert input mass to M500c for normalization, since this profile was calibrated for 500c
        mass_def_old = halo_model.mass_definition
        mass_def_500c = MassDefinition(500, "critical")
        c_old = jnp.reshape(halo_model.concentration.c_delta(halo_model, m, z), (len(m), len(z)))
        m500c = jnp.reshape(convert_m_delta(halo_model.cosmology, m, z, mass_def_old, mass_def_500c, c_old=c_old), (len(m), len(z)))

        r_500c = jnp.reshape(mass_def_500c.r_delta(halo_model.cosmology, m500c, z), m500c.shape)  # (Nm, Nz)
    
        # Convert the comoving radius to the calibrated physical 500c coordinate.
        x_500c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_500c[None, :, :])  # (Nr, Nm, Nz)
    
        # Compute normalization P_500c (with hydrostatic bias)
        h = H0 / 100.0
        H = halo_model.cosmology.hubble_parameter(z)  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)
        m500c_tilde = (m500c * h / B)[None, :, None]  # (1, Nm, 1)
        P_500c = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m500c_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # GNFW profile
        scaled_x = c500 * x_500c  # (Nr, Nm, Nz)
        Pe = P_500c * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return jnp.squeeze(Pe)

jax.tree_util.register_pytree_node(
    GNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GNFWPressureProfile._tree_unflatten(aux_data, children)
)


class B12PressureProfile(PressureProfile):
    """
    Electron pressure profile from `Battaglia et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...758...74B/abstract>`_.

    The profile is evaluated as a function of the comoving radius
    :math:`r`, but its normalization and shape are defined using the native
    :math:`200c` calibration mass and radius:

    .. math::

        P_e(r, M, z)
        = P_{200c} \\, P_0
        \\left(\\frac{x_{200c}}{x_c}\\right)^\\gamma
        \\left[1 + \\left(\\frac{x_{200c}}{x_c}\\right)^\\alpha\\right]^{-\\beta}
        \\tag{1}

    where :math:`x_{200c} = r / r_{200c}` and :math:`r_{200c}` has the same
    units as :math:`r`.
    In this implementation, :math:`\\alpha = 1` and :math:`\\gamma = -0.3`,
    and the remaining profile parameters follow the Battaglia scaling

    .. math::

        X(M_{200c}, z) = A_X
        \\left(\\frac{M_{200c} / h}{10^{14} M_\\odot}\\right)^{\\alpha_m^X}
        (1 + z)^{\\alpha_z^X}
        \\tag{2}

    where :math:`X \\in \\{P_0, x_c, \\beta\\}`. In the implementation, the
    input halo mass is first converted from the halo model's mass definition to
    :math:`M_{200c}`.

    The projected Fourier-space pressure profile is evaluated as

    .. math::

        u_\\ell(\\ell, M, z) =
        \\frac{4 \\pi (1+z) r_\\Delta}{\\ell_\\Delta^2}
        \\int dx \\, x^2 \\, P_e(x, M, z)
        \\, \\frac{\\sin\\!\\left[(\\ell / \\ell_\\Delta) x\\right]}
        {(\\ell / \\ell_\\Delta) x}
        \\tag{3}

    where :math:`\\ell_\\Delta(M, z) = d_A(z) / r_\\Delta(M, z)` and
    :math:`\\chi(z) = (1+z) d_A(z)`.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_\\Delta` used to tabulate the profile and define the Hankel transform, with :math:`r_\\Delta` expressed in the same units as :math:`r`.
    A_P0 : float
        Amplitude :math:`A_{P_0}` of the pressure normalization scaling.
    A_xc : float
        Amplitude :math:`A_{x_c}` of the core-radius scaling.
    A_beta : float
        Amplitude :math:`A_\\beta` of the outer-slope scaling.
    alpha_m_P0 : float
        Mass-scaling exponent :math:`\\alpha_m^{P_0}`.
    alpha_m_xc : float
        Mass-scaling exponent :math:`\\alpha_m^{x_c}`.
    alpha_m_beta : float
        Mass-scaling exponent :math:`\\alpha_m^\\beta`.
    alpha_z_P0 : float
        Redshift-scaling exponent :math:`\\alpha_z^{P_0}`.
    alpha_z_xc : float
        Redshift-scaling exponent :math:`\\alpha_z^{x_c}`.
    alpha_z_beta : float
        Redshift-scaling exponent :math:`\\alpha_z^\\beta`.
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
         obj.alpha_z_P0, obj.alpha_z_xc, obj.alpha_z_beta) = leaves

        obj._x = x
        obj._hankel = hankel
        return obj

    def update(self, A_P0=None, A_xc=None, A_beta=None,
               alpha_m_P0=None, alpha_m_xc=None, alpha_m_beta=None,
               alpha_z_P0=None, alpha_z_xc=None, alpha_z_beta=None):
        """
        Return a new profile instance with updated B12 parameters.
    
        Parameters
        ----------
        A_P0, A_xc, A_beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc, alpha_z_beta : float, optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.
    
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

    @partial(jax.jit, static_argnums=(0,))
    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-pressure profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, mass-definition conversion, and halo
            radius.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron pressure profile with shape :math:`(N_r, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """
        cparams = halo_model.cosmology._cosmo_params()
        h = cparams["h"]
        alpha, gamma = 1.0, -0.3
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Convert input mass to M200c for normalization
        mass_def_old = halo_model.mass_definition
        mass_def_200c = MassDefinition(200, "critical")
        c_old = jnp.reshape(halo_model.concentration.c_delta(halo_model, m, z), (len(m), len(z)))
        m200c = jnp.reshape(convert_m_delta(halo_model.cosmology, m, z, mass_def_old, mass_def_200c, c_old=c_old), (len(m), len(z)))

        r_200c = jnp.reshape(mass_def_200c.r_delta(halo_model.cosmology, m200c, z), m200c.shape)  # (Nm, Nz)
    
        # Convert the comoving radius to the calibrated physical 200c coordinate.
        x_200c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_200c[None, :, :])  # (Nr, Nm, Nz)
        m200c_b = m200c[None, :, None]
        z_b = z[None, None, :]
        mass_ratio = m200c_b / 1e14
    
        # Compute shape parameters using M200c
        P0 = self.A_P0 * mass_ratio**self.alpha_m_P0 * (1 + z_b)**self.alpha_z_P0
        xc = self.A_xc * mass_ratio**self.alpha_m_xc * (1 + z_b)**self.alpha_z_xc
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta
    
        # Normalized GNFW shape
        scaled_x = x_200c / xc
        p_x = (scaled_x)**gamma * (1 + scaled_x**alpha)**(-beta)
    
        # Thermal Pressure Normalization (P200c)
        H = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        r_200c = r_200c * h
        # Use M200c and r_200c for normalization
        P_200c = ((m200c_b / r_200c[None, :, :]) * f_b * 2.61051e-18 * (H[None, None, :])**2)
    
        return jnp.squeeze(P_200c * P0 * p_x)

jax.tree_util.register_pytree_node(
    B12PressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: B12PressureProfile._tree_unflatten(aux_data, children)
)