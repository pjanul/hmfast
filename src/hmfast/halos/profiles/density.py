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
    pass



class B16DensityProfile(DensityProfile):
    """
    Electron density profile from `Battaglia et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016JCAP...08..058B/abstract>`_.

    The profile is evaluated as a function of the comoving radius
    :math:`r`, while its shape is defined using the physical
    :math:`200c` radius:

    .. math::

        \\rho_{\\mathrm{gas,free}}(r)
        = f_b f_{\\mathrm{free}} \\rho_{\\mathrm{crit}}(z) \\, C
        \\left(\\frac{x_{200c}}{x_c}\\right)^{\\gamma}
        \\left[1 + \\left(\\frac{x_{200c}}{x_c}\\right)^{\\alpha}\\right]^{-\\frac{\\beta+\\gamma}{\\alpha}}
        \\tag{1}

    where :math:`x_{200c} = r / r_{200c}` and :math:`r_{200c}` has the same
    units as :math:`r`. With :math:`x_c = 0.5` and
    :math:`\\gamma = -0.2` fixed, the mass- and redshift-dependent parameters
    obey

    .. math::

        X(M_{200c}, z) = A_X
        \\left(\\frac{M_{200c} / h}{10^{14} M_\\odot}\\right)^{\\alpha_m^X}
        (1 + z)^{\\alpha_z^X}
        \\tag{2}

    where :math:`X \\in \\{C, \\alpha, \\beta\\}`.

    The projected Fourier-space profile is evaluated as

    .. math::

        u_k(k, M, z) =
        4 \\pi \\, r_\\Delta^3 \\, \\frac{1}{\\mu_e}
        \\, \\frac{(1+z)^3}{\\chi^2(z)} \\, v_{\\mathrm{rms}}(z)
        \\int dx \\, x^2 \\, \\rho(x, M, z)
        \\, \\frac{\\sin\\!\\left[(k r_\\Delta) x\\right]}
        {(k r_\\Delta) x}
        \\tag{3}

    where :math:`x = r / [(1+z) r_\\Delta]`, :math:`\\mu_e = 1.14`, and
    :math:`\\chi(z) = (1+z) d_A(z)` is the comoving distance.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / [(1+z) r_\\Delta]` used to tabulate the profile and define the Hankel transform.
    A_rho0 : float
        Amplitude :math:`A_C` controlling the normalization of the density profile.
    A_alpha : float
        Amplitude :math:`A_\\alpha` controlling the transition width.
    A_beta : float
        Amplitude :math:`A_\\beta` controlling the outer slope.
    alpha_m_rho0 : float
        Mass-scaling exponent :math:`\\alpha_m^C`.
    alpha_m_alpha : float
        Mass-scaling exponent :math:`\\alpha_m^\\alpha`.
    alpha_m_beta : float
        Mass-scaling exponent :math:`\\alpha_m^\\beta`.
    alpha_z_rho0 : float
        Redshift-scaling exponent :math:`\\alpha_z^C`.
    alpha_z_alpha : float
        Redshift-scaling exponent :math:`\\alpha_z^\\alpha`.
    alpha_z_beta : float
        Redshift-scaling exponent :math:`\\alpha_z^\\beta`.
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


    def update(self, A_rho0=None, A_alpha=None, A_beta=None,
               alpha_m_rho0=None, alpha_m_alpha=None, alpha_m_beta=None,
               alpha_z_rho0=None, alpha_z_alpha=None, alpha_z_beta=None):
        """
        Return a new profile instance with updated Battaglia density parameters.

        Parameters
        ----------
        A_rho0, A_alpha, A_beta, alpha_m_rho0, alpha_m_alpha, alpha_m_beta, alpha_z_rho0, alpha_z_alpha, alpha_z_beta : float, optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

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

    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-density profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology.
        r : float or jnp.ndarray
            Comoving radius or radii in Mpc.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron-density profile with shape :math:`(N_r, N_M, N_z)`.
        """
        cparams = halo_model.cosmology._cosmo_params()
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        h = cparams["h"]
        f_free = 1.0

        gamma = -0.2
        xc = 0.5
        
        # Ensure 1D and setup broadcasting shapes
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m),  jnp.atleast_1d(z)
        r_b, m_b, z_b = r[:, None, None], m[None, :, None], z[None, None, :]

        r_200c = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z)
        x_200c = r_b / ((1.0 + z_b) * r_200c[None, :, :])
        
        # Critical density broadcast to (1, 1, Nz) in physical units.
        rho_crit_z = jnp.atleast_1d(halo_model.cosmology.critical_density(z))[None, None, :]
        
        # Mass scaling logic
        m_200c_msun = m_b
        mass_ratio = m_200c_msun / 1e14 
       
        # Compute Shape Parameters (Equations A1, A2 from B16)
        rho0 = self.A_rho0 * mass_ratio**self.alpha_m_rho0 * (1 + z_b)**self.alpha_z_rho0 
        alpha = self.A_alpha * mass_ratio**self.alpha_m_alpha * (1 + z_b)**self.alpha_z_alpha 
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta 
        
        # Profile Shape Function (Nx, Nm, Nz)
        p_x = (x_200c / xc)**gamma * (1 + (x_200c / xc)**alpha)**(-(beta + gamma) / alpha)
        
        # Final result: M_sun / Mpc^3.
        rho_gas = rho0 * rho_crit_z * f_b * f_free * p_x 
        
        return rho_gas


    def u_k(self, halo_model, k, m, z):
        """
        Compute the projected Fourier-space density profile for halo-model calculations.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and halo-radius relation.
        k : float or jnp.ndarray
            Comoving wavenumber(s) in Mpc^-1.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Transformed profile with shape :math:`(N_k, N_M, N_z)`.
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        r_delta = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z)
        d_A_z = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))
        ell_delta = d_A_z[None, :] / r_delta

        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5

        vrms = jnp.sqrt(halo_model.cosmology.v_rms_squared(z))
        mu_e = 1.14
        prefactor = (
            4 * jnp.pi * r_delta**3 / mu_e
            * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :]
        )

        r = self.x[:, None, None] * r_delta[None, :, :] * (1.0 + z[None, None, :])
        k_native, u_k_native = self._u_k_hankel(halo_model, self.x, r, m, z)

        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None]))
        ell_native = k_native[:, None, None] * ell_delta[None, :, :]
        u_ell_val = prefactor[None, :, :] * u_ell_native

        def interp_single_column(target_x, native_x, native_y):
            return jnp.interp(target_x, native_x, native_y)

        vmapped_interp = jax.vmap(
            jax.vmap(interp_single_column, in_axes=(None, 1, 1), out_axes=1),
            in_axes=(1, 2, 2), out_axes=2
        )

        return vmapped_interp(ell_target, ell_native, u_ell_val)

jax.tree_util.register_pytree_node(
    B16DensityProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: B16DensityProfile._tree_unflatten(aux_data, children)
)

        

class NFWDensityProfile(DensityProfile):
    """
    Electron density profile based on `Navarro, Frenk & White (1997) <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_.

    The profile is evaluated as a function of the comoving radius
    :math:`r` and is obtained by scaling the NFW matter density by the cosmic
    baryon fraction,

    .. math::

        \\rho_e(r, M, z) = f_b \\, f_{\\mathrm{free}} \\, \\rho_{\\mathrm{NFW}}(r)
        \\tag{1}

    where

    .. math::

        \\rho_{\\mathrm{NFW}}(r)
        = \\frac{\\rho_s}{x_s \\left(1+x_s\\right)^2}
        \\tag{2}

    .. math::

        \\rho_s = \\frac{M}{4\\pi r_s^3}
        \\left[\\ln(1+c_\\Delta) - \\frac{c_\\Delta}{1+c_\\Delta}\\right]^{-1}
        \\tag{3}

    with :math:`x_s = r / r_s`, where :math:`r_s` has the same units as
    :math:`r`, and :math:`r_s = r_\\Delta / c_\\Delta`.

    The projected Fourier-space profile is evaluated as

    .. math::

        u_k(k, M, z) =
        4 \\pi \\, r_s^3 \\, \\frac{f_{\\mathrm{free}}}{\\mu_e}
        \\, \\frac{(1+z)^3}{\\chi^2(z)} \\, v_{\\mathrm{rms}}(z)
        \\int dx \\, x^2 \\, \\rho(x, M, z)
        \\, \\frac{\\sin\\!\\left[(k r_s) x\\right]}
        {(k r_s) x}
        \\tag{4}

    where :math:`x = r / r_s`, :math:`r_s` has the same units as
    :math:`r`, :math:`\\mu_e = 1.14`,
    :math:`f_{\\mathrm{free}} = 1`, and
    :math:`\\chi(z) = (1+z) d_A(z)` is the comoving distance.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_s` used to tabulate the profile and define the Hankel transform, with :math:`r_s` expressed in the same units as :math:`r`.
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

    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-density profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, halo radius, and concentration model.
        r : float or jnp.ndarray
            Comoving radius or radii in Mpc.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron-density profile with shape :math:`(N_r, N_M, N_z)`.
        """
        cparams = halo_model.cosmology._cosmo_params()
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        m_internal = m * cparams["h"]
       
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Get scale radius r_s
        r_delta = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z) * cparams["h"]
        c_delta = halo_model.concentration.c_delta(halo_model, m, z)
        r_s = r_delta / c_delta # (Nm, Nz)
        x_s = r[:, None, None] * cparams["h"] / ((1.0 + z[None, None, :]) * r_s[None, :, :])
        
        # Calculate rho_s
        m_nfw = jnp.log(1 + c_delta) - c_delta / (1 + c_delta) # (Nm, Nz)
        rho_s = m_internal[:, None] / (4 * jnp.pi * r_s**3 * m_nfw)    # (Nm, Nz)
        
        # Final broadcast to (Nx, Nm, Nz)
        rho_gas = f_b * rho_s[None, :, :] / (x_s * (1 + x_s)**2)
        
        return rho_gas
        

    def u_k(self, halo_model, k, m, z):
        """
        Compute the projected Fourier-space density profile for halo-model calculations.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, halo radius, and concentration model.
        k : float or jnp.ndarray
            Comoving wavenumber(s) in Mpc^-1.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Transformed profile with shape :math:`(N_k, N_M, N_z)`.
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        r_delta = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z)
        c_delta = halo_model.concentration.c_delta(halo_model, m, z)
        r_s = r_delta / c_delta
        d_A_z = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))
        ell_s = d_A_z[None, :] / r_s

        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5

        vrms = jnp.sqrt(halo_model.cosmology.v_rms_squared(z))
        mu_e = 1.14
        prefactor = (
            4 * jnp.pi * r_s**3 / mu_e
            * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :]
        )

        r = self.x[:, None, None] * r_s[None, :, :] * (1.0 + z[None, None, :])
        k_native, u_k_native = self._u_k_hankel(halo_model, self.x, r, m, z)

        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None]))
        ell_native = k_native[:, None, None] * ell_s[None, :, :]
        u_ell_val = prefactor[None, :, :] * u_ell_native

        def interp_single_column(target_x, native_x, native_y):
            return jnp.interp(target_x, native_x, native_y)

        vmapped_interp = jax.vmap(
            jax.vmap(interp_single_column, in_axes=(None, 1, 1), out_axes=1),
            in_axes=(1, 2, 2), out_axes=2
        )

        return vmapped_interp(ell_target, ell_native, u_ell_val)

    




class BCMDensityProfile(DensityProfile):
    """
    Electron density profile from `Schneider et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019JCAP...03..020S/abstract>`_, 
    also known as the Baryon Correction Model (BCM).

    The profile is evaluated as a function of the comoving radius
    :math:`r`, with shape defined relative to the physical virial radius
    through :math:`x_{\\mathrm{vir}} = r / r_{\\mathrm{vir}}`, where
    :math:`r_{\\mathrm{vir}}` has the same units as :math:`r`:

    .. math::

        \\rho_{\\mathrm{gas}}(r, M, z)
        = \\frac{f_b - f_\\star(M)}
        {\\left(1 + 10 x_{\\mathrm{vir}}\\right)^{\\beta_M(M, z)}
        \\left[1 + \\left(\\frac{x_{\\mathrm{vir}}}{\\theta_{\\mathrm{ej}}}\\right)^\\gamma\\right]^{(\\delta - \\beta_M(M, z))/\\gamma}}
        \\tag{1}

    where

    .. math::

        f_\\star(M) = f_{\\star, M_s}
        \\left(\\frac{M}{M_s}\\right)^{-\\eta_\\star}
        \\tag{2}

    .. math::

        \\beta_M(M, z) =
        \\frac{3 (M / M_c(z))^\\mu}{1 + (M / M_c(z))^\\mu}
        \\tag{3}

    .. math::

        \\log_{10} M_c(z) = \\log_{10} M_c \\, (1+z)^{\\nu_{\\log_{10} M_c}}
        \\tag{4}

    In the implementation, :math:`M_s = 2.5 \\times 10^{11} \\, M_\\odot / h`
    and :math:`f_{\\star, M_s} = 0.055` are fixed constants.

    The projected Fourier-space profile is evaluated as

    .. math::

        u_k(k, M, z) =
        4 \\pi \\, r_{\\mathrm{vir}}^3 \\, \\frac{f_{\\mathrm{free}}}{\\mu_e}
        \\, \\frac{(1+z)^3}{\\chi^2(z)} \\, v_{\\mathrm{rms}}(z)
        \\int dx \\, x^2 \\, \\rho(x, M, z)
        \\, \\frac{\\sin\\!\\left[(k r_{\\mathrm{vir}}) x\\right]}
        {(k r_{\\mathrm{vir}}) x}
        \\tag{5}

    where :math:`x = r / r_{\\mathrm{vir}}`, :math:`r_{\\mathrm{vir}}` has the
    same units as :math:`r`, :math:`\\mu_e = 1.14`,
    :math:`f_{\\mathrm{free}} = 1`, and
    :math:`\\chi(z) = (1+z) d_A(z)` is the comoving distance.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_{\\mathrm{vir}}` used to tabulate the profile and define the Hankel transform, with :math:`r_{\\mathrm{vir}}` expressed in the same units as :math:`r`.
    log10Mc : float
        Characteristic mass scale :math:`\\log_{10} M_c` controlling the gas fraction suppression.
    theta_ej : float
        Ejection-radius parameter :math:`\\theta_{\\mathrm{ej}}` in units of the virial radius.
    eta_star : float
        Stellar-fraction parameter :math:`\\eta_\\star`.
    delta : float
        Inner-slope parameter :math:`\\delta` of the gas profile.
    gamma : float
        Outer-slope parameter :math:`\\gamma` of the gas profile.
    mu : float
        Transition-shape parameter :math:`\\mu` controlling the stellar component.
    nu_log10Mc : float
        Redshift exponent :math:`\\nu_{\\log_{10} M_c}` of the characteristic mass scale.
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
        Return a new profile instance with updated BCM parameters.

        Parameters
        ----------
        log10Mc, theta_ej, eta_star, delta, gamma, mu, nu_log10Mc : float, optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

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

    def u_r(self, halo_model, r, m, z):
        """
        Compute the gas-density profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and virial radius.
        r : float or jnp.ndarray
            Comoving radius or radii in Mpc.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Gas-density profile with shape :math:`(N_r, N_M, N_z)`.
        """
       
        cparams = halo_model.cosmology._cosmo_params()
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # Broadcasting shapes: (Nx, 1, 1), (1, Nm, 1), (1, 1, Nz)
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        m_internal = m * cparams["h"]
        rb, mb, zb = r[:, None, None], m_internal[None, :, None], z[None, None, :]
        
        # This model is calibrated for the virial radius 
        r_vir = MassDefinition("vir", "critical").r_delta(halo_model.cosmology, m, z)
        x_vir = rb / ((1.0 + zb) * r_vir[None, :, :])
        
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
        denom1 = (1. + 10. * x_vir)**beta_m
        
        # Denominator 2: Ejected gas / transition,
        scaled_r = x_vir / self.theta_ej
        denom2 = (1. + (scaled_r)**self.gamma)**((self.delta - beta_m) / self.gamma)
    
        
        return num / (denom1 * denom2) 


    
    def u_k(self, halo_model, k, m, z):
        """
        Compute the projected Fourier-space gas-density profile for halo-model calculations.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and virial radius.
        k : float or jnp.ndarray
            Comoving wavenumber(s) in Mpc^-1.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Transformed profile with shape :math:`(N_k, N_M, N_z)`.
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        r_vir = MassDefinition("vir", "critical").r_delta(
            halo_model.cosmology,
            m,
            z,
        )
        d_A_z = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))
        ell_vir = d_A_z[None, :] / r_vir

        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5

        vrms = jnp.sqrt(halo_model.cosmology.v_rms_squared(z))
        mu_e = 1.14
        prefactor = (
            4 * jnp.pi * r_vir**3 / mu_e
            * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :]
        )

        r = self.x[:, None, None] * r_vir[None, :, :] * (1.0 + z[None, None, :])
        k_native, u_k_native = self._u_k_hankel(halo_model, self.x, r, m, z)

        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None]))
        ell_native = k_native[:, None, None] * ell_vir[None, :, :]
        u_ell_val = prefactor[None, :, :] * u_ell_native

        def interp_single_column(target_x, native_x, native_y):
            return jnp.interp(target_x, native_x, native_y)

        vmapped_interp = jax.vmap(
            jax.vmap(interp_single_column, in_axes=(None, 1, 1), out_axes=1),
            in_axes=(1, 2, 2), out_axes=2
        )

        return vmapped_interp(ell_target, ell_native, u_ell_val)

jax.tree_util.register_pytree_node(
    BCMDensityProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: BCMDensityProfile._tree_unflatten(aux_data, children)
)

    
   