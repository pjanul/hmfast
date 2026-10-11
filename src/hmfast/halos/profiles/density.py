import jax
import jax.numpy as jnp
import numpy as np

from hmfast.halos.massdef import MassDefinition, mass_translator
from hmfast.halos.profiles import HaloProfile, HankelTransform


class DensityProfile(HaloProfile):
    """
    Parent electron density profile class from which density profile classes inherit.

    Child profile classes must implement :meth:`real` and :meth:`fourier`.
    """

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

    where :math:`X \\in \\{C, \\alpha, \\beta\\}`. Note that the scaling parameters must be calibrated with respect to a :math:`200c` mass definition.


    The Fourier-space density profile used by the halo model is evaluated as

    .. math::

        u_k(k, M, z) =
        4 \\pi \\, r_\\Delta^3 \\, (1+z)^3
        \\int dx \\, x^2 \\, \\rho(x, M, z)
        \\, \\frac{\\sin\\!\\left[(k r_\\Delta) x\\right]}
        {(k r_\\Delta) x}
        \\tag{3}

    where :math:`x = r / [(1+z) r_\\Delta]`.

    Attributes
    ----------
    x_range : tuple
        ``(x_min, x_max)`` spanning the dimensionless radial grid :math:`x = r / [(1+z) r_\\Delta]`, log-spaced internally to tabulate the profile and define the Hankel transform.
    n_x : int
        Number of log-spaced points in the transform grid.
    x_out : float
        Outer truncation radius in units of :math:`r_{200c}`. The profile is set to zero
        for :math:`x > x_{\\mathrm{out}}`.
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

    def __init__(
        self,
        *,
        x_range=(1e-2, 1.0),
        n_x=100,
        x_out=1.0,
        A_rho0=4000.0,
        A_alpha=0.88,
        A_beta=3.83,
        alpha_m_rho0=0.29,
        alpha_m_alpha=-0.03,
        alpha_m_beta=0.04,
        alpha_z_rho0=-0.66,
        alpha_z_alpha=0.19,
        alpha_z_beta=-0.025,
    ):

        x_grid = jnp.logspace(jnp.log10(x_range[0]), jnp.log10(x_range[1]), int(n_x))
        self._hankel = HankelTransform(x_grid, nu=0.5)
        self.x_out = x_out

        self.A_rho0, self.A_alpha, self.A_beta = A_rho0, A_alpha, A_beta
        self.alpha_m_rho0, self.alpha_m_alpha, self.alpha_m_beta = (
            alpha_m_rho0,
            alpha_m_alpha,
            alpha_m_beta,
        )
        self.alpha_z_rho0, self.alpha_z_alpha, self.alpha_z_beta = (
            alpha_z_rho0,
            alpha_z_alpha,
            alpha_z_beta,
        )

    @property
    def x_grid(self):
        return self._hankel.x

    @property
    def x_range(self):
        return (self._hankel.x[0], self._hankel.x[-1])

    @property
    def n_x(self):
        return self._hankel.x.shape[0]

    def _tree_flatten(self):
        leaves = (
            self.A_rho0,
            self.A_alpha,
            self.A_beta,
            self.alpha_m_rho0,
            self.alpha_m_alpha,
            self.alpha_m_beta,
            self.alpha_z_rho0,
            self.alpha_z_alpha,
            self.alpha_z_beta,
            self.x_out,
        )
        # Only the Hankel object: pytree aux must be hashable, which an array is not.
        aux_data = (self._hankel,)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        hankel, = aux_data
        obj = cls.__new__(cls)

        (
            obj.A_rho0,
            obj.A_alpha,
            obj.A_beta,
            obj.alpha_m_rho0,
            obj.alpha_m_alpha,
            obj.alpha_m_beta,
            obj.alpha_z_rho0,
            obj.alpha_z_alpha,
            obj.alpha_z_beta,
            obj.x_out,
        ) = leaves

        obj._hankel = hankel
        return obj

    def update(
        self,
        *,
        x_out=None,
        A_rho0=None,
        A_alpha=None,
        A_beta=None,
        alpha_m_rho0=None,
        alpha_m_alpha=None,
        alpha_m_beta=None,
        alpha_z_rho0=None,
        alpha_z_alpha=None,
        alpha_z_beta=None,
        x_range=None,
        n_x=None,
    ):
        """
        Return a new profile instance with updated Battaglia density parameters.

        Parameters
        ----------
        x_out : float, optional
            Replacement truncation radius in units of :math:`r_{200c}`.
        A_rho0, A_alpha, A_beta, alpha_m_rho0, alpha_m_alpha, alpha_m_beta, alpha_z_rho0, alpha_z_alpha, alpha_z_beta : float, optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.
        x_range : tuple, optional
            New ``(x_min, x_max)`` for the dimensionless radial grid. Rebuilds the Hankel transform on a fresh log-spaced grid.
        n_x : int, optional
            New number of log-spaced grid points. Rebuilds the Hankel transform.

        Returns
        -------
        B16DensityProfile
            New profile instance with updated parameters.
        """
        _, aux_data = self._tree_flatten()

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
            x_out if x_out is not None else self.x_out,
        )

        if x_range is not None or n_x is not None:
            new_x_range = x_range if x_range is not None else self.x_range
            new_n_x = n_x if n_x is not None else self.n_x
            x_grid = jnp.logspace(jnp.log10(new_x_range[0]), jnp.log10(new_x_range[1]), int(new_n_x))
            aux_data = (HankelTransform(x_grid, nu=0.5),)

        return self._tree_unflatten(aux_data, new_leaves)

    _PRESETS = {
        "agn": dict(
            A_rho0=4000.0,
            A_alpha=0.88,
            A_beta=3.83,
            alpha_m_rho0=0.29,
            alpha_m_alpha=-0.03,
            alpha_m_beta=0.04,
            alpha_z_rho0=-0.66,
            alpha_z_alpha=0.19,
            alpha_z_beta=-0.025,
        ),
        "shock": dict(
            A_rho0=1.9e4,
            A_alpha=0.70,
            A_beta=4.43,
            alpha_m_rho0=0.09,
            alpha_m_alpha=-0.017,
            alpha_m_beta=0.005,
            alpha_z_rho0=-0.95,
            alpha_z_alpha=0.27,
            alpha_z_beta=0.037,
        ),
    }

    def calibrate(self, model_key):
        """
        Return a new profile with shape parameters set to a named Battaglia et al. (2016) calibration.
        This acts as a wrapper around :meth:`update` that allows setting all nine shape parameters at once based on the calibration name.

        Parameters
        ----------
        model_key : str
            Case-insensitive calibration name.  Supported values: ``'agn'``, ``'shock'``.

        Returns
        -------
        B16DensityProfile
            New profile instance with all nine shape parameters replaced. The radial
            grid (``x_range``/``n_x``) and truncation radius ``x_out`` are preserved unchanged.

        """
        key = model_key.lower()
        if key not in self._PRESETS:
            raise ValueError(
                f"Unknown calibration '{model_key}'. Choose from: {list(self._PRESETS)}."
            )
        return self.update(**self._PRESETS[key])

    @jax.jit
    def real(self, halo_model, r, m, z):
        """
        Compute the electron-density profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron-density profile with shape :math:`(N_r, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """
        f_b = halo_model.cosmology.omega_b / (halo_model.cosmology.H0 / 100.0) ** 2 / halo_model.cosmology.omega_x(0.0, "matter")
        f_free = 1.0

        gamma = -0.2
        xc = 0.5

        # Ensure 1D and setup broadcasting shapes
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        r_b, m_b, z_b = r[:, None, None], m[None, :, None], z[None, None, :]

        mass_def_200c = MassDefinition(200, "critical")
        m_200c = jnp.reshape(
            mass_translator(
                halo_model.mass_def, mass_def_200c, halo_model.concentration
            )(halo_model.cosmology, m, z),
            (len(m), len(z)),
        )
        r_200c = jnp.reshape(
            mass_def_200c.r_delta(halo_model.cosmology, m_200c, z), (len(m), len(z))
        )

        x_200c = r_b / ((1.0 + z_b) * r_200c[None, :, :])

        # Critical density broadcast to (1, 1, Nz) in physical units.
        rho_crit_z = jnp.atleast_1d(halo_model.cosmology.critical_density(z))[
            None, None, :
        ]

        # Mass scaling logic
        m_200c_msun = m_200c[None, :, :]
        mass_ratio = m_200c_msun / 1e14

        # Compute Shape Parameters (Equations A1, A2 from B16)
        rho0 = (
            self.A_rho0 * mass_ratio**self.alpha_m_rho0 * (1 + z_b) ** self.alpha_z_rho0
        )
        alpha = (
            self.A_alpha
            * mass_ratio**self.alpha_m_alpha
            * (1 + z_b) ** self.alpha_z_alpha
        )
        beta = (
            self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b) ** self.alpha_z_beta
        )

        # Profile Shape Function (Nx, Nm, Nz)
        p_x = (x_200c / xc) ** gamma * (1 + (x_200c / xc) ** alpha) ** (
            -(beta + gamma) / alpha
        )

        rho_gas = rho0 * rho_crit_z * f_b * f_free * p_x
        # NaN-safe: a bare `x_200c <= self.x_out` would silently zero out NaN instead of propagating it.
        rho_gas = jnp.where(jnp.isnan(x_200c) | (x_200c <= self.x_out), rho_gas, 0.0)

        return jnp.squeeze(rho_gas)

    @jax.jit
    def fourier(self, halo_model, k, m, z):
        """
        Compute the projected Fourier-space density profile for halo-model calculations.

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
        mass_def_200c = MassDefinition(200, "critical")
        m_200c = jnp.reshape(mass_translator(halo_model.mass_def, mass_def_200c, halo_model.concentration)(halo_model.cosmology, m, z), (len(m), len(z)))
        r_delta = jnp.reshape(mass_def_200c.r_delta(halo_model.cosmology, m_200c, z), (len(m), len(z)))
        halo_model_200c = halo_model.update(mass_def=mass_def_200c)
        return self._fourier_via_hankel_transform(halo_model_200c, k, m_200c, z, r_delta)


jax.tree_util.register_pytree_node(
    B16DensityProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: B16DensityProfile._tree_unflatten(aux_data, children),
)
