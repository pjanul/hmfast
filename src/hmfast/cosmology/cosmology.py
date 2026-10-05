import difflib
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
from mcfit import TophatVar

from hmfast.cosmology.halofit import halofit
from hmfast.utils import Const, log_interp1d_extrap

jax.config.update("jax_enable_x64", True)


_C_KMS = Const._c_ / 1e3
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)
_Z_TAB_BG, _Z_TAB_PK = 20.0, 10.0  # top of the background and P(k) tables of a cosmology with no z limit

_FUNCTION_NAMES = ("hubble_parameter", "pk_linear", "angular_diameter_distance", "pk_nonlinear", "growth_factor")
_Functions = namedtuple("_Functions", _FUNCTION_NAMES)


class _KGrid:
    """A k grid and its sigma(R) transform; one instance per distinct grid, so treedefs compare by identity."""

    def __init__(self, k):
        self.k = k
        self.tophat = partial(TophatVar(k, lowring=True, backend="jax"), extrap=True)


_K_GRIDS = {}


def _k_grid(k):
    k = np.asarray(k, dtype=float)
    if k.tobytes() not in _K_GRIDS:
        _K_GRIDS[k.tobytes()] = _KGrid(k)
    return _K_GRIDS[k.tobytes()]


def _unknown(names, known, what):
    """Message for parameter names that do not exist, with the closest match for each."""
    lower = {k.lower(): k for k in known}
    hints = []
    for n in names:
        close = difflib.get_close_matches(n.lower(), lower, n=1, cutoff=0.8)
        hints.append(f"{n!r} (did you mean {lower[close[0]]!r}?)" if close else repr(n))
    return f"{', '.join(hints)} is not a parameter of {what}, whose parameters are {', '.join(known)}."


def standard_densities(p, *, m_ncdm=0.06, deg_ncdm=1.0, N_ur=3.046, T_cmb=2.7255, w0=-1.0):
    """
    Densities today of a flat universe, from ``H0``, ``omega_b`` and ``omega_cdm`` in ``p``.

    Photons at ``T_cmb``, ``N_ur`` massless neutrinos and ``deg_ncdm`` massive states of ``m_ncdm`` each
    (:math:`\\Omega_\\nu h^2 = m/93.14\\,\\mathrm{eV}`); dark energy closes the budget. A keyword that
    is also a key of ``p`` is read from ``p``.

    Parameters
    ----------
    p : dict
        Cosmological parameters, containing at least ``H0``, ``omega_b`` and ``omega_cdm``.
    m_ncdm : float
        Mass per massive neutrino state in eV.
    deg_ncdm : float
        Number of degenerate massive states.
    N_ur : float
        Number of massless neutrino species.
    T_cmb : float
        CMB temperature today in K.
    w0 : float
        Dark-energy equation of state.

    Returns
    -------
    dict
        ``Omega0_m``, ``Omega0_cb``, ``Omega0_b``, ``Omega0_r`` and ``w0``.
    """
    c = {"m_ncdm": m_ncdm, "deg_ncdm": deg_ncdm, "N_ur": N_ur, "T_cmb": T_cmb, "w0": w0}
    c.update({n: p[n] for n in c if n in p})
    G, sigma_B, Mpc_over_m = Const._G_, Const._sigma_B_, Const._Mpc_over_m_
    h = p["H0"] / 100.0
    omega_g = (4.0 * sigma_B / Const._c_ * c["T_cmb"] ** 4) / (3.0 * Const._c_**2 * 1e10 * h**2 / Mpc_over_m**2 / 8.0 / jnp.pi / G)
    omega_ur = c["N_ur"] * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * omega_g
    cb = (p["omega_b"] + p["omega_cdm"]) / h**2
    return {"Omega0_m": cb + c["deg_ncdm"] * c["m_ncdm"] / (93.14 * h**2), "Omega0_cb": cb,
            "Omega0_b": p["omega_b"] / h**2, "Omega0_r": omega_g + omega_ur, "w0": c["w0"]}


class Cosmology:
    """
    General cosmology class: cosmological parameters plus your own functions for :math:`H(z)` and
    :math:`P(k, z)`, and optionally :math:`D_A(z)`, the nonlinear :math:`P(k, z)` and :math:`D(z)`.
    Everything else in hmfast (growth, :math:`\\sigma(M)`, the halo model, statistics) is built on them.

    Use it to define a cosmology hmfast does not provide. Each function takes ``p``,
    the dict of the cosmology's parameter values, as its last argument and must be JAX-traceable. The functions
    are traced once on construction, so a parameter they read that does not exist, or a wrong output shape,
    raises a ``TypeError`` immediately. They must be valid at every redshift the calculation reaches.

    Parameters are passed as keywords, read as attributes (``cosmo.H0``) and changed with :meth:`update`.
    The densities today follow from ``H0``, ``omega_b``, ``omega_cdm``, ``m_ncdm``, ``N_ur``, ``w0`` and
    ``T_cmb`` for a flat universe; any further keyword becomes a parameter that the functions can read.

    Parameters
    ----------
    hubble_parameter : callable
        ``(z, p) -> H(z)`` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`, shape :math:`(N_z,)`.
    pk_linear : callable
        ``(k, z, p) -> P_L(k, z)`` of total matter in :math:`\\mathrm{Mpc}^3`, shape :math:`(N_k, N_z)`, ``k`` in :math:`\\mathrm{Mpc}^{-1}`.
    angular_diameter_distance : callable, optional
        ``(z, p) -> D_A(z)`` in Mpc. Default: integral of :math:`c/H` in a flat universe.
    pk_nonlinear : callable, optional
        As ``pk_linear``. Default: halofit (Takahashi et al. 2012; Bird et al. 2012) on ``pk_linear``.
    growth_factor : callable, optional
        ``(z, p) -> D(z)``, :math:`D(0) = 1`. Default: from ``pk_linear`` at :math:`k = 0.01\\,\\mathrm{Mpc}^{-1}`.
    k_grid : array_like, optional
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}` for :math:`\\sigma(M)` and FFTLog (static). Default: ``np.geomspace(1e-4, 50, 500)``.
    H0 : float
        Hubble constant in :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}` (default 68.0).
    omega_cdm : float
        Physical cold dark matter density, :math:`\\omega_{\\mathrm{cdm}} = \\Omega_{\\mathrm{cdm}} h^2` (default 0.12).
    omega_b : float
        Physical baryon density, :math:`\\omega_b = \\Omega_b h^2` (default 0.02246576).
    A_s : float
        Amplitude of the primordial scalar power spectrum (default :math:`2.1053 \\times 10^{-9}`).
    n_s : float
        Scalar spectral index (default 0.965).
    m_ncdm : float
        Neutrino mass in eV, as a single massive state (default 0.06).
    N_ur : float
        Effective number of ultra-relativistic species (default 3.046).
    w0 : float
        Dark energy equation of state (default -1).
    T_cmb : float
        CMB temperature today in K (default 2.7255).
    ncdm_mode : {"cb", "m"}
        Mean density, :math:`\\bar\\rho_{cb}` (default) or :math:`\\bar\\rho_m`, used for
        :math:`M(R)` in :math:`\\sigma(M)` and the mass function. :math:`\\sigma(M)` always uses
        the total-matter linear spectrum, and everything else uses total matter.
    extrapolate_k : bool
        If True (default), :meth:`pk` power-law extrapolates in log-log beyond
        ``k_grid``; if False, it returns NaN there.
        Also sets whether :func:`~hmfast.stats.corr_3d` and :func:`~hmfast.stats.corr_angular`
        power-law extrapolate their input beyond the ends of its grid.
    **params
        Further parameters the functions read, e.g. ``f_R0=1e-5``.

    Attributes
    ----------
    params : dict
        Parameter names and values; these are the pytree leaves.

    Examples
    --------
    >>> def hubble(z, p):
    ...     om = (p["omega_b"] + p["omega_cdm"]) / (p["H0"] / 100) ** 2
    ...     return p["H0"] * jnp.sqrt(om * (1 + z) ** 3 + 1 - om)
    >>> cosmo = Cosmology(hubble, my_pk, H0=70.0)
    """
    _STATIC = ("_functions", "_grid", "ncdm_mode", "extrapolate_k")
    _SETTINGS = ("ncdm_mode", "extrapolate_k")

    def __init__(self, hubble_parameter=None, pk_linear=None, *, angular_diameter_distance=None, pk_nonlinear=None,
                 growth_factor=None, k_grid=None,
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, A_s=2.1053e-9, n_s=0.965,
                 m_ncdm=0.06, N_ur=3.046, w0=-1.0, T_cmb=2.7255,
                 ncdm_mode="cb", extrapolate_k=True, **params):
        if hubble_parameter is None or pk_linear is None:
            raise TypeError("Cosmology needs hubble_parameter and pk_linear; "
                            "for the default emulated cosmology use CosmoPowerCosmology().")
        functions = _Functions(hubble_parameter, pk_linear, angular_diameter_distance, pk_nonlinear, growth_factor)
        for name, f in zip(_FUNCTION_NAMES, functions):
            if f is not None and not callable(f):
                raise TypeError(f"{name} must be callable or None; got {f!r}.")
        standard = dict(H0=H0, omega_cdm=omega_cdm, omega_b=omega_b, A_s=A_s, n_s=n_s, m_ncdm=m_ncdm, N_ur=N_ur,
                        w0=w0, T_cmb=T_cmb)
        self._setup({**standard, **params}, functions,
                    np.geomspace(1e-4, 50.0, 500) if k_grid is None else k_grid, ncdm_mode, extrapolate_k)
        self._check_functions()

    def _setup(self, params, functions, k_grid, ncdm_mode, extrapolate_k):
        if ncdm_mode not in ("cb", "m"):
            raise ValueError(f'ncdm_mode must be "cb" or "m", got {ncdm_mode!r}.')
        self.params = dict(params)
        self._functions = functions
        self._grid = _k_grid(k_grid)
        self.ncdm_mode = ncdm_mode
        self.extrapolate_k = extrapolate_k

    def _check_functions(self):
        """Trace the given functions once, so a missing parameter or a wrong shape fails here."""
        p = self.params
        z, k = jnp.zeros(3), jnp.asarray(self._grid.k[:4])
        args = {"hubble_parameter": ((z, p), (3,)), "pk_linear": ((k, z, p), (4, 3)),
                "angular_diameter_distance": ((z, p), (3,)), "pk_nonlinear": ((k, z, p), (4, 3)),
                "growth_factor": ((z, p), (3,))}
        for name, f in zip(_FUNCTION_NAMES, self._functions):
            if f is None:
                continue
            (inputs, shape) = args[name]
            try:
                out = jax.eval_shape(f, *inputs)
            except KeyError as err:
                raise TypeError(f"{name} reads " + _unknown([err.args[0]], tuple(p), type(self).__name__)) from None
            if out.shape != shape:
                raise TypeError(f"{name} must return shape {shape} for these inputs, got {out.shape}.")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(cls, lambda obj: obj._tree_flatten(), cls._tree_unflatten)

    def __getattr__(self, name):
        # Only reached when normal lookup fails, i.e. for parameter names.
        if name.startswith("_") or name == "params":
            raise AttributeError(name)
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __repr__(self):
        return f"{type(self).__name__}({self._repr_values()})"

    def _repr_values(self):
        return ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())

    # ------------------------------------------------------------------
    # PyTree registration
    # ------------------------------------------------------------------

    def _tree_flatten(self):
        # Children are the parameter values; their names, the functions and the settings are static.
        names = tuple(self.params)
        children = tuple(self.params[n] for n in names)
        return children, (names, *(getattr(self, a) for a in self._STATIC))

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # Bypass __init__ so no checks run under a trace.
        obj = cls.__new__(cls)
        names, *static = aux_data
        for attr, value in zip(cls._STATIC, static):
            setattr(obj, attr, value)
        obj.params = dict(zip(names, children))
        return obj

    def update(self, **changes):
        """
        Return a new cosmology of the same class with some parameters or settings changed.

        The cosmology itself is not modified. Changing parameter values does not recompile
        under ``jit``, so use this to sweep or differentiate parameters; changing a setting does.

        Parameters
        ----------
        **changes
            New values, by name, for any parameter in :attr:`params` (e.g. ``H0=70.0``), or for the
            settings ``ncdm_mode`` and ``extrapolate_k``; None leaves a value unchanged.

        Returns
        -------
        Cosmology
            New instance of the same class.

        Raises
        ------
        TypeError
            If a name is not a parameter of this cosmology.

        Examples
        --------
        >>> cosmo = cosmo.update(H0=70.0, omega_cdm=0.11)
        >>> dH = jax.grad(lambda h: cosmo.update(H0=h).hubble_parameter(1.0))(70.0)
        """
        settings = {n: changes.pop(n) for n in self._SETTINGS if n in changes}
        params = {n: v for n, v in changes.items() if v is not None}
        unknown = [n for n in params if n not in self.params]
        if unknown:
            raise TypeError(_unknown(unknown, tuple(self.params), type(self).__name__))
        if settings.get("ncdm_mode") not in (None, "cb", "m"):
            raise ValueError(f'ncdm_mode must be "cb" or "m", got {settings["ncdm_mode"]!r}.')
        children, aux_data = self._tree_flatten()
        obj = self._tree_unflatten(aux_data, children)
        for name, value in settings.items():
            if value is not None:
                setattr(obj, name, value)
        obj.params.update(params)
        return obj

    # ------------------------------------------------------------------
    # The functions and their defaults
    # ------------------------------------------------------------------

    def _call(self, name, *args):
        """Call the given function, or its default if it is None."""
        f = getattr(self._functions, name)
        return f(*args, self.params) if f is not None else getattr(self, "_default_" + name)(*args)

    def _densities(self):
        return standard_densities(self.params)

    def _default_angular_diameter_distance(self, z):
        z = jnp.asarray(z)
        x_max = jnp.log1p(z)[..., None]
        zp1 = jnp.exp(0.5 * x_max * (_GL_NODES + 1.0))
        integrand = _C_KMS * zp1 / self._call("hubble_parameter", (zp1 - 1.0).ravel()).reshape(zp1.shape)
        return 0.5 * x_max[..., 0] * jnp.sum(_GL_WEIGHTS * integrand, axis=-1) / (1.0 + z)

    def _default_pk_nonlinear(self, k, z):
        d = self._densities()
        k_grid = jnp.asarray(self._grid.k)
        H0 = self.params["H0"]
        f_nu = 1.0 - d["Omega0_cb"] / d["Omega0_m"]

        def one_z(z_i):
            z_i = jnp.atleast_1d(z_i)
            omega_m_z = d["Omega0_m"] * (1.0 + z_i[0]) ** 3 * (H0 / self._call("hubble_parameter", z_i)[0]) ** 2
            pk_lin = self._call("pk_linear", k_grid, z_i)[:, 0]
            pk_nl = halofit(k_grid, pk_lin, omega_m_z, d["Omega0_m"], d["w0"], f_nu, H0 / 100.0)
            return log_interp1d_extrap(k, k_grid, pk_nl)

        return jax.vmap(one_z, out_axes=1)(z)

    def _default_growth_factor(self, z, k0=1e-2):
        k = jnp.array([k0])
        return jnp.sqrt(self._call("pk_linear", k, z)[0] / self._call("pk_linear", k, jnp.zeros(1))[0, 0])

    # ------------------------------------------------------------------
    # shared grids
    # ------------------------------------------------------------------

    def _z_grid_bg(self):
        return jnp.linspace(0.0, _Z_TAB_BG, 5000, dtype=jnp.float64)

    def _z_grid_pk(self):
        return jnp.linspace(0.0, _Z_TAB_PK, 100, dtype=jnp.float64)

    def _pk_grid(self):
        # numpy, not jnp: fixed by the cosmology's class, so it stays concrete for mcfit to plan on.
        return self._grid.k

    @property
    def _tophat_instance(self):
        return self._grid.tophat

    @partial(jax.jit, static_argnums=(0,))
    def _compute_sigma_grid(self):
        """
        Compute the interpolation grid for :math:`\\sigma(M, z)`.

        The interpolation mass grid returned here is in physical
        :math:`M_\\odot`.

        Returns
        -------
        ln_x : array_like
            :math:`\\ln(1+z)` grid.
        ln_M : array_like
            :math:`\\ln M` grid.
        sigma_grid : array_like
            :math:`\\sigma(M, z)` values.
        """

        z_grid = self._z_grid_pk()
        cparams = self._cosmo_params()

        # Power spectra for all redshifts, shape: (n_k, n_z)
        k_grid = self._pk_grid()
        pk_grid = self.pk(k_grid, z_grid, linear=True)

        # Compute σ²(R, z) using the cached top-hat helper.
        R_grid, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()

        # Compute σ(R, z)
        sigma_grid = jnp.exp(0.5 * jnp.log(var))
        # Mass grid, shape: (n_R,)
        rho_crit_0 = cparams["Rho_crit_0"]
        # Lagrangian mass of the field halos form from (set by ncdm_mode).
        M_grid = 4.0 * jnp.pi / 3.0 * cparams['Omega0_halo'] * rho_crit_0 * (R_grid ** 3)

        ln_x = jnp.log1p(z_grid)
        ln_M = jnp.log(M_grid)
        return ln_x, ln_M, sigma_grid


    @partial(jax.jit, static_argnums=(0,))
    def sigma_m(self, m, z):
        """
        Evaluate :math:`\\sigma(M, z)` on a physical mass-redshift grid.

        The variance is defined by

        .. math::

            \\sigma^2(M, z) = \\frac{1}{2\\pi^2} \\int_0^\\infty dk\\, k^2\\,
            P_{\\mathrm{L}}(k, z)\\, \\hat{W}^2(kR),

        with Fourier-space top-hat window

        .. math::

            \\hat{W}(x) = \\frac{3}{x^3}\\left[\\sin x - x \\cos x\\right].

        Parameters
        ----------
        m : float or jnp.ndarray
            Halo mass or mass grid in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift or redshift grid.

        Returns
        -------
        float or jnp.ndarray
            Values of :math:`\\sigma(M, z)` with shape :math:`(N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """

        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        ln_x_grid, ln_M_grid, sigma_grid = self._compute_sigma_grid()
        sigma_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), jnp.log(sigma_grid))

        mm, zz = jnp.meshgrid(m, z, indexing='ij')
        pts = jnp.stack([jnp.log1p(zz), jnp.log(mm)], axis=-1)

        return jnp.squeeze(jnp.exp(sigma_interp(pts)))


    @partial(jax.jit, static_argnums=(0,))
    def sigma_r(self, r, z):
        """
        Evaluate :math:`\\sigma(R, z)` on a physical radius-redshift grid.

        The variance is defined by

        .. math::

            \\sigma^2(R, z) = \\frac{1}{2\\pi^2} \\int_0^\\infty dk\\, k^2\\,
            P_{\\mathrm{L}}(k, z)\\, \\hat{W}^2(kR),

        with Fourier-space top-hat window

        .. math::

            \\hat{W}(x) = \\frac{3}{x^3}\\left[\\sin x - x \\cos x\\right].

        Parameters
        ----------
        r : float or jnp.ndarray
            Comoving top-hat radius or radius grid in physical
            :math:`\\mathrm{Mpc}`.
        z : float or jnp.ndarray
            Redshift or redshift grid.

        Returns
        -------
        float or jnp.ndarray
            Values of :math:`\\sigma(R, z)` with shape :math:`(N_r, N_z)`,
            where singleton dimensions get squeezed before return.
        """

        r = jnp.atleast_1d(r)
        cparams = self._cosmo_params()
        rho_mean_0 = cparams["Omega0_halo"] * cparams["Rho_crit_0"]
        m = 4.0 * jnp.pi / 3.0 * rho_mean_0 * r**3

        return self.sigma_m(m, z)


    # ------------------------------------------------------------------
    # JAX-safe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _squeeze_single(values):
        return values[0] if values.shape[0] == 1 else values


    # ------------------------------------------------------------------
    # Cosmology
    # ------------------------------------------------------------------

    @jax.jit
    def hubble_parameter(self, z):
        """
        Hubble parameter :math:`H(z)` at redshift :math:`z`.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Hubble parameter(s) in :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}`
        """
        return self._squeeze_single(self._call("hubble_parameter", jnp.atleast_1d(z)))

    @jax.jit
    def angular_diameter_distance(self, z):
        """
        Angular diameter distance :math:`D_A(z)` at redshift :math:`z`.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Angular diameter distance(s) in :math:`\\mathrm{Mpc}`.
        """
        return self._squeeze_single(self._call("angular_diameter_distance", jnp.atleast_1d(z)))

    @jax.jit
    def sigma8(self, z):
        """
        :math:`\\sigma_8(z)` at redshift :math:`z`, from :meth:`sigma_r`.

        :math:`\\sigma_8(z)` is the dimensionless root-mean-square linear
        matter fluctuation amplitude in spheres of radius
        :math:`8 \\, \\mathrm{Mpc}/h`.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless :math:`\\sigma_8` value(s)
        """
        return self.sigma_r(8.0 / self._cosmo_params()["h"], z)

    @jax.jit
    def _cosmo_params(self):
        """
        Get the parameters together with derived background quantities.

        Returns
        -------
        dict
            Dictionary containing the parameters and the following derived quantities:

            - ``h``: Dimensionless Hubble parameter, :math:`h = H_0 / 100`
            - ``Omega_b``: Present-day baryon density parameter
            - ``Omega_cdm``: Present-day cold dark matter density parameter
            - ``Omega0_ncdm``: Present-day massive neutrino density parameter
            - ``Omega_Lambda``: Present-day dark energy density parameter
            - ``Omega0_m``: Present-day total matter density parameter
            - ``Omega0_r``: Present-day total radiation density parameter
            - ``w0``: Dark-energy equation of state
            - ``Omega0_m_nonu``: Present-day matter density parameter excluding
              massive neutrinos
            - ``Omega0_cb``: Present-day CDM+baryon density parameter
            - ``Omega0_halo``: Density parameter of the field halos form from:
              ``Omega0_cb`` if :attr:`ncdm_mode` is ``"cb"``,
              ``Omega0_m`` if it is ``"m"``
            - ``Rho_crit_0``: Present-day critical density in :math:`M_\\odot \\, \\mathrm{Mpc}^{-3}`

        """

        c, G, M_sun, Mpc_over_m = Const._c_, Const._G_, Const._M_sun_, Const._Mpc_over_m_
        d = self._densities()
        p = dict(self.params)
        p['h'] = p['H0'] / 100.
        p.update(Omega_b=d['Omega0_b'], Omega_cdm=d['Omega0_cb'] - d['Omega0_b'], Omega0_cb=d['Omega0_cb'],
                 Omega0_m=d['Omega0_m'], Omega0_ncdm=d['Omega0_m'] - d['Omega0_cb'], Omega0_r=d['Omega0_r'],
                 Omega_Lambda=1.0 - d['Omega0_m'] - d['Omega0_r'], w0=d['w0'])
        p['Omega0_m_nonu'] = p['Omega0_cb']
        # sigma(M) and the mass function use Omega0_halo; matter profiles, counterterms and lensing use Omega0_m.
        p['Omega0_halo'] = p['Omega0_cb'] if self.ncdm_mode == "cb" else p['Omega0_m']

        # Critical density
        H0 = p['H0'] / (c / 1e3) # Convert to H0 over c (c being in km/s)
        p['Rho_crit_0'] = (3.0 / (8.0 * jnp.pi * G * M_sun)) * Mpc_over_m * c**2 * H0**2

        return p

    @jax.jit
    def critical_density(self, z):
        """
        Get critical density :math:`\\rho_{\\mathrm{crit}}(z)` at redshift :math:`z`.

        .. math::

            \\rho_{\\mathrm{crit}}(z) = \\frac{3 H(z)^2}{8 \\pi G}

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Critical density in :math:`M_\\odot \\, \\mathrm{Mpc}^{-3}`
        """
        
        # Get Hubble parameter    
        H_z = self.hubble_parameter(z)
        
        # Convert H(z) from km/s/Mpc to s^-1 inside the prefactor.
        G, M_sun, Mpc_over_m = Const._G_, Const._M_sun_, Const._Mpc_over_m_
        rho_crit_factor = (3.0 / (8.0 * jnp.pi * G * M_sun)) * (1e6 * Mpc_over_m)
        
        return rho_crit_factor * H_z**2 
        
    @jax.jit
    def omega_m(self, z):
        """
        Total matter density parameter, including massive neutrinos.

        .. math::

            \\Omega_m(z) = \\Omega_{m,0}(1+z)^3 \\left[\\frac{H_0}{H(z)}\\right]^2

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
    
        Returns
        -------
        float or jnp.ndarray
            Dimensionless matter density at redshift :math:`z`
        """
       
        params = self._cosmo_params()
        om0 = params['Omega0_m']
        # rho_m(z)/rho_crit(z), exact for any expansion history via the emulated H(z).
        Omega_m_z = om0 * (1. + z)**3. * (self.H0 / self.hubble_parameter(z))**2.
        
        return Omega_m_z

    @partial(jax.jit, static_argnames=("prescription",))
    def delta_c(self, z, *, prescription="EdS"):
        """
        Spherical-collapse threshold :math:`\\delta_c(z)`.

        Supported prescriptions are:

        - ``"EdS"`` for the Einstein-de Sitter exact value,
          :math:`\\delta_c = \\frac{3}{20}(12\\pi)^{2/3}`.
        - ``"EdS_approx"`` for the standard Einstein-de Sitter approximation,
          :math:`\\delta_c = 1.686`.
        - ``"NS97"`` for the Nakamura and Suto (1997) fit,

          .. math::

              \\delta_c(z) = \\frac{3}{20}(12\\pi)^{2/3}
              \\left[1 + 0.012299 \\, \\log_{10}(\\Omega_m(z))\\right].

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s).
        prescription : str, optional
            Collapse-threshold prescription. Supported values are ``"EdS"``,
            ``"EdS_approx"``, and ``"NS97"``. Input is case-insensitive.

        Returns
        -------
        float or jnp.ndarray
            Collapse threshold evaluated at :math:`z`.
        """

        prescription_key = prescription.lower()
        delta_eds = (3.0 / 20.0) * jnp.power(12.0 * jnp.pi, 2.0 / 3.0)

        if prescription_key == "eds":
            return delta_eds
        if prescription_key == "eds_approx":
            return 1.686
        if prescription_key == "ns97":
            return delta_eds * (1.0 + 0.012299 * jnp.log10(self.omega_m(z)))

        raise ValueError(
            "Unknown delta_c prescription "
            f"{prescription!r}. Allowed values are: 'EdS', 'EdS_approx', 'NS97'."
        )

    @jax.jit
    def growth_factor(self, z):
        """
        Linear growth factor :math:`D(z)`, normalized to :math:`D(0)=1`.

        From the ``growth_factor`` function, or by default from the linear power spectrum at
        :math:`k = 0.01\\,\\mathrm{Mpc}^{-1}`.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless linear growth factor at :math:`z`, with shape
            :math:`(N_z,)`, where singleton dimensions get squeezed before
            return.
        """
        return jnp.squeeze(self._call("growth_factor", jnp.atleast_1d(z)))

    def _growth_rate_tabulated(self, z):
        """:math:`f(z)` from finite differences of :meth:`growth_factor` on the P(k) redshift table; NaN above it."""
        z_grid_pk = self._z_grid_pk()
        D_grid = self.growth_factor(z_grid_pk)
        a_grid = 1.0 / (1.0 + z_grid_pk)
        ln_D, ln_a = jnp.log(D_grid), jnp.log(a_grid)
        f_grid = jnp.gradient(ln_D, ln_a)

        # jnp.gradient is first order at the end nodes (no edge_order=2): use second-order one-sided stencils there.
        def one_sided(y0, y1, y2, x0, x1, x2):
            h1, h2 = x1 - x0, x2 - x1
            return -(2 * h1 + h2) * y0 / (h1 * (h1 + h2)) + (h1 + h2) * y1 / (h1 * h2) - h1 * y2 / (h2 * (h1 + h2))

        f_grid = f_grid.at[0].set(one_sided(*ln_D[:3], *ln_a[:3]))
        f_grid = f_grid.at[-1].set(one_sided(*ln_D[-1:-4:-1], *ln_a[-1:-4:-1]))

        return jnp.interp(z, z_grid_pk, f_grid, left=jnp.nan, right=jnp.nan)

    @jax.jit
    def growth_rate(self, z):
        """
        Linear growth rate

        .. math::

            f(z) = \\frac{d \\ln D}{d \\ln a}

        from finite differences of :meth:`growth_factor` on the P(k) redshift table, NaN above it.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless linear growth rate at :math:`z`, with shape
            :math:`(N_z,)`, where singleton dimensions get squeezed before
            return.
        """
        return jnp.squeeze(self._growth_rate_tabulated(jnp.atleast_1d(z)))

    @jax.jit
    def velocity_dispersion(self, z):
        """
        Compute the dimensionless velocity dispersion

        .. math::

            \\frac{1}{3} \\frac{v_\\mathrm{rms}^2}{c^2}

        from the linear growth factor and matter power spectrum.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless velocity dispersion at :math:`z`, equal to
            :math:`\\frac{1}{3} \\frac{v_\\mathrm{rms}^2}{c^2}`, with shape
            :math:`(N_z,)`, where singleton dimensions get squeezed before
            return.
        """
        
        z = jnp.atleast_1d(z)
        c_km_s = Const._c_ / 1e3
        k_grid = self._pk_grid()
        z_grid_pk = self._z_grid_pk()

        P_grid = self.pk(k_grid, z_grid_pk, linear=True).T
    
        a_grid = 1.0 / (1.0 + z_grid_pk)
        H_grid = self.hubble_parameter(z_grid_pk)
        f_grid = self.growth_rate(z_grid_pk)
    
        W_grid = f_grid * a_grid * H_grid / c_km_s
        integrand = (W_grid[:, None]**2 / 3) * P_grid * k_grid / (2 * jnp.pi**2)
        velocity_dispersion_grid = jax.scipy.integrate.trapezoid(integrand, x=jnp.log(k_grid), axis=1)

        return jnp.squeeze(jnp.interp(z, z_grid_pk, velocity_dispersion_grid, left=jnp.nan, right=jnp.nan))

    @jax.jit
    def comoving_volume_element(self, z):
        """
        Comoving volume element per unit redshift and solid angle.
    
        .. math::
    
            \\frac{dV}{dz\\,d\\Omega} = \\frac{(1+z)^2\\, D_A(z)^2 \\, c}{H(z)}
    
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
    
        Returns
        -------
        float or jnp.ndarray
            :math:`\\frac{dV}{dz\\,d\\Omega}` in :math:`\\mathrm{Mpc}^3 \\, \\mathrm{sr}^{-1}`
        """

        dAz = self.angular_diameter_distance(z)
        Hz = self.hubble_parameter(z)

        return (1 + z)**2 * dAz**2 * (Const._c_ / 1e3) / Hz
   

    # ------------------------------------------------------------------
    # Matter power spectra
    # ------------------------------------------------------------------

    def _pk_k_masked(self, k, z, linear):
        """P(k, z) of shape (N_k, N_z), NaN outside ``k_grid`` unless :attr:`extrapolate_k`."""
        pk_out = self._call("pk_linear" if linear else "pk_nonlinear", k, z)
        if not self.extrapolate_k:
            k_grid = self._pk_grid()
            in_k_bounds = (k >= k_grid[0]) & (k <= k_grid[-1])
            pk_out = jnp.where(in_k_bounds[:, None], pk_out, jnp.nan)
        return pk_out

    @partial(jax.jit, static_argnames=("linear",))
    def pk(self, k, z, *, linear=True):
        """
        Get the matter power spectrum :math:`P(k, z)` interpolated at
        requested wavenumbers `k` and redshifts `z`.

        Parameters
        ----------
        k : float or jnp.ndarray
            Wavenumber(s) in :math:`\\mathrm{Mpc}^{-1}` to evaluate the power spectrum at.
        z : float or jnp.ndarray
            Redshift(s) at which to evaluate the power spectrum.
        linear : bool
            True for linear :math:`P(k)`, False for nonlinear :math:`P(k)`.

        Returns
        -------
        P : jnp.ndarray
            Power spectrum values with shape :math:`(N_k, N_z)`, where singleton
            dimensions get squeezed before return.
        """
        return jnp.squeeze(self._pk_k_masked(jnp.atleast_1d(k), jnp.atleast_1d(z), linear))


jax.tree_util.register_pytree_node(Cosmology, lambda obj: obj._tree_flatten(), Cosmology._tree_unflatten)
