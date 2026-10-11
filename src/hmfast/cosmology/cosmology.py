import difflib
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from mcfit import TophatVar

from hmfast.utils import Const

jax.config.update("jax_enable_x64", True)


_FUNCTION_NAMES = ("hubble_parameter", "pk_linear", "angular_diameter_distance", "pk_nonlinear", "growth_factor")
_Functions = namedtuple("_Functions", _FUNCTION_NAMES)
# Cosmological parameters every cosmology class lists; each is free, fixed or absent (None) in a given class.
_PARAM_NAMES = ("H0", "omega_cdm", "omega_b", "Omega_k", "A_s", "n_s", "tau", "m_ncdm", "N_ur", "w0", "wa", "T_cmb",
                "f_ede", "z_c", "theta_i", "r")


def _unknown(names, known, what):
    """Message for parameter names that do not exist, with the closest match for each."""
    lower = {k.lower(): k for k in known}
    hints = []
    for n in names:
        close = difflib.get_close_matches(n.lower(), lower, n=1, cutoff=0.8)
        hints.append(f"{n!r} (did you mean {lower[close[0]]!r}?)" if close else repr(n))
    return f"{', '.join(hints)} is not a parameter of {what}, whose parameters are {', '.join(known)}."


class Cosmology:
    """
    General cosmology class: cosmological parameters plus your own functions for :math:`H(z)`, the linear
    :math:`P(k, z)`, :math:`D_A(z)`, the nonlinear :math:`P(k, z)` and :math:`D(z)`.
    Everything else in hmfast (:math:`\\sigma(M)`, the halo model, statistics) is built on them.

    Use it to define a cosmology hmfast does not provide. Each function takes ``p``,
    the dict of the cosmology's parameter values, as its last argument and must be JAX-traceable. The functions
    are traced once on construction, so a parameter they read that does not exist, or a wrong output shape,
    raises a ``TypeError`` immediately. They must be valid at every redshift the calculation reaches, and
    ``growth_factor`` must be differentiable in :math:`z`, with each value depending only on its own redshift
    (:meth:`growth_rate` is its derivative).

    Parameters are passed as keywords, read as attributes (``cosmo.H0``) and changed with :meth:`update`.
    The densities follow from ``H0``, ``omega_b``, ``omega_cdm``, ``Omega_k``, ``m_ncdm``, ``N_ur``, ``w0``,
    ``wa`` and ``T_cmb`` (see :meth:`rho_x`); any further keyword becomes a parameter that the functions can read.

    Every cosmology class takes the parameters below. In a given class each one is free (changed with
    :meth:`update`), fixed by the class (read-only) or absent (``None``); for instance
    ``CosmoPowerCosmology("lcdm:v1")`` fixes ``m_ncdm`` at 0.06 and has no ``f_ede``. Here a parameter
    left at ``None`` is absent, and one given a value is free.

    Parameters
    ----------
    hubble_parameter : callable
        ``(z, p) -> H(z)`` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`, shape :math:`(N_z,)`.
    pk_linear : callable
        ``(k, z, p) -> P_L(k, z)`` of total matter in :math:`\\mathrm{Mpc}^3`, shape :math:`(N_k, N_z)`, ``k`` in :math:`\\mathrm{Mpc}^{-1}`.
    angular_diameter_distance : callable
        ``(z, p) -> D_A(z)`` in Mpc, shape :math:`(N_z,)`.
    pk_nonlinear : callable
        ``(k, z, p) -> P_{NL}(k, z)``, as ``pk_linear`` for the nonlinear spectrum.
    growth_factor : callable
        ``(z, p) -> D(z)``, linear growth factor normalised to :math:`D(0) = 1`, shape :math:`(N_z,)`.
    H0 : float
        Hubble constant in :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}` (default 68.0).
    omega_cdm : float
        Physical cold dark matter density, :math:`\\omega_{\\mathrm{cdm}} = \\Omega_{\\mathrm{cdm}} h^2` (default 0.12).
    omega_b : float
        Physical baryon density, :math:`\\omega_b = \\Omega_b h^2` (default 0.02246576).
    Omega_k : float
        Curvature density parameter today, positive for an open universe (default 0). Distances extrapolated
        beyond the redshift range and the projected statistics assume a flat geometry.
    A_s : float
        Amplitude of the primordial scalar power spectrum (default :math:`2.1053 \\times 10^{-9}`).
    n_s : float
        Scalar spectral index (default 0.965).
    tau : float or None
        Optical depth to reionization (default None).
    m_ncdm : float
        Mass in eV of the massive neutrino state; this class has one (default 0.06).
    N_ur : float
        Effective number of ultra-relativistic species (default 3.046).
    w0 : float
        Dark energy equation of state today, :math:`w(a) = w_0 + w_a (1 - a)` (default -1).
    wa : float
        Rate of change of the dark energy equation of state, :math:`w(a) = w_0 + w_a (1 - a)` (default 0).
    T_cmb : float
        CMB temperature today in K (default 2.7255).
    f_ede : float or None
        Maximum fractional contribution of early dark energy (default None).
    z_c : float or None
        Critical redshift of the early dark energy transition (default None).
    theta_i : float or None
        Initial early dark energy field displacement in radians (default None).
    r : float or None
        Tensor-to-scalar ratio (default None).
    ncdm_mode : {"cb", "m"}
        Mean density, :math:`\\bar\\rho_{cb}` (default) or :math:`\\bar\\rho_m`, used for
        :math:`M(R)` in :math:`\\sigma(M)` and the mass function. :math:`\\sigma(M)` always uses
        the total-matter linear spectrum, and everything else uses total matter.
    k_range : tuple of float
        Range in :math:`\\mathrm{Mpc}^{-1}` of the wavenumber grid used for :math:`\\sigma(M)` and other k integrals (default (1e-4, 50)).
    n_k : int
        Number of log-spaced points in that grid (default 500).
    **params
        Further parameters the functions read, e.g. ``f_R0=1e-5``.

    Attributes
    ----------
    params : dict
        Names and values of the free parameters; these are the pytree leaves.

    Examples
    --------
    >>> def hubble(z, p):
    ...     om = (p["omega_b"] + p["omega_cdm"]) / (p["H0"] / 100) ** 2
    ...     return p["H0"] * jnp.sqrt(om * (1 + z) ** 3 + 1 - om)
    >>> cosmo = Cosmology(hubble, my_pk, my_da, my_pknl, my_growth, H0=70.0)
    """
    _STATIC = ("_functions", "_k", "ncdm_mode")
    _SETTINGS = ("ncdm_mode",)
    _deg_ncdm = 1  # number of massive neutrino states of mass m_ncdm each

    def __init__(self, hubble_parameter, pk_linear, angular_diameter_distance, pk_nonlinear, growth_factor, *,
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, Omega_k=0.0, A_s=2.1053e-9, n_s=0.965, tau=None,
                 m_ncdm=0.06, N_ur=3.046, w0=-1.0, wa=0.0, T_cmb=2.7255,
                 f_ede=None, z_c=None, theta_i=None, r=None,
                 ncdm_mode="cb", k_range=(1e-4, 50.0), n_k=500, **params):
        functions = _Functions(hubble_parameter, pk_linear, angular_diameter_distance, pk_nonlinear, growth_factor)
        for name, f in zip(_FUNCTION_NAMES, functions):
            if not callable(f):
                raise TypeError(f"{name} must be callable; got {f!r}.")
        standard = dict(H0=H0, omega_cdm=omega_cdm, omega_b=omega_b, Omega_k=Omega_k, A_s=A_s, n_s=n_s, tau=tau,
                        m_ncdm=m_ncdm, N_ur=N_ur, w0=w0, wa=wa, T_cmb=T_cmb, f_ede=f_ede, z_c=z_c, theta_i=theta_i, r=r)
        standard = {n: v for n, v in standard.items() if v is not None}
        self._setup({**standard, **params}, functions, np.geomspace(*k_range, n_k), ncdm_mode)
        self._check_functions()

    def _setup(self, params, functions, k_grid, ncdm_mode):
        if ncdm_mode not in ("cb", "m"):
            raise ValueError(f'ncdm_mode must be "cb" or "m", got {ncdm_mode!r}.')
        self.params = dict(params)
        self._functions = functions
        self._k = tuple(np.asarray(k_grid, dtype=float))  # static, so equal grids give equal treedefs
        self.ncdm_mode = ncdm_mode

    def _fixed_params(self):
        """Parameters the class fixes, by name; they enter the densities but are not leaves."""
        return {}

    def _label(self):
        """How errors name this cosmology."""
        return type(self).__name__

    def _not_parameters(self, names):
        """Message for names that cannot be set: fixed by the class, absent, or unknown."""
        fixed = self._fixed_params()
        for n in names:
            if n in fixed:
                return f"{n!r} is fixed at {fixed[n]:g} in {self._label()}, whose free parameters are {', '.join(self.params)}."
        return _unknown(names, tuple(self.params), self._label())

    def _check_functions(self):
        """Trace the given functions once, so a missing parameter or a wrong shape fails here."""
        p = self.params
        z, k = jnp.zeros(3), jnp.asarray(self._pk_grid()[:4])
        args = {"hubble_parameter": ((z, p), (3,)), "pk_linear": ((k, z, p), (4, 3)),
                "angular_diameter_distance": ((z, p), (3,)), "pk_nonlinear": ((k, z, p), (4, 3)),
                "growth_factor": ((z, p), (3,))}
        for name, f in zip(_FUNCTION_NAMES, self._functions):
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
        if name in _PARAM_NAMES:
            return self._fixed_params().get(name)
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
            setting ``ncdm_mode``; None leaves a value unchanged.

        Returns
        -------
        Cosmology
            New instance of the same class.

        Raises
        ------
        TypeError
            If a name is not a free parameter of this cosmology.

        Examples
        --------
        >>> cosmo = cosmo.update(H0=70.0, omega_cdm=0.11)
        >>> dH = jax.grad(lambda h: cosmo.update(H0=h).hubble_parameter(1.0))(70.0)
        """
        settings = {n: changes.pop(n) for n in self._SETTINGS if n in changes}
        params = {n: v for n, v in changes.items() if v is not None}
        unknown = [n for n in params if n not in self.params]
        if unknown:
            raise TypeError(self._not_parameters(unknown))
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
    # The functions
    # ------------------------------------------------------------------

    def _call(self, name, *args):
        """Call the cosmology's function for ``name`` with its parameters."""
        return getattr(self._functions, name)(*args, self.params)

    @classmethod
    def _from_params(cls, params):
        """Cosmology of this class with the given parameter values, for functions that only receive ``p``."""
        obj = cls.__new__(cls)
        obj.params = dict(params)
        return obj

    # ------------------------------------------------------------------
    # shared grids
    # ------------------------------------------------------------------

    def _pk_grid(self):
        # numpy, not jnp: fixed by the cosmology's class, so it stays concrete for mcfit to plan on.
        return np.asarray(self._k)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_sigma_grid(self, z):
        """
        Compute :math:`\\sigma(M, z)` at the requested redshifts, on a mass grid
        for interpolation in mass only.

        The interpolation mass grid returned here is in physical
        :math:`M_\\odot`.

        Returns
        -------
        ln_M : array_like
            :math:`\\ln M` grid.
        sigma_grid : array_like
            :math:`\\sigma(M, z)` values, shape :math:`(N_z, N_M)`.
        """

        z_grid = jnp.atleast_1d(z)

        # Power spectra at the requested redshifts, shape: (n_k, n_z)
        k_grid = self._pk_grid()
        pk_grid = jnp.reshape(self.pk(k_grid, z_grid, linear=True), (len(k_grid), len(z_grid)))

        # mcfit plans on concrete values, so build the transform eagerly even under a trace.
        with jax.ensure_compile_time_eval():
            tophat = TophatVar(k_grid, lowring=True, backend="jax")
        R_grid, var = jax.vmap(partial(tophat, extrap=True), in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()

        # Compute σ(R, z)
        sigma_grid = jnp.exp(0.5 * jnp.log(var))
        # Mass grid, shape: (n_R,)
        # Lagrangian mass of the field halos form from: CDM and baryons, or all matter (set by ncdm_mode).
        rho_mean_0 = self.rho_x(0.0, "matter")
        if self.ncdm_mode == "cb":
            rho_mean_0 = rho_mean_0 - self.rho_x(0.0, "neutrinos_massive")
        M_grid = 4.0 * jnp.pi / 3.0 * rho_mean_0 * (R_grid ** 3)

        ln_M = jnp.log(M_grid)
        return ln_M, sigma_grid


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

        # σ is computed at each requested z, so only the mass is interpolated.
        ln_M_grid, sigma_grid = self._compute_sigma_grid(z)
        ln_sigma = jax.vmap(lambda row: jnp.interp(jnp.log(m), ln_M_grid, row, left=jnp.nan, right=jnp.nan))(jnp.log(sigma_grid))

        return jnp.squeeze(jnp.exp(ln_sigma.T))


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
        rho_mean_0 = self.rho_x(0.0, "matter")
        if self.ncdm_mode == "cb":
            rho_mean_0 = rho_mean_0 - self.rho_x(0.0, "neutrinos_massive")
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
        :math:`\\sigma_8(z)` at redshift :math:`z`.

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
        return self._sigma8(z)

    def _sigma8(self, z):
        return self.sigma_r(8.0 / (self.H0 / 100.0), z)

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
        
    def rho_x(self, z, species):
        """
        Physical (not comoving) density of a species at redshift :math:`z`, from the cosmological parameters.

        Matter (massive neutrinos included) dilutes as :math:`(1+z)^3`, photons and massless neutrinos as
        :math:`(1+z)^4` and curvature as :math:`(1+z)^2`; dark energy follows :math:`w(a) = w_0 + w_a (1 - a)`
        and closes the budget today.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s).
        species : str
            ``"matter"`` (cold dark matter, baryons and massive neutrinos), ``"dark_energy"``, ``"radiation"``
            (photons), ``"neutrinos_rel"`` (massless neutrinos), ``"neutrinos_massive"``, ``"curvature"`` or
            ``"critical"`` (the sum of all of them, i.e. the critical density these parameters imply).

        Returns
        -------
        jnp.ndarray
            Physical density in :math:`M_\\odot \\, \\mathrm{Mpc}^{-3}`, with the shape of ``z``.
        """
        p = {**self._fixed_params(), **self.params}
        sigma_B, c, M_sun, Mpc = Const._sigma_B_, Const._c_, Const._M_sun_, Const._Mpc_over_m_
        rho_crit_100 = 3.0e10 * Mpc / (8.0 * np.pi * Const._G_ * M_sun)  # critical density for H0 = 100 km/s/Mpc
        rho_crit_0 = rho_crit_100 * (p["H0"] / 100.0) ** 2
        rho_photons = 4.0 * sigma_B * p.get("T_cmb", 0.0) ** 4 / c**3 * Mpc**3 / M_sun
        rho_nu = self._deg_ncdm * p.get("m_ncdm", 0.0) / 93.14 * rho_crit_100
        rho_matter = (p["omega_b"] + p["omega_cdm"]) * rho_crit_100 + rho_nu
        rho_ur = p.get("N_ur", 0.0) * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * rho_photons
        rho_k = p.get("Omega_k", 0.0) * rho_crit_0
        w0, wa = p.get("w0", -1.0), p.get("wa", 0.0)

        zp1 = 1.0 + jnp.asarray(z)
        # Each species as (density today, dilution with redshift).
        components = {"matter": (rho_matter, zp1**3), "neutrinos_massive": (rho_nu, zp1**3),
                      "radiation": (rho_photons, zp1**4), "neutrinos_rel": (rho_ur, zp1**4), "curvature": (rho_k, zp1**2),
                      "dark_energy": (rho_crit_0 - rho_matter - rho_photons - rho_ur - rho_k,
                                      zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * (1.0 - 1.0 / zp1)))}

        if species == "critical":
            return sum(rho_0 * f for name, (rho_0, f) in components.items() if name != "neutrinos_massive")
        if species not in components:
            raise ValueError(f"species must be one of {', '.join([*components, 'critical'])}; got {species!r}.")
        rho_0, f = components[species]
        return rho_0 * f

    def omega_x(self, z, species):
        """
        Density parameter of a species at redshift :math:`z`, from the cosmological parameters.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s).
        species : str
            As in :meth:`rho_x`.

        Returns
        -------
        jnp.ndarray
            :math:`\\rho_x(z) / \\rho_{\\mathrm{crit}}(z)`, both physical and from :meth:`rho_x`, with the shape of ``z``.
        """
        return self.rho_x(z, species) / self.rho_x(z, "critical")

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
       
        om0 = self.omega_x(0.0, "matter")
        # rho_m(z)/rho_crit(z) for any expansion history, through the cosmology's H(z).
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

    @jax.jit
    def growth_rate(self, z):
        """
        Linear growth rate

        .. math::

            f(z) = \\frac{d \\ln D}{d \\ln a},

        differentiated from :meth:`growth_factor`.

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
        z = jnp.atleast_1d(z)
        _, dlnD_dz = jax.jvp(lambda x: jnp.log(self._call("growth_factor", x)), (z,), (jnp.ones_like(z),))
        return jnp.squeeze(-(1.0 + z) * dlnD_dz)

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
        k = self._pk_grid()
        P = jnp.reshape(self.pk(k, z, linear=True), (len(k), len(z)))
        W = jnp.atleast_1d(self.growth_rate(z)) * jnp.atleast_1d(self.hubble_parameter(z)) / (1.0 + z) / (Const._c_ / 1e3)
        integrand = W[None, :] ** 2 / 3 * P * k[:, None] / (2 * jnp.pi**2)
        return jnp.squeeze(jax.scipy.integrate.trapezoid(integrand, x=jnp.log(k), axis=0))

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

    @partial(jax.jit, static_argnames=("linear",))
    def pk(self, k, z, *, linear=True):
        """
        Matter power spectrum :math:`P(k, z)` at wavenumbers ``k`` and redshifts ``z``.

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
        return jnp.squeeze(self._call("pk_linear" if linear else "pk_nonlinear", jnp.atleast_1d(k), jnp.atleast_1d(z)))


jax.tree_util.register_pytree_node(Cosmology, lambda obj: obj._tree_flatten(), Cosmology._tree_unflatten)
