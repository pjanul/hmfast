import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy as jscipy
from mcfit import TophatVar
from hmfast.cosmology.engines import EmulatorEngine
from hmfast.utils import Const, dopri5_integrate
from functools import partial

jax.config.update("jax_enable_x64", True)


_TOPHATS = {}  # engine -> sigma(R) transform on its k grid, so equal engines give equal treedefs


def _tophat(engine):
    if engine not in _TOPHATS:
        _TOPHATS[engine] = partial(TophatVar(engine.k_grid, lowring=True, backend="jax"), extrap=True)
    return _TOPHATS[engine]


def _check_params(engine, names):
    """Raise if a name is not a parameter of the engine, saying whether it is fixed or unknown."""
    bad = [n for n in names if n not in engine.params]
    if bad:
        why = [f"{n} is fixed at {engine.fixed[n]:.6g}" if n in engine.fixed else f"{n} is not a parameter" for n in bad]
        raise TypeError(f"{'; '.join(why)} for {engine!r}, whose parameters are {', '.join(engine.params)}.")


class Cosmology:
    """
    Cosmology model: cosmological parameters plus the engine that computes from them.

    Provides access to cosmological parameters and engine-based predictions for distances, Hubble parameter, power spectra, CMB spectra, and derived parameters.
    Note that using parameters outside the engine's valid domain (e.g. emulator training bounds) will result in NaN outputs.
    The engine decides which parameters exist: ``print(engine)`` lists the ones it takes, with their
    defaults, and the values it fixes. Parameters are passed as keywords, read as attributes
    (``cosmo.H0``) and changed with :meth:`update`; fixed values can be read but not set.
    The parameters of the built-in engines are listed below.

    Attributes
    ----------
    engine : Engine
        Source of :math:`H(z)`, :math:`D_A(z)` and :math:`P(k, z)`, e.g. ``EmulatorEngine("ede:v2")``.
        Defaults to ``EmulatorEngine("lcdm:v1")``.
    params : dict
        The engine's parameters and their values; these are the pytree leaves.
    H0 : float
        Hubble constant at :math:`z = 0` in units of
        :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}`.
    omega_cdm : float
        Physical cold dark matter density,
        :math:`\\omega_{\\mathrm{cdm}} = \\Omega_{\\mathrm{cdm}} h^2`.
    omega_b : float
        Physical baryon density, :math:`\\omega_b = \\Omega_b h^2`.
    A_s : float
        Amplitude of the primordial scalar power spectrum, :math:`A_s`.
    n_s : float
        Scalar spectral index of primordial perturbations, :math:`n_s`.
    tau : float
        Optical depth to reionization, :math:`\\tau`.
    m_ncdm : float
        Non-cold dark matter mass, used if a massive-neutrino cosmological model is selected. This is the
        mass per state for ``"mnu-3states:v1"`` and the EDE sets, which have three degenerate states.
    N_ur : float
        Effective number of ultra-relativistic species, :math:`N_{\\mathrm{ur}}`,
        used if a model with additional radiation degrees of freedom is
        selected.
    w0 : float
        Present-day dark energy equation-of-state parameter :math:`w_0`,
        used if a cosmological model with dark energy equation-of-state
        parameter :math:`w_0` is selected.
    f_ede : float
        Maximum fractional contribution of early dark energy,
        :math:`f_{\\mathrm{ede}}`, used if an early dark energy cosmological
        model is selected.
    z_c : float
        Critical redshift for the early dark energy transition,
        :math:`z_c`, used if an early dark energy cosmological model is
        selected.
    theta_i : float
        Initial scalar field displacement for the early dark energy model,
        :math:`\\theta_i`, in radians, used if an early dark energy
        cosmological model is selected.
    r : float
        Tensor-to-scalar ratio, used if a cosmological model including primordial tensors is selected.
    T_cmb : float
        CMB temperature today in Kelvin; fixed at 2.7255 by the emulator engines.
    extrapolate_z : bool
        If True, redshifts above the engine's maximum are
        extrapolated. This is less accurate for early dark
        energy models, and for masses/neutrino content where the
        non-relativistic approximation for massive neutrinos breaks down
        before then.
    extrapolate_k : bool
        If True (default), :meth:`pk` power-law extrapolates in log-log beyond
        the engine's :math:`k` grid; if False, it returns NaN there.
        Also sets whether :func:`~hmfast.stats.corr_3d` and :func:`~hmfast.stats.corr_angular`
        power-law extrapolate their input beyond the ends of its grid.
    ncdm_mode : {"cb", "m"}
        Mean density, :math:`\\bar\\rho_{cb}` (default) or :math:`\\bar\\rho_m`, used for
        :math:`M(R)` in :math:`\\sigma(M)` and the mass function. :math:`\\sigma(M)` always uses
        the total-matter linear spectrum, and everything else uses total matter.
    """
    def __init__(self, engine=None, *, extrapolate_z=False, extrapolate_k=True, ncdm_mode="cb", **params):
        engine = engine if engine is not None else EmulatorEngine("lcdm:v1")
        if ncdm_mode not in ("cb", "m"):
            raise ValueError(f'ncdm_mode must be "cb" or "m", got {ncdm_mode!r}.')
        both = engine.params.keys() & engine.fixed.keys()
        if both:
            raise TypeError(f"{engine!r} declares {', '.join(sorted(both))} both as a parameter and fixed.")
        _check_params(engine, params)
        self.engine = engine
        self.params = {**engine.params, **params}
        self.extrapolate_z = extrapolate_z
        self.extrapolate_k = extrapolate_k
        self.ncdm_mode = ncdm_mode
        self._tophat_instance = _tophat(engine)

    def __getattr__(self, name):
        # Only reached when normal lookup fails, i.e. for parameter names.
        if name.startswith("_") or name in ("params", "engine"):
            raise AttributeError(name)
        if name in self.params:
            return self.params[name]
        if name in self.engine.fixed:
            return self.engine.fixed[name]
        raise AttributeError(f"{type(self).__name__!r} with {self.engine!r} has no attribute {name!r}")

    def __repr__(self):
        values = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"Cosmology({self.engine!r}, {values})"

    # ------------------------------------------------------------------
    # PyTree registration
    # ------------------------------------------------------------------

    def _tree_flatten(self):
        # Children are the engine's parameters; the engine and settings are static.
        children = tuple(self.params[name] for name in self.engine.params)
        aux_data = (self.engine, self.extrapolate_z, self.extrapolate_k, self.ncdm_mode, self._tophat_instance)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # Bypass __init__ so no checks run under a trace.
        obj = cls.__new__(cls)
        obj.engine, obj.extrapolate_z, obj.extrapolate_k, obj.ncdm_mode, obj._tophat_instance = aux_data
        obj.params = dict(zip(obj.engine.params, children))
        return obj

    def update(self, *, extrapolate_z=None, extrapolate_k=None, ncdm_mode=None, **params):
        """
        Return a new Cosmology instance with updated parameters.

        Parameters
        ----------
        extrapolate_z : bool or None
            If not None, replaces :attr:`extrapolate_z`.
        extrapolate_k : bool or None
            If not None, replaces :attr:`extrapolate_k`.
        ncdm_mode : {"cb", "m"} or None
            If not None, replaces :attr:`ncdm_mode`.
        **params
            New values for the engine's parameters; None leaves a parameter unchanged.

        Returns
        -------
        Cosmology
            New instance with updated parameters.
        """
        _check_params(self.engine, params)
        if ncdm_mode is not None and ncdm_mode not in ("cb", "m"):
            raise ValueError(f'ncdm_mode must be "cb" or "m", got {ncdm_mode!r}.')
        children, (engine, old_z, old_k, old_ncdm, tophat) = self._tree_flatten()
        obj = self._tree_unflatten((
            engine,
            old_z if extrapolate_z is None else extrapolate_z,
            old_k if extrapolate_k is None else extrapolate_k,
            old_ncdm if ncdm_mode is None else ncdm_mode,
            tophat,
        ), children)
        obj.params.update({name: value for name, value in params.items() if value is not None})
        return obj

    @partial(jax.jit, static_argnums=(0,))
    def _enforce_bounds(self, values):
        valid = self.engine.in_bounds(self._cosmo_params())
        values = jnp.asarray(values)
        return jnp.where(valid, values, jnp.full_like(values, jnp.nan))
                       

    # ------------------------------------------------------------------
    # shared grids 
    # ------------------------------------------------------------------

    def _z_grid_bg(self):
        return jnp.linspace(0.0, self.engine.z_max_bg, 5000, dtype=jnp.float64) 

    def _z_grid_pk(self):
        return jnp.linspace(0.0, self.engine.z_max_pk, 100, dtype=jnp.float64)     # z grid for Pk(z)

    def _pk_grid(self):
        # numpy, not jnp: fixed by the engine alone, so it stays concrete for mcfit to plan on.
        return self.engine.k_grid

    
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
    def _hz_flrw_calibrated(self, z_max):
        """Closed-form flat-FLRW H(z), rescaled to match the emulator's H(z_max) exactly."""
        def hz_flrw(z):
            p = self._cosmo_params()
            zp1 = 1.0 + z
            return self.H0 * jnp.sqrt(
                (p['Omega0_m_nonu'] + p['Omega0_ncdm']) * zp1 ** 3
                + p['Omega0_r'] * zp1 ** 4
                + p['Omega_Lambda'] * zp1 ** (3.0 * (1.0 + p['w0']))
            )
        # Force the non-extrapolated path to avoid recursing into this method via jnp.where's eager evaluation.
        correction = (self.update(extrapolate_z=False).hubble_parameter(z_max) / hz_flrw(z_max)) ** 2
        return lambda z: hz_flrw(z) * jnp.sqrt(correction)

    @jax.jit
    def hubble_parameter(self, z):
        """
        Get Hubble parameter :math:`H(z)` at redshift :math:`z` from the engine.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Hubble parameter(s) in :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}`
        """

        z_arr = jnp.atleast_1d(z)
        Hz = self.engine.compute_hubble_parameter(z_arr, self._cosmo_params())

        if self.extrapolate_z:
            z_max = self._z_grid_bg()[-1]
            Hz = jnp.where(z_arr > z_max, self._hz_flrw_calibrated(z_max)(z_arr), Hz)

        return self._enforce_bounds(self._squeeze_single(Hz))

    @jax.jit
    def angular_diameter_distance(self, z):
        """
        Get angular diameter distance :math:`D_A(z)` at redshift :math:`z` from the engine.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Angular diameter distance(s) in :math:`\\mathrm{Mpc}`.
        """

        z_arr = jnp.atleast_1d(z)
        DA = self.engine.compute_angular_diameter_distance(z_arr, self._cosmo_params())

        if self.extrapolate_z:
            z_max = self._z_grid_bg()[-1]
            # Same recursion hazard as _hz_flrw_calibrated -- force the non-extrapolated path for this boundary-anchor call.
            chi_max = self.update(extrapolate_z=False).angular_diameter_distance(z_max) * (1.0 + z_max)

            Hz_fn = self._hz_flrw_calibrated(z_max)
            x_max = jnp.log1p(z_max)

            def dchi_dx(x, chi):
                z = jnp.expm1(x)
                return (Const._c_ / 1e3) * (1.0 + z) / Hz_fn(z)

            # Integrate to each z independently -- a shared trajectory would be too sparse to interpolate safely.
            def chi_at(z_target):
                _, chi_traj = dopri5_integrate(dchi_dx, chi_max, x_max, jnp.log1p(z_target), rtol=1e-4, atol=1e-7, max_steps=16)
                return chi_traj[-1]

            DA = jnp.where(z_arr > z_max, jax.vmap(chi_at)(z_arr) / (1.0 + z_arr), DA)

        return self._enforce_bounds(self._squeeze_single(DA))

    @jax.jit
    def sigma8(self, z):
        """
        Get :math:`\\sigma_8(z)` at redshift :math:`z`, from the engine if it provides one, else from :meth:`sigma_r`.

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

        p = self._cosmo_params()
        s8 = self.engine.compute_sigma8(jnp.atleast_1d(z), p)
        if s8 is None:
            return self.sigma_r(8.0 / p["h"], z)
        return self._enforce_bounds(self._squeeze_single(s8))

    @jax.jit
    def _cosmo_params(self):
        """
        Get the engine's parameters and fixed values together with derived background quantities.
    
        Returns
        -------
        dict
            Dictionary containing the parameters, the fixed values and the following
            derived quantities (the densities come from :meth:`Engine.compute_densities`):
    
            - ``h``: Dimensionless Hubble parameter, :math:`h = H_0 / 100`
            - ``Omega_b``: Present-day baryon density parameter
            - ``Omega_cdm``: Present-day cold dark matter density parameter
            - ``Omega0_g``: Present-day photon density parameter
            - ``Omega0_ur``: Present-day ultra-relativistic density parameter
            - ``Omega0_ncdm``: Present-day massive neutrino density parameter
            - ``Omega_Lambda``: Present-day dark energy density parameter
            - ``Omega0_m``: Present-day total matter density parameter
            - ``Omega0_r``: Present-day total radiation density parameter
            - ``Omega0_m_nonu``: Present-day matter density parameter excluding
              massive neutrinos
            - ``Omega0_cb``: Present-day CDM+baryon density parameter
            - ``Omega0_halo``: Density parameter of the field halos form from:
              ``Omega0_cb`` if :attr:`ncdm_mode` is ``"cb"``,
              ``Omega0_m`` if it is ``"m"``
            - ``Rho_crit_0``: Present-day critical density in :math:`M_\\odot \\, \\mathrm{Mpc}^{-3}`
    
        """
    
        c, G, M_sun, Mpc_over_m = Const._c_, Const._G_, Const._M_sun_, Const._Mpc_over_m_
        p = {**self.engine.fixed, **self.params}
        p['h'] = p['H0'] / 100.
        p.update(self.engine.compute_densities(p))
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

    def _growth_ode(self, z, z_max):
        """
        Extend both the linear growth factor and the linear growth rate past the
        emulator's ``z_max`` by integrating the growth-rate Riccati equation forward
        (the numerically stable direction) from a deep-matter-domination seed via
        :func:`hmfast.utils.dopri5_integrate`, then calibrating the growth-factor branch
        to :meth:`growth_factor` at ``z_max``.

        The growth-rate branch needs no analogous calibration: the growth-factor
        calibration below is an additive shift to ``ln D``, and :math:`f = d\\ln D/d\\ln a`
        is a derivative, so it's invariant to that shift. ``f``'s own accuracy instead
        comes from the background dynamics (``H(z)``) already being calibrated via
        :meth:`_hz_flrw_calibrated`.

        Returns
        -------
        D : jnp.ndarray
            Extrapolated growth factor at ``z``, calibrated to :meth:`growth_factor` at ``z_max``.
        f : jnp.ndarray
            Extrapolated growth rate at ``z``, from the same ODE trajectory (uncalibrated --
            none is needed, see above).
        """
        Hz_fn = self._hz_flrw_calibrated(z_max)
        z_arr = jnp.atleast_1d(z)

        # Seed well past any requested z so D~a has room to relax onto the growing mode.
        z_start = 4.0 * jnp.maximum(jnp.max(z_arr), z_max)
        x_start = jnp.log(1.0 / (1.0 + z_start))
        x_max = jnp.log(1.0 / (1.0 + z_max))

        p = self._cosmo_params()
        Om0_m = p['Omega0_m_nonu'] + p['Omega0_ncdm']

        def ln_hubble(x):
            return jnp.log(Hz_fn(jnp.expm1(-x)))

        def growth_rhs(x, y):
            _, f = y
            dlnH_dx = jax.grad(ln_hubble)(x)
            z = jnp.expm1(-x)
            Om_a = Om0_m * (1.0 + z) ** 3 * self.H0 ** 2 / Hz_fn(z) ** 2
            return f, -f ** 2 - (2.0 + dlnH_dx) * f + 1.5 * Om_a

        # D_seed=1 (arbitrary norm), f_seed=1; max_h caps node spacing so batched interior queries interpolate safely.
        x_traj, (lnD_traj, f_traj) = dopri5_integrate(growth_rhs, (jnp.array(0.0), jnp.array(1.0)), x_start, x_max, rtol=1e-4, atol=1e-7, max_steps=64, max_h=0.3)

        # Force the non-extrapolated path to avoid recursing back into this method.
        norm = jnp.log(self.update(extrapolate_z=False).growth_factor(z_max)) - jnp.interp(x_max, x_traj, lnD_traj)

        x_target = jnp.log(1.0 / (1.0 + z_arr))
        lnD_target = jnp.interp(x_target, x_traj, lnD_traj) + norm
        f_target = jnp.interp(x_target, x_traj, f_traj)
        return jnp.exp(lnD_target), f_target

    @jax.jit
    def growth_factor(self, z):
        """
        Linear growth factor :math:`D(z)`, normalized to :math:`D(0)=1`.

        Without ``extrapolate_z``, NaN beyond the emulator's trained z-grid (plain
        ``jnp.interp`` clamps by default, which silently returned a flat, wrong value
        here previously -- fixed to NaN instead). With ``extrapolate_z=True``, the
        growth ODE branch below still applies, unchanged.

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

        z = jnp.atleast_1d(z)

        # These pk() calls must bypass self.extrapolate_z to avoid recursing back into this method via pk()'s own extrapolation branch.
        strict = self.update(extrapolate_z=False)
        k0 = 1e-2
        z_grid_pk = self._z_grid_pk()
        pk0_tmp = strict.pk(jnp.array([k0]), z_grid_pk, linear=True)
        pk0_grid = jnp.atleast_2d(pk0_tmp)[0, :]
        pk0_z0_tmp = strict.pk(jnp.array([k0]), jnp.array([0.0]), linear=True)
        D_grid = jnp.sqrt(pk0_grid / jnp.atleast_2d(pk0_z0_tmp)[0, 0])

        D = jnp.interp(z, z_grid_pk, D_grid, left=jnp.nan, right=jnp.nan)

        if self.extrapolate_z:
            z_max = self._z_grid_pk()[-1]
            D_ext, _ = self._growth_ode(z, z_max)
            D = jnp.where(z > z_max, D_ext, D)

        return jnp.squeeze(D)

    @jax.jit
    def growth_rate(self, z):
        """
        Linear growth rate

        .. math::

            f(z) = \\frac{d \\ln D}{d \\ln a}

        Without ``extrapolate_z``, NaN beyond the emulator's trained z-grid (matches
        ``growth_factor``'s convention). With ``extrapolate_z=True``, beyond the grid
        this reuses the same growth-rate ODE trajectory :meth:`growth_factor`'s own
        extrapolation branch already integrates (see :meth:`_growth_ode`) -- both a
        growth factor and a growth rate fall out of that single integration, so this
        doesn't integrate a second time.

        .. warning::
            When extrapolating beyond the emulator's redshift range, this assumes
            massive neutrinos behave as fully non-relativistic matter at every ``z``,
            and does not account for their transition to (semi-)relativistic behavior
            at the high redshifts this extrapolation reaches (e.g. approaching
            recombination).

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

        f = jnp.interp(z, z_grid_pk, f_grid, left=jnp.nan, right=jnp.nan)

        if self.extrapolate_z:
            z_max = z_grid_pk[-1]
            _, f_ext = self._growth_ode(z, z_max)
            f = jnp.where(z > z_max, f_ext, f)

        return jnp.squeeze(f)

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
            True for linear :math:`P(k)`, False for nonlinear :math:`P(k)` (source set by the engine).

        Returns
        -------
        P : jnp.ndarray
            Power spectrum values with shape :math:`(N_k, N_z)`, where singleton
            dimensions get squeezed before return.
        """
        k = jnp.atleast_1d(k)
        z = jnp.atleast_1d(z)

        z_max = self._z_grid_pk()[-1]
        in_z_bounds = z <= z_max
        if self.extrapolate_z:
            growth_ratio_sq = jnp.where(in_z_bounds, 1.0, (self.growth_factor(z) / self.growth_factor(z_max)) ** 2)
            z = jnp.where(in_z_bounds, z, z_max)

        k_grid = self._pk_grid()
        pk_out = self.engine.compute_pk(k, z, self._cosmo_params(), linear=linear)  # shape (Nk, Nz)
        if not self.extrapolate_k:
            in_k_bounds = (k >= k_grid[0]) & (k <= k_grid[-1])
            pk_out = jnp.where(in_k_bounds[:, None], pk_out, jnp.nan)
        if self.extrapolate_z:
            pk_out = pk_out * growth_ratio_sq[None, :]
        else:
            pk_out = jnp.where(in_z_bounds[None, :], pk_out, jnp.nan)
        return jnp.squeeze(self._enforce_bounds(pk_out))

    # ------------------------------------------------------------------
    # CMB angular power spectra
    # ------------------------------------------------------------------

    def cl_cmb(self, type, l):
        """
        Evaluate the CMB power spectrum of the specified type at requested multipoles `l` using the engine.
        This method can be used to evaluate :math:`C_\\ell^{TT}`, :math:`C_\\ell^{EE}`, :math:`C_\\ell^{TE}`, and :math:`C_\\ell^{\\phi\\phi}` by passing the appropriate `type` argument.

        Parameters
        ----------
        type : str
            Power-spectrum specifier, e.g. 'TT', 'EE', 'TE', or 'PP'. Case-insensitive.
        l : int or array-like
            Multipole(s) at which to evaluate C_ell.

        Returns
        -------
        jnp.ndarray
            C_ell for the requested type evaluated at `l`. Out-of-range `l` return NaN.
        """
        s = str(type).upper()
        if s not in ("TT", "EE", "TE", "PP"):
            raise ValueError(f"Unsupported spectrum type: {type}")

        return self._cl_jit(s, l)

    @partial(jax.jit, static_argnums=(1,))
    def _cl_jit(self, s, l):
        cl_out = self.engine.compute_cl_cmb(s, jnp.atleast_1d(l), self._cosmo_params())
        return jnp.squeeze(self._enforce_bounds(cl_out))

    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------

    @jax.jit
    def derived_parameters(self):
        """
        Get derived cosmological parameters from the engine.
    
        Returns
        -------
        dict
            Dictionary of derived parameters with the following keys:
    
            - '100*theta_s' : Sound horizon angle (in units of 1/100 radians)
            - 'sigma8' : Dimensionless RMS linear matter fluctuation in 8 Mpc/h spheres
            - 'YHe' : Primordial helium fraction
            - 'z_reio' : Redshift of reionization
            - 'Neff' : Effective number of relativistic species
            - 'tau_rec' : Conformal time at recombination (maximum visibility)
            - 'z_rec' : Redshift at recombination (maximum visibility)
            - 'rs_rec' : Comoving sound horizon at recombination [Mpc]
            - 'chi_rec' : Comoving distance to recombination [Mpc]
            - 'tau_star' : Conformal time at last scattering (optical depth = 1)
            - 'z_star' : Redshift at last scattering (optical depth = 1)
            - 'rs_star' : Comoving sound horizon at last scattering [Mpc]
            - 'chi_star' : Comoving distance to last scattering [Mpc]
            - 'rs_drag' : Comoving sound horizon at baryon drag [Mpc]
        """
        p = self._cosmo_params()
        out = self.engine.compute_derived_parameters(p)
        valid = self.engine.in_bounds(p)
        return {name: jnp.where(valid, value, jnp.asarray(jnp.nan, dtype=jnp.asarray(value).dtype)) for name, value in out.items()}


jax.tree_util.register_pytree_node(
    Cosmology,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: Cosmology._tree_unflatten(aux_data, children)
)