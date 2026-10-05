"""CosmoPower emulators of CLASS as a Cosmology."""
import os
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.cosmology import Cosmology, _Functions, _unknown, standard_densities
from hmfast.cosmology.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.download import _get_default_data_path
from hmfast.utils import Const, dopri5_integrate, log_interp1d_extrap

_C_KMS = Const._c_ / 1e3

_LCDM_NAMES = ("H0", "omega_cdm", "omega_b", "A_s", "n_s")
_STANDARD_CONSTANTS = MappingProxyType({"m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "T_cmb": 2.7255, "deg_ncdm": 1.0})
_EDE = ("m_ncdm", "N_ur", "f_ede", "z_c", "theta_i", "r")
_V1 = {"z_max_pk": 5.0, "grid": "v1", "log_bg": False, "n_s": (0.8, 1.2), "deg_ncdm": 1.0}

# "free": extension parameters the set takes; "grid"/"log_bg": output layout of the networks.
_EMULATOR_SETS = {
    "lcdm:v1":        {**_V1, "subdir": "lcdm", "suffix": "v1", "free": (), "n_s": (0.8812, 1.0492)},
    "mnu:v1":         {**_V1, "subdir": "mnu", "suffix": "mnu_v1", "free": ("m_ncdm",)},
    "neff:v1":        {**_V1, "subdir": "neff", "suffix": "neff_v1", "free": ("N_ur",)},
    "wcdm:v1":        {**_V1, "subdir": "wcdm", "suffix": "w_v1", "free": ("w0",)},
    "ede:v1":         {**_V1, "subdir": "ede", "suffix": "v1", "free": _EDE, "deg_ncdm": 3.0},
    "mnu-3states:v1": {**_V1, "subdir": "mnu-3states", "suffix": "v1", "free": ("m_ncdm",), "deg_ncdm": 3.0},
    "ede:v2":         {**_V1, "subdir": "ede", "suffix": "v2", "free": _EDE, "deg_ncdm": 3.0,
                       "z_max_pk": 20.0, "grid": "v2", "log_bg": True},
}

# Defaults of every parameter a set can take.
_FIDUCIAL_PARAMS = {"H0": 68.0, "omega_cdm": 0.12, "omega_b": 0.02246576, "A_s": 2.1053e-9, "n_s": 0.965, "tau": 0.0544,
                    "m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "f_ede": 0.1, "z_c": 3162.278, "theta_i": 1.57, "r": 0.01}
_CORE = ("H0", "omega_cdm", "omega_b", "A_s", "n_s", "tau")  # keywords with numeric defaults in the signature

# Training ranges in hmfast parameter names; n_s comes from the set.
_BOUNDS = {"A_s": (np.exp(2.5) / 1e10, np.exp(3.5) / 1e10), "omega_cdm": (0.08, 0.20), "omega_b": (0.01933, 0.02533),
           "H0": (39.99, 100.01), "tau": (0.02, 0.12), "m_ncdm": (0.0, 0.33333), "w0": (-2.0, -0.33),
           "N_ur": (0.49, 4.49), "theta_i": (0.1, 3.1), "z_c": (1e3, 10**4.3), "f_ede": (0.001, 0.5), "r": (0.0, 0.3)}

# hmfast name -> (emulator input name, transform)
_INPUTS = {"H0": ("H0", None), "omega_cdm": ("omega_cdm", None), "omega_b": ("omega_b", None),
           "A_s": ("ln10^{10}A_s", lambda x: jnp.log(1e10 * x)), "n_s": ("n_s", None), "tau": ("tau_reio", None),
           "m_ncdm": ("m_ncdm", None), "N_ur": ("N_ur", None), "w0": ("w0_fld", None), "f_ede": ("fEDE", None),
           "z_c": ("log10z_c", jnp.log10), "theta_i": ("thetai_scf", None), "r": ("r", None)}

_FILES = {
    "HZ": ("growth-and-distances", EmulatorLoader), "DAZ": ("growth-and-distances", EmulatorLoader),
    "S8Z": ("growth-and-distances", EmulatorLoader), "DER": ("derived-parameters", EmulatorLoader),
    "PKL": ("PK", EmulatorLoader), "PKNL": ("PK", EmulatorLoader),
    "TT": ("TTTEEE", EmulatorLoader), "EE": ("TTTEEE", EmulatorLoader), "TE": ("TTTEEE", EmulatorLoaderPCA),
    "PP": ("PP", EmulatorLoader),
}

_DERIVED_NAMES = ("100*theta_s", "sigma8", "YHe", "z_reio", "Neff", "tau_rec", "z_rec", "rs_rec", "chi_rec",
                  "tau_star", "z_star", "rs_star", "chi_star", "rs_drag")

_WEIGHTS = {}  # (set name, key) -> loader, shared by every CosmoPowerCosmology
_Z_BG = np.linspace(0.0, 20.0, 5000)  # redshifts of the emulated background


def _param_names(emulator_set):
    return (*_LCDM_NAMES, "tau", *_EMULATOR_SETS[emulator_set]["free"])


def _constants(emulator_set):
    spec = _EMULATOR_SETS[emulator_set]
    names = _param_names(emulator_set)
    return {n: v for n, v in {**_STANDARD_CONSTANTS, "deg_ncdm": spec["deg_ncdm"]}.items() if n not in names}


def _grid(emulator_set):
    """k grid of the emulated spectra and the factor that turns the network output into P(k)."""
    if _EMULATOR_SETS[emulator_set]["grid"] == "v2":
        k_grid = np.geomspace(5e-4, 10.0, 1000)
        return k_grid, k_grid ** -3
    k_grid = np.geomspace(1e-4, 50.0, 5000)[::10]
    ell = np.arange(2, 5002)[::10]
    return k_grid, (ell * (ell + 1.0) / (2.0 * np.pi)) ** -1


def _emu(emulator_set, key):
    if (emulator_set, key) not in _WEIGHTS:
        spec = _EMULATOR_SETS[emulator_set]
        subdir, loader = _FILES[key]
        path = os.path.join(_get_default_data_path(), spec["subdir"], subdir, f"{key}_{spec['suffix']}")
        # Keep the weights concrete even when first reached under a trace.
        with jax.ensure_compile_time_eval():
            _WEIGHTS[(emulator_set, key)] = loader(path)
    return _WEIGHTS[(emulator_set, key)]


def _inputs(p):
    """Translate hmfast parameter names to the emulators' input names."""
    return {emu: (f(p[name]) if f else p[name]) for name, (emu, f) in _INPUTS.items() if name in p}


def _mask(emulator_set, p, x):
    """NaN wherever the parameters lie outside the training ranges."""
    bounds = {**_BOUNDS, "n_s": _EMULATOR_SETS[emulator_set]["n_s"]}
    valid = True
    for name in _param_names(emulator_set):
        lo, hi = bounds[name]
        valid = valid & (p[name] >= lo) & (p[name] <= hi)
    return jnp.where(valid, x, jnp.nan)


def _hubble_parameter(emulator_set, z, p):
    hz = 10.0 ** _emu(emulator_set, "HZ").predictions(_inputs(p)) * _C_KMS
    return _mask(emulator_set, p, jnp.interp(z, _Z_BG, hz, left=jnp.nan, right=jnp.nan))


def _angular_diameter_distance(emulator_set, z, p):
    da = _emu(emulator_set, "DAZ").predictions(_inputs(p))
    if _EMULATOR_SETS[emulator_set]["log_bg"]:
        da = jnp.insert(10.0 ** da, 0, 0.0)
    return _mask(emulator_set, p, jnp.interp(z, _Z_BG, da, left=jnp.nan, right=jnp.nan))


def _power(emulator_set, key, k, z, p):
    inputs = _inputs(p)
    k_grid, pk_fac = _grid(emulator_set)

    def one_z(z_i):
        pk = 10.0 ** _emu(emulator_set, key).predictions({**inputs, "z_pk_save_nonclass": z_i}) * pk_fac
        return log_interp1d_extrap(k, k_grid, pk)

    return _mask(emulator_set, p, jax.vmap(one_z, out_axes=1)(z))


_FUNCTIONS = {}  # (set name, pknl_mode) -> functions, so equal settings give equal treedefs


def _functions(emulator_set, pknl_mode):
    if (emulator_set, pknl_mode) not in _FUNCTIONS:
        _FUNCTIONS[(emulator_set, pknl_mode)] = _Functions(
            hubble_parameter=partial(_hubble_parameter, emulator_set),
            pk_linear=partial(_power, emulator_set, "PKL"),
            angular_diameter_distance=partial(_angular_diameter_distance, emulator_set),
            pk_nonlinear=partial(_power, emulator_set, "PKNL") if pknl_mode == "hmcode" else None,
            growth_factor=None,
        )
    return _FUNCTIONS[(emulator_set, pknl_mode)]


class CosmoPowerCosmology(Cosmology):
    """
    Cosmology from the CosmoPower emulators of CLASS: background, linear and nonlinear :math:`P(k)`,
    :math:`\\sigma_8(z)`, CMB spectra and derived parameters. Outputs are NaN outside the training ranges.
    Inherits from :class:`~hmfast.cosmology.Cosmology`, with the emulators as its functions; parameters
    are read as attributes (``cosmo.H0``) and changed with :meth:`update`.

    Every set takes ``H0``, ``omega_cdm``, ``omega_b``, ``A_s``, ``n_s`` and ``tau``, plus:

    ==================  ===============================================  ==================  ==============
    Set                 Extra parameters                                 Massive states      ``z_max_pk``
    ==================  ===============================================  ==================  ==============
    ``lcdm:v1``         --                                               1 (0.06 eV)         5
    ``mnu:v1``          ``m_ncdm``                                       1                   5
    ``mnu-3states:v1``  ``m_ncdm`` (per state)                           3                   5
    ``neff:v1``         ``N_ur``                                         1 (0.06 eV)         5
    ``wcdm:v1``         ``w0``                                           1 (0.06 eV)         5
    ``ede:v1``          ``m_ncdm``, ``N_ur``, ``f_ede``, ``z_c``,        3                   5
                        ``theta_i``, ``r``
    ``ede:v2``          as ``ede:v1``                                    3                   20
    ==================  ===============================================  ==================  ==============

    Parameters a set does not take are fixed at :math:`m_\\nu = 0.06` eV, :math:`N_{\\rm ur} = 3.046`,
    :math:`w_0 = -1` and :math:`T_{\\rm cmb} = 2.7255` K. The background is valid to :math:`z = 20`.
    The remaining parameters are as in :class:`~hmfast.cosmology.Cosmology`.

    Parameters
    ----------
    emulator_set : str
        Emulator set, one of the table above (default ``"lcdm:v1"``).
    pknl_mode : {"hmcode", "halofit"}
        Nonlinear :math:`P(k)` from the HMcode emulator, or halofit on the emulated linear :math:`P(k)`.
    tau : float
        Optical depth to reionization (default 0.0544).
    m_ncdm : float
        Neutrino mass in eV per massive state, with the number of states set by the emulator set
        (default 0.06); fixed at 0.06 for sets that do not take it.
    N_ur : float
        Effective number of ultra-relativistic species (default 3.046); fixed for sets that do not take it.
    w0 : float
        Dark energy equation of state (default -1); fixed for sets that do not take it.
    f_ede : float
        Maximum fractional contribution of early dark energy (default 0.1).
    z_c : float
        Critical redshift of the early dark energy transition (default 3162.278).
    theta_i : float
        Initial early dark energy field displacement in radians (default 1.57).
    r : float
        Tensor-to-scalar ratio (default 0.01).
    extrapolate_z : bool
        If True, redshifts above the emulators' maximum are
        extrapolated. This is less accurate for early dark
        energy models, and for masses/neutrino content where the
        non-relativistic approximation for massive neutrinos breaks down
        before then.

    Examples
    --------
    >>> cosmo = CosmoPowerCosmology("ede:v2", f_ede=0.08)
    """
    _STATIC = Cosmology._STATIC + ("emulator_set", "pknl_mode", "extrapolate_z")
    _SETTINGS = Cosmology._SETTINGS + ("extrapolate_z",)

    def __init__(self, emulator_set="lcdm:v1", *, pknl_mode="hmcode",
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, A_s=2.1053e-9, n_s=0.965, tau=0.0544,  # LCDM
                 m_ncdm=None, N_ur=None, w0=None,                                                   # wCDM, Neff, MNU
                 f_ede=None, z_c=None, theta_i=None, r=None,                                        # EDE
                 extrapolate_z=False, extrapolate_k=True, ncdm_mode="cb"):
        if emulator_set not in _EMULATOR_SETS:
            raise ValueError(f"Unknown emulator set {emulator_set!r}. Allowed: {', '.join(_EMULATOR_SETS)}.")
        if pknl_mode not in ("hmcode", "halofit"):
            raise ValueError(f'pknl_mode must be "hmcode" or "halofit", got {pknl_mode!r}.')
        names = _param_names(emulator_set)
        named = dict(H0=H0, omega_cdm=omega_cdm, omega_b=omega_b, A_s=A_s, n_s=n_s, tau=tau, m_ncdm=m_ncdm,
                     N_ur=N_ur, w0=w0, f_ede=f_ede, z_c=z_c, theta_i=theta_i, r=r)
        # A core keyword the set lacks is only an error if moved off its default.
        given = {n: v for n, v in named.items()
                 if v is not None and (n in names or n not in _CORE or v != _FIDUCIAL_PARAMS[n])}
        unknown = [n for n in given if n not in names]
        if unknown:
            raise TypeError(_unknown(unknown, names, repr(emulator_set)))
        self.emulator_set = emulator_set
        self.pknl_mode = pknl_mode
        self.extrapolate_z = extrapolate_z
        self._setup({n: given.get(n, _FIDUCIAL_PARAMS[n]) for n in names}, _functions(emulator_set, pknl_mode),
                    _grid(emulator_set)[0], ncdm_mode, extrapolate_k)

        if os.environ.get("READTHEDOCS") != "True":
            for key in ("HZ", "DAZ", "S8Z", "PKL", "PKNL", "DER"):
                _emu(emulator_set, key)

    def __repr__(self):
        return f"CosmoPowerCosmology({self.emulator_set!r}, pknl_mode={self.pknl_mode!r}, {self._repr_values()})"

    def update(self, *, pknl_mode=None, **changes):
        """
        Return a new cosmology with some parameters or settings changed.

        The cosmology itself is not modified. Changing parameter values does not recompile
        under ``jit``, so use this to sweep or differentiate parameters; changing a setting does.

        Parameters
        ----------
        pknl_mode : {"hmcode", "halofit"} or None
            New source of the nonlinear :math:`P(k)`; None leaves it unchanged.
        **changes
            New values, by name, for any parameter of the emulator set (e.g. ``H0=70.0``, ``f_ede=0.08``),
            or for the settings ``extrapolate_z``, ``extrapolate_k`` and ``ncdm_mode``; None leaves a value unchanged.

        Returns
        -------
        CosmoPowerCosmology
            New instance with the same emulator set.

        Raises
        ------
        TypeError
            If a name is not a parameter of the emulator set, e.g. ``m_ncdm`` for ``lcdm:v1``.
        ValueError
            If ``pknl_mode`` or ``ncdm_mode`` is not one of its allowed values.

        Examples
        --------
        >>> cosmo = CosmoPowerCosmology("ede:v2").update(f_ede=0.08, extrapolate_z=True)
        """
        if pknl_mode not in (None, "hmcode", "halofit"):
            raise ValueError(f'pknl_mode must be "hmcode" or "halofit", got {pknl_mode!r}.')
        obj = super().update(**changes)
        if pknl_mode is not None:
            obj.pknl_mode = pknl_mode
            obj._functions = _functions(obj.emulator_set, pknl_mode)
        return obj

    def _densities(self):
        return standard_densities(self.params, **_constants(self.emulator_set))

    @property
    def _z_max_bg(self):
        return float(_Z_BG[-1])

    @property
    def _z_max_pk(self):
        return _EMULATOR_SETS[self.emulator_set]["z_max_pk"]

    def _z_grid_bg(self):
        return jnp.linspace(0.0, self._z_max_bg, 5000, dtype=jnp.float64)

    def _z_grid_pk(self):
        return jnp.linspace(0.0, self._z_max_pk, 100, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Background, growth and P(k), extrapolated above the emulators' range if extrapolate_z
    # ------------------------------------------------------------------

    def _hz_flrw_calibrated(self, z_max):
        """Closed-form flat-FLRW H(z), rescaled to match the emulator's H(z_max) exactly."""
        def hz_flrw(z):
            p = self._cosmo_params()
            zp1 = 1.0 + z
            return p['H0'] * jnp.sqrt(
                p['Omega0_m'] * zp1 ** 3
                + p['Omega0_r'] * zp1 ** 4
                + p['Omega_Lambda'] * zp1 ** (3.0 * (1.0 + p['w0']))
            )
        # Force the non-extrapolated path to avoid recursing into this method via jnp.where's eager evaluation.
        correction = (self.update(extrapolate_z=False).hubble_parameter(z_max) / hz_flrw(z_max)) ** 2
        return lambda z: hz_flrw(z) * jnp.sqrt(correction)

    @jax.jit
    def hubble_parameter(self, z):
        """
        Hubble parameter :math:`H(z)` at redshift :math:`z`, NaN above :math:`z = 20` unless ``extrapolate_z``.

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
        z_max = self._z_max_bg
        Hz = self._call("hubble_parameter", jnp.minimum(z_arr, z_max))
        fill = self._hz_flrw_calibrated(z_max)(z_arr) if self.extrapolate_z else jnp.nan
        Hz = jnp.where(z_arr > z_max, fill, Hz)

        return self._squeeze_single(Hz)

    @jax.jit
    def angular_diameter_distance(self, z):
        """
        Angular diameter distance :math:`D_A(z)` at redshift :math:`z`, NaN above :math:`z = 20` unless ``extrapolate_z``.

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
        z_max = self._z_max_bg
        DA = self._call("angular_diameter_distance", jnp.minimum(z_arr, z_max))
        if not self.extrapolate_z:
            DA = jnp.where(z_arr > z_max, jnp.nan, DA)

        if self.extrapolate_z:
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

        return self._squeeze_single(DA)

    @jax.jit
    def sigma8(self, z):
        """
        :math:`\\sigma_8(z)` at redshift :math:`z` from its own emulator.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless :math:`\\sigma_8` value(s)
        """
        s8 = _emu(self.emulator_set, "S8Z").predictions(_inputs(self.params))
        if _EMULATOR_SETS[self.emulator_set]["log_bg"]:
            s8 = 10.0 ** s8
        s8 = jnp.interp(jnp.atleast_1d(z), _Z_BG, s8, left=jnp.nan, right=jnp.nan)
        return self._squeeze_single(_mask(self.emulator_set, self.params, s8))

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
        Linear growth factor :math:`D(z)`, normalized to :math:`D(0)=1`, from the linear power spectrum at
        :math:`k = 0.01\\,\\mathrm{Mpc}^{-1}`. NaN above ``z_max_pk`` unless
        ``extrapolate_z``, in which case the growth ODE continues it.

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

        # Clamp to the emulator's range before calling it, so the masked values cannot poison gradients.
        z_max = self._z_max_pk
        D = self._call("growth_factor", jnp.minimum(z, z_max))
        D = jnp.where(z > z_max, jnp.nan, D)

        if self.extrapolate_z:
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
        f = self._growth_rate_tabulated(z)

        if self.extrapolate_z:
            z_max = self._z_max_pk
            _, f_ext = self._growth_ode(z, z_max)
            f = jnp.where(z > z_max, f_ext, f)

        return jnp.squeeze(f)

    @partial(jax.jit, static_argnames=("linear",))
    def pk(self, k, z, *, linear=True):
        """
        Get the matter power spectrum :math:`P(k, z)` interpolated at
        requested wavenumbers `k` and redshifts `z`. NaN above ``z_max_pk`` unless
        ``extrapolate_z``, in which case it is scaled by the squared growth factor.

        Parameters
        ----------
        k : float or jnp.ndarray
            Wavenumber(s) in :math:`\\mathrm{Mpc}^{-1}` to evaluate the power spectrum at.
        z : float or jnp.ndarray
            Redshift(s) at which to evaluate the power spectrum.
        linear : bool
            True for linear :math:`P(k)`, False for nonlinear :math:`P(k)` (HMcode or halofit, set by ``pknl_mode``).

        Returns
        -------
        P : jnp.ndarray
            Power spectrum values with shape :math:`(N_k, N_z)`, where singleton
            dimensions get squeezed before return.
        """
        k = jnp.atleast_1d(k)
        z = jnp.atleast_1d(z)

        z_max = self._z_max_pk
        in_z_bounds = z <= z_max
        if self.extrapolate_z:
            growth_ratio_sq = jnp.where(in_z_bounds, 1.0, (self.growth_factor(z) / self.growth_factor(z_max)) ** 2)

        # Clamp to the emulator's range before calling it, so the masked values cannot poison gradients.
        pk_out = self._pk_k_masked(k, jnp.where(in_z_bounds, z, z_max), linear)
        if self.extrapolate_z:
            pk_out = pk_out * growth_ratio_sq[None, :]
        else:
            pk_out = jnp.where(in_z_bounds[None, :], pk_out, jnp.nan)
        return jnp.squeeze(pk_out)

    # ------------------------------------------------------------------
    # CMB angular power spectra and derived parameters
    # ------------------------------------------------------------------

    def cl_cmb(self, type, l):
        """
        Evaluate the CMB power spectrum of the specified type at requested multipoles `l`.
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
        inputs = _inputs(self.params)
        emu = _emu(self.emulator_set, s)
        cl = emu.predictions(inputs) if s == "TE" else emu.ten_to_predictions(inputs)
        if s == "PP":
            cl = cl / (2.0 * jnp.pi)
        ell = jnp.arange(2, cl.shape[0] + 2)
        cl = jnp.interp(jnp.atleast_1d(l), ell, cl, left=jnp.nan, right=jnp.nan)
        return jnp.squeeze(_mask(self.emulator_set, self.params, cl))

    @jax.jit
    def derived_parameters(self):
        """
        Get derived cosmological parameters from the emulator.

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
        values = _emu(self.emulator_set, "DER").ten_to_predictions(_inputs(self.params))
        return {n: _mask(self.emulator_set, self.params, v) for n, v in zip(_DERIVED_NAMES, values)}
