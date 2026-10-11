"""CosmoPower emulators of CLASS as a Cosmology."""
import os
from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.cosmology import Cosmology, _Functions
from hmfast.cosmology.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.download import _get_default_data_path
from hmfast.utils import Const, dopri5_integrate, log_interp1d_extrap

_C_KMS = Const._c_ / 1e3

_LCDM_NAMES = ("H0", "omega_cdm", "omega_b", "A_s", "n_s")
_STANDARD_CONSTANTS = MappingProxyType({"Omega_k": 0.0, "m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "wa": 0.0,
                                        "T_cmb": 2.7255})
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
    names = _param_names(emulator_set)
    return {n: v for n, v in _STANDARD_CONSTANTS.items() if n not in names}


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


def _growth_factor(emulator_set, z, p, k0=1e-2):
    k = jnp.array([k0])
    return jnp.sqrt(_power(emulator_set, "PKL", k, z, p)[0] / _power(emulator_set, "PKL", k, jnp.zeros(1), p)[0, 0])


# ------------------------------------------------------------------
# Redshift range: NaN above it, or extrapolated with an FLRW H(z) and the growth equation
# ------------------------------------------------------------------

def _cosmology(emulator_set, p):
    """CosmoPowerCosmology with parameters ``p``, so functions that only receive ``p`` can use its methods."""
    cosmo = CosmoPowerCosmology._from_params(p)
    cosmo.emulator_set = emulator_set
    return cosmo


def _hubble_flrw(emulator_set, z, p, z_cal):
    """FLRW H(z) from rho_x, rescaled to the emulated H at ``z_cal``."""
    cosmo = _cosmology(emulator_set, p)
    hz = lambda zz: p["H0"] * jnp.sqrt(cosmo.rho_x(zz, "critical") / cosmo.rho_x(0.0, "critical"))
    return hz(z) * _hubble_parameter(emulator_set, z_cal, p) / hz(z_cal)


def _growth_ode(emulator_set, z, p):
    """ln D above the P(k) range from the growth equation, matched to the emulated D at its top."""
    z_max = _EMULATOR_SETS[emulator_set]["z_max_pk"]
    H0, om0 = p["H0"], _cosmology(emulator_set, p).omega_x(0.0, "matter")
    hz = lambda zz: _hubble_flrw(emulator_set, zz, p, z_max)

    # Seed well past any requested z so D~a has room to relax onto the growing mode.
    z_start = jax.lax.stop_gradient(4.0 * jnp.maximum(jnp.max(z), z_max))
    x_start, x_max = -jnp.log1p(z_start), -jnp.log1p(z_max)

    def growth_rhs(x, y):
        _, f = y
        dlnH_dx = jax.grad(lambda xx: jnp.log(hz(jnp.expm1(-xx))))(x)
        zz = jnp.expm1(-x)
        return f, -f ** 2 - (2.0 + dlnH_dx) * f + 1.5 * om0 * (1.0 + zz) ** 3 * H0 ** 2 / hz(zz) ** 2

    # max_h caps node spacing so batched interior queries interpolate safely.
    x_traj, (lnD_traj, f_traj) = dopri5_integrate(growth_rhs, (jnp.array(0.0), jnp.array(1.0)), x_start, x_max,
                                                  rtol=1e-4, atol=1e-7, max_steps=64, max_h=0.3)
    # Cubic Hermite in ln a with f = dlnD/dlna at the nodes, so D is smooth and growth_rate is its derivative.
    x = jnp.minimum(-jnp.log1p(z), x_max)
    i = jnp.clip(jnp.searchsorted(x_traj, x, side="right") - 1, 0, x_traj.shape[0] - 2)
    h = x_traj[i + 1] - x_traj[i]
    h_safe = jnp.where(h > 0, h, 1.0)  # nodes after convergence repeat x_max
    t = jnp.clip((x - x_traj[i]) / h_safe, 0.0, 1.0)
    lnD = ((2 * t**3 - 3 * t**2 + 1) * lnD_traj[i] + (t**3 - 2 * t**2 + t) * h * f_traj[i]
           + (-2 * t**3 + 3 * t**2) * lnD_traj[i + 1] + (t**3 - t**2) * h * f_traj[i + 1])
    norm = jnp.log(_growth_factor(emulator_set, jnp.atleast_1d(z_max), p)[0]) - jnp.interp(x_max, x_traj, lnD_traj)
    return lnD + norm


def _hubble_parameter_z(emulator_set, extrapolate_z, z, p):
    z_max = float(_Z_BG[-1])
    hz = _hubble_parameter(emulator_set, jnp.minimum(z, z_max), p)
    fill = _hubble_flrw(emulator_set, z, p, z_max) if extrapolate_z else jnp.nan
    return jnp.where(z > z_max, fill, hz)


def _angular_diameter_distance_z(emulator_set, extrapolate_z, z, p):
    z_max = float(_Z_BG[-1])
    da = _angular_diameter_distance(emulator_set, jnp.minimum(z, z_max), p)
    if not extrapolate_z:
        return jnp.where(z > z_max, jnp.nan, da)
    chi_max = _angular_diameter_distance(emulator_set, z_max, p) * (1.0 + z_max)
    dchi_dx = lambda x, chi: _C_KMS * (1.0 + jnp.expm1(x)) / _hubble_flrw(emulator_set, jnp.expm1(x), p, z_max)

    # Integrate to each z independently -- a shared trajectory would be too sparse to interpolate safely.
    def chi_at(z_target):
        _, chi_traj = dopri5_integrate(dchi_dx, chi_max, jnp.log1p(z_max), jnp.log1p(z_target), rtol=1e-4, atol=1e-7,
                                       max_steps=16)
        return chi_traj[-1]

    return jnp.where(z > z_max, jax.vmap(chi_at)(z) / (1.0 + z), da)


def _growth_factor_z(emulator_set, extrapolate_z, z, p):
    z_max = _EMULATOR_SETS[emulator_set]["z_max_pk"]
    D = _growth_factor(emulator_set, jnp.minimum(z, z_max), p)
    fill = jnp.exp(_growth_ode(emulator_set, z, p)) if extrapolate_z else jnp.nan
    return jnp.where(z > z_max, fill, D)


def _power_z(emulator_set, key, extrapolate_z, extrapolate_k, k, z, p):
    z_max = _EMULATOR_SETS[emulator_set]["z_max_pk"]
    pk = _power(emulator_set, key, k, jnp.minimum(z, z_max), p)
    if extrapolate_z:
        D_max = _growth_factor(emulator_set, jnp.atleast_1d(z_max), p)[0]
        growth_ratio = jnp.where(z > z_max, jnp.exp(_growth_ode(emulator_set, z, p)) / D_max, 1.0)
        pk = pk * growth_ratio[None, :] ** 2
    else:
        pk = jnp.where(z[None, :] > z_max, jnp.nan, pk)
    if not extrapolate_k:
        k_grid = _grid(emulator_set)[0]
        pk = jnp.where(((k >= k_grid[0]) & (k <= k_grid[-1]))[:, None], pk, jnp.nan)
    return pk


_FUNCTIONS = {}  # (set name, extrapolate_z, extrapolate_k) -> functions, so equal settings give equal treedefs


def _functions(emulator_set, extrapolate_z, extrapolate_k):
    key = (emulator_set, extrapolate_z, extrapolate_k)
    if key not in _FUNCTIONS:
        _FUNCTIONS[key] = _Functions(
            hubble_parameter=partial(_hubble_parameter_z, emulator_set, extrapolate_z),
            pk_linear=partial(_power_z, emulator_set, "PKL", extrapolate_z, extrapolate_k),
            angular_diameter_distance=partial(_angular_diameter_distance_z, emulator_set, extrapolate_z),
            pk_nonlinear=partial(_power_z, emulator_set, "PKNL", extrapolate_z, extrapolate_k),
            growth_factor=partial(_growth_factor_z, emulator_set, extrapolate_z),
        )
    return _FUNCTIONS[key]


class CosmoPowerCosmology(Cosmology):
    """
    Cosmology from the CosmoPower emulators of CLASS: background, linear and nonlinear (HMcode) :math:`P(k)`,
    :math:`\\sigma_8(z)`, CMB spectra and derived parameters. Outputs are NaN outside the training ranges.
    Inherits from :class:`~hmfast.cosmology.Cosmology`, with the emulators as its functions.

    Every set varies ``H0``, ``omega_cdm``, ``omega_b``, ``A_s``, ``n_s`` and ``tau``, plus:

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

    The background is valid to :math:`z = 20`.

    Parameters
    ----------
    emulator_set : str
        Emulator set, one of the table above (default ``"lcdm:v1"``).
    H0, omega_cdm, omega_b, A_s, n_s, tau : float
        As in :class:`~hmfast.cosmology.Cosmology`; ``tau`` defaults to 0.0544.
    Omega_k, m_ncdm, N_ur, w0, wa, T_cmb, f_ede, z_c, theta_i, r : float or None
        As in :class:`~hmfast.cosmology.Cosmology`. None takes the set's value: the fiducial value if the set
        varies the parameter (``m_ncdm`` 0.06, ``N_ur`` 3.046, ``w0`` -1, ``f_ede`` 0.1, ``z_c`` 3162.278,
        ``theta_i`` 1.57, ``r`` 0.01); otherwise it is fixed (``Omega_k`` 0, ``m_ncdm`` 0.06, ``N_ur`` 3.046,
        ``w0`` -1, ``wa`` 0, ``T_cmb`` 2.7255) or absent, and setting it raises a ``TypeError``. Every set is
        flat with no evolving dark energy, so ``Omega_k`` and ``wa`` are always fixed at 0.
    ncdm_mode : {"cb", "m"}
        As in :class:`~hmfast.cosmology.Cosmology`.
    extrapolate_k : bool
        If True (default), :math:`P(k)` is extrapolated as a power law in log-log beyond the emulators'
        wavenumber range; if False, it is NaN there.
    extrapolate_z : bool
        If True, redshifts above the emulators' range are extrapolated with an FLRW :math:`H(z)` from
        :meth:`~hmfast.cosmology.Cosmology.rho_x` and the linear growth equation, matched to the emulators
        at the top of the range; if False (default), they are NaN. This is less accurate for early dark
        energy models, and for masses/neutrino content where the non-relativistic approximation for
        massive neutrinos breaks down before then.

    Examples
    --------
    >>> cosmo = CosmoPowerCosmology("ede:v2", f_ede=0.08)
    """
    _STATIC = Cosmology._STATIC + ("emulator_set", "extrapolate_z", "extrapolate_k")

    def __init__(self, emulator_set="lcdm:v1", *,
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, Omega_k=None, A_s=2.1053e-9, n_s=0.965, tau=0.0544,
                 m_ncdm=None, N_ur=None, w0=None, wa=None, T_cmb=None,
                 f_ede=None, z_c=None, theta_i=None, r=None,
                 ncdm_mode="cb", extrapolate_k=True, extrapolate_z=False):
        if emulator_set not in _EMULATOR_SETS:
            raise ValueError(f"Unknown emulator set {emulator_set!r}. Allowed: {', '.join(_EMULATOR_SETS)}.")
        self.emulator_set = emulator_set
        self.extrapolate_z, self.extrapolate_k = extrapolate_z, extrapolate_k
        names = _param_names(emulator_set)
        named = dict(H0=H0, omega_cdm=omega_cdm, omega_b=omega_b, Omega_k=Omega_k, A_s=A_s, n_s=n_s, tau=tau,
                     m_ncdm=m_ncdm, N_ur=N_ur, w0=w0, wa=wa, T_cmb=T_cmb, f_ede=f_ede, z_c=z_c, theta_i=theta_i, r=r)
        given = {n: v for n, v in named.items() if v is not None}
        self.params = {n: given.get(n, _FIDUCIAL_PARAMS[n]) for n in names}
        unknown = [n for n in given if n not in names]
        if unknown:
            raise TypeError(self._not_parameters(unknown))
        self._setup(self.params, _functions(emulator_set, extrapolate_z, extrapolate_k), _grid(emulator_set)[0],
                    ncdm_mode)

        if os.environ.get("READTHEDOCS") != "True":
            for key in ("HZ", "DAZ", "S8Z", "PKL", "PKNL", "DER"):
                _emu(emulator_set, key)

    def __repr__(self):
        return f"CosmoPowerCosmology({self.emulator_set!r}, {self._repr_values()})"

    def update(self, *, extrapolate_z=None, extrapolate_k=None, **changes):
        """
        Return a new cosmology with some parameters or settings changed.

        Parameters
        ----------
        extrapolate_z, extrapolate_k : bool or None
            New values of these settings; None leaves a value unchanged.
        **changes
            As in :meth:`Cosmology.update`, for the parameters the emulator set varies.

        Returns
        -------
        CosmoPowerCosmology
            New instance with the same emulator set.

        Examples
        --------
        >>> cosmo = CosmoPowerCosmology("ede:v2").update(f_ede=0.08, extrapolate_z=True)
        """
        obj = super().update(**changes)
        if extrapolate_z is not None:
            obj.extrapolate_z = extrapolate_z
        if extrapolate_k is not None:
            obj.extrapolate_k = extrapolate_k
        obj._functions = _functions(obj.emulator_set, obj.extrapolate_z, obj.extrapolate_k)
        return obj

    def _fixed_params(self):
        return _constants(self.emulator_set)

    def _label(self):
        return repr(self.emulator_set)

    @property
    def _deg_ncdm(self):
        return _EMULATOR_SETS[self.emulator_set]["deg_ncdm"]

    def _sigma8(self, z):
        s8 = _emu(self.emulator_set, "S8Z").predictions(_inputs(self.params))
        if _EMULATOR_SETS[self.emulator_set]["log_bg"]:
            s8 = 10.0 ** s8
        s8 = jnp.interp(jnp.atleast_1d(z), _Z_BG, s8, left=jnp.nan, right=jnp.nan)
        return self._squeeze_single(_mask(self.emulator_set, self.params, s8))

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
