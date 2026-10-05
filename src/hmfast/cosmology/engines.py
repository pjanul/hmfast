"""Engines: the source of the background and matter power spectrum behind a Cosmology."""
import dataclasses
import os
from types import MappingProxyType
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.download import _get_default_data_path
from hmfast.utils import Const, log_interp1d_extrap

_C_KMS = Const._c_ / 1e3
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)

_LCDM_PARAMS = MappingProxyType({"H0": 68.0, "omega_cdm": 0.12, "omega_b": 0.02246576, "A_s": 2.1053e-9, "n_s": 0.965})

_REQUIRED = ("hubble_parameter", "pk_linear", "densities")
_OPTIONAL = ("pk_nonlinear", "angular_diameter_distance", "growth_factor", "sigma8", "cl_cmb",
             "derived_parameters", "in_bounds")
# Replacing a key field resets these to their defaults, so none is left computed from the old function.
_DEPENDENTS = {
    "hubble_parameter": ("angular_diameter_distance",),
    "pk_linear": ("pk_nonlinear", "growth_factor", "sigma8"),
}


@dataclasses.dataclass(frozen=True, eq=False)
class Engine:
    """
    Source of the background and matter power spectrum behind a :class:`~hmfast.cosmology.Cosmology`.

    An engine is a set of functions of ``p``, a dict of the engine's :attr:`params` at the cosmology's
    values. ``Cosmology`` jits, differentiates and vmaps through them, so each must be JAX-traceable
    and read every parameter from ``p``; a value taken from an enclosing scope is a constant with zero gradient.

    Three functions are required. Each optional one left as None falls back to the default given
    below, computed from the required ones of the same engine. On construction the required functions
    are traced once, without being evaluated, so a parameter missing from ``params`` or a wrong
    output shape raises a ``TypeError`` immediately.

    Parameters
    ----------
    params : dict
        Parameter names and default values: the keywords of ``Cosmology`` and its differentiable leaves.
    hubble_parameter : callable
        ``(z, p) -> H(z)`` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`, shape :math:`(N_z,)`.
        If ``H0`` is not in ``params``, ``Cosmology`` takes :math:`H_0 = H(0)` from it.
    pk_linear : callable
        ``(k, z, p) -> P_L(k, z)`` of total matter (massive neutrinos included) in :math:`\\mathrm{Mpc}^3`,
        shape :math:`(N_k, N_z)`, with ``k`` in :math:`\\mathrm{Mpc}^{-1}`.
    densities : callable
        ``p -> dict`` of ``Omega0_m`` (total matter), ``Omega0_cb`` (CDM + baryons) and ``Omega0_b`` today;
        optionally ``Omega0_r`` (default 0) and ``w0`` (default -1), used by halofit and redshift extrapolation.
        :func:`standard_densities` covers the usual case.
    pk_nonlinear : callable, optional
        ``(k, z, p) -> P_NL(k, z)``, as ``pk_linear``. Default: halofit (Takahashi et al. 2012; Bird et al. 2012) on ``pk_linear``.
    angular_diameter_distance : callable, optional
        ``(z, p) -> D_A(z)`` in Mpc. Default: integral of :math:`c/H` over :math:`\\ln(1+z)`, divided by :math:`1+z` (flat).
    growth_factor : callable, optional
        ``(z, p) -> D(z)`` with :math:`D(0) = 1`. Default: :math:`\\sqrt{P_L(k_0, z)/P_L(k_0, 0)}`, :math:`k_0 = 0.01\\,\\mathrm{Mpc}^{-1}`.
    sigma8 : callable, optional
        ``(z, p) -> sigma_8(z)``. Default: top-hat variance of ``pk_linear`` at :math:`R = 8\\,h^{-1}\\mathrm{Mpc}`.
    cl_cmb : callable, optional
        ``(type, l, p) -> C_l``, shape :math:`(N_\\ell,)`, type ∈ {"TT", "EE", "TE", "PP"} (static). Default: not available.
    derived_parameters : callable, optional
        ``p -> dict`` of scalars. CMB lensing reads ``z_star`` and ``chi_star``. Default: not available,
        so CMB-lensing tracers need ``z_source``.
    in_bounds : callable, optional
        ``p -> bool`` (scalar); ``Cosmology`` returns NaN where False. Default: always True.
    k_grid : array_like, optional
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}` for :math:`\\sigma(M)` and FFTLog (static);
        ``Cosmology.pk`` extrapolates beyond them if ``extrapolate_k``. Default: ``np.geomspace(1e-4, 50, 500)``.
    z_max_bg : float, optional
        Highest redshift of ``hubble_parameter`` and ``angular_diameter_distance``; NaN above unless ``extrapolate_z``. Default: inf.
    z_max_pk : float, optional
        Highest redshift of the power spectra and growth; NaN above unless ``extrapolate_z``. Default: inf.
    name : str, optional
        Label shown by ``repr``.

    Examples
    --------
    >>> engine = Engine(
    ...     params={"H0": 67.7, "Omega_m": 0.31, "Omega_b": 0.049},
    ...     hubble_parameter=lambda z, p: p["H0"] * jnp.sqrt(p["Omega_m"] * (1 + z) ** 3 + 1 - p["Omega_m"]),
    ...     pk_linear=my_pk,
    ...     densities=lambda p: {"Omega0_m": p["Omega_m"], "Omega0_cb": p["Omega_m"], "Omega0_b": p["Omega_b"]},
    ... )
    >>> cosmo = Cosmology(engine, Omega_m=0.3)

    Emulated background with an analytic linear spectrum, which needs two extra parameters:

    >>> emu = EmulatorEngine("lcdm:v1")
    >>> mixed = emu.replace(pk_linear=AnalyticEngine().pk_linear, params={**emu.params, "T_cmb": 2.7255, "w0": -1.0})
    """
    params: dict
    hubble_parameter: Callable
    pk_linear: Callable
    densities: Callable
    pk_nonlinear: Optional[Callable] = None
    angular_diameter_distance: Optional[Callable] = None
    growth_factor: Optional[Callable] = None
    sigma8: Optional[Callable] = None
    cl_cmb: Optional[Callable] = None
    derived_parameters: Optional[Callable] = None
    in_bounds: Optional[Callable] = None
    k_grid: np.ndarray = dataclasses.field(default_factory=lambda: np.geomspace(1e-4, 50.0, 500))
    z_max_bg: float = np.inf
    z_max_pk: float = np.inf
    name: Optional[str] = None

    # Built-in engines skip the trace check: their functions are known to fit the contract.
    _check_on_init = True

    def __post_init__(self):
        for name in _REQUIRED:
            if not callable(getattr(self, name)):
                raise TypeError(f"Engine needs {name}, a callable; got {getattr(self, name)!r}.")
        for name in _OPTIONAL:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"Engine {name} must be callable or None; got {value!r}.")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "k_grid", np.asarray(self.k_grid, dtype=float))
        object.__setattr__(self, "z_max_bg", float(self.z_max_bg))
        object.__setattr__(self, "z_max_pk", float(self.z_max_pk))
        if self._check_on_init:
            self._check()

    def _check(self):
        """Trace the required functions once, so a missing parameter or a wrong shape fails here."""
        p = {n: jnp.asarray(v, dtype=float) for n, v in self.params.items()}
        z, k = jnp.zeros(3), jnp.asarray(self.k_grid[:4])
        expected = {"hubble_parameter": ((z, p), (3,)), "pk_linear": ((k, z, p), (4, 3))}
        if self.pk_nonlinear is not None:
            expected["pk_nonlinear"] = ((k, z, p), (4, 3))
        for name, (args, shape) in expected.items():
            try:
                out = jax.eval_shape(getattr(self, name), *args)
            except KeyError as err:
                raise TypeError(f"{name} reads parameter {err}, which is not in params ({', '.join(self.params)}).") from None
            if out.shape != shape:
                raise TypeError(f"{name} must return shape {shape} for these inputs, got {out.shape}.")
        try:
            d = jax.eval_shape(self.densities, p)
        except KeyError as err:
            raise TypeError(f"densities reads parameter {err}, which is not in params ({', '.join(self.params)}).") from None
        missing = {"Omega0_m", "Omega0_cb", "Omega0_b"} - set(d)
        if missing:
            raise TypeError(f"densities must return {', '.join(sorted(missing))}.")

    def replace(self, **changes):
        """
        New engine with the given fields replaced.

        Replacing ``pk_linear`` also resets ``pk_nonlinear``, ``growth_factor`` and ``sigma8``, and
        replacing ``hubble_parameter`` resets ``angular_diameter_distance``, to their defaults unless
        they are given too, so no output is left computed from the old function.

        Parameters
        ----------
        **changes
            Fields of :class:`Engine` and their new values; ``params`` is replaced whole, not merged.

        Returns
        -------
        Engine
        """
        for key, dependents in _DEPENDENTS.items():
            if key in changes:
                for name in dependents:
                    changes.setdefault(name, None)
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(Engine)}
        unknown = changes.keys() - fields.keys()
        if unknown:
            raise TypeError(f"{', '.join(sorted(unknown))} is not a field of Engine.")
        return Engine(**{**fields, **changes})

    def _call(self, name, *args):
        """Call an optional function, or its default if it is None."""
        f = getattr(self, name)
        if f is not None:
            return f(*args)
        if name in _DEFAULTS:
            return _DEFAULTS[name](self, *args)
        raise NotImplementedError(f"{self!r} provides no {name}.")

    def _densities(self, p):
        d = dict(self.densities(p))
        d.setdefault("Omega0_r", 0.0)
        d.setdefault("w0", -1.0)
        return d

    def _H0(self, p):
        return p["H0"] if "H0" in self.params else self.hubble_parameter(jnp.zeros(1), p)[0]

    # Engines that compare equal share jit caches, so the key holds everything that changes the output.
    def _key(self):
        return (tuple(self.params.items()), *(getattr(self, n) for n in _REQUIRED + _OPTIONAL),
                self.k_grid.tobytes(), self.z_max_bg, self.z_max_pk)

    def __eq__(self, other):
        return type(other) is type(self) and other._key() == self._key()

    def __hash__(self):
        return hash((type(self), self._key()))

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})" if self.name else f"{type(self).__name__}()"

    def __str__(self):
        provides = [n for n in _REQUIRED + _OPTIONAL if getattr(self, n) is not None]
        params = ", ".join(f"{k}={v:.6g}" for k, v in self.params.items())
        return f"{self!r}\n  params  : {params}\n  provides: {', '.join(provides)}"


# ----------------------------------------------------------------------
# Defaults for the optional functions
# ----------------------------------------------------------------------

def _default_angular_diameter_distance(engine, z, p):
    z = jnp.asarray(z)
    x_max = jnp.log1p(z)[..., None]
    zp1 = jnp.exp(0.5 * x_max * (_GL_NODES + 1.0))
    integrand = _C_KMS * zp1 / engine.hubble_parameter((zp1 - 1.0).ravel(), p).reshape(zp1.shape)
    return 0.5 * x_max[..., 0] * jnp.sum(_GL_WEIGHTS * integrand, axis=-1) / (1.0 + z)


def _default_pk_nonlinear(engine, k, z, p):
    d = engine._densities(p)
    k_grid = jnp.asarray(engine.k_grid)
    H0 = engine._H0(p)
    f_nu = 1.0 - d["Omega0_cb"] / d["Omega0_m"]

    def one_z(z_i):
        z_i = jnp.atleast_1d(z_i)
        omega_m_z = d["Omega0_m"] * (1.0 + z_i[0]) ** 3 * (H0 / engine.hubble_parameter(z_i, p)[0]) ** 2
        pk_lin = engine.pk_linear(k_grid, z_i, p)[:, 0]
        pk_nl = halofit(k_grid, pk_lin, omega_m_z, d["Omega0_m"], d["w0"], f_nu, H0 / 100.0)
        return log_interp1d_extrap(k, k_grid, pk_nl)

    return jax.vmap(one_z, out_axes=1)(z)


def _default_growth_factor(engine, z, p, k0=1e-2):
    k = jnp.array([k0])
    return jnp.sqrt(engine.pk_linear(k, z, p)[0] / engine.pk_linear(k, jnp.zeros(1), p)[0, 0])


_DEFAULTS = {
    "pk_nonlinear": _default_pk_nonlinear,
    "angular_diameter_distance": _default_angular_diameter_distance,
    "growth_factor": _default_growth_factor,
    "in_bounds": lambda engine, p: True,
}


def standard_densities(p, *, m_ncdm=0.06, deg_ncdm=1.0, N_ur=3.046, T_cmb=2.7255, w0=-1.0):
    """
    Densities of a flat universe for :attr:`Engine.densities`, from ``H0``, ``omega_b`` and ``omega_cdm`` in ``p``.

    Photons at ``T_cmb``, ``N_ur`` massless neutrinos and ``deg_ncdm`` massive states of ``m_ncdm`` each
    (:math:`\\Omega_\\nu h^2 = m/93.14\\,\\mathrm{eV}`); dark energy closes the budget. A keyword that
    is also a key of ``p`` is read from ``p``, so engines that vary it need no special case.

    Parameters
    ----------
    p : dict
        Engine parameters, containing at least ``H0``, ``omega_b`` and ``omega_cdm``.
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


# ----------------------------------------------------------------------
# Emulator engine
# ----------------------------------------------------------------------

_STANDARD_CONSTANTS = MappingProxyType({"m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "T_cmb": 2.7255, "deg_ncdm": 1.0})
_EXTENSION_DEFAULTS = {"m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "f_ede": 0.1, "z_c": 3162.278,
                       "theta_i": 1.57, "r": 0.01}
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

_WEIGHTS = {}  # (set name, key) -> loader, shared by every EmulatorEngine
_Z_BG = np.linspace(0.0, 20.0, 5000)  # redshifts of the emulated background


class EmulatorEngine(Engine):
    """
    Engine from the neural-network emulators of CLASS: background, linear and nonlinear power spectra,
    :math:`\\sigma_8(z)`, CMB spectra and derived parameters. Outputs are NaN outside the training ranges.

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

    Constants a set does not take are :math:`m_\\nu = 0.06` eV, :math:`N_{\\rm ur} = 3.046`, :math:`w_0 = -1`
    and :math:`T_{\\rm cmb} = 2.7255` K. The background is valid to :math:`z = 20` for every set, and
    ``print(engine)`` lists a set's parameters, defaults and constants.

    Parameters
    ----------
    name : str
        Emulator set, one of the table above.
    pknl_mode : {"hmcode", "halofit"}
        Nonlinear :math:`P(k)` from the HMcode emulator, or halofit on the emulated linear :math:`P(k)`.

    Attributes
    ----------
    k_grid : numpy.ndarray
        The emulators' output wavenumbers: 500 points in :math:`[10^{-4}, 50]\\,\\mathrm{Mpc}^{-1}` (v1 sets)
        or 1000 in :math:`[5 \\times 10^{-4}, 10]\\,\\mathrm{Mpc}^{-1}` (``ede:v2``).

    Examples
    --------
    >>> cosmo = Cosmology(EmulatorEngine("ede:v2"), f_ede=0.08)
    """
    _check_on_init = False

    def __init__(self, name="lcdm:v1", pknl_mode="hmcode"):
        if name not in _EMULATOR_SETS:
            raise ValueError(f"Unknown emulator set {name!r}. Allowed: {', '.join(_EMULATOR_SETS)}.")
        if pknl_mode not in ("hmcode", "halofit"):
            raise ValueError(f'pknl_mode must be "hmcode" or "halofit", got {pknl_mode!r}.')
        spec = _EMULATOR_SETS[name]
        params = {**_LCDM_PARAMS, "tau": 0.0544, **{n: _EXTENSION_DEFAULTS[n] for n in spec["free"]}}
        constants = {n: v for n, v in {**_STANDARD_CONSTANTS, "deg_ncdm": spec["deg_ncdm"]}.items() if n not in params}
        if spec["grid"] == "v2":
            k_grid = np.geomspace(5e-4, 10.0, 1000)
            pk_fac = k_grid ** -3
        else:
            k_grid = np.geomspace(1e-4, 50.0, 5000)[::10]
            ell = np.arange(2, 5002)[::10]
            pk_fac = (ell * (ell + 1.0) / (2.0 * np.pi)) ** -1
        for attr, value in (("pknl_mode", pknl_mode), ("constants", MappingProxyType(constants)),
                            ("_spec", spec), ("_pk_fac", pk_fac)):
            object.__setattr__(self, attr, value)

        super().__init__(
            params=params,
            hubble_parameter=self._hubble_parameter,
            pk_linear=self._pk_linear,
            densities=self._densities_of_set,
            pk_nonlinear=self._pk_nonlinear if pknl_mode == "hmcode" else None,
            angular_diameter_distance=self._angular_diameter_distance,
            sigma8=self._sigma8,
            cl_cmb=self._cl_cmb,
            derived_parameters=self._derived_parameters,
            in_bounds=self._in_bounds,
            k_grid=k_grid,
            z_max_bg=float(_Z_BG[-1]),
            z_max_pk=spec["z_max_pk"],
            name=name,
        )

        if os.environ.get("READTHEDOCS") != "True":
            for key in ("HZ", "DAZ", "S8Z", "PKL", "PKNL", "DER"):
                self._emu(key)

    def _key(self):
        return (self.name, self.pknl_mode)

    def __repr__(self):
        return f"EmulatorEngine({self.name!r}, pknl_mode={self.pknl_mode!r})"

    def __str__(self):
        return f"{super().__str__()}\n  constants: {', '.join(f'{k}={v:.6g}' for k, v in self.constants.items())}"

    def _emu(self, key):
        if (self.name, key) not in _WEIGHTS:
            subdir, loader = _FILES[key]
            path = os.path.join(_get_default_data_path(), self._spec["subdir"], subdir, f"{key}_{self._spec['suffix']}")
            # Keep the weights concrete even when first reached under a trace.
            with jax.ensure_compile_time_eval():
                _WEIGHTS[(self.name, key)] = loader(path)
        return _WEIGHTS[(self.name, key)]

    @staticmethod
    def _inputs(p):
        """Translate hmfast parameter names to the emulators' input names."""
        return {emu: (f(p[name]) if f else p[name]) for name, (emu, f) in _INPUTS.items() if name in p}

    def _densities_of_set(self, p):
        return standard_densities(p, **self.constants)

    def _hubble_parameter(self, z, p):
        hz = 10.0 ** self._emu("HZ").predictions(self._inputs(p)) * _C_KMS
        return jnp.interp(z, _Z_BG, hz, left=jnp.nan, right=jnp.nan)

    def _angular_diameter_distance(self, z, p):
        da = self._emu("DAZ").predictions(self._inputs(p))
        if self._spec["log_bg"]:
            da = jnp.insert(10.0 ** da, 0, 0.0)
        return jnp.interp(z, _Z_BG, da, left=jnp.nan, right=jnp.nan)

    def _sigma8(self, z, p):
        s8 = self._emu("S8Z").predictions(self._inputs(p))
        if self._spec["log_bg"]:
            s8 = 10.0 ** s8
        return jnp.interp(z, _Z_BG, s8, left=jnp.nan, right=jnp.nan)

    def _power(self, key, k, z, p):
        inputs = self._inputs(p)

        def one_z(z_i):
            pk = 10.0 ** self._emu(key).predictions({**inputs, "z_pk_save_nonclass": z_i}) * self._pk_fac
            return log_interp1d_extrap(k, self.k_grid, pk)

        return jax.vmap(one_z, out_axes=1)(z)

    def _pk_linear(self, k, z, p):
        return self._power("PKL", k, z, p)

    def _pk_nonlinear(self, k, z, p):
        return self._power("PKNL", k, z, p)

    def _derived_parameters(self, p):
        return dict(zip(_DERIVED_NAMES, self._emu("DER").ten_to_predictions(self._inputs(p))))

    def _cl_cmb(self, type, l, p):
        inputs = self._inputs(p)
        cl = self._emu(type).predictions(inputs) if type == "TE" else self._emu(type).ten_to_predictions(inputs)
        if type == "PP":
            cl = cl / (2.0 * jnp.pi)
        ell = jnp.arange(2, cl.shape[0] + 2)
        return jnp.interp(l, ell, cl, left=jnp.nan, right=jnp.nan)

    def _in_bounds(self, p):
        bounds = {**_BOUNDS, "n_s": self._spec["n_s"]}
        valid = True
        for name in self.params:
            lo, hi = bounds[name]
            valid = valid & (p[name] >= lo) & (p[name] <= hi)
        return valid


# ----------------------------------------------------------------------
# Halofit (Takahashi et al. 2012) with the Bird et al. 2012 massive-neutrino correction
# ----------------------------------------------------------------------

def _gaussian_moments(ln_r, ln_k, delta2_lin):
    """Gaussian-filtered variance :math:`\\sigma^2(R)` and its first two derivatives in :math:`\\ln R`."""
    y2 = jnp.exp(2.0 * (ln_k + ln_r))
    w = delta2_lin * jnp.exp(-y2)
    s0 = jnp.trapezoid(w, x=ln_k)
    s1 = jnp.trapezoid(-2.0 * y2 * w, x=ln_k)
    s2 = jnp.trapezoid((4.0 * y2**2 - 4.0 * y2) * w, x=ln_k)
    return s0, s1, s2


def halofit(k, pk_lin, omega_m, omega_m0, w0, f_nu, h):
    """
    Halofit nonlinear matter power spectrum (Takahashi et al. 2012) with the
    massive-neutrino correction of Bird et al. (2012), for a flat cosmology at a single redshift.

    Parameters
    ----------
    k : jnp.ndarray
        Log-spaced wavenumbers in :math:`\\mathrm{Mpc}^{-1}`, wide enough to resolve
        the Gaussian-filtered variance at the nonlinear scale.
    pk_lin : jnp.ndarray
        Linear total-matter power spectrum at ``k`` in :math:`\\mathrm{Mpc}^3`.
    omega_m : float
        Total matter density parameter :math:`\\Omega_m(z)` at this redshift.
    omega_m0 : float
        Total matter density parameter today, :math:`\\Omega_{m,0}`.
    w0 : float
        Dark energy equation of state.
    f_nu : float
        Massive-neutrino fraction :math:`\\Omega_{\\nu,0} / \\Omega_{m,0}`.
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    jnp.ndarray
        Nonlinear power spectrum at ``k`` in :math:`\\mathrm{Mpc}^3`. Equal to ``pk_lin``
        if :math:`\\sigma(R) < 1` on every scale the ``k`` range resolves.
    """
    ln_k = jnp.log(k)
    delta2_lin = k**3 * pk_lin / (2.0 * jnp.pi**2)

    # Nonlinear scale sigma(R) = 1: bracket on a ln R grid, then Newton-refine so it is smooth in the parameters.
    ln_r_grid = jnp.linspace(-ln_k[-1], -ln_k[0], 200)
    ln_s0_grid = jnp.log(jax.vmap(lambda x: _gaussian_moments(x, ln_k, delta2_lin)[0])(ln_r_grid))
    ln_r = jnp.interp(0.0, -ln_s0_grid, ln_r_grid)
    for _ in range(3):
        s0, s1, _ = _gaussian_moments(ln_r, ln_k, delta2_lin)
        ln_r = ln_r - jnp.log(s0) * s0 / s1
    s0, s1, s2 = _gaussian_moments(ln_r, ln_k, delta2_lin)
    n = -3.0 - s1 / s0
    c = (s1 / s0) ** 2 - s2 / s0

    de_w = (1.0 - omega_m) * (1.0 + w0)
    a = 10.0 ** (1.5222 + 2.8553 * n + 2.3706 * n**2 + 0.9903 * n**3 + 0.2250 * n**4 - 0.6038 * c + 0.1749 * de_w)
    b = 10.0 ** (-0.5642 + 0.5864 * n + 0.5716 * n**2 - 1.5474 * c + 0.2279 * de_w)
    cc = 10.0 ** (0.3698 + 2.0404 * n + 0.8161 * n**2 + 0.5869 * c)
    gamma = 0.1971 - 0.0843 * n + 0.8460 * c
    alpha = jnp.abs(6.0835 + 1.3373 * n - 0.1959 * n**2 - 5.5274 * c)
    beta = 2.0379 - 0.7354 * n + 0.3157 * n**2 + 1.2490 * n**3 + 0.3980 * n**4 - 0.1682 * c + f_nu * (1.081 + 0.395 * n**2)
    nu = 10.0 ** (5.2105 + 3.6902 * n)
    f1, f2, f3 = omega_m**-0.0307, omega_m**-0.0585, omega_m**0.0743

    y = k * jnp.exp(ln_r)
    k_h = k / h
    delta2_lin_nu = delta2_lin * (1.0 + f_nu * 47.48 * k_h**2 / (1.0 + 1.5 * k_h**2))
    delta2_q = delta2_lin * (1.0 + delta2_lin_nu) ** beta / (1.0 + alpha * delta2_lin_nu) * jnp.exp(-y / 4.0 - y**2 / 8.0)
    delta2_h = a * y ** (3.0 * f1) / (1.0 + b * y**f2 + (f3 * cc * y) ** (3.0 - gamma))
    delta2_h = delta2_h / (1.0 + nu / y**2) * (1.0 + f_nu * (0.977 - 18.015 * (omega_m0 - 0.3)))

    pk_nl = (delta2_q + delta2_h) * 2.0 * jnp.pi**2 / k**3
    return jnp.where(ln_s0_grid[0] > 0.0, pk_nl, pk_lin)


# ----------------------------------------------------------------------
# Analytic engine
# ----------------------------------------------------------------------

def eisenstein_hu(k, omega_cb, omega_b, T_cmb):
    """
    Eisenstein & Hu (1998) matter transfer function with baryon acoustic oscillations.

    Parameters
    ----------
    k : jnp.ndarray
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
    omega_cb : float
        Physical CDM + baryon density, :math:`\\omega_{cb} = \\Omega_{cb} h^2`.
    omega_b : float
        Physical baryon density, :math:`\\omega_b = \\Omega_b h^2`.
    T_cmb : float
        CMB temperature today in Kelvin.

    Returns
    -------
    jnp.ndarray
        Transfer function :math:`T(k)`, normalised to 1 as :math:`k \\to 0`.
    """
    theta = T_cmb / 2.7
    f_b = omega_b / omega_cb
    f_c = 1.0 - f_b

    # Matter-radiation equality, drag epoch and sound horizon (eqs. 2-6)
    z_eq = 2.50e4 * omega_cb * theta**-4
    k_eq = 7.46e-2 * omega_cb * theta**-2
    b1 = 0.313 * omega_cb**-0.419 * (1.0 + 0.607 * omega_cb**0.674)
    b2 = 0.238 * omega_cb**0.223
    z_d = 1291.0 * omega_cb**0.251 / (1.0 + 0.659 * omega_cb**0.828) * (1.0 + b1 * omega_b**b2)
    R_eq = 31.5 * omega_b * theta**-4 * (1e3 / z_eq)
    R_d = 31.5 * omega_b * theta**-4 * (1e3 / z_d)
    s = 2.0 / (3.0 * k_eq) * jnp.sqrt(6.0 / R_eq) * jnp.log((jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq)))
    k_silk = 1.6 * omega_b**0.52 * omega_cb**0.73 * (1.0 + (10.4 * omega_cb) ** -0.95)
    q = k / (13.41 * k_eq)

    def t0(alpha, beta):
        ln = jnp.log(jnp.e + 1.8 * beta * q)
        return ln / (ln + (14.2 / alpha + 386.0 / (1.0 + 69.9 * q**1.08)) * q**2)

    # CDM part (eqs. 9-12, 17-18)
    a1 = (46.9 * omega_cb) ** 0.670 * (1.0 + (32.1 * omega_cb) ** -0.532)
    a2 = (12.0 * omega_cb) ** 0.424 * (1.0 + (45.0 * omega_cb) ** -0.582)
    alpha_c = a1**-f_b * a2 ** (-f_b**3)
    bb1 = 0.944 / (1.0 + (458.0 * omega_cb) ** -0.708)
    bb2 = (0.395 * omega_cb) ** -0.0266
    beta_c = 1.0 / (1.0 + bb1 * (f_c**bb2 - 1.0))
    f = 1.0 / (1.0 + (k * s / 5.4) ** 4)
    T_c = f * t0(1.0, beta_c) + (1.0 - f) * t0(alpha_c, beta_c)

    # Baryon part (eqs. 13-15, 19-24)
    y = (1.0 + z_eq) / (1.0 + z_d)
    G = y * (-6.0 * jnp.sqrt(1.0 + y) + (2.0 + 3.0 * y) * jnp.log((jnp.sqrt(1.0 + y) + 1.0) / (jnp.sqrt(1.0 + y) - 1.0)))
    alpha_b = 2.07 * k_eq * s * (1.0 + R_d) ** -0.75 * G
    beta_node = 8.41 * omega_cb**0.435
    beta_b = 0.5 + f_b + (3.0 - 2.0 * f_b) * jnp.sqrt((17.2 * omega_cb) ** 2 + 1.0)
    s_tilde = s / (1.0 + (beta_node / (k * s)) ** 3) ** (1.0 / 3.0)
    T_b = (t0(1.0, 1.0) / (1.0 + (k * s / 5.2) ** 2)
           + alpha_b / (1.0 + (beta_b / (k * s)) ** 3) * jnp.exp(-((k / k_silk) ** 1.4))) * jnp.sinc(k * s_tilde / jnp.pi)

    return f_b * T_b + f_c * T_c


_ANALYTIC_PARAMS = MappingProxyType({**_LCDM_PARAMS, "m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "T_cmb": 2.7255})


def _analytic_hubble_parameter(z, p):
    d = standard_densities(p)
    zp1 = 1.0 + z
    omega_de = 1.0 - d["Omega0_m"] - d["Omega0_r"]
    return p["H0"] * jnp.sqrt(d["Omega0_r"] * zp1**4 + d["Omega0_m"] * zp1**3 + omega_de * zp1 ** (3.0 * (1.0 + p["w0"])))


def _analytic_growth(z, om, w, n_steps=400):
    """Linear growth normalised to :math:`D = a` in matter domination, from RK4 in :math:`\\ln a`."""
    x = jnp.linspace(jnp.log(1e-3), 0.0, n_steps)
    dx = x[1] - x[0]

    # Radiation is left out, so D = a holds exactly at the starting point.
    def rhs(xx, y):
        a = jnp.exp(xx)
        m, de = om * a**-3, (1.0 - om) * a ** (-3.0 * (1.0 + w))
        dlnh = -1.5 * (m + (1.0 + w) * de) / (m + de)
        return jnp.array([y[1], -(2.0 + dlnh) * y[1] + 1.5 * m / (m + de) * y[0]])

    def step(y, xx):
        k1 = rhs(xx, y)
        k2 = rhs(xx + 0.5 * dx, y + 0.5 * dx * k1)
        k3 = rhs(xx + 0.5 * dx, y + 0.5 * dx * k2)
        k4 = rhs(xx + dx, y + dx * k3)
        y = y + dx / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y, y[0]

    y0 = jnp.array([1e-3, 1e-3])
    _, D = jax.lax.scan(step, y0, x[:-1])
    return jnp.interp(-jnp.log1p(z), x, jnp.concatenate([y0[:1], D]))


def _analytic_pk_linear(k, z, p):
    om = standard_densities(p)["Omega0_m"]
    T = eisenstein_hu(k, p["omega_cdm"] + p["omega_b"], p["omega_b"], p["T_cmb"])
    delta2 = 0.16 * p["A_s"] * (k / 0.05) ** (p["n_s"] - 1.0) * (k * _C_KMS / p["H0"]) ** 4 * T**2
    D = _analytic_growth(z, om, p["w0"]) / om
    return (2.0 * jnp.pi**2 / k**3 * delta2)[:, None] * D[None, :] ** 2


class AnalyticEngine(Engine):
    """
    Engine for a flat :math:`w_0\\mathrm{CDM}` cosmology from analytic formulae, valid for any
    parameter values and redshift.

    :math:`H(z)` is the Friedmann equation with photons, ``N_ur`` massless neutrinos, massive neutrinos
    counted as matter, and constant-:math:`w_0` dark energy. The linear :math:`P(k, z)` is
    :math:`A_s` and :math:`n_s` times the Eisenstein & Hu (1998) transfer function and the linear
    growth factor, without massive-neutrino suppression; the nonlinear one is halofit. There are no
    CMB spectra or derived parameters, so CMB-lensing tracers need ``z_source``.

    Its parameters are ``H0``, ``omega_cdm``, ``omega_b``, ``A_s``, ``n_s``, ``m_ncdm`` (one state),
    ``N_ur``, ``w0`` and ``T_cmb``.

    Examples
    --------
    >>> cosmo = Cosmology(AnalyticEngine(), w0=-0.9)
    """
    _check_on_init = False

    def __init__(self):
        super().__init__(params=_ANALYTIC_PARAMS, hubble_parameter=_analytic_hubble_parameter,
                         pk_linear=_analytic_pk_linear, densities=standard_densities)

    def __repr__(self):
        return "AnalyticEngine()"


# ----------------------------------------------------------------------
# Combined engine
# ----------------------------------------------------------------------

class CombinedEngine(Engine):
    """
    Engine with the background of one engine and the power spectrum of another; the same as
    ``background.replace(...)`` with the power-spectrum fields of ``power``.

    Its parameters are those of both engines, with ``background``'s defaults where they share a name.

    Parameters
    ----------
    background : Engine
        Source of :math:`H(z)`, :math:`D_A(z)`, the densities and any CMB spectra or derived parameters.
    power : Engine
        Source of the linear and nonlinear :math:`P(k, z)`, the growth factor and :math:`\\sigma_8(z)`.
    """
    _check_on_init = False

    def __init__(self, background, power):
        for attr, value in (("background", background), ("power", power)):
            object.__setattr__(self, attr, value)
        fields = {f.name: getattr(background, f.name) for f in dataclasses.fields(Engine)}
        fields.update({n: getattr(power, n) for n in ("pk_linear", "pk_nonlinear", "growth_factor", "sigma8",
                                                      "k_grid", "z_max_pk")})
        fields.update(params={**power.params, **background.params}, in_bounds=self._both_in_bounds, name=None)
        super().__init__(**fields)

    def _both_in_bounds(self, p):
        return self.background._call("in_bounds", p) & self.power._call("in_bounds", p)

    def _key(self):
        return (self.background, self.power)

    def __repr__(self):
        return f"CombinedEngine(background={self.background!r}, power={self.power!r})"
