"""Engines: the source of the background and matter power spectrum behind a Cosmology."""
import abc
import os
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.download import _get_default_data_path
from hmfast.utils import Const, log_interp1d_extrap

_C_KMS = Const._c_ / 1e3

_LCDM_PARAMS = MappingProxyType({"H0": 68.0, "omega_cdm": 0.12, "omega_b": 0.02246576, "A_s": 2.1053e-9, "n_s": 0.965})
_STANDARD_FIXED = MappingProxyType({"m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "T_cmb": 2.7255, "deg_ncdm": 1.0})


def _freeze(params, fixed):
    """Read-only copies of an engine's ``params`` and ``fixed``, which must not share a name."""
    both = params.keys() & fixed.keys()
    if both:
        raise TypeError(f"{', '.join(sorted(both))} cannot be both a parameter and fixed.")
    return MappingProxyType(dict(params)), MappingProxyType(dict(fixed))


class Engine(abc.ABC):
    """
    Parent engine class from which cosmology engines inherit.

    An engine is what a :class:`~hmfast.cosmology.Cosmology` computes with. Each ``Cosmology`` method
    listed below calls the engine method of the same name with a ``compute_`` prefix,
    passing the same arguments plus ``p``. ``Cosmology`` then sets values outside
    :meth:`in_bounds` to NaN and, if requested, extrapolates beyond :attr:`z_max_bg`,
    :attr:`z_max_pk` and :attr:`k_grid`; everything else (growth, :math:`\\sigma(M)`,
    halo model, statistics) is derived from these methods.

    ================================  =========================================  ========
    ``Cosmology`` method              ``Engine`` method                          Required
    ================================  =========================================  ========
    ``hubble_parameter(z)``           :meth:`compute_hubble_parameter`           yes
    ``pk(k, z, linear)``              :meth:`compute_pk`                         yes
    ``angular_diameter_distance(z)``  :meth:`compute_angular_diameter_distance`  no
    ``sigma8(z)``                     :meth:`compute_sigma8`                     no
    ``cl_cmb(type, l)``               :meth:`compute_cl_cmb`                     no
    ``derived_parameters()``          :meth:`compute_derived_parameters`         no
    ================================  =========================================  ========

    :meth:`compute_densities` supplies the density parameters that all of these and ``Cosmology``
    itself use, and :meth:`in_bounds` decides where ``Cosmology`` returns NaN.

    Child classes must implement :meth:`compute_hubble_parameter` and :meth:`compute_pk`.

    Every method receives ``p``, a dict holding the cosmology's :attr:`params`, the
    engine's :attr:`fixed` values, ``h`` and the output of :meth:`compute_densities`.

    Attributes
    ----------
    params : Mapping
        Parameters the engine takes, with their defaults: the keywords of ``Cosmology``
        and its differentiable leaves. Read-only.
    fixed : Mapping
        Values the engine assumes for anything not in ``params`` (by default ``m_ncdm``,
        ``N_ur``, ``w0``, ``T_cmb``, ``deg_ncdm``). Read-only and never shares a name with
        ``params``: a subclass that declares only one of the two removes its names from the inherited other.
    k_grid : numpy.ndarray
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}` on which ``Cosmology`` tabulates :math:`P(k)`
        for :math:`\\sigma(M)` and FFTLog transforms; beyond them ``pk`` is NaN unless ``extrapolate_k``.
    z_max_bg : float
        Highest redshift of the background; ``Cosmology`` extrapolates :math:`H` and :math:`D_A` above it if ``extrapolate_z``.
    z_max_pk : float
        Highest redshift of :math:`P(k, z)`; ``Cosmology`` scales by the growth factor above it if ``extrapolate_z``.
    """
    params, fixed = _freeze(_LCDM_PARAMS, _STANDARD_FIXED)
    k_grid = np.geomspace(1e-4, 50.0, 500)
    z_max_bg = 20.0
    z_max_pk = 10.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        own = vars(cls)
        params, fixed = cls.params, cls.fixed
        # The attribute declared in the subclass wins over the inherited one.
        if "params" in own and "fixed" not in own:
            fixed = {n: v for n, v in fixed.items() if n not in params}
        elif "fixed" in own and "params" not in own:
            params = {n: v for n, v in params.items() if n not in fixed}
        cls.params, cls.fixed = _freeze(params, fixed)

    @abc.abstractmethod
    def compute_hubble_parameter(self, z, p):
        """
        Hubble parameter, returned by :meth:`Cosmology.hubble_parameter <hmfast.cosmology.Cosmology.hubble_parameter>`. Required.

        Parameters
        ----------
        z : jnp.ndarray
            Redshifts, shape :math:`(N_z,)`.
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).

        Returns
        -------
        jnp.ndarray
            :math:`H(z)` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`, shape :math:`(N_z,)`.
        """

    @abc.abstractmethod
    def compute_pk(self, k, z, p, linear=True):
        """
        Total-matter power spectrum, returned by :meth:`Cosmology.pk <hmfast.cosmology.Cosmology.pk>`. Required.

        The linear spectrum must be implemented. Calling ``super().compute_pk(k, z, p, linear=False)``
        returns halofit (Takahashi et al. 2012, with the Bird et al. 2012 neutrino correction)
        applied to it, which is the nonlinear spectrum unless the child class provides its own.

        Parameters
        ----------
        k : jnp.ndarray
            Wavenumbers in :math:`\\mathrm{Mpc}^{-1}`, shape :math:`(N_k,)`.
        z : jnp.ndarray
            Redshifts, shape :math:`(N_z,)`.
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).
        linear : bool
            Linear or nonlinear :math:`P(k, z)` (static).

        Returns
        -------
        jnp.ndarray
            :math:`P(k, z)` in :math:`\\mathrm{Mpc}^3`, shape :math:`(N_k, N_z)`.
        """
        if linear:
            raise NotImplementedError
        k_grid = jnp.asarray(self.k_grid)
        f_nu = p["Omega0_ncdm"] / p["Omega0_m"]

        def one_z(z_i):
            omega_m_z = p["Omega0_m"] * (1.0 + z_i) ** 3 * (p["H0"] / self.compute_hubble_parameter(z_i, p)) ** 2
            pk_lin = self.compute_pk(k_grid, jnp.atleast_1d(z_i), p)[:, 0]
            pk_nl = halofit(k_grid, pk_lin, omega_m_z, p["Omega0_m"], p["w0"], f_nu, p["h"])
            return log_interp1d_extrap(k, k_grid, pk_nl)

        return jax.vmap(one_z, out_axes=1)(z)

    def compute_angular_diameter_distance(self, z, p):
        """
        Angular diameter distance, returned by
        :meth:`Cosmology.angular_diameter_distance <hmfast.cosmology.Cosmology.angular_diameter_distance>`.

        By default, the trapezoidal integral of :math:`c/H(z)` on 5000 points up to
        :attr:`z_max_bg`, divided by :math:`1+z` (flat universe).

        Parameters
        ----------
        z : jnp.ndarray
            Redshifts, shape :math:`(N_z,)`.
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).

        Returns
        -------
        jnp.ndarray
            :math:`D_A(z)` in :math:`\\mathrm{Mpc}`, shape :math:`(N_z,)`.
        """
        z_grid = jnp.linspace(0.0, self.z_max_bg, 5000)
        integrand = _C_KMS / self.compute_hubble_parameter(z_grid, p)
        chi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(z_grid))])
        return jnp.interp(z, z_grid, chi, right=jnp.nan) / (1.0 + z)

    def compute_densities(self, p):
        """
        Present-day density parameters, which ``Cosmology`` adds to ``p`` before calling any other
        engine method and uses itself for :math:`\\rho_{\\rm crit}`, :math:`\\Omega_m(z)`, growth and :math:`M(R)`.

        By default, a flat universe with photons at ``T_cmb``, ``N_ur`` massless neutrinos and
        ``deg_ncdm`` massive states of mass ``m_ncdm`` (:math:`\\Omega_\\nu h^2 = m/93.14\\,\\mathrm{eV}`),
        with dark energy closing the budget.

        Parameters
        ----------
        p : dict
            The cosmology's :attr:`params`, the engine's :attr:`fixed` values and ``h``.

        Returns
        -------
        dict
            ``Omega_b``, ``Omega_cdm``, ``Omega0_g``, ``Omega0_ur``, ``Omega0_ncdm``, ``Omega0_cb``,
            ``Omega0_m``, ``Omega0_r`` and ``Omega_Lambda``, all at :math:`z = 0`.
        """
        c, G, sigma_B, Mpc_over_m = Const._c_, Const._G_, Const._sigma_B_, Const._Mpc_over_m_
        h = p["h"]
        d = {"Omega_b": p["omega_b"] / h**2, "Omega_cdm": p["omega_cdm"] / h**2}
        d["Omega0_g"] = (4.0 * sigma_B / c * p["T_cmb"] ** 4) / (3.0 * c**2 * 1e10 * h**2 / Mpc_over_m**2 / 8.0 / jnp.pi / G)
        d["Omega0_ur"] = p["N_ur"] * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * d["Omega0_g"]
        d["Omega0_ncdm"] = p["deg_ncdm"] * p["m_ncdm"] / (93.14 * h**2)
        d["Omega0_cb"] = d["Omega_b"] + d["Omega_cdm"]
        d["Omega0_m"] = d["Omega0_cb"] + d["Omega0_ncdm"]
        d["Omega0_r"] = d["Omega0_g"] + d["Omega0_ur"]
        d["Omega_Lambda"] = 1.0 - d["Omega0_m"] - d["Omega0_r"]
        return d

    def compute_sigma8(self, z, p):
        """
        :math:`\\sigma_8(z)`, returned by :meth:`Cosmology.sigma8 <hmfast.cosmology.Cosmology.sigma8>`.

        By default, None, in which case ``Cosmology`` integrates the linear :meth:`compute_pk`
        with a top-hat window of radius :math:`8\\,h^{-1}\\mathrm{Mpc}`.

        Parameters
        ----------
        z : jnp.ndarray
            Redshifts, shape :math:`(N_z,)`.
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).

        Returns
        -------
        jnp.ndarray or None
            :math:`\\sigma_8(z)`, shape :math:`(N_z,)`, or None.
        """
        return None

    def compute_cl_cmb(self, type, l, p):
        """
        CMB power spectrum, returned by :meth:`Cosmology.cl_cmb <hmfast.cosmology.Cosmology.cl_cmb>`.

        By default, not provided (raises ``NotImplementedError``).

        Parameters
        ----------
        type : str
            Spectrum, one of "TT", "EE", "TE", "PP" (static).
        l : jnp.ndarray
            Multipoles, shape :math:`(N_\\ell,)`.
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).

        Returns
        -------
        jnp.ndarray
            :math:`C_\\ell`, shape :math:`(N_\\ell,)`.
        """
        raise NotImplementedError(f"{self!r} provides no CMB spectra.")

    def compute_derived_parameters(self, p):
        """
        Derived parameters, returned by :meth:`Cosmology.derived_parameters <hmfast.cosmology.Cosmology.derived_parameters>`.

        By default, not provided (raises ``NotImplementedError``); CMB-lensing tracers then need
        an explicit ``z_source``, since they read ``z_star`` and ``chi_star`` from here.

        Parameters
        ----------
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).

        Returns
        -------
        dict
            Scalars by name, e.g. ``z_star`` and ``chi_star`` (comoving distance in :math:`\\mathrm{Mpc}`).
        """
        raise NotImplementedError(f"{self!r} provides no derived parameters; give CMB-lensing tracers a z_source.")

    def in_bounds(self, p):
        """
        Whether the cosmology lies in the engine's valid domain; ``Cosmology`` returns NaN from every
        engine-based method where it does not.

        By default, always True.

        Parameters
        ----------
        p : dict
            Parameters, fixed values and densities of the cosmology (see :class:`Engine`).

        Returns
        -------
        bool or jnp.ndarray
            Scalar boolean.
        """
        return True

    # Engines that compare equal share jit caches; override _key with the settings that change the output.
    def _key(self):
        return id(self)

    def __eq__(self, other):
        return type(other) is type(self) and other._key() == self._key()

    def __hash__(self):
        return hash((type(self), self._key()))

    def __repr__(self):
        return f"{type(self).__name__}()"

    def __str__(self):
        fmt = lambda d: ", ".join(f"{k}={v:.6g}" for k, v in d.items())
        return f"{self!r}\n  free : {fmt(self.params)}\n  fixed: {fmt(self.fixed)}"


# ----------------------------------------------------------------------
# Emulator engine
# ----------------------------------------------------------------------

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


class EmulatorEngine(Engine):
    """
    Engine that calls the neural-network emulators of CLASS for the background, power spectra,
    CMB spectra and derived parameters. Outputs are NaN outside the emulators' training ranges.

    Parameters
    ----------
    name : str
        Emulator set: ``"lcdm:v1"``, ``"mnu:v1"``, ``"neff:v1"``, ``"wcdm:v1"``, ``"ede:v1"``,
        ``"mnu-3states:v1"`` or ``"ede:v2"``. ``print(engine)`` lists its free and fixed parameters.
    pknl_mode : {"hmcode", "halofit"}
        Nonlinear P(k) from the HMcode emulator, or halofit on the emulated linear P(k).
    """

    def __init__(self, name="lcdm:v1", pknl_mode="hmcode"):
        if name not in _EMULATOR_SETS:
            raise ValueError(f"Unknown emulator set {name!r}. Allowed: {', '.join(_EMULATOR_SETS)}.")
        if pknl_mode not in ("hmcode", "halofit"):
            raise ValueError(f'pknl_mode must be "hmcode" or "halofit", got {pknl_mode!r}.')
        self.name, self.pknl_mode = name, pknl_mode
        self._spec = spec = _EMULATOR_SETS[name]
        params = {**_LCDM_PARAMS, "tau": 0.0544, **{n: _EXTENSION_DEFAULTS[n] for n in spec["free"]}}
        fixed = {n: v for n, v in {**_STANDARD_FIXED, "deg_ncdm": spec["deg_ncdm"]}.items() if n not in params}
        self.params, self.fixed = _freeze(params, fixed)

        self.z_max_pk = spec["z_max_pk"]
        self._z_bg = jnp.linspace(0.0, self.z_max_bg, 5000)
        if spec["grid"] == "v2":
            self.k_grid = np.geomspace(5e-4, 10.0, 1000)
            self._pk_fac = self.k_grid ** -3
        else:
            self.k_grid = np.geomspace(1e-4, 50.0, 5000)[::10]
            ell = np.arange(2, 5002)[::10]
            self._pk_fac = (ell * (ell + 1.0) / (2.0 * np.pi)) ** -1

        if os.environ.get("READTHEDOCS") != "True":
            for key in ("HZ", "DAZ", "S8Z", "PKL", "PKNL", "DER"):
                self._emu(key)

    def _key(self):
        return (self.name, self.pknl_mode)

    def __repr__(self):
        return f"EmulatorEngine({self.name!r}, pknl_mode={self.pknl_mode!r})"

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

    def compute_hubble_parameter(self, z, p):
        """Emulated :math:`H(z)`; see :meth:`Engine.compute_hubble_parameter`."""
        hz = 10.0 ** self._emu("HZ").predictions(self._inputs(p)) * _C_KMS
        return jnp.interp(z, self._z_bg, hz, left=jnp.nan, right=jnp.nan)

    def compute_angular_diameter_distance(self, z, p):
        """Emulated :math:`D_A(z)`; see :meth:`Engine.compute_angular_diameter_distance`."""
        da = self._emu("DAZ").predictions(self._inputs(p))
        if self._spec["log_bg"]:
            da = jnp.insert(10.0 ** da, 0, 0.0)
        return jnp.interp(z, self._z_bg, da, left=jnp.nan, right=jnp.nan)

    def compute_sigma8(self, z, p):
        """Emulated :math:`\\sigma_8(z)`; see :meth:`Engine.compute_sigma8`."""
        s8 = self._emu("S8Z").predictions(self._inputs(p))
        if self._spec["log_bg"]:
            s8 = 10.0 ** s8
        return jnp.interp(z, self._z_bg, s8, left=jnp.nan, right=jnp.nan)

    def _power(self, key, k, z, p):
        inputs = self._inputs(p)

        def one_z(z_i):
            pk = 10.0 ** self._emu(key).predictions({**inputs, "z_pk_save_nonclass": z_i}) * self._pk_fac
            return log_interp1d_extrap(k, self.k_grid, pk)

        return jax.vmap(one_z, out_axes=1)(z)

    def compute_pk(self, k, z, p, linear=True):
        """Emulated linear :math:`P(k, z)`, nonlinear from the HMcode emulator or halofit per :attr:`pknl_mode`; see :meth:`Engine.compute_pk`."""
        if not linear and self.pknl_mode == "halofit":
            return super().compute_pk(k, z, p, linear=False)
        return self._power("PKL" if linear else "PKNL", k, z, p)

    def compute_derived_parameters(self, p):
        """Emulated derived parameters (``100*theta_s``, ``sigma8``, ``z_star``, ``chi_star``, ``rs_drag``, ...); see :meth:`Engine.compute_derived_parameters`."""
        return dict(zip(_DERIVED_NAMES, self._emu("DER").ten_to_predictions(self._inputs(p))))

    def compute_cl_cmb(self, type, l, p):
        """Emulated CMB :math:`C_\\ell`; see :meth:`Engine.compute_cl_cmb`."""
        inputs = self._inputs(p)
        cl = self._emu(type).predictions(inputs) if type == "TE" else self._emu(type).ten_to_predictions(inputs)
        if type == "PP":
            cl = cl / (2.0 * jnp.pi)
        ell = jnp.arange(2, cl.shape[0] + 2)
        return jnp.interp(l, ell, cl, left=jnp.nan, right=jnp.nan)

    def in_bounds(self, p):
        """Whether the parameters the set takes lie inside its training ranges; see :meth:`Engine.in_bounds`."""
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

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


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


class AnalyticEngine(Engine):
    """
    Engine that computes a flat :math:`w_0\\mathrm{CDM}` cosmology from analytic formulae, for any parameter values.

    :math:`H(z)` comes from the Friedmann equation (massive neutrinos counted as matter) and
    :math:`D_A(z)` from its integral. The linear :math:`P(k, z)` uses the Eisenstein & Hu (1998)
    transfer function without massive-neutrino suppression, and the nonlinear one is halofit.
    There are no CMB spectra or derived parameters, so CMB-lensing tracers need ``z_source``.
    """
    params = {**_LCDM_PARAMS, "m_ncdm": 0.06, "N_ur": 3.046, "w0": -1.0, "T_cmb": 2.7255}

    def _key(self):
        return ()

    def compute_hubble_parameter(self, z, p):
        """:math:`H(z)` from the Friedmann equation with radiation, matter and constant-:math:`w_0` dark energy; see :meth:`Engine.compute_hubble_parameter`."""
        zp1 = 1.0 + z
        return p["H0"] * jnp.sqrt(p["Omega0_r"] * zp1**4 + p["Omega0_m"] * zp1**3
                                  + p["Omega_Lambda"] * zp1 ** (3.0 * (1.0 + p["w0"])))

    def compute_angular_diameter_distance(self, z, p):
        """:math:`D_A(z)` by 64-point Gauss-Legendre quadrature of :math:`c/H` in :math:`\\ln(1+z)`; see :meth:`Engine.compute_angular_diameter_distance`."""
        z = jnp.asarray(z)
        x_max = jnp.log1p(z)[..., None]
        zp1 = jnp.exp(0.5 * x_max * (_GL_NODES + 1.0))
        chi = 0.5 * x_max[..., 0] * jnp.sum(_GL_WEIGHTS * _C_KMS * zp1 / self.compute_hubble_parameter(zp1 - 1.0, p), axis=-1)
        return chi / (1.0 + z)

    def _growth(self, z, p, n_steps=400):
        """Linear growth normalised to :math:`D = a` in matter domination, from RK4 in :math:`\\ln a`."""
        om, w = p["Omega0_m"], p["w0"]
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

    def compute_pk(self, k, z, p, linear=True):
        """Linear :math:`P(k, z)` from :math:`A_s`, :math:`n_s`, the Eisenstein & Hu transfer function and the growth factor, nonlinear from halofit; see :meth:`Engine.compute_pk`."""
        if not linear:
            return super().compute_pk(k, z, p, linear=False)
        T = eisenstein_hu(k, p["omega_cdm"] + p["omega_b"], p["omega_b"], p["T_cmb"])
        delta2 = 0.16 * p["A_s"] * (k / 0.05) ** (p["n_s"] - 1.0) * (k * _C_KMS / p["H0"]) ** 4 * T**2
        D = self._growth(z, p) / p["Omega0_m"]
        return (2.0 * jnp.pi**2 / k**3 * delta2)[:, None] * D[None, :] ** 2


# ----------------------------------------------------------------------
# Combined engine
# ----------------------------------------------------------------------

class CombinedEngine(Engine):
    """
    Engine that takes the background from one engine and the power spectrum from another.

    A parameter fixed by either engine is fixed for both, so the two always see the same cosmology;
    two engines that fix the same parameter at different values cannot be combined.

    Parameters
    ----------
    background : Engine
        Source of :math:`H(z)`, :math:`D_A(z)`, the densities and any CMB spectra or derived parameters.
    power : Engine
        Source of the linear and nonlinear :math:`P(k, z)` and :math:`\\sigma_8(z)`.
    """

    def __init__(self, background, power):
        self.background, self.power = background, power
        fixed_bg, fixed_pk = background.fixed, power.fixed
        clash = {n for n in fixed_bg.keys() & fixed_pk.keys() if fixed_bg[n] != fixed_pk[n]}
        if clash:
            raise ValueError(f"{background!r} and {power!r} fix {', '.join(sorted(clash))} at different values.")
        fixed = {**fixed_pk, **fixed_bg}
        params = {n: v for n, v in {**power.params, **background.params}.items() if n not in fixed}
        self.params, self.fixed = _freeze(params, fixed)
        self.k_grid, self.z_max_pk = power.k_grid, power.z_max_pk
        self.z_max_bg = background.z_max_bg

    def _key(self):
        return (self.background, self.power)

    def __repr__(self):
        return f"CombinedEngine(background={self.background!r}, power={self.power!r})"

    def compute_hubble_parameter(self, z, p):
        """From the background engine; see :meth:`Engine.compute_hubble_parameter`."""
        return self.background.compute_hubble_parameter(z, p)

    def compute_angular_diameter_distance(self, z, p):
        """From the background engine; see :meth:`Engine.compute_angular_diameter_distance`."""
        return self.background.compute_angular_diameter_distance(z, p)

    def compute_densities(self, p):
        """From the background engine; see :meth:`Engine.compute_densities`."""
        return self.background.compute_densities(p)

    def compute_pk(self, k, z, p, linear=True):
        """From the power engine; see :meth:`Engine.compute_pk`."""
        return self.power.compute_pk(k, z, p, linear)

    def compute_sigma8(self, z, p):
        """From the power engine; see :meth:`Engine.compute_sigma8`."""
        return self.power.compute_sigma8(z, p)

    def compute_cl_cmb(self, type, l, p):
        """From the background engine; see :meth:`Engine.compute_cl_cmb`."""
        return self.background.compute_cl_cmb(type, l, p)

    def compute_derived_parameters(self, p):
        """From the background engine; see :meth:`Engine.compute_derived_parameters`."""
        return self.background.compute_derived_parameters(p)

    def in_bounds(self, p):
        """In bounds for both engines; see :meth:`Engine.in_bounds`."""
        return self.background.in_bounds(p) & self.power.in_bounds(p)
