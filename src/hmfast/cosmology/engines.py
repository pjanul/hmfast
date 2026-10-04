"""Engines: the source of the background and matter power spectrum behind a Cosmology."""
import os

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.download import _get_default_data_path
from hmfast.utils import Const, log_interp1d_extrap

_C_KMS = Const._c_ / 1e3


class Engine:
    """
    Parent engine class from which cosmology engines inherit.

    An engine computes :math:`H(z)`, :math:`D_A(z)` and :math:`P(k, z)` for a
    :class:`~hmfast.cosmology.Cosmology`. Its methods receive ``p``, the cosmology's
    parameters and derived densities (``h``, ``Omega0_m``, ``Omega_Lambda``, ...).

    Child classes must implement :meth:`hubble_parameter` and :meth:`pk` (linear).

    Attributes
    ----------
    free_params : tuple of str
        Extension parameters (``m_ncdm``, ``N_ur``, ``w0``, ``f_ede``, ``z_c``, ``theta_i``, ``r``)
        the engine takes; the others are fixed at their defaults.
    extra_params : dict
        Parameters beyond the cosmology's own, with their default values.
    deg_ncdm : float
        Number of degenerate massive-neutrino states.
    k_grid : numpy.ndarray
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}` on which :math:`\\sigma(M)` and FFTLog transforms are computed.
    z_max_bg, z_max_pk : float
        Maximum redshift of the background and of :math:`P(k, z)`.
    """
    free_params = ()
    extra_params = {}
    deg_ncdm = 1.0
    k_grid = np.geomspace(1e-4, 50.0, 500)
    z_max_bg = 20.0
    z_max_pk = 10.0

    def hubble_parameter(self, z, p):
        """Required Hubble parameter evaluator, :math:`H(z)` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`."""
        raise NotImplementedError

    def pk(self, k, z, p, linear=True):
        """Required matter power spectrum evaluator, :math:`P(k, z)` in :math:`\\mathrm{Mpc}^3` with shape :math:`(N_k, N_z)`; the nonlinear default is halofit on the linear spectrum."""
        if linear:
            raise NotImplementedError
        k_grid = jnp.asarray(self.k_grid)
        f_nu = p["Omega0_ncdm"] / p["Omega0_m"]

        def one_z(z_i):
            omega_m_z = p["Omega0_m"] * (1.0 + z_i) ** 3 * (p["H0"] / self.hubble_parameter(z_i, p)) ** 2
            pk_lin = self.pk(k_grid, jnp.atleast_1d(z_i), p, linear=True)[:, 0]
            pk_nl = halofit(k_grid, pk_lin, omega_m_z, p["Omega0_m"], p["w0"], f_nu, p["h"])
            return log_interp1d_extrap(k, k_grid, pk_nl)

        return jax.vmap(one_z, out_axes=1)(z)

    def angular_diameter_distance(self, z, p):
        """Angular diameter distance :math:`D_A(z)` in :math:`\\mathrm{Mpc}`; the default integrates :math:`c/H` in a flat universe."""
        z_grid = jnp.linspace(0.0, self.z_max_bg, 5000)
        integrand = _C_KMS / self.hubble_parameter(z_grid, p)
        chi = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * jnp.diff(z_grid))])
        return jnp.interp(z, z_grid, chi, right=jnp.nan) / (1.0 + z)

    def derived_parameters(self, p):
        """Derived parameters such as ``z_star`` and ``chi_star`` (needed for CMB lensing); none by default."""
        raise NotImplementedError(f"{self.__class__.__name__} provides no derived parameters.")

    def cl_cmb(self, type, l, p):
        """CMB power spectrum :math:`C_\\ell` of ``type`` in {"TT", "EE", "TE", "PP"}; none by default."""
        raise NotImplementedError(f"{self.__class__.__name__} provides no CMB spectra.")

    def in_bounds(self, p):
        """Whether ``p`` is inside the engine's valid domain, outside which outputs are NaN; always True by default."""
        return True

    def __repr__(self):
        return f"{type(self).__name__}()"


# ----------------------------------------------------------------------
# Emulator engine
# ----------------------------------------------------------------------

# "free": extension parameters the set takes as inputs; "deg_ncdm": degenerate massive states.
_EMULATOR_SETS = {
    "lcdm:v1":        {"suffix": "v1",     "subdir": "lcdm",        "free": ()},
    "mnu:v1":         {"suffix": "mnu_v1", "subdir": "mnu",         "free": ("m_ncdm",)},
    "neff:v1":        {"suffix": "neff_v1", "subdir": "neff",       "free": ("N_ur",)},
    "wcdm:v1":        {"suffix": "w_v1",   "subdir": "wcdm",        "free": ("w0",)},
    "ede:v1":         {"suffix": "v1",     "subdir": "ede",         "free": ("m_ncdm", "N_ur", "f_ede", "z_c", "theta_i", "r"), "deg_ncdm": 3.0},
    "mnu-3states:v1": {"suffix": "v1",     "subdir": "mnu-3states", "free": ("m_ncdm",), "deg_ncdm": 3.0},
    "ede:v2":         {"suffix": "v2",     "subdir": "ede",         "free": ("m_ncdm", "N_ur", "f_ede", "z_c", "theta_i", "r"), "deg_ncdm": 3.0},
}

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
        ``"mnu-3states:v1"`` or ``"ede:v2"``.
    pknl_mode : {"hmcode", "halofit"}
        Nonlinear P(k) from the HMcode emulator, or halofit on the emulated linear P(k).
    """

    def __init__(self, name="lcdm:v1", pknl_mode="hmcode"):
        if name not in _EMULATOR_SETS:
            raise ValueError(f"Unknown emulator set {name!r}. Allowed: {', '.join(_EMULATOR_SETS)}.")
        if pknl_mode not in ("hmcode", "halofit"):
            raise ValueError(f'pknl_mode must be "hmcode" or "halofit", got {pknl_mode!r}.')
        self.name, self.pknl_mode = name, pknl_mode
        spec = _EMULATOR_SETS[name]
        self.free_params = spec["free"]
        self.deg_ncdm = spec.get("deg_ncdm", 1.0)

        is_v2 = name == "ede:v2"
        self.z_max_pk = 20.0 if is_v2 else 5.0
        self._z_bg = jnp.linspace(0.0, self.z_max_bg, 5000)
        if is_v2:
            self.k_grid = np.geomspace(5e-4, 10.0, 1000)
            self._pk_fac = self.k_grid ** -3
        else:
            self.k_grid = np.geomspace(1e-4, 50.0, 5000)[::10]
            ell = np.arange(2, 5002)[::10]
            self._pk_fac = (ell * (ell + 1.0) / (2.0 * np.pi)) ** -1

        if os.environ.get("READTHEDOCS") != "True":
            for key in ("HZ", "DAZ", "S8Z", "PKL", "PKNL", "DER"):
                self._emu(key)

    # Engines with the same settings are interchangeable, so jit caches are shared between them.
    def __eq__(self, other):
        return type(other) is EmulatorEngine and (self.name, self.pknl_mode) == (other.name, other.pknl_mode)

    def __hash__(self):
        return hash((EmulatorEngine, self.name, self.pknl_mode))

    def __repr__(self):
        return f"EmulatorEngine({self.name!r}, pknl_mode={self.pknl_mode!r})"

    def _emu(self, key):
        if (self.name, key) not in _WEIGHTS:
            spec = _EMULATOR_SETS[self.name]
            subdir, loader = _FILES[key]
            path = os.path.join(_get_default_data_path(), spec["subdir"], subdir, f"{key}_{spec['suffix']}")
            # Keep the weights concrete even when first reached under a trace.
            with jax.ensure_compile_time_eval():
                _WEIGHTS[(self.name, key)] = loader(path)
        return _WEIGHTS[(self.name, key)]

    @staticmethod
    def _inputs(p):
        """Translate hmfast parameter names to the emulators' input names."""
        return {"H0": p["H0"], "omega_cdm": p["omega_cdm"], "omega_b": p["omega_b"],
                "ln10^{10}A_s": jnp.log(1e10 * p["A_s"]), "n_s": p["n_s"], "tau_reio": p["tau"],
                "m_ncdm": p["m_ncdm"], "N_ur": p["N_ur"], "w0_fld": p["w0"], "fEDE": p["f_ede"],
                "log10z_c": jnp.log10(p["z_c"]), "thetai_scf": p["theta_i"], "r": p["r"]}

    def hubble_parameter(self, z, p):
        """Emulated :math:`H(z)` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}`."""
        hz = 10.0 ** self._emu("HZ").predictions(self._inputs(p)) * _C_KMS
        return jnp.interp(z, self._z_bg, hz, left=jnp.nan, right=jnp.nan)

    def angular_diameter_distance(self, z, p):
        """Emulated :math:`D_A(z)` in :math:`\\mathrm{Mpc}`."""
        da = self._emu("DAZ").predictions(self._inputs(p))
        if self.name == "ede:v2":
            da = jnp.insert(10.0 ** da, 0, 0.0)
        return jnp.interp(z, self._z_bg, da, left=jnp.nan, right=jnp.nan)

    def sigma8(self, z, p):
        """Emulated :math:`\\sigma_8(z)`."""
        s8 = self._emu("S8Z").predictions(self._inputs(p))
        if self.name == "ede:v2":
            s8 = 10.0 ** s8
        return jnp.interp(z, self._z_bg, s8, left=jnp.nan, right=jnp.nan)

    def _pk(self, key, k, z, p):
        inputs = self._inputs(p)

        def one_z(z_i):
            pk = 10.0 ** self._emu(key).predictions({**inputs, "z_pk_save_nonclass": z_i}) * self._pk_fac
            return log_interp1d_extrap(k, self.k_grid, pk)

        return jax.vmap(one_z, out_axes=1)(z)

    def pk(self, k, z, p, linear=True):
        """Emulated linear :math:`P(k, z)`; nonlinear from the HMcode emulator or halofit, per :attr:`pknl_mode`."""
        if not linear and self.pknl_mode == "halofit":
            return super().pk(k, z, p, linear=False)
        return self._pk("PKL" if linear else "PKNL", k, z, p)

    def derived_parameters(self, p):
        """Emulated derived parameters (``100*theta_s``, ``sigma8``, ``z_star``, ``chi_star``, ``rs_drag``, ...)."""
        values = self._emu("DER").ten_to_predictions(self._inputs(p))
        return dict(zip(_DERIVED_NAMES, values))

    def cl_cmb(self, type, l, p):
        """Emulated CMB power spectrum :math:`C_\\ell` of ``type`` in {"TT", "EE", "TE", "PP"}."""
        inputs = self._inputs(p)
        if type == "TE":
            cl = self._emu("TE").predictions(inputs)
        else:
            cl = self._emu(type).ten_to_predictions(inputs)
        if type == "PP":
            cl = cl / (2.0 * jnp.pi)
        ell = jnp.arange(2, cl.shape[0] + 2)
        return jnp.interp(l, ell, cl, left=jnp.nan, right=jnp.nan)

    def in_bounds(self, p):
        """Whether ``p`` lies inside the emulators' training ranges."""
        ns_min, ns_max = (0.8812, 1.0492) if self.name == "lcdm:v1" else (0.8, 1.2)
        ln_as = jnp.log(1e10 * p["A_s"])
        log10_zc = jnp.log10(p["z_c"])
        return ((ln_as >= 2.5) & (ln_as <= 3.5)
                & (p["omega_cdm"] >= 0.08) & (p["omega_cdm"] <= 0.20)
                & (p["omega_b"] >= 0.01933) & (p["omega_b"] <= 0.02533)
                & (p["H0"] >= 39.99) & (p["H0"] <= 100.01)
                & (p["n_s"] >= ns_min) & (p["n_s"] <= ns_max)
                & (p["tau"] >= 0.02) & (p["tau"] <= 0.12)
                & (p["m_ncdm"] >= 0.0) & (p["m_ncdm"] <= 0.33333)
                & (p["w0"] >= -2.0) & (p["w0"] <= -0.33)
                & (p["N_ur"] >= 0.49) & (p["N_ur"] <= 4.49)
                & (p["theta_i"] >= 0.1) & (p["theta_i"] <= 3.1)
                & (log10_zc >= 3.0) & (log10_zc <= 4.3)
                & (p["f_ede"] >= 0.001) & (p["f_ede"] <= 0.5)
                & (p["r"] >= 0.0) & (p["r"] <= 0.3))



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
    free_params = ("m_ncdm", "N_ur", "w0")

    def __eq__(self, other):
        return type(other) is AnalyticEngine

    def __hash__(self):
        return hash(AnalyticEngine)

    def hubble_parameter(self, z, p):
        """:math:`H(z)` in :math:`\\mathrm{km\\,s^{-1}\\,Mpc^{-1}}` from the Friedmann equation with radiation, matter and constant-:math:`w_0` dark energy."""
        zp1 = 1.0 + z
        return p["H0"] * jnp.sqrt(p["Omega0_r"] * zp1**4 + p["Omega0_m"] * zp1**3
                                  + p["Omega_Lambda"] * zp1 ** (3.0 * (1.0 + p["w0"])))

    def angular_diameter_distance(self, z, p):
        """:math:`D_A(z)` in :math:`\\mathrm{Mpc}`, by 64-point Gauss-Legendre quadrature of :math:`c/H` in :math:`\\ln(1+z)`."""
        z = jnp.asarray(z)
        x_max = jnp.log1p(z)[..., None]
        x = 0.5 * x_max * (_GL_NODES + 1.0)
        zp1 = jnp.exp(x)
        chi = 0.5 * x_max[..., 0] * jnp.sum(_GL_WEIGHTS * _C_KMS * zp1 / self.hubble_parameter(zp1 - 1.0, p), axis=-1)
        return chi / (1.0 + z)

    def growth_factor(self, z, p, n_steps=400):
        """Linear growth factor normalised to :math:`D = a` in matter domination, from RK4 in :math:`\\ln a`."""
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

    def pk(self, k, z, p, linear=True):
        """Linear :math:`P(k, z)` from :math:`A_s`, :math:`n_s`, the Eisenstein & Hu transfer function and the growth factor; nonlinear from halofit."""
        if not linear:
            return super().pk(k, z, p, linear=False)
        k_pivot = 0.05
        H0_c = p["H0"] / _C_KMS
        T = eisenstein_hu(k, p["omega_cdm"] + p["omega_b"], p["omega_b"], p["T_cmb"])
        delta2 = 0.16 * p["A_s"] * (k / k_pivot) ** (p["n_s"] - 1.0) * (k / H0_c) ** 4 * T**2
        D = self.growth_factor(z, p) / p["Omega0_m"]
        return (2.0 * jnp.pi**2 / k**3 * delta2)[:, None] * D[None, :] ** 2
