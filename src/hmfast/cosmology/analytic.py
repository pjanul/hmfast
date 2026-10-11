"""Flat w0CDM cosmology from analytic formulae."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.cosmology.cosmology import Cosmology, _Functions
from hmfast.cosmology.halofit import _halofit_pk_nonlinear
from hmfast.utils import Const

_C_KMS = Const._c_ / 1e3
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


def _analytic_omega_m(p):
    return AnalyticCosmology._from_params(p).omega_x(0.0, "matter")


def _analytic_hubble_parameter(z, p):
    cosmo = AnalyticCosmology._from_params(p)
    return p["H0"] * jnp.sqrt(cosmo.rho_x(z, "critical") / cosmo.rho_x(0.0, "critical"))


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
        return y, y

    y0 = jnp.array([1e-3, 1e-3])
    _, Y = jax.lax.scan(step, y0, x[:-1])
    D, dD = jnp.concatenate([y0[None], Y]).T
    # Cubic Hermite with dD/dln a at the nodes, so D is smooth and growth_rate is its derivative.
    xq = jnp.clip(-jnp.log1p(z), x[0], x[-1])
    i = jnp.clip(jnp.floor((xq - x[0]) / dx).astype(int), 0, n_steps - 2)
    t = (xq - x[i]) / dx
    return ((2 * t**3 - 3 * t**2 + 1) * D[i] + (t**3 - 2 * t**2 + t) * dx * dD[i]
            + (-2 * t**3 + 3 * t**2) * D[i + 1] + (t**3 - t**2) * dx * dD[i + 1])


def _analytic_pk_linear(k, z, p):
    om = _analytic_omega_m(p)
    T = eisenstein_hu(k, p["omega_cdm"] + p["omega_b"], p["omega_b"], p["T_cmb"])
    delta2 = 0.16 * p["A_s"] * (k / 0.05) ** (p["n_s"] - 1.0) * (k * _C_KMS / p["H0"]) ** 4 * T**2
    D = _analytic_growth(z, om, p["w0"]) / om
    return (2.0 * jnp.pi**2 / k**3 * delta2)[:, None] * D[None, :] ** 2


def _analytic_angular_diameter_distance(z, p):
    z = jnp.asarray(z)
    x_max = jnp.log1p(z)[..., None]
    zp1 = jnp.exp(0.5 * x_max * (_GL_NODES + 1.0))
    integrand = _C_KMS * zp1 / _analytic_hubble_parameter(zp1 - 1.0, p)
    return 0.5 * x_max[..., 0] * jnp.sum(_GL_WEIGHTS * integrand, axis=-1) / (1.0 + z)


def _analytic_growth_factor(z, p):
    om = _analytic_omega_m(p)
    return _analytic_growth(z, om, p["w0"]) / _analytic_growth(0.0, om, p["w0"])


_FUNCTIONS = {}  # k grid -> functions, so equal settings give equal treedefs


def _functions(k_grid):
    key = k_grid.tobytes()
    if key not in _FUNCTIONS:
        _FUNCTIONS[key] = _Functions(
            _analytic_hubble_parameter, _analytic_pk_linear, _analytic_angular_diameter_distance,
            partial(_halofit_pk_nonlinear, hubble_parameter=_analytic_hubble_parameter,
                    pk_linear=_analytic_pk_linear, k_grid=k_grid,
                    from_params=AnalyticCosmology._from_params),
            _analytic_growth_factor)
    return _FUNCTIONS[key]


class AnalyticCosmology(Cosmology):
    """
    Flat :math:`w_0\\mathrm{CDM}` cosmology from analytic formulae, valid for any parameter values and redshift.

    :math:`H(z)` is the Friedmann equation with photons, ``N_ur`` massless neutrinos, massive neutrinos counted
    as matter and constant-:math:`w_0` dark energy. The linear :math:`P(k, z)` is the Eisenstein & Hu (1998)
    spectrum times the linear growth factor, without massive-neutrino suppression, and the nonlinear one is halofit.
    Inherits from :class:`~hmfast.cosmology.Cosmology`, with these formulae as its functions.

    Parameters
    ----------
    H0, omega_cdm, omega_b, A_s, n_s, m_ncdm, N_ur, w0, T_cmb : float
        As in :class:`~hmfast.cosmology.Cosmology`.
    Omega_k, wa : None
        Fixed at 0: these formulae are for a flat universe with constant ``w0``, and setting one raises a ``TypeError``.
    tau, f_ede, z_c, theta_i, r : None
        Absent: these formulae do not use them, and setting one raises a ``TypeError``.
    ncdm_mode, k_range, n_k : str, tuple of float, int
        As in :class:`~hmfast.cosmology.Cosmology`.

    Examples
    --------
    >>> cosmo = AnalyticCosmology(w0=-0.9)
    """

    def __init__(self, *, H0=68.0, omega_cdm=0.12, omega_b=0.02246576, Omega_k=None, A_s=2.1053e-9, n_s=0.965,
                 tau=None, m_ncdm=0.06, N_ur=3.046, w0=-1.0, wa=None, T_cmb=2.7255,
                 f_ede=None, z_c=None, theta_i=None, r=None,
                 ncdm_mode="cb", k_range=(1e-4, 50.0), n_k=500):
        not_free = dict(Omega_k=Omega_k, tau=tau, wa=wa, f_ede=f_ede, z_c=z_c, theta_i=theta_i, r=r)
        params = dict(H0=H0, omega_cdm=omega_cdm, omega_b=omega_b, A_s=A_s, n_s=n_s, m_ncdm=m_ncdm,
                      N_ur=N_ur, w0=w0, T_cmb=T_cmb)
        k_grid = np.geomspace(*k_range, n_k)
        self._setup(params, _functions(k_grid), k_grid, ncdm_mode)
        self._check_functions()
        given = [n for n, v in not_free.items() if v is not None]
        if given:
            raise TypeError(self._not_parameters(given))

    def _fixed_params(self):
        return {"Omega_k": 0.0, "wa": 0.0}
