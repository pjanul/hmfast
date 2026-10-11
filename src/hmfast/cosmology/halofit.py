"""Halofit nonlinear matter power spectrum (Takahashi et al. 2012) with the Bird et al. 2012 massive-neutrino correction."""
import jax
import jax.numpy as jnp

from hmfast.utils import log_interp1d_extrap


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


def _halofit_pk_nonlinear(k, z, p, *, hubble_parameter, pk_linear, k_grid, from_params):
    """Halofit :math:`P_{NL}(k, z)` of shape :math:`(N_k, N_z)`, from the linear spectrum tabulated on ``k_grid``."""
    cosmo = from_params(p)
    omega_m0, f_nu = cosmo.omega_x(0.0, "matter"), cosmo.omega_x(0.0, "neutrinos_massive") / cosmo.omega_x(0.0, "matter")
    k_grid = jnp.asarray(k_grid)

    def one_z(z_i):
        z_i = jnp.atleast_1d(z_i)
        omega_m_z = omega_m0 * (1.0 + z_i[0]) ** 3 * (p["H0"] / hubble_parameter(z_i, p)[0]) ** 2
        pk_lin = pk_linear(k_grid, z_i, p)[:, 0]
        pk_nl = halofit(k_grid, pk_lin, omega_m_z, omega_m0, cosmo.w0, f_nu, p["H0"] / 100.0)
        return log_interp1d_extrap(k, k_grid, pk_nl)

    return jax.vmap(one_z, out_axes=1)(z)
