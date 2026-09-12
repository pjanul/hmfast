"""
Non-Limber angular power spectrum for the 2-halo term -- FFTLog/Hankel-
transform engine only.

Implements a SwiftCl-style (Reymond, Reeves, Zhang, Refregier, arXiv:2505.22718)
FFTLog decomposition of the tracer kernel, exploiting the separable structure
already present in hmfast's own 2-halo model,

    P_2h(k, z) = P_lin(k, z) * I_1^1(k, z)^2,

to compute the "beyond-Limber" unequal-time projection without the double
line-of-sight integral against oscillatory spherical Bessel functions. The
1-halo term is not treated here -- it has no clean unequal-time
generalisation and dominates only at high ell, where the existing Limber
``Pk.cl_1h`` is already accurate.

This module holds only the low-level engine (``_cl_2h_nonlimber`` and its
``_D_kz``/``_fftlog_biased_coeffs`` helpers); the public entry point is
:meth:`~hmfast.stats.pk.Pk.cl_2h`, which dispatches per-ell between
``_cl_2h_nonlimber`` here and the Limber fallback also defined on ``Pk``.
"""

from functools import partial

import jax
import jax.numpy as jnp

from hmfast.tracers.cmb_lensing import CMBLensingTracer

jax.config.update("jax_enable_x64", True)


# FFTLog / Hankel-transform primitives

def _fftlog_biased_coeffs(f_chi, chi_min, chi_max, n_fft, bias, window=0.2):
    """FFTLog-decompose ``f_chi`` into c_n such that f_chi(chi) ~= sum_n c_n * chi**(bias + 1j*eta_n),
    with a Hann-type anti-aliasing roll-off (ported from fftlog-lss's CoefWindow) on the outer ``window`` fraction of modes."""
    chi = jnp.geomspace(chi_min, chi_max, n_fft)
    d_ln_chi = jnp.log(chi_max / chi_min) / (n_fft - 1)
    eta_n = 2.0 * jnp.pi * jnp.fft.fftfreq(n_fft, d=d_ln_chi)

    g = f_chi * chi ** (-bias)
    c_n = jnp.fft.fft(g, axis=-1) / n_fft

    m = jnp.round(eta_n * n_fft * d_ln_chi / (2.0 * jnp.pi))  # signed integer mode index
    n_cut = int(window * n_fft // 2)
    n_edge = n_fft // 2 - n_cut
    theta = (n_fft // 2 - jnp.abs(m)) / (n_cut - 1)
    ramp = theta - jnp.sin(2.0 * jnp.pi * theta) / (2.0 * jnp.pi)
    c_n = c_n * jnp.where(jnp.abs(m) > n_edge, ramp, 1.0)

    c_n = c_n * chi_min ** (-1j * eta_n)
    return c_n, eta_n


# Closed-form Hankel-transform coefficients for the n-th derivative of the spherical Bessel
# function, generalizing the plain (n=0) transform used throughout this module.

def _hankel_A_table(l_arr, p, der_bessel):
    """
    A_n(l,p) = (-1)**n * sqrt(pi)/4 * 2**(p+1-n) * Gamma(p+1)/Gamma(p+1-n)
               * Gamma((l+p+1-n)/2) / Gamma((2+n+l-p)/2),
    the closed-form Hankel-transform coefficient for ``\\int dx x**p j_l^(n)(x)`` (Fang, Krause,
    Eifler & MacCrann 2020, arXiv:1911.11947, eq. A.6, in this module's own (l, p) convention).
    ``der_bessel=0`` reduces exactly to the plain spherical-Bessel transform used elsewhere here.
    """
    n = float(der_bessel)
    log_poly = jax.scipy.special.loggamma(p[None, :] + 1.0) - jax.scipy.special.loggamma(p[None, :] + 1.0 - n)
    log_num = jax.scipy.special.loggamma(0.5 * (l_arr[:, None] + p[None, :] + 1.0 - n))
    log_den = jax.scipy.special.loggamma(0.5 * (2.0 + n + l_arr[:, None] - p[None, :]))
    return ((-1.0) ** n) * (jnp.sqrt(jnp.pi) / 4.0) * (2.0 ** (p[None, :] + 1.0 - n)) * jnp.exp(log_poly + log_num - log_den)


# The D(k, z) growth/bias factor, and the per-tracer kernel*D(k,chi) product

def _D_kz(halo_model, profile, k, z, z_fid=0.0):
    """D(k,z) = sqrt(P_lin(k,z)/P_lin(k,z_fid)) * I_1^1(k,z), the per-time factor of the separable P_2h(k;z1,z2) ~= P_lin(k,z_fid)*D(k,z1)*D(k,z2) ansatz."""
    cosmology = halo_model.cosmology
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)

    z_b = cosmology._z_grid_pk()[-1]
    in_bounds = z <= z_b
    growth_ratio = jnp.where(in_bounds, 1.0, cosmology.growth_factor(z) / cosmology.growth_factor(z_b))
    z_eval = jnp.where(in_bounds, z, z_b)

    I1 = jnp.reshape(halo_model._I(profile, k, z_eval, bias_order=1), (len(k), len(z)))

    Plin_zeval = jnp.reshape(cosmology.pk(k, z_eval, linear=True), (len(k), len(z)))
    Plin_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=True), (len(k), 1))

    return growth_ratio[None, :] * jnp.sqrt(Plin_zeval / Plin_zfid) * I1


# Non-Limber 2-halo term (private; use the public Pk.cl_2h)

@partial(jax.jit, static_argnames=("n_fft", "n_interp", "bias", "window"))
def _cl_2h_nonlimber(
    halo_model, tracer1, tracer2, l, z,
    z_fid=0.0,
    n_fft=None, n_interp=200,
    bias=0.1,
    window=0.2,
):
    """
    Non-Limber 2-halo C_ell via an FFTLog/closed-form Hankel transform.
    Fully jax-native: ``l`` and ``z`` may both be traced/jitted here (the
    public ``Pk.cl_2h`` still needs ``l`` concrete for its own per-ell
    Limber/non-Limber dispatch, independent of this function).
    ``z`` only sets the range/resolution of an internal chi
    grid (via min, max, length) and is auto-widened to cover each tracer's
    own support. ``k`` is always the cosmology's own native P(k) grid.
    ``n_fft``/``n_interp`` control internal grid resolutions; ``bias``
    (<1) is the FFTLog de-trending exponent; ``window`` is the fraction of
    FFTLog modes anti-aliased at the high-frequency end (see
    :func:`_fftlog_biased_coeffs`). ``tracer1``/``tracer2`` share one chi
    grid, so they can't use independent z-ranges.
    """
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    cosmology = halo_model.cosmology

    l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
    z = jnp.atleast_1d(z)
    z_min = jnp.min(z)
    z_max = jnp.max(z)
    n_fft = n_fft if n_fft is not None else len(z)

    # Widen z_max (never narrow) to cover each tracer's own declared support.
    z_max = jnp.max(jnp.array([z_max, *(float(t.z_max) for t in (tracer1, tracer2) if hasattr(t, "z_max")),
                                *(jnp.max(t.dndz[0]) for t in (tracer1, tracer2) if hasattr(t, "dndz"))]))

    involves_cmb_lensing = isinstance(tracer1, CMBLensingTracer) or isinstance(tracer2, CMBLensingTracer)
    chi_star = None
    if involves_cmb_lensing:
        # z_star~1090 is beyond the trained z-grid, so `cosmology` needs extrapolate_z=True or this silently returns NaN.
        derived = cosmology.derived_parameters()
        chi_star = derived["chi_star"]
        z_max = jnp.maximum(z_max, 1.05 * derived["z_star"])  # margin gives the taper room past chi_star

    # chi_min/chi_max are the FFTLog grid's own boundary, deliberately non-differentiable (stop_gradient), not part of the physics being fit.
    chi_min = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_min) * (1.0 + z_min))
    chi_max = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_max) * (1.0 + z_max))

    k_fine, _ = cosmology._pk_grid()
    k_min, k_max = k_fine[0], k_fine[-1]

    chi_nodes = jnp.geomspace(chi_min, chi_max, n_fft)
    # Invert chi(z) = angular_diameter_distance(z)*(1+z) via interpolation on a dense z grid extended to z=1200.
    z_bg = cosmology._z_grid_bg()
    z_dense = jnp.concatenate([z_bg, jnp.geomspace(z_bg[-1] + 1.0, 1200.0, 500)])
    chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)
    z_nodes = jnp.minimum(jnp.interp(chi_nodes, chi_dense, z_dense), z_max)  # guard chi->z round-trip overshoot past z_max

    k_anchors = jnp.geomspace(k_min, k_max, n_interp)
    log_ka, log_kf = jnp.log(k_anchors), jnp.log(k_fine)

    # Each tracer may contribute more than one kernel() entry (e.g. density + RSD); every
    # entry gets its own row here, tagged with which tracer and which der_bessel it belongs to.
    # chi_star is real only when a CMBLensingTracer is present; jnp.inf makes the mask a no-op for every other tracer.
    term_tracer_idx = []
    term_der_bessel = []
    f_chi_list = []
    for t_idx, t in enumerate(tracers):
        D_kz_t = _D_kz(halo_model, t.profile, k_anchors, z_nodes, z_fid=z_fid)
        chi_mask = chi_nodes < (chi_star if isinstance(t, CMBLensingTracer) else jnp.inf)
        for weight, der_bessel in t.kernel(cosmology, z_nodes):
            f_chi_list.append(jnp.where(chi_mask, jnp.atleast_1d(weight), 0.0)[None, :] * D_kz_t)
            term_tracer_idx.append(t_idx)
            term_der_bessel.append(der_bessel)

    f_chi_stack = jnp.stack(f_chi_list)  # (n_terms, n_interp, n_fft)
    c_n_stack, eta_n = _fftlog_biased_coeffs(f_chi_stack, chi_min, chi_max, n_fft, bias, window=window)  # leading axis broadcasts through for free

    interp_col = lambda col: jnp.interp(log_kf, log_ka, col)
    interp_batched = jax.vmap(jax.vmap(interp_col, in_axes=1, out_axes=1))  # outer: per term, inner: per chi/eta_n mode
    c_stack = interp_batched(jnp.real(c_n_stack)) + 1j * interp_batched(jnp.imag(c_n_stack))  # (n_terms, n_k, n_fft)

    p = bias + 1j * eta_n  # (n_fft,); already gradient-free via chi_min/chi_max's own stop_gradient above
    kp = k_fine[:, None] ** (-1.0 - bias) * jnp.exp(-1j * eta_n[None, :] * log_kf[:, None])  # (n_k, n_fft); n-independent

    # Sum each tracer's own terms' contributions into that tracer's Delta before cross-multiplying.
    Delta_per_tracer = [0.0] * len(tracers)
    for term_idx, (t_idx, der_bessel) in enumerate(zip(term_tracer_idx, term_der_bessel)):
        A_table = _hankel_A_table(l_arr, p, der_bessel)  # (N_ell, n_fft)
        Delta_per_tracer[t_idx] = Delta_per_tracer[t_idx] + jnp.real(
            jnp.einsum('en,kn->ke', A_table, c_stack[term_idx] * kp)
        )
    Delta1, Delta2 = Delta_per_tracer[0], Delta_per_tracer[-1]  # Delta2 is Delta1 when n_tracers == 1

    # Delta_l(k) is only trustworthy where its Bessel turning point (l+0.5)/k falls within [chi_min, chi_max].
    resonant_chi = (l_arr[None, :] + 0.5) / k_fine[:, None]  # (n_k, N_ell)
    validity_mask = (resonant_chi >= chi_min) & (resonant_chi <= chi_max)

    Plin_fid = jnp.reshape(cosmology.pk(k_fine, jnp.atleast_1d(z_fid), linear=True), (k_fine.shape[0],))

    integrand = k_fine[:, None] ** 2 * Plin_fid[:, None] * Delta1 * Delta2 * validity_mask
    Cl = (2.0 / jnp.pi) * jnp.trapezoid(integrand * k_fine[:, None], x=log_kf, axis=0)

    return jnp.squeeze(Cl)

