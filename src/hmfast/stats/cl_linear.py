"""
Private engine behind :meth:`hmfast.stats.pk.Pk.cl_linear`, the linearly-biased
angular power spectrum -- the linear-bias analogue of ``Pk.cl_1h``/``cl_2h``.

Projects ``kernel1(z) * bias1(z) * kernel2(z) * bias2(z) * P(k, z)`` over
redshift via the Limber approximation, or exactly (beyond-Limber) via the
same SwiftCl-style FFTLog engine used by ``Pk.cl_2h``'s non-Limber branch.
No halo-model mass integral is performed here -- this is the standard
large-scale linear-bias approximation used e.g. in CCL and class_sz.

This module plays the same role for ``Pk.cl_linear`` that ``nonlimber.py``
plays for ``Pk.cl_2h``: a private low-level engine, not a public API surface
of its own (importing only *from* ``nonlimber.py`` and
``tracers/cmb_lensing.py``, never editing them).
"""

from functools import partial

import jax
import jax.numpy as jnp

from hmfast.stats.nonlimber import _fftlog_biased_coeffs
from hmfast.tracers.cmb_lensing import CMBLensingTracer

jax.config.update("jax_enable_x64", True)


def _resolve_bias(bias, z):
    """
    Resolve a tracer's bias attribute into an array on ``z``.

    ``None`` -> unbiased (1); a scalar -> constant bias; a ``(z_b, b_vals)``
    tuple -> interpolated onto ``z`` (same convention as ``dndz``/``mag_bias``
    elsewhere in the tracer code); otherwise an array already matching
    ``z``'s shape.
    """
    z = jnp.atleast_1d(z)
    if bias is None:
        return jnp.ones_like(z)
    if isinstance(bias, tuple):
        z_b, b_vals = bias
        return jnp.interp(z, z_b, b_vals)
    return jnp.broadcast_to(jnp.atleast_1d(bias), z.shape)


def _D_kz_linear(cosmology, k, z, z_fid=0.0, nonlinear=False):
    """
    D(k,z) = sqrt(P(k,z)/P(k,z_fid)), the bias-free per-time factor of the
    separable P(k;z1,z2) ~= P(k,z_fid)*D(k,z1)*D(k,z2) ansatz used by the
    non-Limber branch of :func:`cl_linear`. This is the analogue of
    :func:`hmfast.stats.nonlimber._D_kz`, but without its halo-model mass
    integral (``I_1^1``) factor, since there is no halo occupation weighting
    in the linear-bias approximation. ``nonlinear`` selects P_nl over P_lin.
    """
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)

    z_b = cosmology._z_grid_pk()[-1]
    in_bounds = z <= z_b
    growth_ratio = jnp.where(in_bounds, 1.0, cosmology.growth_factor(z) / cosmology.growth_factor(z_b))
    z_eval = jnp.where(in_bounds, z, z_b)

    P_zeval = jnp.reshape(cosmology.pk(k, z_eval, linear=not nonlinear), (len(k), len(z)))
    P_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=not nonlinear), (len(k), 1))

    return growth_ratio[None, :] * jnp.sqrt(P_zeval / P_zfid)


@partial(jax.jit, static_argnames=("n_fft", "n_interp", "bias", "window", "nonlinear"))
def _cl_linear_nonlimber(
    cosmology, tracer1, tracer2, l, z,
    nonlinear=False,
    z_fid=0.0, n_fft=None, n_interp=200,
    bias=0.1, window=0.2,
):
    """
    Non-Limber linearly-biased C_ell via an FFTLog/closed-form Hankel
    transform -- a close copy of
    :func:`hmfast.stats.nonlimber._cl_2h_nonlimber`, with the halo-model
    ``D(k,z)`` factor (:func:`hmfast.stats.nonlimber._D_kz`) replaced by the
    bias-free :func:`_D_kz_linear`, and each tracer kernel multiplied by that
    tracer's own bias attribute if it has one (resolved via
    :func:`_resolve_bias`), instead of a profile's own occupation weighting.

    ``bias`` here is the FFTLog de-trending exponent (same name/meaning as
    ``Pk.cl_2h``'s own ``bias`` kwarg) -- unrelated to a tracer's own bias.

    Parameters
    ----------
    cosmology : Cosmology
        No halo-model mass integral is performed here, so this takes a
        ``Cosmology`` directly rather than a ``HaloModel``.
    """
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    bias1 = getattr(tracer1, "bias", None)
    bias2 = getattr(tracer2, "bias", None)
    biases = (bias1,) if tracer2 is tracer1 else (bias1, bias2)

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

    # chi_star is real only when a CMBLensingTracer is present; jnp.inf makes the mask a no-op for every other tracer.
    f_chi_stack = jnp.stack([
        jnp.where(chi_nodes < (chi_star if isinstance(t, CMBLensingTracer) else jnp.inf),
                  jnp.atleast_1d(t.kernel(cosmology, z_nodes)) * _resolve_bias(b, z_nodes), 0.0)[None, :]
        * _D_kz_linear(cosmology, k_anchors, z_nodes, z_fid=z_fid, nonlinear=nonlinear)
        for t, b in zip(tracers, biases)
    ])
    c_n_stack, eta_n = _fftlog_biased_coeffs(f_chi_stack, chi_min, chi_max, n_fft, bias, window=window)  # leading axis broadcasts through for free

    interp_col = lambda col: jnp.interp(log_kf, log_ka, col)
    interp_batched = jax.vmap(jax.vmap(interp_col, in_axes=1, out_axes=1))  # outer: per tracer, inner: per chi/eta_n mode
    c_stack = interp_batched(jnp.real(c_n_stack)) + 1j * interp_batched(jnp.imag(c_n_stack))  # (n_tracers, n_k, n_fft)

    # Closed-form Hankel-transform coefficients A_n(l) = 2**(p_n-1)*sqrt(pi)*Gamma((1+l+p_n)/2)/Gamma((2+l-p_n)/2), p_n = bias + 1j*eta_n.
    p = bias + 1j * eta_n  # (n_fft,); already gradient-free via chi_min/chi_max's own stop_gradient above
    log_num = jax.scipy.special.loggamma(0.5 * (1.0 + l_arr[:, None] + p[None, :]))
    log_den = jax.scipy.special.loggamma(0.5 * (2.0 + l_arr[:, None] - p[None, :]))
    A_table = (2.0 ** (p[None, :] - 1.0)) * jnp.sqrt(jnp.pi) * jnp.exp(log_num - log_den)  # (N_ell, n_fft)

    kp = k_fine[:, None] ** (-1.0 - bias) * jnp.exp(-1j * eta_n[None, :] * log_kf[:, None])  # (n_k, n_fft)
    Delta_stack = jnp.real(jnp.einsum('en,tkn->tke', A_table, c_stack * kp))  # (n_tracers, n_k, N_ell)
    Delta1, Delta2 = Delta_stack[0], Delta_stack[-1]  # Delta2 is Delta1 when n_tracers == 1

    # Delta_l(k) is only trustworthy where its Bessel turning point (l+0.5)/k falls within [chi_min, chi_max].
    resonant_chi = (l_arr[None, :] + 0.5) / k_fine[:, None]  # (n_k, N_ell)
    validity_mask = (resonant_chi >= chi_min) & (resonant_chi <= chi_max)

    P_fid = jnp.reshape(cosmology.pk(k_fine, jnp.atleast_1d(z_fid), linear=not nonlinear), (k_fine.shape[0],))

    integrand = k_fine[:, None] ** 2 * P_fid[:, None] * Delta1 * Delta2 * validity_mask
    Cl = (2.0 / jnp.pi) * jnp.trapezoid(integrand * k_fine[:, None], x=log_kf, axis=0)

    return jnp.squeeze(Cl)


@partial(jax.jit, static_argnames=("nonlinear",))
def _cl_linear_limber(cosmology, tracer1, tracer2, l, z, nonlinear=False):
    """
    Limber branch of :meth:`hmfast.stats.pk.Pk.cl_linear`, mirroring
    :meth:`hmfast.stats.pk.Pk._cl_limber` but with a direct
    ``cosmology.pk(...)`` call in place of ``pk_1h``/``pk_2h``'s halo-model
    mass integrals, and each tracer's own bias attribute (if it has one)
    multiplied into its kernel. ``l`` may be traced.

    Parameters
    ----------
    cosmology : Cosmology
        No halo-model mass integral is performed here, so this takes a
        ``Cosmology`` directly rather than a ``HaloModel``.
    """
    tracer2 = tracer1 if tracer2 is None else tracer2
    z = jnp.atleast_1d(z)

    def get_pk_slice(zi):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        return jnp.atleast_1d(cosmology.pk(ki, jnp.atleast_1d(zi), linear=not nonlinear)).flatten()

    P_grid = jax.vmap(get_pk_slice)(z)

    kernel1 = jnp.atleast_1d(tracer1.kernel(cosmology, z)) * _resolve_bias(getattr(tracer1, "bias", None), z)
    kernel2 = jnp.atleast_1d(tracer2.kernel(cosmology, z)) * _resolve_bias(getattr(tracer2, "bias", None), z)

    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)
    limber_weight = cosmology.comoving_volume_element(z) / chi**4

    integrand = P_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])
    return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))
