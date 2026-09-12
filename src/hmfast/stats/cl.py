"""
Angular power spectrum (C_ell) helpers: Limber and non-Limber, halo-model and
linear-bias engines. Private -- every name here is a helper used by Pk.cl_1h/cl_2h/
cl_linear (stats/pk.py), the only public C_ell entry points.
"""

from functools import partial

import jax
import jax.numpy as jnp

from hmfast.tracers.cmb_lensing import CMBLensingTracer

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# Shared primitives
# ------------------------------------------------------------------

def _clip_to_trained_grid(cosmology, z):
    """Clip z to the cosmology's trained P(k) grid; return (z_eval, growth_ratio) for z beyond it."""
    z_b = cosmology._z_grid_pk()[-1]
    in_bounds = z <= z_b
    growth_ratio = jnp.where(in_bounds, 1.0, cosmology.growth_factor(z) / cosmology.growth_factor(z_b))
    return jnp.where(in_bounds, z, z_b), growth_ratio


def _sum_der0(terms, z):
    """Sum every der_bessel=0 kernel term into one (Nz,)-shaped array."""
    total = jnp.zeros_like(jnp.atleast_1d(z))
    for weight, der_bessel in terms:
        if der_bessel == 0:
            total = total + jnp.atleast_1d(weight)
    return total


def _resolve_bias(bias, z):
    """Resolve a tracer's bias attribute (None / scalar / (z, b) tuple / array) onto z."""
    z = jnp.atleast_1d(z)
    if bias is None:
        return jnp.ones_like(z)
    if isinstance(bias, tuple):
        z_b, b_vals = bias
        return jnp.interp(z, z_b, b_vals)
    return jnp.broadcast_to(jnp.atleast_1d(bias), z.shape)


def _limber_trapz(cosmology, z, chi, P_grid, kernel1, kernel2):
    """Limber line-of-sight integral: P(k,z) * comoving-volume weight * kernel1 * kernel2, over z."""
    limber_weight = cosmology.comoving_volume_element(z) / chi**4
    integrand = P_grid * limber_weight[:, None] * kernel1 * kernel2
    return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))


def _dispatch_by_ell(l, l_limber, nonlimber_fn, limber_fn):
    """Route each multipole to the non-Limber (l < l_limber) or Limber (l >= l_limber) engine."""
    l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
    l_vals = [float(x) for x in l_arr]
    idx_low = [i for i, li in enumerate(l_vals) if li < l_limber]
    idx_high = [i for i, li in enumerate(l_vals) if li >= l_limber]

    result = jnp.zeros(len(l_vals), dtype=jnp.float64)
    if idx_low:
        low_idx = jnp.array(idx_low)
        result = result.at[low_idx].set(jnp.atleast_1d(nonlimber_fn(l_arr[low_idx])))
    if idx_high:
        high_idx = jnp.array(idx_high)
        result = result.at[high_idx].set(jnp.atleast_1d(limber_fn(l_arr[high_idx])))
    return jnp.squeeze(result)


def _raise_if_rsd_in_limber(tracer1, tracer2, l_limber):
    """Refuse an RSD (der_bessel=2) term under cl_linear's Limber branch instead of silently dropping it."""
    for t in (tracer1, tracer2):
        if getattr(t, "rsd", False):
            raise ValueError(
                f"{type(t).__name__} has rsd=True, but l_limber={l_limber} routes some multipoles "
                "through cl_linear's Limber approximation, which cannot project RSD -- raise "
                "l_limber to use the exact non-Limber branch everywhere, or set rsd=False."
            )


# ------------------------------------------------------------------
# Non-Limber core: SwiftCl-style FFTLog/Hankel transform (Reymond et al. 2025, arXiv:2505.22718)
# ------------------------------------------------------------------

def _fftlog_biased_coeffs(f_chi, chi_min, chi_max, n_fft, bias, window=0.2):
    """FFTLog-decompose f_chi(chi) ~= sum_n c_n * chi**(bias + 1j*eta_n), Hann-tapered at high n."""
    chi = jnp.geomspace(chi_min, chi_max, n_fft)
    d_ln_chi = jnp.log(chi_max / chi_min) / (n_fft - 1)
    eta_n = 2.0 * jnp.pi * jnp.fft.fftfreq(n_fft, d=d_ln_chi)

    g = f_chi * chi ** (-bias)
    c_n = jnp.fft.fft(g, axis=-1) / n_fft

    m = jnp.round(eta_n * n_fft * d_ln_chi / (2.0 * jnp.pi))
    n_cut = int(window * n_fft // 2)
    n_edge = n_fft // 2 - n_cut
    theta = (n_fft // 2 - jnp.abs(m)) / (n_cut - 1)
    ramp = theta - jnp.sin(2.0 * jnp.pi * theta) / (2.0 * jnp.pi)
    c_n = c_n * jnp.where(jnp.abs(m) > n_edge, ramp, 1.0)

    c_n = c_n * chi_min ** (-1j * eta_n)
    return c_n, eta_n


def _hankel_A_table(l_arr, p, der_bessel):
    """Closed-form Hankel-transform coefficient for the der_bessel-th derivative of j_l (FKEM20 eq. A.6)."""
    n = float(der_bessel)
    log_poly = jax.scipy.special.loggamma(p[None, :] + 1.0) - jax.scipy.special.loggamma(p[None, :] + 1.0 - n)
    log_num = jax.scipy.special.loggamma(0.5 * (l_arr[:, None] + p[None, :] + 1.0 - n))
    log_den = jax.scipy.special.loggamma(0.5 * (2.0 + n + l_arr[:, None] - p[None, :]))
    return ((-1.0) ** n) * (jnp.sqrt(jnp.pi) / 4.0) * (2.0 ** (p[None, :] + 1.0 - n)) * jnp.exp(log_poly + log_num - log_den)


def _nonlimber_cl(cosmology, tracer1, tracer2, l, z, D_kz_fns, bias_scale_fns, P_fid,
                   z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
    """Shared non-Limber engine behind cl_2h_nonlimber/cl_linear_nonlimber.

    D_kz_fns[i](k, z) is tracer i's separable factor; bias_scale_fns (or None) optionally
    rescales each kernel term; P_fid(k) gives the fiducial P(k, z_fid)."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)

    l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
    z = jnp.atleast_1d(z)
    z_min, z_max = jnp.min(z), jnp.max(z)
    n_fft = n_fft if n_fft is not None else len(z)

    # Widen z_max (never narrow) to cover each tracer's own declared support.
    z_max = jnp.max(jnp.array([z_max, *(float(t.z_max) for t in (tracer1, tracer2) if hasattr(t, "z_max")),
                                *(jnp.max(t.dndz[0]) for t in (tracer1, tracer2) if hasattr(t, "dndz"))]))

    involves_cmb_lensing = isinstance(tracer1, CMBLensingTracer) or isinstance(tracer2, CMBLensingTracer)
    chi_star = None
    if involves_cmb_lensing:
        derived = cosmology.derived_parameters()
        chi_star = derived["chi_star"]
        z_max = jnp.maximum(z_max, 1.05 * derived["z_star"])  # margin gives the taper room past chi_star

    # chi_min/chi_max are the FFTLog grid's own boundary, deliberately non-differentiable.
    chi_min = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_min) * (1.0 + z_min))
    chi_max = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_max) * (1.0 + z_max))

    k_fine, _ = cosmology._pk_grid()
    k_min, k_max = k_fine[0], k_fine[-1]

    chi_nodes = jnp.geomspace(chi_min, chi_max, n_fft)
    # Invert chi(z) via interpolation on a dense z grid extended to z=1200.
    z_bg = cosmology._z_grid_bg()
    z_dense = jnp.concatenate([z_bg, jnp.geomspace(z_bg[-1] + 1.0, 1200.0, 500)])
    chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)
    z_nodes = jnp.minimum(jnp.interp(chi_nodes, chi_dense, z_dense), z_max)

    k_anchors = jnp.geomspace(k_min, k_max, n_interp)
    log_ka, log_kf = jnp.log(k_anchors), jnp.log(k_fine)

    # Every kernel() term of every tracer gets its own row, tagged by tracer and der_bessel.
    term_tracer_idx, term_der_bessel, f_chi_list = [], [], []
    for t_idx, t in enumerate(tracers):
        D_kz_t = D_kz_fns[t_idx](k_anchors, z_nodes)
        chi_mask = chi_nodes < (chi_star if isinstance(t, CMBLensingTracer) else jnp.inf)
        for weight, der_bessel in t.kernel(cosmology, z_nodes):
            weight = jnp.atleast_1d(weight)
            if bias_scale_fns is not None:
                weight = bias_scale_fns[t_idx](weight, der_bessel, z_nodes)
            f_chi_list.append(jnp.where(chi_mask, weight, 0.0)[None, :] * D_kz_t)
            term_tracer_idx.append(t_idx)
            term_der_bessel.append(der_bessel)

    f_chi_stack = jnp.stack(f_chi_list)  # (n_terms, n_interp, n_fft)
    c_n_stack, eta_n = _fftlog_biased_coeffs(f_chi_stack, chi_min, chi_max, n_fft, bias, window=window)

    interp_col = lambda col: jnp.interp(log_kf, log_ka, col)
    interp_batched = jax.vmap(jax.vmap(interp_col, in_axes=1, out_axes=1))
    c_stack = interp_batched(jnp.real(c_n_stack)) + 1j * interp_batched(jnp.imag(c_n_stack))

    p = bias + 1j * eta_n
    kp = k_fine[:, None] ** (-1.0 - bias) * jnp.exp(-1j * eta_n[None, :] * log_kf[:, None])

    # Sum each tracer's own terms' contributions into that tracer's Delta before cross-multiplying.
    Delta_per_tracer = [0.0] * len(tracers)
    for term_idx, (t_idx, der_bessel) in enumerate(zip(term_tracer_idx, term_der_bessel)):
        A_table = _hankel_A_table(l_arr, p, der_bessel)
        Delta_per_tracer[t_idx] = Delta_per_tracer[t_idx] + jnp.real(
            jnp.einsum('en,kn->ke', A_table, c_stack[term_idx] * kp)
        )
    Delta1, Delta2 = Delta_per_tracer[0], Delta_per_tracer[-1]  # Delta2 is Delta1 when n_tracers == 1

    # Delta_l(k) is only trustworthy where its Bessel turning point (l+0.5)/k falls within [chi_min, chi_max].
    resonant_chi = (l_arr[None, :] + 0.5) / k_fine[:, None]
    validity_mask = (resonant_chi >= chi_min) & (resonant_chi <= chi_max)

    P_fid_vals = P_fid(k_fine)
    integrand = k_fine[:, None] ** 2 * P_fid_vals[:, None] * Delta1 * Delta2 * validity_mask
    Cl = (2.0 / jnp.pi) * jnp.trapezoid(integrand * k_fine[:, None], x=log_kf, axis=0)
    return jnp.squeeze(Cl)


# ------------------------------------------------------------------
# Halo-model Cl (backs Pk.cl_1h / Pk.cl_2h)
# ------------------------------------------------------------------

def _D_kz(halo_model, profile, k, z, z_fid=0.0):
    """D(k,z) = sqrt(P_lin(k,z)/P_lin(k,z_fid)) * I_1^1(k,z), the halo-model separable factor."""
    cosmology = halo_model.cosmology
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
    z_eval, growth_ratio = _clip_to_trained_grid(cosmology, z)

    I1 = jnp.reshape(halo_model._I(profile, k, z_eval, bias_order=1), (len(k), len(z)))
    Plin_zeval = jnp.reshape(cosmology.pk(k, z_eval, linear=True), (len(k), len(z)))
    Plin_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=True), (len(k), 1))
    return growth_ratio[None, :] * jnp.sqrt(Plin_zeval / Plin_zfid) * I1


@partial(jax.jit, static_argnames=("n_fft", "n_interp", "bias", "window"))
def _cl_2h_nonlimber(halo_model, tracer1, tracer2, l, z, z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
    """Non-Limber 2-halo Cl via _nonlimber_cl; backs Pk.cl_2h below l_limber. l/z may both be traced."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    cosmology = halo_model.cosmology

    D_kz_fns = [lambda k, z, t=t: _D_kz(halo_model, t.profile, k, z, z_fid=z_fid) for t in tracers]
    P_fid = lambda k: jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=True), (k.shape[0],))
    return _nonlimber_cl(cosmology, tracer1, tracer2, l, z, D_kz_fns, None, P_fid,
                          z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)


def _effective_kernel_limber(tracer, cosmology, z, l, z_lp, lp1h, lp3h, sqell):
    """Reduce one tracer's kernel() terms to an effective per-(z,l) Limber kernel.

    der_bessel=0 substitutes directly; der_bessel=2 (RSD) uses CCL's extended-Limber
    recipe (Chisari et al. 2019 Sec. 2.4.1) via the shifted-grid pieces z_lp/lp1h/lp3h/
    sqell (precomputed once in cl_limber, shared across tracer1/tracer2; None if unneeded)."""
    terms = tracer.kernel(cosmology, z)
    if all(der_bessel == 0 for _, der_bessel in terms):
        return _sum_der0(terms, z)

    l = jnp.atleast_1d(l)
    terms_lp = tracer.kernel(cosmology, z_lp.reshape(-1))

    total = jnp.zeros((z.shape[0], l.shape[0]))
    for weight, der_bessel in terms:
        weight = jnp.broadcast_to(jnp.atleast_1d(weight)[:, None], total.shape)
        if der_bessel == 0:
            total = total + weight
        elif der_bessel == 2:
            weight_lp = next(jnp.reshape(w2, z_lp.shape) for w2, db2 in terms_lp if db2 == 2)
            total = total + (
                sqell * 2 * weight_lp / lp3h[None, :] - (0.25 + 2 * l[None, :]) * weight / (lp1h[None, :] ** 2)
            )
        else:
            raise NotImplementedError(
                f"{type(tracer).__name__}.kernel() has a der_bessel={der_bessel} entry; "
                "cl_limber only knows how to project der_bessel in {0, 2}."
            )
    return total


@partial(jax.jit, static_argnums=(0,), static_argnames=("include_1h", "include_2h"))
def _cl_limber(pk_obj, halo_model, tracer1, tracer2, l, z, include_1h=False, include_2h=True, k_damp=0.01):
    """Limber Cl for either/both halo terms; helper behind Pk.cl_1h and the Limber branch of Pk.cl_2h.

    l may be traced; jitted with pk_obj static (Pk isn't a registered pytree). An RSD
    (der_bessel=2) term adds ~1.7x cost via CCL's extended-Limber recipe (see
    _effective_kernel_limber), computed once here and shared across tracer1/tracer2."""
    hm = halo_model
    cosmology = hm.cosmology
    tracer2 = tracer1 if tracer2 is None else tracer2
    z = jnp.atleast_1d(z)
    l = jnp.atleast_1d(l)

    z_eval, growth_ratio = _clip_to_trained_grid(cosmology, z)
    growth_ratio_sq = growth_ratio**2

    def get_pk_slice(zi, zi_eval):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        zi_eval = jnp.atleast_1d(zi_eval)
        p = 0.0
        if include_1h:
            p = p + pk_obj.pk_1h(hm, ki, zi_eval, tracer1.profile, tracer2.profile, k_damp=k_damp)
        if include_2h:
            p = p + pk_obj.pk_2h(hm, ki, zi_eval, tracer1.profile, tracer2.profile)
        return jnp.atleast_1d(p).flatten()

    P_grid = jax.vmap(get_pk_slice)(z, z_eval) * growth_ratio_sq[:, None]
    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)

    # Extended-Limber setup, done once here (not per-tracer) since it's tracer-independent.
    needs_extended = getattr(tracer1, "rsd", False) or getattr(tracer2, "rsd", False)
    if needs_extended:
        z_dense = cosmology._z_grid_bg()
        chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)

        lp1h, lp3h = l + 0.5, l + 1.5
        chi_lp = chi[:, None] * (lp3h / lp1h)[None, :]  # (Nz, Nl): same k_l, shifted chi
        z_lp_raw = jnp.interp(chi_lp.reshape(-1), chi_dense, z_dense).reshape(chi_lp.shape)
        z_lp, growth_ratio_lp = _clip_to_trained_grid(cosmology, z_lp_raw)
        # extrapolate_z=False cosmologies return NaN past z_b; fall back to no growth correction.
        growth_ratio_sq_lp = jnp.nan_to_num(growth_ratio_lp**2, nan=1.0)

        k_l = lp1h[None, :] / chi[:, None]  # (Nz, Nl) -- the SAME k as P_grid, at the shifted z'

        def _pk_pair(k_i, z_i):
            zi = jnp.atleast_1d(z_i)
            p = 0.0
            if include_1h:
                p = p + pk_obj.pk_1h(hm, jnp.atleast_1d(k_i), zi, tracer1.profile, tracer2.profile, k_damp=k_damp)
            if include_2h:
                p = p + pk_obj.pk_2h(hm, jnp.atleast_1d(k_i), zi, tracer1.profile, tracer2.profile)
            return jnp.squeeze(p)

        P_lp = jax.vmap(_pk_pair)(k_l.reshape(-1), z_lp.reshape(-1)).reshape(k_l.shape) * growth_ratio_sq_lp
        pk_ratio = jnp.abs(P_lp / P_grid)
        sqell = jnp.sqrt(lp1h[None, :] * pk_ratio / lp3h[None, :])
    else:
        z_lp = lp1h = lp3h = sqell = None

    kernel1 = _effective_kernel_limber(tracer1, cosmology, z, l, z_lp, lp1h, lp3h, sqell)
    kernel2 = _effective_kernel_limber(tracer2, cosmology, z, l, z_lp, lp1h, lp3h, sqell)
    kernel1 = kernel1[:, None] if kernel1.ndim == 1 else kernel1
    kernel2 = kernel2[:, None] if kernel2.ndim == 1 else kernel2

    return _limber_trapz(cosmology, z, chi, P_grid, kernel1, kernel2)


# ------------------------------------------------------------------
# Linear-bias Cl (backs Pk.cl_linear)
# ------------------------------------------------------------------

def _D_kz_linear(cosmology, k, z, z_fid=0.0, linear=True):
    """D(k,z) = sqrt(P(k,z)/P(k,z_fid)), the bias-free analogue of _D_kz."""
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
    z_eval, growth_ratio = _clip_to_trained_grid(cosmology, z)
    P_zeval = jnp.reshape(cosmology.pk(k, z_eval, linear=linear), (len(k), len(z)))
    P_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=linear), (len(k), 1))
    return growth_ratio[None, :] * jnp.sqrt(P_zeval / P_zfid)


@partial(jax.jit, static_argnames=("n_fft", "n_interp", "bias", "window", "linear"))
def _cl_linear_nonlimber(cosmology, tracer1, tracer2, l, z, linear=True,
                          z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
    """Non-Limber linearly-biased Cl via _nonlimber_cl; helper behind Pk.cl_linear below l_limber."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    biases = [getattr(t, "bias", None) for t in tracers]

    # A tracer's bias only scales its der_bessel=0 (density) term, never an RSD term.
    def _scale(t_idx):
        return lambda weight, der_bessel, z: weight * _resolve_bias(biases[t_idx], z) if der_bessel == 0 else weight
    bias_scale_fns = [_scale(i) for i in range(len(tracers))]

    D_kz_fns = [lambda k, z: _D_kz_linear(cosmology, k, z, z_fid=z_fid, linear=linear) for _ in tracers]
    P_fid = lambda k: jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=linear), (k.shape[0],))
    return _nonlimber_cl(cosmology, tracer1, tracer2, l, z, D_kz_fns, bias_scale_fns, P_fid,
                          z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)


@partial(jax.jit, static_argnames=("linear",))
def _cl_linear_limber(cosmology, tracer1, tracer2, l, z, linear=True):
    """Limber helper behind Pk.cl_linear: raw cosmology.pk(...) in place of pk_1h/pk_2h, tracer bias in place of halo occupation."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    z = jnp.atleast_1d(z)

    def get_pk_slice(zi):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        return jnp.atleast_1d(cosmology.pk(ki, jnp.atleast_1d(zi), linear=linear)).flatten()

    P_grid = jax.vmap(get_pk_slice)(z)

    kernel1 = _sum_der0(tracer1.kernel(cosmology, z), z) * _resolve_bias(getattr(tracer1, "bias", None), z)
    kernel2 = _sum_der0(tracer2.kernel(cosmology, z), z) * _resolve_bias(getattr(tracer2, "bias", None), z)

    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)
    return _limber_trapz(cosmology, z, chi, P_grid, kernel1[:, None], kernel2[:, None])
