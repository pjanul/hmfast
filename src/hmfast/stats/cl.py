"""
Angular power spectrum (C_ell) helpers: Limber and non-Limber, halo-model and
linear-bias engines. Private -- every name here is a helper used by Pk.cl_hm/
cl_lin (stats/pk.py), the only public C_ell entry points.
"""

from functools import partial

import jax
import jax.numpy as jnp

from hmfast.tracers.cmb_lensing import CMBLensingTracer
from hmfast.utils import gauss_legendre_nodes_weights

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# Shared primitives
# ------------------------------------------------------------------

def _extended_limber_kernel_grid(cosmology, l, z, chi, P_grid, pk_fn):
    """Shifted-grid quantities (z_lp, lp1h, lp3h, sqell) for the extended-Limber RSD
    projection (Chisari et al. 2019 Sec. 2.4.1), shared by cl_limber's and
    cl_linear_limber's Limber engines. pk_fn(k, z) -> P(k, z) in that engine's own P(k)."""
    z_dense = cosmology._z_grid_bg()
    chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)

    lp1h, lp3h = l + 0.5, l + 1.5
    chi_lp = chi[:, None] * (lp3h / lp1h)[None, :]  # (Nz, Nl): same k_l, shifted chi
    z_lp_raw = jnp.interp(chi_lp.reshape(-1), chi_dense, z_dense).reshape(chi_lp.shape)
    # Clip to the cosmology's trained P(k) grid; extrapolate_z=False cosmologies return
    # NaN past it, so fall back to no growth correction there instead.
    z_b = cosmology._z_grid_pk()[-1]
    in_bounds_lp = z_lp_raw <= z_b
    growth_ratio_lp = jnp.where(in_bounds_lp, 1.0, cosmology.growth_factor(z_lp_raw) / cosmology.growth_factor(z_b))
    z_lp = jnp.where(in_bounds_lp, z_lp_raw, z_b)
    growth_ratio_sq_lp = jnp.nan_to_num(growth_ratio_lp**2, nan=1.0)

    k_l = lp1h[None, :] / chi[:, None]  # (Nz, Nl) -- the SAME k as P_grid, at the shifted z'
    pk_pair = lambda k_i, z_i: jnp.squeeze(pk_fn(jnp.atleast_1d(k_i), jnp.atleast_1d(z_i)))
    P_lp = jax.vmap(pk_pair)(k_l.reshape(-1), z_lp.reshape(-1)).reshape(k_l.shape) * growth_ratio_sq_lp

    pk_ratio = jnp.abs(P_lp / P_grid)
    sqell = jnp.sqrt(lp1h[None, :] * pk_ratio / lp3h[None, :])
    return z_lp, lp1h, lp3h, sqell


def _dispatch_by_ell(l, l_limber, nonlimber_fn, limber_fn):
    """Route each multipole to the non-Limber (l < l_limber) or Limber (l >= l_limber) engine."""
    l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))

    # l_limber is a plain scalar, so this resolves at trace time and leaves l traceable.
    if float(l_limber) <= 0.0:
        return jnp.squeeze(jnp.atleast_1d(limber_fn(l_arr)))

    if isinstance(l_arr, jax.core.Tracer):
        # A traced grid cannot be split by value, so run both engines and select pointwise.
        low = jnp.atleast_1d(nonlimber_fn(l_arr))
        high = jnp.atleast_1d(limber_fn(l_arr))
        return jnp.squeeze(jnp.where(l_arr < l_limber, low, high))

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


def _nonlimber_cl(cosmology, tracer1, tracer2, l, z_range, n_z, D_kz_fns, bias_scale_fns, P_fid,
                   z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
    """Shared non-Limber engine behind cl_2h_nonlimber/cl_linear_nonlimber.

    D_kz_fns[i](k, z) is tracer i's separable factor; bias_scale_fns (or None) optionally
    rescales each kernel term; P_fid(k) gives the fiducial P(k, z_fid)."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)

    l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
    z_min, z_max = jnp.asarray(z_range[0]), jnp.asarray(z_range[1])
    n_fft = n_fft if n_fft is not None else n_z

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
# Halo-model Cl (backs Pk.cl_hm)
# ------------------------------------------------------------------

def _D_kz(cosmology, k, z, z_fid=0.0, linear=True, mass_integral=None):
    """D(k,z) = sqrt(P(k,z)/P(k,z_fid)) * mass_integral(k,z_eval); the separable
    growth factor shared by the halo-model (mass_integral=I_1^1) and linear-bias
    (mass_integral=None) engines."""
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
    z_b = cosmology._z_grid_pk()[-1]
    in_bounds = z <= z_b
    growth_ratio = jnp.where(in_bounds, 1.0, cosmology.growth_factor(z) / cosmology.growth_factor(z_b))
    z_eval = jnp.where(in_bounds, z, z_b)
    P_zeval = jnp.reshape(cosmology.pk(k, z_eval, linear=linear), (len(k), len(z)))
    P_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=linear), (len(k), 1))
    D = growth_ratio[None, :] * jnp.sqrt(P_zeval / P_zfid)
    return D if mass_integral is None else D * mass_integral(k, z_eval)


@partial(jax.jit, static_argnums=(5,), static_argnames=("n_fft", "n_interp", "bias", "window"))
def _cl_2h_nonlimber(halo_model, tracer1, tracer2, l, z_range, n_z, z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
    """Non-Limber 2-halo Cl via _nonlimber_cl; backs Pk.cl_hm below l_limber. l/z_range may both be traced."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    cosmology = halo_model.cosmology

    D_kz_fns = [
        lambda k, z, t=t: _D_kz(
            cosmology, k, z, z_fid=z_fid, linear=True,
            mass_integral=lambda k, z: jnp.reshape(halo_model._I(t.profile, k, z, bias_order=1), (len(k), len(z))),
        )
        for t in tracers
    ]
    P_fid = lambda k: jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=True), (k.shape[0],))
    return _nonlimber_cl(cosmology, tracer1, tracer2, l, z_range, n_z, D_kz_fns, None, P_fid,
                          z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)


def _effective_kernel_limber(tracer, cosmology, z, l, z_lp, lp1h, lp3h, sqell, bias=None):
    """Reduce one tracer's kernel() terms to an effective per-(z,l) Limber kernel.

    der_bessel=0 substitutes directly; der_bessel=2 (RSD) uses CCL's extended-Limber
    recipe (Chisari et al. 2019 Sec. 2.4.1) via the shifted-grid pieces z_lp/lp1h/lp3h/
    sqell (precomputed once by the caller, shared across tracer1/tracer2; None if
    unneeded). If given, bias scales only the der_bessel=0 (density) term(s), never an
    RSD term -- matches linear Kaiser, where only the density term picks up galaxy bias."""
    terms = tracer.kernel(cosmology, z)
    b = (jnp.ones_like(z) if bias is None else jnp.interp(z, *bias) if isinstance(bias, tuple)
         else jnp.broadcast_to(jnp.atleast_1d(bias), z.shape))
    terms = [(weight * b if der_bessel == 0 else weight, der_bessel) for weight, der_bessel in terms]
    if all(der_bessel == 0 for _, der_bessel in terms):
        return sum((jnp.atleast_1d(weight) for weight, _ in terms), jnp.zeros_like(jnp.atleast_1d(z)))

    l = jnp.atleast_1d(l)
    z_lp_flat = z_lp.reshape(-1)
    terms_lp = tracer.kernel(cosmology, z_lp_flat)
    b_lp = (jnp.ones_like(z_lp_flat) if bias is None else jnp.interp(z_lp_flat, *bias) if isinstance(bias, tuple)
            else jnp.broadcast_to(jnp.atleast_1d(bias), z_lp_flat.shape))
    terms_lp = [(weight * b_lp if der_bessel == 0 else weight, der_bessel) for weight, der_bessel in terms_lp]

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


@partial(jax.jit, static_argnums=(6,), static_argnames=("include_1h", "include_2h"))
def _cl_limber(pk_obj, halo_model, tracer1, tracer2, l, z_range, n_z, include_1h=False, include_2h=True):
    """Limber Cl for either/both halo terms; helper behind the Limber branch of Pk.cl_hm.

    l may be traced; pk_obj is a registered Pk pytree (its k_damp attribute is a dynamic
    leaf), so it is no longer marked static. An RSD (der_bessel=2) term adds ~1.7x cost
    via the extended-Limber correction (see _effective_kernel_limber), computed once here
    and shared across tracer1/tracer2."""
    hm = halo_model
    cosmology = hm.cosmology
    tracer2 = tracer1 if tracer2 is None else tracer2
    logz, z_gl_w = gauss_legendre_nodes_weights(jnp.log(z_range[0]), jnp.log(z_range[1]), n_z)
    z = jnp.exp(logz)
    z_gl_w = z_gl_w * z  # Gauss-Legendre in ln(z); z spans orders of magnitude (e.g. to z_star)
    l = jnp.atleast_1d(l)

    z_b = cosmology._z_grid_pk()[-1]
    in_bounds = z <= z_b
    growth_ratio = jnp.where(in_bounds, 1.0, cosmology.growth_factor(z) / cosmology.growth_factor(z_b))
    z_eval = jnp.where(in_bounds, z, z_b)
    growth_ratio_sq = growth_ratio**2

    def pk_fn(k, z):
        p = 0.0
        if include_1h:
            p = p + pk_obj.pk_1h(hm, k, z, tracer1.profile, tracer2.profile)
        if include_2h:
            p = p + pk_obj.pk_2h(hm, k, z, tracer1.profile, tracer2.profile)
        return p

    def get_pk_slice(zi, zi_eval):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        return jnp.atleast_1d(pk_fn(ki, jnp.atleast_1d(zi_eval))).flatten()

    P_grid = jax.vmap(get_pk_slice)(z, z_eval) * growth_ratio_sq[:, None]
    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)

    # Extended-Limber setup, done once here (not per-tracer) since it's tracer-independent.
    needs_extended = (
        any(der_bessel != 0 for _, der_bessel in tracer1.kernel(cosmology, z))
        or any(der_bessel != 0 for _, der_bessel in tracer2.kernel(cosmology, z))
    )
    if needs_extended:
        z_lp, lp1h, lp3h, sqell = _extended_limber_kernel_grid(cosmology, l, z, chi, P_grid, pk_fn)
    else:
        z_lp = lp1h = lp3h = sqell = None

    kernel1 = _effective_kernel_limber(tracer1, cosmology, z, l, z_lp, lp1h, lp3h, sqell)
    kernel2 = _effective_kernel_limber(tracer2, cosmology, z, l, z_lp, lp1h, lp3h, sqell)
    kernel1 = kernel1[:, None] if kernel1.ndim == 1 else kernel1
    kernel2 = kernel2[:, None] if kernel2.ndim == 1 else kernel2

    limber_weight = cosmology.comoving_volume_element(z) / chi**4
    integrand = P_grid * limber_weight[:, None] * kernel1 * kernel2
    return jnp.squeeze(jnp.sum(integrand * z_gl_w[:, None], axis=0))


# ------------------------------------------------------------------
# Linear-bias Cl (backs Pk.cl_lin)
# ------------------------------------------------------------------

@partial(jax.jit, static_argnums=(5,), static_argnames=("n_fft", "n_interp", "bias", "window", "linear"))
def _cl_linear_nonlimber(cosmology, tracer1, tracer2, l, z_range, n_z, linear=True,
                          z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
    """Non-Limber linearly-biased Cl via _nonlimber_cl; helper behind Pk.cl_lin below l_limber."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    # A tracer's bias only scales its der_bessel=0 (density) term, never an RSD term.
    bias_scale_fns = [lambda weight, der_bessel, z, b=getattr(t, "bias", None): (
        weight * (jnp.ones_like(z) if b is None else jnp.interp(z, *b) if isinstance(b, tuple)
                  else jnp.broadcast_to(jnp.atleast_1d(b), z.shape))
        if der_bessel == 0 else weight) for t in tracers]

    D_kz_fns = [lambda k, z: _D_kz(cosmology, k, z, z_fid=z_fid, linear=linear) for _ in tracers]
    P_fid = lambda k: jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=linear), (k.shape[0],))
    return _nonlimber_cl(cosmology, tracer1, tracer2, l, z_range, n_z, D_kz_fns, bias_scale_fns, P_fid,
                          z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)


@partial(jax.jit, static_argnums=(5,), static_argnames=("linear",))
def _cl_linear_limber(cosmology, tracer1, tracer2, l, z_range, n_z, linear=True):
    """Limber helper behind Pk.cl_lin: raw cosmology.pk(...) in place of pk_1h/pk_2h, tracer
    bias in place of halo occupation. An RSD (der_bessel=2) kernel term is projected via the
    same extended-Limber correction as cl_limber's halo-model engine."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    logz, z_gl_w = gauss_legendre_nodes_weights(jnp.log(z_range[0]), jnp.log(z_range[1]), n_z)
    z = jnp.exp(logz)
    z_gl_w = z_gl_w * z  # Gauss-Legendre in ln(z); z spans orders of magnitude (e.g. to z_star)
    l = jnp.atleast_1d(l)
    pk_fn = lambda k, z: cosmology.pk(k, z, linear=linear)

    def get_pk_slice(zi):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        return jnp.atleast_1d(pk_fn(ki, jnp.atleast_1d(zi))).flatten()

    P_grid = jax.vmap(get_pk_slice)(z)
    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)

    needs_extended = (
        any(der_bessel != 0 for _, der_bessel in tracer1.kernel(cosmology, z))
        or any(der_bessel != 0 for _, der_bessel in tracer2.kernel(cosmology, z))
    )
    if needs_extended:
        z_lp, lp1h, lp3h, sqell = _extended_limber_kernel_grid(cosmology, l, z, chi, P_grid, pk_fn)
    else:
        z_lp = lp1h = lp3h = sqell = None

    kernel1 = _effective_kernel_limber(tracer1, cosmology, z, l, z_lp, lp1h, lp3h, sqell,
                                        bias=getattr(tracer1, "bias", None))
    kernel2 = _effective_kernel_limber(tracer2, cosmology, z, l, z_lp, lp1h, lp3h, sqell,
                                        bias=getattr(tracer2, "bias", None))
    kernel1 = kernel1[:, None] if kernel1.ndim == 1 else kernel1
    kernel2 = kernel2[:, None] if kernel2.ndim == 1 else kernel2

    limber_weight = cosmology.comoving_volume_element(z) / chi**4
    integrand = P_grid * limber_weight[:, None] * kernel1 * kernel2
    return jnp.squeeze(jnp.sum(integrand * z_gl_w[:, None], axis=0))
