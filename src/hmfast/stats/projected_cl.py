"""
Angular power spectrum (C_ell): Limber and non-Limber, halo-model and
linear-bias engines. Every name except the public cl/cl_linbias functions at
the bottom of this module is a private helper.
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

# Lower end of the inferred redshift range; lensing and magnification kernels extend to z=0.
_Z_MIN = 1e-5
# Upper end of the inferred redshift range when no tracer kernel has a cutoff.
_Z_MAX_DEFAULT = 10.0
# Redshifts on which chi(z) is tabulated for inverting it.
_Z_INVERT = jnp.linspace(0.0, 20.0, 5000)


def _resolve_z_range(cosmology, *tracers):
    """``(_Z_MIN, z_max)`` with z_max the lowest tracer cutoff, or ``_Z_MAX_DEFAULT`` if none has one."""
    z_maxes = [z for z in (t._z_max(cosmology) for t in tracers) if z is not None] or [_Z_MAX_DEFAULT]
    z_max = z_maxes[0]
    for z in z_maxes[1:]:
        z_max = jnp.minimum(z_max, z)
    return (_Z_MIN, z_max)


def _extended_limber_kernel_grid(cosmology, l, z, chi, P_grid, pk_fn, z_cut):
    """Shifted-grid quantities (z_lp, lp1h, lp3h, sqell) for the extended-Limber RSD
    projection (Chisari et al. 2019 Sec. 2.4.1), shared by cl_limber's and
    cl_linear_limber's Limber engines. pk_fn(k, z) -> P(k, z) in that engine's own P(k);
    z_cut is the top of the redshift range, above which every kernel term vanishes."""
    z_dense = _Z_INVERT
    chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)

    lp1h, lp3h = l + 0.5, l + 1.5
    chi_lp = chi[:, None] * (lp3h / lp1h)[None, :]  # (Nz, Nl): same k_l, shifted chi
    z_lp_raw = jnp.interp(chi_lp.reshape(-1), chi_dense, z_dense).reshape(chi_lp.shape)
    # Shifted redshifts past the range carry no kernel weight, so evaluate them at its top and zero them below.
    in_range_lp = z_lp_raw <= z_cut
    z_lp = jnp.where(in_range_lp, z_lp_raw, z_cut)

    k_l = lp1h[None, :] / chi[:, None]  # (Nz, Nl) -- the SAME k as P_grid, at the shifted z'
    pk_pair = lambda k_i, z_i: jnp.squeeze(pk_fn(jnp.atleast_1d(k_i), jnp.atleast_1d(z_i)))
    P_lp = jax.vmap(pk_pair)(k_l.reshape(-1), z_lp.reshape(-1)).reshape(k_l.shape)

    pk_ratio = jnp.abs(P_lp / P_grid)
    sqell = jnp.where(in_range_lp, jnp.sqrt(lp1h[None, :] * pk_ratio / lp3h[None, :]), 0.0)
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

# Fraction of C_l the non-Limber k integral may drop below its first node; lensing at l=2, the slowest case, falls as k^4 there.
_K_TAIL_TOL = 1e-8


def _k_lo_nonlimber(k_grid, l_min, chi_max):
    """Lowest k of the non-Limber integral: _K_TAIL_TOL**(1/4) times the turn-on k = (l_min + 1/2) / chi_max, never above the P(k) grid."""
    return jnp.minimum(_K_TAIL_TOL ** 0.25 * (l_min + 0.5) / chi_max, k_grid[0])


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


def _hankel_A_table_term(l_arr, p, der_bessel):
    """Hankel-transform coefficient for one kernel term; der_bessel=-1 (j_l(x)/x^2) is the n=0 table at p-2."""
    if der_bessel == -1:
        return _hankel_A_table(l_arr, p - 2.0, 0)
    return _hankel_A_table(l_arr, p, der_bessel)


def _f_ell(l, der_angles):
    """Angular prefactor f_l^(a): 1, l(l+1), or sqrt((l+2)!/(l-2)!) in CCL's piecewise form (ccl_cl_tracer_t_get_f_ell)."""
    l = jnp.asarray(l, dtype=jnp.float64)
    if der_angles == 0:
        return jnp.ones_like(l)
    if der_angles == 1:
        return l * (l + 1.0)
    if der_angles == 2:
        lp1h2 = (l + 0.5) ** 2
        exact = jnp.sqrt(jnp.clip((l + 2.0) * (l + 1.0) * l * (l - 1.0), 0.0))
        return jnp.where(l <= 1, 0.0, jnp.where(l <= 10, exact, jnp.where(l <= 1000, lp1h2 * (1.0 - 1.25 / lp1h2), lp1h2)))
    raise ValueError(f"der_angles must be 0, 1 or 2; got {der_angles}.")


def _nonlimber_cl(cosmology, tracer1, tracer2, l, z_range, n_z, D_kz_fns, bias_scale_fns, P_fid,
                   z_fid=0.0, n_fft=512, n_interp=200, bias=0.1, window=0.2, n_k=1024):
    """Shared non-Limber engine behind cl_2h_nonlimber/cl_linear_nonlimber.

    D_kz_fns[i](k, z) is tracer i's separable factor; bias_scale_fns (or None) optionally
    rescales each kernel term; P_fid(k) gives the fiducial P(k, z_fid)."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)

    l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
    z_min, z_max = jnp.asarray(z_range[0]), jnp.asarray(z_range[1])

    # Widen z_max (never narrow) to cover each tracer's own declared support.
    z_max = jnp.max(jnp.array([z_max, *(float(t.z_max) for t in (tracer1, tracer2) if hasattr(t, "z_max")),
                                *(jnp.max(t.dndz[0]) for t in (tracer1, tracer2) if hasattr(t, "dndz"))]))

    # Widen z_max past any CMB-lensing source plane, so the kernel's support ends inside the grid.
    for t in (tracer1, tracer2):
        if isinstance(t, CMBLensingTracer):
            z_max = jnp.maximum(z_max, 1.05 * t.z_source)

    # chi_min/chi_max bound the kernels' support, deliberately non-differentiable.
    chi_min = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_min) * (1.0 + z_min))
    chi_max = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_max) * (1.0 + z_max))

    k_grid = cosmology._pk_grid()
    # The projection has support below the cosmology's k grid, down to its turn-on.
    k_lo = _k_lo_nonlimber(k_grid, jnp.min(l_arr), chi_max)
    # Integrated on its own grid, not the P(k) nodes: Delta_l(k) oscillates faster than the P(k) grid resolves.
    k_fine = jnp.geomspace(k_lo, k_grid[-1], n_k)
    k_max = k_fine[-1]

    # Zero-pad n_fft//4 nodes on each side, as SwiftCl does, so FFTLog's periodic wrap never meets a non-zero kernel edge.
    n_pad = n_fft // 4
    n_tot = n_fft + 2 * n_pad
    d_ln_chi = jnp.log(chi_max / chi_min) / (n_fft - 1)
    node_idx = jnp.arange(n_tot) - n_pad
    chi_nodes = chi_min * jnp.exp(d_ln_chi * node_idx)
    in_support = (node_idx >= 0) & (node_idx <= n_fft - 1)
    # Invert chi(z) via interpolation on a dense z grid extended to z=1200.
    z_dense = jnp.concatenate([_Z_INVERT, jnp.geomspace(_Z_INVERT[-1] + 1.0, 1200.0, 500)])
    chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)
    z_nodes = jnp.minimum(jnp.interp(jnp.clip(chi_nodes, chi_min, chi_max), chi_dense, z_dense), z_max)

    # D(k, z) is flat below the P(k) grid, so the anchors stay on it and the interpolation below holds the end value.
    k_anchors = jnp.geomspace(k_grid[0], k_max, n_interp)
    log_ka, log_kf = jnp.log(k_anchors), jnp.log(k_fine)

    # Every kernel() term of every tracer gets its own row, tagged by tracer, der_bessel and der_angles.
    term_tracer_idx, term_der_bessel, term_der_angles, f_chi_list = [], [], [], []
    for t_idx, t in enumerate(tracers):
        D_kz_t = D_kz_fns[t_idx](k_anchors, z_nodes)
        for weight, der_bessel, der_angles in t._kernel_terms(cosmology, z_nodes):
            weight = jnp.atleast_1d(weight)
            if bias_scale_fns is not None:
                weight = bias_scale_fns[t_idx](weight, der_bessel, z_nodes)
            weight = jnp.where(in_support, weight, 0.0)
            f_chi_list.append(weight[None, :] * D_kz_t)
            term_tracer_idx.append(t_idx)
            term_der_bessel.append(der_bessel)
            term_der_angles.append(der_angles)

    f_chi_stack = jnp.stack(f_chi_list)  # (n_terms, n_interp, n_tot)
    c_n_stack, eta_n = _fftlog_biased_coeffs(f_chi_stack, chi_nodes[0], chi_nodes[-1], n_tot, bias, window=window)

    interp_col = lambda col: jnp.interp(log_kf, log_ka, col)
    interp_batched = jax.vmap(jax.vmap(interp_col, in_axes=1, out_axes=1))
    c_stack = interp_batched(jnp.real(c_n_stack)) + 1j * interp_batched(jnp.imag(c_n_stack))

    p = bias + 1j * eta_n
    kp = k_fine[:, None] ** (-1.0 - bias) * jnp.exp(-1j * eta_n[None, :] * log_kf[:, None])

    # Sum each tracer's own terms' contributions into that tracer's Delta before cross-multiplying.
    Delta_per_tracer = [0.0] * len(tracers)
    for term_idx, (t_idx, der_bessel, der_angles) in enumerate(zip(term_tracer_idx, term_der_bessel, term_der_angles)):
        A_table = _hankel_A_table_term(l_arr, p, der_bessel) * _f_ell(l_arr, der_angles)[:, None]
        Delta_per_tracer[t_idx] = Delta_per_tracer[t_idx] + jnp.real(
            jnp.einsum('en,kn->ke', A_table, c_stack[term_idx] * kp)
        )
    Delta1, Delta2 = Delta_per_tracer[0], Delta_per_tracer[-1]  # Delta2 is Delta1 when n_tracers == 1

    P_fid_vals = P_fid(k_fine)
    integrand = k_fine[:, None] ** 2 * P_fid_vals[:, None] * Delta1 * Delta2
    Cl = (2.0 / jnp.pi) * jnp.trapezoid(integrand * k_fine[:, None], x=log_kf, axis=0)
    return jnp.squeeze(Cl)


# ------------------------------------------------------------------
# Halo-model Cl (backs cl below)
# ------------------------------------------------------------------

def _D_kz(cosmology, k, z, z_fid=0.0, linear=True, mass_integral=None):
    """D(k,z) = sqrt(P(k,z)/P(k,z_fid)) * mass_integral(k,z); the separable
    growth factor shared by the halo-model (mass_integral=I_1^1) and linear-bias
    (mass_integral=None) engines."""
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
    P_z = jnp.reshape(cosmology.pk(k, z, linear=linear), (len(k), len(z)))
    P_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=linear), (len(k), 1))
    D = jnp.sqrt(P_z / P_zfid)
    return D if mass_integral is None else D * mass_integral(k, z)


@partial(jax.jit, static_argnums=(5,), static_argnames=("n_fft", "n_interp", "bias", "window", "n_k"))
def _cl_2h_nonlimber(halo_model, tracer1, tracer2, l, z_range, n_z, z_fid=0.0, n_fft=512, n_interp=200, bias=0.1, window=0.2,
                     n_k=1024):
    """Non-Limber 2-halo Cl via _nonlimber_cl; backs cl below l_limber. l/z_range may both be traced."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    cosmology = halo_model.cosmology

    D_kz_fns = [
        lambda k, z, t=t: _D_kz(
            cosmology, k, z, z_fid=z_fid, linear=True,
            mass_integral=lambda k, z: jnp.reshape(halo_model.mass_integral(k, z, t.profile, bias_order=1), (len(k), len(z))),
        )
        for t in tracers
    ]
    P_fid = lambda k: jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=True), (k.shape[0],))
    return _nonlimber_cl(cosmology, tracer1, tracer2, l, z_range, n_z, D_kz_fns, None, P_fid,
                          z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window, n_k=n_k)


def _effective_kernel_limber(tracer, cosmology, z, l, z_lp, lp1h, lp3h, sqell, bias=None):
    """Reduce one tracer's kernel() terms to an effective per-(z,l) Limber kernel.

    der_bessel=0 substitutes directly; der_bessel=-1 (j_l/x^2) divides by (l+1/2)^2;
    der_bessel=2 (RSD) uses CCL's extended-Limber recipe (Chisari et al. 2019 Sec. 2.4.1)
    via the shifted-grid pieces z_lp/lp1h/lp3h/sqell (precomputed once by the caller, shared
    across tracer1/tracer2; None if unneeded). Every term is then multiplied by its angular
    prefactor f_l^(der_angles). If given, bias scales only the der_bessel=0 (density) term(s),
    never an RSD, magnification or IA term -- matches linear Kaiser, where only the density
    term picks up galaxy bias."""
    terms = tracer._kernel_terms(cosmology, z)
    b = (jnp.ones_like(z) if bias is None else jnp.interp(z, *bias) if isinstance(bias, tuple)
         else jnp.broadcast_to(jnp.atleast_1d(bias), z.shape))
    terms = [(weight * b if der_bessel == 0 else weight, der_bessel, der_angles)
             for weight, der_bessel, der_angles in terms]
    if all(der_bessel == 0 and der_angles == 0 for _, der_bessel, der_angles in terms):
        return sum((jnp.atleast_1d(weight) for weight, _, _ in terms), jnp.zeros_like(jnp.atleast_1d(z)))

    l = jnp.atleast_1d(l)
    if any(der_bessel > 0 for _, der_bessel, _ in terms):
        z_lp_flat = z_lp.reshape(-1)
        terms_lp = tracer._kernel_terms(cosmology, z_lp_flat)
        b_lp = (jnp.ones_like(z_lp_flat) if bias is None else jnp.interp(z_lp_flat, *bias) if isinstance(bias, tuple)
                else jnp.broadcast_to(jnp.atleast_1d(bias), z_lp_flat.shape))
        weights_lp = [weight * b_lp if der_bessel == 0 else weight for weight, der_bessel, _ in terms_lp]

    total = jnp.zeros((z.shape[0], l.shape[0]))
    for term_idx, (weight, der_bessel, der_angles) in enumerate(terms):
        weight = jnp.broadcast_to(jnp.atleast_1d(weight)[:, None], total.shape)
        if der_bessel == 0:
            term = weight
        elif der_bessel == -1:
            term = weight / ((l[None, :] + 0.5) ** 2)
        elif der_bessel == 2:
            weight_lp = jnp.reshape(weights_lp[term_idx], z_lp.shape)
            term = sqell * 2 * weight_lp / lp3h[None, :] - (0.25 + 2 * l[None, :]) * weight / (lp1h[None, :] ** 2)
        else:
            raise NotImplementedError(
                f"{type(tracer).__name__}.kernel() has a der_bessel={der_bessel} entry; "
                "cl_limber only knows how to project der_bessel in {-1, 0, 2}."
            )
        total = total + term * _f_ell(l, der_angles)[None, :]
    return total


@partial(jax.jit, static_argnums=(6,), static_argnames=("include_1h", "include_2h"))
def _cl_limber(pk_obj, halo_model, tracer1, tracer2, l, z_range, n_z, include_1h=False, include_2h=True):
    """Limber Cl for either/both halo terms; helper behind the Limber branch of cl.

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

    def pk_fn(k, z):
        p = 0.0
        if include_1h:
            p = p + pk_obj.pk_1h(hm, k, z, tracer1.profile, tracer2.profile)
        if include_2h:
            p = p + pk_obj.pk_2h(hm, k, z, tracer1.profile, tracer2.profile)
        return p

    def get_pk_slice(zi):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        return jnp.atleast_1d(pk_fn(ki, jnp.atleast_1d(zi))).flatten()

    P_grid = jax.vmap(get_pk_slice)(z)
    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)

    # Extended-Limber setup, done once here (not per-tracer) since it's tracer-independent.
    needs_extended = (
        any(der_bessel > 0 for _, der_bessel, _ in tracer1._kernel_terms(cosmology, z))
        or any(der_bessel > 0 for _, der_bessel, _ in tracer2._kernel_terms(cosmology, z))
    )
    if needs_extended:
        z_lp, lp1h, lp3h, sqell = _extended_limber_kernel_grid(cosmology, l, z, chi, P_grid, pk_fn, z_range[1])
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
# Linear-bias Cl (backs cl_linbias below)
# ------------------------------------------------------------------

@partial(jax.jit, static_argnums=(5,), static_argnames=("n_fft", "n_interp", "bias", "window", "linear", "n_k"))
def _cl_linear_nonlimber(cosmology, tracer1, tracer2, l, z_range, n_z, linear=True,
                          z_fid=0.0, n_fft=512, n_interp=200, bias=0.1, window=0.2, n_k=1024):
    """Non-Limber linearly-biased Cl via _nonlimber_cl; helper behind cl_linbias below l_limber."""
    tracer2 = tracer1 if tracer2 is None else tracer2
    tracers = (tracer1,) if tracer2 is tracer1 else (tracer1, tracer2)
    # A tracer's bias only scales its der_bessel=0 (density) term, never an RSD, magnification or IA term.
    bias_scale_fns = [lambda weight, der_bessel, z, b=getattr(t, "bias", None): (
        weight * (jnp.ones_like(z) if b is None else jnp.interp(z, *b) if isinstance(b, tuple)
                  else jnp.broadcast_to(jnp.atleast_1d(b), z.shape))
        if der_bessel == 0 else weight) for t in tracers]

    D_kz_fns = [lambda k, z: _D_kz(cosmology, k, z, z_fid=z_fid, linear=linear) for _ in tracers]
    P_fid = lambda k: jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=linear), (k.shape[0],))
    return _nonlimber_cl(cosmology, tracer1, tracer2, l, z_range, n_z, D_kz_fns, bias_scale_fns, P_fid,
                          z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window, n_k=n_k)


@partial(jax.jit, static_argnums=(5,), static_argnames=("linear",))
def _cl_linear_limber(cosmology, tracer1, tracer2, l, z_range, n_z, linear=True):
    """Limber helper behind cl_linbias: raw cosmology.pk(...) in place of pk_1h/pk_2h, tracer
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
        any(der_bessel > 0 for _, der_bessel, _ in tracer1._kernel_terms(cosmology, z))
        or any(der_bessel > 0 for _, der_bessel, _ in tracer2._kernel_terms(cosmology, z))
    )
    if needs_extended:
        z_lp, lp1h, lp3h, sqell = _extended_limber_kernel_grid(cosmology, l, z, chi, P_grid, pk_fn, z_range[1])
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


# ------------------------------------------------------------------
# Public entry points
# ------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_z", "l_limber", "n_fft", "n_interp", "fftlog_bias", "window", "n_k"))
def cl(pk, halo_model, l, tracer1, tracer2=None, *, n_z=100, l_limber=0.0,
       z_fid=0.0, n_fft=512, n_interp=200, fftlog_bias=0.1, window=0.2, n_k=1024):
    """
    Halo-model angular power spectrum :math:`C_\\ell` between two tracers.

    By default this uses the Limber approximation,

    .. math::

        C_\\ell = \\int d\\chi\\, \\frac{W_1(\\chi)\\, W_2(\\chi)}{\\chi^2}\\,
        P\\!\\left(k = \\frac{\\ell + 1/2}{\\chi}, z(\\chi)\\right),

    where :math:`W_i` are the tracer kernels and :math:`P = P_{1h} + P_{2h}`
    is the halo-model power spectrum, with the terms selected by ``pk``.

    The redshift integral runs from :math:`z = 10^{-5}` to the lowest redshift at
    which a tracer kernel vanishes (e.g. the top of ``dndz``, or ``z_max`` for tSZ),
    or to :math:`z = 10` if no tracer has one.

    Below ``l_limber``, the 2-halo term is instead projected exactly with the
    SwiftCl algorithm (`Reymond et al. 2025 <https://arxiv.org/abs/2505.22718>`_),
    which factorises the unequal-time 2-halo power spectrum as

    .. math::

        P_{2h}(k; z_1, z_2) = P_{\\rm lin}(k, z_{\\rm fid})\\, D_1(k, z_1)\\, D_2(k, z_2),

    with :math:`D_i(k, z)` absorbing the growth and the tracer's large-scale
    bias. Each tracer's line-of-sight integral against :math:`j_\\ell(k\\chi)`
    then becomes a single FFTLog transform, avoiding the double integral over
    oscillating Bessel functions. The 1-halo term is projected with Limber at
    every multipole: it is flat in :math:`k` on the scales where Limber fails.

    Parameters
    ----------
    pk : Pk
        Power spectrum object; ``pk.include_1h``/``include_2h`` select the terms.
    halo_model : HaloModel
        Halo model object.
    l : array-like
        Multipoles, shape :math:`(N_\\ell,)`.
    tracer1 : Tracer
        First tracer object.
    tracer2 : Tracer or None, default None
        Second tracer object. If None, defaults to ``tracer1``.
    n_z : int, default 100
        Number of redshift nodes of the Limber projection: the 1-halo term,
        and the 2-halo term at and above ``l_limber``.
    l_limber : float, default 0.0
        Multipoles below this use SwiftCl for the 2-halo term; at or above it,
        Limber. The default uses Limber everywhere.
    z_fid : float, default 0.0
        SwiftCl parameter, only used below ``l_limber``. Fiducial redshift
        :math:`z_{\\rm fid}` of the factorisation above.
    n_fft : int, default 512
        SwiftCl parameter, only used below ``l_limber``. Number of FFTLog
        nodes.
    n_interp : int, default 200
        SwiftCl parameter, only used below ``l_limber``. Number of wavenumbers
        at which :math:`D_i(k, z)` is evaluated before interpolation.
    fftlog_bias : float, default 0.1
        SwiftCl parameter, only used below ``l_limber``. FFTLog de-trending
        exponent, must be less than 1.
    window : float, default 0.2
        SwiftCl parameter, only used below ``l_limber``. Fraction of
        high-frequency FFTLog modes tapered to suppress ringing.
    n_k : int, default 1024
        SwiftCl parameter, only used below ``l_limber``. Number of
        log-spaced wavenumbers in the :math:`k` integral.

    Returns
    -------
    cl : jnp.ndarray
        Angular power spectrum with shape :math:`(N_\\ell,)`, where singleton
        dimensions get squeezed before return.
    """
    tracer2 = tracer1 if tracer2 is None else tracer2
    z_range = _resolve_z_range(halo_model.cosmology, tracer1, tracer2)

    def cl_low(l_low):
        # Only the 2-halo term goes non-Limber; the 1-halo term stays Limber at every multipole.
        out = jnp.zeros_like(jnp.atleast_1d(l_low))
        if pk.include_2h:
            out = out + jnp.atleast_1d(_cl_2h_nonlimber(halo_model, tracer1, tracer2, l_low, z_range, n_z, z_fid=z_fid,
                                                        n_fft=n_fft, n_interp=n_interp, bias=fftlog_bias, window=window,
                                                        n_k=n_k))
        if pk.include_1h:
            out = out + jnp.atleast_1d(_cl_limber(pk, halo_model, tracer1, tracer2, l_low, z_range, n_z,
                                                  include_1h=True, include_2h=False))
        return out

    return _dispatch_by_ell(
        l, l_limber, cl_low,
        lambda l_high: _cl_limber(pk, halo_model, tracer1, tracer2, l_high, z_range, n_z,
                                   include_1h=pk.include_1h, include_2h=pk.include_2h),
    )


@partial(jax.jit, static_argnames=("n_z", "linear", "l_limber", "n_fft", "n_interp", "fftlog_bias", "window", "n_k"))
def cl_linbias(cosmology, l, tracer1, tracer2=None, *, n_z=100, linear=True,
               l_limber=0.0, z_fid=0.0, n_fft=512, n_interp=200, fftlog_bias=0.1, window=0.2, n_k=1024):
    """
    Angular power spectrum :math:`C_\\ell` between two linearly biased tracers.

    By default this uses the Limber approximation,

    .. math::

        C_\\ell = \\int d\\chi\\, \\frac{W_1(\\chi)\\, W_2(\\chi)}{\\chi^2}\\,
        P_m\\!\\left(k = \\frac{\\ell + 1/2}{\\chi}, z(\\chi)\\right),

    where :math:`P_m` is the linear or nonlinear matter power spectrum and
    :math:`W_i` are the tracer kernels, with the density term scaled by the
    tracer's ``bias`` :math:`b_i(z)` (unbiased if the tracer has none).

    The redshift integral runs from :math:`z = 10^{-5}` to the lowest redshift at
    which a tracer kernel vanishes (e.g. the top of ``dndz``, or ``z_max`` for tSZ),
    or to :math:`z = 10` if no tracer has one.

    Below ``l_limber``, the spectrum is instead projected exactly with the
    SwiftCl algorithm (`Reymond et al. 2025 <https://arxiv.org/abs/2505.22718>`_),
    which factorises the unequal-time matter power spectrum as

    .. math::

        P_m(k; z_1, z_2) = P_m(k, z_{\\rm fid})\\, D(k, z_1)\\, D(k, z_2),
        \\qquad D(k, z) = \\sqrt{P_m(k, z) / P_m(k, z_{\\rm fid})}.

    Each tracer's line-of-sight integral against :math:`j_\\ell(k\\chi)` then
    becomes a single FFTLog transform, avoiding the double integral over
    oscillating Bessel functions.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object.
    l : array-like
        Multipoles, shape :math:`(N_\\ell,)`.
    tracer1 : Tracer
        First tracer object.
    tracer2 : Tracer or None, default None
        Second tracer object. If None, defaults to ``tracer1``.
    n_z : int, default 100
        Number of redshift nodes of the Limber projection, used at and above
        ``l_limber``.
    linear : bool, default True
        If True, use the linear matter power spectrum; if False, the nonlinear one.
    l_limber : float, default 0.0
        Multipoles below this use SwiftCl; at or above it, Limber. The default
        uses Limber everywhere.
    z_fid : float, default 0.0
        SwiftCl parameter, only used below ``l_limber``. Fiducial redshift
        :math:`z_{\\rm fid}` of the factorisation above.
    n_fft : int, default 512
        SwiftCl parameter, only used below ``l_limber``. Number of FFTLog
        nodes.
    n_interp : int, default 200
        SwiftCl parameter, only used below ``l_limber``. Number of wavenumbers
        at which :math:`D(k, z)` is evaluated before interpolation.
    fftlog_bias : float, default 0.1
        SwiftCl parameter, only used below ``l_limber``. FFTLog de-trending
        exponent, must be less than 1.
    window : float, default 0.2
        SwiftCl parameter, only used below ``l_limber``. Fraction of
        high-frequency FFTLog modes tapered to suppress ringing.
    n_k : int, default 1024
        SwiftCl parameter, only used below ``l_limber``. Number of
        log-spaced wavenumbers in the :math:`k` integral.

    Returns
    -------
    cl_linbias : jnp.ndarray
        Angular power spectrum with shape :math:`(N_\\ell,)`, where singleton
        dimensions get squeezed before return.
    """
    tracer2 = tracer1 if tracer2 is None else tracer2
    z_range = _resolve_z_range(cosmology, tracer1, tracer2)
    return _dispatch_by_ell(
        l, l_limber,
        lambda l_low: _cl_linear_nonlimber(cosmology, tracer1, tracer2, l_low, z_range, n_z, linear=linear,
                                            z_fid=z_fid, n_fft=n_fft, n_interp=n_interp,
                                            bias=fftlog_bias, window=window, n_k=n_k),
        lambda l_high: _cl_linear_limber(cosmology, tracer1, tracer2, l_high, z_range, n_z, linear=linear),
    )
