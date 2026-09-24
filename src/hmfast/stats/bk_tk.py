from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.halos.profiles.profiles_2pt import _fourier_2pt
from hmfast.halos.profiles.hod import GalaxyHODProfile
from hmfast.utils import gauss_legendre_nodes_weights

from . import cl as _cl
from .pk import Pk as _Pk


def _extended_limber_grid_for_pair(hm, tracer_a, tracer_b, profile_a, profile_b, z, l, chi, k_damp=0.01):
    """
    Extended-Limber shifted-grid quantities (see :func:`hmfast.stats.cl._extended_limber_kernel_grid`)
    for one leg of a covariance -- ``tracer_a``/``tracer_b`` share a single multipole
    ``l`` and wavenumber :math:`k=(\\ell+1/2)/\\chi`, exactly as the two tracers of a
    single :math:`C_\\ell` would. Returns ``(None, None, None, None)`` if neither
    tracer has a ``der_bessel!=0`` (e.g. RSD) kernel term, in which case the caller's
    kernel product stays scalar-in-``l``, identical to the pre-extended-Limber behavior.

    The reference :math:`P(k,z)` used for the correction is this leg's own halo-model
    :math:`P_{1h}+P_{2h}` (built from ``profile_a``, ``profile_b``) -- the same
    :math:`P(k,z)` that would enter this leg's own :math:`C_\\ell` if computed directly
    via :meth:`~hmfast.stats.pk.Pk.cl_hm`, so that
    RSD's projection correction is treated consistently between the trispectrum/
    covariance code here and the two-point ``C_\\ell`` code in ``stats/cl.py``.
    """
    cosmology = hm.cosmology
    z_arr = jnp.atleast_1d(z)
    needs_extended = (
        any(der_bessel != 0 for _, der_bessel in tracer_a.kernel(cosmology, z_arr))
        or any(der_bessel != 0 for _, der_bessel in tracer_b.kernel(cosmology, z_arr))
    )
    if not needs_extended:
        return None, None, None, None

    pk = _Pk(k_damp=k_damp)

    def pk_fn(k, z):
        return pk.pk_1h(hm, k, z, profile_a, profile_b) + pk.pk_2h(hm, k, z, profile_a, profile_b)

    k_l = (l + 0.5) / chi
    P_grid = jnp.atleast_1d(pk_fn(k_l, z_arr)).flatten()[None, :]  # (1, Nl) -- single-z slice
    return _cl._extended_limber_kernel_grid(cosmology, l, z_arr, jnp.atleast_1d(chi), P_grid, pk_fn)


def _kernel_pair_effective(cosmology, tracer_a, tracer_b, z, l, z_lp, lp1h, lp3h, sqell):
    """
    Product of two tracers' effective per-``(z,l)`` Limber kernels
    (:func:`hmfast.stats.cl._effective_kernel_limber`), evaluated at a single ``z``
    (called once per redshift slice inside ``covariance_cng``/``covariance_ssc``'s
    ``vmap`` over ``z``). Every ``der_bessel=0`` term of each tracer (density,
    magnification bias, IA, ...) is summed directly; a ``der_bessel=2`` (RSD) term is
    projected through the extended-Limber correction (``z_lp``/``lp1h``/``lp3h``/
    ``sqell``, or ``None`` if neither tracer needs it -- see
    :func:`_extended_limber_grid_for_pair`).

    Returns shape ``(Nl,)``: a plain per-``z`` scalar broadcastable against ``Nl`` when
    neither tracer has an RSD term (identical to the old ``_kernel_density`` product),
    ``l``-dependent otherwise.
    """
    z_arr = jnp.atleast_1d(z)
    ka = _cl._effective_kernel_limber(tracer_a, cosmology, z_arr, l, z_lp, lp1h, lp3h, sqell)
    kb = _cl._effective_kernel_limber(tracer_b, cosmology, z_arr, l, z_lp, lp1h, lp3h, sqell)
    ka = ka[:, None] if ka.ndim == 1 else ka  # (1,1) or (1,Nl)
    kb = kb[:, None] if kb.ndim == 1 else kb
    return jnp.squeeze(ka * kb, axis=0)  # (Nl,) or (1,)


# -------------------------
# Perturbation theory helpers
# -------------------------
def _F2(k1, k2, mu):
    """Leading-order SPT kernel F2 (EdS approximation)."""
    k1_over_k2 = jnp.where(k2 == 0.0, 0.0, k1 / k2)
    k2_over_k1 = jnp.where(k1 == 0.0, 0.0, k2 / k1)
    return 5.0 / 7.0 + 0.5 * (k1_over_k2 + k2_over_k1) * mu + 2.0 / 7.0 * mu ** 2

@jax.jit
def _mu(k1, k2, k3):
    """Cosine between k1 and k2 given opposite side k3."""
    den = 2.0 * k1 * k2
    return jnp.where(
        den == 0.0,
        0.0,
        (k3 ** 2 - k1 ** 2 - k2 ** 2) / den
    )

@jax.jit
def _ksum(k1, k2, mu):
    """Magnitude of the vector sum of k1, k2 given the cosine of the angle between them."""
    return jnp.sqrt(k1 ** 2 + k2 ** 2 + 2.0 * k1 * k2 * mu)


def _check_bk_inputs(k1, k2, mu12):
    """k1, k2 must be positive wavenumbers; mu12 must be a genuine cosine.

    Validated with numpy rather than jnp: a jnp comparison inside a jit trace is staged
    into the jaxpr even for concrete inputs, so the bool() below would see a tracer and
    make every caller un-jittable. Traced inputs carry no values to check and are
    skipped.
    """
    if any(isinstance(x, jax.core.Tracer) for x in (k1, k2, mu12)):
        return
    if np.any(np.asarray(k1) <= 0.0) or np.any(np.asarray(k2) <= 0.0):
        raise ValueError("k1 and k2 must be positive.")
    if np.any(np.abs(np.asarray(mu12)) > 1.0):
        raise ValueError("mu12 must be in [-1, 1].")


# Trispectrum tree-level helpers (4-halo term)
#
# The 4-halo trispectrum term needs the tree-level kernel angle-averaged over
# the relative orientation of the k_u and k_v leg pairs -- a genuinely free
# angle in the parallelogram configuration, unlike the bispectrum's triangle
# angle which is fixed by k1,k2,k3. This is new numerical machinery not
# needed elsewhere in hmfast: a fixed-order Gauss-Legendre quadrature over
# theta in [0, pi], evaluating the tree-level trispectrum kernel of
# Takada & Hu (2013), Eq. 30 (arXiv:1302.6994).
_TRISPEC_N_THETA = 96
_trispec_gl_x, _trispec_gl_w = np.polynomial.legendre.leggauss(_TRISPEC_N_THETA)
_trispec_gl_x = jnp.asarray(_trispec_gl_x)
_trispec_gl_w = jnp.asarray(_trispec_gl_w)
_TRISPEC_COS_THETA = jnp.cos(0.5 * jnp.pi * (_trispec_gl_x + 1.0))
_TRISPEC_THETA_WEIGHT = _trispec_gl_w * (jnp.pi / 2.0) / jnp.pi


@jax.jit
def _X3(k, kp):
    """
    Closed-form, angle-averaged tree-level F3-type kernel entering the
    "1113" diagram of the 4-halo trispectrum, following Eq. 30 of
    Takada & Hu (2013). Depends only on the ratio r = kp/k.

    ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays, broadcast into an
    (Nk, Nkp) grid of pairs. Returns an (Nk, Nkp) array, not squeezed --
    this is only ever used as an internal building block of ``Tk.tk_4h``.
    """
    r = kp[None, :] / k[:, None]              # (Nk, Nkp)
    cth = _TRISPEC_COS_THETA[None, None, :]   # (1, 1, N_theta)
    wgt = _TRISPEC_THETA_WEIGHT[None, None, :]

    r_b = r[:, :, None]  # (Nk, Nkp, 1)
    kr = _ksum(k[:, None, None], kp[None, :, None], cth)  # (Nk, Nkp, N_theta)
    intd = (
        (5.0 * r_b + (7.0 - 2.0 * r_b ** 2) * cth) / (1.0 + r_b ** 2 + 2.0 * r_b * cth)
        * (3.0 / 7.0 * r_b + 0.5 * (1.0 + r_b ** 2) * cth + 4.0 / 7.0 * r_b * cth ** 2)
    )
    intd = jnp.where(kr == 0.0, 0.0, intd)

    isotropized = jnp.sum(intd * wgt, axis=2)  # (Nk, Nkp)
    return -7.0 / 4.0 * (1.0 + r ** 2) + isotropized



# -------------------------
# Halo model mass-integral building blocks
# -------------------------
#
# Shared across Bk and Tk (not tied to either class's state), so kept at
# module level rather than duplicated as a private method on each.

def _pair_integral(halo_model, p1, p2, k1, k2, z, outer=False, bias_order=1):
    """
    ∫ dn/dlnM * b_beta(M) * p1.fourier(k1,M,z) * p2.fourier(k2,M,z) dlnM

    Pair integral with halo bias of order ``beta = bias_order`` included
    (``0``: unweighted, ``1``: linear bias -- the default, matching every
    existing caller below, which all omit the argument). Used as a building
    block for ``Bk.bk_2h`` (``outer=False``: ``k1`` and ``k2`` share a single
    batch axis, paired elementwise across a set of triangles) and the
    halo-model trispectrum's 2h/3h terms (``outer=True``: ``k1`` and ``k2``
    are independent and broadcast into an (N1, N2) grid).

    Future generalisation: replace ``u1 * u2`` with a ``_fourier_2pt``
    variant that handles different k values when specialised 2-point kernels
    (HOD, CIB) are needed.

    Returns
    -------
    array
        ``outer=False``: shape (Nk, Nz), singleton dimensions squeezed.
        ``outer=True``: shape (N1, N2, Nz), not squeezed -- this branch is
        only ever used as an internal building block of ``Tk.tk_2h``/
        ``Tk.tk_3h``.
    """
    hm = halo_model
    z_arr = jnp.atleast_1d(z)
    logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
    m = jnp.exp(logm)

    dndlnm = jnp.reshape(
        hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def),
        (len(m), len(z_arr)),
    )
    if bias_order == 0:
        bias_w = jnp.ones((len(m), len(z_arr)))
    else:
        bias_w = jnp.reshape(
            hm.halo_bias.bias(hm.cosmology, m, z_arr, hm.mass_def, order=bias_order),
            (len(m), len(z_arr)),
        )
    total_weights = dndlnm * bias_w * w[:, None]  # (Nm, Nz)

    k1s, k2s = jnp.atleast_1d(k1), jnp.atleast_1d(k2)
    u1 = jnp.reshape(p1.fourier(hm, k1s, m, z_arr), (len(k1s), len(m), len(z_arr)))
    u2 = jnp.reshape(p2.fourier(hm, k2s, m, z_arr), (len(k2s), len(m), len(z_arr)))

    n_min, b1_min, b2_min = hm._counter_terms(z_arr)
    b_min = {0: jnp.ones_like(b1_min), 1: b1_min, 2: b2_min}[bias_order]

    if outer:
        u1e = u1[:, None, :, :]  # (N1, 1, Nm, Nz)
        u2e = u2[None, :, :, :]  # (1, N2, Nm, Nz)
        integral = jnp.sum(u1e * u2e * total_weights[None, None, :, :], axis=2)  # (N1, N2, Nz)
        correction = (
            n_min[None, None, :] * b_min[None, None, :]
            * u1[:, None, 0, :] * u2[None, :, 0, :]
        )
        return integral + hm.hm_consistency * correction

    integral = jnp.sum(u1 * u2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)
    correction = n_min[None, :] * b_min[None, :] * u1[:, 0, :] * u2[:, 0, :]
    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _pair_integral_2pt(halo_model, p1, p2, k, z):
    """
    ∫ dn/dlnM * ⟨p1(k,M,z) p2(k,M,z)⟩_2pt dlnM

    The bias_order=0 (unweighted) pair integral :math:`I_1^2`, built from the
    specialised 1-halo 2-point kernel :func:`_fourier_2pt` instead of a naive
    ``p1.fourier(k,...) * p2.fourier(k,...)`` product -- the same
    generalisation :func:`_pair_integral`'s own docstring anticipates, applied
    here specifically for the SSC number-counts counter-term (see
    :func:`_dPk_response`). Unlike ``_pair_integral``, there is no
    ``outer``/independent-k1-k2 mode: ``_fourier_2pt`` has no such notion (it
    takes a single shared ``k`` for both legs), which is all the counter-term
    ever needs (matching the existing ``i12_uv`` term's own single-k
    convention in ``_dPk_response``).

    Returns
    -------
    array
        Shape (Nk, Nz), singleton dimensions squeezed.
    """
    hm = halo_model
    z_arr = jnp.atleast_1d(z)
    logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
    m = jnp.exp(logm)

    dndlnm = jnp.reshape(
        hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def),
        (len(m), len(z_arr)),
    )
    total_weights = dndlnm * w[:, None]  # (Nm, Nz), bias_order=0 (unweighted)

    ks = jnp.atleast_1d(k)
    u2pt = jnp.reshape(_fourier_2pt(hm, p1, p2, ks, m, z_arr), (len(ks), len(m), len(z_arr)))

    n_min, _, _ = hm._counter_terms(z_arr)
    integral = jnp.sum(u2pt * total_weights[None, :, :], axis=1)  # (Nk, Nz)
    correction = n_min[None, :] * u2pt[:, 0, :]
    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _dPk_response(halo_model, k, z, profile1, profile2=None,
                   needs_counterterm1=None, needs_counterterm2=None):
    """
    Halo-model power-spectrum response :math:`\\partial P_{u,v}(k,z) /
    \\partial\\delta_b`, the effective "trispectrum" entering the
    super-sample covariance (Wagner et al. 2015; Takada & Hu 2013):

    .. math::

        \\frac{\\partial P_{u,v}(k,z)}{\\partial\\delta_b} =
        \\left(\\frac{47}{21} - \\frac{1}{3}\\frac{d\\ln P_{\\rm lin}}{d\\ln k}\\right)
        P_{\\rm lin}(k,z)\\, I_1^1(k \\,|\\, u)\\, I_1^1(k \\,|\\, v)
        + I_1^2(k \\,|\\, u, v)

    where :math:`I_1^1` is the halo model's linearly-biased single-profile
    mass integral (:meth:`~hmfast.halos.HaloModel._I` with
    ``bias_order=1``) and :math:`I_1^2` is the equivalent paired-profile
    integral (:func:`_pair_integral` with ``bias_order=1``, evaluated at
    matching :math:`k` for both legs).

    If ``needs_counterterm1``/``needs_counterterm2`` mark :math:`u`/:math:`v`
    as discrete number-counts observables (e.g. galaxy clustering/HOD) rather
    than a continuous field, the number-counts counter-term is subtracted:

    .. math::

        \\partial P_{u,v}/\\partial\\delta_b \\; \\mathrel{-}= \\;
        (b_u+b_v)\\,P_{u,v}(k,z), \\qquad
        P_{u,v} = P_{\\rm lin}\\,I_1^1(k\\,|\\,u)\\,I_1^1(k\\,|\\,v) + I_1^{2,\\rm 2pt}(k\\,|\\,u,v)

    where :math:`b_u = I_1^1(k\\,|\\,u)` (only included for the legs flagged
    ``True``) and :math:`I_1^{2,\\rm 2pt}` is :func:`_pair_integral_2pt`, the
    bias_order=0 pair integral built from the specialised 1-halo 2-point
    kernel (:func:`_fourier_2pt`) rather than a naive profile product.

    Parameters
    ----------
    halo_model : HaloModel
    k : float or jnp.ndarray
        Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
    z : float or jnp.ndarray
        Redshift grid.
    profile1 : HaloProfile
        First profile (:math:`u`).
    profile2 : HaloProfile or None, default None
        Second profile (:math:`v`). If None, defaults to ``profile1``.
    needs_counterterm1 : bool or None, default None
        Whether ``profile1`` is a discrete number-counts observable requiring
        the counter-term above. ``None`` (the default) auto-detects via
        ``isinstance(profile1, GalaxyHODProfile)`` -- ``True`` for HOD-like
        profiles, ``False`` for everything else (matter, lensing, pressure,
        CIB). An explicit ``True``/``False`` always overrides auto-detection.
    needs_counterterm2 : bool or None, default None
        As ``needs_counterterm1``, but for ``profile2``.

    Returns
    -------
    array
        Response with shape :math:`(N_k, N_z)`, where singleton dimensions
        are squeezed before return.
    """
    hm = halo_model
    profile2 = profile2 if profile2 is not None else profile1
    if needs_counterterm1 is None:
        needs_counterterm1 = isinstance(profile1, GalaxyHODProfile)
    if needs_counterterm2 is None:
        needs_counterterm2 = isinstance(profile2, GalaxyHODProfile)
    k_arr, z_arr = jnp.atleast_1d(k), jnp.atleast_1d(z)
    nk, nz = len(k_arr), len(z_arr)

    i11_u = jnp.reshape(hm._I(profile1, k_arr, z_arr, bias_order=1), (nk, nz))
    i11_v = jnp.reshape(hm._I(profile2, k_arr, z_arr, bias_order=1), (nk, nz))
    i12_uv = jnp.reshape(
        _pair_integral(hm, profile1, profile2, k_arr, k_arr, z_arr, outer=False, bias_order=1),
        (nk, nz),
    )
    pk_lin = jnp.reshape(hm.cosmology.pk(k_arr, z_arr, linear=True), (nk, nz))

    # dlnP/dlnk needs a properly resolved k grid -- a finite difference on a sparse query k_arr is inaccurate.
    k_fine, _ = hm.cosmology._pk_grid()
    pk_fine = jnp.reshape(hm.cosmology.pk(k_fine, z_arr, linear=True), (len(k_fine), nz))
    dlnp_fine = jnp.gradient(jnp.log(pk_fine), jnp.log(k_fine), axis=0)
    dlnp_dlnk = jax.vmap(
        lambda dp: jnp.interp(jnp.log(k_arr), jnp.log(k_fine), dp), in_axes=1, out_axes=1
    )(dlnp_fine)

    response = (47.0 / 21.0 - dlnp_dlnk / 3.0) * pk_lin * i11_u * i11_v + i12_uv

    if needs_counterterm1 or needs_counterterm2:
        i02_uv = jnp.reshape(_pair_integral_2pt(hm, profile1, profile2, k_arr, z_arr), (nk, nz))
        P_uv = pk_lin * i11_u * i11_v + i02_uv
        b_u = i11_u if needs_counterterm1 else 0.0
        b_v = i11_v if needs_counterterm2 else 0.0
        response = response - (b_u + b_v) * P_uv

    return jnp.squeeze(response)


def _triple_integral(halo_model, p_single, p_pair1, p_pair2, k_single, k_pair, z, outer=False):
    """
    ∫ dn/dlnM * b1(M) * p_single(k_single,M) * p_pair1(k_pair,M) * p_pair2(k_pair,M) dlnM

    Triple integral with first-order halo bias, generalising ``_pair_integral``
    to a "1x2" moment: one profile alone at ``k_single``, the other two both
    at ``k_pair``. Needed by the trispectrum's 2-halo "13" term.

    ``outer=False`` (default) pairs ``k_single``/``k_pair`` elementwise,
    sharing a single batch axis. ``outer=True`` broadcasts them into an
    independent (N_single, N_pair) grid, as needed by ``Tk.tk_2h``.

    Returns
    -------
    array
        ``outer=False``: shape (Nk, Nz), singleton dimensions squeezed.
        ``outer=True``: shape (N_single, N_pair, Nz), not squeezed -- this
        branch is only ever used as an internal building block of
        ``Tk.tk_2h``.
    """
    hm = halo_model
    z_arr = jnp.atleast_1d(z)
    logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
    m = jnp.exp(logm)

    dndlnm = jnp.reshape(
        hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def),
        (len(m), len(z_arr)),
    )
    bias_w = jnp.reshape(
        hm.halo_bias.bias(hm.cosmology, m, z_arr, hm.mass_def, order=1),
        (len(m), len(z_arr)),
    )
    total_weights = dndlnm * bias_w * w[:, None]  # (Nm, Nz)

    k_s, k_p = jnp.atleast_1d(k_single), jnp.atleast_1d(k_pair)
    u_s = jnp.reshape(p_single.fourier(hm, k_s, m, z_arr), (len(k_s), len(m), len(z_arr)))
    u_p1 = jnp.reshape(p_pair1.fourier(hm, k_p, m, z_arr), (len(k_p), len(m), len(z_arr)))
    u_p2 = jnp.reshape(p_pair2.fourier(hm, k_p, m, z_arr), (len(k_p), len(m), len(z_arr)))

    n_min, b1_min, _ = hm._counter_terms(z_arr)

    if outer:
        u_se = u_s[:, None, :, :]            # (Ns, 1, Nm, Nz)
        u_pe = (u_p1 * u_p2)[None, :, :, :]  # (1, Np, Nm, Nz)
        integral = jnp.sum(u_se * u_pe * total_weights[None, None, :, :], axis=2)  # (Ns, Np, Nz)
        correction = (
            n_min[None, None, :] * b1_min[None, None, :]
            * u_s[:, None, 0, :] * u_p1[None, :, 0, :] * u_p2[None, :, 0, :]
        )
        return integral + hm.hm_consistency * correction

    integral = jnp.sum(u_s * u_p1 * u_p2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)
    correction = n_min[None, :] * b1_min[None, :] * u_s[:, 0, :] * u_p1[:, 0, :] * u_p2[:, 0, :]
    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _kr_pkr(hm, k, kp, z_arr):
    """
    Shared per-theta ``kr = |k + kp|`` and :math:`P_{\\mathrm{lin}}(k_r)`
    arrays entering every angle-averaged trispectrum kernel
    (``_Pbar_kernel``, ``_P3_kernel``, ``_P4_kernel``). Computing this once
    and reusing it avoids repeating the (relatively expensive) emulator
    evaluation of :math:`P_{\\mathrm{lin}}` across the 2h, 3h and 4h terms.

    ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays, broadcast into an
    (Nk, Nkp) grid of pairs. Returns ``(k_b, kp_b, kr, pkr)`` with ``k_b``
    of shape (Nk,1,1), ``kp_b`` of shape (1,Nkp,1), ``kr`` of shape
    (Nk,Nkp,N_theta), and ``pkr`` of shape (Nk,Nkp,N_theta,Nz).
    """
    k_b, kp_b = k[:, None, None], kp[None, :, None]
    cth = _TRISPEC_COS_THETA[None, None, :]
    kr = _ksum(k_b, kp_b, cth)  # (Nk, Nkp, N_theta)
    nz = len(z_arr)
    pkr = jnp.reshape(
        hm.cosmology.pk(kr.flatten(), z_arr, linear=True),
        kr.shape + (nz,),
    )
    return k_b, kp_b, kr, pkr


# -------------------------
# Halo model bispectrum
# -------------------------

class Bk:
    """
    Halo model bispectrum.

    .. math::

        B(k_1, k_2, k_3, z) = B_{1h} + B_{2h} + B_{3h}

    where the three terms are built from the generalised halo-model mass
    integral

    .. math::

        I_\\mu^\\beta(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\,|\\, M, z)

    where :math:`\\mu` is the number of profiles/wavenumbers in the
    product, :math:`b_\\beta` is the :math:`\\beta`-th order halo bias
    (:math:`b_0 = 1` unweighted, :math:`b_1` linear, :math:`b_2`
    quadratic), and :math:`u_i` are the Fourier-space profiles (first
    moments). See :meth:`bk_1h`, :meth:`bk_2h` and :meth:`bk_3h` for how
    each term is assembled from :math:`I_\\mu^\\beta`.

    .. note::

        This implementation is limited to profiles whose 3-point function
        within a single halo reduces to the product of their (1-point)
        Fourier-space profiles, i.e. :math:`u_{123}(k_1,k_2,k_3 \\,|\\, M) =
        u_1(k_1 \\,|\\, M)\\, u_2(k_2 \\,|\\, M)\\, u_3(k_3 \\,|\\, M)`. This holds
        for matter density and electron pressure/density profiles, but
        not in general for profiles with non-trivial intra-halo occupancy
        statistics such as HOD or CIB.

    Attributes
    ----------
    include_1h : bool
        Whether :meth:`bk_tot` includes the 1-halo term.
    include_2h : bool
        Whether :meth:`bk_tot` includes the 2-halo term.
    include_3h : bool
        Whether :meth:`bk_tot` includes the 3-halo term.
    k_damp : float
        Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for :meth:`bk_1h`'s
        low-k suppression factor.
    """

    def __init__(self, include_1h=True, include_2h=True, include_3h=True, k_damp=0.01):
        """
        Parameters
        ----------
        include_1h : bool, default True
            Whether :meth:`bk_tot` includes the 1-halo term.
        include_2h : bool, default True
            Whether :meth:`bk_tot` includes the 2-halo term.
        include_3h : bool, default True
            Whether :meth:`bk_tot` includes the 3-halo term.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for :meth:`bk_1h`'s
            low-k suppression factor.
        """
        self.include_1h = include_1h
        self.include_2h = include_2h
        self.include_3h = include_3h
        self.k_damp = jnp.asarray(k_damp)

    def _tree_flatten(self):
        return (self.k_damp,), (self.include_1h, self.include_2h, self.include_3h)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (obj.k_damp,) = children
        obj.include_1h, obj.include_2h, obj.include_3h = aux_data
        return obj

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def bk_1h(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        """
        1-halo bispectrum term.


        .. math::

            B_{1h}(k_1, k_2, k_3, z) = I_3^0(k_1, k_2, k_3, z)

        where :math:`I_3^0` is the unweighted (:math:`\\beta=0`) triple mass
        integral :math:`I_\\mu^\\beta` with :math:`\\mu=3`.

        A low-k suppression factor :math:`1 - e^{-(k_{\\min}/k_{\\mathrm{damp}})^2}`
        is applied at the smallest wavenumber of the triplet.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2 : float or jnp.ndarray
            Two triangle sides in :math:`\\mathrm{Mpc}^{-1}`. Must be the same size.
        mu12 : float or jnp.ndarray
            Cosine of the angle between the :math:`k_1` and :math:`k_2`
            vectors. Must be a scalar or an array of shape :math:`(N_k,)`
            matching k1, k2.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.

        Returns
        -------
        array
            1-halo bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        # Validated out here, where k1/k2/mu12 are still concrete.
        _check_bk_inputs(k1, k2, mu12)
        return self._bk_1h(halo_model, k1, k2, mu12, z, profile1, profile2, profile3)

    @jax.jit
    def _bk_1h(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        z_arr = jnp.atleast_1d(z)
        logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
        m = jnp.exp(logm)

        dndlnm = jnp.reshape(
            hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def),
            (len(m), len(z_arr)),
        )
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        k1a, k2a = jnp.asarray(k1), jnp.asarray(k2)
        k3 = _ksum(k1a, k2a, jnp.asarray(mu12))

        k1s, k2s, k3s = jnp.atleast_1d(k1a), jnp.atleast_1d(k2a), jnp.atleast_1d(k3)
        u1 = jnp.reshape(profile1.fourier(hm, k1s, m, z_arr), (len(k1s), len(m), len(z_arr)))
        u2 = jnp.reshape(profile2.fourier(hm, k2s, m, z_arr), (len(k2s), len(m), len(z_arr)))
        u3 = jnp.reshape(profile3.fourier(hm, k3s, m, z_arr), (len(k3s), len(m), len(z_arr)))

        triple = u1 * u2 * u3  # (1, Nm, Nz)
        bk1h = jnp.sum(triple * total_weights[None, :, :], axis=1)  # (1, Nz)

        n_min, _, _ = hm._counter_terms(z_arr)
        correction = n_min[None, :] * u1[:, 0, :] * u2[:, 0, :] * u3[:, 0, :]
        bk1h = bk1h + hm.hm_consistency * correction

        k_min = jnp.minimum(jnp.minimum(k1a, k2a), k3)
        mask = self.k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k_min / jnp.where(mask, self.k_damp, 1.0))**2), 1.0)

        # Reshape damping to (N_k, 1) so it broadcasts correctly over (N_k, N_z)
        damping_bc = jnp.reshape(damping, jnp.shape(damping) + (1,))
        return jnp.squeeze(bk1h * damping_bc)

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def bk_2h(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        """
        2-halo bispectrum term.

        .. math::

            B_{2h}(k_1, k_2, k_3, z) = P_{\\mathrm{lin}}(k_1)\\, I_1^1(k_1)\\, I_2^1(k_2, k_3)
            \\;+\\; 2\\,\\mathrm{cyc.}

        where :math:`I_1^1` and :math:`I_2^1` are the linearly-biased
        (:math:`\\beta=1`) single- and pair-profile mass integrals
        :math:`I_\\mu^\\beta`, and
        "+ 2 cyc" denotes the sum over the two cyclic permutations of
        :math:`(1,2,3)`.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2 : float or jnp.ndarray
            Two triangle sides in :math:`\\mathrm{Mpc}^{-1}`. Must be the same size.
        mu12 : float or jnp.ndarray
            Cosine of the angle between the :math:`k_1` and :math:`k_2`
            vectors. Must be a scalar or an array of shape :math:`(N_k,)`
            matching k1, k2.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.

        Returns
        -------
        array
            2-halo bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        # Validated out here, where k1/k2/mu12 are still concrete.
        _check_bk_inputs(k1, k2, mu12)
        return self._bk_2h(halo_model, k1, k2, mu12, z, profile1, profile2, profile3)

    @jax.jit
    def _bk_2h(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        z_arr = jnp.atleast_1d(z)

        k1a, k2a = jnp.asarray(k1), jnp.asarray(k2)
        k3 = _ksum(k1a, k2a, jnp.asarray(mu12))

        I1 = hm._I(profile1, k1, z, bias_order=1)
        I2 = hm._I(profile2, k2, z, bias_order=1)
        I3 = hm._I(profile3, k3, z, bias_order=1)

        J23 = _pair_integral(hm, profile2, profile3, k2, k3, z)
        J13 = _pair_integral(hm, profile1, profile3, k1, k3, z)
        J12 = _pair_integral(hm, profile1, profile2, k1, k2, z)

        P1 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k1), z_arr, linear=True))
        P2 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k2), z_arr, linear=True))
        P3 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k3), z_arr, linear=True))

        return jnp.squeeze(P1 * I1 * J23 + P2 * I2 * J13 + P3 * I3 * J12)

    # ------------------------------------------------------------------
    # 3-halo term
    # ------------------------------------------------------------------

    def bk_3h(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        """
        3-halo bispectrum term.

        .. math::

            \\begin{aligned}
                B_{3h}(k_1, k_2, k_3, z) &= B^{\\mathrm{PT}}(k_1, k_2, k_3)\\,
                I_1^1(k_1)\\, I_1^1(k_2)\\, I_1^1(k_3) \\\\
                &\\quad +\\; \\Big[\\, I_1^2(k_1)\\, I_1^1(k_2)\\, I_1^1(k_3)\\,
                P_{\\mathrm{lin}}(k_2)\\, P_{\\mathrm{lin}}(k_3) \\;+\\; 2\\,\\mathrm{cyc.} \\,\\Big]
            \\end{aligned}

        where :math:`I_1^\\beta(k_i)` is the single-profile mass integral
        :math:`I_\\mu^\\beta` (with
        :math:`\\mu=1`; :math:`\\beta=1` linear or :math:`\\beta=2`
        quadratic bias), "+ 2 cyc" denotes the sum over the two cyclic
        permutations of :math:`(1,2,3)`, and

        .. math::

            B^{\\mathrm{PT}}(k_1, k_2, k_3) = 2\\, F_2(k_1, k_2)\\, P_{\\mathrm{lin}}(k_1)\\,
            P_{\\mathrm{lin}}(k_2) \\;+\\; 2\\,\\mathrm{cyc.}

        is the tree-level SPT bispectrum and :math:`F_2` is the standard second-order SPT kernel.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2 : float or jnp.ndarray
            Two triangle sides in :math:`\\mathrm{Mpc}^{-1}`. Must be the same size.
        mu12 : float or jnp.ndarray
            Cosine of the angle between the :math:`k_1` and :math:`k_2`
            vectors. Must be a scalar or an array of shape :math:`(N_k,)`
            matching k1, k2.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.

        Returns
        -------
        array
            3-halo bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        # Validated out here, where k1/k2/mu12 are still concrete.
        _check_bk_inputs(k1, k2, mu12)
        return self._bk_3h(halo_model, k1, k2, mu12, z, profile1, profile2, profile3)

    @jax.jit
    def _bk_3h(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        z_arr = jnp.atleast_1d(z)

        k1a, k2a = jnp.asarray(k1), jnp.asarray(k2)
        mu12 = jnp.asarray(mu12)
        k3a = _ksum(k1a, k2a, mu12)

        I1_b1 = hm._I(profile1, k1, z, bias_order=1)
        I2_b1 = hm._I(profile2, k2, z, bias_order=1)
        I3_b1 = hm._I(profile3, k3a, z, bias_order=1)
        I1_b2 = hm._I(profile1, k1, z, bias_order=2)
        I2_b2 = hm._I(profile2, k2, z, bias_order=2)
        I3_b2 = hm._I(profile3, k3a, z, bias_order=2)

        P1 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k1), z_arr, linear=True))
        P2 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k2), z_arr, linear=True))
        P3 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k3a), z_arr, linear=True))

        # Tree-level SPT bispectrum with correct cosine convention:
        # mu_ij = (k_k^2 - k_i^2 - k_j^2) / (2 k_i k_j)  [opposite-side law]
        mu23 = _mu(k2a, k3a, k1a)
        mu31 = _mu(k3a, k1a, k2a)
        B_tree = (
            2.0 * _F2(k1a, k2a, mu12) * P1 * P2
            + 2.0 * _F2(k2a, k3a, mu23) * P2 * P3
            + 2.0 * _F2(k3a, k1a, mu31) * P3 * P1
        )

        tree_term = B_tree * I1_b1 * I2_b1 * I3_b1

        # Quadratic-bias corrections
        b2_term = (
            I1_b2 * I2_b1 * I3_b1 * P2 * P3
            + I1_b1 * I2_b2 * I3_b1 * P1 * P3
            + I1_b1 * I2_b1 * I3_b2 * P1 * P2
        )

        return jnp.squeeze(tree_term + b2_term)

    # ------------------------------------------------------------------
    # Combined 1-halo + 2-halo + 3-halo term
    # ------------------------------------------------------------------

    def bk_tot(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        """
        Combine the 1-halo, 2-halo and 3-halo terms into the total halo-model bispectrum.

        .. math::

            B(k_1, k_2, k_3, z) = B_{1h} + B_{2h} + B_{3h}

        A term excluded via :attr:`include_1h`/:attr:`include_2h`/:attr:`include_3h`
        is simply left out of the sum.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2 : float or jnp.ndarray
            Two triangle sides in :math:`\\mathrm{Mpc}^{-1}`. Must be the same size.
        mu12 : float or jnp.ndarray
            Cosine of the angle between the :math:`k_1` and :math:`k_2`
            vectors. Must be a scalar or an array of shape :math:`(N_k,)`
            matching k1, k2.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.

        Returns
        -------
        array
            Combined bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        # Validated out here, where k1/k2/mu12 are still concrete.
        _check_bk_inputs(k1, k2, mu12)
        return self._bk_tot(halo_model, k1, k2, mu12, z, profile1, profile2, profile3)

    @jax.jit
    def _bk_tot(self, halo_model, k1, k2, mu12, z, profile1, profile2=None, profile3=None):
        b1h = self._bk_1h(halo_model, k1, k2, mu12, z, profile1, profile2, profile3) if self.include_1h else 0.0
        b2h = self._bk_2h(halo_model, k1, k2, mu12, z, profile1, profile2, profile3) if self.include_2h else 0.0
        b3h = self._bk_3h(halo_model, k1, k2, mu12, z, profile1, profile2, profile3) if self.include_3h else 0.0
        return b1h + b2h + b3h


jax.tree_util.register_pytree_node(
    Bk,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: Bk._tree_unflatten(aux_data, children)
)


# -------------------------
# Halo model trispectrum
# -------------------------

class Tk:
    """
    Isotropized halo model trispectrum in the parallelogram ("covariance")
    configuration.

    .. math::

        T(k_u, k_v, z) = T_{1h} + T_{2h} + T_{3h} + T_{4h}

    with :math:`k_1 = -k_2 = k_u` and :math:`k_3 = -k_4 = k_v`,
    angle-averaged over the angle between the two diagonals, where the
    four terms are built from the generalised halo-model mass integral

    .. math::

        I_\\mu^\\beta(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\,|\\, M, z)

    where :math:`\\mu` is the number of
    profiles/wavenumbers in the product, :math:`b_\\beta` is the
    :math:`\\beta`-th order halo bias (:math:`b_0 = 1` unweighted,
    :math:`b_1` linear, :math:`b_2` quadratic), and :math:`u_i` are the
    Fourier-space profiles (first moments). See :meth:`tk_1h`, :meth:`tk_2h`,
    :meth:`tk_3h` and :meth:`tk_4h` for how each term is assembled from
    :math:`I_\\mu^\\beta`.

    .. note::

        This implementation is limited to profiles whose 3- and 4-point
        functions within a single halo reduce to products of their
        (1-point) Fourier-space profiles, i.e.
        :math:`u_{1234}(k_1,k_2,k_3,k_4 \\,|\\, M) = u_1(k_1 \\,|\\, M)\\,
        u_2(k_2 \\,|\\, M)\\, u_3(k_3 \\,|\\, M)\\, u_4(k_4 \\,|\\, M)` (and
        similarly for the 3-point sub-clumps entering the 2-halo "13" term).
        This holds for matter density and electron pressure/density profiles,
        but not in general for profiles with non-trivial
        intra-halo occupancy statistics such as HOD or CIB.

    Attributes
    ----------
    include_1h : bool
        Whether :meth:`tk_tot` includes the 1-halo term.
    include_2h : bool
        Whether :meth:`tk_tot` includes the 2-halo term.
    include_3h : bool
        Whether :meth:`tk_tot` includes the 3-halo term.
    include_4h : bool
        Whether :meth:`tk_tot` includes the 4-halo term.
    """

    def __init__(self, include_1h=True, include_2h=True, include_3h=True, include_4h=True):
        """
        Parameters
        ----------
        include_1h : bool, default True
            Whether :meth:`tk_tot` includes the 1-halo term.
        include_2h : bool, default True
            Whether :meth:`tk_tot` includes the 2-halo term.
        include_3h : bool, default True
            Whether :meth:`tk_tot` includes the 3-halo term.
        include_4h : bool, default True
            Whether :meth:`tk_tot` includes the 4-halo term.
        """
        self.include_1h = include_1h
        self.include_2h = include_2h
        self.include_3h = include_3h
        self.include_4h = include_4h

    def _tree_flatten(self):
        return (), (self.include_1h, self.include_2h, self.include_3h, self.include_4h)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.include_1h, obj.include_2h, obj.include_3h, obj.include_4h = aux_data
        return obj

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def tk_1h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        1-halo trispectrum term.

        .. math::

            T_{1h}(k_u, k_v, z) = I_4^0\\!\\left(k_u \\,|\\, u_1,\\, k_u \\,|\\, u_2,
            \\, k_v \\,|\\, v_1,\\, k_v \\,|\\, v_2\\right)

        where :math:`I_4^0` is the unweighted (:math:`\\beta=0`) quadruple
        mass integral :math:`I_\\mu^\\beta` with :math:`\\mu=4`.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or jnp.ndarray
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length; the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u`` (the :math:`u_1` leg).
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u`` (:math:`u_2`) and the two profiles at
            ``k_v`` (:math:`v_1, v_2`), respectively. If None, profile2 and
            profile3 default to profile1, and profile4 defaults to
            (the resolved) profile2.

        Returns
        -------
        array
            1-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)
        logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
        m = jnp.exp(logm)

        dndlnm = jnp.reshape(
            hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def),
            (len(m), len(z_arr)),
        )
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        u1 = jnp.reshape(profile1.fourier(hm, k_us, m, z_arr), (len(k_us), len(m), len(z_arr)))
        u2 = jnp.reshape(profile2.fourier(hm, k_us, m, z_arr), (len(k_us), len(m), len(z_arr)))
        u3 = jnp.reshape(profile3.fourier(hm, k_vs, m, z_arr), (len(k_vs), len(m), len(z_arr)))
        u4 = jnp.reshape(profile4.fourier(hm, k_vs, m, z_arr), (len(k_vs), len(m), len(z_arr)))

        u12 = (u1 * u2)[:, None, :, :]  # (Nu, 1, Nm, Nz)
        u34 = (u3 * u4)[None, :, :, :]  # (1, Nv, Nm, Nz)
        tk1h = jnp.sum(u12 * u34 * total_weights[None, None, :, :], axis=2)  # (Nu, Nv, Nz)

        n_min, _, _ = hm._counter_terms(z_arr)
        correction = (
            n_min[None, None, :]
            * u1[:, None, 0, :] * u2[:, None, 0, :] * u3[None, :, 0, :] * u4[None, :, 0, :]
        )
        tk1h = tk1h + hm.hm_consistency * correction

        return jnp.squeeze(tk1h)

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def _Pbar_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged isotropized linear power spectrum
        :math:`\\bar P(k,k') = \\langle P_{\\mathrm{lin}}(|{\\bf k}+{\\bf k}'|)\\rangle_\\theta`
        entering the "22" diagram of the 2-halo trispectrum term.

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays. Returns an
        (Nk, Nkp, Nz) array, not squeezed -- this is only ever used as an
        internal building block of ``tk_2h``.
        """
        _, _, _, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, None, :, None]
        return jnp.sum(pkr * wgt, axis=2)

    @jax.jit
    def tk_2h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        2-halo trispectrum term (sum of the "22" and "13" diagrams).

        .. math::

            T_{2h}^{(22)}(k_u,k_v,z) = \\bar P(k_u,k_v)\\, \\Big[\\,
                I_2^1\\!\\left(k_u \\,|\\, u_1, k_v \\,|\\, v_1\\right)\\,
                I_2^1\\!\\left(k_u \\,|\\, u_2, k_v \\,|\\, v_2\\right)
                + (v_1 \\leftrightarrow v_2) \\Big]

        where :math:`I_2^1` is the linearly-biased (:math:`\\beta=1`)
        pair-profile mass integral :math:`I_\\mu^\\beta`, and
        :math:`\\bar P` is the relative-angle average
        of :math:`P_{\\mathrm{lin}}(|{\\bf k}_u+{\\bf k}_v|)`.

        .. math::

            \\begin{aligned}
                T_{2h}^{(13)}(k_u,k_v,z) &= P_{\\mathrm{lin}}(k_u)\\, \\Big[
                    I_1^1\\!\\left(k_u \\,|\\, u_1\\right)\\,
                    I_3^1\\!\\left(k_u \\,|\\, u_2, k_v \\,|\\, v_1, k_v \\,|\\, v_2\\right)
                    + (u_1 \\leftrightarrow u_2) \\Big] \\\\
                &\\quad + P_{\\mathrm{lin}}(k_v)\\, \\Big[
                    I_1^1\\!\\left(k_v \\,|\\, v_1\\right)\\,
                    I_3^1\\!\\left(k_v \\,|\\, v_2, k_u \\,|\\, u_1, k_u \\,|\\, u_2\\right)
                    + (v_1 \\leftrightarrow v_2) \\Big]
            \\end{aligned}

        where :math:`I_1^1`, :math:`I_3^1` are the linearly-biased (:math:`\\beta=1`)
        single- and triple-profile mass integrals :math:`I_\\mu^\\beta`.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or jnp.ndarray
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            2-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)
        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        nu, nv, nz = len(k_us), len(k_vs), len(z_arr)

        # "22" diagram: two halos, each hosting a pair of legs.
        Pbar = self._Pbar_kernel(hm, k_us, k_vs, z_arr)  # (Nu, Nv, Nz)
        pair_a = (
            _pair_integral(hm, profile1, profile3, k_u, k_v, z, outer=True)
            * _pair_integral(hm, profile2, profile4, k_u, k_v, z, outer=True)
        )
        pair_b = (
            _pair_integral(hm, profile1, profile4, k_u, k_v, z, outer=True)
            * _pair_integral(hm, profile3, profile2, k_u, k_v, z, outer=True)
        )
        tk_22 = Pbar * (pair_a + pair_b)  # (Nu, Nv, Nz)

        # "13" diagram: one leg alone in a halo, the other three together.
        P_u = jnp.reshape(hm.cosmology.pk(k_us, z_arr, linear=True), (nu, 1, nz))
        P_v = jnp.reshape(hm.cosmology.pk(k_vs, z_arr, linear=True), (1, nv, nz))

        I1u = jnp.reshape(hm._I(profile1, k_u, z, bias_order=1), (nu, 1, nz))
        I2u = jnp.reshape(hm._I(profile2, k_u, z, bias_order=1), (nu, 1, nz))
        I3v = jnp.reshape(hm._I(profile3, k_v, z, bias_order=1), (1, nv, nz))
        I4v = jnp.reshape(hm._I(profile4, k_v, z, bias_order=1), (1, nv, nz))

        K_u2 = _triple_integral(hm, profile2, profile3, profile4, k_u, k_v, z, outer=True)
        K_u1 = _triple_integral(hm, profile1, profile3, profile4, k_u, k_v, z, outer=True)
        K_v4 = jnp.swapaxes(
            _triple_integral(hm, profile4, profile1, profile2, k_v, k_u, z, outer=True), 0, 1
        )
        K_v3 = jnp.swapaxes(
            _triple_integral(hm, profile3, profile1, profile2, k_v, k_u, z, outer=True), 0, 1
        )

        tk_13 = P_u * (I1u * K_u2 + I2u * K_u1) + P_v * (I3v * K_v4 + I4v * K_v3)  # (Nu, Nv, Nz)

        return jnp.squeeze(tk_22 + tk_13)

    # ------------------------------------------------------------------
    # 3-halo term
    # ------------------------------------------------------------------

    def _P3_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged :math:`P_3(k,k') = \\langle P_{\\mathrm{lin}}(|{\\bf
        k}+{\\bf k}'|)\\, F_2(k,k',\\theta)\\rangle_\\theta` entering the
        tree-level bispectrum-type kernel of the 3-halo trispectrum term,
        built from the SPT kernels evaluated between leg ``k`` and the
        vector sum ``kr = |k + kp|``.

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays. Returns an
        (Nk, Nkp, Nz) array, not squeezed -- this is only ever used as an
        internal building block of ``tk_3h``.
        """
        k_b, kp_b, kr, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, None, :]
        f2_kkp = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(k_b, kr, _mu(k_b, kr, kp_b)))
        return jnp.sum(pkr * (f2_kkp * wgt)[..., None], axis=2)

    def _Bpt_kernel(self, hm, k, kp, z_arr):
        """
        Tree-level, angle-averaged bispectrum-type kernel entering the
        3-halo trispectrum term, following Eq. 30 of Takada & Hu (2013):

        .. math::

            B^{\\mathrm{PT}}(k,k') = \\frac{12}{7} P_{\\mathrm{lin}}(k)
            P_{\\mathrm{lin}}(k') + 2\\big[P_{\\mathrm{lin}}(k)\\, P_3(k,k')
            + P_{\\mathrm{lin}}(k')\\, P_3(k',k)\\big]

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays. Returns an
        (Nk, Nkp, Nz) array, not squeezed -- this is only ever used as an
        internal building block of ``tk_3h``.
        """
        nk, nkp, nz = len(k), len(kp), len(z_arr)
        P_k = jnp.reshape(hm.cosmology.pk(k, z_arr, linear=True), (nk, 1, nz))
        P_kp = jnp.reshape(hm.cosmology.pk(kp, z_arr, linear=True), (1, nkp, nz))
        P3_kkp = self._P3_kernel(hm, k, kp, z_arr)
        P3_kpk = jnp.swapaxes(self._P3_kernel(hm, kp, k, z_arr), 0, 1)
        return 12.0 / 7.0 * P_k * P_kp + 2.0 * (P_k * P3_kkp + P_kp * P3_kpk)

    @jax.jit
    def tk_3h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        3-halo trispectrum term.

        .. math::

            T_{3h}(k_u,k_v,z) = B^{\\mathrm{PT}}(k_u,k_v)\\, \\Big[\\,
                I_1^1\\!\\left(k_u \\,|\\, u_1\\right)\\, I_1^1\\!\\left(k_v \\,|\\, v_1\\right)\\,
                I_2^1\\!\\left(k_u \\,|\\, u_2, k_v \\,|\\, v_2\\right)
                + (v_1 \\leftrightarrow v_2) + (u_1 \\leftrightarrow u_2) \\Big]

        where :math:`B^{\\mathrm{PT}}`
        is the tree-level bispectrum-type kernel, and :math:`I_1^1`,
        :math:`I_2^1` are the linearly-biased (:math:`\\beta=1`) single- and
        pair-profile mass integrals :math:`I_\\mu^\\beta`.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or jnp.ndarray
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            3-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)
        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        nu, nv, nz = len(k_us), len(k_vs), len(z_arr)

        Bpt = self._Bpt_kernel(hm, k_us, k_vs, z_arr)  # (Nu, Nv, Nz)

        I1u = jnp.reshape(hm._I(profile1, k_u, z, bias_order=1), (nu, 1, nz))
        I2u = jnp.reshape(hm._I(profile2, k_u, z, bias_order=1), (nu, 1, nz))
        I3v = jnp.reshape(hm._I(profile3, k_v, z, bias_order=1), (1, nv, nz))
        I4v = jnp.reshape(hm._I(profile4, k_v, z, bias_order=1), (1, nv, nz))

        J24 = _pair_integral(hm, profile2, profile4, k_u, k_v, z, outer=True)
        J32 = _pair_integral(hm, profile3, profile2, k_u, k_v, z, outer=True)
        J14 = _pair_integral(hm, profile1, profile4, k_u, k_v, z, outer=True)
        J31 = _pair_integral(hm, profile3, profile1, k_u, k_v, z, outer=True)

        tk3h = Bpt * (
            I1u * I3v * J24
            + I1u * I4v * J32
            + I3v * I2u * J14
            + I4v * I2u * J31
        )

        return jnp.squeeze(tk3h)

    # ------------------------------------------------------------------
    # 4-halo term
    # ------------------------------------------------------------------

    def _P4_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged P4A(k,kp), P4X(k,kp) kernels ("1122" diagram) entering
        the tree-level 4h trispectrum, built from the SPT kernels evaluated
        between leg ``k`` and the vector sum ``kr = |k + kp|``, following
        Eq. 30 of Takada & Hu (2013).

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays — the two leg
        magnitudes for this ordering. Returns ``(P4A, P4X)``, each of shape
        (Nk, Nkp, Nz), not squeezed -- this is only ever used as an
        internal building block of ``tk_4h``.
        """
        k_b, kp_b, kr, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, None, :]

        # F2 between leg k (or kp) and the (negative of the) internal
        # propagator -kr, whose opposite side is the other leg (kp or k).
        f2_kkp = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(k_b, kr, _mu(k_b, kr, kp_b)))
        f2_kpk = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(kp_b, kr, _mu(kp_b, kr, k_b)))

        P4A = jnp.sum(pkr * (f2_kkp ** 2 * wgt)[..., None], axis=2)
        P4X = jnp.sum(pkr * (f2_kkp * f2_kpk * wgt)[..., None], axis=2)
        return P4A, P4X

    @jax.jit
    def tk_4h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        4-halo (tree-level) trispectrum term.

        .. math::

            T_{4h}(k_u, k_v, z) = T^{\\mathrm{PT}}(k_u,-k_u,k_v,-k_v)\\,
            I_1^1\\!\\left(k_u \\,|\\, u_1\\right)\\, I_1^1\\!\\left(k_u \\,|\\, u_2\\right)\\,
            I_1^1\\!\\left(k_v \\,|\\, v_1\\right)\\, I_1^1\\!\\left(k_v \\,|\\, v_2\\right)

        where :math:`I_1^1` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^\\beta` and the
        tree-level trispectrum :math:`T^{\\mathrm{PT}}` for the
        parallelogram configuration is
        angle-averaged over the relative
        orientation of the :math:`k_u` and :math:`k_v` pairs.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or jnp.ndarray
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            4-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)

        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        nu, nv, nz = len(k_us), len(k_vs), len(z_arr)

        P_u = jnp.reshape(hm.cosmology.pk(k_us, z_arr, linear=True), (nu, 1, nz))
        P_v = jnp.reshape(hm.cosmology.pk(k_vs, z_arr, linear=True), (1, nv, nz))

        # "1113" diagram (tree-level F3-type kernel), symmetrized k_u <-> k_v
        X_uv = _X3(k_us, k_vs)[:, :, None]                            # (Nu, Nv, 1)
        X_vu = jnp.swapaxes(_X3(k_vs, k_us), 0, 1)[:, :, None]        # (Nu, Nv, 1)
        t1113 = 4.0 / 9.0 * P_u ** 2 * P_v * X_uv + 4.0 / 9.0 * P_v ** 2 * P_u * X_vu

        # "1122" diagram (two F2-type vertices), symmetrized k_u <-> k_v
        P4A_uv, P4X_uv = self._P4_kernel(hm, k_us, k_vs, z_arr)  # (Nu, Nv, Nz)
        P4A_vu, P4X_vu = self._P4_kernel(hm, k_vs, k_us, z_arr)  # (Nv, Nu, Nz)
        P4A_vu, P4X_vu = jnp.swapaxes(P4A_vu, 0, 1), jnp.swapaxes(P4X_vu, 0, 1)
        t1122 = (
            8.0 * (P_u ** 2 * P4A_uv + P_u * P_v * P4X_uv)
            + 8.0 * (P_v ** 2 * P4A_vu + P_v * P_u * P4X_vu)
        )

        T_pt = t1113 + t1122  # (Nu, Nv, Nz)

        I1 = jnp.reshape(hm._I(profile1, k_u, z, bias_order=1), (nu, 1, nz))
        I2 = jnp.reshape(hm._I(profile2, k_u, z, bias_order=1), (nu, 1, nz))
        I3 = jnp.reshape(hm._I(profile3, k_v, z, bias_order=1), (1, nv, nz))
        I4 = jnp.reshape(hm._I(profile4, k_v, z, bias_order=1), (1, nv, nz))

        return jnp.squeeze(T_pt * I1 * I2 * I3 * I4)

    # ------------------------------------------------------------------
    # Combined 1-halo + 2-halo + 3-halo + 4-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def tk_tot(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        Combine the 1-halo, 2-halo, 3-halo and 4-halo terms into the total
        halo-model trispectrum.

        .. math::

            T(k_u, k_v, z) = T_{1h} + T_{2h} + T_{3h} + T_{4h}

        A term excluded via :attr:`include_1h`/:attr:`include_2h`/
        :attr:`include_3h`/:attr:`include_4h` is simply left out of the sum.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or jnp.ndarray
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            Combined trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        t1h = self.tk_1h(halo_model, k_u, k_v, z, profile1, profile2, profile3, profile4) if self.include_1h else 0.0
        t2h = self.tk_2h(halo_model, k_u, k_v, z, profile1, profile2, profile3, profile4) if self.include_2h else 0.0
        t3h = self.tk_3h(halo_model, k_u, k_v, z, profile1, profile2, profile3, profile4) if self.include_3h else 0.0
        t4h = self.tk_4h(halo_model, k_u, k_v, z, profile1, profile2, profile3, profile4) if self.include_4h else 0.0
        return t1h + t2h + t3h + t4h

    # ------------------------------------------------------------------
    # Connected (non-Gaussian) angular power spectrum covariance
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(9,))
    def covariance_cng(self, halo_model, tracer1, tracer2, tracer3, tracer4, l1, l2, z_range, n_z, f_sky=1.0):
        """
        Connected (non-Gaussian) covariance between two Limber-projected
        angular power spectra :math:`C_{\\ell_1}^{12}` and
        :math:`C_{\\ell_2}^{34}`, sourced by the halo-model trispectrum.

        Under the Limber approximation, the two independent line-of-sight
        integrals of the exact covariance collapse to a single integral
        over comoving distance, since both multipoles map to a wavenumber
        at the *same* :math:`\\chi`:

        .. math::

            {\\rm Cov}(\\ell_1,\\ell_2) = \\frac{1}{4\\pi f_{\\rm sky}}
            \\int dz\\, \\frac{d\\chi/dz}{\\chi^6}\\,
            W_1(z)\\, W_2(z)\\, W_3(z)\\, W_4(z)\\,
            T\\!\\left(k_1, k_2, z\\right)

        with :math:`k_1 = (\\ell_1 + 1/2)/\\chi(z)`,
        :math:`k_2 = (\\ell_2 + 1/2)/\\chi(z)`, :math:`W_i` the tracer
        kernels, and :math:`T = T_{1h} + T_{2h} + T_{3h} + T_{4h}` the full
        halo-model trispectrum (see :meth:`tk_1h`, :meth:`tk_2h`,
        :meth:`tk_3h`, :meth:`tk_4h`).

        Every ``der_bessel=0`` term of a tracer's kernel (density,
        magnification bias, intrinsic alignment, ...) enters :math:`W_i(z)`
        directly. A ``der_bessel=2`` (RSD) term instead makes :math:`W_1(z)\\,
        W_2(z)` (or :math:`W_3(z)\\,W_4(z)`) depend on :math:`\\ell_1` (or
        :math:`\\ell_2`) too, via the same extended-Limber correction
        (Chisari et al. 2019 Sec. 2.4.1) used by
        :meth:`~hmfast.stats.pk.Pk.cl_hm`
        (see :func:`_extended_limber_grid_for_pair`,
        :func:`_kernel_pair_effective`).

        Parameters
        ----------
        halo_model : HaloModel
        tracer1 : Tracer
            First tracer of the :math:`C_{\\ell_1}` pair.
        tracer2 : Tracer or None
            Second tracer of the :math:`C_{\\ell_1}` pair. If None,
            defaults to ``tracer1``.
        tracer3 : Tracer or None
            First tracer of the :math:`C_{\\ell_2}` pair. If None,
            defaults to ``tracer1``.
        tracer4 : Tracer or None
            Second tracer of the :math:`C_{\\ell_2}` pair. If None,
            defaults to (the resolved) ``tracer3``.
        l1, l2 : float or jnp.ndarray
            Multipole grids for the first and second angular power
            spectrum, respectively. Need not be the same length; the two
            are broadcast into an :math:`(N_{\\ell_1}, N_{\\ell_2})` grid.
        z_range : tuple
            ``(z_min, z_max)`` spanning the Gauss-Legendre redshift integration grid.
        n_z : int
            Number of redshift-integration nodes (static: changing it triggers
            recompilation; sweeping ``z_range`` alone does not).
        f_sky : float, default 1.0
            Observed sky fraction.

        Returns
        -------
        array
            Connected covariance with shape :math:`(N_{\\ell_1},
            N_{\\ell_2})`, where singleton dimensions are squeezed before
            return.
        """
        hm = halo_model
        logz, z_gl_w = gauss_legendre_nodes_weights(jnp.log(z_range[0]), jnp.log(z_range[1]), n_z)
        z = jnp.exp(logz)
        z_gl_w = z_gl_w * z  # Gauss-Legendre in ln(z); z spans orders of magnitude
        tracer2 = tracer2 if tracer2 is not None else tracer1
        tracer3 = tracer3 if tracer3 is not None else tracer1
        tracer4 = tracer4 if tracer4 is not None else tracer3

        def get_cov_slice(z_i):
            chi = hm.cosmology.angular_diameter_distance(z_i) * (1.0 + z_i)
            k1 = (l1 + 0.5) / chi
            k2 = (l2 + 0.5) / chi

            # Respects self.include_1h/2h/3h/4h, so a partial Tk sources a partial covariance.
            T = self.tk_tot(hm, k1, k2, z_i, tracer1.profile, tracer2.profile, tracer3.profile, tracer4.profile)

            z_lp1, lp1h1, lp3h1, sqell1 = _extended_limber_grid_for_pair(
                hm, tracer1, tracer2, tracer1.profile, tracer2.profile, z_i, l1, chi
            )
            kernel12 = _kernel_pair_effective(
                hm.cosmology, tracer1, tracer2, z_i, l1, z_lp1, lp1h1, lp3h1, sqell1
            )  # (N_l1,)

            z_lp2, lp1h2, lp3h2, sqell2 = _extended_limber_grid_for_pair(
                hm, tracer3, tracer4, tracer3.profile, tracer4.profile, z_i, l2, chi
            )
            kernel34 = _kernel_pair_effective(
                hm.cosmology, tracer3, tracer4, z_i, l2, z_lp2, lp1h2, lp3h2, sqell2
            )  # (N_l2,)

            kernels = kernel12[:, None] * kernel34[None, :]  # (N_l1, N_l2)
            weight = jnp.squeeze(hm.cosmology.comoving_volume_element(z_i) / chi ** 8)

            return T * (kernels * weight)

        integrand = jax.vmap(get_cov_slice)(z)  # (Nz, N_l1, N_l2)
        cov = jnp.sum(integrand * z_gl_w[:, None, None], axis=0) / (4.0 * jnp.pi * f_sky)

        return jnp.squeeze(cov)

    # ------------------------------------------------------------------
    # Super-sample covariance
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(9,), static_argnames=("needs_counterterm1", "needs_counterterm2", "needs_counterterm3", "needs_counterterm4"))
    def covariance_ssc(self, halo_model, tracer1, tracer2, tracer3, tracer4, l1, l2, z_range, n_z, f_sky=1.0,
                        needs_counterterm1=None, needs_counterterm2=None,
                        needs_counterterm3=None, needs_counterterm4=None):
        """
        Super-sample covariance (SSC) between two Limber-projected angular
        power spectra :math:`C_{\\ell_1}^{12}` and :math:`C_{\\ell_2}^{34}`.

        Sourced by long-wavelength density modes larger than the survey
        footprint, which are not measured directly but instead shift the
        mean background density of the observed volume -- rescaling every
        halo-model quantity inside it. Structurally a sibling of
        :meth:`covariance_cng`: the same Limber-collapsed single
        line-of-sight integral and tracer-kernel product, but with the real
        trispectrum replaced by a factorised power-spectrum *response*
        product, and one extra ingredient, the survey-footprint variance of
        the background mode:

        .. math::

            {\\rm Cov}(\\ell_1,\\ell_2) =
            \\int dz\\, \\frac{d\\chi/dz}{\\chi^4}\\,
            W_1(z)\\, W_2(z)\\, W_3(z)\\, W_4(z)\\,
            \\sigma_B^2(z)\\,
            \\frac{\\partial P_{12}(k_1,z)}{\\partial\\delta_b}\\,
            \\frac{\\partial P_{34}(k_2,z)}{\\partial\\delta_b}

        with :math:`k_1 = (\\ell_1 + 1/2)/\\chi(z)`,
        :math:`k_2 = (\\ell_2 + 1/2)/\\chi(z)`, :math:`W_i` the tracer
        kernels, and :math:`\\sigma_B^2(z)` the disc-footprint variance (see
        :meth:`~hmfast.cosmology.Cosmology.sigma2_b_disc`).

        As in :meth:`covariance_cng`, a ``der_bessel=2`` (RSD) kernel term
        makes :math:`W_1(z)\\,W_2(z)` (or :math:`W_3(z)\\,W_4(z)`)
        :math:`\\ell`-dependent via the extended-Limber correction (see
        :func:`_extended_limber_grid_for_pair`, :func:`_kernel_pair_effective`);
        every ``der_bessel=0`` term (density, magnification bias, IA, ...)
        enters :math:`W_i(z)` directly, unaffected.

        For a profile pair :math:`(u,v)`, the response (Wagner et al. 2015;
        Takada & Hu 2013) is

        .. math::

            \\frac{\\partial P_{u,v}(k,z)}{\\partial\\delta_b} =
            \\left(\\frac{47}{21} - \\frac{1}{3}\\frac{d\\ln P_{\\rm lin}}{d\\ln k}\\right)
            P_{\\rm lin}(k,z)\\, I_1^1(k \\,|\\, u)\\, I_1^1(k \\,|\\, v)
            + I_1^2(k \\,|\\, u, v)
            \\; - \\; \\left[\\theta_u\\, I_1^1(k \\,|\\, u) + \\theta_v\\, I_1^1(k \\,|\\, v)\\right]
            P_{u,v}(k,z)

        with :math:`P_{u,v}(k,z) = P_{\\rm lin}(k,z)\\, I_1^1(k \\,|\\, u)\\,
        I_1^1(k \\,|\\, v) + I_1^{2,\\rm 2pt}(k \\,|\\, u, v)`, where
        :math:`I_1^1(k\\,|\\,u) = \\int d\\ln M\\, (dn/d\\ln M)\\, b_1(M,z)\\,
        u(k\\,|\\,M,z)` is the linearly-biased single-profile mass integral,
        :math:`I_1^2` is the same integral with a naive product
        :math:`u(k\\,|\\,M,z)\\,v(k\\,|\\,M,z)` in place of a single profile,
        and :math:`I_1^{2,\\rm 2pt}` replaces that naive product with the
        halo's joint 2-point cumulant of :math:`u,v` -- a shot-noise-aware
        kernel that reduces to the naive product for continuous-field
        profiles.

        :math:`\\theta_u, \\theta_v \\in \\{0, 1\\}` flag whether
        :math:`u`/:math:`v` is a discrete number-counts observable (e.g.
        HOD), via ``needs_counterterm1``..``4`` below; the bracketed
        counter-term vanishes when both are 0.

        Parameters
        ----------
        halo_model : HaloModel
        tracer1 : Tracer
            First tracer of the :math:`C_{\\ell_1}` pair.
        tracer2 : Tracer or None
            Second tracer of the :math:`C_{\\ell_1}` pair. If None,
            defaults to ``tracer1``.
        tracer3 : Tracer or None
            First tracer of the :math:`C_{\\ell_2}` pair. If None,
            defaults to ``tracer1``.
        tracer4 : Tracer or None
            Second tracer of the :math:`C_{\\ell_2}` pair. If None,
            defaults to (the resolved) ``tracer3``.
        l1, l2 : float or jnp.ndarray
            Multipole grids for the first and second angular power
            spectrum, respectively. Need not be the same length; the two
            are broadcast into an :math:`(N_{\\ell_1}, N_{\\ell_2})` grid.
        z_range : tuple
            ``(z_min, z_max)`` spanning the Gauss-Legendre redshift integration grid.
        n_z : int
            Number of redshift-integration nodes (static: changing it triggers
            recompilation; sweeping ``z_range`` alone does not).
        f_sky : float, default 1.0
            Observed sky fraction. Also sets the disc footprint used for
            :math:`\\sigma_B^2(z)`.
        needs_counterterm1, needs_counterterm2, needs_counterterm3, needs_counterterm4 : bool or None, default None
            Whether ``tracer1``/``tracer2``/``tracer3``/``tracer4``'s
            profile is a discrete number-counts observable requiring the
            SSC counter-term above (:math:`\\theta_u`/:math:`\\theta_v` for
            the first/second response, respectively). ``None`` auto-detects
            from the tracer's profile type.

        Returns
        -------
        array
            Super-sample covariance with shape :math:`(N_{\\ell_1},
            N_{\\ell_2})`, where singleton dimensions are squeezed before
            return.
        """
        hm = halo_model
        logz, z_gl_w = gauss_legendre_nodes_weights(jnp.log(z_range[0]), jnp.log(z_range[1]), n_z)
        z = jnp.exp(logz)
        z_gl_w = z_gl_w * z  # Gauss-Legendre in ln(z); z spans orders of magnitude
        tracer2 = tracer2 if tracer2 is not None else tracer1
        tracer3 = tracer3 if tracer3 is not None else tracer1
        tracer4 = tracer4 if tracer4 is not None else tracer3

        def get_cov_slice(z_i):
            chi = hm.cosmology.angular_diameter_distance(z_i) * (1.0 + z_i)
            k1 = (l1 + 0.5) / chi
            k2 = (l2 + 0.5) / chi

            response1 = _dPk_response(
                hm, k1, z_i, tracer1.profile, tracer2.profile,
                needs_counterterm1=needs_counterterm1, needs_counterterm2=needs_counterterm2,
            )  # (N_l1,)
            response2 = _dPk_response(
                hm, k2, z_i, tracer3.profile, tracer4.profile,
                needs_counterterm1=needs_counterterm3, needs_counterterm2=needs_counterterm4,
            )  # (N_l2,)
            response_outer = response1[:, None] * response2[None, :]  # (N_l1, N_l2)

            z_lp1, lp1h1, lp3h1, sqell1 = _extended_limber_grid_for_pair(
                hm, tracer1, tracer2, tracer1.profile, tracer2.profile, z_i, l1, chi
            )
            kernel12 = _kernel_pair_effective(
                hm.cosmology, tracer1, tracer2, z_i, l1, z_lp1, lp1h1, lp3h1, sqell1
            )  # (N_l1,)

            z_lp2, lp1h2, lp3h2, sqell2 = _extended_limber_grid_for_pair(
                hm, tracer3, tracer4, tracer3.profile, tracer4.profile, z_i, l2, chi
            )
            kernel34 = _kernel_pair_effective(
                hm.cosmology, tracer3, tracer4, z_i, l2, z_lp2, lp1h2, lp3h2, sqell2
            )  # (N_l2,)

            kernels = kernel12[:, None] * kernel34[None, :]  # (N_l1, N_l2)
            sigma2_b = jnp.squeeze(hm.cosmology.sigma2_b_disc(z_i, f_sky=f_sky))
            weight = jnp.squeeze(hm.cosmology.comoving_volume_element(z_i) / chi ** 6)

            return response_outer * (kernels * weight * sigma2_b)

        # No 1/(4*pi*f_sky) prefactor here -- f_sky's effect is already fully carried by sigma2_b_disc(z, f_sky).
        integrand = jax.vmap(get_cov_slice)(z)  # (Nz, N_l1, N_l2)
        cov = jnp.sum(integrand * z_gl_w[:, None, None], axis=0)

        return jnp.squeeze(cov)


jax.tree_util.register_pytree_node(
    Tk,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: Tk._tree_unflatten(aux_data, children)
)
