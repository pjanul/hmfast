"""
Angular power spectrum covariances (connected non-Gaussian and super-sample),
sourced by the halo-model trispectrum via a Tk instance passed in explicitly.
"""

from functools import partial

import jax
import jax.numpy as jnp

from hmfast.halos.profiles.hod import GalaxyHODProfile
from hmfast.halos.profiles.profiles_2pt import _fourier_2pt
from hmfast.utils import gauss_legendre_nodes_weights

from . import cl as _cl
from .bk_tk import _pair_integral
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
    via :func:`~hmfast.stats.cl.cl_hm`, so that
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


def _pair_integral_2pt(halo_model, p1, p2, k, z):
    """
    ∫ dn/dlnM * ⟨p1(k,M,z) p2(k,M,z)⟩_2pt dlnM

    The bias_order=0 (unweighted) pair integral :math:`I_1^2`, built from the
    specialised 1-halo 2-point kernel :func:`~hmfast.halos.profiles.profiles_2pt._fourier_2pt`
    instead of a naive ``p1.fourier(k,...) * p2.fourier(k,...)`` product -- the same
    generalisation :func:`~hmfast.stats.bk_tk._pair_integral`'s own docstring anticipates, applied
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
    integral (:func:`~hmfast.stats.bk_tk._pair_integral` with ``bias_order=1``, evaluated at
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
    kernel (:func:`~hmfast.halos.profiles.profiles_2pt._fourier_2pt`) rather than a naive profile product.

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


# ------------------------------------------------------------------
# Connected (non-Gaussian) angular power spectrum covariance
# ------------------------------------------------------------------

@partial(jax.jit, static_argnums=(9,))
def covariance_cng(tk, halo_model, tracer1, tracer2, tracer3, tracer4, l1, l2, z_range, n_z, f_sky=1.0):
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
    halo-model trispectrum (see :meth:`~hmfast.stats.bk_tk.Tk.tk_1h`,
    :meth:`~hmfast.stats.bk_tk.Tk.tk_2h`, :meth:`~hmfast.stats.bk_tk.Tk.tk_3h`,
    :meth:`~hmfast.stats.bk_tk.Tk.tk_4h`).

    Every ``der_bessel=0`` term of a tracer's kernel (density,
    magnification bias, intrinsic alignment, ...) enters :math:`W_i(z)`
    directly. A ``der_bessel=2`` (RSD) term instead makes :math:`W_1(z)\\,
    W_2(z)` (or :math:`W_3(z)\\,W_4(z)`) depend on :math:`\\ell_1` (or
    :math:`\\ell_2`) too, via the same extended-Limber correction
    (Chisari et al. 2019 Sec. 2.4.1) used by
    :func:`~hmfast.stats.cl.cl_hm`
    (see :func:`_extended_limber_grid_for_pair`,
    :func:`_kernel_pair_effective`).

    Parameters
    ----------
    tk : Tk
        Trispectrum object; ``tk.include_1h``/``include_2h``/``include_3h``/
        ``include_4h`` select which terms enter :math:`T` via :meth:`~hmfast.stats.bk_tk.Tk.tk_tot`.
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

        # Respects tk.include_1h/2h/3h/4h, so a partial Tk sources a partial covariance.
        T = tk.tk_tot(hm, k1, k2, z_i, tracer1.profile, tracer2.profile, tracer3.profile, tracer4.profile)

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

@partial(jax.jit, static_argnums=(8,), static_argnames=("needs_counterterm1", "needs_counterterm2", "needs_counterterm3", "needs_counterterm4"))
def covariance_ssc(halo_model, tracer1, tracer2, tracer3, tracer4, l1, l2, z_range, n_z, f_sky=1.0,
                    needs_counterterm1=None, needs_counterterm2=None,
                    needs_counterterm3=None, needs_counterterm4=None):
    """
    Super-sample covariance (SSC) between two Limber-projected angular
    power spectra :math:`C_{\\ell_1}^{12}` and :math:`C_{\\ell_2}^{34}`.

    Sourced by long-wavelength density modes larger than the survey
    footprint, which are not measured directly but instead shift the
    mean background density of the observed volume -- rescaling every
    halo-model quantity inside it. Structurally a sibling of
    :func:`covariance_cng`: the same Limber-collapsed single
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

    As in :func:`covariance_cng`, a ``der_bessel=2`` (RSD) kernel term
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
