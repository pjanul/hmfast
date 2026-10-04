"""
Angular power spectrum covariances (connected non-Gaussian and super-sample),
sourced by the halo-model trispectrum via a Tk instance passed in explicitly.
"""

from functools import partial

import jax
import jax.numpy as jnp
import mcfit
import numpy as np

from hmfast.halos.profiles.hod import GalaxyHODProfile
from hmfast.utils import gauss_legendre_nodes_weights, log_interp1d_extrap

from . import projected_cl as _cl
from .pk import Pk as _Pk


def _extended_limber_grid_for_pair(hm, tracer_a, tracer_b, profile_a, profile_b, z, l, chi, k_damp=0.01):
    """
    Extended-Limber shifted-grid quantities (see :func:`hmfast.stats.projected_cl._extended_limber_kernel_grid`)
    for one leg of a covariance -- ``tracer_a``/``tracer_b`` share a single multipole
    ``l`` and wavenumber :math:`k=(\\ell+1/2)/\\chi`, exactly as the two tracers of a
    single :math:`C_\\ell` would. Returns ``(None, None, None, None)`` if neither
    tracer has a ``der_bessel!=0`` (e.g. RSD) kernel term, in which case the caller's
    kernel product stays scalar-in-``l``, identical to the pre-extended-Limber behavior.

    The reference :math:`P(k,z)` used for the correction is this leg's own halo-model
    :math:`P_{1h}+P_{2h}` (built from ``profile_a``, ``profile_b``) -- the same
    :math:`P(k,z)` that would enter this leg's own :math:`C_\\ell` if computed directly
    via :func:`~hmfast.stats.cl`, so that
    RSD's projection correction is treated consistently between the trispectrum/
    covariance code here and the two-point ``C_\\ell`` code in ``stats/cl.py``.
    """
    cosmology = hm.cosmology
    z_arr = jnp.atleast_1d(z)
    needs_extended = (
        any(der_bessel > 0 for _, der_bessel, _ in tracer_a._kernel_terms(cosmology, z_arr))
        or any(der_bessel > 0 for _, der_bessel, _ in tracer_b._kernel_terms(cosmology, z_arr))
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
    (:func:`hmfast.stats.projected_cl._effective_kernel_limber`), evaluated at a single ``z``
    (called once per redshift slice inside ``cov_cng``/``cov_ssc``'s
    ``vmap`` over ``z``). Every ``der_bessel=0`` term of each tracer (density, tSZ, ...)
    is summed directly; a ``der_bessel=-1`` term (lensing, magnification bias, IA) picks
    up :math:`f^{(a)}_\\ell/(\\ell+1/2)^2`; a ``der_bessel=2`` (RSD) term is
    projected through the extended-Limber correction (``z_lp``/``lp1h``/``lp3h``/
    ``sqell``, or ``None`` if neither tracer needs it -- see
    :func:`_extended_limber_grid_for_pair`).

    Returns shape ``(Nl,)``: a plain per-``z`` scalar broadcastable against ``Nl`` when
    every term of both tracers is ``(W, 0, 0)``, ``l``-dependent otherwise.
    """
    z_arr = jnp.atleast_1d(z)
    ka = _cl._effective_kernel_limber(tracer_a, cosmology, z_arr, l, z_lp, lp1h, lp3h, sqell)
    kb = _cl._effective_kernel_limber(tracer_b, cosmology, z_arr, l, z_lp, lp1h, lp3h, sqell)
    ka = ka[:, None] if ka.ndim == 1 else ka  # (1,1) or (1,Nl)
    kb = kb[:, None] if kb.ndim == 1 else kb
    return jnp.squeeze(ka * kb, axis=0)  # (Nl,) or (1,)


def _dPk_response(halo_model, k, z, profile1, profile2=None, include_1h=True, include_2h=True,
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

    where :math:`I_1^1` and :math:`I_1^2` are the linearly-biased single- and
    paired-profile mass integrals (:meth:`~hmfast.halos.HaloModel.mass_integral`
    with ``bias_order=1``), the latter the plain product of first moments at
    matching :math:`k` for both legs.

    If :math:`u` or :math:`v` is a discrete number-counts observable
    (a :class:`GalaxyHODProfile`) rather than a continuous field, the
    number-counts counter-term is subtracted:

    .. math::

        \\partial P_{u,v}/\\partial\\delta_b \\; \\mathrel{-}= \\;
        (b_u+b_v)\\,P_{u,v}(k,z), \\qquad
        P_{u,v} = P_{\\rm lin}\\,I_1^1(k\\,|\\,u)\\,I_1^1(k\\,|\\,v) + I_1^{2,\\rm 2pt}(k\\,|\\,u,v)

    where :math:`b_u = I_1^1(k\\,|\\,u)` (only included for number-counts legs) and :math:`I_1^{2,\\rm 2pt}` is the unweighted pair integral
    :meth:`~hmfast.halos.HaloModel.mass_integral` at a shared :math:`k`, which
    uses the 1-halo second moment rather than a naive profile product.

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
    include_1h, include_2h : bool, default True
        Whether to keep the terms built from :math:`I_1^2`/:math:`I_1^{2,\\rm 2pt}` (1-halo)
        and from :math:`P_{\\rm lin}` (2-halo) (static).
    needs_counterterm1, needs_counterterm2 : bool or None, default None
        Whether ``profile1``/``profile2`` gets the counter-term; None detects a
        :class:`GalaxyHODProfile` (static).

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

    i11_u = jnp.reshape(hm.mass_integral(k_arr, z_arr, profile1, bias_order=1), (nk, nz))
    i11_v = jnp.reshape(hm.mass_integral(k_arr, z_arr, profile2, bias_order=1), (nk, nz))
    # (k, k) rather than a shared k keeps the plain product of first moments, not the 1-halo second moment.
    i12_uv = jnp.reshape(hm.mass_integral((k_arr, k_arr), z_arr, (profile1, profile2), bias_order=1), (nk, nz))
    pk_lin = jnp.reshape(hm.cosmology.pk(k_arr, z_arr, linear=True), (nk, nz))

    # dlnP/dlnk needs a properly resolved k grid -- a finite difference on a sparse query k_arr is inaccurate.
    k_fine = hm.cosmology._pk_grid()
    pk_fine = jnp.reshape(hm.cosmology.pk(k_fine, z_arr, linear=True), (len(k_fine), nz))
    dlnp_fine = jnp.gradient(jnp.log(pk_fine), jnp.log(k_fine), axis=0)
    dlnp_dlnk = jax.vmap(
        lambda dp: jnp.interp(jnp.log(k_arr), jnp.log(k_fine), dp), in_axes=1, out_axes=1
    )(dlnp_fine)

    pk_2h = pk_lin * i11_u * i11_v if include_2h else 0.0
    response = (47.0 / 21.0 - dlnp_dlnk / 3.0) * pk_2h + (i12_uv if include_1h else 0.0)

    if needs_counterterm1 or needs_counterterm2:
        i02_uv = jnp.reshape(hm.mass_integral(k_arr, z_arr, (profile1, profile2), bias_order=0), (nk, nz))
        P_uv = pk_2h + (i02_uv if include_1h else 0.0)
        b_u = i11_u if needs_counterterm1 else 0.0
        b_v = i11_v if needs_counterterm2 else 0.0
        response = response - (b_u + b_v) * P_uv

    return jnp.squeeze(response)


# ------------------------------------------------------------------
# Connected (non-Gaussian) angular power spectrum covariance
# ------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_z",))
def cov_cng(tk, halo_model, l1, l2, tracer1, tracer2=None, tracer3=None, tracer4=None, *, z_range=None, n_z=100,
            f_sky=1.0):
    """
    Connected non-Gaussian covariance between two angular power spectra
    :math:`C_{\\ell_1}^{12}` and :math:`C_{\\ell_2}^{34}`,

    .. math::

        {\\rm Cov}(\\ell_1, \\ell_2) = \\frac{T_{\\ell_1 \\ell_2}}{4\\pi f_{\\rm sky}},

    where :math:`T_{\\ell_1 \\ell_2}` is the angular trispectrum, in the Limber approximation

    .. math::

        T_{\\ell_1 \\ell_2} = \\int d\\chi\\,
        \\frac{W_1(\\chi)\\, W_2(\\chi)\\, W_3(\\chi)\\, W_4(\\chi)}{\\chi^6}\\,
        T\\!\\left(k_1 = \\frac{\\ell_1 + 1/2}{\\chi}, k_2 = \\frac{\\ell_2 + 1/2}{\\chi}, z(\\chi)\\right),

    with :math:`W_i` the tracer kernels and :math:`T` the halo-model trispectrum,
    with the terms selected by ``tk``.

    Parameters
    ----------
    tk : Tk
        Trispectrum object; ``tk.include_1h``/``include_2h``/``include_3h``/``include_4h`` select the terms.
    halo_model : HaloModel
        Halo model object.
    l1, l2 : float or jnp.ndarray
        Multipoles of the first and second angular power spectrum, scalar or
        shapes :math:`(N_{\\ell_1},)` and :math:`(N_{\\ell_2},)`.
    tracer1 : Tracer
        First tracer of the :math:`C_{\\ell_1}` pair.
    tracer2 : Tracer or None, default None
        Second tracer of the :math:`C_{\\ell_1}` pair. If None, defaults to ``tracer1``.
    tracer3 : Tracer or None, default None
        First tracer of the :math:`C_{\\ell_2}` pair. If None, defaults to ``tracer1``.
    tracer4 : Tracer or None, default None
        Second tracer of the :math:`C_{\\ell_2}` pair. If None, defaults to (the resolved) ``tracer3``.
    z_range : tuple or None, default None
        ``(z_min, z_max)`` of the redshift integration. If None, inferred from all
        four tracers as in :func:`~hmfast.stats.cl`.
    n_z : int, default 100
        Number of redshift nodes (static).
    f_sky : float, default 1.0
        Observed sky fraction.

    Returns
    -------
    cov : jnp.ndarray
        Covariance with shape :math:`(N_{\\ell_1}, N_{\\ell_2})`, where singleton
        dimensions get squeezed before return.
    """
    hm = halo_model
    l1, l2 = jnp.atleast_1d(l1), jnp.atleast_1d(l2)
    tracer2 = tracer2 if tracer2 is not None else tracer1
    tracer3 = tracer3 if tracer3 is not None else tracer1
    tracer4 = tracer4 if tracer4 is not None else tracer3
    z_range = _cl._resolve_z_range(hm.cosmology, z_range, tracer1, tracer2, tracer3, tracer4)
    logz, z_gl_w = gauss_legendre_nodes_weights(jnp.log(z_range[0]), jnp.log(z_range[1]), n_z)
    z = jnp.exp(logz)
    z_gl_w = z_gl_w * z  # Gauss-Legendre in ln(z); z spans orders of magnitude

    def get_cov_slice(z_i):
        chi = hm.cosmology.angular_diameter_distance(z_i) * (1.0 + z_i)
        k1 = (l1 + 0.5) / chi
        k2 = (l2 + 0.5) / chi

        # Respects tk.include_1h/2h/3h/4h, so a partial Tk sources a partial covariance.
        T = jnp.reshape(
            tk.tk_tot(hm, k1, k2, z_i, tracer1.profile, tracer2.profile, tracer3.profile, tracer4.profile),
            (l1.size, l2.size),
        )

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

# One mcfit plan per emulator set: the disc-window transform depends only on the emulator's k grid.
_DISC_VAR_TRANSFORMS = {}
# Decades of power-law P_lin below the emulator's k_min: the output radii then reach ~1e6 Mpc, past chi(z) * pi.
_DISC_VAR_LOW_K_DECADES = 2


def _disc_var_transform(cosmology):
    """The disc-variance transform and its input k grid, the emulator's extended below k_min at the same log spacing."""
    key = cosmology.engine
    if key not in _DISC_VAR_TRANSFORMS:
        k_grid = cosmology._pk_grid()
        dlnk = np.log(k_grid[1] / k_grid[0])
        n_low = int(np.ceil(_DISC_VAR_LOW_K_DECADES * np.log(10.0) / dlnk))
        k_ext = np.concatenate([k_grid[0] * np.exp(dlnk * np.arange(-n_low, 0)), k_grid])
        # Build eagerly even when first reached inside jit: mcfit needs a concrete grid to plan on.
        with jax.ensure_compile_time_eval():
            # Same as Cosmology's TophatVar, but dim=2 (disc window) instead of dim=3 (sphere).
            disc_var = mcfit.mcfit(k_ext, mcfit.kernels.Mellin_TophatSq(2), q=1.5, lowring=True, backend='jax')
            disc_var.prefac *= disc_var.x**2 / (2.0 * jnp.pi)
        _DISC_VAR_TRANSFORMS[key] = (partial(disc_var, extrap=True), k_ext)
    return _DISC_VAR_TRANSFORMS[key]


@jax.jit
def sigma2_b_disc(cosmology, z, *, f_sky=1.0):
    """
    Variance of the linear density field over a circular footprint of sky
    fraction :math:`f_{\\rm sky}`, entering :func:`cov_ssc`,

    .. math::

        \\sigma_B^2(z) = \\int_0^\\infty \\frac{k\\,dk}{2\\pi}\\,
        P_{\\rm lin}(k, z)\\, \\left[\\frac{2 J_1(kR)}{kR}\\right]^2,
        \\qquad R(z) = \\chi(z) \\arccos(1 - 2 f_{\\rm sky}),

    by an FFTLog transform.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object.
    z : float or jnp.ndarray
        Redshifts, scalar or shape :math:`(N_z,)`.
    f_sky : float, default 1.0
        Observed sky fraction.

    Returns
    -------
    sigma2_b : jnp.ndarray
        Footprint variance with shape :math:`(N_z,)`, where singleton dimensions
        get squeezed before return.
    """
    z_arr = jnp.atleast_1d(z)
    k_grid = cosmology._pk_grid()
    pk_grid = jnp.reshape(cosmology.pk(k_grid, z_arr, linear=True), (len(k_grid), len(z_arr)))
    transform, k_ext = _disc_var_transform(cosmology)
    # Power-law continuation of P_lin below k_min, independent of cosmology.extrapolate_k.
    pk_ext = jax.vmap(lambda pk_i: log_interp1d_extrap(k_ext, k_grid, pk_i), in_axes=1, out_axes=1)(pk_grid)

    R_grid, var_grid = jax.vmap(transform, in_axes=1, out_axes=(0, 0))(pk_ext)
    R_grid = R_grid[0]  # same R grid for every z -- only depends on k_grid

    chi = cosmology.angular_diameter_distance(z_arr) * (1.0 + z_arr)
    R_target = chi * jnp.arccos(1.0 - 2.0 * f_sky)

    ln_var = jax.vmap(
        lambda r_i, var_i: jnp.interp(jnp.log(r_i), jnp.log(R_grid), jnp.log(var_i))
    )(R_target, var_grid)

    return jnp.squeeze(jnp.exp(ln_var))


@partial(jax.jit, static_argnames=("n_z", "counterterms"))
def cov_ssc(pk, halo_model, l1, l2, tracer1, tracer2=None, tracer3=None, tracer4=None, *, z_range=None, n_z=100,
            f_sky=1.0, sigma2_b=None, counterterms=None):
    """
    Super-sample covariance between two angular power spectra
    :math:`C_{\\ell_1}^{12}` and :math:`C_{\\ell_2}^{34}`, from density modes larger
    than the survey footprint. In the Limber approximation

    .. math::

        {\\rm Cov}(\\ell_1, \\ell_2) = \\int d\\chi\\,
        \\frac{W_1(\\chi)\\, W_2(\\chi)\\, W_3(\\chi)\\, W_4(\\chi)}{\\chi^4}\\,
        \\sigma_B^2(z)\\,
        \\frac{\\partial P_{12}(k_1, z)}{\\partial \\delta_b}\\,
        \\frac{\\partial P_{34}(k_2, z)}{\\partial \\delta_b},

    with :math:`k_i = (\\ell_i + 1/2)/\\chi`, :math:`W_i` the tracer kernels and
    :math:`\\sigma_B^2(z)` the variance of the background mode over the footprint
    (:func:`sigma2_b_disc` by default). The halo-model response of a profile pair
    :math:`(u, v)` (Takada & Hu 2013) is

    .. math::

        \\frac{\\partial P_{uv}}{\\partial \\delta_b} =
        \\left(\\frac{47}{21} - \\frac{1}{3}\\frac{d\\ln P_{\\rm lin}}{d\\ln k}\\right) P_{2h}
        + I_1^2(k \\,|\\, u, v)
        - \\left(\\theta_u b_u + \\theta_v b_v\\right) P_{uv},

    where :math:`I_1^2` is the bias-weighted 1-halo integral, :math:`b_u(k, z)` the
    large-scale bias of :math:`u`, :math:`P_{uv} = P_{\\rm lin} b_u b_v + I_1^{2,\\rm 2pt}(k \\,|\\, u, v)`
    with :math:`I_1^{2,\\rm 2pt}` the halo's 2-point pair integral (the profile product for
    continuous fields), and :math:`\\theta_u \\in \\{0, 1\\}` switches on the number-count
    counter-term (see ``counterterms``).

    Parameters
    ----------
    pk : Pk
        Power spectrum object; ``pk.include_1h``/``include_2h`` select the terms of the response.
    halo_model : HaloModel
        Halo model object.
    l1, l2 : float or jnp.ndarray
        Multipoles of the first and second angular power spectrum, scalar or
        shapes :math:`(N_{\\ell_1},)` and :math:`(N_{\\ell_2},)`.
    tracer1 : Tracer
        First tracer of the :math:`C_{\\ell_1}` pair.
    tracer2 : Tracer or None, default None
        Second tracer of the :math:`C_{\\ell_1}` pair. If None, defaults to ``tracer1``.
    tracer3 : Tracer or None, default None
        First tracer of the :math:`C_{\\ell_2}` pair. If None, defaults to ``tracer1``.
    tracer4 : Tracer or None, default None
        Second tracer of the :math:`C_{\\ell_2}` pair. If None, defaults to (the resolved) ``tracer3``.
    z_range : tuple or None, default None
        ``(z_min, z_max)`` of the redshift integration. If None, inferred from all
        four tracers as in :func:`~hmfast.stats.cl`.
    n_z : int, default 100
        Number of redshift nodes (static).
    f_sky : float, default 1.0
        Observed sky fraction, used only by the default ``sigma2_b``.
    sigma2_b : tuple of jnp.ndarray or None, default None
        :math:`\\sigma_B^2(z)` tabulated as ``(z, sigma2_b)``, each of shape :math:`(N,)`,
        linearly interpolated in :math:`z` and held constant beyond the ends.
        For gradients with respect to cosmology, compute it from the same ``Cosmology``.
        If None, :func:`sigma2_b_disc` for a circular footprint of ``f_sky``.
    counterterms : tuple of 4 bool or None, default None
        :math:`\\theta` for ``tracer1``..``tracer4``, e.g. ``(1, 1, 0, 0)`` (static).
        If None, a tracer gets the counter-term if its profile is a
        :class:`GalaxyHODProfile`.

    Returns
    -------
    cov : jnp.ndarray
        Covariance with shape :math:`(N_{\\ell_1}, N_{\\ell_2})`, where singleton
        dimensions get squeezed before return.
    """
    hm = halo_model
    l1, l2 = jnp.atleast_1d(l1), jnp.atleast_1d(l2)
    tracer2 = tracer2 if tracer2 is not None else tracer1
    tracer3 = tracer3 if tracer3 is not None else tracer1
    tracer4 = tracer4 if tracer4 is not None else tracer3
    z_range = _cl._resolve_z_range(hm.cosmology, z_range, tracer1, tracer2, tracer3, tracer4)
    logz, z_gl_w = gauss_legendre_nodes_weights(jnp.log(z_range[0]), jnp.log(z_range[1]), n_z)
    z = jnp.exp(logz)
    z_gl_w = z_gl_w * z  # Gauss-Legendre in ln(z); z spans orders of magnitude
    if counterterms is None:
        counterterms = (None,) * 4
    elif len(counterterms) != 4:
        raise ValueError(f"counterterms must have one entry per tracer (4); got {counterterms!r}.")
    ct1, ct2, ct3, ct4 = (None if c is None else bool(c) for c in counterterms)
    if sigma2_b is None:
        sigma2_b_fn = lambda z_i: sigma2_b_disc(hm.cosmology, z_i, f_sky=f_sky)
    else:
        sigma2_b_fn = lambda z_i: jnp.interp(z_i, jnp.asarray(sigma2_b[0]), jnp.asarray(sigma2_b[1]))

    def get_cov_slice(z_i):
        chi = hm.cosmology.angular_diameter_distance(z_i) * (1.0 + z_i)
        k1 = (l1 + 0.5) / chi
        k2 = (l2 + 0.5) / chi

        response1 = jnp.reshape(_dPk_response(
            hm, k1, z_i, tracer1.profile, tracer2.profile, include_1h=pk.include_1h, include_2h=pk.include_2h,
            needs_counterterm1=ct1, needs_counterterm2=ct2,
        ), (l1.size,))  # (N_l1,)
        response2 = jnp.reshape(_dPk_response(
            hm, k2, z_i, tracer3.profile, tracer4.profile, include_1h=pk.include_1h, include_2h=pk.include_2h,
            needs_counterterm1=ct3, needs_counterterm2=ct4,
        ), (l2.size,))  # (N_l2,)
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
        sigma2_b_i = jnp.squeeze(sigma2_b_fn(z_i))
        weight = jnp.squeeze(hm.cosmology.comoving_volume_element(z_i) / chi ** 6)

        return response_outer * (kernels * weight * sigma2_b_i)

    # No 1/(4*pi*f_sky) prefactor: the footprint enters only through sigma2_b.
    integrand = jax.vmap(get_cov_slice)(z)  # (Nz, N_l1, N_l2)
    cov = jnp.sum(integrand * z_gl_w[:, None, None], axis=0)

    return jnp.squeeze(cov)
