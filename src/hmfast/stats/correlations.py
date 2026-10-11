"""
Real-space correlation functions: FFTLog transforms of a tabulated 3D power
spectrum or angular power spectrum, from any source (``Cosmology.pk``,
``Pk.pk_tot``, ``cl``, ``cl_linbias``, ...).
"""

import functools

import jax
import jax.numpy as jnp
import mcfit
import numpy as np


# Bessel order of the flat-sky Hankel kernel for each angular correlation type.
_ANGULAR_NU = {"NN": 0, "NG": 2, "GG+": 0, "GG-": 4}


def _log_grid(x, name):
    """Return ``x`` as a concrete 1D numpy array, checking it is log-uniformly spaced."""
    try:
        x = np.asarray(x, dtype=np.float64)
    except jax.errors.TracerArrayConversionError:
        raise ValueError(
            f"`{name}` must be concrete (e.g. a numpy array), not traced: the FFTLog plan is built from its values."
        ) from None
    if x.ndim != 1 or x.size < 2 or np.any(x <= 0):
        raise ValueError(f"`{name}` must be a 1D array of at least 2 positive values.")
    dlnx = np.diff(np.log(x))
    if not np.allclose(dlnx, dlnx[0], rtol=1e-6, atol=0.0):
        raise ValueError(f"`{name}` must be log-uniformly spaced, e.g. np.geomspace({x[0]:g}, {x[-1]:g}, {x.size}).")
    return x


@functools.lru_cache(maxsize=None)
def _transform(kind, x_bytes, nu, extrap):
    """Jitted mcfit transform along axis 0, built once per grid outside any trace."""
    x = np.frombuffer(x_bytes, dtype=np.float64)
    # mcfit plans on concrete values, so keep its setup concrete even when first reached under a trace.
    with jax.ensure_compile_time_eval():
        if kind == "P2xi":
            T = mcfit.P2xi(x, lowring=True, backend="jax")
        else:
            T = mcfit.C2w(x, nu=nu, lowring=True, backend="jax")
    return jax.jit(functools.partial(T, axis=0, extrap=extrap))


def _fftlog_interp(kind, x, f, y, nu, extrap):
    """Transform ``f`` (tabulated on ``x`` along axis 0) and interpolate the result onto ``y``."""
    f = jnp.asarray(f)
    if f.shape[0] != x.size:
        raise ValueError(f"Leading axis of the input ({f.shape[0]}) must match the grid length ({x.size}).")
    trailing = f.shape[1:]
    y_native, g_native = _transform(kind, x.tobytes(), nu, bool(extrap))(f.reshape(x.size, -1))

    # Linear in the output against ln y rather than log-log, so sign changes are safe
    ln_y, ln_y_native = jnp.log(jnp.atleast_1d(y)), jnp.log(y_native)
    g = jax.vmap(lambda col: jnp.interp(ln_y, ln_y_native, col), in_axes=1, out_axes=1)(g_native)
    return jnp.squeeze(g.reshape(ln_y.shape + trailing))


def corr_3d(cosmology, k, pk, r, *, extrapolate=True):
    """
    3D correlation function of a tabulated power spectrum,

    .. math::

        \\xi(r) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\, P(k)\\, j_0(kr),

    by an FFTLog transform interpolated onto ``r``. Works for matter power
    spectra (:meth:`~hmfast.cosmology.Cosmology.pk`) and halo-model power
    spectra (:class:`~hmfast.stats.Pk`).

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object.
    k : array-like
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}`, shape :math:`(N_k,)`;
        concrete, positive and log-uniformly spaced.
    pk : jnp.ndarray
        Power spectrum in :math:`\\mathrm{Mpc}^3` on ``k``, shape :math:`(N_k,)` at
        one redshift or :math:`(N_k, N_z)` at :math:`N_z` redshifts.
    r : float or jnp.ndarray
        Comoving separations in :math:`\\mathrm{Mpc}`, scalar or shape :math:`(N_r,)`.
        Only reliable well inside :math:`[1/k_{\\max}, 1/k_{\\min}]`; FFTLog rings near the edges.
    extrapolate : bool
        If True (default), ``pk`` is extended as a power law beyond the ends of ``k`` before the transform;
        if False, it is padded with zeros.

    Returns
    -------
    xi : jnp.ndarray
        Correlation function with shape :math:`(N_r, N_z)`, where singleton
        dimensions get squeezed before return.
    """
    k = _log_grid(k, "k")
    return _fftlog_interp("P2xi", k, pk, r, 0, extrapolate)


def corr_angular(cosmology, l, cl, theta, *, type="NN", extrapolate=True):
    """
    Flat-sky angular correlation function of a tabulated angular power spectrum,

    .. math::

        \\xi(\\theta) = \\int \\frac{d\\ell\\, \\ell}{2\\pi}\\, C_\\ell\\, J_\\nu(\\ell\\theta),

    by an FFTLog transform interpolated onto ``theta``, with
    :math:`\\nu = 0, 2, 0, 4` for ``type`` = NN (e.g. :math:`w(\\theta)`),
    NG (:math:`\\gamma_t`), GG+ (:math:`\\xi_+`), GG- (:math:`\\xi_-`).
    Works for halo-model (:func:`~hmfast.stats.cl`) and linear-bias
    (:func:`~hmfast.stats.cl_linbias`) angular power spectra.

    .. warning::

        This is the flat-sky (small-angle) approximation: sub-percent below a
        few degrees, increasingly wrong above. Do not use it for large-angle
        measurements without checking against a full-sky Legendre sum.

    For clustering at large ``theta``, compute :math:`C_\\ell` with
    ``l_limber > 0``, since Limber is inaccurate at low :math:`\\ell`. Shear
    :math:`C_\\ell` vanish at :math:`\\ell \\le 1`, so with ``extrapolate=True`` start ``l`` at
    :math:`\\ell \\ge 2`; power-law extrapolation of a zero gives NaN.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object.
    l : array-like
        Multipoles, shape :math:`(N_\\ell,)`; concrete, positive and log-uniformly spaced.
    cl : jnp.ndarray
        Angular power spectrum on ``l``, shape :math:`(N_\\ell,)` for one spectrum
        or :math:`(N_\\ell, N_c)` for :math:`N_c` spectra stacked as columns
        (e.g. tomographic bin pairs).
    theta : float or jnp.ndarray
        Angular separations in degrees, scalar or shape :math:`(N_\\theta,)`. Only reliable
        well inside :math:`[1/\\ell_{\\max}, 1/\\ell_{\\min}]` (radians); FFTLog rings near the edges.
    type : str, default "NN"
        Spin combination; ``type`` ∈ {"NN", "NG", "GG+", "GG-"} (static).
    extrapolate : bool
        If True (default), ``cl`` is extended as a power law beyond the ends of ``l`` before the transform;
        if False, it is padded with zeros.

    Returns
    -------
    xi : jnp.ndarray
        Correlation function with shape :math:`(N_\\theta, N_c)`, where singleton
        dimensions get squeezed before return.
    """
    if type not in _ANGULAR_NU:
        raise ValueError(f"type must be one of {sorted(_ANGULAR_NU)}; got {type!r}.")
    l = _log_grid(l, "l")
    return _fftlog_interp("C2w", l, cl, jnp.deg2rad(theta), _ANGULAR_NU[type], extrapolate)
