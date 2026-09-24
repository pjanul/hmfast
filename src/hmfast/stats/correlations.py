"""
Real-space correlation functions. Currently just the 3D halo-model
correlation function; angular correlation functions (w(theta)) will live
here too once implemented.
"""

import functools

import jax
import jax.numpy as jnp
import mcfit


def _p2xi(halo_model):
    """Build the P2xi FFTLog transform (:class:`mcfit.P2xi`) on the halo model's
    own cosmology's tabulated wavenumber grid (:meth:`Cosmology._pk_grid`), rather
    than an independent FFTLog grid -- mcfit needs this instantiated before it can
    be used in a jitted function, so this returns an already-jitted callable."""
    k, _ = halo_model.cosmology._pk_grid()
    return jax.jit(functools.partial(
        mcfit.P2xi(k, lowring=True, backend='jax'),
        axis=0, extrap=False,
    ))


def xi_hm(pk, halo_model, r, z, profile1, profile2=None):
    """
    Halo-model 3D correlation function, combining the 1-halo and 2-halo
    terms according to ``pk.include_1h``/``pk.include_2h`` (and
    ``pk.alpha_smooth``).

    .. math::

        \\xi(r, z) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\, P(k, z)\\, j_0(kr)

    obtained by an FFTLog transform (:class:`mcfit.P2xi`) of
    :meth:`~hmfast.stats.pk.Pk.pk_tot`, tabulated on ``halo_model``'s own cosmology's native
    log-spaced ``k`` grid (:meth:`Cosmology._pk_grid`) and interpolated
    onto the requested ``r``. Transforming the already-combined
    ``pk_tot`` (rather than transforming each term and summing) is
    what makes this correct when ``alpha_smooth`` differs from 1,
    since the FFTLog transform is linear but ``pk_tot``'s combination
    need not be.

    Parameters
    ----------
    pk : Pk
        Power spectrum object; ``pk.include_1h``/``include_2h``/``alpha_smooth``
        select and combine the terms entering :math:`P(k,z)` via :meth:`~hmfast.stats.pk.Pk.pk_tot`.
    halo_model : HaloModel
    r : float or jnp.ndarray
        Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
        well inside the range dual to ``halo_model``'s cosmology's own
        ``k`` grid; values of ``r`` too close to that range's edges are
        affected by FFTLog ringing.
    z : float or jnp.ndarray
        Redshift grid.
    profile1 : HaloProfile
        First halo profile object.
    profile2 : HaloProfile or None, default None
        Second halo profile object. If None, defaults to profile1.

    Returns
    -------
    xi_hm : array
        Halo-model correlation function (dimensionless), with shape
        :math:`(N_r, N_z)`, where singleton dimensions get squeezed
        before return.
    """
    r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

    k, _ = halo_model.cosmology._pk_grid()
    p_of_k = pk.pk_tot(halo_model, k, z, profile1, profile2)
    p_of_k = jnp.reshape(p_of_k, (len(k), len(z)))

    r_native, xi_native = _p2xi(halo_model)(p_of_k)
    ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

    # Linear in xi against ln r rather than log-log
    def interp_col(xi_col):
        return jnp.interp(ln_r, ln_r_native, xi_col)

    xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
    return jnp.squeeze(xi)
