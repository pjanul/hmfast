import jax.numpy as jnp

from .cib import CIBProfile
from .hod import GalaxyHODProfile


def _fourier_2pt_hod(halo_model, profile1, profile2, k, m, z):
    k = jnp.atleast_1d(k)
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)

    ns1 = jnp.atleast_1d(profile1.n_sat(halo_model, m))
    ns2 = jnp.atleast_1d(profile2.n_sat(halo_model, m))
    ng1 = jnp.atleast_1d(profile1.ng_bar(halo_model, z))
    ng2 = jnp.atleast_1d(profile2.ng_bar(halo_model, z))

    _, u1 = profile1._u_k_nfw(halo_model, k, m, z)
    _, u2 = profile2._u_k_nfw(halo_model, k, m, z)
    u1 = jnp.reshape(u1, (len(k), len(m), len(z)))
    u2 = jnp.reshape(u2, (len(k), len(m), len(z)))

    sat_sat = ns1[None, :, None] * ns2[None, :, None] * u1 * u2
    sat_cen = ns1[None, :, None] * u1 + ns2[None, :, None] * u2
    return jnp.squeeze((sat_sat + sat_cen) / (ng1[None, None, :] * ng2[None, None, :]))


def _fourier_2pt_cib(halo_model, profile1, profile2, k, m, z):
    k = jnp.atleast_1d(k)
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)

    ls1 = jnp.reshape(profile1.l_sat(halo_model, m, z), (len(m), len(z)))
    lc1 = jnp.reshape(profile1.l_cen(halo_model, m, z), (len(m), len(z)))
    ls2 = jnp.reshape(profile2.l_sat(halo_model, m, z), (len(m), len(z)))
    lc2 = jnp.reshape(profile2.l_cen(halo_model, m, z), (len(m), len(z)))

    _, u1 = profile1._u_k_nfw(halo_model, k, m, z)
    _, u2 = profile2._u_k_nfw(halo_model, k, m, z)
    u1 = jnp.reshape(u1, (len(k), len(m), len(z)))
    u2 = jnp.reshape(u2, (len(k), len(m), len(z)))

    prefactor = 1.0 / (4.0 * jnp.pi) ** 2
    sat_sat = ls1[None, :, :] * ls2[None, :, :] * u1 * u2
    sat_cen = ls1[None, :, :] * lc2[None, :, :] * u1
    cen_sat = ls2[None, :, :] * lc1[None, :, :] * u2
    return jnp.squeeze(prefactor * (sat_sat + sat_cen + cen_sat))


def _fourier_2pt(halo_model, profile1, profile2, k, m, z):
    """
    Transitional helper for 1-halo Fourier-space second moments.

    This private API is intended to replace the legacy central/satellite split
    in the halo-model 1-halo term. For now it supports special handling only
    for HOD x HOD and CIB x CIB. Mixed HOD x CIB pairs fall back to the
    product of the two first moments.
    """
    k = jnp.atleast_1d(k)
    m = jnp.atleast_1d(m)
    z = jnp.atleast_1d(z)

    if isinstance(profile1, GalaxyHODProfile) and isinstance(profile2, GalaxyHODProfile):
        return _fourier_2pt_hod(halo_model, profile1, profile2, k, m, z)

    if isinstance(profile1, CIBProfile) and isinstance(profile2, CIBProfile):
        return _fourier_2pt_cib(halo_model, profile1, profile2, k, m, z)

    u1 = jnp.reshape(profile1.fourier(halo_model, k, m, z), (len(k), len(m), len(z)))
    u2 = jnp.reshape(profile2.fourier(halo_model, k, m, z), (len(k), len(m), len(z)))
    return jnp.squeeze(u1 * u2)