import jax
import jax.numpy as jnp


# -------------------------
# F2 kernel (unchanged)
# -------------------------
def _F2(k1, k2, mu):
    """Leading-order SPT kernel F2 (EdS approximation)."""
    k1_over_k2 = jnp.where(k2 == 0.0, 0.0, k1 / k2)
    k2_over_k1 = jnp.where(k1 == 0.0, 0.0, k2 / k1)
    return 5.0 / 7.0 + 0.5 * (k1_over_k2 + k2_over_k1) * mu + 2.0 / 7.0 * mu ** 2


# -------------------------
# G2 kernel (unchanged)
# -------------------------
def _G2(k1, k2, mu):
    """Leading-order velocity-divergence kernel G2 (EdS approximation)."""
    k1_over_k2 = jnp.where(k2 == 0.0, 0.0, k1 / k2)
    k2_over_k1 = jnp.where(k1 == 0.0, 0.0, k2 / k1)
    return 3.0 / 7.0 + 0.5 * (k1_over_k2 + k2_over_k1) * mu + 4.0 / 7.0 * mu ** 2


# -------------------------
# cosine law helper (safe)
# -------------------------
@jax.jit
def _mu(k1, k2, k3):
    """Cosine between k1 and k2 given opposite side k3."""
    den = 2.0 * k1 * k2
    return jnp.where(
        den == 0.0,
        0.0,
        (k3 ** 2 - k1 ** 2 - k2 ** 2) / den
    )


# -------------------------
# FIXED F3 (symmetrized approximation)
# -------------------------
@jax.jit
def _F3(k1, k2, k3, k4):
    """
    Symmetry-fixed EdS-inspired F3 approximation.

    Key fix: no single ordering; average over channels.
    """

    def channel(a, b, c):
        k_ab = jnp.sqrt(a**2 + b**2)
        mu_ab = _mu(a, b, k_ab)
        F2_ab = _F2(a, b, mu_ab)

        k_abc = jnp.sqrt(k_ab**2 + c**2)
        mu_abc = _mu(k_ab, c, k_abc)
        F2_abc = _F2(k_ab, c, mu_abc)

        return F2_ab * F2_abc

    return (
        channel(k1, k2, k3) +
        channel(k1, k2, k4) +
        channel(k1, k3, k4) +
        channel(k2, k3, k4)
    ) / 4.0


# -------------------------
# Bispectrum (unchanged)
# -------------------------
def _bk(cosmology, k1, k2, k3, z):

    k1, k2, k3 = jnp.asarray(k1), jnp.asarray(k2), jnp.asarray(k3)

    mu12 = (k1**2 + k2**2 - k3**2) / (2.0 * k1 * k2)
    mu23 = (k2**2 + k3**2 - k1**2) / (2.0 * k2 * k3)
    mu31 = (k3**2 + k1**2 - k2**2) / (2.0 * k3 * k1)

    P1 = cosmology.pk(k1, z)
    P2 = cosmology.pk(k2, z)
    P3 = cosmology.pk(k3, z)

    B12 = 2.0 * _F2(k1, k2, mu12) * P1 * P2
    B23 = 2.0 * _F2(k2, k3, mu23) * P2 * P3
    B31 = 2.0 * _F2(k3, k1, mu31) * P3 * P1

    return B12 + B23 + B31


# -------------------------
# FIXED trispectrum
# -------------------------
@jax.jit
def _tk(cosmology, k1, k2, k3, k4, z):

    k = jnp.array([k1, k2, k3, k4])
    P = cosmology.pk(k, z)

    # ---------------------
    # T22 term (stable pairing)
    # ---------------------
    mu12 = _mu(k1, k2, jnp.sqrt(k1**2 + k2**2))
    mu34 = _mu(k3, k4, jnp.sqrt(k3**2 + k4**2))

    F12 = _F2(k1, k2, mu12)
    F34 = _F2(k3, k4, mu34)

    T22 = 4.0 * F12 * F34 * P[0] * P[1] * P[2] * P[3]

    # ---------------------
    # T13 term (FIXED symmetry)
    # ---------------------
    T13 = (
        _F3(k1, k2, k3, k4) * P[0] * P[1] * P[2] +
        _F3(k1, k2, k4, k3) * P[0] * P[1] * P[3] +
        _F3(k1, k3, k4, k2) * P[0] * P[2] * P[3] +
        _F3(k2, k3, k4, k1) * P[1] * P[2] * P[3]
    )

    return T22 + T13


# -------------------------
# vectorized version
# -------------------------
_vtk = jax.vmap(_tk, in_axes=(None, 0, 0, 0, 0, None))