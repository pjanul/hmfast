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

    mu12 = _mu(k1, k2, k3)
    mu23 = _mu(k2, k3, k1)
    mu31 = _mu(k3, k1, k2)

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


# -------------------------
# Halo model bispectrum
# -------------------------

class _Bk:
    """
    Halo model bispectrum B(k1, k2, k3, z).

    Computes the 1-, 2-, and 3-halo contributions following the standard halo
    model decomposition, matching class_sz ``bk_at_z_hm``.

    The three-halo term includes both the linear-bias (tree-level SPT) piece and
    the quadratic-bias (b2) correction.

    For profiles whose higher moments reduce to simple products of their
    Fourier-space first moments (e.g. matter, tSZ, CMB lensing), the nth-order
    profile product within a single halo is taken as u1(k1,M)*...*un(kn,M).
    Profiles with more complex intra-halo occupancy statistics (HOD, CIB) will
    require a dedicated ``_fourier_3pt`` helper analogous to ``_fourier_2pt`` in
    ``profiles_2pt.py`` — this is left as a future extension point.

    ``HaloModel`` is **not** stored as class state — it is passed explicitly to
    each method so that the same ``_Bk`` instance can be reused across different
    halo model configurations.
    """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pair_integral(self, halo_model, p1, p2, k1, k2, z):
        """
        ∫ dn/dlnM * b1(M) * p1.fourier(k1,M,z) * p2.fourier(k2,M,z) dlnM

        Pair integral with first-order halo bias included, matching the
        class_sz 2h convention.  Used as a building block for ``bk_2h``.

        Future generalisation: replace ``u1 * u2`` with a ``_fourier_2pt``
        variant that handles different k values when specialised 2-point kernels
        (HOD, CIB) are needed.
        """
        hm = halo_model
        m, z_arr = hm.m_grid, jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(
            hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses),
            (len(m), len(z_arr)),
        )
        bias_w = jnp.reshape(
            hm.halo_bias.bias(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses, order=1),
            (len(m), len(z_arr)),
        )
        total_weights = dndlnm * bias_w * w[:, None]  # (Nm, Nz)

        k1s, k2s = jnp.atleast_1d(k1), jnp.atleast_1d(k2)
        u1 = jnp.reshape(p1.fourier(hm, k1s, m, z_arr), (len(k1s), len(m), len(z_arr)))
        u2 = jnp.reshape(p2.fourier(hm, k2s, m, z_arr), (len(k2s), len(m), len(z_arr)))

        integral = jnp.sum(u1 * u2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)

        n_min, b1_min, _ = hm._counter_terms(z_arr)
        correction = n_min[None, :] * b1_min[None, :] * u1[:, 0, :] * u2[:, 0, :]

        return jnp.squeeze(integral + hm.hm_consistency * correction)

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def bk_1h(self, halo_model, profile1, profile2, profile3, k1, k2, k3, z, k_damp=0.01):
        """
        1-halo bispectrum term.

        .. math::

            B_{1h}(k_1, k_2, k_3, z) =
            \\int \\frac{dn}{d\\ln M}\\, u_1(k_1 \\mid M, z)\\,
            u_2(k_2 \\mid M, z)\\, u_3(k_3 \\mid M, z)\\, d\\ln M

        where :math:`u_i` are the Fourier-space profiles (first moments).
        For profiles with non-trivial intra-halo occupancy statistics, replace
        the triple product with a ``_fourier_3pt`` helper.

        A low-k suppression factor :math:`1 - e^{-(k_{\\min}/k_{\\mathrm{damp}})^2}`
        is applied at the smallest wavenumber of the triplet.

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2, profile3 : HaloProfile
            The three halo profiles at wavenumbers k1, k2, k3 respectively.
        k1, k2, k3 : float
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping wavenumber for the low-k suppression.

        Returns
        -------
        array
            1-halo bispectrum in :math:`\\mathrm{Mpc}^6`, squeezed.
        """
        hm = halo_model
        m, z_arr = hm.m_grid, jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(
            hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses),
            (len(m), len(z_arr)),
        )
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        k1s, k2s, k3s = jnp.atleast_1d(k1), jnp.atleast_1d(k2), jnp.atleast_1d(k3)
        u1 = jnp.reshape(profile1.fourier(hm, k1s, m, z_arr), (len(k1s), len(m), len(z_arr)))
        u2 = jnp.reshape(profile2.fourier(hm, k2s, m, z_arr), (len(k2s), len(m), len(z_arr)))
        u3 = jnp.reshape(profile3.fourier(hm, k3s, m, z_arr), (len(k3s), len(m), len(z_arr)))

        triple = u1 * u2 * u3  # (1, Nm, Nz)
        bk1h = jnp.sum(triple * total_weights[None, :, :], axis=1)  # (1, Nz)

        n_min, _, _ = hm._counter_terms(z_arr)
        correction = n_min[None, :] * u1[:, 0, :] * u2[:, 0, :] * u3[:, 0, :]
        bk1h = bk1h + hm.hm_consistency * correction

        k_min = jnp.minimum(jnp.minimum(jnp.asarray(k1), jnp.asarray(k2)), jnp.asarray(k3))
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k_min / jnp.where(mask, k_damp, 1.0))**2), 1.0)

        # Reshape damping to (N_k, 1) so it broadcasts correctly over (N_k, N_z)
        damping_bc = jnp.reshape(damping, jnp.shape(damping) + (1,))
        return jnp.squeeze(bk1h * damping_bc)

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def bk_2h(self, halo_model, profile1, profile2, profile3, k1, k2, k3, z):
        """
        2-halo bispectrum term.

        .. math::

            B_{2h} = P_{\\mathrm{lin}}(k_1)\\, I^{(1)}_1(k_1)\\, J^{(1)}_{23}(k_2,k_3)
                   + P_{\\mathrm{lin}}(k_2)\\, I^{(1)}_2(k_2)\\, J^{(1)}_{13}(k_1,k_3)
                   + P_{\\mathrm{lin}}(k_3)\\, I^{(1)}_3(k_3)\\, J^{(1)}_{12}(k_1,k_2)

        where :math:`I^{(1)}_i = \\int dn/d\\ln M\\, b_1 u_i` and
        :math:`J^{(1)}_{ij} = \\int dn/d\\ln M\\, b_1 u_i u_j` (bias included in
        both the single and pair integrals, matching class_sz convention).

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2, profile3 : HaloProfile
        k1, k2, k3 : float
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            2-halo bispectrum in :math:`\\mathrm{Mpc}^6`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)

        I1 = hm._I(profile1, k1, z, bias_order=1)
        I2 = hm._I(profile2, k2, z, bias_order=1)
        I3 = hm._I(profile3, k3, z, bias_order=1)

        J23 = self._pair_integral(hm, profile2, profile3, k2, k3, z)
        J13 = self._pair_integral(hm, profile1, profile3, k1, k3, z)
        J12 = self._pair_integral(hm, profile1, profile2, k1, k2, z)

        P1 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k1), z_arr, linear=True))
        P2 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k2), z_arr, linear=True))
        P3 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k3), z_arr, linear=True))

        return jnp.squeeze(P1 * I1 * J23 + P2 * I2 * J13 + P3 * I3 * J12)

    # ------------------------------------------------------------------
    # 3-halo term
    # ------------------------------------------------------------------

    def bk_3h(self, halo_model, profile1, profile2, profile3, k1, k2, k3, z):
        """
        3-halo bispectrum term.

        .. math::

            B_{3h} = B_{\\mathrm{tree}}(k_1,k_2,k_3)\\,
                     I^{(1)}_1 I^{(1)}_2 I^{(1)}_3
                   + I^{(2)}_1 I^{(1)}_2 I^{(1)}_3 P_2 P_3
                   + I^{(1)}_1 I^{(2)}_2 I^{(1)}_3 P_1 P_3
                   + I^{(1)}_1 I^{(1)}_2 I^{(2)}_3 P_1 P_2

        :math:`B_{\\mathrm{tree}}` uses the correct SPT F2 kernel
        (:math:`\\hat k_i \\cdot \\hat k_j = (k_3^2-k_1^2-k_2^2)/(2k_1k_2)`).

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2, profile3 : HaloProfile
        k1, k2, k3 : float
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            3-halo bispectrum in :math:`\\mathrm{Mpc}^6`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)

        I1_b1 = hm._I(profile1, k1, z, bias_order=1)
        I2_b1 = hm._I(profile2, k2, z, bias_order=1)
        I3_b1 = hm._I(profile3, k3, z, bias_order=1)
        I1_b2 = hm._I(profile1, k1, z, bias_order=2)
        I2_b2 = hm._I(profile2, k2, z, bias_order=2)
        I3_b2 = hm._I(profile3, k3, z, bias_order=2)

        P1 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k1), z_arr, linear=True))
        P2 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k2), z_arr, linear=True))
        P3 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k3), z_arr, linear=True))

        # Tree-level SPT bispectrum with correct cosine convention:
        # mu_ij = (k_k^2 - k_i^2 - k_j^2) / (2 k_i k_j)  [opposite-side law]
        k1a, k2a, k3a = jnp.asarray(k1), jnp.asarray(k2), jnp.asarray(k3)
        mu12 = _mu(k1a, k2a, k3a)
        mu23 = _mu(k2a, k3a, k1a)
        mu31 = _mu(k3a, k1a, k2a)
        B_tree = (
            2.0 * _F2(k1a, k2a, mu12) * P1 * P2
            + 2.0 * _F2(k2a, k3a, mu23) * P2 * P3
            + 2.0 * _F2(k3a, k1a, mu31) * P3 * P1
        )

        tree_term = B_tree * I1_b1 * I2_b1 * I3_b1

        # Quadratic-bias corrections (no 1/2 prefactor — b2 convention matches class_sz)
        b2_term = (
            I1_b2 * I2_b1 * I3_b1 * P2 * P3
            + I1_b1 * I2_b2 * I3_b1 * P1 * P3
            + I1_b1 * I2_b1 * I3_b2 * P1 * P2
        )

        return jnp.squeeze(tree_term + b2_term)