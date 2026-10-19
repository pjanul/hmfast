"""
Unit tests for the 3D halo-model statistics: Pk, Bk, Tk and corr_3d.

Internal contracts only -- output shapes, exact identities between terms, symmetries,
large-scale limits and input validation. Comparisons against CCL and class_sz are in
tests/benchmarks/benchmark_stats_3d.py, jittability in test_jax_transforms.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.stats import Bk, Pk, Tk, corr_3d

from .._shared import GNFW, K_GRID, K_GRID_BT, NFW, Z_GRID, stats_halo_model

HM = stats_halo_model()
# Same shape as K_GRID, so calls on it reuse the compiled kernels.
K_LOW = jnp.geomspace(1e-4, 1e-3, K_GRID.shape[0])

PK = Pk()
PK_1H, PK_2H = Pk(include_2h=False), Pk(include_1h=False)
BK = Bk()
TK = Tk()

K_VALS = {"scalar": jnp.array(0.1), "array1": jnp.array([0.1]), "arrayN": jnp.geomspace(0.05, 1.0, 5)}
Z_VALS = {"scalar": jnp.array(0.5), "array1": jnp.array([0.5]), "arrayM": jnp.array([0.0, 0.5, 1.0, 2.0])}
SHAPE_MATRIX_2D = [
    ("scalar", "scalar", ()),
    ("array1", "array1", ()),
    ("arrayN", "scalar", (5,)),
    ("scalar", "arrayM", (4,)),
    ("arrayN", "arrayM", (5, 4)),
]


class TestPk:
    # pk_1h/pk_2h/pk_tot follow the (k, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("method", ["pk_1h", "pk_2h", "pk_tot"])
    @pytest.mark.parametrize("k_key,z_key,expected_shape", SHAPE_MATRIX_2D)
    def test_shape_matrix(self, method, k_key, z_key, expected_shape):
        out = jax.eval_shape(lambda: getattr(PK, method)(HM, K_VALS[k_key], Z_VALS[z_key], NFW))
        assert out.shape == expected_shape

    # profile2=None is the auto spectrum of profile1.
    def test_profile2_defaults_to_profile1(self):
        np.testing.assert_allclose(PK.pk_tot(HM, K_GRID, Z_GRID, NFW), PK.pk_tot(HM, K_GRID, Z_GRID, NFW, NFW), rtol=1e-14)

    # A cross spectrum does not depend on which profile is passed first.
    @pytest.mark.parametrize("method", ["pk_1h", "pk_2h"])
    def test_cross_spectrum_symmetric_in_profiles(self, method):
        f = getattr(PK, method)
        np.testing.assert_allclose(f(HM, K_GRID, Z_GRID, NFW, GNFW), f(HM, K_GRID, Z_GRID, GNFW, NFW), rtol=1e-12)

    # With alpha_smooth=1, pk_tot is exactly pk_1h + pk_2h, and include_1h/include_2h drop a term.
    def test_pk_tot_composition(self):
        p1h, p2h = PK.pk_1h(HM, K_GRID, Z_GRID, NFW), PK.pk_2h(HM, K_GRID, Z_GRID, NFW)
        np.testing.assert_allclose(PK.pk_tot(HM, K_GRID, Z_GRID, NFW), p1h + p2h, rtol=1e-14)
        np.testing.assert_allclose(PK_1H.pk_tot(HM, K_GRID, Z_GRID, NFW), p1h, rtol=1e-14)
        np.testing.assert_allclose(PK_2H.pk_tot(HM, K_GRID, Z_GRID, NFW), p2h, rtol=1e-14)

    # alpha_smooth != 1 smooths the transition: pk_tot lies between the larger term and the plain sum.
    def test_alpha_smooth_bounds(self):
        p1h, p2h = PK.pk_1h(HM, K_GRID, Z_GRID, NFW), PK.pk_2h(HM, K_GRID, Z_GRID, NFW)
        p = Pk(alpha_smooth=2.0).pk_tot(HM, K_GRID, Z_GRID, NFW)
        assert jnp.all(p >= jnp.maximum(p1h, p2h) * (1 - 1e-12))
        assert jnp.all(p <= (p1h + p2h) * (1 + 1e-12))

    # pk_1h is the undamped integral times 1 - exp(-(k/k_damp)^2), and k_damp=0 switches damping off.
    def test_k_damp_factor(self):
        k_damp = 0.01
        undamped = Pk(k_damp=0.0).pk_1h(HM, K_LOW, Z_GRID, NFW)
        damped = Pk(k_damp=k_damp).pk_1h(HM, K_LOW, Z_GRID, NFW)
        factor = 1.0 - jnp.exp(-((K_LOW / k_damp) ** 2))
        np.testing.assert_allclose(damped, undamped * factor[:, None], rtol=1e-12)

    # On large scales the 2-halo matter term recovers the linear power spectrum (the consistency counterterm forces I_1^1 -> 1).
    def test_pk_2h_recovers_linear_on_large_scales(self):
        p2h = PK.pk_2h(HM, K_LOW, Z_GRID, NFW)
        plin = HM.cosmology.pk(K_LOW, Z_GRID, linear=True)
        np.testing.assert_allclose(p2h, plin, rtol=1e-5)

    # The matter auto spectrum is positive on every term.
    @pytest.mark.parametrize("method", ["pk_1h", "pk_2h"])
    def test_matter_auto_positive(self, method):
        assert jnp.all(getattr(PK, method)(HM, K_GRID, Z_GRID, NFW) > 0)


class TestBk:
    # Non-positive wavenumbers and cosines outside [-1, 1] are rejected before any compile.
    @pytest.mark.parametrize("method", ["bk_1h", "bk_2h", "bk_3h", "bk_tot"])
    @pytest.mark.parametrize("k1,k2,mu", [(0.0, 0.1, 0.5), (0.1, -0.1, 0.5), (0.1, 0.1, 1.5)])
    def test_rejects_invalid_triangle(self, method, k1, k2, mu):
        with pytest.raises(ValueError):
            getattr(BK, method)(HM, k1, k2, mu, Z_GRID, NFW)

    # bk_* follow the (k, z) broadcast-then-squeeze convention, with k1 and k2 paired element-wise.
    @pytest.mark.parametrize("method", ["bk_1h", "bk_2h", "bk_3h", "bk_tot"])
    @pytest.mark.parametrize("k_key,z_key,expected_shape", SHAPE_MATRIX_2D)
    def test_shape_matrix(self, method, k_key, z_key, expected_shape):
        k = K_VALS[k_key]
        out = jax.eval_shape(lambda: getattr(BK, method)(HM, k, 2 * k, -0.3, Z_VALS[z_key], NFW))
        assert out.shape == expected_shape

    # Every term of an auto bispectrum is symmetric under exchanging k1 and k2 at fixed angle.
    @pytest.mark.parametrize("method", ["bk_1h", "bk_2h", "bk_3h"])
    def test_symmetric_under_k1_k2_exchange(self, method):
        k1, k2 = K_GRID_BT, K_GRID_BT[::-1]
        f = getattr(BK, method)
        np.testing.assert_allclose(f(HM, k1, k2, -0.3, Z_GRID, NFW), f(HM, k2, k1, -0.3, Z_GRID, NFW), rtol=1e-10)

    # bk_tot restricted to one term is exactly that term.
    def test_bk_tot_single_term(self):
        only_1h = Bk(include_2h=False, include_3h=False)
        np.testing.assert_allclose(
            only_1h.bk_tot(HM, K_GRID_BT, K_GRID_BT, -0.3, Z_GRID, NFW), BK.bk_1h(HM, K_GRID_BT, K_GRID_BT, -0.3, Z_GRID, NFW),
            rtol=1e-14,
        )


class TestTk:
    # tk_* return (N_u, N_v, N_z) with singleton dimensions squeezed.
    @pytest.mark.parametrize("method", ["tk_1h", "tk_2h", "tk_3h", "tk_4h", "tk_tot"])
    @pytest.mark.parametrize(
        "ku,kv,z,expected_shape",
        [
            (jnp.array(0.1), jnp.array(0.2), jnp.array(0.5), ()),
            (jnp.geomspace(0.05, 1.0, 3), jnp.geomspace(0.05, 1.0, 5), jnp.array(0.5), (3, 5)),
            (jnp.geomspace(0.05, 1.0, 3), jnp.array(0.2), jnp.array([0.0, 1.0]), (3, 2)),
            (jnp.geomspace(0.05, 1.0, 3), jnp.geomspace(0.05, 1.0, 5), jnp.array([0.0, 1.0]), (3, 5, 2)),
        ],
    )
    def test_shape_matrix(self, method, ku, kv, z, expected_shape):
        out = jax.eval_shape(lambda: getattr(TK, method)(HM, ku, kv, z, NFW))
        assert out.shape == expected_shape

    # With legs (A, B | A, B) the u and v pairs carry the same profiles, so every term is symmetric under k_u <-> k_v;
    # a leg evaluated at the wrong one of k_u, k_v breaks this.
    @pytest.mark.parametrize("method", ["tk_1h", "tk_2h", "tk_3h", "tk_4h"])
    def test_mixed_legs_symmetric_under_ku_kv_exchange(self, method):
        k = K_GRID_BT
        t = np.asarray(getattr(TK, method)(HM, k, k, Z_GRID, NFW, GNFW, NFW, GNFW))
        np.testing.assert_allclose(t, np.swapaxes(t, 0, 1), rtol=1e-10)

    # tk_tot restricted to one term is exactly that term.
    def test_tk_tot_single_term(self):
        only_1h = Tk(include_2h=False, include_3h=False, include_4h=False)
        np.testing.assert_allclose(
            only_1h.tk_tot(HM, K_GRID_BT, K_GRID_BT, Z_GRID, NFW), TK.tk_1h(HM, K_GRID_BT, K_GRID_BT, Z_GRID, NFW), rtol=1e-14
        )


# A Gaussian P(k) = exp(-k^2 R^2 / 2) has xi(r) = exp(-r^2 / 2R^2) / ((2 pi)^{3/2} R^3).
R_GAUSS = np.array([1.0, 2.0])
K_GAUSS = np.geomspace(1e-4, 1e2, 512)
R_EVAL = jnp.geomspace(0.3, 3.0, 8)


def _gaussian_pk():
    return jnp.exp(-0.5 * (K_GAUSS[:, None] * R_GAUSS[None, :]) ** 2)


def _gaussian_xi(r):
    return np.exp(-0.5 * (np.asarray(r)[:, None] / R_GAUSS[None, :]) ** 2) / ((2 * np.pi) ** 1.5 * R_GAUSS**3)


class TestCorr3d:
    # corr_3d of a Gaussian power spectrum matches its analytic transform, to 1e-3 of each column's peak.
    def test_gaussian_transform_pair(self):
        xi, truth = np.asarray(corr_3d(HM.cosmology, K_GAUSS, _gaussian_pk(), R_EVAL, extrapolate=False)), _gaussian_xi(R_EVAL)
        assert np.all(np.abs(xi - truth) <= 1e-3 * truth.max(axis=0))

    # Trailing axes are transformed independently: a (N_k, 2) input equals two (N_k,) calls.
    def test_trailing_axes_independent(self):
        pk = _gaussian_pk()
        batched = corr_3d(HM.cosmology, K_GAUSS, pk, R_EVAL, extrapolate=False)
        for i in range(pk.shape[1]):
            np.testing.assert_allclose(batched[:, i], corr_3d(HM.cosmology, K_GAUSS, pk[:, i], R_EVAL, extrapolate=False), rtol=1e-12)

    # The k grid must be concrete, 1D, positive and log-uniform, and match pk's leading axis.
    @pytest.mark.parametrize(
        "k,pk",
        [
            (np.linspace(1e-3, 10.0, 32), np.ones(32)),
            (np.geomspace(1e-3, 10.0, 32), np.ones(31)),
            (-np.geomspace(1e-3, 10.0, 32), np.ones(32)),
            (np.geomspace(1e-3, 10.0, 32)[None, :], np.ones(32)),
        ],
        ids=["linear_grid", "length_mismatch", "negative_k", "2d_grid"],
    )
    def test_rejects_bad_grid(self, k, pk):
        with pytest.raises(ValueError):
            corr_3d(HM.cosmology, k, pk, R_EVAL)
