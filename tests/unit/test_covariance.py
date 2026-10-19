"""
Unit tests for the angular covariances: cov_g, cov_cng, cov_ssc and sigma2_b_disc.

Internal contracts only -- output shapes, symmetries, f_sky scaling and additivity over
trispectrum terms. The connected covariance is sourced from single-term Tk objects, whose
compiles are far cheaper than the full trispectrum's. Comparisons against CCL are in
tests/benchmarks/benchmark_covariance.py, jittability in test_jax_transforms.py.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.stats import Pk, Tk, cov_cng, cov_g, cov_ssc, sigma2_b_disc

from .._shared import GAL_TRACER_BIASED, GLENS_TRACER, L_GRID, N_Z, TK, Z_GRID, stats_halo_model

HM = stats_halo_model()
PK = Pk()
TK_1H = Tk(include_2h=False, include_3h=False, include_4h=False)
TK_4H = Tk(include_1h=False, include_2h=False, include_3h=False)
TK_1H_4H = Tk(include_2h=False, include_3h=False)
F_SKY = 0.4

# Each l input as (value, its contribution to the output shape); one entry is squeezed away.
L_INPUTS = {
    "scalar": (jnp.array(200.0), ()),
    "array1": (jnp.array([200.0]), ()),
    "array3": (jnp.geomspace(20.0, 1000.0, 3), (3,)),
    "array5": (jnp.geomspace(20.0, 1000.0, 5), (5,)),
}
L_SHAPES = [
    pytest.param(L_INPUTS[a][0], L_INPUTS[b][0], L_INPUTS[a][1] + L_INPUTS[b][1], id=f"{a}-{b}")
    for a in L_INPUTS for b in L_INPUTS
]


def _cng(tk, t1, t2, t3, t4, l1=L_GRID, l2=L_GRID, f_sky=F_SKY):
    return np.asarray(cov_cng(tk, HM, l1, l2, t1, t2, t3, t4, n_z=N_Z, f_sky=f_sky))


def _ssc(t1, t2, t3, t4, f_sky=F_SKY):
    return np.asarray(cov_ssc(PK, HM, L_GRID, L_GRID, t1, t2, t3, t4, n_z=N_Z, f_sky=f_sky))


# Random positive-definite 4x4 C_l matrices on a few multipoles, spanning orders of magnitude.
_RNG = np.random.default_rng(0)
L_G = jnp.array([2.0, 30.0, 500.0])
_A = _RNG.normal(size=(3, 4, 4)) * np.logspace(-1, 1, 4)
C_G = jnp.asarray((_A @ np.swapaxes(_A, 1, 2) + 0.1 * np.eye(4)) * np.asarray(L_G)[:, None, None] ** -2)


def _g(a, b, c, d, **kw):
    # Cov(C^{ab}, C^{cd}) from the tracer indices into C_G.
    return np.asarray(cov_g(L_G, C_G[:, a, c], C_G[:, b, d], C_G[:, a, d], C_G[:, b, c], **kw))


class TestCovG:
    @pytest.mark.parametrize("l,expected_shape", [(200.0, ()), (jnp.array([200.0]), ()), (L_G, (3, 3))])
    def test_shape_matrix(self, l, expected_shape):
        assert cov_g(l, 1.0).shape == expected_shape

    # The Gaussian covariance couples no two different multipoles.
    def test_diagonal(self):
        cov = _g(0, 1, 2, 3)
        np.testing.assert_array_equal(cov - np.diag(np.diag(cov)), 0.0)

    # Omitted spectra default so that one spectrum gives the auto covariance 2 C^2 / ((2l + 1) f_sky dl).
    def test_auto_default(self):
        c, dl = C_G[:, 0, 0], jnp.array([1.0, 5.0, 21.0])
        expected = np.diag(2 * c**2 / ((2 * L_G + 1) * F_SKY * dl))
        np.testing.assert_allclose(cov_g(L_G, c, f_sky=F_SKY, delta_l=dl), expected, rtol=1e-14)

    # Exchanging the two C_l pairs leaves the covariance unchanged.
    def test_pair_exchange(self):
        np.testing.assert_allclose(_g(0, 1, 2, 3), _g(2, 3, 0, 1), rtol=1e-14)

    # Matches the covariance of full-sky Gaussian fields, (E[y_a y_b y_c y_d] - C^{ab} C^{cd}) / (2l + 1),
    # with the fourth moment from 3-point Gauss-Hermite quadrature, exact for this degree.
    def test_gaussian_fourth_moment(self):
        x, w = np.polynomial.hermite_e.hermegauss(3)
        z = np.array(list(itertools.product(x, repeat=4)))  # (81, 4) standard-normal nodes
        wz = np.prod(np.array(list(itertools.product(w, repeat=4))), axis=1) / (2 * np.pi) ** 2
        y = np.einsum("lij,nj->lni", np.linalg.cholesky(np.asarray(C_G)), z)  # (N_l, 81, 4) with covariance C_G
        M4 = np.einsum("n,lna,lnb,lnc,lnd->labcd", wz, y, y, y, y)
        for a, b, c, d in itertools.product(range(4), repeat=4):
            exact = (M4[:, a, b, c, d] - C_G[:, a, b] * C_G[:, c, d]) / (2 * L_G + 1)
            np.testing.assert_allclose(np.diag(_g(a, b, c, d)), exact, rtol=1e-10)


class TestCovCng:
    # cov_cng follows the (l1, l2) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("l1,l2,expected_shape", L_SHAPES)
    def test_shape_matrix(self, l1, l2, expected_shape):
        g = GLENS_TRACER
        out = jax.eval_shape(lambda: cov_cng(TK, HM, l1, l2, g, g, g, g, n_z=N_Z, f_sky=F_SKY))
        assert out.shape == expected_shape

    # An auto covariance on a shared l grid is a symmetric matrix.
    def test_auto_symmetric(self):
        g = GLENS_TRACER
        cov = _cng(TK_1H, g, g, g, g)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12)

    # Exchanging the two C_l pairs transposes the covariance.
    def test_pair_exchange_transposes(self):
        g, k = GAL_TRACER_BIASED, GLENS_TRACER
        l1, l2 = L_GRID, L_GRID[:4]
        np.testing.assert_allclose(_cng(TK_1H, g, g, k, k, l1, l2), _cng(TK_1H, k, k, g, g, l2, l1).T, rtol=1e-12)

    # A single l2 gives the matching column of a wider calculation, not a broadcast across l1.
    def test_single_l2_is_a_column(self):
        g = GLENS_TRACER
        np.testing.assert_allclose(
            _cng(TK_1H, g, g, g, g, l2=L_GRID[0]), _cng(TK_1H, g, g, g, g, l2=L_GRID[:2])[:, 0], rtol=1e-12
        )

    # The connected covariance scales exactly as 1 / f_sky.
    def test_inverse_f_sky_scaling(self):
        g = GLENS_TRACER
        np.testing.assert_allclose(_cng(TK_1H, g, g, g, g, f_sky=0.2), 2.0 * _cng(TK_1H, g, g, g, g, f_sky=0.4), rtol=1e-12)

    # The covariance is linear in the trispectrum: per-term covariances add up to the combined one.
    def test_additive_over_trispectrum_terms(self):
        g = GLENS_TRACER
        np.testing.assert_allclose(
            _cng(TK_1H, g, g, g, g) + _cng(TK_4H, g, g, g, g), _cng(TK_1H_4H, g, g, g, g), rtol=1e-12
        )


class TestCovSsc:
    # cov_ssc follows the (l1, l2) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("l1,l2,expected_shape", L_SHAPES)
    def test_shape_matrix(self, l1, l2, expected_shape):
        g = GLENS_TRACER
        out = jax.eval_shape(lambda: cov_ssc(PK, HM, l1, l2, g, g, g, g, n_z=N_Z, f_sky=F_SKY))
        assert out.shape == expected_shape

    # An auto covariance on a shared l grid is a symmetric matrix.
    def test_auto_symmetric(self):
        g = GLENS_TRACER
        cov = _ssc(g, g, g, g)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12)

    # A larger footprint averages down the super-sample mode, so the covariance shrinks with f_sky.
    def test_decreases_with_f_sky(self):
        g = GLENS_TRACER
        assert np.all(np.abs(_ssc(g, g, g, g, f_sky=0.4)) < np.abs(_ssc(g, g, g, g, f_sky=0.1)))


class TestSigma2BDisc:
    # sigma2_b_disc is positive, keeps z's shape and squeezes a scalar z.
    @pytest.mark.parametrize("z,expected_shape", [(jnp.array(0.5), ()), (Z_GRID, Z_GRID.shape)])
    def test_positive_with_z_shape(self, z, expected_shape):
        s2b = sigma2_b_disc(HM.cosmology, z, f_sky=F_SKY)
        assert s2b.shape == expected_shape and jnp.all(s2b > 0)

    # A larger footprint has a smaller background variance, and the variance falls with redshift as structure shrinks.
    def test_decreases_with_f_sky_and_redshift(self):
        small, large = sigma2_b_disc(HM.cosmology, Z_GRID, f_sky=0.1), sigma2_b_disc(HM.cosmology, Z_GRID, f_sky=0.4)
        assert jnp.all(large < small)
        assert jnp.all(jnp.diff(large) < 0)
