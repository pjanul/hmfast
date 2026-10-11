"""
Unit tests for the projected statistics: cl, cl_linbias and corr_angular.

Internal contracts only -- output shapes, exact identities, symmetries, the Limber/non-Limber
split, the large-scale limit of the halo model and input validation. Comparisons against CCL
and SwiftCl are in tests/benchmarks/benchmark_stats_projected.py, jittability in
test_jax_transforms.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.stats import Pk, cl, cl_linbias, corr_angular
from hmfast.stats.projected_cl import _f_ell
from hmfast.tracers import GalaxyTracer

from .._shared import GAL_TRACER_BIASED, GLENS_TRACER, HOD, L_GRID, N_Z, Z_BIAS, Z_RANGE, stats_halo_model

HM = stats_halo_model()
COSMO = HM.cosmology
COSMO_NO_EXTRAP = COSMO.update(extrapolate_k=False)
PK, PK_1H, PK_2H = Pk(), Pk(include_2h=False), Pk(include_1h=False)
# Same shape as L_GRID, so calls on it reuse the compiled kernels.
L_LOW = jnp.geomspace(2.0, 50.0, L_GRID.shape[0])

L_VALS = {"scalar": jnp.array(100.0), "array1": jnp.array([100.0]), "arrayN": jnp.geomspace(20.0, 1000.0, 4)}
L_SHAPES = [("scalar", ()), ("array1", ()), ("arrayN", (4,))]


def _galaxy_with_bias(scale):
    """GAL_TRACER_BIASED with its bias multiplied by `scale` (same pytree structure, so no recompile)."""
    return GalaxyTracer(profile=HOD, bias=(Z_BIAS, scale * (1.0 + 0.5 * Z_BIAS)))


class TestCl:
    # cl follows the l broadcast-then-squeeze convention.
    @pytest.mark.parametrize("l_key,expected_shape", L_SHAPES)
    def test_shape_matrix(self, l_key, expected_shape):
        out = jax.eval_shape(lambda: cl(PK, HM, L_VALS[l_key], GLENS_TRACER, GLENS_TRACER, z_range=Z_RANGE, n_z=N_Z))
        assert out.shape == expected_shape

    # The 1-halo and 2-halo spectra add up to the total: the projection is linear in P(k).
    def test_halo_terms_add_up(self):
        c = lambda pk: cl(pk, HM, L_GRID, GLENS_TRACER, None, z_range=Z_RANGE, n_z=N_Z)  # noqa: E731
        np.testing.assert_allclose(c(PK_1H) + c(PK_2H), c(PK), rtol=1e-12)

    # Below l_limber only the 2-halo term goes non-Limber; the 1-halo term stays the Limber one at every multipole.
    def test_l_limber_keeps_1h(self):
        c = lambda pk, **kw: cl(pk, HM, L_GRID, GLENS_TRACER, None, z_range=Z_RANGE, n_z=N_Z, **kw)  # noqa: E731
        np.testing.assert_allclose(c(PK, l_limber=100.0) - c(PK_2H, l_limber=100.0), c(PK_1H), rtol=1e-12)

    # On large scales the 2-halo matter spectrum reduces to the linear-bias spectrum with b = 1.
    def test_2h_matches_linear_bias_on_large_scales(self):
        c_2h = cl(PK_2H, HM, L_LOW, GLENS_TRACER, None, z_range=Z_RANGE, n_z=N_Z)
        c_lin = cl_linbias(COSMO, L_LOW, GLENS_TRACER, None, z_range=Z_RANGE, n_z=N_Z, linear=True)
        np.testing.assert_allclose(c_2h, c_lin, rtol=1e-3)


class TestClLinbias:
    # cl_linbias follows the l broadcast-then-squeeze convention.
    @pytest.mark.parametrize("l_key,expected_shape", L_SHAPES)
    def test_shape_matrix(self, l_key, expected_shape):
        out = jax.eval_shape(lambda: cl_linbias(COSMO, L_VALS[l_key], GAL_TRACER_BIASED, GLENS_TRACER, z_range=Z_RANGE, n_z=N_Z))
        assert out.shape == expected_shape

    # A cross spectrum does not depend on which tracer is passed first.
    def test_cross_symmetric_in_tracers(self):
        np.testing.assert_allclose(
            cl_linbias(COSMO, L_GRID, GAL_TRACER_BIASED, GLENS_TRACER, z_range=Z_RANGE, n_z=N_Z),
            cl_linbias(COSMO, L_GRID, GLENS_TRACER, GAL_TRACER_BIASED, z_range=Z_RANGE, n_z=N_Z),
            rtol=1e-12,
        )

    # Doubling a density-only galaxy bias quadruples its auto spectrum and doubles its cross with lensing.
    def test_bias_scaling(self):
        g1, g2 = _galaxy_with_bias(1.0), _galaxy_with_bias(2.0)
        auto = lambda g: cl_linbias(COSMO, L_GRID, g, None, z_range=Z_RANGE, n_z=N_Z)  # noqa: E731
        cross = lambda g: cl_linbias(COSMO, L_GRID, g, GLENS_TRACER, z_range=Z_RANGE, n_z=N_Z)  # noqa: E731
        np.testing.assert_allclose(auto(g2), 4.0 * auto(g1), rtol=1e-12)
        np.testing.assert_allclose(cross(g2), 2.0 * cross(g1), rtol=1e-12)

    # Multipoles at or above l_limber use the Limber engine exactly; those below it go through the non-Limber one.
    def test_limber_split(self):
        l_limber = 100.0
        limber = np.asarray(cl_linbias(COSMO, L_GRID, GAL_TRACER_BIASED, None, z_range=Z_RANGE, n_z=N_Z))
        split = np.asarray(cl_linbias(COSMO, L_GRID, GAL_TRACER_BIASED, None, z_range=Z_RANGE, n_z=N_Z, l_limber=l_limber))
        high = np.asarray(L_GRID) >= l_limber
        assert high.any() and (~high).any()
        np.testing.assert_allclose(split[high], limber[high], rtol=1e-12)
        assert np.all(np.isfinite(split[~high])) and np.all(split[~high] != limber[~high])


# A Gaussian C_l = exp(-l^2 sigma^2 / 2) has the flat-sky w(theta) = exp(-theta^2 / 2 sigma^2) / (2 pi sigma^2).
SIGMA = 0.01  # rad
L_GAUSS = np.geomspace(1.0, 1e4, 512)
THETA_EVAL = jnp.geomspace(0.05, 1.5, 8)  # deg


def _gaussian_cl():
    return jnp.exp(-0.5 * (L_GAUSS * SIGMA) ** 2)


class TestCorrAngular:
    # corr_angular NN of a Gaussian C_l matches its analytic flat-sky transform, to 1e-3 of the peak.
    def test_gaussian_transform_pair(self):
        w = np.asarray(corr_angular(COSMO_NO_EXTRAP, L_GAUSS, _gaussian_cl(), THETA_EVAL, type="NN"))
        theta = np.deg2rad(np.asarray(THETA_EVAL))
        truth = np.exp(-0.5 * (theta / SIGMA) ** 2) / (2 * np.pi * SIGMA**2)
        assert np.all(np.abs(w - truth) <= 1e-3 * truth.max())

    # GG+ uses the same J_0 kernel as NN.
    def test_gg_plus_equals_nn(self):
        nn = corr_angular(COSMO_NO_EXTRAP, L_GAUSS, _gaussian_cl(), THETA_EVAL, type="NN")
        gp = corr_angular(COSMO_NO_EXTRAP, L_GAUSS, _gaussian_cl(), THETA_EVAL, type="GG+")
        np.testing.assert_allclose(gp, nn, rtol=1e-14)

    # corr_angular follows the theta broadcast-then-squeeze convention, keeping trailing axes of the input.
    @pytest.mark.parametrize(
        "theta,trailing,expected_shape",
        [(jnp.array(0.5), (), ()), (THETA_EVAL, (), (8,)), (THETA_EVAL, (3,), (8, 3))],
    )
    def test_shape_matrix(self, theta, trailing, expected_shape):
        c = jnp.ones(L_GAUSS.shape + trailing)
        assert jax.eval_shape(lambda: corr_angular(COSMO, L_GAUSS, c, theta)).shape == expected_shape

    # Unknown spin combinations are rejected.
    def test_rejects_unknown_type(self):
        with pytest.raises(ValueError):
            corr_angular(COSMO, L_GAUSS, _gaussian_cl(), THETA_EVAL, type="GN")

    # The l grid must be log-uniform.
    def test_rejects_linear_l_grid(self):
        with pytest.raises(ValueError):
            corr_angular(COSMO, np.linspace(1.0, 1e4, 512), _gaussian_cl(), THETA_EVAL)


class TestAngularPrefactor:
    # The angular prefactor exists only for der_angles 0, 1 and 2.
    def test_rejects_unknown_der_angles(self):
        with pytest.raises(ValueError):
            _f_ell(jnp.array([10.0]), 3)
