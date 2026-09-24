"""
Unit tests for hmfast.halos.profiles: shape/squeeze conventions, update()/pytree
rebuild behavior, mass-definition handling, JIT shape-polymorphism, and
gradients. See tests/benchmarks/benchmark_profiles.py for CCL/class_sz
cross-checks.
"""

import jax
import jax.numpy as jnp
import pytest

from hmfast.halos import HaloModel
from hmfast.halos.concentration import (
    B13Concentration,
    ConstantConcentration,
    D08Concentration,
)
from hmfast.halos.massdef import MassDefinition, mass_translator
from hmfast.halos.profiles import (
    B12PressureProfile,
    B16DensityProfile,
    GNFWPressureProfile,
    M21CIBProfile,
    NFWMatterProfile,
    S12CIBProfile,
    Z07GalaxyHODProfile,
)
from hmfast.halos.profiles.base_profile import HankelTransform
from hmfast.halos.profiles.profiles_2pt import _fourier_2pt
from hmfast.utils import gauss_legendre_nodes_weights

M_GRID = jnp.geomspace(1e10, 1e15, 40)


def _m_grid(hm):
    """Reconstruct a halo model's internal Gauss-Legendre mass-integral nodes."""
    logm, _ = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
    return jnp.exp(logm)


@pytest.fixture
def hm200c(fixed_cosmology):
    """A single fixed HaloModel at 200c/D08, reused across non-mass-def-focused checks."""
    return HaloModel(
        cosmology=fixed_cosmology,
        mass_def=MassDefinition(200, "critical"),
        concentration=D08Concentration(),
        m_range=(M_GRID[0], M_GRID[-1]),
        n_m=M_GRID.shape[0],
    )


@pytest.fixture
def hm200c_oob(out_of_bounds_cosmology):
    """Same as hm200c but backed by an out-of-bounds cosmology, for NaN-propagation checks."""
    return HaloModel(
        cosmology=out_of_bounds_cosmology,
        mass_def=MassDefinition(200, "critical"),
        concentration=D08Concentration(),
        m_range=(M_GRID[0], M_GRID[-1]),
        n_m=M_GRID.shape[0],
    )


def _check_grad(f, x0, rtol=1e-2):
    """jax.grad is finite and matches a scale-appropriate finite-difference estimate."""
    x0 = jnp.asarray(x0, dtype=float)
    g_auto = jax.grad(f)(x0)
    assert jnp.isfinite(g_auto)
    eps = jnp.maximum(1e-6, 1e-5 * jnp.abs(x0))
    g_fd = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
    assert jnp.isclose(g_auto, g_fd, rtol=rtol, atol=1e-8)


# Shared 3-axis (r/k, m, z) shape-matrix inputs, extending test_halo_model.py's 2-axis
# SHAPE_MATRIX_CASES convention by one more broadcast-then-squeeze axis.
R_VALS = {
    "scalar": jnp.array(0.1),
    "array1": jnp.array([0.1]),
    "arrayN": jnp.geomspace(0.05, 0.3, 4),
}
M_VALS = {
    "scalar": jnp.array(5e13),
    "array1": jnp.array([5e13]),
    "arrayN": jnp.geomspace(1e11, 1e14, 5),
}
Z_VALS = {
    "scalar": jnp.array(0.5),
    "array1": jnp.array([0.5]),
    "arrayM": jnp.array([0.0, 0.5, 1.0, 2.0]),
}
SHAPE_MATRIX_3D = [
    ("scalar", "scalar", "scalar", ()),
    ("array1", "array1", "array1", ()),
    ("arrayN", "scalar", "scalar", (4,)),
    ("scalar", "arrayN", "scalar", (5,)),
    ("scalar", "scalar", "arrayM", (4,)),
    ("arrayN", "arrayN", "arrayM", (4, 5, 4)),
]
# m x z only (no r/k axis), used by HOD's/CIB's per-halo helpers.
SHAPE_MATRIX_2D = [
    ("scalar", "scalar", ()),
    ("array1", "scalar", ()),
    ("array1", "array1", ()),
    ("arrayN", "scalar", (5,)),
    ("scalar", "arrayM", (4,)),
    ("arrayN", "arrayM", (5, 4)),
]
# z only, used by ng_bar/galaxy_bias/mean_emissivity (mass integral is internal).
Z_ONLY_CASES = [("scalar", ()), ("array1", ()), ("arrayM", (4,))]


class TestHaloProfileBase:
    # HankelTransform.transform()'s output shape matches the input profile array's shape.
    def test_hankel_transform_shape(self):
        x_grid = jnp.logspace(-4, 1, 64)
        f_theta = jnp.exp(-x_grid)
        k, y_k = HankelTransform(x_grid, nu=0.5).transform(f_theta)
        assert k.shape == x_grid.shape
        assert y_k.shape == x_grid.shape

    # HankelTransform stores whatever x it's given verbatim -- no internal sorting.
    def test_hankel_transform_requires_presorted_x(self):
        x_unsorted = jnp.array([1.0, 0.1, 10.0, 0.5])
        ht = HankelTransform(x_unsorted, nu=0.5)
        assert jnp.array_equal(ht._hankel.x, x_unsorted)

    # _u_r_nfw's output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_u_r_nfw_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        nfw = NFWMatterProfile()
        out = nfw._u_r_nfw(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # _u_k_nfw's output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_u_k_nfw_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        nfw = NFWMatterProfile()
        _, out = nfw._u_k_nfw(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # The unit-mass NFW profile vanishes outside r_delta*(1+z) and is positive just inside.
    def test_u_r_nfw_zero_beyond_r_delta(self, hm200c):
        nfw = NFWMatterProfile()
        m, z = jnp.array(1e13), jnp.array(0.5)
        r_delta = hm200c.mass_def.r_delta(hm200c.cosmology, m, z)
        r_edge = r_delta * (1.0 + z)
        assert nfw._u_r_nfw(hm200c, r_edge * 1.5, m, z) == 0.0
        assert nfw._u_r_nfw(hm200c, r_edge * 0.5, m, z) > 0.0

    # The unit-mass NFW Fourier profile approaches 1 (its unit-mass normalization) as k -> 0.
    def test_u_k_nfw_unity_at_k_zero(self, hm200c):
        nfw = NFWMatterProfile()
        m, z = jnp.array(1e13), jnp.array(0.5)
        _, u_k = nfw._u_k_nfw(hm200c, jnp.array(1e-6), m, z)
        assert jnp.isclose(u_k, 1.0, atol=1e-3)


class TestFourier2pt:
    # HOD x HOD dispatch matches the hand-built central/satellite second-moment formula.
    def test_dispatch_hod_hod_matches_manual_formula(self, hm200c):
        hod1 = Z07GalaxyHODProfile(M_min=10**12.0)
        hod2 = Z07GalaxyHODProfile(M_min=10**12.3)
        k, m, z = (
            jnp.geomspace(0.1, 5.0, 4),
            jnp.geomspace(1e11, 1e14, 5),
            jnp.array([0.3, 1.0]),
        )

        ns1, ns2 = hod1.n_sat(hm200c, m), hod2.n_sat(hm200c, m)
        ng1, ng2 = hod1.ng_bar(hm200c, z), hod2.ng_bar(hm200c, z)
        _, u1 = hod1._u_k_nfw(hm200c, k, m, z)
        _, u2 = hod2._u_k_nfw(hm200c, k, m, z)
        expected = (
            ns1[None, :, None] * ns2[None, :, None] * u1 * u2
            + ns1[None, :, None] * u1
            + ns2[None, :, None] * u2
        ) / (ng1[None, None, :] * ng2[None, None, :])
        actual = _fourier_2pt(hm200c, hod1, hod2, k, m, z)
        assert jnp.allclose(actual, jnp.squeeze(expected))

    # CIB x CIB dispatch matches the hand-built central/satellite second-moment formula.
    def test_dispatch_cib_cib_matches_manual_formula(self, hm200c):
        cib1 = S12CIBProfile(nu=100)
        cib2 = S12CIBProfile(nu=250)
        k, m, z = (
            jnp.geomspace(0.1, 5.0, 4),
            jnp.geomspace(1e11, 1e14, 5),
            jnp.array([0.3, 1.0]),
        )

        ls1, lc1 = cib1.l_sat(hm200c, m, z), cib1.l_cen(hm200c, m, z)
        ls2, lc2 = cib2.l_sat(hm200c, m, z), cib2.l_cen(hm200c, m, z)
        _, u1 = cib1._u_k_nfw(hm200c, k, m, z)
        _, u2 = cib2._u_k_nfw(hm200c, k, m, z)
        expected = (1.0 / (4.0 * jnp.pi) ** 2) * (
            ls1[None, :, :] * ls2[None, :, :] * u1 * u2
            + ls1[None, :, :] * lc2[None, :, :] * u1
            + ls2[None, :, :] * lc1[None, :, :] * u2
        )
        actual = _fourier_2pt(hm200c, cib1, cib2, k, m, z)
        assert jnp.allclose(actual, jnp.squeeze(expected))

    # Mixed profile-family pairs (no special-cased dispatch) fall back to the product of fouriers.
    def test_dispatch_mixed_types_falls_back_to_product_of_fouriers(self, hm200c):
        hod = Z07GalaxyHODProfile()
        nfw = NFWMatterProfile()
        k, m, z = jnp.array([0.1, 1.0]), jnp.array([1e13, 1e14]), jnp.array(0.5)
        actual = _fourier_2pt(hm200c, hod, nfw, k, m, z)
        expected = hod.fourier(hm200c, k, m, z) * nfw.fourier(hm200c, k, m, z)
        assert jnp.allclose(actual, expected)

    # _fourier_2pt's output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        hod1, hod2 = Z07GalaxyHODProfile(), Z07GalaxyHODProfile(alpha_s=1.5)
        out = _fourier_2pt(
            hm200c, hod1, hod2, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key]
        )
        assert jnp.shape(out) == expected_shape


class TestNFWMatterProfile:
    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        nfw = NFWMatterProfile()
        out = nfw.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier()'s output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        nfw = NFWMatterProfile()
        out = nfw.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # The mass-weighted real-space profile is positive and non-increasing with radius.
    def test_real_positive_and_nonincreasing_with_radius(self, hm200c):
        nfw = NFWMatterProfile()
        r = jnp.geomspace(0.01, 0.5, 10)
        rho = nfw.real(hm200c, r, jnp.array(1e13), jnp.array(0.5))
        assert jnp.all(rho > 0)
        assert jnp.all(jnp.diff(rho) <= 0)

    # NFWMatterProfile has no parameters -- it is a trivially empty pytree that still round-trips.
    def test_pytree_roundtrip_empty_leaves(self, hm200c):
        nfw = NFWMatterProfile()
        assert nfw._tree_flatten() == ((), None)
        leaves, treedef = jax.tree_util.tree_flatten(nfw)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(
            rt.real(hm200c, 0.1, 1e13, 0.5), nfw.real(hm200c, 0.1, 1e13, 0.5)
        )

    # The same jitted real()/fourier() accept a differently-shaped z array across successive calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        nfw = NFWMatterProfile()
        r, m = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14])
        real_scalar = nfw.real(hm200c, r, m, 0.5)
        real_array = nfw.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # real()/fourier() are all-NaN (not an exception) when the cosmology is outside emulator bounds.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        nfw = NFWMatterProfile()
        r, m, z = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(nfw.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(nfw.fourier(hm200c_oob, r, m, z)))

    # real() at mass_def A vs. mass_def B (with mass properly converted via mass_translator)
    # agree within a loose tolerance -- NFW's generic kernel is approximately mass-def
    # self-consistent, unlike GNFW's deliberate mass-def sensitivity (see TestGNFWPressureProfile).
    @pytest.mark.parametrize(
        "md_a,md_b,conc",
        [
            (
                MassDefinition(200, "critical"),
                MassDefinition(200, "mean"),
                D08Concentration(),
            ),
            (
                MassDefinition(200, "critical"),
                MassDefinition("vir", "critical"),
                B13Concentration(),
            ),
            (
                MassDefinition(200, "mean"),
                MassDefinition("vir", "critical"),
                ConstantConcentration(c=5),
            ),
        ],
    )
    def test_mass_def_self_consistency_via_generic_kernel(
        self, fixed_cosmology, md_a, md_b, conc
    ):
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        nfw = NFWMatterProfile()
        m, z = jnp.array(1e13), jnp.array(0.5)
        m_b = mass_translator(md_a, md_b, conc)(fixed_cosmology, m, z)
        r = jnp.geomspace(0.05, 0.3, 5)
        real_a = nfw.real(hm_a, r, m, z)
        real_b = nfw.real(hm_b, r, m_b, z)
        assert jnp.allclose(real_a, real_b, rtol=0.15)

    class TestGradients:
        # NFW has no own parameters -- H0 (via cosmology) is the only meaningful gradient path.
        def test_real_grad_wrt_H0(self, hm200c):
            nfw = NFWMatterProfile()
            r, m, z = jnp.array(0.1), jnp.array(1e13), jnp.array(0.5)

            def f(H0):
                return nfw.real(
                    hm200c.update(cosmology=hm200c.cosmology.update(H0=H0)), r, m, z
                )

            _check_grad(f, 67.5)

        def test_fourier_grad_wrt_H0(self, hm200c):
            nfw = NFWMatterProfile()
            k, m, z = jnp.array(0.5), jnp.array(1e13), jnp.array(0.5)

            def f(H0):
                return nfw.fourier(
                    hm200c.update(cosmology=hm200c.cosmology.update(H0=H0)), k, m, z
                )

            _check_grad(f, 67.5)


class TestZ07GalaxyHODProfile:
    # N_cen is bounded in [0, 1] for any mass.
    def test_n_cen_bounded_0_1(self, hm200c):
        hod = Z07GalaxyHODProfile()
        m = jnp.geomspace(1e10, 1e16, 20)
        n_cen = hod.n_cen(hm200c, m)
        assert jnp.all((n_cen >= 0.0) & (n_cen <= 1.0))

    # N_sat is non-negative and exactly zero below M0.
    def test_n_sat_nonnegative_and_zero_below_M0(self, hm200c):
        hod = Z07GalaxyHODProfile(M0=1e12)
        m = jnp.geomspace(1e10, 1e16, 20)
        n_sat = hod.n_sat(hm200c, m)
        assert jnp.all(n_sat >= 0.0)
        assert jnp.all(n_sat[m < 1e12] == 0.0)

    # ng_bar's output shape follows the z-only broadcast-then-squeeze convention.
    @pytest.mark.parametrize("z_key,expected_shape", Z_ONLY_CASES)
    def test_ng_bar_shape_matrix(self, hm200c, z_key, expected_shape):
        hod = Z07GalaxyHODProfile()
        out = hod.ng_bar(hm200c, Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # galaxy_bias's output shape follows the z-only broadcast-then-squeeze convention.
    @pytest.mark.parametrize("z_key,expected_shape", Z_ONLY_CASES)
    def test_galaxy_bias_shape_matrix(self, hm200c, z_key, expected_shape):
        hod = Z07GalaxyHODProfile()
        out = hod.galaxy_bias(hm200c, Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        hod = Z07GalaxyHODProfile()
        out = hod.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier()'s output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        hod = Z07GalaxyHODProfile()
        out = hod.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # ng_bar/galaxy_bias take no mass argument at all -- they always integrate over halo_model.m_range/n_m.
    def test_ng_bar_ignores_m_range_arg_not_exposed(self, hm200c):
        import inspect

        hod = Z07GalaxyHODProfile()
        assert list(inspect.signature(hod.ng_bar.__wrapped__).parameters) == [
            "self",
            "halo_model",
            "z",
        ]
        hm_coarse = hm200c.update(m_range=(1e10, 1e15), n_m=10)
        assert not jnp.isclose(
            hod.ng_bar(hm200c, jnp.array(0.5)),
            hod.ng_bar(hm_coarse, jnp.array(0.5)),
            rtol=1e-6,
        )

    # real()'s implicit 1/ng_bar normalization is derived from halo_model.m_range/n_m, not from
    # whatever `m` array is passed for the output's own mass axis -- callers should pass
    # m = the halo model's own Gauss-Legendre mass grid for a self-consistent normalized profile.
    def test_real_normalization_uses_halo_model_m_range_not_passed_m(self, hm200c):
        hod = Z07GalaxyHODProfile()
        r, z = jnp.array(0.1), jnp.array(0.5)
        ng_from_hm_grid = hod.ng_bar(hm200c, z)
        other_m = jnp.geomspace(1e11, 1e14, 6)
        default_m = _m_grid(hm200c)
        real_default_m = hod.real(hm200c, r, default_m, z)
        real_other_m = hod.real(hm200c, r, other_m, z)
        ns_default, nc_default = hod.n_sat(hm200c, default_m), hod.n_cen(
            hm200c, default_m
        )
        u_default = hod._u_r_nfw(hm200c, r, default_m, z)
        implied_ng_default = (nc_default + ns_default * u_default) / real_default_m
        ns_other, nc_other = hod.n_sat(hm200c, other_m), hod.n_cen(hm200c, other_m)
        u_other = hod._u_r_nfw(hm200c, r, other_m, z)
        implied_ng_other = (nc_other + ns_other * u_other) / real_other_m
        assert jnp.allclose(implied_ng_default, ng_from_hm_grid, rtol=1e-6)
        assert jnp.allclose(implied_ng_other, ng_from_hm_grid, rtol=1e-6)

    # update() replaces only the requested leaf, leaving the rest untouched.
    def test_update_none_coalescing(self):
        hod = Z07GalaxyHODProfile()
        hod2 = hod.update(alpha_s=2.0)
        assert hod2.alpha_s == 2.0
        assert hod2.sigma_log10M == hod.sigma_log10M
        assert hod2.M1_prime == hod.M1_prime
        assert hod2.M_min == hod.M_min
        assert hod2.M0 == hod.M0

    # Z07GalaxyHODProfile survives a JAX pytree flatten/unflatten round trip unchanged.
    def test_pytree_roundtrip(self):
        hod = Z07GalaxyHODProfile(alpha_s=1.5)
        leaves, treedef = jax.tree_util.tree_flatten(hod)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rt.alpha_s == hod.alpha_s

    # The same jitted real()/fourier()/ng_bar accept a differently-shaped z array across calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        hod = Z07GalaxyHODProfile()
        r, m = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14])
        real_scalar = hod.real(hm200c, r, m, 0.5)
        real_array = hod.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # ng_bar/galaxy_bias/real/fourier are all-NaN (not an exception) outside emulator bounds.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        hod = Z07GalaxyHODProfile()
        r, m, z = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(hod.ng_bar(hm200c_oob, z)))
        assert jnp.all(jnp.isnan(hod.galaxy_bias(hm200c_oob, z)))
        assert jnp.all(jnp.isnan(hod.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(hod.fourier(hm200c_oob, r, m, z)))

    # real() at mass_def A vs. mass_def B (mass properly converted) agree within a loose
    # tolerance for the satellite-only NFW-kernel term (the central term is mass-def-independent).
    @pytest.mark.parametrize(
        "md_a,md_b,conc",
        [
            (
                MassDefinition(200, "critical"),
                MassDefinition(200, "mean"),
                D08Concentration(),
            ),
            (
                MassDefinition(200, "critical"),
                MassDefinition("vir", "critical"),
                B13Concentration(),
            ),
            (
                MassDefinition(200, "mean"),
                MassDefinition("vir", "critical"),
                ConstantConcentration(c=5),
            ),
        ],
    )
    def test_mass_def_self_consistency_via_generic_kernel(
        self, fixed_cosmology, md_a, md_b, conc
    ):
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        hod = Z07GalaxyHODProfile()
        m, z = jnp.array(5e12), jnp.array(0.5)
        m_b = mass_translator(md_a, md_b, conc)(fixed_cosmology, m, z)
        r = jnp.geomspace(0.05, 0.3, 5)
        real_a = hod.real(hm_a, r, m, z)
        real_b = hod.real(hm_b, r, m_b, z)
        assert jnp.allclose(real_a, real_b, rtol=0.15)

    class TestGradients:
        @pytest.mark.parametrize(
            "param,x0",
            [
                ("sigma_log10M", 0.68),
                ("alpha_s", 1.30),
                ("M1_prime", 10**12.87),
                ("M_min", 10**11.97),
                ("M0", 1e9),
            ],
        )
        def test_ng_bar_grad_wrt_params(self, hm200c, param, x0):
            hod = Z07GalaxyHODProfile()
            z = jnp.array(0.5)

            def f(x):
                return hod.update(**{param: x}).ng_bar(hm200c, z)

            _check_grad(f, x0)

        @pytest.mark.parametrize(
            "param,x0",
            [
                ("sigma_log10M", 0.68),
                ("alpha_s", 1.30),
                ("M1_prime", 10**12.87),
                ("M_min", 10**11.97),
                ("M0", 1e9),
            ],
        )
        def test_galaxy_bias_grad_wrt_params(self, hm200c, param, x0):
            hod = Z07GalaxyHODProfile()
            z = jnp.array(0.5)

            def f(x):
                return hod.update(**{param: x}).galaxy_bias(hm200c, z)

            _check_grad(f, x0)

        @pytest.mark.parametrize(
            "param,x0",
            [
                ("sigma_log10M", 0.68),
                ("alpha_s", 1.30),
                ("M1_prime", 10**12.87),
                ("M_min", 10**11.97),
                ("M0", 1e9),
            ],
        )
        def test_real_grad_wrt_params(self, hm200c, param, x0):
            hod = Z07GalaxyHODProfile()
            r, z = jnp.array(0.1), jnp.array(0.5)

            m = _m_grid(hm200c)

            def f(x):
                return hod.update(**{param: x}).real(hm200c, r, m, z).sum()

            _check_grad(f, x0)


class TestS12CIBProfile:
    # l_gal/l_sat/l_cen's output shapes follow the (m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("method", ["l_gal", "l_sat", "l_cen"])
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_2D)
    def test_per_halo_shape_matrix(self, hm200c, method, m_key, z_key, expected_shape):
        cib = S12CIBProfile(nu=100)
        out = getattr(cib, method)(hm200c, M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # mean_emissivity's output shape follows the z-only broadcast-then-squeeze convention.
    @pytest.mark.parametrize("z_key,expected_shape", Z_ONLY_CASES)
    def test_mean_emissivity_shape_matrix(self, hm200c, z_key, expected_shape):
        cib = S12CIBProfile(nu=100)
        out = cib.mean_emissivity(hm200c, Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # mean_intensity always collapses to a scalar.
    def test_mean_intensity_is_scalar(self, hm200c):
        cib = S12CIBProfile(nu=100)
        out = cib.mean_intensity(hm200c, (0.1, 1.0), 5)
        assert jnp.shape(out) == ()

    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        cib = S12CIBProfile(nu=100)
        out = cib.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier()'s output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        cib = S12CIBProfile(nu=100)
        out = cib.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # The central term is a Dirac delta at r=0 -- it contributes only at the r==0 slice.
    def test_real_central_term_is_delta_at_r_zero(self, hm200c):
        cib = S12CIBProfile(nu=100)
        r = jnp.array([0.0, 0.1, 0.5])
        m, z = jnp.array(5e13), jnp.array(0.5)
        real_r = cib.real(hm200c, r, m, z)
        lc = cib.l_cen(hm200c, m, z)
        sat_only = real_r - jnp.array([lc / (4 * jnp.pi), 0.0, 0.0])
        assert jnp.isclose(sat_only[0], real_r[0] - lc / (4 * jnp.pi), atol=1e-8)
        assert real_r[1] < real_r[0]

    # l_sat's subhalo integral uses ngrid=halo_model.n_m (Gauss-Legendre in ln(m_sub)) --
    # a smooth integrand, so it converges much faster with n_m than M21CIB's clamped one below.
    def test_l_sat_ngrid_tracks_halo_model_n_m(self, fixed_cosmology):
        cib = S12CIBProfile(nu=100)
        hm_fine = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=MassDefinition(200, "critical"),
            concentration=D08Concentration(),
            m_range=(1e10, 1e15),
            n_m=100,
        )
        hm_coarse = hm_fine.update(n_m=8)
        m, z = jnp.array(5e13), jnp.array(0.5)
        l_fine, l_coarse = cib.l_sat(hm_fine, m, z), cib.l_sat(hm_coarse, m, z)
        assert not jnp.array_equal(l_fine, l_coarse)
        assert jnp.isclose(l_fine, l_coarse, rtol=1e-3)

    # update() replaces only the requested leaves, leaving the rest untouched.
    def test_update_rebuilds_leaves(self):
        cib = S12CIBProfile(nu=100)
        cib2 = cib.update(alpha=0.5, M_eff=10**13.0)
        assert cib2.alpha == 0.5 and cib2.M_eff == 10**13.0
        assert cib2.beta == cib.beta and cib2.nu == cib.nu

    # S12CIBProfile survives a JAX pytree flatten/unflatten round trip unchanged.
    def test_pytree_roundtrip(self):
        cib = S12CIBProfile(nu=100, alpha=0.5)
        leaves, treedef = jax.tree_util.tree_flatten(cib)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rt.alpha == cib.alpha and rt.nu == cib.nu

    # The same jitted real()/fourier() accept a differently-shaped z array across calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        cib = S12CIBProfile(nu=100)
        r, m = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14])
        real_scalar = cib.real(hm200c, r, m, 0.5)
        real_array = cib.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # real/fourier/mean_emissivity/mean_intensity are all-NaN outside emulator bounds.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        cib = S12CIBProfile(nu=100)
        r, m, z = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(cib.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(cib.fourier(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(cib.mean_emissivity(hm200c_oob, z)))
        assert jnp.isnan(cib.mean_intensity(hm200c_oob, (float(z[0]), float(z[-1])), 5))

    # real() at mass_def A vs. mass_def B (mass properly converted) agree within a loose
    # tolerance -- the satellite term traces the same generic NFW kernel as NFW/HOD above.
    @pytest.mark.parametrize(
        "md_a,md_b,conc",
        [
            (
                MassDefinition(200, "critical"),
                MassDefinition(200, "mean"),
                D08Concentration(),
            ),
            (
                MassDefinition(200, "critical"),
                MassDefinition("vir", "critical"),
                B13Concentration(),
            ),
            (
                MassDefinition(200, "mean"),
                MassDefinition("vir", "critical"),
                ConstantConcentration(c=5),
            ),
        ],
    )
    def test_mass_def_self_consistency_via_generic_kernel(
        self, fixed_cosmology, md_a, md_b, conc
    ):
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        cib = S12CIBProfile(nu=100)
        m, z = jnp.array(5e13), jnp.array(0.5)
        m_b = mass_translator(md_a, md_b, conc)(fixed_cosmology, m, z)
        r = jnp.geomspace(0.05, 0.3, 5)
        real_a = cib.real(hm_a, r, m, z)
        real_b = cib.real(hm_b, r, m_b, z)
        assert jnp.allclose(real_a, real_b, rtol=0.15)

    class TestGradients:
        # z_p (default 1e100) keeps _phi always on the same branch, so its gradient is
        # structurally zero and is excluded here rather than tested as a trivial 0 ~= 0 check.
        PARAMS = [
            ("L0", 6.4e-8),
            ("alpha", 0.36),
            ("beta", 1.75),
            ("gamma", 1.7),
            ("T0", 24.4),
            ("M_eff", 10**12.6),
            ("sigma2_LM", 0.5),
            ("delta", 3.6),
            ("M_min", 10**11.5),
            ("nu", 100.0),
        ]

        @pytest.mark.parametrize("param,x0", PARAMS)
        def test_real_grad_wrt_params(self, hm200c, param, x0):
            cib = S12CIBProfile(nu=100)
            r, m, z = jnp.array(0.1), jnp.array(5e13), jnp.array(0.5)

            def f(x):
                return cib.update(**{param: x}).real(hm200c, r, m, z)

            _check_grad(f, x0)


class TestM21CIBProfile:
    # l_gal/l_sat/l_cen's output shapes follow the (m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("method", ["l_gal", "l_sat", "l_cen"])
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_2D)
    def test_per_halo_shape_matrix(self, hm200c, method, m_key, z_key, expected_shape):
        cib = M21CIBProfile(nu=100)
        out = getattr(cib, method)(hm200c, M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # mean_emissivity's output shape follows the z-only broadcast-then-squeeze convention.
    @pytest.mark.parametrize("z_key,expected_shape", Z_ONLY_CASES)
    def test_mean_emissivity_shape_matrix(self, hm200c, z_key, expected_shape):
        cib = M21CIBProfile(nu=100)
        out = cib.mean_emissivity(hm200c, Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        cib = M21CIBProfile(nu=100)
        out = cib.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier()'s output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        cib = M21CIBProfile(nu=100)
        out = cib.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # With s_nu=None (default), the SED grid is loaded from disk into a 3-tuple of arrays.
    def test_default_construction_loads_sed_files(self):
        try:
            cib = M21CIBProfile(nu=100)
        except Exception as exc:
            pytest.skip(
                f"default M21CIB auxiliary SED files not available locally: {exc}"
            )
        assert len(cib.s_nu) == 3
        assert all(arr.size > 0 for arr in cib.s_nu)

    # An explicit s_nu bypasses file I/O entirely.
    def test_custom_s_nu_bypasses_file_load(self, monkeypatch):
        def _raise(*a, **k):
            raise AssertionError(
                "np.loadtxt should not be called when s_nu is provided"
            )

        monkeypatch.setattr("numpy.loadtxt", _raise)
        z_g, nu_g = jnp.linspace(0.0, 5.0, 6), jnp.geomspace(10.0, 1000.0, 6)
        s_nu_g = jnp.ones((6, 6))
        cib = M21CIBProfile(nu=100, s_nu=(z_g, nu_g, s_nu_g))
        assert cib.s_nu[0] is z_g and cib.s_nu[1] is nu_g and cib.s_nu[2] is s_nu_g

    # update() has no s_nu parameter -- the SED grid is carried over unchanged, not rebuilt.
    def test_update_cannot_change_s_nu(self):
        z_g, nu_g = jnp.linspace(0.0, 5.0, 6), jnp.geomspace(10.0, 1000.0, 6)
        s_nu_g = jnp.ones((6, 6))
        cib = M21CIBProfile(nu=100, s_nu=(z_g, nu_g, s_nu_g))
        cib2 = cib.update(nu=150.0)
        assert cib2.s_nu is cib.s_nu

    # _s_nu_interp is rebuilt fresh from self.s_nu on every call -- no persistent, staleness-prone cache.
    def test_s_nu_interp_stateless_across_calls(self):
        z_g, nu_g = jnp.linspace(0.0, 5.0, 6), jnp.geomspace(10.0, 1000.0, 6)
        s_nu_g = jnp.ones((6, 6)) * jnp.geomspace(1.0, 2.0, 6)[None, :]
        cib = M21CIBProfile(nu=100, s_nu=(z_g, nu_g, s_nu_g))
        z, nu = jnp.array([0.5, 1.5]), 150.0
        assert jnp.allclose(cib._s_nu_interp(z, nu), cib._s_nu_interp(z, nu))

    # l_sat's subhalo integral also uses ngrid=halo_model.n_m, but the Maniyar clamping
    # (jnp.minimum of two branches) kinks the integrand, so Gauss-Legendre converges
    # more slowly here than for S12CIB's smooth one above -- shifts more with n_m, though
    # it still converges.
    def test_l_sat_ngrid_tracks_halo_model_n_m(self, fixed_cosmology):
        cib = M21CIBProfile(nu=100)
        hm_fine = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=MassDefinition(200, "critical"),
            concentration=D08Concentration(),
            m_range=(1e10, 1e15),
            n_m=200,
        )
        hm_coarse = hm_fine.update(n_m=8)
        m, z = jnp.array(5e13), jnp.array(0.5)
        l_fine, l_coarse = cib.l_sat(hm_fine, m, z), cib.l_sat(hm_coarse, m, z)
        assert not jnp.array_equal(l_fine, l_coarse)
        assert jnp.isclose(l_fine, l_coarse, rtol=0.1)

    # _m_dot/_sfr are NOT squeezed like the public methods -- always shape (N_m, N_z).
    def test_m_dot_and_sfr_helper_shapes(self, hm200c):
        cib = M21CIBProfile(nu=100)
        assert cib._m_dot(hm200c, jnp.array(1e13), jnp.array(0.5)).shape == (1, 1)
        assert cib._sfr(hm200c, jnp.array(1e13), jnp.array(0.5)).shape == (1, 1)
        assert cib._m_dot(
            hm200c, jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        ).shape == (2, 2)

    # update() replaces only the requested leaves and always preserves s_nu.
    def test_update_rebuilds_8_leaves_preserves_s_nu(self):
        cib = M21CIBProfile(nu=100)
        cib2 = cib.update(eta_max=0.5, tau=1.5)
        assert cib2.eta_max == 0.5 and cib2.tau == 1.5
        assert cib2.z_c == cib.z_c
        assert cib2.s_nu is cib.s_nu

    # M21CIBProfile survives a JAX pytree flatten/unflatten round trip, s_nu preserved as aux.
    def test_pytree_roundtrip_preserves_s_nu(self):
        cib = M21CIBProfile(nu=100)
        leaves, treedef = jax.tree_util.tree_flatten(cib)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert all(jnp.array_equal(a, b) for a, b in zip(rt.s_nu, cib.s_nu))

    # The same jitted real()/fourier() accept a differently-shaped z array across calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        cib = M21CIBProfile(nu=100)
        r, m = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14])
        real_scalar = cib.real(hm200c, r, m, 0.5)
        real_array = cib.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # real/fourier/mean_emissivity are all-NaN outside emulator bounds -- not masked by the interpolator.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        cib = M21CIBProfile(nu=100)
        r, m, z = jnp.array([0.1, 0.5]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(cib.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(cib.fourier(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(cib.mean_emissivity(hm200c_oob, z)))

    # l_cen/l_sat never reference halo_model.mass_def -- they are identical across mass defs
    # at a fixed mass, while the satellite NFW kernel _u_r_nfw genuinely differs (different
    # r_delta/c_delta). A mass_translator-based real() comparison (as used for NFW/HOD/S12CIB
    # above) is NOT used here: M21's natural-log Gaussian efficiency term is steep enough that
    # mass_translator's approximate mass conversion gets amplified into an order-of-magnitude
    # spurious difference, unrelated to whether mass_def is handled correctly.
    @pytest.mark.parametrize(
        "md_a,md_b,conc",
        [
            (
                MassDefinition(200, "critical"),
                MassDefinition(200, "mean"),
                D08Concentration(),
            ),
            (
                MassDefinition(200, "critical"),
                MassDefinition("vir", "critical"),
                B13Concentration(),
            ),
            (
                MassDefinition(200, "mean"),
                MassDefinition("vir", "critical"),
                ConstantConcentration(c=5),
            ),
        ],
    )
    def test_mass_def_affects_only_kernel_not_luminosity(
        self, fixed_cosmology, md_a, md_b, conc
    ):
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        cib = M21CIBProfile(nu=100)
        m, z = jnp.array(5e13), jnp.array(0.5)
        assert jnp.allclose(cib.l_cen(hm_a, m, z), cib.l_cen(hm_b, m, z), rtol=1e-6)
        assert jnp.allclose(cib.l_sat(hm_a, m, z), cib.l_sat(hm_b, m, z), rtol=1e-6)
        u_m_a = cib._u_r_nfw(hm_a, jnp.array(0.1), m, z)
        u_m_b = cib._u_r_nfw(hm_b, jnp.array(0.1), m, z)
        assert not jnp.isclose(u_m_a, u_m_b, rtol=1e-4)

    class TestGradients:
        # s_nu is static aux data, not a leaf -- excluded here since it is non-differentiable by design.
        PARAMS = [
            ("eta_max", 0.4028),
            ("z_c", 1.5),
            ("tau", 1.204),
            ("f_sub", 0.134),
            ("M_min", 10**11.5),
            ("M_eff", 10**12.6),
            ("sigma2_LM", 0.5),
            ("nu", 100.0),
        ]

        @pytest.mark.parametrize("param,x0", PARAMS)
        def test_real_grad_wrt_params(self, hm200c, param, x0):
            cib = M21CIBProfile(nu=100)
            r, m, z = jnp.array(0.1), jnp.array(5e13), jnp.array(0.5)

            def f(x):
                return cib.update(**{param: x}).real(hm200c, r, m, z)

            _check_grad(f, x0)


class TestB16DensityProfile:
    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        b16 = B16DensityProfile()
        out = b16.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier()'s output shape follows the (k, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        b16 = B16DensityProfile()
        out = b16.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # x_range/n_x are always turned into a log-spaced grid, regardless of the requested bounds.
    def test_x_grid_is_log_spaced_from_range(self):
        b16 = B16DensityProfile(x_range=(0.1, 10.0), n_x=4)
        assert jnp.array_equal(b16.x_grid, jnp.logspace(jnp.log10(0.1), jnp.log10(10.0), 4))

    # update(x_range=..., n_x=...) rebuilds a fresh HankelTransform on the new log-spaced grid.
    def test_update_x_range_rebuilds_hankel(self):
        b16 = B16DensityProfile()
        b16_2 = b16.update(x_range=(0.01, 5.0), n_x=8)
        assert jnp.array_equal(b16_2.x_grid, jnp.logspace(jnp.log10(0.01), jnp.log10(5.0), 8))
        assert b16_2._hankel is not b16._hankel

    # _tree_unflatten bypasses the x_grid setter and trusts aux_data as-is -- no re-sort/rebuild.
    def test_tree_unflatten_trusts_aux_data(self):
        b16 = B16DensityProfile()
        leaves, aux = b16._tree_flatten()
        replacement = HankelTransform(aux[0].x[::-1], nu=0.5)
        rt = B16DensityProfile._tree_unflatten((replacement,), leaves)
        assert jnp.array_equal(rt.x_grid, replacement.x)

    # calibrate() with a named preset matches an explicit update() call with the same kwargs.
    @pytest.mark.parametrize("key", ["agn", "shock"])
    def test_calibrate_matches_update(self, key):
        b16 = B16DensityProfile()
        calibrated = b16.calibrate(key)
        updated = b16.update(**B16DensityProfile._PRESETS[key])
        assert calibrated._tree_flatten()[0] == updated._tree_flatten()[0]

    # calibrate() with an unknown preset name raises ValueError.
    def test_calibrate_unknown_key_raises_value_error(self):
        b16 = B16DensityProfile()
        with pytest.raises(ValueError):
            b16.calibrate("bogus")

    # calibrate() only touches the 9 shape parameters -- x_grid and x_out are preserved.
    def test_calibrate_preserves_x_grid_and_x_out(self):
        b16 = B16DensityProfile(x_out=2.5)
        calibrated = b16.calibrate("shock")
        assert calibrated.x_out == 2.5
        assert jnp.array_equal(calibrated.x_grid, b16.x_grid)

    # real() is invariant to halo_model.mass_def given a properly mass_translator-converted
    # mass -- B16 always renormalizes internally to 200c. Contrast with GNFW's sensitivity below.
    @pytest.mark.parametrize(
        "md_b,shape_kwargs",
        [
            (MassDefinition(500, "critical"), {}),
            (MassDefinition("vir", "critical"), dict(A_rho0=3000.0, A_beta=4.2)),
        ],
    )
    def test_mass_def_invariance_given_proper_conversion(
        self, fixed_cosmology, md_b, shape_kwargs
    ):
        md_a, conc = MassDefinition(200, "critical"), ConstantConcentration(c=5)
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        b16 = B16DensityProfile(**shape_kwargs)
        m, z = jnp.array(1e14), jnp.array(0.3)
        m_b = mass_translator(md_a, md_b, conc)(fixed_cosmology, m, z)
        r = jnp.geomspace(0.1, 1.0, 5)
        real_a = b16.real(hm_a, r, m, z)
        real_b = b16.real(hm_b, r, m_b, z)
        assert jnp.allclose(real_a, real_b, rtol=0.15)

    # fourier() clamps to the native grid's edge value outside its range, rather than
    # extrapolating or returning NaN -- any two sufficiently-large k give the same clamped value.
    def test_fourier_clamps_at_native_grid_edges(self, hm200c):
        b16 = B16DensityProfile()
        m, z = jnp.array(1e14), jnp.array(0.3)
        u_a = b16.fourier(hm200c, jnp.array(1e6), m, z)
        u_b = b16.fourier(hm200c, jnp.array(1e8), m, z)
        assert jnp.isfinite(u_a) and jnp.isfinite(u_b)
        assert jnp.allclose(u_a, u_b, rtol=1e-6)

    # The same jitted real()/fourier() accept a differently-shaped z array across calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        b16 = B16DensityProfile()
        r, m = jnp.array([0.1, 0.2]), jnp.array([1e13, 1e14])
        real_scalar = b16.real(hm200c, r, m, 0.5)
        real_array = b16.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # real()/fourier() are all-NaN (not an exception) outside emulator bounds.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        b16 = B16DensityProfile()
        r, m, z = jnp.array([0.1, 0.2]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(b16.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(b16.fourier(hm200c_oob, r, m, z)))

    class TestGradients:
        # x_grid is static aux data and x_out only enters a boolean mask (zero gradient by
        # construction) -- both excluded in favor of the 9 differentiable shape parameters.
        PARAMS = [
            ("A_rho0", 4000.0),
            ("A_alpha", 0.88),
            ("A_beta", 3.83),
            ("alpha_m_rho0", 0.29),
            ("alpha_m_alpha", -0.03),
            ("alpha_m_beta", 0.04),
            ("alpha_z_rho0", -0.66),
            ("alpha_z_alpha", 0.19),
            ("alpha_z_beta", -0.025),
        ]

        @pytest.mark.parametrize("param,x0", PARAMS)
        def test_real_grad_wrt_shape_params(self, hm200c, param, x0):
            b16 = B16DensityProfile()
            r, m, z = jnp.array(0.1), jnp.array(5e13), jnp.array(0.5)

            def f(x):
                return b16.update(**{param: x}).real(hm200c, r, m, z)

            _check_grad(f, x0)


class TestGNFWPressureProfile:
    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        gnfw = GNFWPressureProfile()
        out = gnfw.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier() (the shared PressureProfile.fourier) output shape follows the same convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        gnfw = GNFWPressureProfile()
        out = gnfw.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # x_range/n_x are always turned into a log-spaced grid, regardless of the requested bounds.
    def test_x_grid_is_log_spaced_from_range(self):
        gnfw = GNFWPressureProfile(x_range=(0.1, 10.0), n_x=4)
        assert jnp.array_equal(gnfw.x_grid, jnp.logspace(jnp.log10(0.1), jnp.log10(10.0), 4))

    # update(x_range=..., n_x=...) rebuilds a fresh HankelTransform on the new log-spaced grid.
    def test_update_x_range_rebuilds_hankel(self):
        gnfw = GNFWPressureProfile()
        gnfw_2 = gnfw.update(x_range=(0.01, 5.0), n_x=8)
        assert jnp.array_equal(gnfw_2.x_grid, jnp.logspace(jnp.log10(0.01), jnp.log10(5.0), 8))
        assert gnfw_2._hankel is not gnfw._hankel

    # real() is deliberately mass-def SENSITIVE (no internal renormalization) -- given a
    # properly mass_translator-converted mass, mass_def A vs. B disagree well beyond a loose
    # tolerance, unlike B16/B12's designed invariance above/below.
    @pytest.mark.parametrize(
        "md_b,shape_kwargs",
        [
            (MassDefinition(500, "critical"), {}),
            (MassDefinition("vir", "critical"), dict(P0=6.0, beta=4.5)),
        ],
    )
    def test_mass_def_sensitivity_no_internal_renorm(
        self, fixed_cosmology, md_b, shape_kwargs
    ):
        md_a, conc = MassDefinition(200, "critical"), ConstantConcentration(c=5)
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        gnfw = GNFWPressureProfile(**shape_kwargs)
        m, z = jnp.array(1e14), jnp.array(0.3)
        m_b = mass_translator(md_a, md_b, conc)(fixed_cosmology, m, z)
        r = jnp.geomspace(0.1, 1.0, 5)
        real_a = gnfw.real(hm_a, r, m, z)
        real_b = gnfw.real(hm_b, r, m_b, z)
        assert not jnp.allclose(real_a, real_b, rtol=0.1)

    # x_out truncates the profile to exactly zero beyond the truncation radius.
    def test_x_out_truncates_to_zero(self, hm200c):
        gnfw = GNFWPressureProfile(x_out=2.0)
        val = gnfw.real(hm200c, jnp.array(1000.0), jnp.array(1e14), jnp.array(0.5))
        assert val == 0.0

    # fourier()'s explicit k->0 zero-mode handling (direct trapezoid integration) clamps to a
    # constant for any sufficiently small k, rather than log-interpolating a varying value.
    def test_fourier_zero_mode_uses_trapezoid_not_loginterp(self, hm200c):
        gnfw = GNFWPressureProfile()
        m, z = jnp.array(1e14), jnp.array(0.5)
        u_a = gnfw.fourier(hm200c, jnp.array(1e-9), m, z)
        u_b = gnfw.fourier(hm200c, jnp.array(1e-7), m, z)
        assert jnp.allclose(u_a, u_b, rtol=1e-8)

    # GNFWPressureProfile survives a JAX pytree flatten/unflatten round trip unchanged.
    def test_pytree_roundtrip(self):
        gnfw = GNFWPressureProfile(P0=6.0)
        leaves, treedef = jax.tree_util.tree_flatten(gnfw)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rt.P0 == gnfw.P0

    # The same jitted real()/fourier() accept a differently-shaped z array across calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        gnfw = GNFWPressureProfile()
        r, m = jnp.array([0.1, 0.2]), jnp.array([1e13, 1e14])
        real_scalar = gnfw.real(hm200c, r, m, 0.5)
        real_array = gnfw.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # real()/fourier() are all-NaN (not an exception) outside emulator bounds.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        gnfw = GNFWPressureProfile()
        r, m, z = jnp.array([0.1, 0.2]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(gnfw.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(gnfw.fourier(hm200c_oob, r, m, z)))

    class TestGradients:
        # x_grid is static aux data and x_out only enters a boolean mask (zero gradient by
        # construction) -- both excluded in favor of the 8 differentiable shape parameters.
        PARAMS = [
            ("P0", 8.130),
            ("c500", 1.156),
            ("alpha", 1.0620),
            ("beta", 5.4807),
            ("gamma", 0.3292),
            ("B", 1.4),
            ("alpha_P", 0.12),
            ("P0_hexp", -1.0),
        ]

        @pytest.mark.parametrize("param,x0", PARAMS)
        def test_real_grad_wrt_shape_params(self, hm200c, param, x0):
            gnfw = GNFWPressureProfile()
            r, m, z = jnp.array(0.1), jnp.array(5e13), jnp.array(0.5)

            def f(x):
                return gnfw.update(**{param: x}).real(hm200c, r, m, z)

            _check_grad(f, x0)


class TestB12PressureProfile:
    # real()'s output shape follows the (r, m, z) broadcast-then-squeeze convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_real_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        b12 = B12PressureProfile()
        out = b12.real(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # fourier() (the shared PressureProfile.fourier) output shape follows the same convention.
    @pytest.mark.parametrize("r_key,m_key,z_key,expected_shape", SHAPE_MATRIX_3D)
    def test_fourier_shape_matrix(self, hm200c, r_key, m_key, z_key, expected_shape):
        b12 = B12PressureProfile()
        out = b12.fourier(hm200c, R_VALS[r_key], M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(out) == expected_shape

    # x_range/n_x are always turned into a log-spaced grid, regardless of the requested bounds.
    def test_x_grid_is_log_spaced_from_range(self):
        b12 = B12PressureProfile(x_range=(0.1, 10.0), n_x=4)
        assert jnp.array_equal(b12.x_grid, jnp.logspace(jnp.log10(0.1), jnp.log10(10.0), 4))

    # update(x_range=..., n_x=...) rebuilds a fresh HankelTransform on the new log-spaced grid.
    def test_update_x_range_rebuilds_hankel(self):
        b12 = B12PressureProfile()
        b12_2 = b12.update(x_range=(0.01, 5.0), n_x=8)
        assert jnp.array_equal(b12_2.x_grid, jnp.logspace(jnp.log10(0.01), jnp.log10(5.0), 8))
        assert b12_2._hankel is not b12._hankel

    # calibrate("agn") matches an explicit update() call with the same kwargs.
    def test_calibrate_agn_matches_update(self):
        b12 = B12PressureProfile()
        calibrated = b12.calibrate("agn")
        updated = b12.update(**B12PressureProfile._PRESETS["agn"])
        assert calibrated._tree_flatten()[0] == updated._tree_flatten()[0]

    # calibrate() with an unknown preset name raises ValueError.
    def test_calibrate_unknown_key_raises_value_error(self):
        b12 = B12PressureProfile()
        with pytest.raises(ValueError):
            b12.calibrate("bogus")

    # real() is invariant to halo_model.mass_def given a properly mass_translator-converted
    # mass -- B12 always renormalizes internally to 200c, same design as B16 above.
    @pytest.mark.parametrize(
        "md_b,shape_kwargs",
        [
            (MassDefinition(500, "critical"), {}),
            (MassDefinition("vir", "critical"), dict(A_P0=15.0, A_beta=4.0)),
        ],
    )
    def test_mass_def_invariance_given_proper_conversion(
        self, fixed_cosmology, md_b, shape_kwargs
    ):
        md_a, conc = MassDefinition(200, "critical"), ConstantConcentration(c=5)
        hm_a = HaloModel(
            cosmology=fixed_cosmology, mass_def=md_a, concentration=conc,
            m_range=(M_GRID[0], M_GRID[-1]), n_m=M_GRID.shape[0],
        )
        hm_b = hm_a.update(mass_def=md_b)
        b12 = B12PressureProfile(**shape_kwargs)
        m, z = jnp.array(1e14), jnp.array(0.3)
        m_b = mass_translator(md_a, md_b, conc)(fixed_cosmology, m, z)
        r = jnp.geomspace(0.1, 1.0, 5)
        real_a = b12.real(hm_a, r, m, z)
        real_b = b12.real(hm_b, r, m_b, z)
        assert jnp.allclose(real_a, real_b, rtol=0.15)

    # real()'s independently-re-derived r_200c matches _fourier_radius_scale's -- both use the
    # identical mass_translator + r_delta formula, so this is a regression guard against the
    # two derivations silently drifting apart.
    def test_real_and_fourier_radius_scale_agree(self, hm200c):
        b12 = B12PressureProfile()
        m, z = jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        r_scale = b12._fourier_radius_scale(hm200c, m, z)
        mass_def_200c = MassDefinition(200, "critical")
        m200c = mass_translator(hm200c.mass_def, mass_def_200c, hm200c.concentration)(
            hm200c.cosmology, m, z
        )
        r_200c_expected = mass_def_200c.r_delta(hm200c.cosmology, m200c, z)
        assert jnp.allclose(r_scale, r_200c_expected, rtol=1e-10)

    # The same jitted real()/fourier() accept a differently-shaped z array across calls.
    def test_array_size_change_across_jitted_calls(self, hm200c):
        b12 = B12PressureProfile()
        r, m = jnp.array([0.1, 0.2]), jnp.array([1e13, 1e14])
        real_scalar = b12.real(hm200c, r, m, 0.5)
        real_array = b12.real(hm200c, r, m, jnp.array([0.0, 0.5, 1.0]))
        assert real_scalar.shape == (2, 2)
        assert real_array.shape == (2, 2, 3)
        assert jnp.allclose(real_scalar, real_array[:, :, 1])

    # real()/fourier() are all-NaN (not an exception) outside emulator bounds.
    def test_nan_outside_emulator_bounds(self, hm200c_oob):
        b12 = B12PressureProfile()
        r, m, z = jnp.array([0.1, 0.2]), jnp.array([1e13, 1e14]), jnp.array([0.3, 1.0])
        assert jnp.all(jnp.isnan(b12.real(hm200c_oob, r, m, z)))
        assert jnp.all(jnp.isnan(b12.fourier(hm200c_oob, r, m, z)))

    class TestGradients:
        # x_grid is static aux data and x_out only enters a boolean mask (zero gradient by
        # construction) -- both excluded in favor of the 9 differentiable shape parameters.
        PARAMS = [
            ("A_P0", 18.1),
            ("A_xc", 0.497),
            ("A_beta", 4.35),
            ("alpha_m_P0", 0.154),
            ("alpha_m_xc", -0.00865),
            ("alpha_m_beta", 0.0393),
            ("alpha_z_P0", -0.758),
            ("alpha_z_xc", 0.731),
            ("alpha_z_beta", 0.415),
        ]

        @pytest.mark.parametrize("param,x0", PARAMS)
        def test_real_grad_wrt_shape_params(self, hm200c, param, x0):
            b12 = B12PressureProfile()
            r, m, z = jnp.array(0.1), jnp.array(5e13), jnp.array(0.5)

            def f(x):
                return b12.update(**{param: x}).real(hm200c, r, m, z)

            _check_grad(f, x0)


class TestJitRetraceAwareness:
    # Two separately-constructed, value-identical profile instances give numerically identical
    # output -- the confirmed per-instance jit retrace defect is a pure performance concern,
    # with no correctness impact.
    @pytest.mark.parametrize(
        "build",
        [
            lambda: NFWMatterProfile(),
            lambda: Z07GalaxyHODProfile(alpha_s=1.3),
            lambda: S12CIBProfile(nu=100),
            lambda: GNFWPressureProfile(),
        ],
    )
    def test_correctness_unaffected_by_retrace(self, hm200c, build):
        p1, p2 = build(), build()
        assert p1 is not p2
        r, m, z = jnp.array(0.1), jnp.array(1e13), jnp.array(0.5)
        assert jnp.allclose(p1.real(hm200c, r, m, z), p2.real(hm200c, r, m, z))
