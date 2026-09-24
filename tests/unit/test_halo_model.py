"""
Unit tests for hmfast.halos: mass definitions/conversions, halo mass function,
halo bias, concentration, subhalo mass function, and the profile-independent
parts of HaloModel. See tests/benchmarks/benchmark_halo_model.py for CCL cross-checks.
"""
import jax
import jax.numpy as jnp
import pytest

from hmfast.halos.massdef import MassDefinition, mass_translator, _convert_m_delta
from hmfast.halos.massfunc import (
    T08HaloMassFunction,
    T10HaloMassFunction,
    TW10SubHaloMassFunction,
    JvdB14SubHaloMassFunction,
)
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import ConstantConcentration, D08Concentration, B13Concentration
from hmfast.halos import HaloModel


# Shared shape-matrix inputs: scalar/scalar and array1/array1 collapse to (),
# arrayN/scalar broadcasts to (N,), arrayN/arrayM broadcasts to (N, M).
M_VALS = {
    "scalar": jnp.array(1e13),
    "array1": jnp.array([1e13]),
    "arrayN": jnp.geomspace(1e11, 1e14, 5),
}
Z_VALS = {
    "scalar": jnp.array(0.5),
    "array1": jnp.array([0.5]),
    "arrayM": jnp.array([0.0, 0.5, 1.0, 2.0]),
}
SHAPE_MATRIX_CASES = [
    ("scalar", "scalar", ()),
    ("array1", "scalar", ()),
    ("array1", "array1", ()),
    ("arrayN", "scalar", (5,)),
    ("scalar", "arrayM", (4,)),
    ("arrayN", "arrayM", (5, 4)),
]


class TestMassDefinition:
    # 'vir' delta requires reference='critical'; any other reference raises ValueError.
    def test_vir_requires_critical_reference(self):
        with pytest.raises(ValueError):
            MassDefinition(delta="vir", reference="mean")
        MassDefinition(delta="vir", reference="critical")

    # r_delta's output shape follows the scalar/array broadcast-then-squeeze convention.
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_CASES)
    def test_r_delta_shape_matrix(self, fixed_cosmology, mass_def, m_key, z_key, expected_shape):
        r = mass_def.r_delta(fixed_cosmology, M_VALS[m_key], Z_VALS[z_key])
        assert jnp.shape(r) == expected_shape

    # r_delta increases monotonically with mass at fixed redshift.
    def test_r_delta_increases_with_mass(self, fixed_cosmology, mass_def):
        m = jnp.geomspace(1e10, 1e16, 20)
        r = mass_def.r_delta(fixed_cosmology, m, 0.5)
        assert jnp.all(jnp.diff(r) > 0)

    # r_delta shrinks as the overdensity threshold increases (200c vs 500c).
    def test_r_delta_decreases_with_delta(self, fixed_cosmology):
        m, z = jnp.array(1e13), jnp.array(0.5)
        r200 = MassDefinition(200, "critical").r_delta(fixed_cosmology, m, z)
        r500 = MassDefinition(500, "critical").r_delta(fixed_cosmology, m, z)
        assert r500 < r200

    # MassDefinition survives a JAX pytree flatten/unflatten round trip unchanged.
    def test_pytree_roundtrip(self, mass_def):
        leaves, treedef = jax.tree_util.tree_flatten(mass_def)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rt.delta == mass_def.delta
        assert rt.reference == mass_def.reference


class TestMassConversion:
    # mass_translator(d, d, ...) is the identity map, regardless of the concentration model passed.
    def test_identity_translator(self, fixed_cosmology, mass_def):
        f = mass_translator(mass_def, mass_def, D08Concentration())
        m, z = jnp.geomspace(1e11, 1e15, 10), jnp.array([0.0, 0.5, 1.0])
        r = f(fixed_cosmology, m, z)
        assert jnp.allclose(r, jnp.broadcast_to(m[:, None], (len(m), len(z))))

    # Converting mass_def A -> B -> A recovers the original mass within tolerance.
    @pytest.mark.parametrize("def_a,def_b", [
        (MassDefinition(200, "critical"), MassDefinition(200, "mean")),
        (MassDefinition(200, "critical"), MassDefinition("vir", "critical")),
        (MassDefinition(200, "mean"), MassDefinition("vir", "critical")),
    ])
    @pytest.mark.parametrize("z_val", [0.0, 0.5, 1.0, 2.0])
    def test_roundtrip_conversion(self, fixed_cosmology, def_a, def_b, z_val):
        m = jnp.geomspace(1e10, 1e16, 50)
        conc = D08Concentration()
        f_fwd = mass_translator(def_a, def_b, conc)
        f_bwd = mass_translator(def_b, def_a, conc)
        m_converted = f_fwd(fixed_cosmology, m, z_val)
        m_roundtrip = f_bwd(fixed_cosmology, m_converted, z_val)
        assert jnp.allclose(m_roundtrip, m, rtol=0.03)

    # _convert_m_delta's output shape follows the scalar/array broadcast-then-squeeze convention.
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_CASES)
    def test_convert_m_delta_shape_matrix(self, fixed_cosmology, m_key, z_key, expected_shape):
        md_old, md_new = MassDefinition(200, "critical"), MassDefinition(200, "mean")
        c_old = D08Concentration().c_delta(fixed_cosmology, M_VALS[m_key], Z_VALS[z_key], md_old)
        r = _convert_m_delta(fixed_cosmology, M_VALS[m_key], Z_VALS[z_key], md_old, md_new, c_old)
        assert jnp.shape(r) == expected_shape


class TestHaloMassFunction:
    # dn/dlnM is positive everywhere and decreases monotonically with mass.
    @pytest.mark.parametrize("hmf_cls", [T08HaloMassFunction, T10HaloMassFunction])
    def test_positive_and_monotonic_decreasing_in_mass(self, fixed_cosmology, hmf_cls, mass_def, m_grid, z_grid):
        dndlnm = hmf_cls().dndlnm(fixed_cosmology, m_grid, z_grid, mass_def)
        assert jnp.all(dndlnm > 0)
        assert jnp.all(jnp.diff(dndlnm, axis=0) <= 0)

    # dndlnm's output shape follows the scalar/array broadcast-then-squeeze convention.
    @pytest.mark.parametrize("hmf_cls", [T08HaloMassFunction, T10HaloMassFunction])
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_CASES)
    def test_shape_matrix(self, fixed_cosmology, hmf_cls, mass_def, m_key, z_key, expected_shape):
        r = hmf_cls().dndlnm(fixed_cosmology, M_VALS[m_key], Z_VALS[z_key], mass_def)
        assert jnp.shape(r) == expected_shape

    # The same jitted dndlnm call accepts a differently-shaped z array across successive calls.
    @pytest.mark.parametrize("hmf_cls", [T08HaloMassFunction, T10HaloMassFunction])
    def test_array_size_change_across_jitted_calls(self, fixed_cosmology, hmf_cls):
        m = jnp.geomspace(1e11, 1e14, 5)
        md = MassDefinition(200, "critical")
        hmf = hmf_cls()
        r_scalar = hmf.dndlnm(fixed_cosmology, m, 0.5, md)
        r_array = hmf.dndlnm(fixed_cosmology, m, jnp.array([0.0, 0.5, 1.0]), md)
        assert r_scalar.shape == (5,)
        assert r_array.shape == (5, 3)
        assert jnp.allclose(r_scalar, r_array[:, 1])

    # dndlnm is all-NaN (not an exception) when the cosmology is outside the emulator's trained bounds.
    @pytest.mark.parametrize("hmf_cls", [T08HaloMassFunction, T10HaloMassFunction])
    def test_nan_outside_emulator_bounds(self, out_of_bounds_cosmology, hmf_cls, m_grid, z_grid):
        md = MassDefinition(200, "mean")
        dndlnm = hmf_cls().dndlnm(out_of_bounds_cosmology, m_grid, z_grid, md)
        assert jnp.all(jnp.isnan(dndlnm))


class TestHaloBias:
    # bias() raises ValueError for any order other than 1 or 2.
    def test_invalid_order_raises(self, fixed_cosmology, m_grid, z_grid):
        md = MassDefinition(200, "mean")
        with pytest.raises(ValueError):
            T10HaloBias().bias(fixed_cosmology, m_grid, z_grid, md, order=3)

    # bias's output shape follows the scalar/array broadcast-then-squeeze convention, for order 1 and 2.
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_CASES)
    def test_shape_matrix(self, fixed_cosmology, mass_def, order, m_key, z_key, expected_shape):
        r = T10HaloBias().bias(fixed_cosmology, M_VALS[m_key], Z_VALS[z_key], mass_def, order=order)
        assert jnp.shape(r) == expected_shape

    # Linear bias (order=1) is positive and increases monotonically with mass.
    def test_order_1_positive_and_monotonic_increasing_in_mass(self, fixed_cosmology, mass_def, m_grid, z_grid):
        b1 = T10HaloBias().bias(fixed_cosmology, m_grid, z_grid, mass_def, order=1)
        assert jnp.all(jnp.diff(b1, axis=0) >= 0)

    # Calling bias() with order=1, then order=2, then order=1 again gives distinct, correctly-retraced results.
    def test_sequential_jit_calls_with_different_order_give_distinct_results(self, fixed_cosmology, m_grid, z_grid):
        md = MassDefinition(200, "mean")
        bias_obj = T10HaloBias()
        b1 = bias_obj.bias(fixed_cosmology, m_grid, z_grid, md, order=1)
        b2 = bias_obj.bias(fixed_cosmology, m_grid, z_grid, md, order=2)
        assert not jnp.allclose(b1, b2)
        b1_again = bias_obj.bias(fixed_cosmology, m_grid, z_grid, md, order=1)
        assert jnp.array_equal(b1, b1_again)

    # bias() is all-NaN when the cosmology is outside the emulator's trained bounds.
    def test_nan_outside_emulator_bounds(self, out_of_bounds_cosmology, m_grid, z_grid):
        md = MassDefinition(200, "mean")
        b1 = T10HaloBias().bias(out_of_bounds_cosmology, m_grid, z_grid, md, order=1)
        assert jnp.all(jnp.isnan(b1))


class TestConcentration:
    # D08/B13 raise ValueError for mass definitions outside their calibrated set (e.g. 500c).
    @pytest.mark.parametrize("conc_cls", [D08Concentration, B13Concentration])
    def test_raises_for_unsupported_mass_def(self, fixed_cosmology, conc_cls):
        m, z = jnp.array([1e12, 1e14]), jnp.array([0.5])
        with pytest.raises(ValueError):
            conc_cls().c_delta(fixed_cosmology, m, z, MassDefinition(500, "critical"))

    # c_delta's output shape follows the scalar/array broadcast-then-squeeze convention, for D08 and B13.
    @pytest.mark.parametrize("conc_cls", [D08Concentration, B13Concentration])
    @pytest.mark.parametrize("delta,reference", [(200, "critical"), (200, "mean"), ("vir", "critical")])
    @pytest.mark.parametrize("m_key,z_key,expected_shape", SHAPE_MATRIX_CASES)
    def test_shape_matrix(self, fixed_cosmology, conc_cls, delta, reference, m_key, z_key, expected_shape):
        md = MassDefinition(delta, reference)
        r = conc_cls().c_delta(fixed_cosmology, M_VALS[m_key], Z_VALS[z_key], md)
        assert jnp.shape(r) == expected_shape

    # ConstantConcentration returns the fixed value broadcast to shape, regardless of mass_def.
    def test_constant_concentration_ignores_mass_def(self, fixed_cosmology, mass_def):
        m, z = jnp.array([1e12, 1e14]), jnp.array([0.5])
        c = ConstantConcentration(4.5)
        r = c.c_delta(fixed_cosmology, m, z, mass_def)
        assert jnp.allclose(r, 4.5)

    # B13's c_delta is all-NaN outside emulator bounds (unlike D08, which only reads H0 and stays finite).
    def test_b13_nan_outside_emulator_bounds(self, out_of_bounds_cosmology):
        m, z = jnp.array([1e12, 1e14]), jnp.array([0.5])
        r = B13Concentration().c_delta(out_of_bounds_cosmology, m, z, MassDefinition(200, "critical"))
        assert jnp.all(jnp.isnan(r))


class TestSubHaloMassFunction:
    # dN/dlnmu is positive and decreases monotonically with mu = m_sub/m_host.
    @pytest.mark.parametrize("subhmf_cls", [TW10SubHaloMassFunction, JvdB14SubHaloMassFunction])
    def test_positive_and_monotonic_decreasing_in_mu(self, subhmf_cls, fixed_cosmology):
        mu = jnp.geomspace(1e-4, 1.0, 20)
        r = subhmf_cls().dndlnmu(fixed_cosmology, 1.0, mu)
        assert jnp.all(r > 0)
        assert jnp.all(jnp.diff(r) <= 0)

    # dndlnmu's output shape for scalar/array m_host, m_sub (no outer-product broadcast, unlike other components).
    @pytest.mark.parametrize("subhmf_cls", [TW10SubHaloMassFunction, JvdB14SubHaloMassFunction])
    @pytest.mark.parametrize("m_host,m_sub,expected_shape", [
        (jnp.array(1e14), jnp.array(1e12), ()),
        (jnp.array([1e14, 1e13]), jnp.array(1e12), (2,)),
        (jnp.array(1e14), jnp.array([1e12, 1e11, 1e10]), (3,)),
        (jnp.array([1e14, 1e13, 1e12]), jnp.array([1e12, 1e11, 1e10]), (3,)),
    ])
    def test_shape_matrix(self, subhmf_cls, fixed_cosmology, m_host, m_sub, expected_shape):
        r = subhmf_cls().dndlnmu(fixed_cosmology, m_host, m_sub)
        assert jnp.shape(r) == expected_shape

    # dndlnmu raises when m_host and m_sub are non-scalar arrays of different, non-broadcastable lengths.
    @pytest.mark.parametrize("subhmf_cls", [TW10SubHaloMassFunction, JvdB14SubHaloMassFunction])
    def test_mismatched_array_shapes_raise(self, subhmf_cls, fixed_cosmology):
        with pytest.raises(Exception):
            subhmf_cls().dndlnmu(fixed_cosmology, jnp.array([1e14, 1e13]), jnp.array([1e12, 1e11, 1e10]))

    # dndlnmu gives identical results regardless of which cosmology object is passed in.
    @pytest.mark.parametrize("subhmf_cls", [TW10SubHaloMassFunction, JvdB14SubHaloMassFunction])
    def test_independent_of_cosmology(self, subhmf_cls, fixed_cosmology, out_of_bounds_cosmology):
        m_host, m_sub = jnp.array([1e14, 1e13]), jnp.array([1e12, 1e11])
        r1 = subhmf_cls().dndlnmu(fixed_cosmology, m_host, m_sub)
        r2 = subhmf_cls().dndlnmu(out_of_bounds_cosmology, m_host, m_sub)
        assert jnp.array_equal(r1, r2)


class TestHaloModelCore:
    # HaloModel()'s default mass_def, component models, m_range/n_m, and hm_consistency match the documented defaults.
    def test_constructor_defaults(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)
        assert hm.mass_def.delta == 200 and hm.mass_def.reference == "critical"
        assert isinstance(hm.halo_mass_function, T08HaloMassFunction)
        assert isinstance(hm.halo_bias, T10HaloBias)
        assert isinstance(hm.subhalo_mass_function, TW10SubHaloMassFunction)
        assert isinstance(hm.concentration, D08Concentration)
        assert hm.hm_consistency is True
        assert hm.n_m == 100
        assert (float(hm.m_range[0]), float(hm.m_range[1])) == (1e10, 1e15)

    # update() returns a new HaloModel, changing only the field passed and leaving all others identical.
    def test_update_returns_new_object_preserving_untouched_fields(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)
        hm2 = hm.update(mass_def=MassDefinition(500, "critical"))
        assert hm2 is not hm
        assert hm2.mass_def.delta == 500 and hm2.mass_def.reference == "critical"
        assert hm2.halo_mass_function is hm.halo_mass_function
        assert hm2.halo_bias is hm.halo_bias
        assert hm2.subhalo_mass_function is hm.subhalo_mass_function
        assert hm2.concentration is hm.concentration
        assert hm2.hm_consistency == hm.hm_consistency

    # update() can replace halo_mass_function or concentration independently of one another.
    def test_update_each_component_individually(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)
        new_hmf = T10HaloMassFunction()
        new_conc = B13Concentration()
        assert hm.update(halo_mass_function=new_hmf).halo_mass_function is new_hmf
        assert hm.update(concentration=new_conc).concentration is new_conc

    # update(m_range=..., n_m=...) replaces the mass-integral bounds and node count.
    def test_update_replaces_m_range_and_n_m(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)
        hm2 = hm.update(m_range=(1e11, 1e14), n_m=50)
        assert (float(hm2.m_range[0]), float(hm2.m_range[1])) == (1e11, 1e14)
        assert hm2.n_m == 50

    # HaloModel survives a JAX pytree flatten/unflatten round trip unchanged.
    def test_pytree_roundtrip(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology, mass_def=MassDefinition(500, "critical"))
        leaves, treedef = jax.tree_util.tree_flatten(hm)
        hm_rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert hm_rt.mass_def.delta == hm.mass_def.delta
        assert hm_rt.mass_def.reference == hm.mass_def.reference
        assert jnp.allclose(hm_rt.m_range[0], hm.m_range[0])
        assert jnp.allclose(hm_rt.m_range[1], hm.m_range[1])
        assert hm_rt.n_m == hm.n_m

    # _counter_terms returns shape-(Nz,) arrays (even for scalar z, which never gets squeezed away) with n_min > 0.
    def test_counter_terms_shape_and_sign(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)
        n_min, b1_min, b2_min = hm._counter_terms(jnp.array(0.5))
        assert n_min.shape == (1,)
        assert b1_min.shape == (1,)
        assert b2_min.shape == (1,)
        assert n_min[0] > 0

        n_min_arr, _, _ = hm._counter_terms(jnp.array([0.0, 0.5, 1.0]))
        assert n_min_arr.shape == (3,)

    # _counter_terms works under an outer jax.jit and across successive calls with differently-shaped z.
    def test_counter_terms_under_outer_jit_and_array_size_change(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)

        @jax.jit
        def call(hm, z):
            return hm._counter_terms(z)

        n_min_scalar, _, _ = call(hm, 0.5)
        assert n_min_scalar.shape == (1,)
        n_min_array, _, _ = call(hm, jnp.array([0.0, 0.5, 1.0]))
        assert n_min_array.shape == (3,)


class TestGradients:
    EPS = 1e-3
    RTOL = 1e-3

    def _check(self, f, x0):
        g_auto = jax.grad(f)(x0)
        assert jnp.isfinite(g_auto)
        g_fd = (f(x0 + self.EPS) - f(x0 - self.EPS)) / (2 * self.EPS)
        assert jnp.isclose(g_auto, g_fd, rtol=self.RTOL)

    # jax.grad of dndlnm wrt H0 is finite and matches a finite-difference estimate.
    def test_dndlnm_grad_wrt_H0(self, fixed_cosmology):
        m, z, md = jnp.array(1e13), jnp.array(0.5), MassDefinition(200, "critical")

        def f(H0):
            return T08HaloMassFunction().dndlnm(fixed_cosmology.update(H0=H0), m, z, md)

        self._check(f, 67.5)

    # jax.grad of bias wrt H0 is finite and matches a finite-difference estimate.
    def test_bias_grad_wrt_H0(self, fixed_cosmology):
        m, z, md = jnp.array(1e13), jnp.array(0.5), MassDefinition(200, "critical")

        def f(H0):
            return T10HaloBias().bias(fixed_cosmology.update(H0=H0), m, z, md, order=1)

        self._check(f, 67.5)

    # jax.grad of D08 concentration wrt H0 is finite and matches a finite-difference estimate.
    def test_concentration_grad_wrt_H0(self, fixed_cosmology):
        m, z, md = jnp.array(1e13), jnp.array(0.5), MassDefinition(200, "critical")

        def f(H0):
            return D08Concentration().c_delta(fixed_cosmology.update(H0=H0), m, z, md)

        self._check(f, 67.5)

    # jax.grad of r_delta wrt H0 is finite and matches a finite-difference estimate.
    def test_r_delta_grad_wrt_H0(self, fixed_cosmology):
        m, z, md = jnp.array(1e13), jnp.array(0.5), MassDefinition(200, "critical")

        def f(H0):
            return md.r_delta(fixed_cosmology.update(H0=H0), m, z)

        self._check(f, 67.5)

    # jax.grad through mass_translator's Newton solve wrt H0 is finite and matches a finite-difference estimate.
    def test_mass_translator_grad_wrt_H0(self, fixed_cosmology):
        m, z = jnp.array(1e13), jnp.array(0.5)
        mt = mass_translator(MassDefinition(200, "critical"), MassDefinition(200, "mean"), D08Concentration())

        def f(H0):
            return mt(fixed_cosmology.update(H0=H0), m, z)

        self._check(f, 67.5)

    # jax.grad of _counter_terms' n_min wrt H0 is finite and matches a finite-difference estimate.
    def test_counter_terms_grad_wrt_H0(self, fixed_cosmology):
        hm = HaloModel(cosmology=fixed_cosmology)
        z = jnp.array(0.5)

        def f(H0):
            hm2 = hm.update(cosmology=hm.cosmology.update(H0=H0))
            n_min, _, _ = hm2._counter_terms(z)
            return n_min[0]

        self._check(f, 67.5)
