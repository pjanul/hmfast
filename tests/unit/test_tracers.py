"""
Unit tests for hmfast.tracers: the profile-type compatibility guardrail
(e.g. CMB lensing must not accept a pressure profile), dndz normalization,
kernel shape/squeeze conventions, z_max truncation, NaN propagation, and
gradients.
"""

import jax
import jax.numpy as jnp
import pytest

from hmfast.halos.profiles import (
    B16DensityProfile,
    GNFWPressureProfile,
    MatterProfile,
    NFWMatterProfile,
    S12CIBProfile,
    Z07GalaxyHODProfile,
)
from hmfast.tracers import (
    CIBTracer,
    CMBLensingTracer,
    GalaxyLensingTracer,
    GalaxyTracer,
    kSZTracer,
    tSZTracer,
)

_SYNTHETIC_DNDZ = (jnp.linspace(0.0, 2.0, 20), jnp.ones(20))

# Each tracer paired with a profile from a genuinely incompatible family.
_WRONG_PROFILE_CASES = [
    (CIBTracer, NFWMatterProfile()),
    (CMBLensingTracer, Z07GalaxyHODProfile()),
    (GalaxyTracer, GNFWPressureProfile()),
    (GalaxyLensingTracer, B16DensityProfile()),
    (kSZTracer, S12CIBProfile(nu=100)),
    (tSZTracer, NFWMatterProfile()),
]

# Each tracer paired with a profile from its own required family (not necessarily the default).
_CORRECT_PROFILE_CASES = [
    (CIBTracer, S12CIBProfile(nu=150)),
    (CMBLensingTracer, NFWMatterProfile()),
    (GalaxyTracer, Z07GalaxyHODProfile()),
    (GalaxyLensingTracer, NFWMatterProfile()),
    (kSZTracer, B16DensityProfile()),
    (tSZTracer, GNFWPressureProfile()),
]


def _build(tracer_cls, profile):
    """Construct a tracer with a valid profile, supplying dndz explicitly where needed."""
    if tracer_cls in (GalaxyTracer, GalaxyLensingTracer):
        return tracer_cls(profile=profile, dndz=_SYNTHETIC_DNDZ)
    return tracer_cls(profile=profile)


def _build_default(tracer_cls, **kwargs):
    """Construct a tracer with its default profile, supplying dndz explicitly where needed."""
    if tracer_cls in (GalaxyTracer, GalaxyLensingTracer):
        return tracer_cls(dndz=_SYNTHETIC_DNDZ, **kwargs)
    return tracer_cls(**kwargs)


_ALL_TRACER_CLASSES = [
    CMBLensingTracer,
    GalaxyLensingTracer,
    GalaxyTracer,
    CIBTracer,
    kSZTracer,
    tSZTracer,
]


class TestTracerProfileGuardrail:
    # Constructing a tracer with a profile from the wrong family raises TypeError.
    @pytest.mark.parametrize("tracer_cls,wrong_profile", _WRONG_PROFILE_CASES)
    def test_rejects_wrong_profile_family_at_construction(
        self, tracer_cls, wrong_profile
    ):
        with pytest.raises(TypeError):
            _build(tracer_cls, wrong_profile)

    # .update(profile=...) routes through the same validating setter and also raises TypeError.
    @pytest.mark.parametrize(
        "tracer_cls,correct_profile,wrong_profile",
        [
            (tc, cp, wp)
            for (tc, cp), (_, wp) in zip(_CORRECT_PROFILE_CASES, _WRONG_PROFILE_CASES)
        ],
    )
    def test_rejects_wrong_profile_family_via_update(
        self, tracer_cls, correct_profile, wrong_profile
    ):
        tracer = _build(tracer_cls, correct_profile)
        with pytest.raises(TypeError):
            tracer.update(profile=wrong_profile)

    # A profile from the tracer's own required family is accepted at construction and via update().
    @pytest.mark.parametrize("tracer_cls,profile", _CORRECT_PROFILE_CASES)
    def test_accepts_correct_profile_family(self, tracer_cls, profile):
        tracer = _build(tracer_cls, profile)
        assert tracer.profile is profile
        tracer2 = tracer.update(profile=profile)
        assert tracer2.profile is profile


class TestDndzNormalization:
    # A default (file-loaded) dndz integrates to 1.
    def test_default_dndz_is_normalized(self):
        try:
            tracer = GalaxyTracer()
        except Exception as exc:
            pytest.skip(f"default GalaxyTracer dndz file not available locally: {exc}")
        z, phi = tracer.dndz
        assert jnp.isclose(jnp.trapezoid(phi, x=z), 1.0)

    # update(dndz=unnormalized_array) re-normalizes explicitly before storing.
    @pytest.mark.parametrize("tracer_cls", [GalaxyTracer, GalaxyLensingTracer])
    def test_update_dndz_renormalizes(self, tracer_cls):
        tracer = tracer_cls(dndz=_SYNTHETIC_DNDZ)
        unnormalized = (jnp.linspace(0.0, 3.0, 10), jnp.linspace(1.0, 5.0, 10))
        tracer2 = tracer.update(dndz=unnormalized)
        z, phi = tracer2.dndz
        assert jnp.isclose(jnp.trapezoid(phi, x=z), 1.0)

    # Direct attribute assignment (tracer.dndz = raw_array) also normalizes via the property setter.
    @pytest.mark.parametrize("tracer_cls", [GalaxyTracer, GalaxyLensingTracer])
    def test_direct_assignment_renormalizes(self, tracer_cls):
        tracer = tracer_cls(dndz=_SYNTHETIC_DNDZ)
        unnormalized = (jnp.linspace(0.0, 3.0, 10), jnp.linspace(1.0, 5.0, 10))
        tracer.dndz = unnormalized
        z, phi = tracer.dndz
        assert jnp.isclose(jnp.trapezoid(phi, x=z), 1.0)


class TestKernelShapeAndSqueeze:
    # kernel() follows the same broadcast-then-squeeze convention as profile real()/fourier().
    @pytest.mark.parametrize("tracer_cls", _ALL_TRACER_CLASSES)
    @pytest.mark.parametrize(
        "z_key,expected_shape",
        [("scalar", ()), ("array1", ()), ("arrayN", (4,))],
    )
    def test_kernel_shape_matrix(
        self, fixed_cosmology, tracer_cls, z_key, expected_shape
    ):
        z_vals = {
            "scalar": jnp.array(0.5),
            "array1": jnp.array([0.5]),
            "arrayN": jnp.array([0.1, 0.5, 1.0, 2.0]),
        }
        tracer = _build_default(tracer_cls)
        out = tracer.kernel(fixed_cosmology, z_vals[z_key])
        assert jnp.shape(out) == expected_shape


class TestKernelZMaxTruncation:
    # kernel() is exactly zero beyond z_max and finite/nonzero just inside it.
    @pytest.mark.parametrize("tracer_cls", [CIBTracer, kSZTracer, tSZTracer])
    def test_truncates_at_z_max(self, fixed_cosmology, tracer_cls):
        tracer = tracer_cls(z_max=3.0)
        assert tracer.kernel(fixed_cosmology, jnp.array(3.5)) == 0.0
        inside = tracer.kernel(fixed_cosmology, jnp.array(2.5))
        assert jnp.isfinite(inside) and inside != 0.0


class TestTracerUpdateAndPytree:
    # update(profile=...) swaps in the new profile and leaves the original tracer untouched.
    def test_update_cmb_lensing_swaps_profile(self):
        original_profile = NFWMatterProfile()
        tracer = CMBLensingTracer(profile=original_profile)
        new_profile = NFWMatterProfile()
        tracer2 = tracer.update(profile=new_profile)
        assert tracer2.profile is new_profile
        assert tracer.profile is original_profile

    # update(z_max=...) changes only z_max, leaving the profile untouched.
    @pytest.mark.parametrize("tracer_cls", [CIBTracer, kSZTracer, tSZTracer])
    def test_update_z_max_preserves_profile(self, tracer_cls):
        tracer = _build_default(tracer_cls, z_max=3.0)
        tracer2 = tracer.update(z_max=4.0)
        assert tracer2.z_max == 4.0
        assert tracer2.profile is tracer.profile

    # update(profile=...) changes only the profile, leaving dndz untouched.
    @pytest.mark.parametrize("tracer_cls", [GalaxyTracer, GalaxyLensingTracer])
    def test_update_profile_preserves_dndz(self, tracer_cls):
        tracer = _build_default(tracer_cls)
        new_profile = (
            NFWMatterProfile()
            if tracer_cls is GalaxyLensingTracer
            else Z07GalaxyHODProfile(alpha_s=1.5)
        )
        tracer2 = tracer.update(profile=new_profile)
        assert tracer2.profile is new_profile
        assert jnp.array_equal(tracer2.dndz[0], tracer.dndz[0])
        assert jnp.array_equal(tracer2.dndz[1], tracer.dndz[1])

    # Every tracer survives a JAX pytree flatten/unflatten round trip unchanged.
    @pytest.mark.parametrize("tracer_cls", _ALL_TRACER_CLASSES)
    def test_pytree_roundtrip(self, tracer_cls):
        tracer = _build_default(tracer_cls)
        leaves, treedef = jax.tree_util.tree_flatten(tracer)
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rt) is type(tracer)
        assert isinstance(rt.profile, type(tracer.profile))


class TestKernelArraySizeChange:
    # kernel(), wrapped in an outer jax.jit, accepts a differently-shaped z array across calls.
    @pytest.mark.parametrize("tracer_cls", _ALL_TRACER_CLASSES)
    def test_array_size_change_across_outer_jit(self, fixed_cosmology, tracer_cls):
        tracer = _build_default(tracer_cls)
        call = jax.jit(lambda t, c, z: t.kernel(c, z))
        out_scalar = call(tracer, fixed_cosmology, jnp.array(0.5))
        out_array = call(tracer, fixed_cosmology, jnp.array([0.0, 0.5, 1.0]))
        assert out_scalar.shape == ()
        assert out_array.shape == (3,)
        assert jnp.allclose(out_scalar, out_array[1])


class TestKernelNaNPropagation:
    # Kernels that genuinely depend on cosmology (chi(z), H(z), or velocity dispersion) go
    # all-NaN under an out-of-bounds cosmology.
    @pytest.mark.parametrize(
        "tracer_cls", [CMBLensingTracer, GalaxyLensingTracer, GalaxyTracer, kSZTracer]
    )
    def test_nan_outside_emulator_bounds(self, out_of_bounds_cosmology, tracer_cls):
        tracer = (
            _build_default(tracer_cls, z_max=3.0)
            if tracer_cls is kSZTracer
            else _build_default(tracer_cls)
        )
        out = tracer.kernel(out_of_bounds_cosmology, jnp.array([0.3, 1.0]))
        assert jnp.all(jnp.isnan(out))

    # tSZ/CIB's kernels are purely analytic in z (and z_max), with no cosmology dependence at
    # all, so they stay finite regardless of the cosmology's validity.
    @pytest.mark.parametrize("tracer_cls", [tSZTracer, CIBTracer])
    def test_unaffected_by_cosmology_bounds(self, out_of_bounds_cosmology, tracer_cls):
        tracer = _build_default(tracer_cls)
        out = tracer.kernel(out_of_bounds_cosmology, jnp.array([0.3, 1.0]))
        assert jnp.all(jnp.isfinite(out))


def _check_grad(f, x0, rtol=1e-2):
    """jax.grad is finite and matches a scale-appropriate finite-difference estimate."""
    x0 = jnp.asarray(x0, dtype=float)
    g_auto = jax.grad(f)(x0)
    assert jnp.isfinite(g_auto)
    eps = jnp.maximum(1e-6, 1e-5 * jnp.abs(x0))
    g_fd = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
    assert jnp.isclose(g_auto, g_fd, rtol=rtol, atol=1e-8)


class TestKernelGradients:
    # jax.grad of kernel() wrt H0 (via cosmology) is finite and matches a finite-difference estimate.
    @pytest.mark.parametrize(
        "tracer_cls", [CMBLensingTracer, GalaxyLensingTracer, GalaxyTracer]
    )
    def test_kernel_grad_wrt_H0(self, fixed_cosmology, tracer_cls):
        tracer = _build_default(tracer_cls)
        z = jnp.array(0.5)

        def f(H0):
            return tracer.kernel(fixed_cosmology.update(H0=H0), z)

        _check_grad(f, fixed_cosmology.H0)

    # kSZTracer needs a finite z_max and a z safely inside velocity_dispersion's valid z range.
    def test_ksz_kernel_grad_wrt_H0(self, fixed_cosmology):
        tracer = kSZTracer(z_max=3.0)
        z = jnp.array(1.0)

        def f(H0):
            return tracer.kernel(fixed_cosmology.update(H0=H0), z)

        _check_grad(f, fixed_cosmology.H0)


class TestGalaxyLensingEfficiency:
    # I_s(z) is exactly zero at/beyond the source distribution's maximum redshift.
    def test_zero_beyond_max_source_z(self, fixed_cosmology):
        tracer = GalaxyLensingTracer(dndz=_SYNTHETIC_DNDZ)
        z_max_source = float(_SYNTHETIC_DNDZ[0][-1])
        I_s = tracer._lensing_efficiency_integral(
            fixed_cosmology, jnp.array([z_max_source, z_max_source + 1.0]), tracer.dndz
        )
        assert jnp.all(I_s == 0.0)

    # I_s(z) decreases monotonically with z (fewer sources remain "behind" the lens).
    def test_monotonically_decreasing(self, fixed_cosmology):
        tracer = GalaxyLensingTracer(dndz=_SYNTHETIC_DNDZ)
        z = jnp.linspace(0.0, 1.9, 10)
        I_s = tracer._lensing_efficiency_integral(fixed_cosmology, z, tracer.dndz)
        assert jnp.all(jnp.diff(I_s) <= 0.0)


class TestMagnificationBias:
    # Default mag_bias is s(z)=2/5, which makes the magnification-bias term vanish
    # (CCL's (1 - 5s/2) weighting is exactly zero at s=2/5).
    def test_default_slope_makes_term_vanish(self):
        tracer = GalaxyTracer(dndz=_SYNTHETIC_DNDZ)
        _, s_vals = tracer.mag_bias
        assert jnp.all(s_vals == 0.4)

    # A nontrivial magnification-bias slope changes the kernel relative to the default.
    def test_nontrivial_slope_changes_kernel(self, fixed_cosmology):
        z = jnp.array([0.3, 0.8, 1.5])
        default_tracer = GalaxyTracer(dndz=_SYNTHETIC_DNDZ)
        biased_tracer = GalaxyTracer(
            dndz=_SYNTHETIC_DNDZ, mag_bias=(jnp.array([0.0, 2.0]), jnp.array([0.6, 0.6]))
        )
        assert not jnp.allclose(
            default_tracer.kernel(fixed_cosmology, z), biased_tracer.kernel(fixed_cosmology, z)
        )

    # jax.grad of kernel() wrt the magnification-bias slope is finite and matches finite differences.
    def test_kernel_grad_wrt_mag_bias_amplitude(self, fixed_cosmology):
        z = jnp.array(0.5)

        def f(s):
            tracer = GalaxyTracer(
                dndz=_SYNTHETIC_DNDZ, mag_bias=(jnp.array([0.0, 2.0]), jnp.array([s, s]))
            )
            return tracer.kernel(fixed_cosmology, z)

        _check_grad(f, 0.6)


class TestIntrinsicAlignment:
    # Default ia_bias is A_IA(z)=0, which makes the intrinsic-alignment term vanish.
    def test_default_amplitude_is_identically_zero(self):
        tracer = GalaxyLensingTracer(dndz=_SYNTHETIC_DNDZ)
        _, a_vals = tracer.ia_bias
        assert jnp.all(a_vals == 0.0)

    # A nontrivial IA amplitude changes the kernel relative to the default.
    def test_nontrivial_amplitude_changes_kernel(self, fixed_cosmology):
        z = jnp.array([0.3, 0.8, 1.5])
        default_tracer = GalaxyLensingTracer(dndz=_SYNTHETIC_DNDZ)
        ia_tracer = GalaxyLensingTracer(
            dndz=_SYNTHETIC_DNDZ, ia_bias=(jnp.array([0.0, 2.0]), jnp.array([1.0, 1.0]))
        )
        assert not jnp.allclose(
            default_tracer.kernel(fixed_cosmology, z), ia_tracer.kernel(fixed_cosmology, z)
        )

    # jax.grad of kernel() wrt the IA amplitude is finite and matches finite differences.
    def test_kernel_grad_wrt_ia_amplitude(self, fixed_cosmology):
        z = jnp.array(0.5)

        def f(a):
            tracer = GalaxyLensingTracer(
                dndz=_SYNTHETIC_DNDZ, ia_bias=(jnp.array([0.0, 2.0]), jnp.array([a, a]))
            )
            return tracer.kernel(fixed_cosmology, z)

        _check_grad(f, 1.0)


class TestDefaultProfileMatchesRequiredType:
    # Each tracer's own default profile is an instance of its own _required_profile_type.
    @pytest.mark.parametrize("tracer_cls", _ALL_TRACER_CLASSES)
    def test_default_profile_matches_required_type(self, tracer_cls):
        tracer = _build_default(tracer_cls)
        assert isinstance(tracer.profile, tracer_cls._required_profile_type)
