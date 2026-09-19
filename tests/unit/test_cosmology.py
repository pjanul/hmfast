"""
Unit tests for hmfast.cosmology.Cosmology: construction/validation, the full
background/growth/power-spectrum/CMB API, NaN/bounds behavior, extrapolate_z,
update()/pytree behavior, gradients, and light coverage of the non-lcdm
emulator sets. Primary focus is lcdm:v1. No external ground truth here -- see
tests/benchmarks/benchmark_cosmology.py for CCL cross-checks.
"""
import jax
import jax.numpy as jnp
import pytest

from hmfast.cosmology import Cosmology


def _construct_or_skip(emulator_set, **kwargs):
    try:
        return Cosmology(emulator_set=emulator_set, **kwargs)
    except Exception as exc:
        pytest.skip(f"{emulator_set} emulator files not available locally: {exc}")


def _rel_grad_check(f, x0, rel_eps=1e-5, rtol=1e-3):
    eps = abs(x0) * rel_eps
    g = jax.grad(f)(x0)
    assert jnp.isfinite(g)
    fd = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
    if fd == 0:
        assert g == 0
    else:
        assert jnp.isclose(g, fd, rtol=rtol)


class TestConstruction:
    # An unrecognized emulator_set raises ValueError listing the allowed values.
    def test_invalid_emulator_set_raises(self):
        with pytest.raises(ValueError):
            Cosmology(emulator_set="not-a-real-set")

    # The shared fixed_cosmology fixture is wired to lcdm:v1 as expected.
    def test_default_construction_uses_lcdm_v1(self, fixed_cosmology):
        assert fixed_cosmology.emulator_set == "lcdm:v1"


class TestBackgroundQuantities:
    # hubble_parameter/angular_diameter_distance/critical_density/omega_m/comoving_volume_element
    # all follow the scalar-z -> () , array-z -> (Nz,) shape convention.
    @pytest.mark.parametrize("method", [
        "hubble_parameter", "angular_diameter_distance", "critical_density",
        "omega_m", "comoving_volume_element",
    ])
    def test_shape_matrix(self, fixed_cosmology, method):
        f = getattr(fixed_cosmology, method)
        assert jnp.shape(f(jnp.array(0.5))) == ()
        assert jnp.shape(f(jnp.array([0.0, 0.5, 1.0, 2.0]))) == (4,)

    # H(z) increases monotonically with redshift.
    def test_hubble_parameter_increases_with_redshift(self, fixed_cosmology):
        Hz = fixed_cosmology.hubble_parameter(jnp.linspace(0.01, 5.0, 30))
        assert jnp.all(jnp.diff(Hz) > 0)

    # critical_density(z) increases monotonically with redshift (tracks H(z)^2).
    def test_critical_density_increases_with_redshift(self, fixed_cosmology):
        rho = fixed_cosmology.critical_density(jnp.linspace(0.01, 5.0, 30))
        assert jnp.all(jnp.diff(rho) > 0)

    # omega_m(z) increases monotonically over z in [0,10] and approaches ~1 (matter domination);
    # deliberately not asserted at very high z, where the radiation term makes it turn over.
    def test_omega_m_increases_toward_one(self, fixed_cosmology):
        om = fixed_cosmology.omega_m(jnp.linspace(0.0, 10.0, 30))
        assert jnp.all(jnp.diff(om) > 0)
        assert om[-1] > 0.95

    # comoving_volume_element(0) == 0 exactly, since D_A(0) = 0.
    def test_comoving_volume_element_zero_at_z_zero(self, fixed_cosmology):
        assert fixed_cosmology.comoving_volume_element(jnp.array(0.0)) == 0.0


class TestGrowthAndPerturbations:
    # growth_factor is normalized to D(0)=1 and decreases monotonically with redshift.
    def test_growth_factor_normalized_and_decreasing(self, fixed_cosmology):
        assert jnp.isclose(fixed_cosmology.growth_factor(jnp.array(0.0)), 1.0)
        D = fixed_cosmology.growth_factor(jnp.linspace(0.0, 3.0, 20))
        assert jnp.all(D > 0)
        assert jnp.all(jnp.diff(D) < 0)

    # growth_rate f(z) stays within its expected physical range over z in [0,3].
    def test_growth_rate_in_sane_range(self, fixed_cosmology):
        f = fixed_cosmology.growth_rate(jnp.linspace(0.0, 3.0, 20))
        assert jnp.all((f > 0) & (f < 1.5))

    # delta_c gives a value close to the classic 1.686 at z=0 for all three supported prescriptions
    # (they differ slightly from each other by design -- NS97 sits closer to 1.676, EdS/EdS_approx to 1.686).
    @pytest.mark.parametrize("prescription", ["EdS", "EdS_approx", "NS97"])
    def test_delta_c_prescriptions(self, fixed_cosmology, prescription):
        dc = fixed_cosmology.delta_c(jnp.array(0.0), prescription=prescription)
        assert jnp.isclose(dc, 1.686, atol=0.02)

    def test_delta_c_invalid_prescription_raises(self, fixed_cosmology):
        with pytest.raises(ValueError):
            fixed_cosmology.delta_c(jnp.array(0.0), prescription="not-a-prescription")

    # sigma_m(M,z) is positive and decreases monotonically with mass at fixed z.
    def test_sigma_m_positive_and_decreasing(self, fixed_cosmology):
        sm = fixed_cosmology.sigma_m(jnp.geomspace(1e10, 1e16, 20), jnp.array(0.5))
        assert jnp.all(sm > 0)
        assert jnp.all(jnp.diff(sm) < 0)

    # sigma_r(R,z) is positive and decreases monotonically with radius at fixed z.
    def test_sigma_r_positive_and_decreasing(self, fixed_cosmology):
        sr = fixed_cosmology.sigma_r(jnp.geomspace(1.0, 100.0, 20), jnp.array(0.5))
        assert jnp.all(sr > 0)
        assert jnp.all(jnp.diff(sr) < 0)

    # sigma_m(M,z) and sigma_r(R,z) agree for R/M related by M = 4/3 pi rho_mean R^3 (same underlying transform).
    def test_sigma_m_sigma_r_cross_consistency(self, fixed_cosmology):
        r = jnp.geomspace(1.0, 100.0, 20)
        z = jnp.array(0.5)
        cparams = fixed_cosmology._cosmo_params()
        rho_mean_0 = cparams["Omega0_cb"] * cparams["Rho_crit_0"]
        m_from_r = 4.0 / 3.0 * jnp.pi * rho_mean_0 * r**3
        sigma_m_at_r = fixed_cosmology.sigma_m(m_from_r, z)
        sigma_r = fixed_cosmology.sigma_r(r, z)
        assert jnp.allclose(sigma_m_at_r, sigma_r, rtol=1e-4)

    # sigma8(0) and derived_parameters()['sigma8'] agree closely despite coming from two independently-trained NNs.
    def test_sigma8_matches_derived_parameters(self, fixed_cosmology):
        s8_direct = fixed_cosmology.sigma8(jnp.array(0.0))
        s8_derived = fixed_cosmology.derived_parameters()["sigma8"]
        assert jnp.isclose(s8_direct, s8_derived, rtol=1e-3)

    # Nonlinear P(k) is at least as large as linear P(k) at high k (nonlinear boost), both positive.
    def test_pk_nonlinear_at_least_linear_at_high_k(self, fixed_cosmology):
        k = jnp.geomspace(1e-2, 10.0, 30)
        pk_lin = fixed_cosmology.pk(k, jnp.array(0.0), linear=True)
        pk_nl = fixed_cosmology.pk(k, jnp.array(0.0), linear=False)
        assert jnp.all(pk_lin > 0) and jnp.all(pk_nl > 0)
        assert jnp.all(pk_nl[-5:] >= pk_lin[-5:])

    # sigma2_b_disc is positive with the expected shape.
    def test_sigma2_b_disc_positive(self, fixed_cosmology):
        s2b = fixed_cosmology.sigma2_b_disc(jnp.array([0.0, 0.5, 1.0]), f_sky=0.5)
        assert jnp.all(s2b > 0) and s2b.shape == (3,)

    # velocity_dispersion is positive with the expected shape.
    def test_velocity_dispersion_positive(self, fixed_cosmology):
        vd = fixed_cosmology.velocity_dispersion(jnp.array([0.0, 0.5, 1.0]))
        assert jnp.all(vd > 0) and vd.shape == (3,)


class TestCMB:
    # TT/EE/PP are non-negative; TE is allowed to be negative (it isn't log-emulated like the others).
    @pytest.mark.parametrize("spectrum_type,allow_negative", [
        ("TT", False), ("EE", False), ("TE", True), ("PP", False),
    ])
    def test_shape_and_sign(self, fixed_cosmology, spectrum_type, allow_negative):
        l = jnp.array([10.0, 100.0, 1000.0])
        cl = fixed_cosmology.cl(spectrum_type, l)
        assert cl.shape == (3,)
        if not allow_negative:
            assert jnp.all(cl > 0)

    # cl()'s type argument is case-insensitive.
    def test_case_insensitive_type(self, fixed_cosmology):
        l = jnp.array([10.0, 100.0])
        assert jnp.allclose(fixed_cosmology.cl("tt", l), fixed_cosmology.cl("TT", l))

    # l values outside the emulator's valid multipole range return NaN (both below and above).
    def test_out_of_range_l_returns_nan(self, fixed_cosmology):
        assert jnp.isnan(fixed_cosmology.cl("TT", jnp.array(1.0)))
        assert jnp.isnan(fixed_cosmology.cl("TT", jnp.array(1e7)))

    # An unsupported spectrum type (including "BB", which has no working code path) raises ValueError.
    @pytest.mark.parametrize("bad_type", ["BB", "XX"])
    def test_unsupported_type_raises(self, fixed_cosmology, bad_type):
        with pytest.raises(ValueError):
            fixed_cosmology.cl(bad_type, jnp.array([10.0]))

    # TT is genuinely lazy: absent from _emu before the first cl("TT",...) call, present after.
    def test_lazy_loading(self):
        cosmo = Cosmology(emulator_set="lcdm:v1", H0=67.5, omega_cdm=0.12, omega_b=0.022, A_s=2.1e-9, n_s=0.965)
        assert "TT" not in cosmo._emu
        cosmo.cl("TT", jnp.array([10.0]))
        assert "TT" in cosmo._emu


class TestDerivedParameters:
    # derived_parameters() returns all 14 documented keys, all finite for an in-bounds cosmology.
    def test_all_keys_present_and_finite(self, fixed_cosmology):
        dp = fixed_cosmology.derived_parameters()
        expected_keys = {
            "100*theta_s", "sigma8", "YHe", "z_reio", "Neff", "tau_rec", "z_rec",
            "rs_rec", "chi_rec", "tau_star", "z_star", "rs_star", "chi_star", "rs_drag",
        }
        assert expected_keys <= dp.keys()
        for value in dp.values():
            assert jnp.isfinite(jnp.asarray(value))

    # Recombination happens well before reionization looking backward in time: z_star > z_reio > 0.
    def test_z_star_and_z_reio_ordering(self, fixed_cosmology):
        dp = fixed_cosmology.derived_parameters()
        assert dp["z_star"] > dp["z_reio"] > 0


class TestBoundsAndNaN:
    # hubble_parameter/angular_diameter_distance/sigma8/pk/cl/omega_m are all-NaN for an out-of-bounds cosmology.
    def test_enforce_bounds_gated_functions_all_nan(self, out_of_bounds_cosmology):
        z = jnp.array([0.0, 0.5, 1.0])
        assert jnp.all(jnp.isnan(out_of_bounds_cosmology.hubble_parameter(z)))
        # omega_m is rho_m(z)/rho_crit(z), built from H(z), so it inherits the NaN gate.
        assert jnp.all(jnp.isnan(out_of_bounds_cosmology.omega_m(z)))
        assert jnp.all(jnp.isnan(out_of_bounds_cosmology.angular_diameter_distance(z)))
        assert jnp.all(jnp.isnan(out_of_bounds_cosmology.sigma8(z)))
        assert jnp.all(jnp.isnan(out_of_bounds_cosmology.pk(jnp.geomspace(1e-2, 1, 5), z, linear=True)))
        assert jnp.all(jnp.isnan(out_of_bounds_cosmology.cl("TT", jnp.array([10.0, 100.0]))))

    # derived_parameters() is all-NaN for an out-of-bounds cosmology too (masked inline, not via _enforce_bounds).
    def test_derived_parameters_all_nan_out_of_bounds(self, out_of_bounds_cosmology):
        dp = out_of_bounds_cosmology.derived_parameters()
        for value in dp.values():
            assert jnp.all(jnp.isnan(jnp.asarray(value)))

    # delta_c(z) is purely analytic and stays finite even for an out-of-bounds cosmology.
    def test_delta_c_stays_finite_out_of_bounds(self, out_of_bounds_cosmology):
        z = jnp.array([0.0, 0.5, 1.0])
        assert jnp.all(jnp.isfinite(out_of_bounds_cosmology.delta_c(z)))


class TestExtrapolateZ:
    # Without extrapolate_z, hubble_parameter/angular_diameter_distance are NaN beyond the background grid (z_max=20).
    def test_hz_da_nan_beyond_grid_without_extrapolation(self, fixed_cosmology):
        z_beyond = jnp.array(float(fixed_cosmology._z_grid_bg()[-1]) + 5.0)
        assert jnp.isnan(fixed_cosmology.hubble_parameter(z_beyond))
        assert jnp.isnan(fixed_cosmology.angular_diameter_distance(z_beyond))

    # With extrapolate_z=True, both are finite beyond the grid and exactly continuous with the non-extrapolated value at z_max.
    def test_hz_da_finite_and_continuous_with_extrapolation(self, fixed_cosmology):
        cosmo_ext = fixed_cosmology.update(extrapolate_z=True)
        z_max = float(fixed_cosmology._z_grid_bg()[-1])
        z_beyond = jnp.array(z_max + 5.0)
        assert jnp.isfinite(cosmo_ext.hubble_parameter(z_beyond))
        assert jnp.isfinite(cosmo_ext.angular_diameter_distance(z_beyond))
        assert jnp.isclose(cosmo_ext.hubble_parameter(jnp.array(z_max)), fixed_cosmology.hubble_parameter(jnp.array(z_max)))
        assert jnp.isclose(cosmo_ext.angular_diameter_distance(jnp.array(z_max)), fixed_cosmology.angular_diameter_distance(jnp.array(z_max)))

    # pk is NaN beyond its grid without extrapolate_z; with it, a finite growth-ratio-rescaled value instead.
    def test_pk_nan_beyond_grid_without_extrapolation_finite_with_it(self, fixed_cosmology):
        cosmo_ext = fixed_cosmology.update(extrapolate_z=True)
        z_beyond = jnp.array(float(fixed_cosmology._z_grid_pk()[-1]) + 2.0)
        k = jnp.array([0.05])
        pk_unflagged = fixed_cosmology.pk(k, z_beyond, linear=True)
        pk_flagged = cosmo_ext.pk(k, z_beyond, linear=True)
        assert jnp.isnan(pk_unflagged)
        assert jnp.isfinite(pk_flagged)

    # growth_factor is NaN beyond its grid without extrapolate_z (plain jnp.interp with
    # left/right=nan -- it used to silently clamp to the z_max value instead, which was
    # judged worse than NaN since it looks like a real, if wrong, value);
    # extrapolate_z=True instead integrates the growth ODE forward, giving a finite value.
    def test_growth_factor_nan_beyond_grid_without_extrapolation_finite_with_it(self, fixed_cosmology):
        cosmo_ext = fixed_cosmology.update(extrapolate_z=True)
        z_max = float(fixed_cosmology._z_grid_pk()[-1])
        z_beyond = jnp.array(z_max + 2.0)
        D_unflagged = fixed_cosmology.growth_factor(z_beyond)
        D_flagged = cosmo_ext.growth_factor(z_beyond)
        assert jnp.isnan(D_unflagged)
        assert jnp.isfinite(D_flagged)

    # growth_rate/sigma8/velocity_dispersion have no extrapolation branch at all: still NaN
    # beyond their grid regardless of extrapolate_z.
    def test_growth_extrapolates_but_sigma8_velocity_dispersion_do_not(self, fixed_cosmology):
        cosmo_ext = fixed_cosmology.update(extrapolate_z=True)
        z_beyond = jnp.array(float(fixed_cosmology._z_grid_pk()[-1]) + 2.0)
        # growth_factor/growth_rate honour extrapolate_z; without it they stay NaN.
        assert jnp.isnan(fixed_cosmology.growth_rate(z_beyond))
        assert jnp.isfinite(cosmo_ext.growth_rate(z_beyond))
        assert jnp.isfinite(cosmo_ext.growth_factor(z_beyond))
        # velocity_dispersion has no extrapolation branch, so it is NaN either way.
        assert jnp.isnan(cosmo_ext.velocity_dispersion(z_beyond))
        z_beyond_bg = jnp.array(float(fixed_cosmology._z_grid_bg()[-1]) + 5.0)
        assert jnp.isnan(cosmo_ext.sigma8(z_beyond_bg))


class TestUpdateAndPytree:
    # update() returns a new instance, changes only the given field, and reuses the cached emulator dict (no reload).
    def test_update_returns_new_instance_without_reloading(self, fixed_cosmology):
        updated = fixed_cosmology.update(H0=68.0)
        assert updated is not fixed_cosmology
        assert updated.H0 == 68.0
        assert updated.omega_cdm == fixed_cosmology.omega_cdm
        assert updated._emu is fixed_cosmology._emu

    # update() cannot switch emulator_set -- it's not an accepted keyword argument.
    def test_update_cannot_switch_emulator_set(self, fixed_cosmology):
        with pytest.raises(TypeError):
            fixed_cosmology.update(emulator_set="mnu:v1")

    # Cosmology survives a JAX pytree flatten/unflatten round trip, preserving all 15 leaves and both aux_data fields.
    def test_pytree_roundtrip(self, fixed_cosmology):
        leaves, treedef = jax.tree_util.tree_flatten(fixed_cosmology)
        assert len(leaves) == 15
        rt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rt.emulator_set == fixed_cosmology.emulator_set
        assert rt.extrapolate_z == fixed_cosmology.extrapolate_z
        assert rt.H0 == fixed_cosmology.H0 and rt.omega_cdm == fixed_cosmology.omega_cdm


class TestGradients:
    # jax.grad wrt H0 is finite and matches a finite-difference estimate, for every listed background/growth quantity.
    @pytest.mark.parametrize("method,kwargs", [
        ("hubble_parameter", {}),
        ("angular_diameter_distance", {}),
        ("sigma8", {}),
        ("growth_factor", {}),
        ("omega_m", {}),
    ])
    def test_grad_wrt_H0(self, fixed_cosmology, method, kwargs):
        z = jnp.array(0.5)

        def f(H0):
            return getattr(fixed_cosmology.update(H0=H0), method)(z, **kwargs)

        _rel_grad_check(f, 67.5)

    # jax.grad of pk wrt H0 is finite and matches a finite-difference estimate.
    def test_pk_grad_wrt_H0(self, fixed_cosmology):
        def f(H0):
            return fixed_cosmology.update(H0=H0).pk(jnp.array([0.1]), jnp.array(0.5), linear=True)

        _rel_grad_check(f, 67.5)

    # jax.grad of sigma8 wrt A_s is finite and matches a finite-difference estimate (A_s ~ 1e-9, so the
    # finite-difference step must be scaled relative to its magnitude, not a fixed absolute step).
    def test_sigma8_grad_wrt_A_s(self, fixed_cosmology):
        def f(A_s):
            return fixed_cosmology.update(A_s=A_s).sigma8(jnp.array(0.5))

        _rel_grad_check(f, 2.1e-9)

    # delta_c's gradient wrt H0 is exactly zero for the EdS prescription (a cosmology-independent constant)
    # but nonzero for NS97 (which depends on omega_m(z)) -- both match their finite-difference estimate.
    def test_delta_c_grad_wrt_H0_prescription_dependence(self, fixed_cosmology):
        def f_eds(H0):
            return fixed_cosmology.update(H0=H0).delta_c(jnp.array(0.5), prescription="EdS")

        def f_ns97(H0):
            return fixed_cosmology.update(H0=H0).delta_c(jnp.array(0.5), prescription="NS97")

        _rel_grad_check(f_eds, 67.5)
        _rel_grad_check(f_ns97, 67.5, rel_eps=1e-6)
        assert jax.grad(f_eds)(67.5) == 0.0
        assert jax.grad(f_ns97)(67.5) != 0.0


class TestOtherEmulatorSets:
    # hubble_parameter/pk/sigma8/derived_parameters() are all finite at default parameters, for every non-lcdm set.
    def test_smoke_finite_outputs(self, other_emulator_cosmology):
        cosmo = other_emulator_cosmology
        z = jnp.array(0.5)
        assert jnp.isfinite(cosmo.hubble_parameter(z))
        assert jnp.isfinite(cosmo.pk(jnp.array([0.1]), z, linear=True))
        assert jnp.isfinite(cosmo.sigma8(jnp.array(0.0)))
        for value in cosmo.derived_parameters().values():
            assert jnp.isfinite(jnp.asarray(value))

    # mnu:v1's extra free parameter (m_ncdm) actually changes sigma8 -- it isn't silently ignored.
    def test_mnu_sensitive_to_m_ncdm(self):
        cosmo = _construct_or_skip("mnu:v1")
        s8_default = cosmo.sigma8(jnp.array(0.0))
        s8_changed = cosmo.update(m_ncdm=0.3).sigma8(jnp.array(0.0))
        assert not jnp.isclose(s8_default, s8_changed)

    # neff:v1's extra free parameter (N_ur) actually changes sigma8.
    def test_neff_sensitive_to_N_ur(self):
        cosmo = _construct_or_skip("neff:v1")
        s8_default = cosmo.sigma8(jnp.array(0.0))
        s8_changed = cosmo.update(N_ur=1.0).sigma8(jnp.array(0.0))
        assert not jnp.isclose(s8_default, s8_changed)

    # wcdm:v1's extra free parameter (w0) actually changes H(z).
    def test_wcdm_sensitive_to_w0(self):
        cosmo = _construct_or_skip("wcdm:v1")
        hz_default = cosmo.hubble_parameter(jnp.array(1.0))
        hz_changed = cosmo.update(w0=-1.3).hubble_parameter(jnp.array(1.0))
        assert not jnp.isclose(hz_default, hz_changed)

    # ede:v1's extra free parameter (f_ede) actually changes theta_s -- probed there rather than low-z H(z),
    # since EDE only affects the expansion history near z_c (early times), not the late-time background.
    def test_ede_sensitive_to_f_ede(self):
        cosmo = _construct_or_skip("ede:v1")
        theta_default = cosmo.derived_parameters()["100*theta_s"]
        theta_changed = cosmo.update(f_ede=0.3).derived_parameters()["100*theta_s"]
        assert not jnp.isclose(theta_default, theta_changed)

    # mnu-3states:v1 has no code-visible distinguishing parameter from mnu:v1 -- confirms it's a genuinely
    # different trained emulator (not an accidental alias) by giving a different sigma8 at the same m_ncdm.
    def test_mnu_3states_differs_from_mnu(self):
        s8_mnu = _construct_or_skip("mnu:v1").sigma8(jnp.array(0.0))
        s8_mnu3 = _construct_or_skip("mnu-3states:v1").sigma8(jnp.array(0.0))
        assert not jnp.isclose(s8_mnu, s8_mnu3)
