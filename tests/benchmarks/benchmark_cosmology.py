"""
CCL benchmark comparisons for hmfast.cosmology.Cosmology (background, growth,
matter power spectrum, RMS fluctuations, collapse threshold), used as an
external ground truth. Ported from the validated comparisons in
tests/test_cosmology.ipynb.

Requires pyccl; the whole module is skipped if it isn't installed. All tests
are marked `ccl` (see pyproject.toml) so `pytest -m "not ccl"` skips this file
without needing pyccl at all. The nonlinear-P(k) test additionally requires
`camb` (marked `camb`) -- CCL's native halofit disagrees with the HMcode-trained
PKNL emulator by orders of magnitude, so CAMB+HMcode (reached via CCL's
`boltzmann_camb`/`camb` backend) is the only fair ground truth for it; every
other test in this file runs fine without camb installed.

Tolerances were derived empirically (not guessed): for each comparison, we ran
this exact check at the fixed cosmology below and recorded the observed max %
residual, then set rtol to a round number with margin above it (see each
test's comment for the number).

"""
import jax.numpy as jnp
import numpy as np
import pytest

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl


def z_to_a(z):
    return 1.0 / (1.0 + np.asarray(z))


Z_LIST = [0.0, 0.5, 1.0, 2.0]
# growth_rate only: excludes the jnp.gradient table's edges (z=0, z_max_pk=5) -- see module docstring.
Z_LIST_GROWTH_RATE = [0.1, 0.5, 1.0, 2.0]
M_GRID = np.geomspace(1e10, 1e15, 20)
R_GRID = np.geomspace(1e-1, 1e2, 20)


@pytest.fixture(scope="session")
def cosmo_hmfast_ext(fixed_cosmology):
    return fixed_cosmology.update(extrapolate_z=True)


@pytest.fixture(scope="session")
def cosmo_ccl_bg(fixed_cosmology):
    # Independent pyccl.Cosmology (CLASS background), matched in neutrino mass; deliberately not passing w0 since lcdm:v1 always assumes w=-1.
    h = fixed_cosmology.H0 / 100.0
    return pyccl.Cosmology(
        Omega_c=fixed_cosmology.omega_cdm / h**2,
        Omega_b=fixed_cosmology.omega_b / h**2,
        h=h, A_s=fixed_cosmology.A_s, n_s=fixed_cosmology.n_s,
        m_nu=[0.0, 0.0, float(fixed_cosmology.m_ncdm)], mass_split="list",
        transfer_function="boltzmann_class",
    )


@pytest.fixture(scope="session")
def cosmo_ccl_pkl(fixed_cosmology):
    # Same as cosmo_ccl_bg but forcing linear-only P(k), for the P_lin(k,z) check.
    h = fixed_cosmology.H0 / 100.0
    return pyccl.Cosmology(
        Omega_c=fixed_cosmology.omega_cdm / h**2,
        Omega_b=fixed_cosmology.omega_b / h**2,
        h=h, A_s=fixed_cosmology.A_s, n_s=fixed_cosmology.n_s,
        m_nu=[0.0, 0.0, float(fixed_cosmology.m_ncdm)], mass_split="list",
        transfer_function="boltzmann_class", matter_power_spectrum="linear",
    )


@pytest.fixture(scope="session")
def cosmo_ccl_pknl(fixed_cosmology):
    # CAMB+HMcode nonlinear P(k) -- the only fair ground truth for the HMcode-trained PKNL emulator; requires the separate `camb` package.
    pytest.importorskip("camb")
    h = fixed_cosmology.H0 / 100.0
    return pyccl.Cosmology(
        Omega_c=fixed_cosmology.omega_cdm / h**2,
        Omega_b=fixed_cosmology.omega_b / h**2,
        h=h, A_s=fixed_cosmology.A_s, n_s=fixed_cosmology.n_s,
        m_nu=[0.0, 0.0, float(fixed_cosmology.m_ncdm)], mass_split="list",
        transfer_function="boltzmann_camb", matter_power_spectrum="camb",
    )


@pytest.fixture(scope="session")
def cosmo_ccl(fixed_cosmology):
    # pyccl CosmologyCalculator fed hmfast's own P_lin(k, z), isolating the sigma(M)/sigma(R) formula from P(k) differences.
    h = fixed_cosmology.H0 / 100.0
    Omega_c = fixed_cosmology.omega_cdm / h**2
    Omega_b = fixed_cosmology.omega_b / h**2

    z_wide = np.linspace(0.0, 3.0, 50)
    a_wide = np.sort(1.0 / (1.0 + z_wide))
    k_ref_jnp, _ = fixed_cosmology._pk_grid()
    k_ref = np.asarray(k_ref_jnp)
    pk_lin = np.asarray(
        fixed_cosmology.pk(k_ref_jnp, jnp.asarray(1.0 / a_wide - 1.0), linear=True)
    ).T  # (Nz, Nk), CCL's expected ordering

    return pyccl.CosmologyCalculator(
        Omega_c=Omega_c, Omega_b=Omega_b, h=h,
        A_s=fixed_cosmology.A_s, n_s=fixed_cosmology.n_s,
        pk_linear={"a": a_wide, "k": k_ref, "delta_matter:delta_matter": pk_lin},
    )


class TestGrowthAndDistancesCCL:
    # H(z), D_A(z), sigma8(z) match CCL in-grid (rtol 0.05%/0.05%/0.1%, observed max ~0.011%/0.0018%/0.013%); D_A(z=0)=0 skipped to avoid 0/0.
    def test_hz_da_sigma8_in_grid(self, fixed_cosmology, cosmo_ccl_bg):
        h = fixed_cosmology.H0 / 100.0
        z_arr = jnp.asarray(Z_LIST)
        a_arr = z_to_a(Z_LIST)

        Hz_hmf = np.asarray(fixed_cosmology.hubble_parameter(z_arr))
        Hz_ccl = 100.0 * h * np.asarray(pyccl.background.h_over_h0(cosmo_ccl_bg, a_arr))
        assert np.allclose(Hz_hmf, Hz_ccl, rtol=0.0005)

        DA_hmf = np.asarray(fixed_cosmology.angular_diameter_distance(z_arr))[1:]
        DA_ccl = np.asarray(pyccl.background.angular_diameter_distance(cosmo_ccl_bg, a_arr))[1:]
        assert np.allclose(DA_hmf, DA_ccl, rtol=0.0005)

        R8 = 8.0 / h
        s8_hmf = np.asarray(fixed_cosmology.sigma8(z_arr))
        s8_ccl = np.array([pyccl.power.sigmaR(cosmo_ccl_bg, R8, float(ai)) for ai in a_arr])
        assert np.allclose(s8_hmf, s8_ccl, rtol=0.001)

    # H(z), D_A(z) with extrapolate_z=True still match CCL a few z past z_max_bg=20 (rtol 0.2%, observed max ~0.069%/0.0032%).
    def test_hz_da_beyond_bg_grid_with_extrapolation(self, cosmo_hmfast_ext, cosmo_ccl_bg):
        h = cosmo_hmfast_ext.H0 / 100.0
        z_max_bg = float(cosmo_hmfast_ext._z_grid_bg()[-1])
        z_arr = jnp.array([z_max_bg + 5.0, z_max_bg + 50.0, 100.0])
        a_arr = z_to_a(np.asarray(z_arr))

        Hz_hmf = np.asarray(cosmo_hmfast_ext.hubble_parameter(z_arr))
        Hz_ccl = 100.0 * h * np.asarray(pyccl.background.h_over_h0(cosmo_ccl_bg, a_arr))
        assert np.allclose(Hz_hmf, Hz_ccl, rtol=0.002)

        DA_hmf = np.asarray(cosmo_hmfast_ext.angular_diameter_distance(z_arr))
        DA_ccl = np.asarray(pyccl.background.angular_diameter_distance(cosmo_ccl_bg, a_arr))
        assert np.allclose(DA_hmf, DA_ccl, rtol=0.002)

    # sigma8 has no extrapolation branch at all: NaN beyond z_max_bg regardless of extrapolate_z (unlike H(z)/D_A(z)).
    def test_sigma8_always_nan_beyond_grid(self, fixed_cosmology, cosmo_hmfast_ext):
        z_beyond = jnp.array(float(fixed_cosmology._z_grid_bg()[-1]) + 5.0)
        assert jnp.isnan(fixed_cosmology.sigma8(z_beyond))
        assert jnp.isnan(cosmo_hmfast_ext.sigma8(z_beyond))


class TestMatterPowerSpectrumCCL:
    # P_lin(k, z=1) matches CCL's independent CLASS linear P(k) (rtol 0.15%, observed max ~0.042%).
    def test_pk_linear_matches_ccl(self, fixed_cosmology, cosmo_ccl_pkl):
        k_test = jnp.geomspace(1e-3, 1.0, 20)
        pkl_hmf = np.asarray(fixed_cosmology.pk(k_test, jnp.array([1.0]), linear=True)).flatten()
        pkl_ccl = pyccl.linear_matter_power(cosmo_ccl_pkl, np.asarray(k_test), z_to_a(1.0))
        assert np.allclose(pkl_hmf, pkl_ccl, rtol=0.0015)

    # P_nl(k, z=1) matches CCL+CAMB's HMcode nonlinear P(k) (rtol 0.5%, observed max ~0.265%). Requires camb.
    @pytest.mark.camb
    def test_pk_nonlinear_matches_ccl(self, fixed_cosmology, cosmo_ccl_pknl):
        k_test = jnp.geomspace(1e-3, 1.0, 20)
        pknl_hmf = np.asarray(fixed_cosmology.pk(k_test, jnp.array([1.0]), linear=False)).flatten()
        pknl_ccl = pyccl.nonlin_matter_power(cosmo_ccl_pknl, np.asarray(k_test), z_to_a(1.0))
        assert np.allclose(pknl_hmf, pknl_ccl, rtol=0.005)

    # Without extrapolate_k, pk is NaN beyond the trained k-grid; with it, finite and close to CCL a modest factor beyond it (rtol 1%, observed max ~0.79%).
    def test_pk_k_extrapolation_matches_ccl(self, fixed_cosmology, cosmo_ccl_pkl):
        k_grid_native, _ = fixed_cosmology._pk_grid()
        k_min, k_max = float(k_grid_native.min()), float(k_grid_native.max())
        k_beyond = jnp.array([k_min * 0.5, k_max * 2.0])

        pk_noext = fixed_cosmology.pk(k_beyond, jnp.array([1.0]), linear=True, extrapolate_k=False)
        assert jnp.all(jnp.isnan(pk_noext))

        pk_ext = np.asarray(
            fixed_cosmology.pk(k_beyond, jnp.array([1.0]), linear=True, extrapolate_k=True)
        ).flatten()
        pk_ccl = pyccl.linear_matter_power(cosmo_ccl_pkl, np.asarray(k_beyond), z_to_a(1.0))
        assert np.allclose(pk_ext, pk_ccl, rtol=0.01)

    # Without extrapolate_z, pk is NaN beyond z_max_pk; with it, the growth-ratio-rescaled value matches CCL's growth-factor ratio (rtol 0.5%, observed max ~0.23%).
    def test_pk_z_extrapolation_growth_ratio_matches_ccl(self, fixed_cosmology, cosmo_hmfast_ext, cosmo_ccl_bg):
        z_max_pk = float(fixed_cosmology._z_grid_pk()[-1])
        k0 = jnp.array([0.05])
        z_beyond = jnp.array([z_max_pk + 2.0, z_max_pk + 10.0])

        pk_noext = fixed_cosmology.pk(k0, z_beyond, linear=True)
        assert jnp.all(jnp.isnan(pk_noext))

        pk_ext = np.asarray(cosmo_hmfast_ext.pk(k0, z_beyond, linear=True)).flatten()
        pk_zmax = float(np.asarray(fixed_cosmology.pk(k0, jnp.array([z_max_pk]), linear=True)).flatten()[0])
        growth_ratio_hmf = pk_ext / pk_zmax

        a_beyond = z_to_a(np.asarray(z_beyond))
        D_ccl_beyond = np.asarray(pyccl.background.growth_factor(cosmo_ccl_bg, a_beyond))
        D_ccl_zmax = float(pyccl.background.growth_factor(cosmo_ccl_bg, z_to_a(z_max_pk)))
        growth_ratio_ccl = (D_ccl_beyond / D_ccl_zmax) ** 2

        assert np.allclose(growth_ratio_hmf, growth_ratio_ccl, rtol=0.005)


class TestDensitiesAndMatterFractionCCL:
    # critical_density(z) matches CCL's rho_x(..., 'critical') in-grid and a bit beyond z_max_bg=20 (rtol 0.2%, observed max ~0.091% at z=70).
    def test_critical_density_matches_ccl(self, cosmo_hmfast_ext, cosmo_ccl_bg):
        z_max_bg = float(cosmo_hmfast_ext._z_grid_bg()[-1])
        z_arr = jnp.asarray(Z_LIST + [z_max_bg + 50.0])
        a_arr = z_to_a(np.asarray(z_arr))

        rho_crit_hmf = np.asarray(cosmo_hmfast_ext.critical_density(z_arr))
        rho_crit_ccl = pyccl.background.rho_x(cosmo_ccl_bg, a_arr, "critical", is_comoving=False)
        assert np.allclose(rho_crit_hmf, rho_crit_ccl, rtol=0.002)

    # omega_m(z) matches CCL's neutrino-corrected matter fraction in-grid and a bit beyond z_max_bg (rtol 0.3%, observed max ~0.184% at z=70).
    def test_omega_m_matches_ccl(self, cosmo_hmfast_ext, cosmo_ccl_bg):
        z_max_bg = float(cosmo_hmfast_ext._z_grid_bg()[-1])
        z_arr = jnp.asarray(Z_LIST + [z_max_bg + 50.0])
        a_arr = z_to_a(np.asarray(z_arr))

        om_hmf = np.asarray(cosmo_hmfast_ext.omega_m(z_arr))
        om_ccl = pyccl.background.omega_x(cosmo_ccl_bg, a_arr, "matter")
        assert np.allclose(om_hmf, om_ccl, rtol=0.003)

    # comoving_volume_element(z), built from the same D_A(z)/H(z) CCL calls, matches in-grid and beyond z_max_bg (rtol 0.1%, max ~0.052%); z=0 skipped (0/0).
    def test_comoving_volume_element_matches_ccl(self, cosmo_hmfast_ext, cosmo_ccl_bg):
        h = cosmo_hmfast_ext.H0 / 100.0
        z_max_bg = float(cosmo_hmfast_ext._z_grid_bg()[-1])
        z_arr = jnp.asarray(Z_LIST[1:] + [z_max_bg + 50.0])
        a_arr = z_to_a(np.asarray(z_arr))

        c_km_s = 299792.458
        dV_hmf = np.asarray(cosmo_hmfast_ext.comoving_volume_element(z_arr))
        DA_ccl = np.asarray(pyccl.background.angular_diameter_distance(cosmo_ccl_bg, a_arr))
        Hz_ccl = 100.0 * h * np.asarray(pyccl.background.h_over_h0(cosmo_ccl_bg, a_arr))
        dV_ccl = (1.0 + np.asarray(z_arr)) ** 2 * DA_ccl**2 * c_km_s / Hz_ccl

        assert np.allclose(dV_hmf, dV_ccl, rtol=0.001)


class TestSigmaMRCCL:
    # sigma(M), sigma(R) match CCL's sigmaM/sigmaR at z=0,1 (rtol 0.05%, observed max ~0.0051%).
    @pytest.mark.parametrize("z", [0.0, 1.0])
    def test_sigma_m_matches_ccl(self, cosmo_ccl, fixed_cosmology, z):
        sm_hmf = np.asarray(fixed_cosmology.sigma_m(jnp.asarray(M_GRID), z))
        sm_ccl = pyccl.power.sigmaM(cosmo_ccl, M_GRID, z_to_a(z))
        assert np.allclose(sm_hmf, sm_ccl, rtol=0.0005)

    @pytest.mark.parametrize("z", [0.0, 1.0])
    def test_sigma_r_matches_ccl(self, cosmo_ccl, fixed_cosmology, z):
        sr_hmf = np.asarray(fixed_cosmology.sigma_r(jnp.asarray(R_GRID), z))
        sr_ccl = pyccl.power.sigmaR(cosmo_ccl, R_GRID, z_to_a(z))
        assert np.allclose(sr_hmf, sr_ccl, rtol=0.0005)


class TestGrowthCCL:
    # growth_factor(z) matches CCL's growth_factor in-grid (rtol 0.15%, observed max ~0.068%).
    def test_growth_factor_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg):
        z_arr = jnp.asarray(Z_LIST)
        a_arr = z_to_a(Z_LIST)
        D_hmf = np.asarray(fixed_cosmology.growth_factor(z_arr))
        D_ccl = np.asarray(pyccl.background.growth_factor(cosmo_ccl_bg, a_arr))
        assert np.allclose(D_hmf, D_ccl, rtol=0.0015)

    # growth_rate(z) matches CCL's growth_rate in-grid, away from the jnp.gradient table's edges (rtol 0.2%, observed max ~0.113%).
    def test_growth_rate_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg):
        z_arr = jnp.asarray(Z_LIST_GROWTH_RATE)
        a_arr = z_to_a(Z_LIST_GROWTH_RATE)
        f_hmf = np.asarray(fixed_cosmology.growth_rate(z_arr))
        f_ccl = np.asarray(pyccl.background.growth_rate(cosmo_ccl_bg, a_arr))
        assert np.allclose(f_hmf, f_ccl, rtol=0.002)

    # growth_factor with extrapolate_z=True is finite and close to CCL past z_max_pk=5 (rtol 0.5%, max ~0.111%); NaN without the flag.
    def test_growth_factor_extrapolates_near_grid_boundary(self, fixed_cosmology, cosmo_hmfast_ext, cosmo_ccl_bg):
        z_max_pk = float(fixed_cosmology._z_grid_pk()[-1])
        z_beyond = jnp.array([z_max_pk + 2.0, z_max_pk + 10.0])

        assert jnp.all(jnp.isnan(fixed_cosmology.growth_factor(z_beyond)))

        D_hmf = np.asarray(cosmo_hmfast_ext.growth_factor(z_beyond))
        D_ccl = np.asarray(pyccl.background.growth_factor(cosmo_ccl_bg, z_to_a(np.asarray(z_beyond))))
        assert np.allclose(D_hmf, D_ccl, rtol=0.005)

    # growth_rate has no extrapolation branch at all: NaN beyond z_max_pk regardless of extrapolate_z.
    def test_growth_rate_nan_beyond_grid_unless_extrapolating(self, fixed_cosmology, cosmo_hmfast_ext):
        z_max_pk = float(fixed_cosmology._z_grid_pk()[-1])
        z_beyond = jnp.array(z_max_pk + 2.0)
        assert jnp.isnan(fixed_cosmology.growth_rate(z_beyond))
        assert jnp.isfinite(cosmo_hmfast_ext.growth_rate(z_beyond))


class TestDeltaCCL:
    # delta_c matches CCL's get_delta_c for all 3 prescriptions (hmfast "NS97" <-> CCL "NakamuraSuto97"; rtol 0.1%, observed max ~0.0025%).
    @pytest.mark.parametrize("hmfast_kind,ccl_kind", [
        ("EdS", "EdS"),
        ("EdS_approx", "EdS_approx"),
        ("NS97", "NakamuraSuto97"),
    ])
    def test_delta_c_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg, hmfast_kind, ccl_kind):
        z_arr = jnp.asarray(Z_LIST)
        a_arr = z_to_a(Z_LIST)
        dc_hmf = np.broadcast_to(
            np.asarray(fixed_cosmology.delta_c(z_arr, prescription=hmfast_kind)), z_arr.shape
        ).astype(float)
        dc_ccl = np.broadcast_to(
            np.asarray(pyccl.halos.get_delta_c(cosmo_ccl_bg, a_arr, kind=ccl_kind)), z_arr.shape
        ).astype(float)
        assert np.allclose(dc_hmf, dc_ccl, rtol=0.001)
