"""
Concentrated stress-test benchmarks for hmfast.halos.profiles against CCL
(NFW, GNFW, HOD, CIB Shang12) and class_sz (B16, B12), plus an inlined-numpy
cross-check of the Maniyar+2021 CIB Mdot/SFR formulas. These are deliberately
NOT the full parameter sweep done in tests/test_profiles.ipynb -- 1-3 canonical
points per profile, not 10-60 random draws.

CCL sections require pyccl; the whole module is skipped if it isn't installed.
Those tests are marked `ccl` (see pyproject.toml) so `pytest -m "not ccl"`
skips them without needing pyccl at all. The B16/B12 sections additionally
require the (separate) `classy_sz` package and are marked `class_sz`; neither
package need be installed for the rest of this file to run.

Tolerances were derived empirically (not guessed): each canonical-point
comparison below was run once at the fixed cosmology, and rtol was set to a
round number with margin above the observed max % residual (noted per test).
Reused from tests/test_profiles.ipynb's own established conventions:
  - hmfast's `real()`/`fourier()` are ng_bar-normalized for HOD; CCL's raw HOD
    output must be divided by ng_bar to match.
  - CCL's GNFW `real()` doesn't enforce its own x_out truncation -- masked to
    NaN for r > x_out*r_delta on the CCL side here, as in the notebook.
  - hmfast's NFW real()/fourier() return density in units of the mean cb
    density (rho_crit_0 * Omega0_cb); multiply by that factor to recover the
    physical Msun/Mpc^3 convention used for comparison in this file.
"""

import jax.numpy as jnp
import numpy as np
import pytest

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

from hmfast.halos import HaloModel
from hmfast.halos.concentration import ConstantConcentration, D08Concentration
from hmfast.halos.massdef import MassDefinition
from hmfast.halos.profiles import (
    GNFWPressureProfile,
    M21CIBProfile,
    NFWMatterProfile,
    S12CIBProfile,
    Z07GalaxyHODProfile,
)
from hmfast.halos.profiles.profiles_2pt import _fourier_2pt

M_GRID = jnp.geomspace(1e10, 1e15, 40)
R_GRID = np.geomspace(0.01, 2.0, 20)
K_GRID = np.geomspace(0.01, 5.0, 20)


def to_ccl_massdef(mass_def):
    reference = "matter" if mass_def.reference == "mean" else mass_def.reference
    return pyccl.halos.MassDef(mass_def.delta, reference)


@pytest.fixture(scope="session")
def cosmo_ccl(fixed_cosmology):
    # A pyccl CosmologyCalculator fed hmfast's own P_lin(k, z), isolating profile-formula
    # differences from any P(k) differences between the two codes' Boltzmann solvers.
    h = fixed_cosmology.H0 / 100.0
    z_wide = np.linspace(0.0, 3.0, 50)
    a_wide = np.sort(1.0 / (1.0 + z_wide))
    k_ref_jnp, _ = fixed_cosmology._pk_grid()
    k_ref = np.asarray(k_ref_jnp)
    pk_lin = np.asarray(
        fixed_cosmology.pk(k_ref_jnp, jnp.asarray(1.0 / a_wide - 1.0), linear=True)
    ).T
    return pyccl.CosmologyCalculator(
        Omega_c=fixed_cosmology.omega_cdm / h**2,
        Omega_b=fixed_cosmology.omega_b / h**2,
        h=h,
        A_s=fixed_cosmology.A_s,
        n_s=fixed_cosmology.n_s,
        pk_linear={"a": a_wide, "k": k_ref, "delta_matter:delta_matter": pk_lin},
    )


class TestNFWMatterProfileCCL:
    # real()/fourier() match CCL's HaloProfileNFW at 200c (rtol 1%, observed max ~0.13%/0.11%).
    def test_real_fourier_match_ccl_at_canonical_point(
        self, fixed_cosmology, cosmo_ccl
    ):
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        conc_ccl = pyccl.halos.concentration.ConcentrationDuffy08(mass_def=md_ccl)
        nfw_ccl = pyccl.halos.profiles.nfw.HaloProfileNFW(
            mass_def=md_ccl,
            concentration=conc_ccl,
            truncated=True,
            fourier_analytic=True,
        )
        m_val, z_val, a_val = 1e14, 0.5, 1.0 / 1.5
        cparams = fixed_cosmology._cosmo_params()
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"]

        nfw = NFWMatterProfile()
        real_hm = (
            np.asarray(nfw.real(hm, jnp.asarray(R_GRID), m_val, z_val)) * rho_mean_0
        )
        real_ccl = np.asarray(nfw_ccl.real(cosmo_ccl, R_GRID, m_val, a_val))
        mask = real_ccl != 0
        assert np.allclose(real_hm[mask], real_ccl[mask], rtol=0.01)

        four_hm = (
            np.asarray(nfw.fourier(hm, jnp.asarray(K_GRID), m_val, z_val)) * rho_mean_0
        )
        four_ccl = np.asarray(nfw_ccl.fourier(cosmo_ccl, K_GRID, m_val, a_val))
        assert np.allclose(four_hm, four_ccl, rtol=0.01)

    # real() matches CCL across all 4 mass defs at the same canonical point (D08 skips 500c).
    @pytest.mark.parametrize(
        "delta,reference", [(200, "critical"), (200, "mean"), ("vir", "critical")]
    )
    def test_real_matches_ccl_across_mass_defs(
        self, fixed_cosmology, cosmo_ccl, delta, reference
    ):
        md = MassDefinition(delta, reference)
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        conc_ccl = pyccl.halos.concentration.ConcentrationDuffy08(mass_def=md_ccl)
        nfw_ccl = pyccl.halos.profiles.nfw.HaloProfileNFW(
            mass_def=md_ccl,
            concentration=conc_ccl,
            truncated=True,
            fourier_analytic=True,
        )
        m_val, z_val, a_val = 1e14, 0.5, 1.0 / 1.5
        cparams = fixed_cosmology._cosmo_params()
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"]

        nfw = NFWMatterProfile()
        real_hm = (
            np.asarray(nfw.real(hm, jnp.asarray(R_GRID), m_val, z_val)) * rho_mean_0
        )
        real_ccl = np.asarray(nfw_ccl.real(cosmo_ccl, R_GRID, m_val, a_val))
        mask = real_ccl != 0
        assert np.allclose(real_hm[mask], real_ccl[mask], rtol=0.01)


class TestGNFWPressureProfileCCL:
    # real() matches CCL's HaloProfilePressureGNFW at 500c (rtol 2%, observed max ~0.26%).
    # CCL's own real() doesn't enforce x_out -- masked to NaN for r > x_out*r500c, as in the notebook.
    def test_real_matches_ccl_at_canonical_point(self, fixed_cosmology, cosmo_ccl):
        md = MassDefinition(500, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=ConstantConcentration(c=5),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        B, x_out = 1.4, 4.0
        gnfw = GNFWPressureProfile(
            P0=8.130,
            c500=1.156,
            alpha=1.062,
            beta=5.4807,
            gamma=0.3292,
            B=B,
            alpha_P=0.12,
            P0_hexp=-1.0,
            x_out=x_out,
        )
        gnfw_ccl = pyccl.halos.profiles.pressure_gnfw.HaloProfilePressureGNFW(
            mass_def=md_ccl,
            mass_bias=1.0 / B,
            P0=8.130,
            c500=1.156,
            alpha=1.062,
            alpha_P=0.12,
            beta=5.4807,
            gamma=0.3292,
            P0_hexp=-1.0,
            qrange=(1e-5, 1e5),
            nq=256,
            x_out=x_out,
        )
        m_val, z_val, a_val = 1e14, 0.5, 1.0 / 1.5

        r500c = md_ccl.get_radius(cosmo_ccl, m_val / B, a_val) / a_val
        real_hm = np.asarray(gnfw.real(hm, jnp.asarray(R_GRID), m_val, z_val))
        real_ccl_raw = np.asarray(gnfw_ccl.real(cosmo_ccl, R_GRID, m_val, a_val))
        real_ccl = np.where(R_GRID <= x_out * r500c, real_ccl_raw, np.nan)
        mask = np.isfinite(real_ccl) & (real_ccl != 0)
        assert np.allclose(real_hm[mask], real_ccl[mask], rtol=0.02)

        # fourier(), checked at a k range pre-verified clear of GNFW's known rare zero-crossing
        # spikes (rtol 3%, observed max ~1.5%) -- not a statistically-bounded statement.
        four_hm = np.asarray(gnfw.fourier(hm, jnp.asarray(K_GRID), m_val, z_val))
        four_ccl = np.asarray(gnfw_ccl.fourier(cosmo_ccl, K_GRID, m_val, a_val))
        assert np.allclose(four_hm, four_ccl, rtol=0.03)

    # real() matches CCL after converting to a non-native (200c) mass def -- 1 extra point,
    # not a full 4-way sweep, since GNFW is deliberately mass-def sensitive (see test_profiles.py).
    def test_real_matches_ccl_at_non_native_mass_def(self, fixed_cosmology, cosmo_ccl):
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=ConstantConcentration(c=5),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        gnfw = GNFWPressureProfile()
        gnfw_ccl = pyccl.halos.profiles.pressure_gnfw.HaloProfilePressureGNFW(
            mass_def=md_ccl,
            mass_bias=1.0 / 1.4,
            P0=8.130,
            c500=1.156,
            alpha=1.062,
            alpha_P=0.12,
            beta=5.4807,
            gamma=0.3292,
            P0_hexp=-1.0,
            qrange=(1e-5, 1e5),
            nq=256,
            x_out=4.0,
        )
        m_val, z_val, a_val = 1e14, 0.5, 1.0 / 1.5
        r200c = md_ccl.get_radius(cosmo_ccl, m_val / 1.4, a_val) / a_val
        real_hm = np.asarray(gnfw.real(hm, jnp.asarray(R_GRID), m_val, z_val))
        real_ccl_raw = np.asarray(gnfw_ccl.real(cosmo_ccl, R_GRID, m_val, a_val))
        real_ccl = np.where(R_GRID <= 4.0 * r200c, real_ccl_raw, np.nan)
        mask = np.isfinite(real_ccl) & (real_ccl != 0)
        assert np.allclose(real_hm[mask], real_ccl[mask], rtol=0.02)


class TestHODProfileCCL:
    # real()/fourier() match CCL's HaloProfileHOD at 200c, dividing CCL's raw output by
    # ng_bar per the notebook's established convention (rtol 1%, observed max ~0.12%/0.02%).
    def test_real_fourier_match_ccl_at_canonical_point(
        self, fixed_cosmology, cosmo_ccl
    ):
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        conc_ccl = pyccl.halos.concentration.ConcentrationDuffy08(mass_def=md_ccl)
        hod = Z07GalaxyHODProfile(
            sigma_log10M=0.68, alpha_s=1.30, M1_prime=10**12.87, M_min=10**11.97, M0=0.0
        )
        hod_ccl = pyccl.halos.profiles.hod.HaloProfileHOD(
            mass_def=md_ccl,
            concentration=conc_ccl,
            log10Mmin_0=11.97,
            siglnM_0=0.68 * np.log(10),
            log10M1_0=12.87,
            alpha_0=1.30,
            log10M0_0=0.0,
        )
        m_val, z_val, a_val = 5e12, 0.5, 1.0 / 1.5
        ngbar = float(hod.ng_bar(hm, z_val))

        real_hm = np.asarray(hod.real(hm, jnp.asarray(R_GRID), m_val, z_val))
        real_ccl = np.asarray(hod_ccl.real(cosmo_ccl, R_GRID, m_val, a_val)) / ngbar
        mask = real_ccl != 0
        assert np.allclose(real_hm[mask], real_ccl[mask], rtol=0.01)

        four_hm = np.asarray(hod.fourier(hm, jnp.asarray(K_GRID), m_val, z_val))
        four_ccl = np.asarray(hod_ccl.fourier(cosmo_ccl, K_GRID, m_val, a_val)) / ngbar
        assert np.allclose(four_hm, four_ccl, rtol=0.01)

    # The 1-halo Fourier 2-point term matches CCL's Profile2ptHOD at the canonical point
    # (rtol 3%, observed max ~1.5%), a single snapshot per the notebook's "bonus" check.
    def test_fourier_2pt_matches_ccl_profile2pthod(self, fixed_cosmology, cosmo_ccl):
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        conc_ccl = pyccl.halos.concentration.ConcentrationDuffy08(mass_def=md_ccl)
        hod = Z07GalaxyHODProfile(
            sigma_log10M=0.68, alpha_s=1.30, M1_prime=10**12.87, M_min=10**11.97, M0=0.0
        )
        hod_ccl = pyccl.halos.profiles.hod.HaloProfileHOD(
            mass_def=md_ccl,
            concentration=conc_ccl,
            log10Mmin_0=11.97,
            siglnM_0=0.68 * np.log(10),
            log10M1_0=12.87,
            alpha_0=1.30,
            log10M0_0=0.0,
        )
        m_val, z_val, a_val = 5e12, 0.5, 1.0 / 1.5
        ngbar = float(hod.ng_bar(hm, z_val))

        prof2pt = pyccl.halos.profiles_2pt.Profile2ptHOD()
        two_pt_ccl = prof2pt.fourier_2pt(
            cosmo_ccl, K_GRID, m_val, a_val, prof=hod_ccl, prof2=hod_ccl
        )
        two_pt_hm = np.asarray(
            _fourier_2pt(hm, hod, hod, jnp.asarray(K_GRID), m_val, z_val)
        ) * (ngbar**2)
        assert np.allclose(two_pt_hm, two_pt_ccl, rtol=0.03)


class TestS12CIBProfileCCL:
    # real()/fourier() match CCL's HaloProfileCIBShang12 at 200c (rtol 1%, observed max ~0.19%/0.06%).
    def test_real_fourier_match_ccl_at_canonical_point(
        self, fixed_cosmology, cosmo_ccl
    ):
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        md_ccl = to_ccl_massdef(md)
        conc_ccl = pyccl.halos.concentration.ConcentrationDuffy08(mass_def=md_ccl)
        cib = S12CIBProfile(
            nu=100,
            L0=6.4e-8,
            alpha=0.36,
            beta=1.75,
            gamma=1.7,
            T0=24.4,
            M_eff=10**12.6,
            sigma2_LM=0.707**2,
            delta=3.6,
            z_p=1e100,
            M_min=10**11.5,
        )
        cib_ccl = pyccl.halos.profiles.cib_shang12.HaloProfileCIBShang12(
            nu_GHz=100,
            mass_def=md_ccl,
            concentration=conc_ccl,
            alpha=0.36,
            T0=24.4,
            beta=1.75,
            gamma=1.7,
            s_z=3.6,
            log10Meff=12.6,
            siglog10M=0.707,
            Mmin=10**11.5,
            L0=6.4e-8,
        )
        m_val, z_val, a_val = 5e13, 0.5, 1.0 / 1.5

        real_hm = np.asarray(cib.real(hm, jnp.asarray(R_GRID), m_val, z_val))
        real_ccl = np.asarray(cib_ccl.real(cosmo_ccl, R_GRID, m_val, a_val))
        mask = real_ccl != 0
        assert np.allclose(real_hm[mask], real_ccl[mask], rtol=0.01)

        four_hm = np.asarray(cib.fourier(hm, jnp.asarray(K_GRID), m_val, z_val))
        four_ccl = np.asarray(cib_ccl.fourier(cosmo_ccl, K_GRID, m_val, a_val))
        assert np.allclose(four_hm, four_ccl, rtol=0.01)


class TestB16DensityProfileClassSZ:
    # real() matches class_sz's get_gas_profile_at_x_M_z_b16_200c at the notebook's canonical
    # point, applying the confirmed 1/h^2 unit correction (rtol 1%, observed max ~0.26%).
    @pytest.mark.class_sz
    def test_real_matches_classy_sz_at_canonical_point(self, fixed_cosmology):
        from hmfast.halos.profiles import B16DensityProfile

        classy_sz = pytest.importorskip("classy_sz")
        ClassSZ = classy_sz.Class
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        b16 = B16DensityProfile(
            A_rho0=4000.0,
            A_alpha=0.88,
            A_beta=3.83,
            alpha_m_rho0=0.29,
            alpha_m_alpha=-0.03,
            alpha_m_beta=0.04,
            alpha_z_rho0=-0.66,
            alpha_z_alpha=0.19,
            alpha_z_beta=-0.025,
        )
        m_val, z_val = 1e14, 0.3
        h = fixed_cosmology.H0 / 100.0

        cosmo_csz = ClassSZ()
        cosmo_csz.set({"output": "gas_pressure_profile_2h,gas_density_profile_2h"})
        cosmo_csz.compute_class_szfast()
        csz_kwargs = dict(
            A_rho0=4000.0,
            A_alpha=0.88,
            A_beta=3.83,
            alpha_m_rho0=0.29,
            alpha_m_alpha=-0.03,
            alpha_m_beta=0.04,
            alpha_z_rho0=-0.66,
            alpha_z_alpha=0.19,
            alpha_z_beta=-0.025,
            alphap_m_rho0=0.29,
            alphap_m_alpha=-0.03,
            alphap_m_beta=0.04,
        )
        real_csz = (
            np.array(
                [
                    cosmo_csz.get_gas_profile_at_x_M_z_b16_200c(
                        x=float(r), M=m_val, z=z_val, **csz_kwargs
                    )
                    for r in R_GRID
                ]
            )
            * h**2
        )
        real_hm = np.asarray(b16.real(hm, jnp.asarray(R_GRID), m_val, z_val))
        mask = real_csz != 0
        assert np.allclose(real_hm[mask], real_csz[mask], rtol=0.01)


class TestB12PressureProfileClassSZ:
    # real() matches class_sz's B12 shape function (dimensionalized via hmfast's own P_200c
    # formula, per the notebook's b12_dimensionalize) at the canonical point (rtol 2%, observed max ~0.9%).
    @pytest.mark.class_sz
    def test_real_matches_classy_sz_at_canonical_point(self, fixed_cosmology):
        from hmfast.halos.profiles import B12PressureProfile

        classy_sz = pytest.importorskip("classy_sz")
        ClassSZ = classy_sz.Class
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        b12 = B12PressureProfile(
            A_P0=18.1,
            A_xc=0.497,
            A_beta=4.35,
            alpha_m_P0=0.154,
            alpha_m_xc=-0.00865,
            alpha_m_beta=0.0393,
            alpha_z_P0=-0.758,
            alpha_z_xc=0.731,
            alpha_z_beta=0.415,
        )
        m_val, z_val = 1e14, 0.3
        h = fixed_cosmology.H0 / 100.0
        cparams = fixed_cosmology._cosmo_params()
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        r200c = float(md.r_delta(fixed_cosmology, m_val, z_val))
        H = float(fixed_cosmology.hubble_parameter(z_val))
        p_200c = (m_val / (r200c * h)) * f_b * 2.61051e-18 * H**2

        cosmo_csz = ClassSZ()
        cosmo_csz.set({"output": "gas_pressure_profile_2h"})
        cosmo_csz.compute_class_szfast()
        csz_kwargs = dict(
            A_P0=18.1,
            A_xc=0.497,
            A_beta=4.35,
            alpha_m_P0=0.154,
            alpha_m_xc=-0.00865,
            alpha_m_beta=0.0393,
            alpha_z_P0=-0.758,
            alpha_z_xc=0.731,
            alpha_z_beta=0.415,
            alphap_m_P0=0.154,
            alphap_m_xc=-0.00865,
            alphap_m_beta=0.0393,
        )
        shape_csz = np.array(
            [
                cosmo_csz.get_pressure_P_over_P_delta_at_x_M_z_b12_200c(
                    x=float(r), M=m_val, z=z_val, **csz_kwargs
                )
                for r in R_GRID
            ]
        )
        real_csz = shape_csz * p_200c
        real_hm = np.asarray(b12.real(hm, jnp.asarray(R_GRID), m_val, z_val))
        mask = real_csz != 0
        assert np.allclose(real_hm[mask], real_csz[mask], rtol=0.02)


def _ref_mdot(m, z, Om0, Ode0):
    """Halo mass accretion rate, transcribed from class_sz's maniyar_cib_Mdot (source/class_sz.c),
    cross-checked against abhimaniyar/halomodel_cib_tsz_cibxtsz v2's independent rewrite.
    """
    m, z = np.atleast_1d(m), np.atleast_1d(z)
    E_z = np.sqrt(Om0 * (1.0 + z) ** 3 + Ode0)
    return 46.1 * (1.0 + 1.11 * z[None, :]) * E_z[None, :] * (m[:, None] / 1e12) ** 1.1


def _ref_sfr(m, z, Om0, Ode0, f_b, eta_max, M_eff, sigma2_LM, tau, z_c):
    """SFR, transcribed from class_sz's maniyar_cib_sfr, same cross-check as _ref_mdot."""
    m, z = np.atleast_1d(m), np.atleast_1d(z)
    mdot = _ref_mdot(m, z, Om0, Ode0)
    sigma2_lnM = np.where(
        m[:, None] < M_eff,
        sigma2_LM,
        (np.sqrt(sigma2_LM) - tau * np.maximum(0.0, z_c - z[None, :])) ** 2,
    )
    log_m, log_meff = np.log(m)[:, None], np.log(M_eff)
    sfr_c = eta_max * np.exp(-((log_m - log_meff) ** 2) / (2.0 * sigma2_lnM))
    return 1e10 * mdot * f_b * sfr_c


class TestManiyarMdotSFR:
    # No package dependency needed -- _ref_mdot/_ref_sfr above are inlined numpy transcriptions
    # of class_sz's C source. Expected residual source: the reference uses the analytic flat-LCDM
    # E(z)=sqrt(Om0(1+z)^3+Ode0), hmfast uses the actual emulator H(z)/H0 (rtol 1%, observed max ~0.04%).
    def test_mdot_and_sfr_match_reference_at_canonical_grid(self, fixed_cosmology):
        md = MassDefinition(200, "critical")
        hm = HaloModel(
            cosmology=fixed_cosmology,
            mass_def=md,
            concentration=D08Concentration(),
            m_grid=M_GRID,
        )
        m21 = M21CIBProfile(nu=100)
        m_grid_np = np.geomspace(1e11, 1e15, 6)
        z_grid_np = np.array([0.0, 0.5, 1.0, 2.0])
        cparams = fixed_cosmology._cosmo_params()
        Om0 = float(cparams["Omega0_m"])
        Ode0 = 1.0 - Om0
        f_b = float(cparams["Omega_b"] / cparams["Omega0_m"])

        mdot_hm = np.asarray(
            m21._m_dot(hm, jnp.asarray(m_grid_np), jnp.asarray(z_grid_np))
        )
        mdot_ref = _ref_mdot(m_grid_np, z_grid_np, Om0, Ode0)
        assert np.allclose(mdot_hm, mdot_ref, rtol=0.01)

        sfr_hm = np.asarray(
            m21._sfr(hm, jnp.asarray(m_grid_np), jnp.asarray(z_grid_np))
        )
        sfr_ref = _ref_sfr(
            m_grid_np,
            z_grid_np,
            Om0,
            Ode0,
            f_b,
            m21.eta_max,
            m21.M_eff,
            m21.sigma2_LM,
            m21.tau,
            m21.z_c,
        )
        mask = sfr_ref != 0
        assert np.allclose(sfr_hm[mask], sfr_ref[mask], rtol=0.01)
