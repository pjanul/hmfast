"""
CCL benchmarks for the 3D halo-model statistics in hmfast.stats: Pk, Bk, Tk and corr_3d.

Every comparison runs hmfast with ncdm_mode="m" against a CCL CosmologyCalculator with the same neutrino masses and fed
hmfast's own P_lin, on the same profiles, mass function, bias and mass range, so residuals measure how each statistic is
assembled from the halo-model ingredients (validated in benchmark_halo_model.py and benchmark_profiles.py).

CCL has no halo-model bispectrum, and its native trispectrum 2- and 3-halo terms assign legs inconsistently when the
four profiles differ, so those references are assembled from CCL's own mass integrals (I_1_1, I_1_2, I_1_3) and P_lin
with the term definitions of Bk and Tk. The HOD second moment follows the hmfast convention, N_c^2 N_s^2 for satellite
pairs, where CCL's Profile2ptHOD uses N_c N_s^2; the reference uses a CCL Profile2pt with the hmfast convention.

Requires pyccl; marked `ccl` and skipped if it is missing.

Thresholds are ~2x the error measured when written (noted per test), so a ~2x degradation fails; lower them when
accuracy improves. Errors are maximum relative errors.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.cosmology import Cosmology
from hmfast.halos.massdef import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile, S12CIBProfile, Z07GalaxyHODProfile
from hmfast.stats import Bk, Pk, Tk, corr_3d

from .._shared import ccl_cosmology, ccl_profile2pt_hod, halo_model, rel_err, shared

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

M_MIN, M_MAX = 1e10, 1e16
K_PK = np.geomspace(1e-3, 30.0, 60)
Z_PK = [0.0, 0.5, 1.0, 2.0]
Z_X = 0.5
PK = Pk(k_damp=0.0)
BK = Bk(k_damp=0.0)
TK = Tk()

NFW = NFWMatterProfile()
GNFW = GNFWPressureProfile(x_range=(1e-5, 1e5), n_x=128, P0=8.13, c500=1.156, alpha=1.062, beta=5.4807, gamma=0.3292,
                           B=1.4, alpha_P=0.12, P0_hexp=-1, x_out=4)
HOD = Z07GalaxyHODProfile(sigma_log10M=0.68, alpha_s=1.3, M1_prime=10**12.7, M_min=10**11.8, M0=0.0)
CIB = S12CIBProfile(nu=545.0, L0=6.4e-8, alpha=0.36, beta=1.75, gamma=1.7, T0=24.4, M_eff=10**12.6, sigma2_LM=0.5,
                    delta=3.6, z_p=1e100, M_min=10**11.5)


def make_setup(cosmo):
    """hmfast halo model and the matching CCL calculator, HMCalculator and profiles at 200c with D08 concentration."""
    hm = halo_model(cosmology=cosmo, mass_def=shared(MassDefinition, 200, "critical"), m_range=(M_MIN, M_MAX), n_m=100)
    ccl = ccl_cosmology(cosmo, pk_linear=True, pk_nonlin=True)  # correlation_3d requires a nonlinear P(k)
    md = pyccl.halos.MassDef(200, "critical")
    conc = pyccl.halos.ConcentrationDuffy08(mass_def=md)
    hmc = pyccl.halos.HMCalculator(mass_function=pyccl.halos.MassFuncTinker08(mass_def=md),
                                   halo_bias=pyccl.halos.HaloBiasTinker10(mass_def=md), mass_def=md,
                                   log10M_min=np.log10(M_MIN), log10M_max=np.log10(M_MAX), nM=256)
    profiles = {
        "m": (NFW, pyccl.halos.HaloProfileNFW(mass_def=md, concentration=conc, truncated=True, fourier_analytic=True)),
        "y": (GNFW, pyccl.halos.HaloProfilePressureGNFW(mass_def=md, mass_bias=1 / 1.4, P0=8.13, c500=1.156,
                                                        alpha=1.062, alpha_P=0.12, beta=5.4807, gamma=0.3292,
                                                        P0_hexp=-1, qrange=(1e-5, 1e5), nq=1000, x_out=4)),
        "g": (HOD, pyccl.halos.HaloProfileHOD(mass_def=md, concentration=conc, log10Mmin_0=11.8,
                                              siglnM_0=0.68 * np.log(10), log10M1_0=12.7, alpha_0=1.3, log10M0_0=0.0)),
        "c": (CIB, pyccl.halos.HaloProfileCIBShang12(mass_def=md, concentration=conc, nu_GHz=545.0, alpha=0.36,
                                                     T0=24.4, beta=1.75, gamma=1.7, s_z=3.6, log10Meff=12.6,
                                                     siglog10M=np.sqrt(0.5), Mmin=10**11.5, L0=6.4e-8)),
    }
    two_pt = {"g": ccl_profile2pt_hod(), "c": pyccl.halos.Profile2ptCIB()}
    return hm, ccl, hmc, profiles, two_pt


@pytest.fixture(scope="module")
def setup(fixed_cosmology_m):
    return make_setup(fixed_cosmology_m)


def ccl_pk(setup, k, z, a_key, b_key=None, **kwargs):
    _, ccl, hmc, profiles, two_pt = setup
    prof2 = None if b_key is None or b_key == a_key else profiles[b_key][1]
    prof_2pt = two_pt.get(a_key) if prof2 is None else None
    return pyccl.halos.halomod_power_spectrum(ccl, hmc, k, 1.0 / (1.0 + z), profiles[a_key][1], prof2=prof2,
                                              prof_2pt=prof_2pt, **kwargs)


def hm_pk(setup, k, z, a_key, b_key=None, pk=PK, term="pk_tot"):
    hm, _, _, profiles, _ = setup
    p2 = None if b_key is None else profiles[b_key][0]
    return np.asarray(getattr(pk, term)(hm, jnp.asarray(k), jnp.array([z]), profiles[a_key][0], p2))


# ---------------------------------------------------------------------------------------------------------------------
# Power spectrum.
# ---------------------------------------------------------------------------------------------------------------------


class TestPowerSpectrumCCL:
    # Matter 1-halo and 2-halo terms against halomod_power_spectrum (measured 9.9e-4 / 8.7e-4).
    @pytest.mark.parametrize("z", Z_PK)
    def test_pk_mm(self, setup, z):
        got_1h, got_2h = hm_pk(setup, K_PK, z, "m", term="pk_1h"), hm_pk(setup, K_PK, z, "m", term="pk_2h")
        assert rel_err(got_1h, ccl_pk(setup, K_PK, z, "m", get_2h=False)) < 2e-3
        assert rel_err(got_2h, ccl_pk(setup, K_PK, z, "m", get_1h=False)) < 2e-3

    # Crosses through each branch of HaloModel.mass_integral: the HOD and CIB second moments for gg and CIB x CIB, the
    # plain product of two different profiles for matter x pressure (measured gg 6.4e-4 / 6.9e-4, CIB 3.3e-4 / 3.6e-3,
    # my 1.0e-3 / 1.3e-3).
    @pytest.mark.parametrize("a_key,b_key,tol_1h,tol_2h", [
        ("g", None, 1.3e-3, 1.4e-3),
        ("c", None, 7e-4, 7e-3),
        ("m", "y", 2e-3, 2.5e-3),
    ])
    def test_pk_cross(self, setup, a_key, b_key, tol_1h, tol_2h):
        got_1h = hm_pk(setup, K_PK, Z_X, a_key, b_key, term="pk_1h")
        got_2h = hm_pk(setup, K_PK, Z_X, a_key, b_key, term="pk_2h")
        assert rel_err(got_1h, ccl_pk(setup, K_PK, Z_X, a_key, b_key, get_2h=False)) < tol_1h
        assert rel_err(got_2h, ccl_pk(setup, K_PK, Z_X, a_key, b_key, get_1h=False)) < tol_2h

    # pk_tot's HMcode transition (P_1h^a + P_2h^a)^(1/a) against CCL's smooth_transition (measured 7.1e-4).
    def test_alpha_smooth(self, setup):
        got = hm_pk(setup, K_PK, Z_X, "m", pk=Pk(k_damp=0.0, alpha_smooth=0.7))
        assert rel_err(got, ccl_pk(setup, K_PK, Z_X, "m", smooth_transition=lambda a: 0.7)) < 1.5e-3

    # Matter P(k) with 0.3 eV neutrinos, whose neutrino fraction exposes any Omega_cb/Omega_m mix-up (measured 8.6e-4).
    def test_pk_mm_massive_neutrinos(self, fixed_cosmology):
        f = fixed_cosmology
        cosmo = Cosmology(emulator_set="mnu:v1", H0=f.H0, omega_cdm=f.omega_cdm, omega_b=f.omega_b, A_s=f.A_s,
                          n_s=f.n_s, m_ncdm=0.3, ncdm_mode="m")
        setup = make_setup(cosmo)
        for z in (0.0, 1.0):
            assert rel_err(hm_pk(setup, K_PK, z, "m"), ccl_pk(setup, K_PK, z, "m")) < 1.7e-3


class TestPowerSpectrumConsistency:
    # The 2-halo term recovers P_lin on large scales, where the consistency counterterms force I_1^1 -> 1
    # (measured 5.6e-8).
    def test_two_halo_recovers_linear_power(self, setup):
        hm = setup[0]
        k = np.geomspace(1e-4, 1e-3, 5)
        for z in Z_PK:
            want = np.ravel(hm.cosmology.pk(jnp.asarray(k), jnp.array([z]), linear=True))
            assert rel_err(hm_pk(setup, k, z, "m", term="pk_2h"), want) < 1.2e-7

    # pk_tot with alpha_smooth = 1 is exactly the sum of its terms.
    def test_pk_tot_is_sum_of_terms(self, setup):
        got = hm_pk(setup, K_PK, Z_X, "m")
        terms = hm_pk(setup, K_PK, Z_X, "m", term="pk_1h") + hm_pk(setup, K_PK, Z_X, "m", term="pk_2h")
        assert rel_err(got, terms) < 1e-12

    # k_damp multiplies the 1-halo term by 1 - exp(-(k / k_damp)^2).
    def test_k_damp_suppression(self, setup):
        k_damp = 0.05
        got = hm_pk(setup, K_PK, Z_X, "m", pk=Pk(k_damp=k_damp), term="pk_1h")
        want = hm_pk(setup, K_PK, Z_X, "m", term="pk_1h") * (1.0 - np.exp(-((K_PK / k_damp) ** 2)))
        assert rel_err(got, want) < 1e-12


# ---------------------------------------------------------------------------------------------------------------------
# Bispectrum, against a reference assembled from CCL's mass integrals.
# ---------------------------------------------------------------------------------------------------------------------

K_BK = np.geomspace(1e-3, 10.0, 30)
# (k2 / k1, k3 / k1) for each triangle shape.
SHAPES = {"equilateral": (1.0, 1.0), "squeezed": (1.0, 0.1), "folded": (1.0, 2.0)}


def _f2(k1, k2, mu):
    return 5.0 / 7.0 + 0.5 * mu * (k1 / k2 + k2 / k1) + 2.0 / 7.0 * mu**2


def _cos_opposite(ka, kb, kc):
    """Cosine between ka and kb in a closed triangle with third side kc."""
    return (kc**2 - ka**2 - kb**2) / (2 * ka * kb)


def ccl_bk(setup, k1, k2, k3, keys, z):
    """1h, 2h and the b1 part of 3h from CCL's mass integrals; profile i sits at k_{i+1}."""
    _, ccl, hmc, profiles, _ = setup
    profs = [profiles[key][1] for key in keys]
    a = 1.0 / (1.0 + z)
    n = len(k1)
    k_all = np.concatenate([k1, k2, k3])
    leg = [slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)]
    norm = [p.get_normalization(ccl, a, hmc=hmc) for p in profs]
    u = [p.fourier(ccl, k_all[leg[i]], hmc._mass, a) / norm[i] for i, p in enumerate(profs)]
    I1 = [hmc.I_1_1(ccl, k_all[leg[i]], a, p) / norm[i] for i, p in enumerate(profs)]
    P = [pyccl.linear_matter_power(ccl, k_all[leg[i]], a) for i in range(3)]
    idx = np.arange(n)

    def I2(i, j):
        # CCL's I_1_2(p, q)[j, i] has p at k[i] and q at k[j].
        m = hmc.I_1_2(ccl, k_all, a, profs[i], prof2=profs[j], prof_2pt=pyccl.halos.Profile2pt(), diag=False)
        return m[leg[j].start + idx, leg[i].start + idx] / (norm[i] * norm[j])

    b_pt = (2 * _f2(k1, k2, _cos_opposite(k1, k2, k3)) * P[0] * P[1]
            + 2 * _f2(k2, k3, _cos_opposite(k2, k3, k1)) * P[1] * P[2]
            + 2 * _f2(k3, k1, _cos_opposite(k3, k1, k2)) * P[2] * P[0])
    return {
        "1h": hmc.integrate_over_massfunc(lambda M: (u[0] * u[1] * u[2]).T, ccl, a),
        "2h": P[0] * I1[0] * I2(1, 2) + P[1] * I1[1] * I2(0, 2) + P[2] * I1[2] * I2(0, 1),
        "3h": b_pt * I1[0] * I1[1] * I1[2],
    }


def hm_bk(setup, k1, k2, mu12, keys, z):
    """hmfast 1h, 2h and 3h, with the second-order-bias part of 3h (absent from the reference) subtracted."""
    hm, _, _, profiles, _ = setup
    profs = [profiles[key][0] for key in keys]
    za = jnp.array([z])
    out = {t: np.asarray(getattr(BK, f"bk_{t}")(hm, jnp.asarray(k1), jnp.asarray(k2), mu12, za, *profs))
           for t in ("1h", "2h", "3h")}
    ks = (k1, k2, np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12))

    def mi(i, order):
        return np.asarray(hm.mass_integral(jnp.asarray(ks[i]), za, profs[i], bias_order=order))

    P = [np.ravel(hm.cosmology.pk(jnp.asarray(kk), za, linear=True)) for kk in ks]
    b2 = (mi(0, 2) * mi(1, 1) * mi(2, 1) * P[1] * P[2] + mi(0, 1) * mi(1, 2) * mi(2, 1) * P[0] * P[2]
          + mi(0, 1) * mi(1, 1) * mi(2, 2) * P[0] * P[1])
    out["3h"] = out["3h"] - b2
    return out


class TestBispectrumCCL:
    # Matter 1h, 2h and 3h (first-order bias part) for three triangle shapes (measured 5.4e-4 / 9.6e-4 / 1.3e-3).
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("term,tol", [("1h", 1.1e-3), ("2h", 2e-3), ("3h", 2.7e-3)])
    def test_bk_mmm(self, setup, shape, term, tol):
        lam2, lam3 = SHAPES[shape]
        mu12 = (lam3**2 - 1.0 - lam2**2) / (2.0 * lam2)
        got = hm_bk(setup, K_BK, lam2 * K_BK, mu12, "mmm", Z_X)[term]
        assert rel_err(got, ccl_bk(setup, K_BK, lam2 * K_BK, lam3 * K_BK, "mmm", Z_X)[term]) < tol

    # Matter x matter x pressure with pressure on each leg in turn, squeezed (measured 1.4e-3).
    @pytest.mark.parametrize("keys", ["ymm", "mym", "mmy"])
    def test_bk_mixed_legs(self, setup, keys):
        lam2, lam3 = SHAPES["squeezed"]
        mu12 = (lam3**2 - 1.0 - lam2**2) / (2.0 * lam2)
        got = hm_bk(setup, K_BK, lam2 * K_BK, mu12, keys, Z_X)
        want = ccl_bk(setup, K_BK, lam2 * K_BK, lam3 * K_BK, keys, Z_X)
        for term in got:
            assert rel_err(got[term], want[term]) < 3e-3

    # For an equilateral triangle the total cannot depend on which leg carries pressure.
    def test_bk_leg_symmetry(self, setup):
        hm, _, _, profiles, _ = setup
        m, y = profiles["m"][0], profiles["y"][0]
        tot = [np.asarray(BK.bk_tot(hm, jnp.asarray(K_BK), jnp.asarray(K_BK), -0.5, jnp.array([Z_X]), *p))
               for p in ((y, m, m), (m, y, m), (m, m, y))]
        assert rel_err(tot[0], tot[1]) < 1e-10 and rel_err(tot[0], tot[2]) < 1e-10


# ---------------------------------------------------------------------------------------------------------------------
# Trispectrum.
# ---------------------------------------------------------------------------------------------------------------------

K_TK = np.geomspace(0.02, 2.0, 20)
TK_TERMS = ("1h", "2h", "3h", "4h")
# Far from k_u = k_v the tree-level trispectrum nearly cancels, amplifying input P_lin differences; checked separately.
NEAR_DIAGONAL = np.abs(np.log(K_TK[:, None] / K_TK[None, :])) <= np.log(2.0) + 1e-9


def hm_tk(setup, keys):
    hm, _, _, profiles, _ = setup
    profs = [profiles[key][0] for key in keys]
    return {t: np.asarray(getattr(TK, f"tk_{t}")(hm, jnp.asarray(K_TK), jnp.asarray(K_TK), jnp.array([Z_X]), *profs))
            for t in TK_TERMS}


@pytest.fixture(scope="module")
def tk_mmmm(setup):
    _, ccl, hmc, profiles, _ = setup
    nfw, a = profiles["m"][1], 1.0 / (1.0 + Z_X)
    halomod = pyccl.halos
    # CCL returns [k_v, k_u], transposed here to hmfast's [k_u, k_v].
    ccl_terms = {
        "1h": halomod.halomod_trispectrum_1h(ccl, hmc, K_TK, a, nfw).T,
        "2h": (halomod.halomod_trispectrum_2h_22(ccl, hmc, K_TK, a, nfw)
               + halomod.halomod_trispectrum_2h_13(ccl, hmc, K_TK, a, nfw)).T,
        "3h": halomod.halomod_trispectrum_3h(ccl, hmc, K_TK, a, nfw).T,
        "4h": halomod.halomod_trispectrum_4h(ccl, hmc, K_TK, a, nfw).T,
    }
    return hm_tk(setup, "mmmm"), ccl_terms


def ccl_tk_mixed(setup, keys, tk_4h_matter):
    """T(k_u, k_v) with legs 1, 2 at k_u and 3, 4 at k_v, from CCL's mass integrals and hmfast's PT kernels."""
    hm, ccl, hmc, profiles, _ = setup
    profs = [profiles[key][1] for key in keys]
    a = 1.0 / (1.0 + Z_X)
    norm = [p.get_normalization(ccl, a, hmc=hmc) for p in profs]
    I1 = [hmc.I_1_1(ccl, K_TK, a, p) / n for p, n in zip(profs, norm)]
    Iu, Iv = [x[:, None] for x in I1], [x[None, :] for x in I1]

    def J(i, j):
        # Leg i at k_u, leg j at k_v, as [iu, iv].
        m = hmc.I_1_2(ccl, K_TK, a, profs[i], prof2=profs[j], prof_2pt=pyccl.halos.Profile2pt(), diag=False)
        return m.T / (norm[i] * norm[j])

    def K(i, j, l):
        # Leg i alone at one wavenumber, legs j and l at the other, as [k_i index, k_jl index].
        m = hmc.I_1_3(ccl, K_TK, a, profs[i], prof2=profs[j], prof3=profs[l], prof_2pt=pyccl.halos.Profile2pt())
        return m.T / (norm[i] * norm[j] * norm[l])

    za = jnp.array([Z_X])
    p_bar = np.asarray(TK._Pbar_kernel(hm, jnp.asarray(K_TK), jnp.asarray(K_TK), za))[..., 0]
    b_pt = np.asarray(TK._Bpt_kernel(hm, jnp.asarray(K_TK), jnp.asarray(K_TK), za))[..., 0]
    p_lin = pyccl.linear_matter_power(ccl, K_TK, a)
    m_ccl = profiles["m"][1]
    I_m = hmc.I_1_1(ccl, K_TK, a, m_ccl) / m_ccl.get_normalization(ccl, a, hmc=hmc)
    t_pt = tk_4h_matter / (I_m[:, None] ** 2 * I_m[None, :] ** 2)
    return {
        "1h": pyccl.halos.halomod_trispectrum_1h(ccl, hmc, K_TK, a, profs[0], prof2=profs[1], prof3=profs[2],
                                                 prof4=profs[3]).T,
        "2h": (p_bar * (J(0, 2) * J(1, 3) + J(0, 3) * J(1, 2))
               + p_lin[:, None] * (Iu[0] * K(1, 2, 3) + Iu[1] * K(0, 2, 3))
               + p_lin[None, :] * (Iv[2] * K(3, 0, 1).T + Iv[3] * K(2, 0, 1).T)),
        "3h": b_pt * (Iu[0] * Iv[2] * J(1, 3) + Iu[0] * Iv[3] * J(1, 2) + Iu[1] * Iv[2] * J(0, 3)
                      + Iu[1] * Iv[3] * J(0, 2)),
        "4h": t_pt * Iu[0] * Iu[1] * Iv[2] * Iv[3],
    }


class TestTrispectrumCCL:
    # Matter 1h-4h against CCL's native trispectrum over the (k_u, k_v) plane, the 4-halo term within a factor 2 of
    # the diagonal (measured 7.5e-4 / 1.2e-3 / 1.3e-3 / 2.6e-3).
    @pytest.mark.parametrize("term,tol", [("1h", 1.5e-3), ("2h", 2.4e-3), ("3h", 2.5e-3), ("4h", 5.2e-3)])
    def test_tk_mmmm(self, tk_mmmm, term, tol):
        got, want = tk_mmmm[0][term], tk_mmmm[1][term]
        if term == "4h":
            got, want = got[NEAR_DIAGONAL], want[NEAR_DIAGONAL]
        assert rel_err(got, want) < tol

    # Too few angle points miss the test_tk_mmmm accuracy (measured 3.9e-2 / 5.1e-2 / 5.2e-2 at n_theta = 32).
    @pytest.mark.xfail(strict=True, reason="angle average unconverged at small n_theta; no accuracy control yet")
    @pytest.mark.parametrize("term,tol", [("2h", 2.4e-3), ("3h", 2.5e-3), ("4h", 5.2e-3)])
    def test_tk_mmmm_few_angles(self, setup, tk_mmmm, term, tol):
        hm, _, _, profiles, _ = setup
        nfw = profiles["m"][0]
        got = np.asarray(getattr(Tk(n_theta=32), f"tk_{term}")(hm, jnp.asarray(K_TK), jnp.asarray(K_TK),
                                                               jnp.array([Z_X]), nfw, nfw, nfw, nfw))
        want = tk_mmmm[1][term]
        if term == "4h":
            got, want = got[NEAR_DIAGONAL], want[NEAR_DIAGONAL]
        assert rel_err(got, want) < tol

    # The 4-halo term over the full plane, out to k_u / k_v = 100. Both codes evaluate the same tree-level formula to
    # 1e-5, but it nearly cancels there and amplifies the up-to-1e-3 difference between hmfast's log-linear P_lin
    # interpolation and CCL's spline of the same nodes (measured 2.4e-2).
    def test_tk_mmmm_4h_far_from_diagonal(self, tk_mmmm):
        assert rel_err(tk_mmmm[0]["4h"], tk_mmmm[1]["4h"]) < 5e-2

    # Matter, matter | pressure, pressure against the assembled reference (measured 2.1e-3 / 2.6e-3 / 1.8e-3 / 3.3e-3).
    @pytest.mark.parametrize("term,tol", [("1h", 4.2e-3), ("2h", 5.2e-3), ("3h", 3.6e-3), ("4h", 6.5e-3)])
    def test_tk_mixed_legs(self, setup, tk_mmmm, term, tol):
        got = hm_tk(setup, "mmyy")[term]
        want = ccl_tk_mixed(setup, "mmyy", tk_mmmm[1]["4h"])[term]
        if term == "4h":
            got, want = got[NEAR_DIAGONAL], want[NEAR_DIAGONAL]
        assert rel_err(got, want) < tol

    # Exchanging the k_u and k_v leg pairs transposes the trispectrum.
    def test_tk_pair_exchange(self, setup):
        a, b = hm_tk(setup, "mmyy"), hm_tk(setup, "yymm")
        for t in TK_TERMS:
            assert rel_err(a[t], b[t].T) < 1e-10


# ---------------------------------------------------------------------------------------------------------------------
# 3D correlation function.
# ---------------------------------------------------------------------------------------------------------------------

K_FFT = np.geomspace(1e-4, 100.0, 4096)
R = np.geomspace(1.0, 150.0, 60)


class TestCorr3dCCL:
    # corr_3d of the linear and halo-model matter P(k) against pyccl.correlation_3d over 1 < r < 150 Mpc, where |xi|
    # exceeds 1e-3 of its maximum (excludes the zero crossing); CCL needs its P(k) tabulated to k = 1e3 or it rings
    # (measured 4.4e-5 / 3.2e-4).
    @pytest.mark.parametrize("which,tol", [("linear", 1e-4), ("halo model", 6.5e-4)])
    def test_corr_3d(self, setup, which, tol):
        hm, ccl, hmc, profiles, _ = setup
        za, a = jnp.array([Z_X]), 1.0 / (1.0 + Z_X)
        if which == "linear":
            pk = hm.cosmology.pk(jnp.asarray(K_FFT), za, linear=True)
            p_of_k_a = ccl.get_linear_power()
        else:
            pk = PK.pk_tot(hm, jnp.asarray(K_FFT), za, NFW)
            lk = np.log(np.geomspace(1e-5, 1e3, 2048))
            p_of_k_a = pyccl.halos.halomod_Pk2D(ccl, hmc, profiles["m"][1], lk_arr=lk, a_arr=np.linspace(0.3, 1.0, 30))
        got = np.asarray(corr_3d(hm.cosmology, K_FFT, pk, R))
        want = pyccl.correlation_3d(ccl, r=R, a=a, p_of_k_a=p_of_k_a)
        keep = np.abs(want) > 1e-3 * np.max(np.abs(want))
        assert rel_err(got[keep], want[keep]) < tol
