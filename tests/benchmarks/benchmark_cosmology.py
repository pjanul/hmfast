"""
Benchmarks for hmfast.cosmology.Cosmology against external ground truth: CCL running CLASS for the background, growth
and power spectra (CAMB for the HMcode nonlinear spectrum), and CLASS directly (classy) for the CMB spectra and derived
parameters. Every comparison runs hmfast with ncdm_mode="m", the neutrino convention CCL uses, against a CCL cosmology
with the same neutrino masses (see `ccl_cosmology`).

CCL tests are marked `ccl`, the CAMB test `camb` and the CLASS tests `classy`; each is skipped if its package is
missing.

Thresholds are ~2x the error measured when written (noted per test), so a ~2x degradation fails; lower them when
accuracy improves. Errors are maximum relative errors unless stated otherwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.cosmology import CosmoPowerCosmology

from .._shared import ccl_cosmology, emulator_z_max_bg, emulator_z_max_pk, peak_err, rel_err

try:
    import pyccl
except ImportError:
    pyccl = None
try:
    import classy
except ImportError:
    classy = None

REQUIRES_CCL = [pytest.mark.ccl, pytest.mark.skipif(pyccl is None, reason="requires pyccl")]


def z_to_a(z):
    return 1.0 / (1.0 + np.asarray(z))


# In-grid redshifts: background emulators run to z = 20, the P(k) emulators to z = 5.
Z_BG = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 19.0]
Z_PK = [0.0, 0.5, 1.0, 2.0, 3.0, 4.9]
# growth_rate is a finite difference of the growth table; its end nodes, less accurate, are tested separately.
Z_GROWTH_RATE = [0.1, 0.5, 1.0, 2.0, 3.0, 4.5]
Z_LOW = [0.001, 0.003, 0.01, 0.03, 0.1]
M_GRID = np.geomspace(1e9, 1e16, 30)
R_GRID = np.geomspace(1e-1, 1e2, 30)
K_FULL = np.geomspace(1e-4, 48.0, 60)


@pytest.fixture(scope="module")
def cosmo(fixed_cosmology_m):
    return fixed_cosmology_m


@pytest.fixture(scope="module")
def cosmo_ext(fixed_cosmology_m):
    return fixed_cosmology_m.update(extrapolate_z=True)


@pytest.fixture(scope="module")
def ccl_class(fixed_cosmology_m):
    """CCL running CLASS for the same cosmology: an independent background, growth and linear P(k)."""
    return ccl_cosmology(fixed_cosmology_m)


@pytest.fixture(scope="module")
def ccl_calc(fixed_cosmology_m):
    """CCL fed hmfast's own P_lin, isolating the sigma(M)/sigma(R) integral from P(k) differences."""
    return ccl_cosmology(fixed_cosmology_m, pk_linear=True)


class TestBackgroundCCL:
    pytestmark = REQUIRES_CCL

    # H(z) and D_A(z) across the background emulators' range (measured 1.4e-4 / 4.1e-5).
    def test_hubble_and_distance_in_grid(self, cosmo, ccl_class):
        z, a = jnp.asarray(Z_BG), z_to_a(Z_BG)
        assert rel_err(cosmo.hubble_parameter(z), cosmo.H0 * pyccl.h_over_h0(ccl_class, a)) < 3e-4
        assert rel_err(cosmo.angular_diameter_distance(z), pyccl.angular_diameter_distance(ccl_class, a)) < 1e-4

    # D_A(z) at low z, where linear interpolation across the first background-grid bin (dz = 0.004) dominates
    # (measured 3.6e-3 at z = 0.001, 4.6e-4 at z = 0.01).
    def test_distance_at_low_redshift(self, cosmo, ccl_class):
        z, a = jnp.asarray(Z_LOW), z_to_a(Z_LOW)
        assert rel_err(cosmo.angular_diameter_distance(z), pyccl.angular_diameter_distance(ccl_class, a)) < 7e-3

    # sigma8(z) across the background emulators' range (measured 6.4e-5).
    def test_sigma8_in_grid(self, cosmo, ccl_class):
        R8 = 8.0 / (cosmo.H0 / 100.0)
        s8_ccl = [pyccl.sigmaR(ccl_class, R8, float(a)) for a in z_to_a(Z_BG)]
        assert rel_err(cosmo.sigma8(jnp.asarray(Z_BG)), s8_ccl) < 1.5e-4

    # With extrapolate_z=True, H(z) and D_A(z) continue past z_max_bg = 20 (measured 1.2e-3 / 1.2e-4).
    def test_hubble_and_distance_beyond_grid_with_extrapolation(self, cosmo_ext, ccl_class):
        z_max_bg = emulator_z_max_bg(cosmo_ext)
        z = jnp.array([z_max_bg + 5.0, z_max_bg + 50.0, 100.0, 1000.0])
        a = z_to_a(z)
        assert rel_err(cosmo_ext.hubble_parameter(z), cosmo_ext.H0 * pyccl.h_over_h0(ccl_class, a)) < 2.5e-3
        assert rel_err(cosmo_ext.angular_diameter_distance(z), pyccl.angular_diameter_distance(ccl_class, a)) < 2.5e-4

    # sigma8 has no extrapolation branch: NaN beyond z_max_bg regardless of extrapolate_z.
    def test_sigma8_always_nan_beyond_grid(self, cosmo, cosmo_ext):
        z_beyond = jnp.array(emulator_z_max_bg(cosmo) + 5.0)
        assert jnp.isnan(cosmo.sigma8(z_beyond))
        assert jnp.isnan(cosmo_ext.sigma8(z_beyond))


class TestDensitiesCCL:
    pytestmark = REQUIRES_CCL

    # critical_density(z) in-grid and beyond z_max_bg (measured 1.0e-3).
    def test_critical_density(self, cosmo_ext, ccl_class):
        z = np.array(Z_BG + [70.0])
        got = cosmo_ext.critical_density(jnp.asarray(z))
        assert rel_err(got, pyccl.rho_x(ccl_class, z_to_a(z), "critical", is_comoving=False)) < 2e-3

    # omega_m(z), total matter including massive neutrinos, in-grid and beyond z_max_bg (measured 2.1e-3).
    def test_omega_m(self, cosmo_ext, ccl_class):
        z = np.array(Z_BG + [70.0])
        assert rel_err(cosmo_ext.omega_m(jnp.asarray(z)), pyccl.omega_x(ccl_class, z_to_a(z), "matter")) < 4e-3

    # comoving_volume_element(z) against CCL's D_A and H, in-grid and beyond z_max_bg (measured 5.9e-4).
    def test_comoving_volume_element(self, cosmo_ext, ccl_class):
        z = np.array(Z_BG + [70.0])
        a = z_to_a(z)
        c_km_s = 299792.458
        D_A = pyccl.angular_diameter_distance(ccl_class, a)
        H = cosmo_ext.H0 * pyccl.h_over_h0(ccl_class, a)
        assert rel_err(cosmo_ext.comoving_volume_element(jnp.asarray(z)), (1 + z) ** 2 * D_A**2 * c_km_s / H) < 1.2e-3


class TestGrowthCCL:
    pytestmark = REQUIRES_CCL

    # growth_factor(z) across the P(k) emulators' range (measured 1.3e-3).
    def test_growth_factor(self, cosmo, ccl_class):
        assert rel_err(cosmo.growth_factor(jnp.asarray(Z_PK)), pyccl.growth_factor(ccl_class, z_to_a(Z_PK))) < 2.5e-3

    # growth_rate(z) away from the growth table's edges (measured 1.1e-3).
    def test_growth_rate(self, cosmo, ccl_class):
        got = cosmo.growth_rate(jnp.asarray(Z_GROWTH_RATE))
        assert rel_err(got, pyccl.growth_rate(ccl_class, z_to_a(Z_GROWTH_RATE))) < 2.5e-3

    # growth_rate at the table's first node, z = 0, from a second-order one-sided difference (measured 3.7e-3).
    def test_growth_rate_at_z0(self, cosmo, ccl_class):
        assert rel_err(cosmo.growth_rate(jnp.array([0.0])), pyccl.growth_rate(ccl_class, z_to_a(0.0))) < 7.5e-3

    # With extrapolate_z=True, growth_factor continues past z_max_pk = 5 (measured 1.3e-3); NaN without the flag.
    def test_growth_factor_beyond_grid_with_extrapolation(self, cosmo, cosmo_ext, ccl_class):
        z_max_pk = emulator_z_max_pk(cosmo)
        z = jnp.array([z_max_pk + 2.0, z_max_pk + 10.0])
        assert jnp.all(jnp.isnan(cosmo.growth_factor(z)))
        assert rel_err(cosmo_ext.growth_factor(z), pyccl.growth_factor(ccl_class, z_to_a(z))) < 2.5e-3

    # growth_rate has no extrapolation branch: NaN beyond z_max_pk unless extrapolating.
    def test_growth_rate_nan_beyond_grid_unless_extrapolating(self, cosmo, cosmo_ext):
        z_beyond = jnp.array(emulator_z_max_pk(cosmo) + 2.0)
        assert jnp.isnan(cosmo.growth_rate(z_beyond))
        assert jnp.isfinite(cosmo_ext.growth_rate(z_beyond))

    # velocity_dispersion(z) against (f a H / c)^2 / 3 * int P_lin dk / 2 pi^2 built from CCL's own ingredients
    # (measured 1.9e-3; 7.1e-3 at z = 0, which inherits growth_rate's larger error there).
    def test_velocity_dispersion(self, cosmo, ccl_class):
        z = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
        a = z_to_a(z)
        k = np.geomspace(1e-5, 50.0, 4000)
        pk = np.array([pyccl.linear_matter_power(ccl_class, k, float(ai)) for ai in a])
        f_aH = pyccl.growth_rate(ccl_class, a) * a * cosmo.H0 * pyccl.h_over_h0(ccl_class, a) / 299792.458
        want = f_aH**2 / 3 * np.trapezoid(pk * k, np.log(k), axis=1) / (2 * np.pi**2)
        got = np.asarray(cosmo.velocity_dispersion(jnp.asarray(z)))
        assert rel_err(got[1:], want[1:]) < 4e-3
        assert rel_err(got[0], want[0]) < 1.5e-2


class TestDeltaCCL:
    pytestmark = REQUIRES_CCL

    # delta_c for all 3 prescriptions (hmfast "NS97" is CCL "NakamuraSuto97"; measured 1.5e-6).
    @pytest.mark.parametrize("hmfast_kind,ccl_kind", [
        ("EdS", "EdS"),
        ("EdS_approx", "EdS_approx"),
        ("NS97", "NakamuraSuto97"),
    ])
    def test_delta_c(self, cosmo, ccl_class, hmfast_kind, ccl_kind):
        z = np.array(Z_BG)
        got = np.broadcast_to(np.asarray(cosmo.delta_c(jnp.asarray(z), prescription=hmfast_kind)), z.shape)
        assert rel_err(got, pyccl.halos.get_delta_c(ccl_class, z_to_a(z), kind=ccl_kind)) < 3e-6


class TestSigmaCCL:
    pytestmark = REQUIRES_CCL

    # sigma(M) and sigma(R) on hmfast's own P_lin (measured 6.3e-5 / 3.6e-5).
    @pytest.mark.parametrize("z", [0.0, 1.0, 3.0])
    def test_sigma_m(self, cosmo, ccl_calc, z):
        assert rel_err(cosmo.sigma_m(jnp.asarray(M_GRID), z), pyccl.sigmaM(ccl_calc, M_GRID, z_to_a(z))) < 1.3e-4

    @pytest.mark.parametrize("z", [0.0, 1.0, 3.0])
    def test_sigma_r(self, cosmo, ccl_calc, z):
        assert rel_err(cosmo.sigma_r(jnp.asarray(R_GRID), z), pyccl.sigmaR(ccl_calc, R_GRID, z_to_a(z))) < 8e-5


class TestLinearPowerCCL:
    pytestmark = REQUIRES_CCL

    # P_lin(k, z) over the emulator's full k range (measured 1.1e-3).
    @pytest.mark.parametrize("z", Z_PK)
    def test_pk_linear(self, cosmo, ccl_class, z):
        got = cosmo.pk(jnp.asarray(K_FULL), jnp.array([z]), linear=True)
        assert rel_err(np.ravel(got), pyccl.linear_matter_power(ccl_class, K_FULL, z_to_a(z))) < 2.2e-3

    # Without extrapolate_k, pk is NaN beyond the trained k range; with it, a modest factor beyond (measured 3.4e-3).
    def test_pk_k_extrapolation(self, cosmo, ccl_class):
        k_native = cosmo._pk_grid()
        k = jnp.array([k_native.min() * 0.5, k_native.max() * 2.0])
        assert jnp.all(jnp.isnan(cosmo.update(extrapolate_k=False).pk(k, jnp.array([1.0]), linear=True)))
        got = np.ravel(cosmo.update(extrapolate_k=True).pk(k, jnp.array([1.0]), linear=True))
        assert rel_err(got, pyccl.linear_matter_power(ccl_class, np.asarray(k), z_to_a(1.0))) < 7e-3

    # Without extrapolate_z, pk is NaN beyond z_max_pk; with it, P(k) scales by CCL's growth-factor ratio
    # (measured 3.3e-5).
    def test_pk_z_extrapolation_growth_ratio(self, cosmo, cosmo_ext, ccl_class):
        z_max_pk = emulator_z_max_pk(cosmo)
        k0 = jnp.array([0.05])
        z = jnp.array([z_max_pk + 2.0, z_max_pk + 10.0])
        assert jnp.all(jnp.isnan(cosmo.pk(k0, z, linear=True)))
        ratio = np.ravel(cosmo_ext.pk(k0, z, linear=True)) / np.ravel(cosmo.pk(k0, jnp.array([z_max_pk])))[0]
        D = pyccl.growth_factor(ccl_class, z_to_a(z)) / pyccl.growth_factor(ccl_class, z_to_a(z_max_pk))
        assert rel_err(ratio, D**2) < 7e-5


class TestNonlinearPowerCCL:
    pytestmark = REQUIRES_CCL + [pytest.mark.camb]

    # HMcode P_nl(k, z) against CCL running CAMB with HMcode, the only fair truth for the HMcode-trained emulator
    # (measured 5.1e-3 at z = 0, 3.0e-3 above).
    @pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0])
    def test_pk_hmcode(self, cosmo, z):
        pytest.importorskip("camb")
        ccl_camb = _ccl_camb(cosmo)
        k = np.geomspace(1e-3, 10.0, 40)
        got = np.ravel(cosmo.pk(jnp.asarray(k), jnp.array([z]), linear=False))
        assert rel_err(got, pyccl.nonlin_matter_power(ccl_camb, k, z_to_a(z))) < 1e-2


_CCL_CAMB = {}


def _ccl_camb(cosmo):
    if id(cosmo) not in _CCL_CAMB:
        _CCL_CAMB[id(cosmo)] = ccl_cosmology(cosmo, transfer_function="boltzmann_camb", matter_power_spectrum="camb")
    return _CCL_CAMB[id(cosmo)]


# EDE sets are excluded throughout: CCL cannot represent an EDE background.
OTHER_SETS = [
    ("lcdm:v1", {}),
    ("wcdm:v1", {"w0": -0.8}),
    ("mnu:v1", {"m_ncdm": 0.3}),
    ("mnu-3states:v1", {"m_ncdm": 0.1}),
    ("neff:v1", {"N_ur": 3.5}),
]


@pytest.fixture(scope="module", params=OTHER_SETS, ids=[s for s, _ in OTHER_SETS])
def emulator_set_cosmology(request, fixed_cosmology):
    """Each non-EDE emulator set at the fixed cosmology plus one extension value, in ncdm_mode="m"."""
    emulator_set, extension = request.param
    f = fixed_cosmology
    try:
        return CosmoPowerCosmology(emulator_set, H0=f.H0, omega_cdm=f.omega_cdm, omega_b=f.omega_b, A_s=f.A_s,
                         n_s=f.n_s, ncdm_mode="m", **extension)
    except Exception as exc:
        pytest.skip(f"{emulator_set} emulator files not available locally: {exc}")


class TestEmulatorSetsCCL:
    pytestmark = REQUIRES_CCL

    @pytest.fixture(scope="class")
    def pair(self, emulator_set_cosmology):
        return emulator_set_cosmology, ccl_cosmology(emulator_set_cosmology)

    # H(z), D_A(z) per emulator set (measured max over sets 1.4e-4 / 4.1e-5).
    def test_background(self, pair):
        cosmo, ccl = pair
        z = jnp.asarray(Z_BG[:-1])
        a = z_to_a(z)
        assert rel_err(cosmo.hubble_parameter(z), cosmo.H0 * pyccl.h_over_h0(ccl, a)) < 3e-4
        assert rel_err(cosmo.angular_diameter_distance(z), pyccl.angular_diameter_distance(ccl, a)) < 1e-4

    # growth_factor(z), growth_rate(z) per emulator set (measured max over sets 3.6e-3 / 3.3e-3, both mnu-3states). With
    # massive neutrinos growth is scale dependent: hmfast reads it off P_lin at k = 0.01 / Mpc, CCL solves an ODE.
    def test_growth(self, pair):
        cosmo, ccl = pair
        assert rel_err(cosmo.growth_factor(jnp.asarray(Z_PK)), pyccl.growth_factor(ccl, z_to_a(Z_PK))) < 7e-3
        got = cosmo.growth_rate(jnp.asarray(Z_GROWTH_RATE))
        assert rel_err(got, pyccl.growth_rate(ccl, z_to_a(Z_GROWTH_RATE))) < 7e-3

    # sigma8(z) per emulator set (measured max over sets 1.3e-4).
    def test_sigma8(self, pair):
        cosmo, ccl = pair
        R8 = 8.0 / (cosmo.H0 / 100.0)
        z = Z_BG[:-2]
        assert rel_err(cosmo.sigma8(jnp.asarray(z)), [pyccl.sigmaR(ccl, R8, float(a)) for a in z_to_a(z)]) < 2.5e-4

    # P_lin(k, z) per emulator set over k <= 10 / Mpc (measured max over sets 1.6e-3).
    def test_pk_linear(self, pair):
        cosmo, ccl = pair
        k = np.geomspace(1e-4, 10.0, 40)
        got = np.asarray(cosmo.pk(jnp.asarray(k), jnp.asarray(Z_PK), linear=True))
        want = np.stack([pyccl.linear_matter_power(ccl, k, float(a)) for a in z_to_a(Z_PK)], axis=1)
        assert rel_err(got, want) < 3.3e-3


# ---------------------------------------------------------------------------------------------------------------------
# CMB spectra and derived parameters against CLASS run with the lcdm:v1 training settings (N_ur = 2.0328 plus one
# massive state, HMcode lensing); the emulators were trained on CLASS v2.9.4, so this includes version differences.
# ---------------------------------------------------------------------------------------------------------------------

L_CMB = np.arange(2, 5001)


@pytest.fixture(scope="module")
def class_reference(cosmo):
    params = {
        "output": "tCl,pCl,lCl,mPk", "lensing": "yes", "non_linear": "hmcode", "hmcode_version": "2016",
        "l_max_scalars": int(L_CMB[-1]), "P_k_max_1/Mpc": 50.0,
        "H0": cosmo.H0, "omega_cdm": cosmo.omega_cdm, "omega_b": cosmo.omega_b, "A_s": cosmo.A_s, "n_s": cosmo.n_s,
        "tau_reio": cosmo.tau, "N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": float(cosmo.m_ncdm),
        "tol_perturbations_integration": 1e-7, "perturbations_sampling_stepsize": 0.05, "accurate_lensing": 1,
        "delta_l_max": 1000, "l_logstep": 1.025, "l_linstep": 15,
    }
    c = classy.Class()
    c.set(params)
    c.compute()
    yield c
    c.struct_cleanup()
    c.empty()


@pytest.mark.classy
@pytest.mark.skipif(classy is None, reason="requires classy")
class TestCMBClass:
    # D_l in hmfast's convention, l(l+1) C_l / 2 pi ([l(l+1)]^2 C_l / 2 pi for PP). Errors are max |difference| over
    # max |D_l|, since EE and TE cross zero (measured TT 3.8e-4, EE 7.0e-4, TE 9.6e-4, PP 2.6e-3).
    @pytest.mark.parametrize("kind,tol", [("TT", 8e-4), ("EE", 1.4e-3), ("TE", 2e-3), ("PP", 5.3e-3)])
    def test_cl_cmb(self, cosmo, class_reference, kind, tol):
        cls = class_reference.lensed_cl(int(L_CMB[-1]))
        ell = L_CMB
        fac = (ell * (ell + 1)) ** 2 / (2 * np.pi) if kind == "PP" else ell * (ell + 1) / (2 * np.pi)
        got = np.asarray(cosmo.cl_cmb(kind, jnp.asarray(ell)))
        assert peak_err(got, cls[kind.lower()][ell] * fac) < tol

    # derived_parameters() against CLASS's own (hmfast chi_rec/chi_star/rs_drag are CLASS ra_rec/ra_star/rs_d); measured
    # 3.6e-4 for YHe, 2.6e-4 for z_reio, 4.7e-5 for sigma8, <= 1.3e-5 for the rest.
    @pytest.mark.parametrize("hmfast_key,class_key,tol", [
        ("100*theta_s", "100*theta_s", 1.6e-5),
        ("sigma8", "sigma8", 1e-4),
        ("YHe", "YHe", 7e-4),
        ("z_reio", "z_reio", 5e-4),
        ("Neff", "Neff", 2e-7),
        ("tau_rec", "tau_rec", 1e-5),
        ("z_rec", "z_rec", 7e-6),
        ("rs_rec", "rs_rec", 1.1e-5),
        ("chi_rec", "ra_rec", 2e-5),
        ("tau_star", "tau_star", 1e-5),
        ("z_star", "z_star", 7e-6),
        ("rs_star", "rs_star", 1.5e-5),
        ("chi_star", "ra_star", 1e-5),
        ("rs_drag", "rs_d", 2.5e-5),
    ])
    def test_derived_parameters(self, cosmo, class_reference, hmfast_key, class_key, tol):
        want = class_reference.get_current_derived_parameters([class_key])[class_key]
        assert rel_err(cosmo.derived_parameters()[hmfast_key], want) < tol
