"""
Benchmarks for hmfast.halos.profiles against external ground truth: CCL for NFW, GNFW, HOD and Shang12 CIB, class_sz
for B16 and B12, inlined transcriptions of class_sz's C source for the Maniyar+2021 CIB Mdot/SFR, and, for the generic
Hankel-transform path (GNFW, B12, B16), the analytic truncated-NFW transform and quad integrals of the truncated GNFW.

CCL comparisons run hmfast with ncdm_mode="m" against a CCL CosmologyCalculator with the same neutrino masses and fed
hmfast's own P_lin, so residuals measure the profile formulas. Each profile is checked at three (M, z) points spanning
group to cluster masses. Conventions bridged here: hmfast's HOD real()/fourier() are ng_bar-normalized, so CCL's raw
HOD output is divided by ng_bar; hmfast's NFW returns M u / rho_m, so it is multiplied by rho_m = Rho_crit_0 * Omega0_m;
CCL's GNFW real() does not apply x_out, so it is masked beyond x_out r_delta.

CCL tests are marked `ccl` and class_sz tests `class_sz`; each is skipped if its package is missing.

Thresholds are ~2x the error measured when written (noted per test), so a ~2x degradation fails; lower them when
accuracy improves. Errors are maximum relative errors unless stated otherwise.
"""

import functools

import jax
import jax.numpy as jnp
import mcfit
import numpy as np
import pytest
from scipy.integrate import quad

from hmfast.halos.concentration import ConstantConcentration, D08Concentration
from hmfast.halos.massdef import MassDefinition
from hmfast.halos.profiles import (
    B12PressureProfile,
    B16DensityProfile,
    GNFWPressureProfile,
    M21CIBProfile,
    NFWMatterProfile,
    S12CIBProfile,
    Z07GalaxyHODProfile,
)
from hmfast.halos.profiles.base_profile import HaloProfile, HankelTransform
from hmfast.halos.profiles.profiles_2pt import _fourier_2pt

from .._shared import ccl_cosmology, peak_err, rel_err, shared
from .._shared import halo_model as shared_halo_model

try:
    import pyccl
except ImportError:
    pyccl = None

REQUIRES_CCL = [pytest.mark.ccl, pytest.mark.skipif(pyccl is None, reason="requires pyccl")]

M_GRID = jnp.geomspace(1e10, 1e15, 40)
R_GRID = np.geomspace(0.01, 2.0, 20)
K_GRID = np.geomspace(0.01, 5.0, 20)


def to_ccl_massdef(mass_def):
    reference = "matter" if mass_def.reference == "mean" else mass_def.reference
    return pyccl.halos.MassDef(mass_def.delta, reference)


def halo_model(cosmology, mass_def, concentration=None, **kwargs):
    """HaloModel with shared components (so jitted kernels are reused across tests), D08 concentration by default."""
    kwargs.setdefault("m_range", (M_GRID[0], M_GRID[-1]))
    kwargs.setdefault("n_m", M_GRID.shape[0])
    return shared_halo_model(cosmology=cosmology, mass_def=mass_def,
                             concentration=concentration or shared(D08Concentration), **kwargs)


@pytest.fixture(scope="module")
def cosmo(fixed_cosmology_m):
    return fixed_cosmology_m


@pytest.fixture(scope="module")
def cosmo_ccl(fixed_cosmology_m):
    return ccl_cosmology(fixed_cosmology_m, pk_linear=True)


# ---------------------------------------------------------------------------------------------------------------------
# Profiles against CCL.
# ---------------------------------------------------------------------------------------------------------------------

D08_MASS_DEFS = [(200, "critical"), (200, "mean"), ("vir", "critical")]


class TestNFWMatterProfileCCL:
    pytestmark = REQUIRES_CCL

    # real() and fourier() against CCL's truncated, analytic-Fourier HaloProfileNFW for each D08 mass definition
    # (measured 9.5e-5 / 7.9e-5).
    @pytest.mark.parametrize("m,z", [(1e12, 0.0), (1e14, 0.5), (1e15, 1.0)])
    @pytest.mark.parametrize("delta,reference", D08_MASS_DEFS)
    def test_real_and_fourier(self, cosmo, cosmo_ccl, delta, reference, m, z):
        md = shared(MassDefinition, delta, reference)
        hm = halo_model(cosmo, md)
        md_ccl = to_ccl_massdef(md)
        nfw_ccl = pyccl.halos.HaloProfileNFW(mass_def=md_ccl, concentration=pyccl.halos.ConcentrationDuffy08(
            mass_def=md_ccl), truncated=True, fourier_analytic=True)
        a = 1.0 / (1.0 + z)
        rho_m = cosmo.rho_x(0.0, "matter")
        nfw = NFWMatterProfile()

        real_ccl = nfw_ccl.real(cosmo_ccl, R_GRID, m, a)
        assert rel_err(np.asarray(nfw.real(hm, jnp.asarray(R_GRID), m, z)) * rho_m, real_ccl) < 2e-4
        four_ccl = nfw_ccl.fourier(cosmo_ccl, K_GRID, m, a)
        assert rel_err(np.asarray(nfw.fourier(hm, jnp.asarray(K_GRID), m, z)) * rho_m, four_ccl) < 1.6e-4


GNFW_A10 = dict(P0=8.130, c500=1.156, alpha=1.062, beta=5.4807, gamma=0.3292, alpha_P=0.12, P0_hexp=-1.0)


def _gnfw_pair(md, B=1.4, x_out=4.0, nq=256):
    gnfw = GNFWPressureProfile(B=B, x_out=x_out, **GNFW_A10)
    gnfw_ccl = pyccl.halos.HaloProfilePressureGNFW(mass_def=to_ccl_massdef(md), mass_bias=1.0 / B, qrange=(1e-5, 1e5),
                                                   nq=nq, x_out=x_out, **GNFW_A10)
    return gnfw, gnfw_ccl


class TestGNFWPressureProfileCCL:
    pytestmark = REQUIRES_CCL

    # real() at 500c, the definition GNFW is calibrated for, and at 200c after conversion (measured 2.3e-4).
    @pytest.mark.parametrize("m,z", [(1e13, 0.0), (3e14, 0.3), (1e15, 1.0)])
    @pytest.mark.parametrize("delta", [500, 200])
    def test_real(self, cosmo, cosmo_ccl, delta, m, z):
        md = shared(MassDefinition, delta, "critical")
        hm = halo_model(cosmo, md, shared(ConstantConcentration, c=5))
        gnfw, gnfw_ccl = _gnfw_pair(md)
        a = 1.0 / (1.0 + z)
        r_delta = to_ccl_massdef(md).get_radius(cosmo_ccl, m / 1.4, a) / a
        inside = R_GRID <= 4.0 * r_delta
        real_ccl = gnfw_ccl.real(cosmo_ccl, R_GRID, m, a)
        assert rel_err(np.asarray(gnfw.real(hm, jnp.asarray(R_GRID), m, z))[inside], real_ccl[inside]) < 5e-4

    # fourier() at 500c against CCL's sine-transform table (nq = 256). Error is max |difference| over the k -> 0 value,
    # since the transform crosses zero (measured 4.0e-4).
    @pytest.mark.parametrize("m,z", [(1e13, 0.0), (3e14, 0.3), (1e15, 1.0)])
    def test_fourier(self, cosmo, cosmo_ccl, m, z):
        md = shared(MassDefinition, 500, "critical")
        hm = halo_model(cosmo, md, shared(ConstantConcentration, c=5))
        gnfw, gnfw_ccl = _gnfw_pair(md)
        k = np.concatenate([[1e-4], K_GRID])
        want = gnfw_ccl.fourier(cosmo_ccl, k, m, 1.0 / (1.0 + z))
        assert peak_err(np.asarray(gnfw.fourier(hm, jnp.asarray(k), m, z)), want) < 8e-4


HOD_PARAMS = dict(sigma_log10M=0.68, alpha_s=1.30, M1_prime=10**12.87, M_min=10**11.97, M0=0.0)


def _hod_pair(md):
    md_ccl = to_ccl_massdef(md)
    hod = Z07GalaxyHODProfile(**HOD_PARAMS)
    hod_ccl = pyccl.halos.HaloProfileHOD(mass_def=md_ccl, concentration=pyccl.halos.ConcentrationDuffy08(
        mass_def=md_ccl), log10Mmin_0=11.97, siglnM_0=0.68 * np.log(10), log10M1_0=12.87, alpha_0=1.30, log10M0_0=0.0)
    return hod, hod_ccl


class TestHODProfileCCL:
    pytestmark = REQUIRES_CCL

    # real() and fourier() against CCL's HaloProfileHOD divided by ng_bar (measured 8.9e-5 / 4.5e-5).
    @pytest.mark.parametrize("m,z", [(1e12, 0.0), (5e12, 0.5), (1e14, 1.0)])
    def test_real_and_fourier(self, cosmo, cosmo_ccl, m, z):
        md = shared(MassDefinition, 200, "critical")
        hm = halo_model(cosmo, md)
        hod, hod_ccl = _hod_pair(md)
        a = 1.0 / (1.0 + z)
        ngbar = float(hod.ng_bar(hm, z))
        real_ccl = hod_ccl.real(cosmo_ccl, R_GRID, m, a) / ngbar
        assert rel_err(hod.real(hm, jnp.asarray(R_GRID), m, z), real_ccl) < 2e-4
        four_ccl = hod_ccl.fourier(cosmo_ccl, K_GRID, m, a) / ngbar
        assert rel_err(hod.fourier(hm, jnp.asarray(K_GRID), m, z), four_ccl) < 1e-4

    # The 1-halo Fourier 2-point term against CCL's Profile2ptHOD (measured 1.0e-4 at 1e14). hmfast's sat-sat term is
    # n_sat^2 u^2 = N_c^2 N_s^2 u^2; Poisson satellites that require a central give <N_s(N_s-1)> = N_c N_s^2
    # (1.7% / 1.5% at 1e12 / 5e12, where N_c < 1).
    @pytest.mark.parametrize("m,z", [
        pytest.param(1e12, 0.0, marks=pytest.mark.xfail(strict=True, reason="sat-sat term carries N_c^2, not N_c")),
        pytest.param(5e12, 0.5, marks=pytest.mark.xfail(strict=True, reason="sat-sat term carries N_c^2, not N_c")),
        (1e14, 1.0),
    ])
    def test_fourier_2pt(self, cosmo, cosmo_ccl, m, z):
        md = shared(MassDefinition, 200, "critical")
        hm = halo_model(cosmo, md)
        hod, hod_ccl = _hod_pair(md)
        ngbar = float(hod.ng_bar(hm, z))
        want = pyccl.halos.Profile2ptHOD().fourier_2pt(cosmo_ccl, K_GRID, m, 1.0 / (1.0 + z), hod_ccl, prof2=hod_ccl)
        got = np.asarray(_fourier_2pt(hm, hod, hod, jnp.asarray(K_GRID), m, z)) * ngbar**2
        assert rel_err(got, want) < 2e-4

    # ng_bar(z) and galaxy_bias(z) against CCL's HMCalculator mass integrals. Both run on fine grids from 1e8, where
    # N_gal vanishes, so neither the quadrature nor CCL's low-mass mass-function correction matters
    # (measured 1.3e-4 / 2.2e-5).
    @pytest.mark.parametrize("z", [0.0, 1.0, 2.0, 3.0])
    def test_ng_bar_and_galaxy_bias(self, cosmo, cosmo_ccl, z):
        md = shared(MassDefinition, 200, "critical")
        hm = halo_model(cosmo, md, m_range=(1e8, 1e16), n_m=256, hm_consistency=False)
        hod, hod_ccl = _hod_pair(md)
        md_ccl = to_ccl_massdef(md)
        hmc = pyccl.halos.HMCalculator(mass_function=pyccl.halos.MassFuncTinker08(mass_def=md_ccl),
                                       halo_bias=pyccl.halos.HaloBiasTinker10(mass_def=md_ccl), mass_def=md_ccl,
                                       log10M_min=8.0, log10M_max=16.0, nM=1024)
        a = 1.0 / (1.0 + z)
        ng_ccl = hod_ccl.get_normalization(cosmo_ccl, a, hmc=hmc)
        bg_ccl = hmc.I_1_1(cosmo_ccl, 1e-5, a, hod_ccl) / ng_ccl
        assert rel_err(hod.ng_bar(hm, z), ng_ccl) < 2.6e-4
        assert rel_err(hod.galaxy_bias(hm, z), bg_ccl) < 5e-5


class TestS12CIBProfileCCL:
    pytestmark = REQUIRES_CCL

    # real() and fourier() against CCL's HaloProfileCIBShang12 (measured 1.6e-4 / 6.3e-5). CCL integrates the satellite
    # luminosity with Simpson on 10 nodes per whole decade above M_min, which is under-resolved below ~1e13.
    @pytest.mark.parametrize("m,z", [(1e13, 0.0), (5e13, 0.5), (1e15, 1.0)])
    def test_real_and_fourier(self, cosmo, cosmo_ccl, m, z):
        md = shared(MassDefinition, 200, "critical")
        hm = halo_model(cosmo, md)
        md_ccl = to_ccl_massdef(md)
        cib = S12CIBProfile(nu=100, L0=6.4e-8, alpha=0.36, beta=1.75, gamma=1.7, T0=24.4, M_eff=10**12.6,
                            sigma2_LM=0.707**2, delta=3.6, z_p=1e100, M_min=10**11.5)
        cib_ccl = pyccl.halos.HaloProfileCIBShang12(
            nu_GHz=100, mass_def=md_ccl, concentration=pyccl.halos.ConcentrationDuffy08(mass_def=md_ccl), alpha=0.36,
            T0=24.4, beta=1.75, gamma=1.7, s_z=3.6, log10Meff=12.6, siglog10M=0.707, Mmin=10**11.5, L0=6.4e-8)
        a = 1.0 / (1.0 + z)
        assert rel_err(cib.real(hm, jnp.asarray(R_GRID), m, z), cib_ccl.real(cosmo_ccl, R_GRID, m, a)) < 3.2e-4
        assert rel_err(cib.fourier(hm, jnp.asarray(K_GRID), m, z), cib_ccl.fourier(cosmo_ccl, K_GRID, m, a)) < 1.3e-4


# ---------------------------------------------------------------------------------------------------------------------
# Profiles against class_sz.
# ---------------------------------------------------------------------------------------------------------------------


class TestB16DensityProfileClassSZ:
    # real() against class_sz's get_gas_profile_at_x_M_z_b16_200c, whose output carries an extra 1/h^2
    # (rtol 1%, measured ~0.26%).
    @pytest.mark.class_sz
    def test_real(self, cosmo):
        classy_sz = pytest.importorskip("classy_sz")
        hm = halo_model(cosmo, shared(MassDefinition, 200, "critical"))
        shape = dict(A_rho0=4000.0, A_alpha=0.88, A_beta=3.83, alpha_m_rho0=0.29, alpha_m_alpha=-0.03,
                     alpha_m_beta=0.04, alpha_z_rho0=-0.66, alpha_z_alpha=0.19, alpha_z_beta=-0.025)
        b16 = B16DensityProfile(**shape)
        m, z = 1e14, 0.3
        csz = classy_sz.Class()
        csz.set({"output": "gas_pressure_profile_2h,gas_density_profile_2h"})
        csz.compute_class_szfast()
        csz_kwargs = dict(shape, alphap_m_rho0=0.29, alphap_m_alpha=-0.03, alphap_m_beta=0.04)
        want = np.array([csz.get_gas_profile_at_x_M_z_b16_200c(x=float(r), M=m, z=z, **csz_kwargs) for r in R_GRID])
        assert rel_err(b16.real(hm, jnp.asarray(R_GRID), m, z), want * (cosmo.H0 / 100.0) ** 2) < 0.01


class TestB12PressureProfileClassSZ:
    # real() against class_sz's B12 shape function, dimensionalized with hmfast's P_200c (rtol 2%, measured ~0.9%).
    @pytest.mark.class_sz
    def test_real(self, cosmo):
        classy_sz = pytest.importorskip("classy_sz")
        md = shared(MassDefinition, 200, "critical")
        hm = halo_model(cosmo, md)
        shape = dict(A_P0=18.1, A_xc=0.497, A_beta=4.35, alpha_m_P0=0.154, alpha_m_xc=-0.00865, alpha_m_beta=0.0393,
                     alpha_z_P0=-0.758, alpha_z_xc=0.731, alpha_z_beta=0.415)
        b12 = B12PressureProfile(**shape)
        m, z = 1e14, 0.3
        h = cosmo.H0 / 100.0
        f_b = cosmo.omega_b / h**2 / cosmo.omega_x(0.0, "matter")
        r200c = float(md.r_delta(cosmo, m, z))
        p_200c = (m / (r200c * h)) * f_b * 2.61051e-18 * float(cosmo.hubble_parameter(z)) ** 2
        csz = classy_sz.Class()
        csz.set({"output": "gas_pressure_profile_2h"})
        csz.compute_class_szfast()
        csz_kwargs = dict(shape, alphap_m_P0=0.154, alphap_m_xc=-0.00865, alphap_m_beta=0.0393)
        want = np.array([csz.get_pressure_P_over_P_delta_at_x_M_z_b12_200c(x=float(r), M=m, z=z, **csz_kwargs)
                         for r in R_GRID])
        assert rel_err(b12.real(hm, jnp.asarray(R_GRID), m, z), want * p_200c) < 0.02


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
    # Against the inlined class_sz transcriptions above, which use the analytic flat-LCDM E(z) where hmfast uses the
    # emulated H(z)/H0, the source of the residual (measured 3.6e-4 / 3.6e-4).
    def test_mdot_and_sfr(self, cosmo):
        hm = halo_model(cosmo, shared(MassDefinition, 200, "critical"))
        m21 = M21CIBProfile(nu=100)
        m = np.geomspace(1e11, 1e15, 6)
        z = np.array([0.0, 0.5, 1.0, 2.0])
        Om0 = float(cosmo.omega_x(0.0, "matter"))
        f_b = float(cosmo.omega_b / (cosmo.H0 / 100.0) ** 2 / Om0)
        assert rel_err(m21._m_dot(hm, jnp.asarray(m), jnp.asarray(z)), _ref_mdot(m, z, Om0, 1.0 - Om0)) < 7e-4
        want = _ref_sfr(m, z, Om0, 1.0 - Om0, f_b, m21.eta_max, m21.M_eff, m21.sigma2_LM, m21.tau, m21.z_c)
        assert rel_err(m21._sfr(hm, jnp.asarray(m), jnp.asarray(z)), want) < 7e-4


# ---------------------------------------------------------------------------------------------------------------------
# Numerical accuracy of the generic Hankel path, HaloProfile._fourier_via_hankel_transform (GNFW, B12, B16), against
# the analytic truncated-NFW transform or a quad integral of the truncated GNFW. Errors are the maximum absolute error
# over the targets divided by u(q -> 0).
# ---------------------------------------------------------------------------------------------------------------------

# Off-grid targets in the dimensionless wavenumber q = k r_scale (1+z).
Q = np.geomspace(1e-2, 20.0, 97)
Q_LOW = np.geomspace(1e-4, 1e-2, 9)
GNFW_SHAPE = dict(c500=1.81, alpha=1.062, beta=4.13, gamma=0.3292)
X_MIN, X_MAX = 1e-5, 4.0


@pytest.fixture(scope="module")
def hm200c(cosmo):
    """A HaloModel at 200c/D08, the native mass definition of the NFW accuracy checks."""
    return halo_model(cosmo, shared(MassDefinition, 200, "critical"))


@pytest.fixture(scope="module")
def hm500c(cosmo):
    """A HaloModel at 500c/D08, the native mass definition of the GNFW accuracy checks."""
    return shared_halo_model(cosmology=cosmo, mass_def=shared(MassDefinition, 500, "critical"))


def _gnfw_shape(x):
    c, a, b, g = GNFW_SHAPE["c500"], GNFW_SHAPE["alpha"], GNFW_SHAPE["beta"], GNFW_SHAPE["gamma"]
    return (c * x) ** -g * (1.0 + (c * x) ** a) ** ((g - b) / a)


def _gnfw_truth_at(q, x_out):
    """int_0^x_out x^2 f(x) j0(q x) dx, to ~1e-12."""
    if q < 1e-8:
        return quad(lambda x: x * x * _gnfw_shape(x), 1e-14, x_out, limit=500, epsabs=0, epsrel=1e-13)[0]
    return quad(lambda x: x * _gnfw_shape(x) / q, 1e-14, x_out, weight="sin", wvar=q,
                limit=2000, epsabs=0, epsrel=1e-13)[0]


@functools.lru_cache(maxsize=None)
def _gnfw_truth(x_out, which="Q"):
    qs = Q if which == "Q" else Q_LOW
    return _gnfw_truth_at(0.0, x_out), np.array([_gnfw_truth_at(q, x_out) for q in qs])


def _gnfw(n_x, x_out, B=1.0):
    return GNFWPressureProfile(x_range=(X_MIN, X_MAX), n_x=n_x, B=B, x_out=x_out, **GNFW_SHAPE)


def _gnfw_u(hm, n_x, x_out, q, m=1e14, z=0.0):
    """hmfast's GNFW transform at q, divided by its constant prefactor so it compares to the truth."""
    p = _gnfw(n_x, x_out)
    m1, z1 = jnp.atleast_1d(m), jnp.atleast_1d(z)
    rs = float(jnp.squeeze(p._fourier_radius_scale(hm, m1, z1)))
    amp = float(jnp.squeeze(p.real(hm, jnp.array([0.5 * rs * (1 + z)]), m, z))) / _gnfw_shape(0.5)
    u = np.asarray(p.fourier(hm, jnp.asarray(q / (rs * (1 + z))), m, z))
    return u / (4 * np.pi * (rs * (1 + z)) ** 3 * amp)


def gnfw_error(hm, n_x, x_out, which="Q"):
    u0, truth = _gnfw_truth(x_out, which)
    u = _gnfw_u(hm, n_x, x_out, Q if which == "Q" else Q_LOW)
    return np.max(np.abs(u - truth)) / u0


def mcfit_ideal_error(n_x, pad_decades=2.0):
    """Best achievable with mcfit on the same padded grid: x_out on the last node at half weight, read at mcfit's own nodes."""
    x = np.logspace(np.log10(X_MIN), np.log10(X_MAX), n_x)
    h = np.log(x[1] / x[0])
    n_pad = int(np.ceil(pad_decades * np.log(10) / h))
    F = _gnfw_shape(x) * x**0.5
    F[-1] *= 0.5
    q, G = mcfit.Hankel(x[0] * np.exp(h * np.arange(n_x + n_pad)), nu=0.5, lowring=True, backend="jax")(
        jnp.asarray(np.concatenate([F, np.zeros(n_pad)])), extrap=False)
    q, G = np.asarray(q), np.asarray(G)
    sel = (q > Q[0]) & (q < Q[-1])
    u = (G * np.sqrt(np.pi / (2 * q)))[sel]
    truth = np.array([_gnfw_truth_at(v, X_MAX) for v in q[sel]])
    return np.max(np.abs(u - truth)) / _gnfw_truth(X_MAX)[0]


def nfw_error(hm, n_x, x_max, m, z):
    """Truncated NFW pushed through the generic Hankel path, against its analytic transform."""
    nfw = NFWMatterProfile()
    x = jnp.logspace(np.log10(X_MIN), np.log10(x_max), n_x)
    nfw._hankel, nfw.x_grid, nfw.x_out = HankelTransform(x, nu=0.5), x, 1.0  # x = r / r_delta, truncated at r_delta
    m1, z1 = jnp.atleast_1d(m), jnp.atleast_1d(z)
    r_delta = jnp.reshape(hm.mass_def.r_delta(hm.cosmology, m1, z1), (1, 1))
    k = jnp.asarray(Q) / (r_delta[0, 0] * (1 + z))
    u_hankel = np.asarray(HaloProfile._fourier_via_hankel_transform(nfw, hm, k, m1, z1, r_delta))
    u_exact = np.asarray(nfw.fourier(hm, k, m, z))
    return np.max(np.abs(u_hankel - u_exact)) / np.max(np.abs(u_exact))


NFW_HALOS = [(1e12, 0.0), (1e14, 0.5), (1e15, 1.0)]


class TestHankelAgainstAnalyticNFW:
    # x_out = 1 on the last grid node (measured max over halos: 1.5e-3 at n_x=100).
    @pytest.mark.parametrize("m,z", NFW_HALOS)
    def test_truncation_on_grid_edge(self, hm200c, m, z):
        assert nfw_error(hm200c, 100, 1.0, m, z) < 3e-3

    # x_out = 1 between nodes of a grid running to 1.5 (measured max over halos: 3.2e-3 at n_x=100).
    @pytest.mark.parametrize("m,z", NFW_HALOS)
    def test_truncation_between_nodes(self, hm200c, m, z):
        assert nfw_error(hm200c, 100, 1.5, m, z) < 6e-3

    # Second-order convergence: doubling n_x cuts the error by >= 3 (measured 4.7).
    def test_convergence_rate(self, hm200c):
        e = [nfw_error(hm200c, n, 1.0, 1e14, 0.5) for n in (100, 200, 400)]
        assert e[0] / e[1] > 3 and e[1] / e[2] > 3


class TestHankelAgainstQuadGNFW:
    # x_out on the grid edge, the GNFW default (measured 5.2e-4 / 1.3e-4 at n_x=100 / 200).
    @pytest.mark.parametrize("n_x,tol", [(100, 1e-3), (200, 2.6e-4)])
    def test_truncation_on_grid_edge(self, hm500c, n_x, tol):
        assert gnfw_error(hm500c, n_x, X_MAX) < tol

    # x_out between nodes (measured 1.7e-3 / 1.3e-4 at n_x=100 / 200).
    @pytest.mark.parametrize("n_x,tol", [(100, 3.4e-3), (200, 2.7e-4)])
    def test_truncation_between_nodes(self, hm500c, n_x, tol):
        assert gnfw_error(hm500c, n_x, 3.0) < tol

    # x_out just inside the last node, where it is not snapped onto it (measured 2.4e-3 at n_x=100).
    def test_truncation_just_inside_node(self, hm500c):
        assert gnfw_error(hm500c, 100, X_MAX * (1 - 1e-3)) < 5e-3

    # Second-order convergence: doubling n_x cuts the error by >= 3 (measured 4.1).
    def test_convergence_rate(self, hm500c):
        e = [gnfw_error(hm500c, n, X_MAX) for n in (100, 200, 400)]
        assert e[0] / e[1] > 3 and e[1] / e[2] > 3

    # Below mcfit's native q range, where the zero padding and the q -> 0 anchor take over (measured 3.8e-4).
    def test_low_q(self, hm500c):
        assert gnfw_error(hm500c, 100, X_MAX, which="low") < 8e-4


class TestHankelMatchesMcfit:
    # hmfast adds no significant error on top of mcfit's own (measured ratio 1.09-1.11).
    @pytest.mark.parametrize("n_x", [100, 200, 400])
    def test_no_loss_relative_to_mcfit(self, hm500c, n_x):
        assert gnfw_error(hm500c, n_x, X_MAX) / mcfit_ideal_error(n_x) < 1.25


class TestHankelEdgeCases:
    # x_out = inf truncates at the grid edge, so it must equal x_out = x_max (round-off in real() must not drop the node).
    def test_untruncated_equals_truncated_at_grid_edge(self, hm500c):
        u_inf = _gnfw_u(hm500c, 100, jnp.inf, Q)
        u_edge = _gnfw_u(hm500c, 100, X_MAX, Q)
        assert np.allclose(u_inf, u_edge, rtol=1e-9, atol=0)

    # x_out a few ulps below the last node snaps onto it; real() must still see that node as inside (1.8e-2 if it doesn't).
    def test_node_kept_when_x_out_rounds_below_it(self, hm500c):
        u_below = _gnfw_u(hm500c, 100, X_MAX * (1 - 1e-14), Q)
        u_edge = _gnfw_u(hm500c, 100, X_MAX, Q)
        assert np.allclose(u_below, u_edge, rtol=1e-9, atol=0)

    # Far beyond the native grid the transform clamps to its last value rather than extrapolating.
    def test_clamps_above_native_grid(self, hm500c):
        u = _gnfw_u(hm500c, 100, X_MAX, np.array([1e7, 1e9]))
        assert np.all(np.isfinite(u)) and u[0] == u[1]


class TestHankelSmoothness:
    # d ln u / d ln B varies smoothly along a dense sweep in B, so the interpolation onto q = k r_500 has no kinks
    # (measured max second difference 1.3e-4; linear interpolation gives 2.2e-2).
    def test_autodiff_derivative_has_no_kinks_in_B(self, hm500c):
        k = jnp.array([0.3, 1.0, 3.0])
        p = _gnfw(100, X_MAX)
        dlnu_dlnB = jax.vmap(jax.jacfwd(lambda lnB: jnp.log(p.update(B=jnp.exp(lnB)).fourier(hm500c, k, 1e14, 0.0))))
        d = np.asarray(dlnu_dlnB(jnp.linspace(0.0, np.log(1.5), 201)))
        assert np.max(np.abs(d[2:] - 2 * d[1:-1] + d[:-2])) < 2.5e-4
