"""
CCL benchmarks for hmfast.halos (mass definitions and conversions, concentration, halo mass function, halo bias).

Every comparison runs hmfast with ncdm_mode="m", the neutrino convention CCL uses, against a CCL CosmologyCalculator
with the same neutrino masses and fed hmfast's own P_lin, so residuals measure the halo-model formulas rather than
P(k). Each test runs at three cosmologies: lcdm:v1 (one 0.06 eV state), mnu:v1 with 0.3 eV, where the neutrino
fraction is large enough to expose any Omega_cb/Omega_m mix-up, and mnu-3states:v1 (three 0.1 eV states).

T10 and B13 are compared with CCL's delta_c set to the constant EdS value both papers calibrate with, where CCL itself
uses the EdS_approx (Tinker10) and NakamuraSuto97 (Bhattacharya13) fits; with CCL's choice they differ by up to 0.7%.

Requires pyccl; marked `ccl` and skipped if it is missing.

Thresholds are ~2x the error measured when written (noted per test, maximum over the three cosmologies), so a ~2x
degradation fails; lower them when accuracy improves. Errors are maximum relative errors. The mass function and bias
are compared where the comoving halo density dn/dlnM exceeds 1e-12 Mpc^-3, beyond which no observable is sensitive to
them and the exponential tail amplifies any sigma(M) difference.
"""

import numpy as np
import pytest

from hmfast.cosmology import CosmoPowerCosmology
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import B13Concentration, D08Concentration
from hmfast.halos.massdef import MassDefinition, mass_translator
from hmfast.halos.massfunc import T08HaloMassFunction, T10HaloMassFunction

from .._shared import ccl_cosmology, rel_err, shared

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

import pyccl.halos.concentration.bhattacharya13 as ccl_b13  # noqa: E402
import pyccl.halos.hmfunc.tinker10 as ccl_t10  # noqa: E402

M_GRID = np.geomspace(1e10, 1e16, 40)
Z_LIST = [0.0, 0.5, 1.0, 2.0, 3.0]
# CCL's Tinker mass function and bias raise for the virial definition at z >~ 1.06 (Delta_m below their 200 floor).
Z_LIST_VIR_HMF = [0.0, 0.5, 1.0]
N_MIN = 1e-12

COSMOLOGIES = [
    ("lcdm:v1", {}),
    ("mnu:v1", {"m_ncdm": 0.3}),
    ("mnu-3states:v1", {"m_ncdm": 0.1}),
]


@pytest.fixture(scope="module", params=COSMOLOGIES, ids=[s for s, _ in COSMOLOGIES])
def cosmologies(request, fixed_cosmology):
    """(hmfast cosmology in ncdm_mode="m", matching CCL calculator) for each benchmark cosmology."""
    emulator_set, extension = request.param
    f = fixed_cosmology
    try:
        cosmo = CosmoPowerCosmology(emulator_set, H0=f.H0, omega_cdm=f.omega_cdm, omega_b=f.omega_b, A_s=f.A_s,
                          n_s=f.n_s, ncdm_mode="m", **extension)
    except Exception as exc:
        pytest.skip(f"{emulator_set} emulator files not available locally: {exc}")
    return cosmo, ccl_cosmology(cosmo, pk_linear=True)


@pytest.fixture
def eds_delta_c(monkeypatch):
    """Make CCL's Tinker10 mass function and Bhattacharya13 concentration use the constant EdS delta_c, like hmfast."""
    def eds(cosmo, a, kind="EdS"):
        return pyccl.halos.get_delta_c(cosmo, a, kind="EdS")

    monkeypatch.setattr(ccl_t10, "get_delta_c", eds)
    monkeypatch.setattr(ccl_b13, "get_delta_c", eds)


def to_ccl_massdef(mass_def):
    reference = "matter" if mass_def.reference == "mean" else mass_def.reference
    return pyccl.halos.MassDef(mass_def.delta, reference)


def z_list_hmf(mass_def):
    return Z_LIST_VIR_HMF if mass_def.delta == "vir" else Z_LIST


class TestMassDefinitionCCL:
    # r_delta for all 4 mass definitions (measured 4.9e-5).
    def test_r_delta(self, cosmologies, mass_def):
        cosmo, ccl = cosmologies
        md_ccl = to_ccl_massdef(mass_def)
        for z in Z_LIST:
            got = mass_def.r_delta(cosmo, M_GRID, z)
            assert rel_err(np.ravel(got), md_ccl.get_radius(ccl, M_GRID, 1.0 / (1.0 + z))) < 1e-4


# All (in, out) pairs among {200c, 200m, vir} x {200c, 200m, 500c, vir} with in != out (D08 has no 500c input).
MASS_CONVERSION_PAIRS = [
    (in_def, out_def)
    for in_def in [(200, "critical"), (200, "mean"), ("vir", "critical")]
    for out_def in [(200, "critical"), (200, "mean"), (500, "critical"), ("vir", "critical")]
    if in_def != out_def
]


class TestMassConversionCCL:
    # mass_translator with D08 concentration for every non-self pair (measured 1.2e-4).
    @pytest.mark.parametrize("in_def,out_def", MASS_CONVERSION_PAIRS)
    def test_translator(self, cosmologies, in_def, out_def):
        cosmo, ccl = cosmologies
        md_in, md_out = shared(MassDefinition, *in_def), shared(MassDefinition, *out_def)
        md_in_ccl, md_out_ccl = to_ccl_massdef(md_in), to_ccl_massdef(md_out)
        f_hmf = mass_translator(md_in, md_out, shared(D08Concentration))
        f_ccl = pyccl.halos.mass_translator(mass_in=md_in_ccl, mass_out=md_out_ccl,
                                            concentration=pyccl.halos.ConcentrationDuffy08(mass_def=md_in_ccl))
        for z in Z_LIST:
            assert rel_err(np.ravel(f_hmf(cosmo, M_GRID, z)), f_ccl(ccl, M_GRID, 1.0 / (1.0 + z))) < 2.5e-4


class TestConcentrationCCL:
    # D08 and B13 at their shared mass definitions (measured 4e-16 / 2.9e-3). B13 inherits the scale-dependent
    # growth-factor difference with massive neutrinos through D(z)^B (see benchmark_cosmology.py); 1.1e-3 for lcdm:v1.
    @pytest.mark.parametrize("conc_cls,ccl_conc_cls,tol", [
        (D08Concentration, "ConcentrationDuffy08", 1e-12),
        (B13Concentration, "ConcentrationBhattacharya13", 6e-3),
    ])
    @pytest.mark.parametrize("delta,reference", [(200, "critical"), (200, "mean"), ("vir", "critical")])
    def test_concentration(self, cosmologies, eds_delta_c, conc_cls, ccl_conc_cls, tol, delta, reference):
        cosmo, ccl = cosmologies
        md = shared(MassDefinition, delta, reference)
        conc_ccl = getattr(pyccl.halos, ccl_conc_cls)(mass_def=to_ccl_massdef(md))
        for z in Z_LIST:
            got = shared(conc_cls).c_delta(cosmo, M_GRID, z, mass_def=md)
            assert rel_err(got, conc_ccl(ccl, M_GRID, 1.0 / (1.0 + z))) < tol


class TestHaloMassFunctionCCL:
    # T08 and T10 dn/dlnM for all 4 mass definitions (measured 2.6e-3 / 2.7e-3).
    @pytest.mark.parametrize("hmf_cls,ccl_hmf_cls,tol", [
        (T08HaloMassFunction, "MassFuncTinker08", 5.5e-3),
        (T10HaloMassFunction, "MassFuncTinker10", 5.5e-3),
    ])
    def test_dndlnm(self, cosmologies, eds_delta_c, mass_def, hmf_cls, ccl_hmf_cls, tol):
        cosmo, ccl = cosmologies
        hmf_ccl = getattr(pyccl.halos, ccl_hmf_cls)(mass_def=to_ccl_massdef(mass_def), mass_def_strict=True)
        for z in z_list_hmf(mass_def):
            want = hmf_ccl(ccl, M_GRID, 1.0 / (1.0 + z)) / np.log(10)
            got = np.asarray(shared(hmf_cls).dndlnm(cosmo, M_GRID, z, mass_def=mass_def))
            keep = want > N_MIN
            assert rel_err(got[keep], want[keep]) < tol


class TestHaloBiasCCL:
    # T10 linear bias for all 4 mass definitions (measured 1.8e-4).
    def test_bias(self, cosmologies, mass_def):
        cosmo, ccl = cosmologies
        md_ccl = to_ccl_massdef(mass_def)
        bias_ccl = pyccl.halos.HaloBiasTinker10(mass_def=md_ccl)
        hmf_ccl = pyccl.halos.MassFuncTinker08(mass_def=md_ccl, mass_def_strict=False)
        for z in z_list_hmf(mass_def):
            a = 1.0 / (1.0 + z)
            keep = hmf_ccl(ccl, M_GRID, a) / np.log(10) > N_MIN
            got = np.asarray(shared(T10HaloBias).bias(cosmo, M_GRID, z, mass_def=mass_def, order=1))
            assert rel_err(got[keep], bias_ccl(ccl, M_GRID, a)[keep]) < 4e-4
