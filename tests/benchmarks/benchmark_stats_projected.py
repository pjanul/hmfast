"""
Benchmarks for the projected statistics in hmfast.stats: the halo-model angular power spectrum `cl`, the linear-bias
`cl_linbias`, their non-Limber branch, and `corr_angular`.

The CCL reference is a CosmologyCalculator fed hmfast's own background (chi, H), growth (D, f) and P(k), with
ncdm_mode="m" and the same profiles, mass function, bias and mass range, so residuals measure the projection: the
kernels, the Limber or non-Limber line-of-sight integral, and the halo-model P(k) it projects (validated in
benchmark_stats_3d.py). Non-Limber spectra are also checked against SwiftCl, which runs the same FFTLog decomposition
through its own integrals on hmfast's kernels and D(k, z), and against CCL's FKEM non-Limber method.

Kernels that extend past z = 3 (CMB lensing) are cut there on both sides, where CCL's halo-model P(k) is tabulated. The
HOD second moment uses hmfast's N_c^2 N_s^2 satellite pairs on the CCL side too (see `ccl_profile2pt_hod`).

CCL tests are marked `ccl` and skipped without pyccl; SwiftCl tests are also skipped without swiftcl. SwiftCl caches
its FFTLog coefficient tables in $SWIFTCL_CACHE, or in a temporary directory if that is unset.

Thresholds are ~2x the error measured when written (noted per test), so a ~2x degradation fails; lower them when
accuracy improves. Errors are maximum relative errors unless stated otherwise.
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.halos.massdef import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile, S12CIBProfile, Z07GalaxyHODProfile
from hmfast.stats import Pk, cl, cl_linbias, corr_angular
from hmfast.stats.projected_cl import _D_kz
from hmfast.tracers import CIBTracer, CMBLensingTracer, GalaxyLensingTracer, GalaxyTracer, tSZTracer

from .._shared import ccl_profile2pt_hod, halo_model, rel_err, shared

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

M_MIN, M_MAX = 1e10, 1e16
Z_RANGE, N_Z = (1e-4, 3.0), 200
ELL = np.geomspace(10, 5000, 30)
PK_1H, PK_2H = Pk(include_2h=False, k_damp=0.0), Pk(include_1h=False)

NFW = NFWMatterProfile()
GNFW = GNFWPressureProfile(x_range=(1e-5, 1e5), n_x=128, P0=8.13, c500=1.156, alpha=1.062, beta=5.4807, gamma=0.3292,
                           B=1.4, alpha_P=0.12, P0_hexp=-1, x_out=4)
HOD = Z07GalaxyHODProfile(sigma_log10M=0.68, alpha_s=1.3, M1_prime=10**12.7, M_min=10**11.8, M0=0.0)
CIB = S12CIBProfile(nu=545, L0=6.4e-8, alpha=0.36, beta=1.75, gamma=1.7, T0=24.4, M_eff=10**12.6, sigma2_LM=0.5,
                    delta=3.6, z_p=1e100, M_min=1e10)


class Setup:
    """hmfast and CCL sides of every comparison, built once per module."""

    def __init__(self, cosmo_fixed, *, halo_pk2d=True):
        self.cosmo = cosmo_fixed.update(extrapolate_z=True)  # CMB lensing reaches z_star
        self.hm = halo_model(cosmology=self.cosmo, mass_def=shared(MassDefinition, 200, "critical"),
                             m_range=(M_MIN, M_MAX), n_m=100)
        self.z_star = float(self.cosmo.derived_parameters()["z_star"])
        self._background()
        self.ccl = self._ccl()
        # A second calculator for non-halo-model spectra: once halo-model Pk2Ds are built on a calculator, CCL's Limber
        # integral of a linear-P galaxy spectrum on it fails with CCL_ERROR_INTEG.
        self.ccl_lin = self._ccl()
        md = pyccl.halos.MassDef(200, "critical")
        conc = pyccl.halos.ConcentrationDuffy08(mass_def=md)
        self.hmc = pyccl.halos.HMCalculator(mass_function=pyccl.halos.MassFuncTinker08(mass_def=md),
                                            halo_bias=pyccl.halos.HaloBiasTinker10(mass_def=md), mass_def=md,
                                            log10M_min=np.log10(M_MIN), log10M_max=np.log10(M_MAX), nM=256)
        nfw = pyccl.halos.HaloProfileNFW(mass_def=md, concentration=conc, truncated=True, fourier_analytic=True)
        gnfw = pyccl.halos.HaloProfilePressureGNFW(mass_def=md, mass_bias=1 / 1.4, P0=8.13, c500=1.156, alpha=1.062,
                                                   alpha_P=0.12, beta=5.4807, gamma=0.3292, P0_hexp=-1,
                                                   qrange=(1e-5, 1e5), nq=1000, x_out=4)
        hod = pyccl.halos.HaloProfileHOD(mass_def=md, concentration=conc, log10Mmin_0=11.8, siglnM_0=0.68 * np.log(10),
                                         log10M1_0=12.7, alpha_0=1.3, log10M0_0=0.0)
        cib = pyccl.halos.HaloProfileCIBShang12(mass_def=md, concentration=conc, nu_GHz=545, alpha=0.36, T0=24.4,
                                                beta=1.75, gamma=1.7, s_z=3.6, log10Meff=12.6, siglog10M=np.sqrt(0.5),
                                                Mmin=1e10, L0=6.4e-8)
        self.nfw_ccl = nfw
        self.ccl_profiles = {"m": nfw, "y": gnfw, "g": hod, "c": cib}
        self.ccl_2pt = {"g": ccl_profile2pt_hod(), "c": pyccl.halos.Profile2ptCIB()}
        self.swiftcl_comps = {}  # SwiftCl's FFTLog coefficient tables, one per chi grid, reused across tests
        self.pk2d = {} if not halo_pk2d else {  # CCL halo-model P(k) tables for the C_ell references
            "m": self._halo_pk2d(nfw), "y": self._halo_pk2d(gnfw), "my": self._halo_pk2d(nfw, gnfw),
            "g": self._halo_pk2d(hod, prof_2pt=ccl_profile2pt_hod()), "gm": self._halo_pk2d(hod, nfw),
            "c": self._halo_pk2d(cib, prof_2pt=pyccl.halos.Profile2ptCIB()),
        }
        self.z_g, self.n_g = (np.asarray(x) for x in GalaxyTracer(profile=HOD).dndz)
        self.z_s, self.n_s = (np.asarray(x) for x in GalaxyLensingTracer(profile=NFW).dndz)

    def _background(self):
        z = np.concatenate([[0.0], np.geomspace(1e-5, 3.0, 600)[:-1], np.geomspace(3.0, 1200.0, 400)])
        self.a_bg = (1.0 / (1.0 + z))[::-1]
        zj = jnp.asarray(1.0 / self.a_bg - 1.0)
        self.chi_bg = np.asarray(self.cosmo.angular_diameter_distance(zj)) / self.a_bg
        self.hz_bg = np.asarray(self.cosmo.hubble_parameter(zj))
        self.D_bg, self.f_bg = np.asarray(self.cosmo.growth_factor(zj)), np.asarray(self.cosmo.growth_rate(zj))

    def _ccl(self):
        c = self.cosmo
        p = c._cosmo_params()
        a_tab = np.sort(1.0 / (1.0 + np.linspace(0.0, 3.0, 60)))
        k_tab = np.asarray(c._pk_grid()[0])
        z_tab = jnp.asarray(1.0 / a_tab - 1.0)
        pk = {lin: np.asarray(c.pk(jnp.asarray(k_tab), z_tab, linear=lin)).T for lin in (True, False)}
        return pyccl.CosmologyCalculator(
            Omega_c=float(p["Omega_cdm"]), Omega_b=float(p["Omega_b"]), h=float(p["h"]), A_s=c.A_s, n_s=c.n_s,
            m_nu=[0.0, 0.0, float(p["m_ncdm"])], mass_split="list",
            background={"a": self.a_bg, "chi": self.chi_bg, "h_over_h0": self.hz_bg / self.hz_bg[-1]},
            growth={"a": self.a_bg, "growth_factor": self.D_bg / self.D_bg[-1], "growth_rate": self.f_bg},
            pk_linear={"a": a_tab, "k": k_tab, "delta_matter:delta_matter": pk[True]},
            pk_nonlin={"a": a_tab, "k": k_tab, "delta_matter:delta_matter": pk[False]},
        )

    def _halo_pk2d(self, prof1, prof2=None, prof_2pt=None):
        kw = dict(prof2=prof2, prof_2pt=prof_2pt, lk_arr=np.log(np.geomspace(1e-5, 1e3, 1024)),
                  a_arr=np.linspace(0.25, 1.0, 40))
        return (pyccl.halos.halomod_Pk2D(self.ccl, self.hmc, prof1, get_2h=False, **kw),
                pyccl.halos.halomod_Pk2D(self.ccl, self.hmc, prof1, get_1h=False, **kw))

    # Tracers, each as an (hmfast, CCL) pair.
    def galaxy(self, *, bias=None, mag=False, rsd=False, density=True, lin=False):
        s = (self.z_g, 0.3 * np.ones_like(self.z_g))
        t = GalaxyTracer(profile=HOD, has_density=density, bias=bias, mag_bias=s if mag else None, has_rsd=rsd)
        ccl_bias = bias if bias is not None else (self.z_g, np.ones_like(self.z_g))
        tc = pyccl.NumberCountsTracer(self.ccl_lin if lin else self.ccl, dndz=(self.z_g, self.n_g), has_rsd=rsd,
                                      bias=ccl_bias if density else None, mag_bias=s if mag else None)
        return t, tc

    def shear(self, *, ia=False, shear=True, lin=False):
        A = (self.z_s, 0.5 * np.ones_like(self.z_s))
        t = GalaxyLensingTracer(profile=NFW, ia_bias=A if ia else None, has_shear=shear)
        tc = pyccl.WeakLensingTracer(self.ccl_lin if lin else self.ccl, dndz=(self.z_s, self.n_s), has_shear=shear,
                                     ia_bias=A if ia else None)
        return t, tc

    def cmb_lensing(self, z_max=3.0, lin=False):
        c = self.ccl_lin if lin else self.ccl
        tc = pyccl.CMBLensingTracer(c, z_source=self.z_star)
        if z_max is not None:
            z = np.geomspace(1e-5, z_max, 3000)
            chi = pyccl.comoving_radial_distance(c, 1.0 / (1.0 + z))
            cut = pyccl.Tracer()
            cut.add_tracer(c, kernel=(chi, tc.get_kernel(chi)[0]), der_bessel=-1, der_angles=1)
            tc = cut
        return CMBLensingTracer(profile=NFW), tc

    def tsz(self):
        return tSZTracer(profile=GNFW, z_max=3.0), pyccl.tSZTracer(self.ccl, z_max=3.0)

    def cib(self):
        return CIBTracer(profile=CIB, z_max=3.0), pyccl.CIBTracer(self.ccl, z_min=1e-4, z_max=3.0)

    # Spectra.
    def hm_cl(self, t1, t2=None, ell=ELL, **kw):
        kw = {"z_range": Z_RANGE, "n_z": N_Z, **kw}
        c1 = np.asarray(cl(PK_1H, self.hm, jnp.asarray(ell), t1, t2, **kw))
        c2 = np.asarray(cl(PK_2H, self.hm, jnp.asarray(ell), t1, t2, **kw))
        return {"1h": c1, "2h": c2, "1h+2h": c1 + c2}

    def ccl_cl(self, t1, t2, key, ell=ELL):
        c1 = pyccl.angular_cl(self.ccl, t1, t2, ell, p_of_k_a=self.pk2d[key][0])
        c2 = pyccl.angular_cl(self.ccl, t1, t2, ell, p_of_k_a=self.pk2d[key][1])
        return {"1h": c1, "2h": c2, "1h+2h": c1 + c2}


@pytest.fixture(scope="module")
def s(fixed_cosmology_m):
    return Setup(fixed_cosmology_m)


# ---------------------------------------------------------------------------------------------------------------------
# Halo-model C_ell, Limber.
# ---------------------------------------------------------------------------------------------------------------------

AUTOS = {
    "tSZ": ("tsz", "y", 2e-3, 3e-3),
    "CIB": ("cib", "c", 2.4e-3, 9e-4),
    "CMB lensing": ("cmb_lensing", "m", 6.5e-4, 2.5e-4),
    "shear": ("shear", "m", 6e-4, 1.5e-3),
    "galaxy": ("galaxy", "g", 1.2e-3, 3.5e-4),
}


class TestHaloModelClLimber:
    # Auto-spectra of each tracer's primary term, 1-halo and 2-halo separately, 10 <= l <= 5000
    # (measured 1h / 2h: tSZ 9.4e-4 / 1.4e-3, CIB 1.2e-3 / 4.2e-4, CMB lensing 3.1e-4 / 1.2e-4, shear 3.0e-4 / 7.3e-4,
    # galaxy 5.6e-4 / 1.7e-4).
    @pytest.mark.parametrize("name", AUTOS)
    def test_auto(self, s, name):
        make, key, tol_1h, tol_2h = AUTOS[name]
        t, tc = getattr(s, make)()
        got, want = s.hm_cl(t), s.ccl_cl(tc, tc, key)
        assert rel_err(got["1h"], want["1h"]) < tol_1h
        assert rel_err(got["2h"], want["2h"]) < tol_2h

    # Secondary kernel terms alone and combined, so the cross terms between them are checked through the totals
    # (measured 7.3e-4, 6.4e-3, 4.9e-4, 1.1e-3, 3.0e-4). IA follows the rough source dN/dz, which needs n_z = 800
    # (0.9% at the default 200). RSD alone is a near-cancellation of the two extended-Limber pieces at high l; it is
    # 0.65% from CCL there, unexplained (same formula, insensitive to n_z, dN/dz sampling and CCL's table density), but
    # <= 5e-4 combined with density.
    @pytest.mark.parametrize("name,make,kw,key,n_z,tol", [
        ("galaxy mag only", "galaxy", dict(density=False, mag=True), "g", N_Z, 1.5e-3),
        ("galaxy RSD only", "galaxy", dict(density=False, rsd=True), "g", N_Z, 1.3e-2),
        ("galaxy dens + mag + RSD", "galaxy", dict(mag=True, rsd=True), "g", N_Z, 1e-3),
        ("shear IA only", "shear", dict(ia=True, shear=False), "m", 800, 2.2e-3),
        ("shear + IA", "shear", dict(ia=True), "m", 800, 6e-4),
    ])
    def test_secondary_terms(self, s, name, make, kw, key, n_z, tol):
        t, tc = getattr(s, make)(**kw)
        assert rel_err(s.hm_cl(t, n_z=n_z)["1h+2h"], s.ccl_cl(tc, tc, key)["1h+2h"]) < tol

    # Crosses with a different tracer on each side (measured 2.7e-4, 7.0e-4, 2.7e-4).
    @pytest.mark.parametrize("make1,make2,key,tol", [
        ("galaxy", "shear", "gm", 5.5e-4),
        ("cmb_lensing", "tsz", "my", 1.4e-3),
        ("galaxy", "cmb_lensing", "gm", 5.5e-4),
    ])
    def test_cross(self, s, make1, make2, key, tol):
        (t1, tc1), (t2, tc2) = getattr(s, make1)(), getattr(s, make2)()
        assert rel_err(s.hm_cl(t1, t2)["1h+2h"], s.ccl_cl(tc1, tc2, key)["1h+2h"]) < tol


# ---------------------------------------------------------------------------------------------------------------------
# Linear-bias C_ell.
# ---------------------------------------------------------------------------------------------------------------------

LINBIAS = {
    "galaxy b(z)": ("galaxy", dict(bias=True)),
    "galaxy b(z) + mag + RSD": ("galaxy", dict(bias=True, mag=True, rsd=True)),
    "shear": ("shear", {}),
    "shear + IA": ("shear", dict(ia=True)),
    "CMB lensing": ("cmb_lensing", {}),
}


def linbias_pair(s, name, z_max=3.0):
    make, kw = LINBIAS[name]
    if kw.get("bias"):
        kw = {**kw, "bias": (s.z_g, 1.0 + s.z_g)}
    if make == "cmb_lensing":
        kw = {**kw, "z_max": z_max}
    return getattr(s, make)(lin=True, **kw)


class TestLinearBiasCl:
    # cl_linbias against CCL's angular_cl with the same linear or nonlinear matter P(k) (measured 9.8e-4 / 6.0e-4).
    @pytest.mark.parametrize("linear,tol", [(True, 2e-3), (False, 1.2e-3)])
    @pytest.mark.parametrize("name", LINBIAS)
    def test_cl_linbias(self, s, name, linear, tol):
        t, tc = linbias_pair(s, name)
        got = cl_linbias(s.cosmo, jnp.asarray(ELL), t, z_range=Z_RANGE, n_z=N_Z, linear=linear)
        p_of_k_a = s.ccl_lin.get_linear_power() if linear else s.ccl_lin.get_nonlin_power()
        assert rel_err(got, pyccl.angular_cl(s.ccl_lin, tc, tc, ELL, p_of_k_a=p_of_k_a)) < tol

    # On large scales the halo-model 2-halo term reduces to cl_linbias with the linear P(k): galaxies with the HOD's
    # effective bias I_1^1(k -> 0, z), matter tracers with b = 1, over 2 <= l <= 50 (measured 1.4e-4 / 2.2e-4).
    @pytest.mark.parametrize("make,tol", [("galaxy", 3e-4), ("shear", 4.5e-4)])
    def test_two_halo_large_scale_limit(self, s, make, tol):
        ell = np.geomspace(2, 50, 12)
        t = getattr(s, make)()[0]
        if make == "galaxy":
            b_eff = np.ravel(s.hm.mass_integral(jnp.array([1e-4]), jnp.asarray(s.z_g), HOD, bias_order=1))
            t_lin = t.update(bias=(jnp.asarray(s.z_g), jnp.asarray(b_eff)))
        else:
            t_lin = t
        got = cl(PK_2H, s.hm, jnp.asarray(ell), t, z_range=Z_RANGE, n_z=N_Z)
        assert rel_err(got, cl_linbias(s.cosmo, jnp.asarray(ell), t_lin, z_range=Z_RANGE, n_z=N_Z)) < tol


# ---------------------------------------------------------------------------------------------------------------------
# Non-Limber, against SwiftCl and CCL's FKEM.
# ---------------------------------------------------------------------------------------------------------------------

ELL_NL = np.arange(2, 201, dtype=float)
L_LIMBER = 1000.0
F_ELL = {0: lambda l: np.ones_like(l), 1: lambda l: l * (l + 1), 2: lambda l: np.sqrt((l + 2) * (l + 1) * l * (l - 1))}
BESSEL_ROW = {0: 0, -1: 1, 2: 2}  # SwiftCl's coefficient tables for j_l, j_l / x^2 and j_l''


@pytest.fixture(scope="module")
def swiftcl_cache(tmp_path_factory):
    return os.environ.get("SWIFTCL_CACHE") or str(tmp_path_factory.mktemp("swiftcl"))


def swiftcl_cl(s, cache, tracer, z_max, D_kz, bias=None, n_chi=512, n_k=400, tracer2=None, D_kz2=None, n_fft=512):
    """SwiftCl's FFTLog C_ell of `tracer` (x `tracer2` if given) with growth D(k, z), on hmfast's own background."""
    swiftcl = pytest.importorskip("swiftcl")
    z_of_a = (1.0 / s.a_bg - 1.0)[::-1]
    chi_of_z = s.chi_bg[::-1]
    chi = np.geomspace(np.interp(1e-4, z_of_a, chi_of_z), np.interp(z_max, z_of_a, chi_of_z), n_chi)
    z_nodes = np.interp(chi, chi_of_z, z_of_a)
    k_tab = np.asarray(s.cosmo._pk_grid()[0])
    k = np.geomspace(k_tab[0], k_tab[-1], n_k)
    key = (cache, float(z_max), n_chi, n_k, n_fft)
    if key not in s.swiftcl_comps:
        s.swiftcl_comps[key] = swiftcl.ClComp(z1=z_nodes, z2=z_nodes, l=ELL_NL.astype(int), k=k, contr1="intjl",
                                              contr2="intjl", path=cache, N=n_fft)
    cc = s.swiftcl_comps[key]

    def delta_of(t, D_kz):
        D = np.asarray(D_kz(jnp.asarray(k), jnp.asarray(z_nodes))).T  # (N_chi, N_k)
        delta = 0.0
        for W, n, a in t.kernel(s.cosmo, jnp.asarray(z_nodes)):
            W = np.asarray(W) * (np.interp(z_nodes, *bias) if bias is not None and n == 0 else 1.0)
            it = swiftcl.interpolate.interp(l=cc.c["l"], N=cc.c["N"], x=chi, xmin=cc.c["xmin1"], xmax=cc.c["xmax1"],
                                            f=W[:, None] * D, a=k, b=cc.c["bias"], g_r=cc.c["g_r_1"][BESSEL_ROW[n]],
                                            N_interp=cc.c["N_interp"])
            if n == 0:
                delta = delta + np.asarray(it.int_jl())
            elif n == 2:
                delta = delta + np.asarray(it.int_ddjl())
            else:
                delta = delta + np.asarray(it.int_k()) * F_ELL[a](ELL_NL)[:, None] / k[None, :] ** 2
        return delta

    delta1 = delta_of(tracer, D_kz)
    delta2 = delta1 if tracer2 is None else delta_of(tracer2, D_kz2)
    P_fid = np.ravel(s.cosmo.pk(jnp.asarray(k), jnp.array([0.0]), linear=True))
    return 2.0 / np.pi * np.trapezoid(k**2 * delta1 * delta2 * P_fid, x=k, axis=1)


def halo_D_kz(s, profile):
    """D(k, z) of the halo-model 2-halo term, carrying the profile's I_1^1(k, z)."""
    def D_kz(k, z):
        def I11(k, z):
            return jnp.reshape(s.hm.mass_integral(k, z, profile, bias_order=1), (len(k), len(z)))

        return _D_kz(s.cosmo, k, z, z_fid=0.0, linear=True, mass_integral=I11)

    return D_kz


NONLIMBER_LIN = {
    "galaxy b(z)": 3.0,
    "galaxy b(z) + mag + RSD": 3.0,
    "shear": 3.0,
    "CMB lensing": None,  # out to z_star
}


class TestNonLimber:
    @pytest.fixture(scope="class")
    def linbias_spectra(self, s):
        """Non-Limber cl_linbias for each case, with its tracer pair and redshift range."""
        out = {}
        for name, z_max in NONLIMBER_LIN.items():
            t, tc = linbias_pair(s, name, z_max=z_max)
            z_hi = s.z_star if z_max is None else z_max
            c = np.asarray(cl_linbias(s.cosmo, jnp.asarray(ELL_NL), t, z_range=(1e-4, z_hi), n_z=N_Z,
                                      l_limber=L_LIMBER))
            out[name] = (c, t, tc, z_hi, (s.z_g, 1.0 + s.z_g) if name.startswith("galaxy") else None)
        return out

    @pytest.fixture(scope="class")
    def swiftcl_linbias(self, s, swiftcl_cache, linbias_spectra):
        out = {}
        for name, (_, t, _, z_hi, bias) in linbias_spectra.items():
            z_sw = 1.05 * z_hi if name == "CMB lensing" else z_hi
            out[name] = swiftcl_cl(s, swiftcl_cache, t, z_sw,
                                   lambda k, z: _D_kz(s.cosmo, k, z, z_fid=0.0, linear=True), bias=bias)
        return out

    # Non-Limber cl_linbias against SwiftCl, 3 <= l <= 200 (measured 1.8e-4, 1.9e-4, 2.1e-4, 2.4e-4); l = 2 below.
    @pytest.mark.parametrize("name,tol", [("galaxy b(z)", 4e-4), ("galaxy b(z) + mag + RSD", 4e-4), ("shear", 6e-4),
                                          ("CMB lensing", 5e-4)])
    def test_linbias_vs_swiftcl(self, linbias_spectra, swiftcl_linbias, name, tol):
        assert rel_err(linbias_spectra[name][0][1:], swiftcl_linbias[name][1:]) < tol

    # l = 2 against SwiftCl, which needs the zero-padded chi grid (measured 2.0e-4, 1.1e-4, 2.3e-4, 2.2e-4).
    @pytest.mark.parametrize("name,tol", [
        ("galaxy b(z)", 4e-4),
        ("galaxy b(z) + mag + RSD", 2.5e-4),
        ("shear", 6e-4),
        ("CMB lensing", 5e-4),
    ])
    def test_linbias_l2_vs_swiftcl(self, linbias_spectra, swiftcl_linbias, name, tol):
        assert rel_err(linbias_spectra[name][0][0], swiftcl_linbias[name][0]) < tol

    # Non-Limber cl_linbias against CCL's FKEM non-Limber method, 3 <= l <= 200; FKEM's own scatter, up to 0.5% at
    # l ~ 10-30 where hmfast and SwiftCl agree to 0.03%, sets the floor (measured 6.6e-3, 1.5e-3, 7.8e-3, 6.6e-3).
    @pytest.mark.parametrize("name,tol", [("galaxy b(z)", 1.3e-2), ("galaxy b(z) + mag + RSD", 3e-3),
                                          ("shear", 1.6e-2), ("CMB lensing", 1.3e-2)])
    def test_linbias_vs_ccl_fkem(self, s, linbias_spectra, name, tol):
        got, _, tc, _, _ = linbias_spectra[name]
        p_lin = s.ccl_lin.get_linear_power()
        want = pyccl.angular_cl(s.ccl_lin, tc, tc, ELL_NL, p_of_k_a=p_lin, p_of_k_a_lin=p_lin, l_limber=L_LIMBER,
                                non_limber_integration_method="FKEM")
        assert rel_err(got[1:], want[1:]) < tol

    # Non-Limber halo-model 2-halo term against SwiftCl with D(k, z) carrying I_1^1, 3 <= l <= 200
    # (measured 1.8e-4 / 3.0e-4).
    @pytest.mark.parametrize("make,profile,tol", [("galaxy", HOD, 4e-4), ("shear", NFW, 6e-4)])
    def test_halo_model_vs_swiftcl(self, s, swiftcl_cache, make, profile, tol):
        t = getattr(s, make)()[0]
        got = cl(PK_2H, s.hm, jnp.asarray(ELL_NL), t, z_range=Z_RANGE, n_z=N_Z, l_limber=L_LIMBER)
        assert rel_err(got[1:], swiftcl_cl(s, swiftcl_cache, t, Z_RANGE[1], halo_D_kz(s, profile))[1:]) < tol

    # Edge kernels vs SwiftCl, measured l=2 / l>=3: y 1.7e-5/4.7e-3, CIB 4.7e-3/2.3e-2, y x kCMB 3.7e-4/1.4e-3
    @pytest.mark.parametrize("make1,make2,n_fft,sw_n_fft,tol_l2,tol", [
        ("tsz", None, None, 2048, 4e-5, 1e-2),
        ("cib", None, 2000, 4096, 1e-2, 4.5e-2),
        ("tsz", "cmb_lensing", None, 2048, 8e-4, 3e-3),
    ])
    def test_edge_kernels_vs_swiftcl(self, s, swiftcl_cache, make1, make2, n_fft, sw_n_fft, tol_l2, tol):
        t1 = getattr(s, make1)()[0]
        t2 = None if make2 is None else getattr(s, make2)()[0]
        z_hi = Z_RANGE[1] if t2 is None else s.z_star
        got = cl(PK_2H, s.hm, jnp.asarray(ELL_NL), t1, t2, z_range=(Z_RANGE[0], z_hi), n_z=N_Z, l_limber=L_LIMBER,
                 n_fft=n_fft)
        profile = {"tsz": GNFW, "cib": CIB, "cmb_lensing": NFW}
        want = swiftcl_cl(s, swiftcl_cache, t1, z_hi if t2 is None else 1.05 * z_hi, halo_D_kz(s, profile[make1]),
                          tracer2=t2, D_kz2=None if t2 is None else halo_D_kz(s, profile[make2]), n_fft=sw_n_fft,
                          n_k=4000)
        assert rel_err(got[0], want[0]) < tol_l2
        assert rel_err(got[1:], want[1:]) < tol

    # Non-Limber halo-model 2-halo shear against CCL's FKEM on CCL's own 2-halo P(k), 3 <= l <= 200 (measured 7.8e-3).
    def test_halo_model_shear_vs_ccl_fkem(self, s):
        t, tc = s.shear()
        got = cl(PK_2H, s.hm, jnp.asarray(ELL_NL), t, z_range=Z_RANGE, n_z=N_Z, l_limber=L_LIMBER)
        p2h = s.pk2d["m"][1]
        want = pyccl.angular_cl(s.ccl, tc, tc, ELL_NL, p_of_k_a=p2h, p_of_k_a_lin=p2h, l_limber=L_LIMBER,
                                non_limber_integration_method="FKEM")
        assert rel_err(got[1:], want[1:]) < 1.6e-2


# ---------------------------------------------------------------------------------------------------------------------
# Angular correlation functions.
# ---------------------------------------------------------------------------------------------------------------------


class TestCorrAngular:
    # corr_angular against pyccl.correlation (FFTLog, flat sky) for the same input C_ell over 3' <= theta <= 300', where
    # |xi| exceeds 1% of its maximum (measured 4.4e-3, 4.3e-3, 2.7e-3, 8.4e-3; floor not yet traced).
    @pytest.mark.parametrize("typ,tol", [("NN", 9e-3), ("NG", 9e-3), ("GG+", 5.5e-3), ("GG-", 1.7e-2)])
    def test_corr_angular(self, s, typ, tol):
        ell = np.geomspace(2, 1e5, 1024)
        theta = np.geomspace(3.0, 300.0, 40) / 60.0
        galaxy, shear = s.galaxy(bias=(s.z_g, 1.0 + s.z_g), lin=True)[0], s.shear(lin=True)[0]
        t1, t2 = {"NN": (galaxy, None), "NG": (galaxy, shear)}.get(typ, (shear, None))
        c = np.asarray(cl_linbias(s.cosmo, jnp.asarray(ell), t1, t2, z_range=Z_RANGE, n_z=N_Z, linear=False))
        got = np.asarray(corr_angular(s.cosmo, ell, c, theta, type=typ))
        want = pyccl.correlation(s.ccl_lin, ell=ell, C_ell=c, theta=theta, type=typ, method="fftlog")
        keep = np.abs(want) > 1e-2 * np.max(np.abs(want))
        assert rel_err(got[keep], want[keep]) < tol
