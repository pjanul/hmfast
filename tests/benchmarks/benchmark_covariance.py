"""
Benchmarks for the covariances in hmfast.stats: sigma2_b_disc, the super-sample response, cov_ssc and cov_cng.

sigma2_b_disc is checked against direct quadrature of its defining integral, with P_lin continued below the emulator's
k_min as a power law; CCL's sigma2_B_disc is less accurate at large radii (no P(k) below k = 1e-4).

The covariance reference is assembled directly from CCL rather than through its Tk3D tables and Limber integrator,
which are unconverged at practical resolution for cNG (13% at 50 x 30 nodes, 8% at 100 x 60) and whose integrator
fails here: at each of hmfast's own redshift nodes, CCL's halo-model trispectrum or SSC response is evaluated at
exactly k = (l + 1/2) / chi, multiplied by CCL's tracer kernels and summed with the same quadrature weights. Sharing the
nodes tests the integrand (kernels, weights, trispectrum or response), not quadrature convergence. The nodes span the
redshift range hmfast takes from the tracers.

Every comparison runs hmfast with ncdm_mode="m" against a CCL CosmologyCalculator fed hmfast's background, growth and
P(k). To stay within a few minutes (each tracer compiles its own covariance, 30-80 s), the cases cover distinct code
paths only: the SSC response for a continuous field and a number count, SSC projection with the counter-term
(galaxies), and cNG projection of lensing (shear) and HOD (galaxy) kernels on the 1-halo + 4-halo terms, which carry
the tracer-specific pair moments. Every trispectrum term is validated in benchmark_stats_3d.py, the projection
kernels in benchmark_tracers.py and benchmark_stats_projected.py, and the term sum in benchmark_jax_transforms.py.
Measured when written but not kept: SSC 2.5e-3 shear, 2.6e-3 CMB lensing, 2.0e-3 tSZ, 1.4e-3 CIB; cNG (1h + 4h)
5.1e-3 CMB lensing, 2.8e-3 tSZ; cNG with the full trispectrum 4.8e-3 for shear.

Requires pyccl; marked `ccl` and skipped if it is missing.

Thresholds are ~2x the error measured when written (noted per test), so a ~2x degradation fails; lower them when
accuracy improves. Errors are maximum relative errors over the (l1, l2) matrix unless stated otherwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import j1

from hmfast.stats import Pk, Tk, cov_cng, cov_ssc, sigma2_b_disc
from hmfast.stats.covariance import _dPk_response
from hmfast.stats.projected_cl import _resolve_z_range
from hmfast.utils import gauss_legendre_nodes_weights

from .._shared import peak_err, rel_err

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

from .benchmark_stats_projected import HOD, NFW, Setup  # noqa: E402

C_KM_S = 299792.458
F_SKY = 0.1
ELL = np.geomspace(20, 2000, 5)
N_Z = 20
TK_1H_4H = Tk(include_2h=False, include_3h=False)
TRACERS = {  # name: (Setup method, CCL profile key)
    "shear": ("shear", "m"),
    "galaxy": ("galaxy", "g"),
    "CIB": ("cib", "c"),
}


@pytest.fixture(scope="module")
def s(fixed_cosmology_m):
    return Setup(fixed_cosmology_m, halo_pk2d=False)


def z_nodes(s, tracer):
    """hmfast's covariance quadrature: Gauss-Legendre in ln z, weights including the dz / dln z Jacobian."""
    z_min, z_max = (float(z) for z in _resolve_z_range(s.cosmo, tracer))
    logz, w = gauss_legendre_nodes_weights(np.log(z_min), np.log(z_max), N_Z)
    z = np.exp(np.asarray(logz))
    return z, np.asarray(w) * z


def ccl_limber_weight(tracer_ccl, chi, a):
    """CCL tracer's Limber weight at chi for every l: kernel x transfer x f_l, divided by (l + 1/2)^2 for j_l / x^2."""
    W = np.atleast_1d(np.atleast_2d(tracer_ccl.get_kernel(np.array([chi])))[:, 0])
    T = np.ravel(tracer_ccl.get_transfer(np.log(1e-2), np.array([a])))
    f = np.atleast_2d(tracer_ccl.get_f_ell(ELL))
    n = np.atleast_1d(tracer_ccl.get_bessel_derivative())
    return sum(W[i] * T[i] * f[i] / ((ELL + 0.5) ** 2 if n[i] == -1 else 1.0) for i in range(len(W)))


def ccl_trispectrum(s, key, k, a, terms):
    """CCL's halo-model T(k, k') summed over `terms`, as [k_u, k_v]; same-k leg pairs take the profile's pair moment."""
    hm, prof, p2pt = pyccl.halos, s.ccl_profiles[key], s.ccl_2pt.get(key)
    pair = {} if p2pt is None else dict(prof12_2pt=p2pt, prof34_2pt=p2pt)
    term_fns = {
        "1h": lambda: hm.halomod_trispectrum_1h(s.ccl, s.hmc, k, a, prof, **pair),
        "2h": lambda: (hm.halomod_trispectrum_2h_22(s.ccl, s.hmc, k, a, prof)
                       + hm.halomod_trispectrum_2h_13(s.ccl, s.hmc, k, a, prof, **pair)),
        "3h": lambda: hm.halomod_trispectrum_3h(s.ccl, s.hmc, k, a, prof),
        "4h": lambda: hm.halomod_trispectrum_4h(s.ccl, s.hmc, k, a, prof),
    }
    return sum(term_fns[t]() for t in terms).T


def ccl_response(s, key, k, a, slope=None):
    """CCL's halo-model SSC response dP/d delta_b (its Tk3D_SSC leg), optionally with a given dlnP/dlnk."""
    prof, p2pt = s.ccl_profiles[key], s.ccl_2pt.get(key)
    norm = prof.get_normalization(s.ccl, a, hmc=s.hmc)
    i11 = s.hmc.I_1_1(s.ccl, k, a, prof) / norm
    # hmfast's 1-halo response term is the plain product of first moments (see covariance._dPk_response).
    i12 = s.hmc.I_1_2(s.ccl, k, a, prof, prof_2pt=pyccl.halos.Profile2pt()) / norm**2
    p_lin = s.ccl.get_linear_power()
    pk = p_lin(k, a)
    slope = p_lin(k, a, derivative=True) if slope is None else slope
    response = (47.0 / 21.0 - slope / 3.0) * i11**2 * pk + i12
    if prof.is_number_counts:
        i02 = s.hmc.I_0_2(s.ccl, k, a, prof, prof_2pt=p2pt) / norm**2
        response = response - 2.0 * i11 * (pk * i11**2 + i02)
    return response


def direct_covariance(s, tracer, tracer_ccl, key, kind, terms=("1h", "2h", "3h", "4h")):
    """The Limber covariance on hmfast's z nodes for `tracer` from CCL's T or response at k = (l + 1/2) / chi."""
    z, w = z_nodes(s, tracer)
    cov = 0.0
    for z_i, w_i in zip(z, w):
        a = 1.0 / (1.0 + z_i)
        chi = pyccl.comoving_radial_distance(s.ccl, a)
        dchi_dz = C_KM_S / (s.cosmo.H0 * pyccl.h_over_h0(s.ccl, a))
        k = (ELL + 0.5) / chi
        K2 = ccl_limber_weight(tracer_ccl, chi, a) ** 2
        if kind == "cng":
            T = ccl_trispectrum(s, key, k, a, terms)
            cov = cov + w_i * dchi_dz * K2[:, None] * K2[None, :] * T / chi**6 / (4.0 * np.pi * F_SKY)
        else:
            R = K2 * ccl_response(s, key, k, a)
            s2b = float(sigma2_b_disc(s.cosmo, jnp.array([z_i]), f_sky=F_SKY))
            cov = cov + w_i * dchi_dz * R[:, None] * R[None, :] / chi**4 * s2b
    return cov


def tracer_pair(s, name):
    return getattr(s, TRACERS[name][0])()


# ---------------------------------------------------------------------------------------------------------------------
# Survey-window variance and SSC response.
# ---------------------------------------------------------------------------------------------------------------------


class TestSigma2BDisc:
    # Against direct quadrature of int k dk / 2 pi P_lin(k) [2 J_1(kR) / kR]^2, P_lin power-law continued below k_min,
    # out to R = chi pi at z = 5 (f_sky = 1), where the output grid must reach (measured 2.6e-5).
    @pytest.mark.parametrize("f_sky", [0.01, 0.1, 0.5, 1.0])
    def test_against_quadrature(self, fixed_cosmology_m, f_sky):
        cosmo = fixed_cosmology_m
        z = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
        R = np.asarray(cosmo.angular_diameter_distance(jnp.asarray(z))) * (1 + z) * np.arccos(1 - 2 * f_sky)
        k = np.geomspace(1e-8, 50.0, 100000)
        p_lin = np.asarray(cosmo.update(extrapolate_k=True).pk(jnp.asarray(k), jnp.asarray(z), linear=True))
        x = k[:, None] * R[None, :]
        window = np.where(x > 1e-8, 2.0 * j1(x) / np.where(x > 1e-8, x, 1.0), 1.0)
        want = np.trapezoid(k[:, None] ** 2 * p_lin * window**2 / (2.0 * np.pi), np.log(k), axis=0)
        assert rel_err(sigma2_b_disc(cosmo, jnp.asarray(z), f_sky=f_sky), want) < 6e-5


class TestSSCResponseCCL:
    # dP/d delta_b against CCL's Tk3D_SSC leg assembled from its I_1_1, I_1_2, I_0_2 and P_lin, given hmfast's own
    # dlnP/dlnk, as max |difference| over the peak: the number-count counter-term drives galaxy responses through zero
    # (measured 1.4e-4 matter, 5.9e-4 galaxies; 1.1e-3 pressure and 2.2e-4 CIB when written, not kept).
    @pytest.mark.parametrize("key,tol", [("m", 3e-4), ("g", 1.2e-3)])
    def test_response(self, s, key, tol):
        profile = {"m": NFW, "g": HOD}[key]
        k = np.geomspace(5e-3, 10.0, 40)
        k_grid = np.asarray(s.cosmo._pk_grid())
        for z in (0.3, 1.0):
            za = jnp.array([z])
            p_grid = np.ravel(s.cosmo.pk(jnp.asarray(k_grid), za, linear=True))
            slope = np.interp(np.log(k), np.log(k_grid), np.gradient(np.log(p_grid), np.log(k_grid)))
            got = _dPk_response(s.hm, jnp.asarray(k), za, profile)
            assert peak_err(got, ccl_response(s, key, k, 1.0 / (1.0 + z), slope=slope)) < tol

    # The same with CCL's own slope, the derivative of its spline of the same P_lin table: hmfast's finite-difference
    # slope differs by up to 0.08 at the BAO scale, ~0.7% of the matter response (measured 6.7e-3).
    def test_matter_response_with_ccl_slope(self, s):
        k = np.geomspace(5e-3, 10.0, 40)
        got = _dPk_response(s.hm, jnp.asarray(k), jnp.array([0.5]), NFW)
        assert rel_err(got, ccl_response(s, "m", k, 1.0 / 1.5)) < 1.4e-2


# ---------------------------------------------------------------------------------------------------------------------
# Covariances against the direct reference.
# ---------------------------------------------------------------------------------------------------------------------


class TestCovSSC:
    # Galaxy super-sample covariance, with CCL's own slope in the reference: the slope difference, amplified where
    # the counter-term drives the response through zero, sets the error (measured 1.6e-2).
    def test_cov_ssc_galaxy(self, s):
        t, tc = tracer_pair(s, "galaxy")
        got = cov_ssc(Pk(), s.hm, jnp.asarray(ELL), jnp.asarray(ELL), t, n_z=N_Z, f_sky=F_SKY)
        assert rel_err(got, direct_covariance(s, t, tc, "g", "ssc")) < 3.2e-2


class TestCovCNG:
    # Connected covariance from the 1-halo + 4-halo trispectrum (measured 1.1e-2 shear, set by the 4-halo term at the
    # most separated (l1, l2), and 2.8e-3 galaxies). hmfast's 1-halo trispectrum pairs legs at the same k with the plain
    # product of first moments, where Pk (and CCL) use the pair moment; that adds central self-pairs, negligible for
    # galaxies but dominant for CIB (T_1h up to 2.3x CCL's, this cNG ~40x).
    @pytest.mark.parametrize("name,tol", [
        ("shear", 2.2e-2),
        ("galaxy", 6e-3),
        pytest.param("CIB", 6e-3, marks=pytest.mark.xfail(
            strict=True, reason="Tk pairs same-k legs with the plain product of first moments, not the pair moment")),
    ])
    def test_cov_cng_1h_4h(self, s, name, tol):
        t, tc = tracer_pair(s, name)
        got = cov_cng(TK_1H_4H, s.hm, jnp.asarray(ELL), jnp.asarray(ELL), t, n_z=N_Z, f_sky=F_SKY)
        assert rel_err(got, direct_covariance(s, t, tc, TRACERS[name][1], "cng", terms=("1h", "4h"))) < tol
