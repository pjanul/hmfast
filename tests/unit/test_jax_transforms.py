"""Every public hmfast entry point must stay usable inside ``jax.jit``.

Jittability is a load-bearing property of this library, not a nice-to-have: an entry
point that cannot be traced cannot be batched with ``vmap``, cannot be placed inside a
gradient-based sampler's log-posterior, and pays eager per-operation dispatch on every
call (measured at 6x for ``pk_*``, 21x for ``bk_*`` and up to 40x for ``tk_*``). It is
also easy to break by accident, because the failure modes are all ordinary Python that
works perfectly well outside a trace -- a ``float()``, a ``bool()``, a ``len()``, or an
array handed to a library that plans on concrete values.

Each case builds its object graph *inside* the timed function from a traced parameter
vector, which is how a sampler uses the library: the cosmology, halo model and any
derived quantity are all constructed under the trace, so nothing can accidentally be
captured as a compile-time constant.

Every entry point is expected to pass. Should one regress and need to be parked, mark it
with ``broken()`` rather than deleting it: that is an ``xfail(strict=True)``, so fixing
the underlying cause turns the run red until the marker is removed, and the gap can
never silently reopen.

Tracing is where every one of these failures surfaces, so the unit test only traces each
case; benchmarks/benchmark_jax_transforms.py compiles them and checks jitted against eager
numbers, and runs the heavier end-to-end gradient cases.

This file also holds every gradient check in the unit suite, from full observables down to
single components, each batched into one Jacobian per group of quantities.
"""

import functools
import inspect

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax.scipy.special  # noqa: F401  (probed below for loggamma)

from hmfast.cosmology import Cosmology, CosmoPowerCosmology
from hmfast.cosmology.cosmopower import _WEIGHTS
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import (
    B13Concentration,
    ConstantConcentration,
    D08Concentration,
)
from hmfast.halos.massdef import MassDefinition, mass_translator
from hmfast.halos.massfunc import (
    JvdB14SubHaloMassFunction,
    T08HaloMassFunction,
    T10HaloMassFunction,
    TW10SubHaloMassFunction,
)
from hmfast.halos.profiles import (
    HaloProfile,
    B12PressureProfile,
    M21CIBProfile,
    B16DensityProfile,
    GNFWPressureProfile,
    NFWMatterProfile,
    S12CIBProfile,
    Z07GalaxyHODProfile,
)
from hmfast.stats import Bk, Pk, Tk, cl, cl_linbias, corr_3d, corr_angular, cov_cng, cov_g, cov_ssc, sigma2_b_disc
from hmfast.stats import projected_cl as _cl_module
from hmfast.stats import covariance as _covariance_module
from hmfast.tracers import CMBLensingTracer, GalaxyLensingTracer, GalaxyTracer, kSZTracer

from hmfast.utils import gauss_legendre_nodes_weights

from .._shared import (
    B12, B16, BIAS, BK, CIB, CIB_TRACER, CMBLENS_TRACER, CONC, FIXED_COSMOLOGY, GAL_TRACER,
    GAL_TRACER_BIASED, GLENS_TRACER, GNFW, HMF, HOD, K_FFT, K_GRID, K_GRID_BT, KSZ_TRACER, L_FFT, L_GRID, M21,
    M_GRID, MD_200C, MD_200M, MD_500C, MD_VIR, N_R, N_Z, NFW, PARAMS, PK, R_GRID, R_GRID_SIGMA, THETA_GRID, TK,
    TSZ_TRACER, Z_GRID, Z_RANGE, Z_SINGLE, Z_STAR, cosmo, shared, stats_halo_model,
)
from .._shared import halo_model as _shared_halo_model

PK0 = Pk(k_damp=0.0)  # damping disabled, for cases that used to pass k_damp=0.0 as a call-time kwarg
BK0 = Bk(k_damp=0.0)  # damping disabled, for cases that used to pass k_damp=0.0 as a call-time kwarg

MASS_TRANSLATOR_200M_500C = mass_translator(MD_200M, MD_500C, CONC)
halo_model = stats_halo_model


# One entry per public entry point; `broken` marks a known gap with a strict xfail.

CASES = []


def case(name, fn):
    CASES.append(pytest.param(fn, id=name))


def broken(name, fn, reason):
    CASES.append(pytest.param(fn, id=name, marks=pytest.mark.xfail(strict=True, reason=reason)))


# _hankel_A_table needs a complex log-gamma, which jax.scipy.special only grew in 0.10.
_NEEDS_LOGGAMMA = pytest.mark.skipif(
    not hasattr(jax.scipy.special, "loggamma"),
    reason=f"jax {jax.__version__} has no jax.scipy.special.loggamma (needs >= 0.10)",
)

# --- cosmology -------------------------------------------------------------------
case("Cosmology.hubble_parameter", lambda p: cosmo(p).hubble_parameter(Z_GRID))
case("Cosmology.angular_diameter_distance", lambda p: cosmo(p).angular_diameter_distance(Z_GRID))
case("Cosmology.critical_density", lambda p: cosmo(p).critical_density(Z_GRID))
case("Cosmology.omega_m", lambda p: cosmo(p).omega_m(Z_GRID))
case("Cosmology.delta_c", lambda p: cosmo(p).delta_c(Z_GRID))
case("Cosmology.growth_factor", lambda p: cosmo(p).growth_factor(Z_GRID))
case("Cosmology.growth_rate", lambda p: cosmo(p).growth_rate(Z_GRID))
case("Cosmology.sigma8", lambda p: cosmo(p).sigma8(Z_GRID))
case("Cosmology.sigma_m", lambda p: cosmo(p).sigma_m(M_GRID, Z_GRID))
case("Cosmology.sigma_r", lambda p: cosmo(p).sigma_r(R_GRID_SIGMA, Z_GRID))
case("Cosmology.velocity_dispersion", lambda p: cosmo(p).velocity_dispersion(Z_GRID))
case("Cosmology.comoving_volume_element", lambda p: cosmo(p).comoving_volume_element(Z_GRID))
case("Cosmology.pk[linear]", lambda p: cosmo(p).pk(K_GRID, Z_GRID, linear=True))
case("Cosmology.pk[nonlinear]", lambda p: cosmo(p).pk(K_GRID, Z_GRID, linear=False))
case("Cosmology.cl[tt]", lambda p: cosmo(p).cl_cmb("tt", jnp.arange(2, 50)))
case("Cosmology.derived_parameters", lambda p: cosmo(p).derived_parameters())

# --- mass definitions ------------------------------------------------------------
case("MassDefinition.r_delta[200m]", lambda p: MD_200M.r_delta(cosmo(p), M_GRID, Z_GRID))
case("MassDefinition.r_delta[vir]", lambda p: MD_VIR.r_delta(cosmo(p), M_GRID, Z_GRID))
case("mass_translator[200m->500c]", lambda p: MASS_TRANSLATOR_200M_500C(cosmo(p), M_GRID, Z_GRID))

# --- mass functions, bias, concentration -----------------------------------------
case("T08HaloMassFunction.dndlnm", lambda p: HMF.dndlnm(cosmo(p), M_GRID, Z_GRID, MD_200M))
case("T10HaloMassFunction.dndlnm",
     lambda p: shared(T10HaloMassFunction).dndlnm(cosmo(p), M_GRID, Z_GRID, MD_200M))
case("TW10SubHaloMassFunction.dndlnmu",
     lambda p: TW10SubHaloMassFunction().dndlnmu(cosmo(p), M_GRID, M_GRID / 10.0))
case("JvdB14SubHaloMassFunction.dndlnmu",
     lambda p: JvdB14SubHaloMassFunction().dndlnmu(cosmo(p), M_GRID, M_GRID / 10.0))
case("T10HaloBias.bias[order=1]", lambda p: BIAS.bias(cosmo(p), M_GRID, Z_GRID, MD_200M, order=1))
case("T10HaloBias.bias[order=2]", lambda p: BIAS.bias(cosmo(p), M_GRID, Z_GRID, MD_200M, order=2))
case("ConstantConcentration.c_delta",
     lambda p: shared(ConstantConcentration, c=5.0).c_delta(cosmo(p), M_GRID, Z_GRID, MD_200C))
case("D08Concentration.c_delta", lambda p: CONC.c_delta(cosmo(p), M_GRID, Z_GRID, MD_200C))
case("B13Concentration.c_delta",
     lambda p: shared(B13Concentration).c_delta(cosmo(p), M_GRID, Z_GRID, MD_200C))

# --- halo model ------------------------------------------------------------------
case("HaloModel._counter_terms", lambda p: halo_model(p)._counter_terms(Z_GRID))
case("HaloModel.mass_integral", lambda p: halo_model(p).mass_integral(K_GRID, Z_GRID, NFW, bias_order=1))

# --- profiles --------------------------------------------------------------------
for _name, _prof, _md in [
    ("NFWMatterProfile", NFW, MD_200M),
    ("GNFWPressureProfile", GNFW, MD_500C),
    ("B12PressureProfile", B12, MD_200C),
    ("B16DensityProfile", B16, MD_200C),
    ("Z07GalaxyHODProfile", HOD, MD_200M),
    ("S12CIBProfile", CIB, MD_200M),
]:
    case(f"{_name}.real",
         (lambda pr, md: lambda p: pr.real(halo_model(p, md), R_GRID, M_GRID, Z_SINGLE))(_prof, _md))
    case(f"{_name}.fourier",
         (lambda pr, md: lambda p: pr.fourier(halo_model(p, md), K_GRID, M_GRID, Z_SINGLE))(_prof, _md))

case("Z07GalaxyHODProfile.n_cen", lambda p: HOD.n_cen(halo_model(p), M_GRID))
case("Z07GalaxyHODProfile.n_sat", lambda p: HOD.n_sat(halo_model(p), M_GRID))
case("Z07GalaxyHODProfile.ng_bar", lambda p: HOD.ng_bar(halo_model(p), Z_GRID))
case("Z07GalaxyHODProfile.galaxy_bias", lambda p: HOD.galaxy_bias(halo_model(p), Z_GRID))
case("S12CIBProfile.l_gal", lambda p: CIB.l_gal(halo_model(p), M_GRID, Z_GRID))
case("S12CIBProfile.l_sat", lambda p: CIB.l_sat(halo_model(p), M_GRID, Z_GRID))
case("S12CIBProfile.l_cen", lambda p: CIB.l_cen(halo_model(p), M_GRID, Z_GRID))
case("S12CIBProfile.mean_emissivity", lambda p: CIB.mean_emissivity(halo_model(p), Z_GRID))
case("S12CIBProfile.mean_intensity", lambda p: CIB.mean_intensity(halo_model(p), z_range=Z_RANGE, n_z=N_Z))

# --- tracers ---------------------------------------------------------------------
for _name, _tracer in [
    ("tSZTracer", TSZ_TRACER),
    ("kSZTracer", KSZ_TRACER),
    ("GalaxyTracer", GAL_TRACER),
    ("GalaxyLensingTracer", GLENS_TRACER),
    ("CMBLensingTracer", CMBLENS_TRACER),
    ("CIBTracer", CIB_TRACER),
]:
    # kernel() returns (weight, der_bessel, der_angles) triples; the tags are Python ints, so weights only.
    # CMB lensing's source plane z_star lies beyond the emulated background, so it needs extrapolate_z.
    case(f"{_name}.kernel",
         (lambda t, ext: lambda p: [w for w, _, _ in t.kernel(cosmo(p).update(extrapolate_z=ext), Z_GRID)])(
             _tracer, _tracer is CMBLENS_TRACER))

# --- 2-point statistics ----------------------------------------------------------
case("Pk.pk_1h", lambda p: PK0.pk_1h(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("Pk.pk_2h", lambda p: PK.pk_2h(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("Pk.pk_tot", lambda p: PK0.pk_tot(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("Pk.pk_tot[1h only]",
     lambda p: Pk(k_damp=0.0, include_2h=False).pk_tot(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("corr_3d[pk_tot]", lambda p: corr_3d(cosmo(p), K_FFT, PK.pk_tot(halo_model(p), K_FFT, Z_SINGLE, NFW), R_GRID))
case("corr_3d[pk_nl]", lambda p: corr_3d(cosmo(p), K_FFT, cosmo(p).pk(K_FFT, Z_GRID, linear=False), R_GRID))
case("cl[limber]",
     lambda p: cl(PK, halo_model(p), L_GRID, GAL_TRACER, GAL_TRACER, n_z=N_Z))
CASES.append(pytest.param(
    lambda p: cl(PK, halo_model(p), L_GRID, GAL_TRACER, GAL_TRACER, n_z=N_Z, l_limber=100.0),
    id="cl[non-limber]", marks=_NEEDS_LOGGAMMA))
case("cl[1h only]",
     lambda p: cl(Pk(k_damp=0.0, include_2h=False), halo_model(p), L_GRID, GAL_TRACER, GAL_TRACER, n_z=N_Z))
case("cl_linbias",
     lambda p: cl_linbias(cosmo(p), L_GRID, GAL_TRACER_BIASED, GAL_TRACER_BIASED, n_z=N_Z))
for _type in ("NN", "NG", "GG+", "GG-"):
    case(f"corr_angular[{_type}]",
         (lambda t: lambda p: corr_angular(
             cosmo(p), L_FFT, cl_linbias(cosmo(p), L_FFT, GLENS_TRACER, GLENS_TRACER, n_z=N_Z), THETA_GRID, type=t))(_type))

# --- higher-order statistics and covariances -------------------------------------
case("Bk.bk_1h",
     lambda p: BK0.bk_1h(halo_model(p), K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW))
case("Bk.bk_2h",
     lambda p: BK.bk_2h(halo_model(p), K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW))
case("Bk.bk_3h",
     lambda p: BK.bk_3h(halo_model(p), K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW))
case("Bk.bk_tot",
     lambda p: BK0.bk_tot(halo_model(p), K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW))
case("Bk.bk_tot[1h only]",
     lambda p: Bk(k_damp=0.0, include_2h=False, include_3h=False).bk_tot(
         halo_model(p), K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW))
for _order in (1, 2, 3, 4):
    case(f"Tk.tk_{_order}h",
         (lambda o: lambda p: getattr(TK, f"tk_{o}h")(halo_model(p), K_GRID_BT, K_GRID_BT, Z_SINGLE, NFW))(_order))
case("Tk.tk_tot",
     lambda p: TK.tk_tot(halo_model(p), K_GRID_BT, K_GRID_BT, Z_SINGLE, NFW))
case("Tk.tk_tot[2h only]",
     lambda p: Tk(include_1h=False, include_3h=False, include_4h=False).tk_tot(
         halo_model(p), K_GRID_BT, K_GRID_BT, Z_SINGLE, NFW))
case("cov_g", lambda p: cov_g(L_GRID, p[0] * L_GRID**-2.0, f_sky=0.4))
case("cov_cng",
     lambda p: cov_cng(TK, halo_model(p), L_GRID[:3], L_GRID[:3], GAL_TRACER, None, None, None, n_z=N_Z))
case("cov_ssc",
     lambda p: cov_ssc(PK, halo_model(p), L_GRID[:3], L_GRID[:3], GAL_TRACER, None, None, None,
                       n_z=N_Z, f_sky=0.4))
case("sigma2_b_disc", lambda p: sigma2_b_disc(cosmo(p), Z_GRID, f_sky=0.4))


@pytest.mark.parametrize("fn", CASES)
def test_jittable(fn):
    """Traces under jax.jit before any eager call, so no cache an eager call fills can mask a tracing failure."""
    jax.jit(fn).trace(PARAMS)


def test_no_retrace_on_new_parameters():
    """A second call at different parameters must reuse the compiled kernel.

    Jittability is worth little if every parameter draw recompiles, which is what a
    sampler would hit if a cosmological parameter leaked into a pytree's aux data
    instead of staying a leaf.
    """
    fn = jax.jit(lambda p: PK0.pk_1h(halo_model(p), K_GRID, Z_SINGLE, NFW))
    jax.block_until_ready(fn(PARAMS))
    n_compiles = fn._cache_size()

    jax.block_until_ready(fn(PARAMS * jnp.array([1.01, 0.99, 1.02, 0.98, 1.0])))
    assert fn._cache_size() == n_compiles, "changing parameter values triggered a retrace"


def test_pk_k_damp_sweep_does_not_recompile():
    """Varying Pk.k_damp must reuse the compiled kernel, not just cosmological parameters.

    Pk.pk_1h pins `self` static if Pk isn't a registered pytree, which hashes it by
    identity and pays a full compile every time k_damp changes (the same failure mode
    test_profile_parameter_sweep_does_not_recompile guards against for profiles).
    """
    hm = halo_model(PARAMS)
    fn = jax.jit(lambda pk_obj: pk_obj.pk_1h(hm, K_GRID, Z_SINGLE, NFW))

    jax.block_until_ready(fn(Pk(k_damp=0.01)))
    n_compiles = fn._cache_size()

    jax.block_until_ready(fn(Pk(k_damp=0.05)))
    assert fn._cache_size() == n_compiles, "changing k_damp triggered a retrace"


def test_bk_k_damp_sweep_does_not_recompile():
    """Varying Bk.k_damp must reuse the compiled kernel (mirrors the Pk.k_damp case)."""
    hm = halo_model(PARAMS)
    fn = jax.jit(lambda bk_obj: bk_obj._bk_1h(hm, K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW))

    jax.block_until_ready(fn(Bk(k_damp=0.01)))
    n_compiles = fn._cache_size()

    jax.block_until_ready(fn(Bk(k_damp=0.05)))
    assert fn._cache_size() == n_compiles, "changing k_damp triggered a retrace"


def test_cl_survives_jit_before_eager():
    """Tracing must not poison a shared CosmoPowerCosmology for later use.

    The first cl_cmb call loads the set's TT emulator; when that happens under a
    trace, the loaded weights must stay concrete. A sampler that jits its likelihood, then takes
    its gradient or inspects a spectrum eagerly on the same object, hits exactly this order.
    """
    _WEIGHTS.pop(("lcdm:v1", "TT"), None)  # so the first load below happens under the trace
    obj = CosmoPowerCosmology(
        "lcdm:v1", H0=67.36, omega_cdm=0.12011, omega_b=0.02242,
        A_s=2.1005e-9, n_s=0.9665,
    )
    ell = jnp.arange(2, 200)
    fn = lambda p: obj.update(H0=p[0], omega_cdm=p[1], omega_b=p[2], A_s=p[3], n_s=p[4]).cl_cmb("tt", ell)  # noqa: E731

    jitted = jax.block_until_ready(jax.jit(fn)(PARAMS))
    grad = jax.block_until_ready(jax.jit(jax.grad(lambda p: jnp.sum(fn(p))))(PARAMS))
    eager = jax.block_until_ready(fn(PARAMS))
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(jitted, eager, rtol=1e-10)


# Internal jitting: a user must never need to write jax.jit to get compiled performance.

# (owner class, attribute), for entry points whose cost justifies a compile.
JITTED_API = [
    (Cosmology, "hubble_parameter"), (Cosmology, "angular_diameter_distance"),
    (Cosmology, "critical_density"), (Cosmology, "omega_m"), (Cosmology, "delta_c"),
    (Cosmology, "growth_factor"), (Cosmology, "growth_rate"), (Cosmology, "sigma8"),
    (Cosmology, "sigma_m"), (Cosmology, "sigma_r"),
    (Cosmology, "pk"),
    (MassDefinition, "r_delta"),
    (T08HaloMassFunction, "dndlnm"), (T10HaloMassFunction, "dndlnm"),
    (TW10SubHaloMassFunction, "dndlnmu"), (JvdB14SubHaloMassFunction, "dndlnmu"),
    (T10HaloBias, "bias"),
    (ConstantConcentration, "c_delta"), (D08Concentration, "c_delta"), (B13Concentration, "c_delta"),
    (NFWMatterProfile, "real"), (NFWMatterProfile, "fourier"),
    (GNFWPressureProfile, "real"), (GNFWPressureProfile, "fourier"),
    (B16DensityProfile, "real"), (B16DensityProfile, "fourier"),
    (Z07GalaxyHODProfile, "real"), (Z07GalaxyHODProfile, "fourier"),
    (Z07GalaxyHODProfile, "ng_bar"), (Z07GalaxyHODProfile, "galaxy_bias"),
    (S12CIBProfile, "real"), (S12CIBProfile, "fourier"), (S12CIBProfile, "mean_emissivity"),
    (Pk, "pk_1h"), (Pk, "pk_2h"), (Pk, "pk_tot"),
    (_cl_module, "cl"), (_cl_module, "cl_linbias"),
    (Bk, "_bk_1h"), (Bk, "_bk_2h"), (Bk, "_bk_3h"), (Bk, "_bk_tot"),
    (Tk, "tk_1h"), (Tk, "tk_2h"), (Tk, "tk_3h"), (Tk, "tk_4h"), (Tk, "tk_tot"),
    (_covariance_module, "cov_g"), (_covariance_module, "cov_cng"), (_covariance_module, "cov_ssc"),
    (_covariance_module, "sigma2_b_disc"),
]


@pytest.mark.parametrize(
    "owner,attr", JITTED_API, ids=[f"{c.__name__}.{a}" for c, a in JITTED_API]
)
def test_public_api_is_jitted_internally(owner, attr):
    """The decorator must be on the library's own function, not supplied by the caller.

    A user calling hmfast should get compiled performance without ever writing jax.jit
    themselves. `bk_1h`/`bk_2h`/`bk_3h` validate their inputs eagerly and delegate to the
    jitted `_bk_*` cores listed above, which is why the core is what gets checked.
    """
    fn = inspect.getattr_static(owner, attr)
    assert isinstance(fn, jax.stages.Wrapped), (
        f"{owner.__name__}.{attr} is a plain function -- it needs a jax.jit decorator, "
        "otherwise every call pays eager per-operation dispatch"
    )


# `self` can only be traced when the profile's pytree aux data holds no array.
PROFILE_SWEEPS = [
    pytest.param(HOD, "fourier", MD_200M, dict(sigma_log10M=0.25, alpha_s=1.05), id="Z07GalaxyHODProfile"),
    pytest.param(CIB, "fourier", MD_200M, dict(L0=7e-8, beta=1.8), id="S12CIBProfile"),
    pytest.param(M21, "fourier", MD_200M, dict(eta_max=0.45, f_sub=0.15), id="M21CIBProfile"),
    pytest.param(GNFW, "real", MD_500C, dict(P0=6.8, c500=1.2), id="GNFWPressureProfile"),
    pytest.param(B12, "real", MD_200C, dict(A_P0=19.0, A_beta=4.2), id="B12PressureProfile"),
    pytest.param(B16, "real", MD_200C, dict(x_out=1.2), id="B16DensityProfile"),
]


@pytest.mark.parametrize("profile,method,mass_def,new_params", PROFILE_SWEEPS)
def test_profile_parameter_sweep_does_not_recompile(profile, method, mass_def, new_params):
    """Varying a profile's own parameters must reuse the compiled kernel.

    This is the inner loop of an MCMC over profile parameters: `update()` hands back a new
    object, so if the jitted method pins `self` static it is hashed by identity, misses the
    cache, and pays a full compile every step (measured at 85x for GNFWPressureProfile.real)
    while leaking one executable per draw.
    """
    hm = halo_model(PARAMS, mass_def)
    arg = jnp.geomspace(1e-2, 5.0, N_R) if method == "real" else K_GRID

    jax.block_until_ready(getattr(profile, method)(hm, arg, M_GRID, Z_SINGLE))
    n_compiles = inspect.getattr_static(type(profile), method)._cache_size()

    jax.block_until_ready(getattr(profile.update(**new_params), method)(hm, arg, M_GRID, Z_SINGLE))

    assert inspect.getattr_static(type(profile), method)._cache_size() == n_compiles


# Differentiability of the full pipeline: for a few key observables, the autodiff Jacobian with respect to the
# cosmological and profile parameters (all in log space) matches a central finite difference, and forward- and
# reverse-mode agree. Together these cover the emulators, mass function and bias, the analytic (NFW, HOD) and Hankel
# (GNFW) profile paths, the Limber projection, and the 3D power spectrum.

def _gnfw_tsz(e):
    return TSZ_TRACER.update(profile=GNFW.update(P0=e[5], beta=e[6], B=e[7]))


def _hod_galaxy(e):
    return GAL_TRACER.update(profile=HOD.update(alpha_s=e[5], M1_prime=e[6], sigma_log10M=e[7]))


def _cl_gy(e):
    t_g = GAL_TRACER.update(profile=HOD.update(alpha_s=e[5]))
    t_y = TSZ_TRACER.update(profile=GNFW.update(beta=e[6]))
    return cl(PK, halo_model(e[:5]), L_GRID, t_g, t_y, n_z=N_Z)


LOG_PARAMS = jnp.log(PARAMS)
GRADIENT_CASES = {
    "cl_yy": (lambda th: (lambda e: cl(PK, halo_model(e[:5]), L_GRID, _gnfw_tsz(e), _gnfw_tsz(e), n_z=N_Z))(jnp.exp(th)),
              jnp.concatenate([LOG_PARAMS, jnp.log(jnp.array([6.41, 4.13, 1.4]))])),
    "cl_gg": (lambda th: (lambda e: cl(PK, halo_model(e[:5]), L_GRID, _hod_galaxy(e), _hod_galaxy(e), n_z=N_Z))(jnp.exp(th)),
              jnp.concatenate([LOG_PARAMS, jnp.log(jnp.array([1.0, 1e13, 0.2]))])),
    "cl_gy": (lambda th: _cl_gy(jnp.exp(th)), jnp.concatenate([LOG_PARAMS, jnp.log(jnp.array([1.0, 4.13]))])),
    "cl_kgkg": (lambda th: cl(PK, halo_model(jnp.exp(th)), L_GRID, GLENS_TRACER, GLENS_TRACER, n_z=N_Z), LOG_PARAMS),
    "pk_mm": (lambda th: (lambda hm: PK.pk_1h(hm, K_GRID, Z_SINGLE, NFW) + PK.pk_2h(hm, K_GRID, Z_SINGLE, NFW))(
        halo_model(jnp.exp(th))), LOG_PARAMS),
    "cl_kgkg_nl[hmcode]": (lambda th: cl_linbias(cosmo(jnp.exp(th)), L_GRID, GLENS_TRACER, GLENS_TRACER, n_z=N_Z,
                                                 linear=False), LOG_PARAMS),
}


@functools.lru_cache(maxsize=None)
def _jacobian_errors(name, step=1e-4):
    fn, theta0 = GRADIENT_CASES[name]
    f = jax.jit(fn)
    jac_fwd = np.asarray(jax.jit(jax.jacfwd(fn))(theta0))
    jac_rev = np.asarray(jax.jit(jax.jacrev(fn))(theta0))
    eye = np.eye(theta0.shape[0])
    jac_fd = np.stack([(np.asarray(f(theta0 + step * e)) - np.asarray(f(theta0 - step * e))) / (2 * step) for e in eye], axis=-1)
    scale = np.max(np.abs(jac_fwd), axis=0)  # each parameter's largest sensitivity
    return jac_fwd, np.max(np.abs(jac_fwd - jac_fd) / scale), np.max(np.abs(jac_fwd - jac_rev) / scale)


# Central-difference step 1e-4 in ln(theta); errors are relative to each parameter's largest sensitivity. A broken
# gradient is off by O(1), so these ~5x-measured bounds leave room (measured: 1.3e-6, 2.0e-5, 1.2e-5, 1.3e-5, 2.8e-9, 2.1e-6).
GRADIENT_FD_TOL = {"cl_yy": 1e-5, "cl_gg": 1e-4, "cl_gy": 6e-5, "cl_kgkg": 6e-5, "pk_mm": 1.5e-8,
                   "cl_kgkg_nl[hmcode]": 1e-5}


# One end-to-end Jacobian through the halo-model mass integrals and one through the Limber projection; the rest run as benchmarks.
UNIT_GRADIENT_CASES = ["pk_mm", "cl_kgkg_nl[hmcode]"]


@pytest.mark.parametrize("name", UNIT_GRADIENT_CASES)
def test_gradient_matches_finite_difference(name):
    jac, fd_err, _ = _jacobian_errors(name)
    assert np.all(np.isfinite(jac)) and np.all(np.any(jac != 0, axis=0)), "a parameter has no effect or a NaN gradient"
    assert fd_err < GRADIENT_FD_TOL[name]


@pytest.mark.parametrize("name", UNIT_GRADIENT_CASES)
def test_forward_and_reverse_mode_agree(name):
    _, _, mode_err = _jacobian_errors(name)
    assert mode_err < 1e-10  # measured <= 2.0e-14


# Component gradients. Each group stacks the quantities that share a parameter vector, so one jitted reverse-mode Jacobian
# replaces a jax.grad per (quantity, parameter). Every entry is still checked against its own central difference, taken
# on the jitted function with the step rule and tolerance of the test it came from.

def _abs_step(h):
    return lambda x0: jnp.full_like(x0, h)


def _rel_step(rel, floor=0.0):
    return lambda x0: jnp.maximum(floor, rel * jnp.abs(x0))


HALO_STEP, HALO_RTOL = _abs_step(1e-3), 1e-3
COSMO_STEP, COSMO_RTOL = _rel_step(1e-5), 1e-3
PROFILE_STEP, PROFILE_RTOL = _rel_step(1e-5, floor=1e-6), 1e-2

_Z_HALF = jnp.array(0.5)
_M_GRID_40 = jnp.geomspace(1e10, 1e15, 40)
HM_200C = _shared_halo_model(
    FIXED_COSMOLOGY, mass_def=MD_200C, concentration=CONC, m_range=(_M_GRID_40[0], _M_GRID_40[-1]), n_m=_M_GRID_40.shape[0],
)
HM_DEFAULT = _shared_halo_model(FIXED_COSMOLOGY)
MASS_TRANSLATOR_200C_200M = mass_translator(MD_200C, MD_200M, CONC)
_TRACER_DNDZ = (jnp.linspace(0.0, 2.0, 20), jnp.ones(20))


def _with_H0(hm, H0):
    return hm.update(cosmology=hm.cosmology.update(H0=H0))


def _kernel_total(tracer, cosmology, z):
    """Sum of every term's weight from kernel()."""
    return sum(weight for weight, _, _ in tracer.kernel(cosmology, z))


def _hod_m_grid(hm):
    """The halo model's Gauss-Legendre mass nodes, which HOD real() normalises over."""
    logm, _ = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
    return jnp.exp(logm)


def _halo_components(x):
    c, m = FIXED_COSMOLOGY.update(H0=x[0]), jnp.array(1e13)
    return jnp.stack([
        HMF.dndlnm(c, m, _Z_HALF, MD_200C),
        BIAS.bias(c, m, _Z_HALF, MD_200C, order=1),
        CONC.c_delta(c, m, _Z_HALF, MD_200C),
        MD_200C.r_delta(c, m, _Z_HALF),
        MASS_TRANSLATOR_200C_200M(c, m, _Z_HALF),
        HM_DEFAULT.update(cosmology=c)._counter_terms(_Z_HALF)[0][0],
    ])


def _cosmology_background(x):
    c = FIXED_COSMOLOGY.update(H0=x[0])
    return jnp.stack([
        c.hubble_parameter(_Z_HALF), c.angular_diameter_distance(_Z_HALF), c.sigma8(_Z_HALF), c.growth_factor(_Z_HALF),
        c.omega_m(_Z_HALF), jnp.squeeze(c.pk(jnp.array([0.1]), _Z_HALF, linear=True)),
    ])


def _delta_c(x):
    c = FIXED_COSMOLOGY.update(H0=x[0])
    return jnp.stack([c.delta_c(_Z_HALF, prescription="EdS"), c.delta_c(_Z_HALF, prescription="NS97")])


# kSZ needs a finite z_max, and is read at a z inside velocity_dispersion's valid range.
_H0_KERNEL_TRACERS = [(CMBLensingTracer(z_source=Z_STAR), _Z_HALF), (GalaxyLensingTracer(dndz=_TRACER_DNDZ), _Z_HALF),
                      (GalaxyTracer(dndz=_TRACER_DNDZ), _Z_HALF), (kSZTracer(z_max=3.0), jnp.array(1.0))]


def _tracer_kernels_H0(x):
    c = FIXED_COSMOLOGY.update(H0=x[0], extrapolate_z=True)  # CMB lensing's source plane is beyond the background grid
    return jnp.stack([_kernel_total(tracer, c, z) for tracer, z in _H0_KERNEL_TRACERS])


def _tracer_kernels_amplitudes(x):
    s, a, edges = x[0], x[1], jnp.array([0.0, 2.0])
    return jnp.stack([
        _kernel_total(GalaxyTracer(dndz=_TRACER_DNDZ, mag_bias=(edges, jnp.array([s, s]))), FIXED_COSMOLOGY, _Z_HALF),
        _kernel_total(GalaxyLensingTracer(dndz=_TRACER_DNDZ, ia_bias=(edges, jnp.array([a, a]))), FIXED_COSMOLOGY, _Z_HALF),
    ])


def _nfw_H0(x):
    hm = _with_H0(HM_200C, x[0])
    m = jnp.array(1e13)
    return jnp.stack([NFW.real(hm, jnp.array(0.1), m, _Z_HALF), NFW.fourier(hm, jnp.array(0.5), m, _Z_HALF)])


def _hod_params(x):
    hod = Z07GalaxyHODProfile().update(**dict(zip(HOD_GRAD_PARAMS, x)))
    return jnp.stack([
        hod.ng_bar(HM_200C, _Z_HALF),
        hod.galaxy_bias(HM_200C, _Z_HALF),
        hod.real(HM_200C, jnp.array(0.1), _hod_m_grid(HM_200C), _Z_HALF).sum(),
    ])


def _real_wrt_params(profile, names):
    """profile.real at one (r, m, z) point as a function of the named parameters."""
    return lambda x: jnp.atleast_1d(
        profile.update(**dict(zip(names, x))).real(HM_200C, jnp.array(0.1), jnp.array(5e13), _Z_HALF)
    )


HOD_GRAD_PARAMS = {"sigma_log10M": 0.68, "alpha_s": 1.30, "M1_prime": 10**12.87, "M_min": 10**11.97, "M0": 1e9}
# z_p (default 1e100) keeps _phi on one branch, so its gradient is structurally zero and it is left out.
S12_GRAD_PARAMS = {"L0": 6.4e-8, "alpha": 0.36, "beta": 1.75, "gamma": 1.7, "T0": 24.4, "M_eff": 10**12.6,
                   "sigma2_LM": 0.5, "delta": 3.6, "M_min": 10**11.5, "nu": 100.0}
# s_nu is static aux data, not a leaf, so it is non-differentiable by design.
M21_GRAD_PARAMS = {"eta_max": 0.4028, "z_c": 1.5, "tau": 1.204, "f_sub": 0.134, "M_min": 10**11.5, "M_eff": 10**12.6,
                   "sigma2_LM": 0.5, "nu": 100.0}
# x_grid is static aux data and x_out only enters a boolean mask, so neither is differentiated (also for GNFW and B12).
B16_GRAD_PARAMS = {"A_rho0": 4000.0, "A_alpha": 0.88, "A_beta": 3.83, "alpha_m_rho0": 0.29, "alpha_m_alpha": -0.03,
                   "alpha_m_beta": 0.04, "alpha_z_rho0": -0.66, "alpha_z_alpha": 0.19, "alpha_z_beta": -0.025}
GNFW_GRAD_PARAMS = {"P0": 8.130, "c500": 1.156, "alpha": 1.0620, "beta": 5.4807, "gamma": 0.3292, "B": 1.4,
                    "alpha_P": 0.12, "P0_hexp": -1.0}
B12_GRAD_PARAMS = {"A_P0": 18.1, "A_xc": 0.497, "A_beta": 4.35, "alpha_m_P0": 0.154, "alpha_m_xc": -0.00865,
                   "alpha_m_beta": 0.0393, "alpha_z_P0": -0.758, "alpha_z_xc": 0.731, "alpha_z_beta": 0.415}


def _params_case(fn, params, outputs):
    return (fn, list(params.values()), list(params), outputs, PROFILE_STEP, PROFILE_RTOL)


# name: (fn, x0, input names, output names, finite-difference step rule, rtol)
COMPONENT_GRADIENT_CASES = {
    "halo_components[H0]": (_halo_components, [67.5], ["H0"],
                            ["dndlnm", "bias", "c_delta", "r_delta", "mass_translator", "counter_terms.n_min"],
                            HALO_STEP, HALO_RTOL),
    "cosmology[H0]": (_cosmology_background, [67.5], ["H0"],
                      ["hubble_parameter", "angular_diameter_distance", "sigma8", "growth_factor", "omega_m", "pk"],
                      COSMO_STEP, COSMO_RTOL),
    # A_s ~ 1e-9, so the step is relative to its magnitude.
    "cosmology[A_s]": (lambda x: jnp.atleast_1d(FIXED_COSMOLOGY.update(A_s=x[0]).sigma8(_Z_HALF)), [2.1e-9], ["A_s"],
                       ["sigma8"], COSMO_STEP, COSMO_RTOL),
    "delta_c[H0]": (_delta_c, [67.5], ["H0"], ["EdS", "NS97"], _rel_step(1e-6), COSMO_RTOL),
    "tracer_kernels[H0]": (_tracer_kernels_H0, [67.5], ["H0"],
                           ["CMBLensingTracer", "GalaxyLensingTracer", "GalaxyTracer", "kSZTracer"],
                           PROFILE_STEP, PROFILE_RTOL),
    "tracer_kernels[mag_bias,ia_bias]": (_tracer_kernels_amplitudes, [0.6, 1.0], ["mag_bias", "ia_bias"],
                                         ["GalaxyTracer", "GalaxyLensingTracer"], PROFILE_STEP, PROFILE_RTOL),
    "NFWMatterProfile[H0]": (_nfw_H0, [67.5], ["H0"], ["real", "fourier"], PROFILE_STEP, PROFILE_RTOL),
    "Z07GalaxyHODProfile": _params_case(_hod_params, HOD_GRAD_PARAMS, ["ng_bar", "galaxy_bias", "real"]),
    "S12CIBProfile": _params_case(_real_wrt_params(S12CIBProfile(nu=100), S12_GRAD_PARAMS), S12_GRAD_PARAMS, ["real"]),
    "M21CIBProfile": _params_case(_real_wrt_params(M21CIBProfile(nu=100), M21_GRAD_PARAMS), M21_GRAD_PARAMS, ["real"]),
    "B16DensityProfile": _params_case(_real_wrt_params(B16DensityProfile(), B16_GRAD_PARAMS), B16_GRAD_PARAMS, ["real"]),
    "GNFWPressureProfile": _params_case(_real_wrt_params(GNFWPressureProfile(), GNFW_GRAD_PARAMS), GNFW_GRAD_PARAMS,
                                        ["real"]),
    "B12PressureProfile": _params_case(_real_wrt_params(B12PressureProfile(), B12_GRAD_PARAMS), B12_GRAD_PARAMS, ["real"]),
}


@functools.lru_cache(maxsize=None)
def _component_jacobians(name):
    """Reverse-mode Jacobian and central-difference estimate, both of shape (n_out, n_in)."""
    fn, x0, _, _, step, _ = COMPONENT_GRADIENT_CASES[name]
    x0 = jnp.asarray(x0, dtype=float)
    f = jax.jit(fn)
    jac = np.asarray(jax.jit(jax.jacrev(fn))(x0))
    eye = np.eye(x0.shape[0])
    fd = np.stack([(np.asarray(f(x0 + h * e)) - np.asarray(f(x0 - h * e))) / (2 * h)
                   for h, e in zip(np.asarray(step(x0)), eye)], axis=-1)
    return jac, fd


@pytest.mark.parametrize("name", COMPONENT_GRADIENT_CASES)
def test_component_gradient_matches_finite_difference(name):
    _, _, inputs, outputs, _, rtol = COMPONENT_GRADIENT_CASES[name]
    jac, fd = _component_jacobians(name)
    assert jac.shape == (len(outputs), len(inputs))
    bad = [f"d {o}/d {i}: autodiff {jac[a, b]:.6e}, finite difference {fd[a, b]:.6e}"
           for a, o in enumerate(outputs) for b, i in enumerate(inputs)
           if not (np.isfinite(jac[a, b]) and np.isclose(jac[a, b], fd[a, b], rtol=rtol, atol=1e-8))]
    assert not bad, "\n".join(bad)


def test_delta_c_gradient_depends_on_prescription():
    """EdS delta_c is a cosmology-independent constant, NS97 depends on omega_m(z)."""
    jac, _ = _component_jacobians("delta_c[H0]")
    assert jac[0, 0] == 0.0
    assert jac[1, 0] != 0.0
