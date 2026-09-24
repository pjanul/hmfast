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
"""

import inspect

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax.scipy.special  # noqa: F401  (probed below for loggamma)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
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
from hmfast.stats import Bk, Pk, Tk, cl_hm, cl_lin, xi_hm, covariance_cng, covariance_ssc
from hmfast.stats import cl as _cl_module
from hmfast.stats import covariance as _covariance_module
from hmfast.tracers import (
    CIBTracer,
    CMBLensingTracer,
    GalaxyLensingTracer,
    GalaxyTracer,
    kSZTracer,
    tSZTracer,
)

# Deliberately tiny: this is about tracing, not accuracy, and every case pays a compile.
N_M, N_K, N_R, N_Z, N_L, N_KBT = 8, 8, 8, 6, 6, 4

M_GRID = jnp.geomspace(1e11, 1e15, N_M)
K_GRID = jnp.geomspace(1e-2, 5.0, N_K)
R_GRID = jnp.geomspace(1e-2, 5.0, N_R)
# sigma(R) below ~0.1 Mpc runs off the tabulated P(k) grid and returns NaN, by design.
R_GRID_SIGMA = jnp.geomspace(1.0, 20.0, N_R)
Z_GRID = jnp.geomspace(0.05, 2.0, N_Z)
Z_RANGE = (Z_GRID[0], Z_GRID[-1])
L_GRID = jnp.geomspace(20.0, 1000.0, N_L)
K_GRID_BT = jnp.geomspace(1e-2, 2.0, N_KBT)
Z_SINGLE = jnp.array([0.5])

# The traced parameter vector: (H0, omega_cdm, omega_b, A_s, n_s).
PARAMS = jnp.array([67.36, 0.12011, 0.02242, 2.1005e-9, 0.9665])

BASE_COSMO = Cosmology(
    emulator_set="lcdm:v1", H0=67.36, omega_cdm=0.12011, omega_b=0.02242,
    A_s=2.1005e-9, n_s=0.9665,
)

MD_200M = MassDefinition(200, "mean")
MD_200C = MassDefinition(200, "critical")
MD_500C = MassDefinition(500, "critical")
MD_VIR = MassDefinition("vir", "critical")

CONC = D08Concentration()
HMF = T08HaloMassFunction()
BIAS = T10HaloBias()

NFW = NFWMatterProfile()
GNFW = GNFWPressureProfile(
    x_range=(1e-5, 1e5), n_x=32, P0=6.41, c500=1.177, alpha=1.33,
    beta=4.13, gamma=0.31, B=1.4, alpha_P=0.12, P0_hexp=-1, x_out=4,
)
B12 = B12PressureProfile(x_range=(1e-5, 1e5), n_x=32)
B16 = B16DensityProfile(x_range=(1e-5, 1e5), n_x=32)
HOD = Z07GalaxyHODProfile(sigma_log10M=0.2, alpha_s=1.0, M1_prime=1e13, M_min=1e12, M0=1e12)
CIB = S12CIBProfile(nu=100.0)
M21 = M21CIBProfile(nu=100.0)

_Z_BIAS = jnp.linspace(0.0, 2.0, 5)
TSZ_TRACER = tSZTracer(profile=GNFW, z_max=2.0)
KSZ_TRACER = kSZTracer(profile=B16, z_max=2.0)
GAL_TRACER = GalaxyTracer(profile=HOD)
GAL_TRACER_BIASED = GalaxyTracer(profile=HOD, bias=(_Z_BIAS, 1.0 + 0.5 * _Z_BIAS))
GLENS_TRACER = GalaxyLensingTracer(profile=NFW)
CMBLENS_TRACER = CMBLensingTracer(profile=NFW)
CIB_TRACER = CIBTracer(profile=CIB)

PK, BK, TK = Pk(), Bk(), Tk()
PK0 = Pk(k_damp=0.0)  # damping disabled, for cases that used to pass k_damp=0.0 as a call-time kwarg
BK0 = Bk(k_damp=0.0)  # damping disabled, for cases that used to pass k_damp=0.0 as a call-time kwarg

MASS_TRANSLATOR_200M_500C = mass_translator(MD_200M, MD_500C, CONC)


def cosmo(p):
    """Cosmology rebuilt from the traced parameter vector."""
    return BASE_COSMO.update(H0=p[0], omega_cdm=p[1], omega_b=p[2], A_s=p[3], n_s=p[4])


def halo_model(p, mass_def=MD_200M):
    return HaloModel(
        cosmology=cosmo(p), mass_def=mass_def, concentration=CONC,
        halo_mass_function=HMF, halo_bias=BIAS, m_range=(M_GRID[0], M_GRID[-1]), n_m=N_M,
    )


# One entry per public entry point; `broken` marks a known gap with a strict xfail.

CASES = []


def case(name, fn):
    CASES.append(pytest.param(fn, id=name))


def broken(name, fn, reason):
    CASES.append(pytest.param(fn, id=name, marks=pytest.mark.xfail(strict=True, reason=reason)))


_P2XI_BUILT_UNDER_TRACE = (
    "stats/pk.py::_p2xi constructs mcfit.P2xi(k, ...) on every call, so inside a jit it "
    "is built under the trace; mcfit plans on concrete values. Cosmology._pk_grid() now "
    "hands it a real numpy grid, but the plan itself still has to be built once, outside "
    "any trace, for xi_hm to be traceable."
)

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
case("Cosmology.sigma2_b_disc", lambda p: cosmo(p).sigma2_b_disc(Z_GRID, f_sky=0.4))
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
     lambda p: T10HaloMassFunction().dndlnm(cosmo(p), M_GRID, Z_GRID, MD_200M))
case("TW10SubHaloMassFunction.dndlnmu",
     lambda p: TW10SubHaloMassFunction().dndlnmu(cosmo(p), M_GRID, M_GRID / 10.0))
case("JvdB14SubHaloMassFunction.dndlnmu",
     lambda p: JvdB14SubHaloMassFunction().dndlnmu(cosmo(p), M_GRID, M_GRID / 10.0))
case("T10HaloBias.bias[order=1]", lambda p: BIAS.bias(cosmo(p), M_GRID, Z_GRID, MD_200M, 1))
case("T10HaloBias.bias[order=2]", lambda p: BIAS.bias(cosmo(p), M_GRID, Z_GRID, MD_200M, 2))
case("ConstantConcentration.c_delta",
     lambda p: ConstantConcentration(5.0).c_delta(cosmo(p), M_GRID, Z_GRID, MD_200C))
case("D08Concentration.c_delta", lambda p: CONC.c_delta(cosmo(p), M_GRID, Z_GRID, MD_200C))
case("B13Concentration.c_delta",
     lambda p: B13Concentration().c_delta(cosmo(p), M_GRID, Z_GRID, MD_200C))

# --- halo model ------------------------------------------------------------------
case("HaloModel._counter_terms", lambda p: halo_model(p)._counter_terms(Z_GRID))
case("HaloModel._I", lambda p: halo_model(p)._I(NFW, K_GRID, Z_GRID, bias_order=1))

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
case("S12CIBProfile.mean_intensity", lambda p: CIB.mean_intensity(halo_model(p), Z_RANGE, N_Z))

# --- tracers ---------------------------------------------------------------------
for _name, _tracer in [
    ("tSZTracer", TSZ_TRACER),
    ("kSZTracer", KSZ_TRACER),
    ("GalaxyTracer", GAL_TRACER),
    ("GalaxyLensingTracer", GLENS_TRACER),
    ("CMBLensingTracer", CMBLENS_TRACER),
    ("CIBTracer", CIB_TRACER),
]:
    # kernel() returns (weight, der_bessel) pairs; der_bessel is a Python int, so weights only.
    case(f"{_name}.kernel",
         (lambda t: lambda p: [w for w, _ in t.kernel(cosmo(p), Z_GRID)])(_tracer))

# --- 2-point statistics ----------------------------------------------------------
case("Pk.pk_1h", lambda p: PK0.pk_1h(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("Pk.pk_2h", lambda p: PK.pk_2h(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("Pk.pk_tot", lambda p: PK0.pk_tot(halo_model(p), K_GRID, Z_SINGLE, NFW))
case("Pk.pk_tot[1h only]",
     lambda p: Pk(k_damp=0.0, include_2h=False).pk_tot(halo_model(p), K_GRID, Z_SINGLE, NFW))
broken("xi_hm", lambda p: xi_hm(PK, halo_model(p), R_GRID, Z_SINGLE, NFW),
       _P2XI_BUILT_UNDER_TRACE)
case("cl_hm[limber]",
     lambda p: cl_hm(PK, halo_model(p), GAL_TRACER, GAL_TRACER, L_GRID, Z_RANGE, N_Z))
CASES.append(pytest.param(
    lambda p: cl_hm(PK, halo_model(p), GAL_TRACER, GAL_TRACER, L_GRID, Z_RANGE, N_Z, l_limber=100.0),
    id="cl_hm[non-limber]", marks=_NEEDS_LOGGAMMA))
case("cl_hm[1h only]",
     lambda p: cl_hm(Pk(k_damp=0.0, include_2h=False), halo_model(p), GAL_TRACER, GAL_TRACER, L_GRID, Z_RANGE, N_Z))
case("cl_lin",
     lambda p: cl_lin(cosmo(p), GAL_TRACER_BIASED, GAL_TRACER_BIASED, L_GRID, Z_RANGE, N_Z))

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
case("covariance_cng",
     lambda p: covariance_cng(TK, halo_model(p), GAL_TRACER, None, None, None,
                              L_GRID[:3], L_GRID[:3], Z_RANGE, N_Z))
case("covariance_ssc",
     lambda p: covariance_ssc(halo_model(p), GAL_TRACER, None, None, None,
                              L_GRID[:3], L_GRID[:3], Z_RANGE, N_Z, f_sky=0.4))


def _leaves(out):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]


@pytest.mark.parametrize("fn", CASES)
def test_jittable(fn):
    """Runs inside jax.jit, and returns the same finite numbers as the eager call."""
    eager = _leaves(jax.block_until_ready(fn(PARAMS)))
    jitted = _leaves(jax.block_until_ready(jax.jit(fn)(PARAMS)))

    assert len(eager) == len(jitted)

    # Floor taken from the whole output: a term that cancels to zero has no scale of its own.
    scale = max((float(np.max(np.abs(x))) for x in eager if x.size), default=0.0)

    for got, want in zip(jitted, eager):
        assert np.all(np.isfinite(want)), "eager call produced non-finite values"
        assert np.all(np.isfinite(got)), "jitted call produced non-finite values"
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12 * scale)


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


@pytest.mark.xfail(
    strict=True,
    reason="Cosmology.cl caches its emulator on self; when the first call on an instance "
           "happens under a trace, the cached state carries a tracer out of it and the "
           "next eager call on that same instance raises UnexpectedTracerError",
)
def test_cl_survives_jit_before_eager():
    """Tracing must not poison a shared Cosmology for later eager use.

    Order matters here: eager-then-jit passes, jit-then-eager does not. A sampler that
    jits its likelihood and then inspects a spectrum eagerly on the same object hits
    exactly this.
    """
    obj = Cosmology(
        emulator_set="lcdm:v1", H0=67.36, omega_cdm=0.12011, omega_b=0.02242,
        A_s=2.1005e-9, n_s=0.9665,
    )
    ell = jnp.arange(2, 200)
    fn = lambda p: obj.update(H0=p[0], omega_cdm=p[1], omega_b=p[2], A_s=p[3], n_s=p[4]).cl_cmb("tt", ell)

    jax.block_until_ready(jax.jit(fn)(PARAMS))
    jax.block_until_ready(fn(PARAMS))


# Internal jitting: a user must never need to write jax.jit to get compiled performance.

# (owner class, attribute), for entry points whose cost justifies a compile.
JITTED_API = [
    (Cosmology, "hubble_parameter"), (Cosmology, "angular_diameter_distance"),
    (Cosmology, "critical_density"), (Cosmology, "omega_m"), (Cosmology, "delta_c"),
    (Cosmology, "growth_factor"), (Cosmology, "growth_rate"), (Cosmology, "sigma8"),
    (Cosmology, "sigma_m"), (Cosmology, "sigma_r"), (Cosmology, "sigma2_b_disc"),
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
    (_cl_module, "cl_hm"), (_cl_module, "cl_lin"),
    (Bk, "_bk_1h"), (Bk, "_bk_2h"), (Bk, "_bk_3h"), (Bk, "_bk_tot"),
    (Tk, "tk_1h"), (Tk, "tk_2h"), (Tk, "tk_3h"), (Tk, "tk_4h"), (Tk, "tk_tot"),
    (_covariance_module, "covariance_cng"), (_covariance_module, "covariance_ssc"),
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
