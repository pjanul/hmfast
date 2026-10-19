"""
Shared instances and tiny inputs for the test suite.

Cosmology, MassDefinition and the halo-model components (mass function, bias, concentration,
subhalo mass function) are hashed by identity in jit caches, so two value-identical instances
compile separately. Building them through `shared` gives every test the same object and lets
compiled kernels be reused across tests.
"""

import functools

import jax.numpy as jnp
import numpy as np

from hmfast.cosmology import CosmoPowerCosmology
from hmfast.cosmology.cosmopower import _EMULATOR_SETS, _Z_BG
from hmfast.halos import HaloModel
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import D08Concentration
from hmfast.halos.massdef import MassDefinition
from hmfast.halos.massfunc import T08HaloMassFunction, TW10SubHaloMassFunction
from hmfast.halos.profiles import (
    B12PressureProfile,
    B16DensityProfile,
    GNFWPressureProfile,
    M21CIBProfile,
    NFWMatterProfile,
    S12CIBProfile,
    Z07GalaxyHODProfile,
)
from hmfast.stats import Bk, Pk, Tk
from hmfast.tracers import (
    CIBTracer,
    CMBLensingTracer,
    GalaxyLensingTracer,
    GalaxyTracer,
    kSZTracer,
    tSZTracer,
)


@functools.cache
def shared(cls, *args, **kwargs):
    """One instance per (class, arguments); arguments must be hashable."""
    return cls(*args, **kwargs)


# Built once; other cosmologies are derived with .update() so they share its emulator objects.
FIXED_COSMOLOGY = CosmoPowerCosmology(
    "lcdm:v1",
    H0=67.5,
    omega_cdm=0.12,
    omega_b=0.022,
    A_s=2.1e-9,
    n_s=0.965,
)


def emulator_z_max_bg(cosmology):
    """Top of a CosmoPowerCosmology's emulated background."""
    return float(_Z_BG[-1])


def emulator_z_max_pk(cosmology):
    """Top of a CosmoPowerCosmology's emulated P(k) and growth."""
    return float(_EMULATOR_SETS[cosmology.emulator_set]["z_max_pk"])


def emulator_z_grid_pk(cosmology):
    """100 redshifts spanning a CosmoPowerCosmology's emulated P(k)."""
    return jnp.linspace(0.0, emulator_z_max_pk(cosmology), 100)


def halo_model(cosmology=FIXED_COSMOLOGY, **kwargs):
    """HaloModel whose unspecified components are the shared defaults rather than fresh instances."""
    kwargs.setdefault("mass_def", shared(MassDefinition, 200, "critical"))
    kwargs.setdefault("halo_mass_function", shared(T08HaloMassFunction))
    kwargs.setdefault("halo_bias", shared(T10HaloBias))
    kwargs.setdefault("subhalo_mass_function", shared(TW10SubHaloMassFunction))
    kwargs.setdefault("concentration", shared(D08Concentration))
    return HaloModel(cosmology=cosmology, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# Tiny stats inputs, shared by the stats unit tests and test_jax_transforms.py so identical calls compile once.
# ---------------------------------------------------------------------------------------------------------------------

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
# FFTLog transforms plan on concrete, log-uniform input grids.
K_FFT = np.geomspace(1e-3, 10.0, 32)
L_FFT = np.geomspace(10.0, 1e4, 32)
THETA_GRID = jnp.geomspace(0.05, 1.0, N_R)

# The cosmology the stats inputs are evaluated at, as the vector (H0, omega_cdm, omega_b, A_s, n_s).
PARAMS = jnp.array([67.36, 0.12011, 0.02242, 2.1005e-9, 0.9665])
BASE_COSMO = FIXED_COSMOLOGY.update(H0=67.36, omega_cdm=0.12011, omega_b=0.02242, A_s=2.1005e-9, n_s=0.9665)

MD_200M = shared(MassDefinition, 200, "mean")
MD_200C = shared(MassDefinition, 200, "critical")
MD_500C = shared(MassDefinition, 500, "critical")
MD_VIR = shared(MassDefinition, "vir", "critical")

CONC = shared(D08Concentration)
HMF = shared(T08HaloMassFunction)
BIAS = shared(T10HaloBias)

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

Z_BIAS = jnp.linspace(0.0, 2.0, 5)
TSZ_TRACER = tSZTracer(profile=GNFW, z_max=2.0)
KSZ_TRACER = kSZTracer(profile=B16, z_max=2.0)
GAL_TRACER = GalaxyTracer(profile=HOD)
GAL_TRACER_BIASED = GalaxyTracer(profile=HOD, bias=(Z_BIAS, 1.0 + 0.5 * Z_BIAS))
GLENS_TRACER = GalaxyLensingTracer(profile=NFW)
# Source plane at last scattering; chi(z_star) lies beyond the emulated background, so kernels there need extrapolate_z.
Z_STAR = float(FIXED_COSMOLOGY.derived_parameters()["z_star"])
CMBLENS_TRACER = CMBLensingTracer(profile=NFW, z_source=Z_STAR)
CIB_TRACER = CIBTracer(profile=CIB)

PK, BK, TK = Pk(), Bk(), Tk()


def cosmo(p):
    """Cosmology rebuilt from a parameter vector (H0, omega_cdm, omega_b, A_s, n_s)."""
    return BASE_COSMO.update(H0=p[0], omega_cdm=p[1], omega_b=p[2], A_s=p[3], n_s=p[4])


def stats_halo_model(p=PARAMS, mass_def=MD_200M):
    """The tiny HaloModel every stats test runs on."""
    return halo_model(
        cosmology=cosmo(p), mass_def=mass_def, concentration=CONC,
        halo_mass_function=HMF, halo_bias=BIAS, m_range=(M_GRID[0], M_GRID[-1]), n_m=N_M,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Benchmark helpers: CCL cosmologies matched to an hmfast Cosmology, and the error metrics benchmarks assert on.
# ---------------------------------------------------------------------------------------------------------------------


def ccl_cosmology(cosmology, *, pk_linear=False, pk_nonlin=False, **kwargs):
    """
    pyccl cosmology with `cosmology`'s neutrino masses, w0 and N_eff. With `pk_linear=True` it is a CosmologyCalculator
    fed hmfast's own P_lin on the emulator's (k, z) nodes, so halo-model comparisons test formulas rather than
    Boltzmann codes, and with `pk_nonlin=True` also hmfast's nonlinear P(k); otherwise CCL runs CLASS itself. Extra
    keyword arguments go to the pyccl constructor.
    """
    import pyccl

    h = cosmology.H0 / 100.0
    m_ncdm = float(cosmology.m_ncdm)
    m_nu = [m_ncdm] * 3 if cosmology.emulator_set == "mnu-3states:v1" else [0.0, 0.0, m_ncdm]
    params = dict(
        Omega_c=float(cosmology.omega_cdm / h**2), Omega_b=float(cosmology.omega_b / h**2), h=float(h),
        A_s=float(cosmology.A_s), n_s=float(cosmology.n_s), m_nu=m_nu, mass_split="list",
        w0=float(cosmology.w0), Neff=float(cosmology.derived_parameters()["Neff"]), **kwargs,
    )
    if not pk_linear:
        return pyccl.Cosmology(**{"transfer_function": "boltzmann_class", **params})
    k, z = cosmology._pk_grid(), emulator_z_grid_pk(cosmology)
    a = np.asarray(1.0 / (1.0 + z))[::-1]

    def table(linear):
        pk = np.asarray(cosmology.pk(k, z, linear=linear)).T[::-1]  # (N_a, N_k) with a increasing, as CCL expects
        return {"a": a, "k": np.asarray(k), "delta_matter:delta_matter": pk}

    if pk_nonlin:
        params["pk_nonlin"] = table(False)
    return pyccl.CosmologyCalculator(pk_linear=table(True), **params)


def ccl_profile2pt_hod():
    """CCL Profile2pt for an HOD with satellite pairs N_c^2 N_s^2 (hmfast); CCL's Profile2ptHOD has N_c N_s^2."""
    import pyccl

    class Profile2ptHODhmfast(pyccl.halos.Profile2pt):
        def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True):
            M_use, k_use = np.atleast_1d(M), np.atleast_1d(k)
            n_cen, n_sat = prof._Nc(M_use, a)[:, None], prof._Ns(M_use, a)[:, None]
            sat = n_cen * n_sat * prof._usat_fourier(cosmo, k_use, M_use, a)
            out = 2 * sat + sat**2
            if np.ndim(k) == 0:
                out = np.squeeze(out, axis=-1)
            if np.ndim(M) == 0:
                out = np.squeeze(out, axis=0)
            return out

    return Profile2ptHODhmfast()


def rel_err(got, want):
    """Maximum relative error |got / want - 1| over the entries where `want` is non-zero; NaN if `got` is not finite."""
    got, want = np.broadcast_arrays(np.asarray(got, dtype=float), np.asarray(want, dtype=float))
    mask = want != 0
    return float(np.max(np.abs(got[mask] / want[mask] - 1.0)))


def peak_err(got, want):
    """Maximum absolute error over max |want|, for quantities that cross zero or have negligible tails."""
    got, want = np.broadcast_arrays(np.asarray(got, dtype=float), np.asarray(want, dtype=float))
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
