"""
Grid-convergence benchmarks for the three numerical grids of a halo-model calculation: the mass integral (n_m), the
redshift integral (n_z) and the profile's Hankel-transform grid (n_x). Each quantity is compared against a finer
reference computed in the same run, so the tests need no external code and check hmfast's numerics only.

Thresholds are ~2x the error measured when written (noted per case), so a change that degrades any grid by ~2x fails;
lower them when accuracy improves. Errors are the maximum relative error over ell (or k).

Marked `accuracy` and skipped unless selected: `pytest -m accuracy tests/benchmarks/benchmark_convergence.py`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile, Z07GalaxyHODProfile
from hmfast.stats import Pk, cl
from hmfast.tracers import GalaxyTracer, tSZTracer

from .._shared import halo_model as shared_halo_model

pytestmark = pytest.mark.accuracy

ELL = jnp.geomspace(20.0, 2000.0, 50)
K = jnp.geomspace(1e-2, 5.0, 50)
Z_RANGE, Z_PK = (0.05, 3.0), jnp.array([0.5])
PRODUCTION = dict(m=50, z=50, x=100)
REFERENCE = dict(m=200, z=200, x=800)
REFERENCE_OVERRIDES = {"Cl_gg": dict(z=800)}  # Cl_gg converges more slowly in z, so its reference is finer there
PK = Pk()


def _gnfw(n_x):
    return GNFWPressureProfile(x_range=(1e-5, 4.0), n_x=n_x, P0=6.41, c500=1.81, alpha=1.062, beta=4.13, gamma=0.3292,
                               B=1.0, alpha_P=0.12, P0_hexp=-1.0, x_out=4.0)


HOD = Z07GalaxyHODProfile(M0=0.0, M_min=1e12, sigma_log10M=0.2, alpha_s=1.0, M1_prime=10**13.3)
QUANTITIES = {
    "Cl_yy": ("m", "z", "x"),
    "Cl_gg": ("m", "z"),
    "Pk_mm": ("m",),
    "Pk_yy": ("m", "x"),
}


def _halo_model(cosmology, n_m):
    return shared_halo_model(cosmology, m_range=(1e10, 5e15), n_m=n_m)


def _evaluate(cosmology, name, m, z, x):
    hm = _halo_model(cosmology, m)
    if name == "Cl_yy":
        t = tSZTracer(profile=_gnfw(x), z_max=Z_RANGE[1])
        return np.asarray(cl(PK, hm, ELL, t, t, n_z=z))
    if name == "Cl_gg":
        t = GalaxyTracer(profile=HOD)
        return np.asarray(cl(PK, hm, ELL, t, t, n_z=z))
    p = NFWMatterProfile() if name == "Pk_mm" else _gnfw(x)
    return np.asarray(PK.pk_1h(hm, K, Z_PK, p) + PK.pk_2h(hm, K, Z_PK, p))


_CACHE = {}


def _cached(cosmology, name, m, z, x):
    key = (id(cosmology), name, m, z, x)
    if key not in _CACHE:
        _CACHE[key] = _evaluate(cosmology, name, m, z, x)
    return _CACHE[key]


def _reference(name):
    return {**REFERENCE, **REFERENCE_OVERRIDES.get(name, {})}


def grid_error(cosmology, name, **grids):
    """Max relative error of `name` at its reference grids overridden by `grids`, against the reference."""
    r = _reference(name)
    ref = _cached(cosmology, name, r["m"], r["z"], r["x"])
    g = {**r, **grids}
    return float(np.max(np.abs(_cached(cosmology, name, g["m"], g["z"], g["x"]) / ref - 1)))


# Production grids (N_m = N_z = 50, n_x = 100) meet the <= 0.1% criterion quoted in the paper
# (measured: Cl_yy 8.0e-4, Cl_gg 9.9e-5, Pk_mm 4.7e-5, Pk_yy 9.8e-4).
PRODUCTION_TOL = {"Cl_yy": 1e-3, "Cl_gg": 2e-4, "Pk_mm": 1e-4, "Pk_yy": 1e-3}


@pytest.mark.parametrize("name", QUANTITIES)
def test_production_grids_converged(fixed_cosmology, name):
    grids = {a: PRODUCTION[a] for a in QUANTITIES[name]}
    assert grid_error(fixed_cosmology, name, **grids) < PRODUCTION_TOL[name]


# The reference itself is converged: halving every grid moves it by far less than the tolerances above
# (measured: Cl_yy 2.4e-5, Cl_gg 1.3e-5, Pk_mm 1.0e-5, Pk_yy 6.5e-5).
REFERENCE_TOL = {"Cl_yy": 5e-5, "Cl_gg": 3e-5, "Pk_mm": 2e-5, "Pk_yy": 1.3e-4}


@pytest.mark.parametrize("name", QUANTITIES)
def test_reference_converged(fixed_cosmology, name):
    half = {a: _reference(name)[a] // 2 for a in QUANTITIES[name]}
    assert grid_error(fixed_cosmology, name, **half) < REFERENCE_TOL[name]


# One grid coarsened at a time, the others at the reference, so a failure names the grid that regressed.
# Each n is coarse enough that its error sits well above the reference's own.
SINGLE_AXIS_CASES = [
    ("Cl_yy", "m", 10, 7.6e-3),  # measured 3.8e-3
    ("Cl_yy", "z", 20, 1.4e-3),  # measured 7.0e-4
    ("Cl_yy", "x", 50, 6.4e-3),  # measured 3.2e-3
    ("Cl_gg", "m", 20, 4.4e-2),  # measured 2.2e-2
    ("Cl_gg", "z", 20, 2.0e-2),  # measured 9.8e-3
    ("Pk_mm", "m", 15, 3.0e-3),  # measured 1.5e-3
    ("Pk_yy", "m", 20, 1.7e-2),  # measured 8.5e-3
    ("Pk_yy", "x", 50, 8.4e-3),  # measured 4.2e-3
]


@pytest.mark.parametrize("name,axis,n,tol", SINGLE_AXIS_CASES)
def test_single_grid_convergence(fixed_cosmology, name, axis, n, tol):
    assert grid_error(fixed_cosmology, name, **{axis: n}) < tol
