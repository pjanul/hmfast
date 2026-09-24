"""
Shared fixtures for the hmfast test suite.

Picked up automatically by both ``tests/unit/`` and ``tests/benchmarks/``
via pytest's conftest inheritance.
"""

import jax.numpy as jnp
import pytest

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.concentration import (
    B13Concentration,
    ConstantConcentration,
    D08Concentration,
)
from hmfast.halos.massdef import MassDefinition


@pytest.fixture(scope="session")
def fixed_cosmology():
    """
    One fixed, documented cosmology used everywhere instead of random sampling,
    so tolerances (especially CCL-comparison ones) are reproducible across runs.
    """
    return Cosmology(
        emulator_set="lcdm:v1",
        H0=67.5,
        omega_cdm=0.12,
        omega_b=0.022,
        A_s=2.1e-9,
        n_s=0.965,
    )


@pytest.fixture(scope="session")
def out_of_bounds_cosmology():
    """
    A cosmology with n_s outside lcdm:v1's trained bounds (0.8812-1.0492),
    used to verify NaN propagation through emulator-dependent quantities.
    """
    return Cosmology(
        emulator_set="lcdm:v1",
        H0=67.5,
        omega_cdm=0.12,
        omega_b=0.022,
        A_s=2.1e-9,
        n_s=5.0,
    )


@pytest.fixture
def m_grid():
    """Mass grid in physical Msun, mirroring compare_ccl.ipynb's range (trimmed for test speed)."""
    return jnp.geomspace(1e10, 1e16, 50)


@pytest.fixture
def z_grid():
    """Redshift grid mirroring compare_ccl.ipynb's range (trimmed for test speed)."""
    return jnp.array([0.0, 0.5, 1.0, 2.0])


@pytest.fixture(
    params=[
        (200, "critical"),
        (200, "mean"),
        (500, "critical"),
        ("vir", "critical"),
    ]
)
def mass_def(request):
    """Parametrized across the four mass definitions compare_ccl.ipynb benchmarks (200c/200m/500c/vir)."""
    delta, reference = request.param
    return MassDefinition(delta=delta, reference=reference)


@pytest.fixture
def halo_model(fixed_cosmology, mass_def):
    """A HaloModel per mass_def, pairing each with a different concentration class."""
    conc = {200: D08Concentration(), "vir": B13Concentration()}.get(
        mass_def.delta, ConstantConcentration(c=5)
    )
    return HaloModel(
        cosmology=fixed_cosmology,
        mass_def=mass_def,
        concentration=conc,
        m_range=(1e10, 1e15),
        n_m=40,
    )


@pytest.fixture(
    scope="session",
    params=[
        "mnu:v1",
        "neff:v1",
        "wcdm:v1",
        "ede:v1",
        "ede:v2",
        "mnu-3states:v1",
    ],
)
def other_emulator_cosmology(request):
    """
    A Cosmology for each non-lcdm emulator set, at default params. Skips cleanly if
    that set's data isn't downloaded locally, rather than assuming it always is.
    """
    try:
        return Cosmology(emulator_set=request.param)
    except Exception as exc:
        pytest.skip(f"{request.param} emulator files not available locally: {exc}")
