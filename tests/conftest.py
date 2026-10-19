"""
Shared fixtures for the hmfast test suite.

Picked up automatically by both ``tests/unit/`` and ``tests/benchmarks/``
via pytest's conftest inheritance.
"""

import jax.numpy as jnp
import pytest

from hmfast.cosmology import CosmoPowerCosmology
from hmfast.halos.concentration import (
    B13Concentration,
    ConstantConcentration,
    D08Concentration,
)
from hmfast.halos.massdef import MassDefinition

from ._shared import FIXED_COSMOLOGY, halo_model as _halo_model, shared


def pytest_collection_modifyitems(config, items):
    """Skip the slow `accuracy` benchmarks unless a -m expression selects them."""
    if "accuracy" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="grid-convergence benchmark; run with -m accuracy")
    for item in items:
        if "accuracy" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixed_cosmology():
    """
    One fixed, documented cosmology used everywhere instead of random sampling,
    so tolerances (especially CCL-comparison ones) are reproducible across runs.
    """
    return FIXED_COSMOLOGY


@pytest.fixture(scope="session")
def fixed_cosmology_m(fixed_cosmology):
    """fixed_cosmology with ncdm_mode="m", the neutrino convention CCL uses; every CCL benchmark runs in it."""
    return fixed_cosmology.update(ncdm_mode="m")


@pytest.fixture(scope="session")
def out_of_bounds_cosmology():
    """
    A cosmology with n_s outside lcdm:v1's trained bounds (0.8812-1.0492),
    used to verify NaN propagation through emulator-dependent quantities.
    """
    return FIXED_COSMOLOGY.update(n_s=5.0)


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
    return shared(MassDefinition, delta, reference)


@pytest.fixture
def halo_model(fixed_cosmology, mass_def):
    """A HaloModel per mass_def, pairing each with a different concentration class."""
    conc = {200: shared(D08Concentration), "vir": shared(B13Concentration)}.get(
        mass_def.delta, shared(ConstantConcentration, c=5)
    )
    return _halo_model(
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
    A CosmoPowerCosmology for each non-lcdm emulator set, at default params. Skips cleanly if
    that set's data isn't downloaded locally, rather than assuming it always is.
    """
    try:
        return shared(CosmoPowerCosmology, request.param)
    except Exception as exc:
        pytest.skip(f"{request.param} emulator files not available locally: {exc}")
