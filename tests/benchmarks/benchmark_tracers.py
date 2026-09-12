"""
Concentrated kernel-vs-CCL benchmarks for hmfast.tracers, covering the 5 tracers that have a
CCL equivalent (CMBLensingTracer, GalaxyLensingTracer, GalaxyTracer, tSZTracer, CIBTracer).
kSZTracer has no CCL equivalent and is excluded -- no ground truth exists for it.

Requires pyccl; the whole module is skipped if it isn't installed (marked `ccl`, see
pyproject.toml), so `pytest -m "not ccl"` skips this file without needing pyccl at all.

Known correction: lcdm:v1 was trained assuming a single massive neutrino of 0.06 eV. Without
`m_nu=[0, 0, 0.06]` on the CCL side, kernel comparisons show a small but real, systematic
~0.3-0.4% median residual; with it, median residuals drop below 0.01%. Every CCL cosmology
built here includes this correction.

Tolerances are asserted on the *median* of the residual distribution, not a full-array
allclose: each kernel's relative residual only blows up in the far tail (below ~1% of its peak
amplitude, where dividing by a near-zero CCL value amplifies the relative error even though the
absolute agreement is fine) -- that tail is masked out before computing the median, matching
this project's established "median is the reliable metric" convention for tail-sensitive
comparisons (see benchmark_profiles.py).
"""

import jax.numpy as jnp
import numpy as np
import pytest

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

from hmfast.cosmology import Cosmology
from hmfast.tracers import (
    CIBTracer,
    CMBLensingTracer,
    GalaxyLensingTracer,
    GalaxyTracer,
    tSZTracer,
)


def _ccl_cosmology(cosmology):
    h = cosmology.H0 / 100.0
    return pyccl.Cosmology(
        Omega_c=cosmology.omega_cdm / h**2,
        Omega_b=cosmology.omega_b / h**2,
        h=h,
        A_s=cosmology.A_s,
        n_s=cosmology.n_s,
        m_nu=[0, 0, 0.06],
        mass_split="list",
    )


def _kernel_scalar(tracer, cosmology, z):
    """Sum every term's weight from kernel() into one array, matching CCL's own
    get_kernel() (which has no notion of a per-term breakdown)."""
    weights = [weight for weight, _ in tracer.kernel(cosmology, z)]
    total = weights[0]
    for w in weights[1:]:
        total = total + w
    return total


def _kernel_median_residual(K_hm, K_ccl, threshold=0.01):
    """Normalize both kernels by their own peak, mask the near-zero tail, return the median % residual."""
    K_hm = np.where(np.isfinite(K_hm), K_hm, 0.0)
    K_hm_n = K_hm / (np.max(np.abs(K_hm)) or 1.0)
    K_ccl_n = K_ccl / (np.max(np.abs(K_ccl)) or 1.0)
    mask = np.abs(K_ccl_n) > threshold
    res = 100.0 * (K_hm_n[mask] - K_ccl_n[mask]) / K_ccl_n[mask]
    return np.median(np.abs(res))


@pytest.fixture(scope="session")
def cosmo_ccl_bg(fixed_cosmology):
    return _ccl_cosmology(fixed_cosmology)


class TestCMBLensingKernelCCL:
    # kernel() matches CCL's CMBLensingTracer.get_kernel() out to z_star (median resid <1%,
    # observed ~0.36%). Needs an extrapolated hmfast cosmology since CMB lensing's kernel only
    # vanishes at z_star (~1086), far beyond the emulator's normally-trained z range.
    def test_kernel_matches_ccl(self, fixed_cosmology):
        cosmo_ext = Cosmology(
            emulator_set="lcdm:v1",
            H0=fixed_cosmology.H0,
            omega_cdm=fixed_cosmology.omega_cdm,
            omega_b=fixed_cosmology.omega_b,
            A_s=fixed_cosmology.A_s,
            n_s=fixed_cosmology.n_s,
            extrapolate_z=True,
        )
        cosmo_ccl = _ccl_cosmology(cosmo_ext)
        z_star = float(cosmo_ext.derived_parameters()["z_star"])

        tracer = CMBLensingTracer()
        tracer_ccl = pyccl.tracers.CMBLensingTracer(cosmo_ccl, z_source=z_star)

        z_grid = np.logspace(np.log10(0.001), np.log10(z_star), 200)
        chi_grid = pyccl.comoving_radial_distance(cosmo_ccl, 1.0 / (1.0 + z_grid))

        K_hm = np.asarray(_kernel_scalar(tracer, cosmo_ext, jnp.asarray(z_grid)))
        K_ccl = np.asarray(tracer_ccl.get_kernel(chi_grid)).squeeze()
        assert _kernel_median_residual(K_hm, K_ccl) < 1.0


class TestGalaxyLensingKernelCCL:
    # kernel() matches CCL's WeakLensingTracer.get_kernel() (median resid <1%, observed ~0.03%).
    def test_kernel_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg):
        try:
            tracer = GalaxyLensingTracer()
        except Exception as exc:
            pytest.skip(
                f"default GalaxyLensingTracer dndz file not available locally: {exc}"
            )
        z_s, nz_s = np.asarray(tracer.dndz[0]), np.asarray(tracer.dndz[1])
        tracer_ccl = pyccl.tracers.WeakLensingTracer(
            cosmo_ccl_bg, dndz=(z_s, nz_s), ia_bias=None
        )

        z_grid = np.logspace(np.log10(0.001), np.log10(3.0), 200)
        chi_grid = pyccl.comoving_radial_distance(cosmo_ccl_bg, 1.0 / (1.0 + z_grid))

        K_hm = np.asarray(_kernel_scalar(tracer, fixed_cosmology, jnp.asarray(z_grid)))
        K_ccl = np.asarray(tracer_ccl.get_kernel(chi_grid)).squeeze()
        assert _kernel_median_residual(K_hm, K_ccl) < 1.0


class TestGalaxyCountsKernelCCL:
    # kernel() matches CCL's NumberCountsTracer.get_kernel() (bias=1, to isolate kernel shape
    # from bias) at median resid <1% (observed ~0.05%).
    def test_kernel_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg):
        try:
            tracer = GalaxyTracer()
        except Exception as exc:
            pytest.skip(f"default GalaxyTracer dndz file not available locally: {exc}")
        z_c, nz_c = np.asarray(tracer.dndz[0]), np.asarray(tracer.dndz[1])
        tracer_ccl = pyccl.tracers.NumberCountsTracer(
            cosmo_ccl_bg,
            dndz=(z_c, nz_c),
            bias=(z_c, np.ones_like(z_c)),
            has_rsd=False,
        )

        z_grid = np.logspace(np.log10(0.001), np.log10(3.0), 200)
        chi_grid = pyccl.comoving_radial_distance(cosmo_ccl_bg, 1.0 / (1.0 + z_grid))

        K_hm = np.asarray(_kernel_scalar(tracer, fixed_cosmology, jnp.asarray(z_grid)))
        K_ccl = np.asarray(tracer_ccl.get_kernel(chi_grid)).squeeze()
        assert _kernel_median_residual(K_hm, K_ccl) < 1.0


class TestTSZKernelCCL:
    # kernel() matches CCL's tSZTracer.get_kernel() essentially exactly (median resid <0.01%,
    # observed ~6e-7%) -- both sides are the same purely-analytic 1/(1+z) formula.
    def test_kernel_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg):
        tracer = tSZTracer(z_max=3.0)
        tracer_ccl = pyccl.tracers.tSZTracer(cosmo_ccl_bg, z_max=3.0)

        z_grid = np.logspace(np.log10(0.001), np.log10(3.0), 200)
        chi_grid = pyccl.comoving_radial_distance(cosmo_ccl_bg, 1.0 / (1.0 + z_grid))

        K_hm = np.asarray(_kernel_scalar(tracer, fixed_cosmology, jnp.asarray(z_grid)))
        K_ccl = np.asarray(tracer_ccl.get_kernel(chi_grid)).squeeze()
        assert _kernel_median_residual(K_hm, K_ccl) < 0.01


class TestCIBKernelCCL:
    # kernel() matches CCL's CIBTracer.get_kernel() essentially exactly (median resid <0.01%,
    # observed ~6e-7%) -- both sides are the same purely-analytic 1/(1+z) formula.
    def test_kernel_matches_ccl(self, fixed_cosmology, cosmo_ccl_bg):
        tracer = CIBTracer(z_max=3.0)
        tracer_ccl = pyccl.tracers.CIBTracer(cosmo_ccl_bg, z_min=1e-4, z_max=3.0)

        z_grid = np.logspace(np.log10(0.001), np.log10(3.0), 200)
        chi_grid = pyccl.comoving_radial_distance(cosmo_ccl_bg, 1.0 / (1.0 + z_grid))

        K_hm = np.asarray(_kernel_scalar(tracer, fixed_cosmology, jnp.asarray(z_grid)))
        K_ccl = np.asarray(tracer_ccl.get_kernel(chi_grid)).squeeze()
        assert _kernel_median_residual(K_hm, K_ccl) < 0.01
