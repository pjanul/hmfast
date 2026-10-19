"""
CCL benchmarks for the radial kernels of hmfast.tracers: every term of CMBLensingTracer, GalaxyLensingTracer (shear,
intrinsic alignments), GalaxyTracer (density, RSD, magnification bias), tSZTracer and CIBTracer. kSZTracer has no CCL
equivalent and is not covered.

Galaxy kernels are compared on the interior nodes of the tabulated dN/dz, where both codes use the same values: between
nodes CCL splines it and hmfast interpolates linearly (~2% of the peak for a narrow bin), and on the end nodes CCL's
kernel is already zero.

hmfast folds each term's z-dependent prefactor (-f for RSD, the IA amplitude) into its kernel W, where CCL splits it
into a kernel and a scale-independent transfer function; CCL's term is compared as their product. Runs hmfast with
ncdm_mode="m" against a CCL cosmology with the same neutrino masses.

Requires pyccl; marked `ccl` and skipped if it is missing.

Errors are the maximum absolute difference over the kernel's peak, max |W - W_CCL| / max |W_CCL|: relative errors are
meaningless where a kernel falls to zero (the edges of dN/dz, the source plane). Thresholds are ~2x the error measured
when written (noted per test), so a ~2x degradation fails; lower them when accuracy improves.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hmfast.tracers import CIBTracer, CMBLensingTracer, GalaxyLensingTracer, GalaxyTracer, tSZTracer

from .._shared import ccl_cosmology, peak_err

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

Z_GRID = np.geomspace(1e-3, 3.0, 300)


@pytest.fixture(scope="module")
def cosmo(fixed_cosmology_m):
    return fixed_cosmology_m


@pytest.fixture(scope="module")
def cosmo_ccl(fixed_cosmology_m):
    return ccl_cosmology(fixed_cosmology_m)


def hmfast_terms(tracer, cosmo, z):
    return [np.asarray(w) for w, _, _ in tracer.kernel(cosmo, jnp.asarray(z))]


def ccl_terms(tracer_ccl, cosmo_ccl, z):
    """Each CCL term's kernel times its transfer function, at the chi(z) of `z`."""
    a = 1.0 / (1.0 + z)
    kernels = np.atleast_2d(tracer_ccl.get_kernel(pyccl.comoving_radial_distance(cosmo_ccl, a)))
    order = np.argsort(a)
    transfer = np.empty_like(kernels)
    transfer[:, order] = np.reshape(tracer_ccl.get_transfer(np.log(1e-2), a[order]), kernels.shape)
    return list(kernels * transfer)


class TestCMBLensingKernelCCL:
    # Out to the source plane z_star, which needs extrapolate_z (measured 5.1e-2). Near z_star, W ~ chi_star - chi
    # amplifies the 1e-4 error of hmfast's chi(z) extrapolation beyond z = 20; CCL agrees with CLASS to 1e-8 there.
    def test_kernel(self, cosmo, cosmo_ccl):
        cosmo_ext = cosmo.update(extrapolate_z=True)
        z_star = float(cosmo_ext.derived_parameters()["z_star"])
        z = np.geomspace(1e-3, z_star, 300)
        tracer_ccl = pyccl.CMBLensingTracer(cosmo_ccl, z_source=z_star)
        (got,), (want,) = hmfast_terms(CMBLensingTracer(z_source=z_star), cosmo_ext, z), ccl_terms(tracer_ccl, cosmo_ccl, z)
        assert peak_err(got, want) < 0.1


class TestGalaxyLensingKernelCCL:
    # Shear and NLA intrinsic-alignment terms against WeakLensingTracer with A_IA = 1 (measured 7.1e-4 / 7.5e-4).
    @pytest.mark.parametrize("term,tol", [("shear", 1.5e-3), ("ia", 1.5e-3)])
    def test_kernel_terms(self, cosmo, cosmo_ccl, term, tol):
        ia = (jnp.linspace(0.0, 4.0, 50), jnp.ones(50))
        tracer = GalaxyLensingTracer(ia_bias=ia)
        dndz = tuple(np.asarray(x) for x in tracer.dndz)
        tracer_ccl = pyccl.WeakLensingTracer(cosmo_ccl, dndz=dndz, ia_bias=tuple(np.asarray(x) for x in ia))
        got = dict(zip(["shear", "ia"], hmfast_terms(tracer, cosmo, dndz[0][1:-1])))
        want = dict(zip(["shear", "ia"], ccl_terms(tracer_ccl, cosmo_ccl, dndz[0][1:-1])))
        assert peak_err(got[term], want[term]) < tol


class TestGalaxyCountsKernelCCL:
    # Density, RSD and magnification-bias terms against NumberCountsTracer with unit bias and s = 0.2
    # (measured 1.2e-4 / 1.2e-3 / 8.1e-5).
    @pytest.mark.parametrize("term,tol", [("density", 2.5e-4), ("rsd", 2.5e-3), ("magnification", 1.7e-4)])
    def test_kernel_terms(self, cosmo, cosmo_ccl, term, tol):
        s = (jnp.linspace(0.0, 4.0, 50), jnp.full(50, 0.2))
        tracer = GalaxyTracer(mag_bias=s, has_rsd=True)
        dndz = tuple(np.asarray(x) for x in tracer.dndz)
        tracer_ccl = pyccl.NumberCountsTracer(cosmo_ccl, dndz=dndz, bias=(dndz[0], np.ones_like(dndz[0])),
                                              has_rsd=True, mag_bias=tuple(np.asarray(x) for x in s))
        got = dict(zip(["density", "magnification", "rsd"], hmfast_terms(tracer, cosmo, dndz[0][1:-1])))
        want = dict(zip(["density", "rsd", "magnification"], ccl_terms(tracer_ccl, cosmo_ccl, dndz[0][1:-1])))
        assert peak_err(got[term], want[term]) < tol


class TestTSZKernelCCL:
    # The same analytic 1/(1+z) on both sides, compared inside z_max where CCL cuts the kernel (measured 1.1e-7).
    def test_kernel(self, cosmo, cosmo_ccl):
        (got,) = hmfast_terms(tSZTracer(z_max=4.0), cosmo, Z_GRID)
        (want,) = ccl_terms(pyccl.tSZTracer(cosmo_ccl, z_max=4.0), cosmo_ccl, Z_GRID)
        assert peak_err(got, want) < 3e-7


class TestCIBKernelCCL:
    # The same analytic 1/(1+z) on both sides, compared inside z_max where CCL cuts the kernel (measured 1.5e-8).
    def test_kernel(self, cosmo, cosmo_ccl):
        (got,) = hmfast_terms(CIBTracer(z_max=4.0), cosmo, Z_GRID)
        (want,) = ccl_terms(pyccl.CIBTracer(cosmo_ccl, z_min=1e-4, z_max=4.0), cosmo_ccl, Z_GRID)
        assert peak_err(got, want) < 3e-8
