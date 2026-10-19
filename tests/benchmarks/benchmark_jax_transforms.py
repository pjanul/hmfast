"""
Compiled counterparts of tests/unit/test_jax_transforms.py.

The unit file only traces each entry point under ``jax.jit``, which is where jittability
failures surface. Here every case is compiled and its jitted numbers are checked against
the eager call, and the end-to-end gradient cases too heavy for the unit suite (halo-model
C_ell through HOD/GNFW profiles) are checked against finite
differences and between forward and reverse mode.

The all-terms trispectrum and bispectrum cases (and cov_cng, which runs the full trispectrum) re-run code that the
per-term cases already compile and check, at several times their cost; here they are replaced by the cheapest
multi-term combination, 1-halo + 4-halo (1-halo + 3-halo for Bk), which still exercises the include_* flags and the sum.
"""

import jax
import numpy as np
import pytest

from hmfast.stats import Bk, Tk, cov_cng

from .._shared import GAL_TRACER, K_GRID_BT, L_GRID, N_Z, NFW, Z_SINGLE, stats_halo_model
from ..unit.test_jax_transforms import (
    CASES,
    GRADIENT_CASES,
    GRADIENT_FD_TOL,
    PARAMS,
    UNIT_GRADIENT_CASES,
    _jacobian_errors,
)

BENCHMARK_GRADIENT_CASES = [name for name in GRADIENT_CASES if name not in UNIT_GRADIENT_CASES]

TK_1H_4H = Tk(include_2h=False, include_3h=False)
BK_1H_3H = Bk(k_damp=0.0, include_2h=False)
REPLACED_CASES = {"Tk.tk_tot", "Tk.tk_tot[2h only]", "Bk.bk_tot", "cov_cng"}
JIT_CASES = [c for c in CASES if c.id not in REPLACED_CASES] + [
    pytest.param(lambda p: TK_1H_4H.tk_tot(stats_halo_model(p), K_GRID_BT, K_GRID_BT, Z_SINGLE, NFW),
                 id="Tk.tk_tot[1h+4h]"),
    pytest.param(lambda p: BK_1H_3H.bk_tot(stats_halo_model(p), K_GRID_BT, K_GRID_BT, -0.5, Z_SINGLE, NFW),
                 id="Bk.bk_tot[1h+3h]"),
    pytest.param(lambda p: cov_cng(TK_1H_4H, stats_halo_model(p), L_GRID[:3], L_GRID[:3], GAL_TRACER, n_z=N_Z),
                 id="cov_cng[1h+4h]"),
]
assert REPLACED_CASES <= {c.id for c in CASES}, "a replaced case was renamed in the unit file"


def _leaves(out):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]


@pytest.mark.parametrize("fn", JIT_CASES)
def test_jitted_matches_eager(fn):
    """Runs inside jax.jit, and returns the same finite numbers as the eager call."""
    jitted = _leaves(jax.block_until_ready(jax.jit(fn)(PARAMS)))
    eager = _leaves(jax.block_until_ready(fn(PARAMS)))

    assert len(eager) == len(jitted)

    # Floor taken from the whole output: a term that cancels to zero has no scale of its own.
    scale = max((float(np.max(np.abs(x))) for x in eager if x.size), default=0.0)

    for got, want in zip(jitted, eager):
        assert np.all(np.isfinite(want)), "eager call produced non-finite values"
        assert np.all(np.isfinite(got)), "jitted call produced non-finite values"
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12 * scale)


@pytest.mark.parametrize("name", BENCHMARK_GRADIENT_CASES)
def test_gradient_matches_finite_difference(name):
    jac, fd_err, _ = _jacobian_errors(name)
    assert np.all(np.isfinite(jac)) and np.all(np.any(jac != 0, axis=0)), "a parameter has no effect or a NaN gradient"
    assert fd_err < GRADIENT_FD_TOL[name]


@pytest.mark.parametrize("name", BENCHMARK_GRADIENT_CASES)
def test_forward_and_reverse_mode_agree(name):
    _, _, mode_err = _jacobian_errors(name)
    assert mode_err < 1e-10  # measured <= 2.0e-14
