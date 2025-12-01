import jax
import jax.numpy as jnp
import functools
import mcfit
from abc import ABC, abstractmethod


class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """
    def __init__(self, x_min=1e-6, x_max=1e6, x_npoints=4096, nu=0.5):
        self._x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), num=x_npoints)
        self._hankel = mcfit.Hankel(self._x_grid, nu=nu, lowring=True, backend='jax')
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


class BaseTracer(ABC):
    """
    Abstract base class for cosmological tracers.

    """
    
    def __init__(self, params):
        """
        Initialize the radial grid and Hankel transform.
        """
   
    @abstractmethod
    def compute_u_ell(self, z, m, moment=1, params=None):
        """
        Compute u_ell(M,z). All child classes must have a version of this function implemented.
        """
        pass 

   
  