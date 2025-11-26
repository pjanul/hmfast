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
        self.params = params
        x_min, x_max, x_npoints = params['x_min'], params['x_max'], params['x_npoints']
        self.x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), x_npoints)
        self.hankel = HankelTransform(x_min=x_min, x_max=x_max, x_npoints=x_npoints, nu=0.5)

    @abstractmethod
    def get_prefactor(self, z, m):
        """Return tracer-specific prefactor(M,z)"""
        pass

    @abstractmethod
    def get_contributions(self, z, m):
        """
        Return (local_contribution, profile_contribution) arrays for each halo.
        - local_contribution: delta-function / pointlike part (typically 0 for continuous tracers)
        - profile_contribution: profile-weighted / extended part (typically 1 for continuous tracers)
        """
        pass

    @abstractmethod
    def get_hankel_integrand(self, x, z, m):
        """
        Return the radial profile (integrand for Hankel transform) evaluated on self.x_grid.
        Shape: (N_m, N_r)
        """
        pass

    def compute_u_ell(self, z, m):
        """
        Compute u_ell(M,z) using the general decomposition:
            u_ell = prefactor * [local_contribution + profile_contribution * u_k]
        """

        hankel_integrand = self.get_hankel_integrand(self.x_grid, z, m)
        k, u_k = self.hankel.transform(hankel_integrand)     # applies Hankel transform
        u_k *=  jnp.sqrt(jnp.pi / (2 * k[None, :]))

        
        local, profile = self.get_contributions(z, m)               # get contributions
        prefactor, scale_factor = self.get_prefactor(z, m)          # get prefactor
        ell = k[None, :] * scale_factor[:, None] 
        u_ell = prefactor[:, None] * (local[:, None] + profile[:, None] * u_k)
        
        return ell, u_ell
