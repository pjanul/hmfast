import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
import functools
from jax.scipy.special import sici
from jax.tree_util import register_pytree_node_class



class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """
    def __init__(self, x, nu=0.5):
        
        self._hankel = mcfit.Hankel(x, nu=nu, lowring=True, backend='jax')
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


class HaloProfile:

    @property
    def has_central_contribution(self):
        """ 
        Indicates whether the profile has a contribution from central terms, such as:
        
            - HOD, which has profile = N_sat * u_k + N_sat 
            - CIB, which has profile = L_sat * u_k + L_sat * L_cen

        For most profiles, profile = prefactor * u_k, meaning that this will be set to False.
        """
        return False


    def _u_r_matter(self, halo_model, r, m, z):
        """
        Calculate the normalized real-space NFW matter profile.

        This is the real-space analogue of ``_u_k_matter`` and returns the
        unit-mass NFW profile sampled on a radial grid.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the concentration relation and mass definition.
        r : float or jnp.ndarray
            Physical radius or radii in the same units as :math:`r_\\Delta`.
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Normalized real-space profile with shape :math:`(N_r, N_M, N_z)`.
        """
        r = jnp.atleast_1d(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        h = halo_model.cosmology.H0 / 100.0
        m_internal = m * h

        c_delta = halo_model.concentration.c_delta(halo_model, m_internal, z)
        r_delta = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z) * h
        r_s = r_delta / c_delta

        f_nfw = 1.0 / (jnp.log1p(c_delta) - c_delta / (1.0 + c_delta))
        x = r[:, None, None] / r_s[None, :, :]
        prefactor = 1.0 / (4.0 * jnp.pi * r_s**3)

        return prefactor[None, :, :] * f_nfw[None, :, :] / (x * (1.0 + x) ** 2)

    
    def _u_k_matter(self, halo_model, k, m, z):
        """
        Calculate u^m(k, M, z) supporting independent dimensions for k, m, and z.
        
        Returns u_k_m with shape (N_k, N_m, N_z).
        """
       
        
        # Ensure all inputs are 1D arrays
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = halo_model.cosmology.H0 / 100.0
        m_internal = m * h
        
        # Get c_delta and r_delta
        c_delta = halo_model.concentration.c_delta(halo_model, m_internal, z)
        r_delta = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z) * h
        lambda_val = 1.0 
        
        # Compute analytical profile q terms with shape: (N_k, N_m, N_z)
        q = k[:, None, None] * r_delta[None, :, :] / c_delta[None, :, :] * (1 + z[None, None, :])
        q_scaled = (1 + lambda_val * c_delta[None, :, :]) * q
        
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        
        # NFW normalization
        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x / (1 + x))
        f_nfw_val = f_nfw(lambda_val * c_delta)
        f_nfw_val = f_nfw_val[None, :, :]  
        
        # Fourier-space profile calculation
        u_k_m = (jnp.cos(q) * (Ci_q_scaled - Ci_q)
                   + jnp.sin(q) * (Si_q_scaled - Si_q)
                   - jnp.sin(lambda_val * c_delta[None,:,:] * q) / q_scaled) * f_nfw_val 
    
        return k, u_k_m
    