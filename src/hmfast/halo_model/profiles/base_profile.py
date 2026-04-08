import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import mcfit
import functools
from jax.scipy.special import sici, erf
from jax.tree_util import register_pytree_node_class

from hmfast.download import get_default_data_path
from hmfast.defaults import merge_with_defaults
from hmfast.utils import lambertw, Const
from hmfast.halo_model.mass_definition import MassDefinition


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

        
    def u_k_hankel(self, halo_model, x, m, z):
        """
        Hankel-transform a 3D halo/tracer profile to u_ell for halo model use.
    
        Parameters
        ----------
        x : arrat like
            Radius r scaled by the scale radius x = r / r_s
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        k : array_like, optional
            k values over which the hankel transform will be evaluated. 
            If None, the transform's natural k grid will be output.
            If not None, the transform will be inteprolated to match this k
       

        Returns ell, u_ell_m
    
        """

       
        cparams = halo_model.emulator.get_all_cosmo_params()
        h = cparams['h']
       
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m_z(m_val, z_val):
            profile = jnp.squeeze(self.profile(halo_model, x, m_val, z_val))  # remove extra axes
            return profile * x**0.5 * W_x  # shape (Nx,)

        hankel_integrand = jax.vmap(jax.vmap(single_m_z, in_axes=(None, 0)), in_axes=(0, None) )(m, z)
            
        # We need u_k_native to have shape (Nx, Nm, Nz)
        k_native, u_k_native = self._hankel.transform(hankel_integrand)
        u_k_native = jnp.swapaxes(u_k_native, 2, 0)
        u_k_native = jnp.swapaxes(u_k_native, 2, 1)
 
        return k_native, u_k_native

    def u_k_matter(self, halo_model, k, m, z):
        """
        Calculate u^m(k, M, z) supporting independent dimensions for k, m, and z.
        
        Returns u_k_m with shape (N_k, N_m, N_z).
        """
       
        
        # Ensure all inputs are 1D arrays
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Get c_delta and r_delta
        c_delta = halo_model.c_delta(m, z)
        r_delta = halo_model.r_delta(m, z)
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
    