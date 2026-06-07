import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
import functools
from abc import ABC, abstractmethod
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


class HaloProfile(ABC):
    """
    Grandparent halo profile class from which all halo profile classes inherit.

    Child profile classes must implement :meth:`real` and :meth:`fourier`.
    """

    @abstractmethod
    def real(self, halo_model, r, m, z):
        pass


    @abstractmethod
    def fourier(self, halo_model, k, m, z):
        pass


    def _u_k_hankel(self, halo_model, x, r, m, z):
        """
        Hankel-transform a real-space profile sampled on a dimensionless grid.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model passed through to ``real``.
        x : array_like
            Dimensionless transform grid.
        r : jnp.ndarray
            Comoving radius grid with shape :math:`(N_x, N_m, N_z)`.
        m : float or array_like
            Halo mass(es).
        z : float or array_like
            Redshift(s).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Native Hankel wavenumbers and transformed profile values with shape
            :math:`(N_k, N_m, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        x = jnp.atleast_1d(x)
        r = jnp.asarray(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m_z(r_vals, m_val, z_val):
            profile = jnp.squeeze(self.real(halo_model, r_vals, m_val, z_val))
            return profile * x**0.5 * W_x

        hankel_integrand = jax.vmap(
            jax.vmap(single_m_z, in_axes=(1, None, 0), out_axes=0),
            in_axes=(1, 0, None), out_axes=0,
        )(r, m, z)

        k_native, u_k_native = self._hankel.transform(hankel_integrand)
        u_k_native = jnp.swapaxes(u_k_native, 2, 0)
        u_k_native = jnp.swapaxes(u_k_native, 2, 1)

        return k_native, jnp.squeeze(u_k_native)


    def _u_r_nfw(self, halo_model, r, m, z):
        """
        Calculate the normalized real-space NFW matter profile.

        This is the real-space analogue of ``_u_k_nfw`` and returns the
        unit-mass NFW profile sampled on a radial grid. The profile is
        truncated at :math:`r_\Delta`, matching the standard finite-mass NFW
        convention assumed by the Fourier-space helper.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the concentration relation and mass definition.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Normalized real-space profile with shape :math:`(N_r, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """
        r = jnp.atleast_1d(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        c_delta = jnp.reshape(
            halo_model.concentration.c_delta(
                halo_model.cosmology,
                m,
                z,
                mass_def=halo_model.mass_def,
            ),
            (len(m), len(z)),
        )
        r_delta = jnp.reshape(halo_model.mass_def.r_delta(halo_model.cosmology, m, z), (len(m), len(z)))
        r_s = r_delta * (1.0 + z[None, :]) / c_delta

        f_nfw = 1.0 / (jnp.log1p(c_delta) - c_delta / (1.0 + c_delta))
        x = r[:, None, None] / r_s[None, :, :]
        prefactor = 1.0 / (4.0 * jnp.pi * r_s**3)
        inside_halo = x <= c_delta[None, :, :]

        profile = prefactor[None, :, :] * f_nfw[None, :, :] / (x * (1.0 + x) ** 2)
        return jnp.squeeze(jnp.where(inside_halo, profile, 0.0))

    
    def _u_k_nfw(self, halo_model, k, m, z):
        """
        Calculate :math:`u^m(k, M, z)` for wavenumbers in :math:`\\mathrm{Mpc}^{-1}`
        supporting independent dimensions for ``k``, ``m``, and ``z``.

        Returns
        -------
        jnp.ndarray
            Fourier-space matter profile with shape :math:`(N_k, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """
       
        
        # Ensure all inputs are 1D arrays
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Get c_delta and r_delta
        c_delta = jnp.reshape(
            halo_model.concentration.c_delta(
                halo_model.cosmology,
                m,
                z,
                mass_def=halo_model.mass_def,
            ),
            (len(m), len(z)),
        )
        r_delta = jnp.reshape(halo_model.mass_def.r_delta(halo_model.cosmology, m, z), (len(m), len(z)))
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
    
        return k, jnp.squeeze(u_k_m)
    