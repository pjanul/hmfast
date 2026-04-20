import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from hmfast.download import get_default_data_path
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import HaloProfile


class MatterProfile(HaloProfile):
    pass



class NFWMatterProfile(MatterProfile):
    """
    Matter density profile from `Navarro, Frenk & White (1997) <https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract>`_.
    """
    def __init__(self):
        pass

    def u_r(self, halo_model, r, m, z):
        """
        Compute the real-space mass-weighted NFW matter profile.

        The returned quantity is the real-space counterpart of ``u_k``:

        .. math::

            u_r(r, M, z) = \\frac{\\rho_{\\mathrm{NFW}}(r, M, z)}{\\bar{\\rho}_{m,0}}
            = \\frac{M}{\\bar{\\rho}_{m,0}} \\, u_r^m(r, M, z),

        where

        .. math::

            u_r^m(r, M, z)
            = \\frac{1}{4\\pi r_s^3}
            \\left[\\ln(1+c) - \\frac{c}{1+c}\\right]^{-1}
            \\frac{1}{x(1+x)^2},

        with :math:`x = r / r_s` and :math:`r_s = r_\\Delta / c`.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, concentration relation, and halo
            radius.
        r : float or jnp.ndarray
            Physical radius or radii in the same units as :math:`r_\\Delta`.
        m : float or jnp.ndarray
            Halo mass(es).
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Real-space profile with shape :math:`(N_r, N_M, N_z)`.
        """
        cparams = halo_model.cosmology._cosmo_params()

        r = jnp.atleast_1d(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        # Normalized real-space profile (unit mass)
        u_r_norm = self._u_r_matter(halo_model, r, m, z)
        # Mass-weighted profile
        return (m[:, None] / rho_mean_0)[None, :, :] * u_r_norm


    def u_k(self, halo_model, k, m, z):
        """
        Compute the mass-weighted NFW matter profile in Fourier space.
    
        The returned quantity is
    
        .. math::
    
            u(k, M, z) = \\frac{M}{\\bar{\\rho}_{m,0}} \\, u^m(k, M, z),
    
        where :math:`u^m(k, M, z)` is the normalized analytic Fourier transform of the NFW
        density profile.
    
        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, concentration relation, and halo
            radius.
        k : float or jnp.ndarray
            Comoving wavenumber(s).
        m : float or jnp.ndarray
            Halo mass(es).
        z : float or jnp.ndarray
            Redshift(s).
        Returns
        -------
        tuple
            :math:`(k, u)``, where :math:`u` has shape :math:`(N_k, N_M, N_z)`.
        """

        
        cparams = halo_model.cosmology._cosmo_params()

        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute u_m_k from Tracer
        k, u_m = self._u_k_matter(halo_model, k, m, z) 
        
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        u_m *= m_over_rho_mean
    
        return k, u_m
