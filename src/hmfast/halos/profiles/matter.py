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

    The real-space mass-weighted matter profile is written as

    .. math::

        u_r(r, M, z) = \\frac{1}{\\bar{\\rho}_{m,0}} \\,
        \\frac{\\rho_s}{(r/r_s) \\left(1+r/r_s\\right)^2}
        \\tag{1}

    .. math::

        \\rho_s = \\frac{M}{4\\pi r_s^3}
        \\left[\\ln(1+c) - \\frac{c}{1+c}\\right]^{-1}
        \\tag{2}

    with :math:`r_s = r_\\Delta / c`.

    The Fourier-space mass-weighted matter profile is written as

    .. math::

        u(k, M, z)
        = \\frac{M}{\\bar{\\rho}_{m,0}}
        \\left[\\ln(1+c) - \\frac{c}{1+c}\\right]^{-1}
        \\Bigg[
        \\cos(q) \\left(\\mathrm{Ci}[(1+c)q] - \\mathrm{Ci}(q)\\right)
        + \\sin(q) \\left(\\mathrm{Si}[(1+c)q] - \\mathrm{Si}(q)\\right)
        - \\frac{\\sin(cq)}{(1+c)q}
        \\Bigg]
        \\tag{3}

    with :math:`q = k \\, r_s \\, (1+z)`.
    """
    def __init__(self):
        pass

    def u_r(self, halo_model, r, m, z):
        """
        Compute the real-space mass-weighted NFW matter profile.

        This evaluates Eqs. (1) and (2).

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, concentration relation, and halo
            radius.
        r : float or jnp.ndarray
            Physical radius or radii in the same units as :math:`r_\\Delta`.
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
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
        m_internal = m * cparams["h"]

        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        # Normalized real-space profile (unit mass)
        u_r_norm = self._u_r_matter(halo_model, r, m, z)
        # Mass-weighted profile
        return (m_internal[:, None] / rho_mean_0)[None, :, :] * u_r_norm


    def u_k(self, halo_model, k, m, z):
        """
        Compute the mass-weighted NFW matter profile in Fourier space.

        This evaluates Eq. (3).
    
        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, concentration relation, and halo
            radius.
        k : float or jnp.ndarray
            Comoving wavenumber(s).
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).
            
        Returns
        -------
            jnp.ndarray
                Fourier-space profile with shape :math:`(N_k, N_M, N_z)`.
        """

        
        cparams = halo_model.cosmology._cosmo_params()

        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        m_internal = m * cparams["h"]
    
        # Compute u_m_k from Tracer
        _, u_m = self._u_k_matter(halo_model, k, m, z) 
        
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        m_over_rho_mean = (m_internal / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        u_m *= m_over_rho_mean
    
        return u_m
