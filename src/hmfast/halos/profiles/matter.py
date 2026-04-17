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


    def u_k(self, halo_model, k, m, z, moment=1):
        """
        Compute the first or second moment of the mass-weighted NFW matter
        profile in Fourier space.
    
        The returned quantity is
    
        .. math::
    
            u(k, M, z) = \\frac{M}{\\bar{\\rho}_{m,0}} \\, u^m(k, M, z),
    
        where :math:`u^m(k, M, z)` is the normalized analytic Fourier transform of the NFW
        density profile.
        
        For ``moment=1``, this method returns :math:`u(k, M, z)`. For
        ``moment=2``, it returns :math:`u^2(k, M, z)`.
    
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
        moment : int, optional
            Moment of the profile to return. Supported values are ``1`` and ``2``.
    
        Returns
        -------
        tuple
            :math:`(k, u)``, where :math:`u` has shape :math:`(N_k, N_M, N_z)`.
        """

        
        cparams = halo_model.cosmology._cosmo_params()

        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute u_m_k from Tracer
        k, u_m = self.u_k_matter(halo_model, k, m, z) 
        
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"]
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _:   u_m ,
            lambda _:   u_m**2,
        ]
    
        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k
