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
from hmfast.halo_model.profiles import HaloProfile


class MatterProfile(HaloProfile):
    pass



class NFWMatterProfile(MatterProfile):
    def __init__(self):
        pass


    def u_k(self, halo_model, k, m, z, moment=1):
        """ 
        Compute either the first or second moment of the CMB lensing tracer u_ell.
        For CMB lensing:, 
            First moment:     W_k_cmb * u_ell_m
            Second moment:    W_k_cmb^2 * u_ell_m^2 
        """

        
        cparams = halo_model.emulator.get_all_cosmo_params()

        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute u_m_k from BaseTracer
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
