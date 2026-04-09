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
from hmfast.utils import lambertw, Const
from hmfast.halo_model.mass_definition import MassDefinition
from hmfast.halo_model.profiles import HaloProfile


class GalaxyHODProfile(HaloProfile):
    pass



@register_pytree_node_class
class StandardGalaxyHODProfile(GalaxyHODProfile):
    """
    Galaxy HOD tracer implementing central + satellite occupation.
    Refactored with individual float attributes to support JAX JIT and Grad.
    """

    def __init__(self, sigma_log10M=0.68, alpha_s=1.30, M1_prime=10**12.7, M_min=10**11.8, M0=0.0):        
        
        self.sigma_log10M, self.alpha_s, self.M1_prime, self.M_min, self.M0  = sigma_log10M, alpha_s, M1_prime, M_min, M0

    @property
    def has_central_contribution(self):
        return True
    
  
    # --- JAX PyTree Registration ---

    def tree_flatten(self):
        # Dynamic leaves (JAX will track these for gradients/jit) and static metadata (changes will trigger a recompile)
        leaves = (self.sigma_log10M, self.alpha_s, self.M1_prime, self.M_min, self.M0)
        return (leaves, None)


    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves)


    def update(self, **kwargs):
        names = ['sigma_log10M', 'alpha_s', 'M1_prime', 'M_min', 'M0']
        
        # Block typos immediately
        if not set(kwargs).issubset(names):
            raise ValueError(f"Invalid galaxy HOD parameter(s): {set(kwargs) - set(names)}")
    
        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    # --- Physics Implementations ---

    def n_cen(self, m):
        """Mean central occupation."""
        # Using attributes directly as they are now JAX-traced leaves
        x = (jnp.log10(m) - jnp.log10(self.M_min)) / self.sigma_log10M
        return 0.5 * (1.0 + erf(x))

    def n_sat(self, m):
        """Mean satellite occupation."""
        pow_term = jnp.maximum((m - self.M0) / self.M1_prime, 0.0)**self.alpha_s
        return self.n_cen(m) * pow_term

    def ng_bar(self, halo_model, m, z):
        """Comoving galaxy number density ng(z)."""
       
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = halo_model.halo_mass_function(m, z)
        ng_val = jnp.trapezoid(dndlnm * Ntot[:, None], x=logm, axis=0)

        # HM Consistency check
        return jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z)[0] * Ntot[0], lambda x: x, ng_val)

    def galaxy_bias(self, halo_model, m, z):
        """Compute the large-scale galaxy bias b_g(z)."""
       
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = halo_model.halo_mass_function(m, z)
        bh = halo_model.halo_bias(m, z, order=1)
        ng = self.ng_bar(halo_model, m, z)

        bg_num = jnp.trapezoid(dndlnm * bh * Ntot[:, None], x=logm, axis=0)
        bg_num = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z)[1] * Ntot[0], lambda x: x, bg_num)
        return bg_num / ng


    def sat_and_cen_contribution(self, halo_model, k, m, z):
        """ 
        Compute either the first or second moment of the galaxy HOD tracer u_ell.
        For galaxy HOD:, 
            First moment:     W_g / ng_bar * [Nc + Ns * u_ell_m]
            Second moment:    W_g^2 / ng_bar^2 * [Ns^2 * u_ell_m^2 + 2 * Ns * u_ell_m]
        You cannot simply take u_ell_g**2.
        """

       
        Ns = self.n_sat(m)
        Nc = self.n_cen(m)
        ng = self.ng_bar(halo_model, m, z) * (halo_model.emulator.H0 / 100)**3

        _, u_m = self.u_k_matter(halo_model, k, m, z)  

        sat_term = (1/ng) * (Ns[None, :, None] * u_m)
        cen_term = (1/ng) * (Nc[None, :, None]**0)
    
        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1):
        """Compute 1st or 2nd moment of the galaxy HOD tracer."""
       
        Ns = self.n_sat(m)
        Nc = self.n_cen(m)
        ng = self.ng_bar(halo_model, m, z) * (halo_model.emulator.H0 / 100)**3

        _, u_m = self.u_k_matter(halo_model, k, m, z)
    
        moment_funcs = [
            
            lambda _: (1/ng) * (Nc[None, :, None] + Ns[None, :, None] * u_m),
            lambda _: (1/ng**2) * (Ns[None, :, None]**2 * u_m**2 + 2 * Ns[None, :, None] * u_m),
        ]
    
        u_k_res = jax.lax.switch(moment - 1, moment_funcs, None)
        return k, u_k_res

