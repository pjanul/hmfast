import os
import jax
import jax.numpy as jnp
from jax.scipy.special import erf 
from jax.tree_util import register_pytree_node_class

from hmfast.tracers.base_tracer import BaseTracer
from hmfast.halo_model.profiles import NFWMatterProfile
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path

# Ensure high precision for cosmological integrations
jax.config.update("jax_enable_x64", True)

@register_pytree_node_class
class GalaxyHODTracer(BaseTracer):
    """
    Galaxy HOD tracer implementing central + satellite occupation.
    Refactored with individual float attributes to support JAX JIT and Grad.
    """

    def __init__(self, halo_model, profile=None, dndz=None, 
                       sigma_log10M_HOD=0.68, alpha_s_HOD=1.30, M1_prime_HOD=10**12.7, M_min_HOD=10**11.8, M0_HOD=0.0
                ):        
        # Static setup (Only runs on fresh instantiation)
        self.halo_model = halo_model
        self.profile = NFWMatterProfile() if profile is None else profile
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")

        if dndz is None:
            dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "normalised_dndz_cosmos_0.txt")
            dndz = self._load_dndz_data(dndz_path)  

        self.dndz = dndz

        # Dynamic leaves (floats)
        self.sigma_log10M_HOD, self.alpha_s_HOD, self.M1_prime_HOD, self.M_min_HOD, self.M0_HOD  = sigma_log10M_HOD, alpha_s_HOD, M1_prime_HOD, M_min_HOD, M0_HOD
       

    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._normalize_dndz(value)

    # --- JAX PyTree Registration ---

    def tree_flatten(self):
        # Dynamic leaves (JAX will track these for gradients/jit) and static metadata (changes will trigger a recompile)
        leaves = (self.sigma_log10M_HOD, self.alpha_s_HOD, self.M1_prime_HOD, self.M_min_HOD, self.M0_HOD)
        aux_data = (self.halo_model, self.profile, self._dndz_data) 
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        """
        The 'Magic' fix: This bypasses __init__ and its reloading logic.
        """
        # Create the object shell without running __init__, then manually assign aux data and leaves without reloading
        obj = cls.__new__(cls)
        obj.halo_model, obj.profile, obj._dndz_data = aux_data
        obj.sigma_log10M_HOD, obj.alpha_s_HOD, obj.M1_prime_HOD, obj.M_min_HOD, obj.M0_HOD = leaves
        
        return obj

    def update_params(self, **kwargs):
        names = ['sigma_log10M_HOD', 'alpha_s_HOD', 'M1_prime_HOD', 'M_min_HOD', 'M0_HOD']
        
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
        x = (jnp.log10(m) - jnp.log10(self.M_min_HOD)) / self.sigma_log10M_HOD
        return 0.5 * (1.0 + erf(x))

    def n_sat(self, m):
        """Mean satellite occupation."""
        pow_term = jnp.maximum((m - self.M0_HOD) / self.M1_prime_HOD, 0.0)**self.alpha_s_HOD
        return self.n_cen(m) * pow_term

    def ng_bar(self, m, z, params=None):
        """Comoving galaxy number density ng(z)."""
        params = merge_with_defaults(params)
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = self.halo_model.halo_mass_function(m, z, params=params)
        ng_val = jnp.trapezoid(dndlnm * Ntot[:, None], x=logm, axis=0)

        # HM Consistency check
        return jax.lax.cond(
            self.halo_model.hm_consistency, 
            lambda x: x + self.halo_model.counter_terms(m, z, params=params)[0] * Ntot[0], 
            lambda x: x, 
            ng_val
        )

    def galaxy_bias(self, m, z, params=None):
        """Compute the large-scale galaxy bias b_g(z)."""
        params = merge_with_defaults(params)
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = self.halo_model.halo_mass_function(m, z, params=params)
        bh = self.halo_model.halo_bias(m, z, order=1, params=params)
        ng = self.ng_bar(m, z, params=params)

        bg_num = jnp.trapezoid(dndlnm * bh * Ntot[:, None], x=logm, axis=0)
        bg_num = jax.lax.cond(
            self.halo_model.hm_consistency, 
            lambda x: x + self.halo_model.counter_terms(m, z, params=params)[1] * Ntot[0], 
            lambda x: x, 
            bg_num
        )
        return bg_num / ng

    def kernel(self, z, params=None):
        """Return Wg_grid at requested z."""
        params = merge_with_defaults(params)
        z = jnp.atleast_1d(z)
        z_g, phi_prime_g = self.dndz
    
        phi_prime_g_at_z = jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)
        H_grid = self.halo_model.emulator.hubble_parameter(z, params=params)
        chi_grid = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1.0 + z)

        return H_grid * (phi_prime_g_at_z / chi_grid**2)

    def u_k(self, k, m, z, moment=1, params=None):
        """Compute 1st or 2nd moment of the galaxy HOD tracer."""
        params = merge_with_defaults(params)
        Ns = self.n_sat(m)
        Nc = self.n_cen(m)
        ng = self.ng_bar(m, z, params=params) * (params["H0"]/100)**3

        _, u_m = self.profile.u_k_matter(self.halo_model, k, m, z, params=params)
    
        moment_funcs = [
            lambda _: (1/ng) * (Nc[None, :, None] + Ns[None, :, None] * u_m),
            lambda _: (1/ng**2) * (Ns[None, :, None]**2 * u_m**2 + 2 * Ns[None, :, None] * u_m),
        ]
    
        u_k_res = jax.lax.switch(moment - 1, moment_funcs, None)
        return k, u_k_res