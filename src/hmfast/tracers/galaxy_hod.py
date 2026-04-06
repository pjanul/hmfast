import os
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from hmfast.tracers.base_tracer import BaseTracer
from hmfast.halo_model.profiles import GalaxyHODProfile, StandardGalaxyHODProfile
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

    _required_profile_type = GalaxyHODProfile

    def __init__(self, profile=None, dndz=None):        
        super().__init__(profile=profile or StandardGalaxyHODProfile())
        
        if dndz is None:
            dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "normalised_dndz_cosmos_0.txt")
            dndz = self._load_dndz_data(dndz_path)  

        self.dndz = dndz


    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._normalize_dndz(value)

    # --- JAX PyTree Registration ---

    def tree_flatten(self):
        # The profile IS the leaf. JAX will automatically 
        # drill down into the profile's own 5 leaves.
        leaves = (self.profile, self._dndz_data) 
        return (leaves, None)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        profile, dndz_data = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._dndz_data = dndz_data
        return obj

    def update_params(self, **kwargs):
        """
        Update HOD parameters.
        """
        
        new_profile = self.profile.update_params(**kwargs)
        return GalaxyHODTracer(profile=new_profile, dndz=self._dndz_data)


    def kernel(self, emulator, z, params=None):
        """Return Wg_grid at requested z."""
        params = merge_with_defaults(params)
        z = jnp.atleast_1d(z)
        z_g, phi_prime_g = self.dndz
    
        phi_prime_g_at_z = jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)
        H_grid = emulator.hubble_parameter(z, params=params)
        chi_grid = emulator.angular_diameter_distance(z, params=params) * (1.0 + z)

        return H_grid * (phi_prime_g_at_z / chi_grid**2)
