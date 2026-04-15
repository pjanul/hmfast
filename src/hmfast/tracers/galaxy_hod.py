import os
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import GalaxyHODProfile, StandardGalaxyHODProfile
from hmfast.download import get_default_data_path

# Ensure high precision for cosmological integrations
jax.config.update("jax_enable_x64", True)

@register_pytree_node_class
class GalaxyHODTracer(Tracer):
    """
    Galaxy counts tracer.
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

    def update(self, **kwargs):
        """
        Update HOD parameters.
        """
        
        new_profile = self.profile.update(**kwargs)
        return GalaxyHODTracer(profile=new_profile, dndz=self._dndz_data)


    def kernel(self, cosmology, z):
        """Return Wg_grid at requested z."""
        
        z = jnp.atleast_1d(z)
        z_g, phi_prime_g = self.dndz
    
        phi_prime_g_at_z = jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)
        H_grid = cosmology.hubble_parameter(z)
        chi_grid = cosmology.angular_diameter_distance(z) * (1.0 + z)

        return H_grid * (phi_prime_g_at_z / chi_grid**2)
