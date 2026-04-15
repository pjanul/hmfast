import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import CIBProfile, S12CIBProfile
from hmfast.utils import lambertw, Const
from hmfast.download import get_default_data_path

@register_pytree_node_class
class CIBTracer(Tracer):
    """
    Cosmic infrared background tracer.
    """

    _required_profile_type = CIBProfile
    
    def __init__(self, profile=None):
        super().__init__(profile=profile or S12CIBProfile(nu=100))
        
    # --- JAX PyTree Registration ---
    def tree_flatten(self):
        # The Tracer's only dynamic component is the Profile PyTree
        leaves = (self.profile,)
        aux_data = None 
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        profile, = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        return obj

    def update(self, **kwargs):
        """
        Passes all parameter updates (including 'nu') down to the profile.
        """
        new_profile = self.profile.update(**kwargs)
        return CIBTracer(profile=new_profile)
    
    def kernel(self, cosmology, z):
        
        h = cosmology.H0 
        chi = cosmology.angular_diameter_distance(z) * (1 + z)

        # If Shang, apply the 1/(a * chi^2) factor. If Maniyar, return 1.0.
        is_shang = isinstance(self.profile, S12CIBProfile)
        s_nu_factor = jnp.where(is_shang, 1.0 / ((1.0 + z) * chi**2), 1.0)
        
        
        return s_nu_factor


  