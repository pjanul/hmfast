import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.tree_util import register_pytree_node_class

from hmfast.tracers.base_tracer import Tracer
from hmfast.utils import Const
from hmfast.halos.profiles import DensityProfile, NFWDensityProfile, B16DensityProfile

jax.config.update("jax_enable_x64", True)

@register_pytree_node_class
class kSZTracer(Tracer):
    """
    tSZ tracer using GNFW profile.
    """
    _required_profile_type = DensityProfile
    
    def __init__(self, profile=None):
        super().__init__(profile=profile or B16DensityProfile())


    # ---------------- Start JAX PyTree Registration ---------------- #

    def tree_flatten(self):
        # The profile is the leaf. JAX will drill down into the profile's leaves.
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
        Updates density profile parameters (e.g., core radius, slopes).
        """
        new_profile = self.profile.update(**kwargs)
        # Returns a new tracer instance with the updated profile
        return kSZTracer(profile=new_profile)


    # ---------------- End JAX PyTree Registration ---------------- #

    def kernel(self, cosmology, z):
        # sigmaT / m_prot in (Mpc/h)**2/(Msun/h) which is required for kSZ
        sigma_T_over_m_p = (Const._sigma_T_ / Const._m_p_) / Const._Mpc_over_m_**2 * Const._M_sun_ * cosmology.H0 / 100
        return sigma_T_over_m_p / (1.0 + z)

        