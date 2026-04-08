import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.tree_util import register_pytree_node_class

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const
from hmfast.halo_model.profiles import PressureProfile, GNFWPressureProfile

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class tSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """

    _required_profile_type = PressureProfile
    
    def __init__(self, profile=None):
        super().__init__(profile=profile or GNFWPressureProfile())


    # --- Begin JAX PyTree Registration ---

    def tree_flatten(self):
        # The profile is the dynamic leaf
        leaves = (self.profile,)
        aux_data = None 
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        profile, = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        return obj

    def update_params(self, **kwargs):
        """
        Updates pressure profile parameters (e.g., P0, alpha, beta).
        """
        new_profile = self.profile.update_params(**kwargs)
        return tSZTracer(profile=new_profile)

    # --- End JAX PyTree Registration ---
        
    def kernel(self,emulator, z):
        
        h = emulator.H0/100 
        
        # Get electon mass in eV, Thomson cross section in cm^2, and Mpc/h in cm
        m_e = Const._m_e_ * Const._c_**2 / Const._eV_
        sigma_T = Const._sigma_T_ * 1e6
        mpc_per_h_to_cm =  Const._Mpc_over_m_ / h
        return (sigma_T / m_e) / (1+z) # Check this

