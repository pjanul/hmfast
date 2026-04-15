import os
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici, erf 
from jax.tree_util import register_pytree_node_class

from hmfast.tracers.base_tracer import Tracer
from hmfast.download import get_default_data_path
from hmfast.utils import Const
from hmfast.halos.profiles import MatterProfile, NFWMatterProfile
jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class CMBLensingTracer(Tracer):
    """
    CMB weak lensing tracer.
    """

    _required_profile_type = MatterProfile

    def __init__(self, profile=None):        
        super().__init__(profile=profile or NFWMatterProfile())

    def tree_flatten(self):
        # We treat the profile as the only leaf. 
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
        Since this tracer has no params of its own, it just 
        passes the request to the profile.
        """
        new_profile = self.profile.update(**kwargs)
        return CMBLensingTracer(profile=new_profile)


    def kernel(self, cosmology, z):
        """
        Compute the CMB lensing kernel W_kappa_cmb at redshift z.
        """
        # Merge default parameters with input
        
        cparams = cosmology.get_all_cosmo_params()
        z = jnp.atleast_1d(z)  # Ensure z is an array
        
        # Cosmological constants
        H0 = cosmology.H0    # Hubble constant in km/s/Mpc
        Omega_m = cparams["Omega0_m"]  # Matter density parameter
        c_km_s = Const._c_ / 1e3  # Speed of light in km/s        
        h = H0 / 100
        
        # Compute comoving distance and Hubble parameter
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z) * h # Comoving distance in Mpc/h
        H_z = cosmology.hubble_parameter(z)   # Hubble parameter in km/s/Mpc
        
        # Comoving distance to the last scattering surface (z ~ 1090) in Mpc/h
        chi_z_cmb = cosmology.derived_parameters()["chi_star"] * h  
        
        # Compute the CMB lensing kernel
        W_kappa_cmb =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            ((chi_z_cmb - chi_z) / chi_z_cmb)
        )    

       
        return W_kappa_cmb 

