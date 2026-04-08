import os
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.halo_model.profiles import MatterProfile, NFWMatterProfile
from hmfast.utils import Const
from hmfast.download import get_default_data_path


jax.config.update("jax_enable_x64", True)

@register_pytree_node_class
class GalaxyLensingTracer(BaseTracer):
    """
    Galaxy lensing tracer. Implements the formalism described in Kusiak et al (2023)
    Link to paper: https://arxiv.org/pdf/2203.12583

    Parameters
    ----------
    halo_model : 
        Halo model used to compute relevant quantities
    dndz :
        The redshift distribution of the galaxy population. This distribution will be normalized if it is not already done.
     
    """

    _required_profile_type = MatterProfile

    
    def __init__(self, profile=None, dndz=None):        

        super().__init__(profile=profile or NFWMatterProfile())

        if dndz is None:
            # Call _load_dndz_data from BaseTracer
            dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "nz_source_normalized_bin4.txt")
            self.dndz = self._load_dndz_data(dndz_path)
        else:
            self.dndz = dndz
            

    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._normalize_dndz(value)


    # --- Begin JAX PyTree Registration ---

    def tree_flatten(self):
        # Exactly like HOD: Profile is leaf 1, dndz array/tuple is leaf 2
        leaves = (self.profile, self._dndz_data)
        aux_data = None 
        return (leaves, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        profile, dndz_data = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._dndz_data = dndz_data
        return obj

    def update(self, **kwargs):
        """
        Updates profile parameters. 
        Passes the current dndz blob to the new instance.
        """
        new_profile = self.profile.update(**kwargs)
        return GalaxyLensingTracer(profile=new_profile, dndz=self._dndz_data)


    # --- End JAX PyTree Registration ---

    
    def I_s(self, emulator, z):
        """
        Return I_s at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        Integrates only over sources behind the lens (z_s > z).
        """
        
        
        z = jnp.atleast_1d(z)
        h = emulator.H0 / 100
        
        # Load source distribution       
        z_s, phi_prime_s = self.dndz
        
        # Angular distances
        chi_z_s = emulator.angular_diameter_distance(z_s) * (1 + z_s) 
        chi_z = emulator.angular_diameter_distance(z) * (1 + z) 
    
        # Reshape for broadcasting
        chi_z_s = chi_z_s[:, None]  # (N_s, 1)
        chi_z = chi_z[None, :]      # (1, N_z)
    
        # Lensing factor
        chi_diff = (chi_z_s - chi_z) / chi_z_s
    
        # Mask: only include sources behind the lens
        mask = (z_s[:, None] > z[None, :])  # (N_s, N_z)
        chi_diff_masked = chi_diff * mask
    
        # Integrate over z_s using trapezoid
        I_s = jnp.trapezoid(phi_prime_s[:, None] * chi_diff_masked, x=z_s, axis=0)
    
        return I_s


    def kernel(self, emulator, z):
        """
        Compute the galaxy lensing kernel W_kappa_g at redshift z.
        """
        # Merge default parameters with input
       
        cparams = emulator.get_all_cosmo_params()
        z = jnp.atleast_1d(z) # Ensure z is an array

        c_km_s = Const._c_ / 1e3  # Speed of light in km/s
       
        # Cosmological constants
        H0 = emulator.H0  # Hubble constant in km/s/Mpc
        h = H0 / 100
        Omega_m = cparams["Omega0_m"]  # Matter density parameter

        # Compute comoving distance and Hubble parameter
        chi_z = emulator.angular_diameter_distance(z) * (1 + z) * h # Comoving distance in Mpc/h
        H_z = emulator.hubble_parameter(z)   # Hubble parameter in km/s/Mpc
    
        I_s = self.I_s(emulator, z) 
    
        # Compute the CMB lensing kernel
        W_kappa_g =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            I_s 
        ) 
    
        return W_kappa_g 


       
