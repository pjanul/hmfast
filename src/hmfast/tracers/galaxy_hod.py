import os
import jax
import jax.numpy as jnp

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import GalaxyHODProfile, StandardGalaxyHODProfile
from hmfast.download import get_default_data_path

# Ensure high precision for cosmological integrations
jax.config.update("jax_enable_x64", True)

class GalaxyHODTracer(Tracer):
    """
    Galaxy counts tracer.

    Attributes
    ----------
    profile : GalaxyHODProfile
        Halo occupation distribution profile used to model galaxy number counts.
    dndz : tuple of jnp.ndarray
        Normalized galaxy redshift distribution stored as :math:`(z, dN/dz)`.
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
        """
        :meta private:
        """
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._normalize_dndz(value)

    # --- JAX PyTree Registration ---

    def _tree_flatten(self):
        # The profile IS the leaf. JAX will automatically 
        # drill down into the profile's own 5 leaves.
        leaves = (self.profile, self._dndz_data) 
        return (leaves, None)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, dndz_data = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._dndz_data = dndz_data
        return obj

    def update(self, profile=None, dndz=None):
        """
        Return a new GalaxyHODTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : GalaxyHODProfile, optional
            New HOD profile to use for the tracer. If None, the profile is unchanged.
        dndz : array_like, optional
            New redshift distribution (z, dN/dz). If None, the distribution is unchanged.

        Returns
        -------
        GalaxyHODTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        new_profile = profile if profile is not None else flat[0]
        new_dndz = dndz if dndz is not None else flat[1]
        return self._tree_unflatten(aux, (new_profile, new_dndz))


    def kernel(self, cosmology, z):
        """
        Compute the galaxy kernel :math:`W_g(z)` at redshift :math:`z`.
    
        The kernel is given by:
    
        .. math::
    
            W_g(z) = \\frac{H(z)}{c\\,\\chi^2(z)} \\frac{dN}{dz}
    
        where :math:`dN/dz` is the normalized redshift distribution of galaxies.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the kernel.
    
        Returns
        -------
        W_g : array_like
            Galaxy kernel evaluated at redshift(s) :math:`z`.
        """
        
        z = jnp.atleast_1d(z)
        z_g, phi_prime_g = self.dndz
    
        phi_prime_g_at_z = jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)
        H_grid = cosmology.hubble_parameter(z)
        chi_grid = cosmology.angular_diameter_distance(z) * (1.0 + z)

        return H_grid * (phi_prime_g_at_z / chi_grid**2)




jax.tree_util.register_pytree_node(
    GalaxyHODTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GalaxyHODTracer._tree_unflatten(aux_data, children)
)
