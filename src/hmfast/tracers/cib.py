import os
import numpy as np
import jax
import jax.numpy as jnp

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import CIBProfile, S12CIBProfile
from hmfast.utils import lambertw, Const
from hmfast.download import _get_default_data_path

class CIBTracer(Tracer):
    """
    Cosmic infrared background tracer.

    Attributes
    ----------
    profile : CIBProfile
        Infrared emissivity profile used to model the cosmic infrared background signal.
    """

    _required_profile_type = CIBProfile
    
    def __init__(self, profile=None):
        super().__init__(profile=profile or S12CIBProfile(nu=100))
        
    # --- JAX PyTree Registration ---
    def _tree_flatten(self):
        # The Tracer's only dynamic component is the Profile PyTree
        leaves = (self.profile,)
        aux_data = None 
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        return obj

    def update(self, profile=None):
        """
        Return a new CIBTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : CIBProfile, optional
            New CIB profile to use for the tracer. If None, the profile is unchanged.

        Returns
        -------
        CIBTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        if profile is not None:
            flat = (profile,)
        return self._tree_unflatten(aux, flat)
    
    def kernel(self, cosmology, z):
        """
        Compute the CIB kernel :math:`W_{\\mathrm{CIB}}(\\chi)` at redshift :math:`z`.

        The kernel is given by:

        .. math::

            W_{\\mathrm{CIB}}(\\chi) = \\frac{1}{1+z}

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object.
        z : float or array_like
            Redshift(s) at which to compute the kernel.

        Returns
        -------
        W_cib : array_like
            CIB kernel evaluated at redshift(s) :math:`z`.
        """
        z = jnp.atleast_1d(z)
        return 1.0 / (1.0 + z)


jax.tree_util.register_pytree_node(
    CIBTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: CIBTracer._tree_unflatten(aux_data, children)
)


  