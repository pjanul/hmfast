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
    z_max : float
        Maximum redshift up to which the kernel has support.
    """

    _required_profile_type = CIBProfile

    def __init__(self, profile=None, z_max=5.0):
        super().__init__(profile=profile or S12CIBProfile(nu=100))
        self.z_max = z_max

    # --- JAX PyTree Registration ---
    def _tree_flatten(self):
        # The Tracer's only dynamic component is the Profile PyTree
        leaves = (self.profile,)
        aux_data = (self.z_max,)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, = leaves
        z_max, = aux_data
        obj = cls.__new__(cls)
        obj.profile = profile
        obj.z_max = z_max
        return obj

    def update(self, profile=None, z_max=None):
        """
        Return a new CIBTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : CIBProfile, optional
            New CIB profile to use for the tracer. If None, the profile is unchanged.
        z_max : float, optional
            New maximum redshift for the kernel. If None, z_max is unchanged.

        Returns
        -------
        CIBTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        if profile is not None:
            flat = (profile,)
        if z_max is not None:
            aux = (z_max,)
        return self._tree_unflatten(aux, flat)

    def _kernel_primary(self, cosmology, z):
        """
        Compute the CIB kernel :math:`W_{\\mathrm{CIB}}(\\chi)` at redshift :math:`z`.

        The kernel is given by:

        .. math::

            W_{\\mathrm{CIB}}(\\chi) = \\frac{1}{1+z}

        for :math:`z \\leq z_{\\max}`, and zero otherwise.

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
        W = 1.0 / (1.0 + z)
        # Tolerance guards against callers' z grids (e.g. jnp.geomspace) not landing bit-exactly on z_max.
        return jnp.squeeze(jnp.where(z <= self.z_max + 1e-8, W, 0.0))

    def kernel(self, cosmology, z):
        return [(self._kernel_primary(cosmology, z), 0)]


jax.tree_util.register_pytree_node(
    CIBTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: CIBTracer._tree_unflatten(aux_data, children)
)


  