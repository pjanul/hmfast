import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.tracers.base_tracer import Tracer
from hmfast.utils import Const
from hmfast.halos.profiles import DensityProfile, NFWDensityProfile, B16DensityProfile

jax.config.update("jax_enable_x64", True)

class kSZTracer(Tracer):
    """
    kinetic Sunyaev-Zeldovich effect tracer.
    """
    _required_profile_type = DensityProfile
    
    def __init__(self, profile=None):
        super().__init__(profile=profile or B16DensityProfile())


    # ---------------- Start JAX PyTree Registration ---------------- #

    def _tree_flatten(self):
        # The profile is the leaf. JAX will drill down into the profile's leaves.
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
        Return a new kSZTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : DensityProfile, optional
            New density profile to use for the tracer. If None, the profile is unchanged.

        Returns
        -------
        kSZTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        new_profile = profile if profile is not None else flat[0]
        return self._tree_unflatten(aux, (new_profile,))


    # ---------------- End JAX PyTree Registration ---------------- #

    def kernel(self, cosmology, z):
        """
        Compute the kSZ kernel $W_{\\mathrm{kSZ}}(z)$ at redshift $z$.

        The kernel is given by:

        .. math::
            W_{\\mathrm{kSZ}}(z) = \\frac{\\sigma_T}{m_p} \\frac{1}{1+z}

        where $\\sigma_T$ is the Thomson cross-section, $m_p$ is the proton mass, and $z$ is the redshift

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the kernel.

        Returns
        -------
        W_ksz : array_like
            kSZ kernel evaluated at redshift(s) $z$.
        """
        # sigmaT / m_prot in (Mpc/h)**2/(Msun/h) which is required for kSZ
        sigma_T_over_m_p = (Const._sigma_T_ / Const._m_p_) / Const._Mpc_over_m_**2 * Const._M_sun_ * cosmology.H0 / 100
        return sigma_T_over_m_p / (1.0 + z)



jax.tree_util.register_pytree_node(
    kSZTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: kSZTracer._tree_unflatten(aux_data, children)
)
        