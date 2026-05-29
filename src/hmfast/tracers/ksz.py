import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.tracers.base_tracer import Tracer
from hmfast.utils import Const
from hmfast.halos.profiles import DensityProfile, B16DensityProfile

jax.config.update("jax_enable_x64", True)

class kSZTracer(Tracer):
    """
    kinetic Sunyaev-Zeldovich effect tracer.

    Attributes
    ----------
    profile : DensityProfile
        Electron density profile used to model the kinetic Sunyaev-Zeldovich signal.
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
        Compute the kSZ kernel :math:`W_{\\mathrm{kSZ}}(z)` at redshift
        :math:`z`.

        The kernel is given by:

        .. math::

            W_{\\mathrm{kSZ}}(z) = \\frac{\\sigma_T}{m_p}
            \\frac{v_{\\mathrm{rms}}(z)}{\\mu_e \\, \\chi^2(z) \\, (1+z)}

        where :math:`\\sigma_T` is the Thomson cross-section, :math:`m_p` is
        the proton mass, :math:`\\mu_e = 1.14`, :math:`\\chi(z)` is the comoving
        distance, and :math:`z` is the redshift. In the implementation,
        :math:`\\sigma_T` is stored in m\\ :sup:`2`, :math:`m_p` is stored in kg,
        and the kernel prefactor is converted to physical
        :math:`\\mathrm{Mpc}^2 \\, M_\\odot^{-1}`.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the kernel.

        Returns
        -------
        W_ksz : array_like
            kSZ kernel evaluated at redshift(s) :math:`z`.
        """
        # sigmaT / m_prot in physical Mpc^2 / Msun.
        sigma_T_over_m_p = (Const._sigma_T_ / Const._m_p_) / Const._Mpc_over_m_**2 * Const._M_sun_
        z = jnp.atleast_1d(z)
        chi = cosmology.angular_diameter_distance(z) * (1.0 + z)
        velocity_dispersion = jnp.sqrt(cosmology.velocity_dispersion(z))
        mu_e = 1.14
        return sigma_T_over_m_p * velocity_dispersion / (mu_e * chi**2 * (1.0 + z))



jax.tree_util.register_pytree_node(
    kSZTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: kSZTracer._tree_unflatten(aux_data, children)
)
        