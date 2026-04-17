import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.tracers.base_tracer import Tracer
from hmfast.utils import Const
from hmfast.halos.profiles import PressureProfile, GNFWPressureProfile

jax.config.update("jax_enable_x64", True)


class tSZTracer(Tracer):
    """
    thermal Sunyaev-Zeldovich effect tracer.
    """

    _required_profile_type = PressureProfile
    
    def __init__(self, profile=None):
        super().__init__(profile=profile or GNFWPressureProfile())


    # --- Begin JAX PyTree Registration ---

    def _tree_flatten(self):
        # The profile is the dynamic leaf
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
        Return a new tSZTracer instance with updated attributes.
    
        Parameters
        ----------
        profile : PressureProfile, optional
            New pressure profile to use for the tracer. If None, the profile is unchanged.
    
        Returns
        -------
        tSZTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        if profile is not None:
            flat = (profile,)
        return self._tree_unflatten(aux, flat)

    # --- End JAX PyTree Registration ---
        
    def kernel(self, cosmology, z):

        """
        Compute the tSZ kernel as a function of redshift.
    
        The kernel is given by:
    
            .. math::
    
               W_{\\mathrm{kSZ}}(z) = \\frac{\\sigma_T}{m_e c^2} \\frac{1}{1+z}
    
        where :math:`\\sigma_T` is the Thomson cross-section, :math:`m_e` is
        the electron mass, and :math:`z` is the redshift.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object.
        z : float or array-like
            Redshift(s).
    
        Returns
        -------
        W_tsz : float or array-like
            tSZ kernel evaluated at redshift(s) :math:`z`.
        """
        
        h = cosmology.H0/100 
        
        # Get electon mass in eV, Thomson cross section in cm^2, and Mpc/h in cm
        m_e = Const._m_e_ * Const._c_**2 / Const._eV_
        sigma_T = Const._sigma_T_ * 1e6
        mpc_per_h_to_cm =  Const._Mpc_over_m_ / h
        return (sigma_T / m_e) / (1+z) # Check this



jax.tree_util.register_pytree_node(
    tSZTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: tSZTracer._tree_unflatten(aux_data, children)
)