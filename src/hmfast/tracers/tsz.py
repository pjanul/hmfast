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

    The kernel has a single term (from :meth:`_kernel_primary`):

    .. math::

        W_y(\\chi) = \\frac{\\sigma_T}{m_e c^2}\\, \\frac{1}{1+z}

    for :math:`z \\leq z_{\\max}`, and zero otherwise, where :math:`\\sigma_T`
    is the Thomson cross-section and :math:`m_e c^2` is the electron
    rest-mass energy. See :meth:`kernel` for how this is returned.

    Attributes
    ----------
    profile : PressureProfile
        Pressure profile used to model the thermal Sunyaev-Zeldovich signal.
    z_max : float
        Maximum redshift up to which the kernel has support.
    """

    _required_profile_type = PressureProfile

    def __init__(self, profile=None, z_max=5.0):
        super().__init__(profile=profile or GNFWPressureProfile())
        self.z_max = z_max


    # --- Begin JAX PyTree Registration ---

    def _tree_flatten(self):
        # The profile is the dynamic leaf
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
        Return a new tSZTracer instance with updated attributes.

        Parameters
        ----------
        profile : PressureProfile, optional
            New pressure profile to use for the tracer. If None, the profile is unchanged.
        z_max : float, optional
            New maximum redshift for the kernel. If None, z_max is unchanged.

        Returns
        -------
        tSZTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        if profile is not None:
            flat = (profile,)
        if z_max is not None:
            aux = (z_max,)
        return self._tree_unflatten(aux, flat)

    # --- End JAX PyTree Registration ---
        
    def _kernel_primary(self, cosmology, z):

        """
        Compute the tSZ kernel as a function of redshift.

        The kernel is given by:

            .. math::

               W_{\\mathrm{tSZ}}(\\chi) = \\frac{\\sigma_T}{m_e c^2} \\frac{1}{1+z}

        for :math:`z \\leq z_{\\max}`, and zero otherwise.

        where :math:`\\sigma_T` is the Thomson cross-section, :math:`m_e c^2` is
        the electron rest-mass energy, and :math:`z` is the redshift.

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

        z = jnp.atleast_1d(z)

        m_e = Const._m_e_ * Const._c_**2 / Const._eV_
        sigma_T = Const._sigma_T_ * 1e6
        W = (sigma_T / m_e) * Const._Mpc_over_m_ / (1.0 + z)
        # Tolerance guards against callers' z grids (e.g. jnp.geomspace) not landing bit-exactly on z_max.
        return jnp.squeeze(jnp.where(z <= self.z_max + 1e-8, W, 0.0))

    def kernel(self, cosmology, z):
        """
        Returns
        -------
        list of (weight, der_bessel)
            A single term, ``(W_y, 0)``.
        """
        return [(self._kernel_primary(cosmology, z), 0)]



jax.tree_util.register_pytree_node(
    tSZTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: tSZTracer._tree_unflatten(aux_data, children)
)