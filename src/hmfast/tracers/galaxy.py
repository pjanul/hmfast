import os
import jax
import jax.numpy as jnp

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import GalaxyHODProfile, Z07GalaxyHODProfile
from hmfast.download import _get_default_data_path
from hmfast.utils import Const

# Ensure high precision for cosmological integrations
jax.config.update("jax_enable_x64", True)

class GalaxyTracer(Tracer):
    """
    Galaxy counts tracer.

    Attributes
    ----------
    profile : GalaxyHODProfile
        Halo occupation distribution profile used to model galaxy number counts.
    dndz : tuple of jnp.ndarray
        Normalized galaxy redshift distribution stored as :math:`(z, dN/dz)`.
    mag_bias : tuple of jnp.ndarray
        Magnification-bias log-slope of number counts (w.r.t. magnitude) stored as
        :math:`(z, s(z))`. Defaults to :math:`s(z)\\equiv 2/5` (no magnification bias).
    bias : tuple of jnp.ndarray or None
        Linear galaxy bias stored as :math:`(z, b(z))`. Only used when this tracer is used to
        compute a linearly-biased angular power spectrum; has no effect when this tracer is used
        with a full halo-model calculation, where bias is instead handled through the halo
        occupation profile. Defaults to `None` (unbiased).
    """

    _required_profile_type = GalaxyHODProfile

    def __init__(self, profile=None, dndz=None, mag_bias=None, bias=None):
        super().__init__(profile=profile or Z07GalaxyHODProfile())

        if dndz is None:
            dndz_path = os.path.join(_get_default_data_path(), "auxiliary_files", "normalised_dndz_cosmos_0.txt")
            dndz = self._load_dndz_data(dndz_path)

        self.dndz = dndz

        if mag_bias is None:
            mag_bias = (jnp.array([0.0, 1.0]), jnp.array([0.4, 0.4]))
        self.mag_bias = mag_bias

        self.bias = bias


    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._prepare_z_function(value)

    @property
    def mag_bias(self):
        return self._mag_bias_data

    @mag_bias.setter
    def mag_bias(self, value):
        self._mag_bias_data = self._prepare_z_function(value, normalize=False)

    @property
    def bias(self):
        return self._bias_data

    @bias.setter
    def bias(self, value):
        self._bias_data = None if value is None else self._prepare_z_function(value, normalize=False)

    # --- JAX PyTree Registration ---

    def _tree_flatten(self):
        # The profile IS the leaf. JAX will automatically
        # drill down into the profile's own 5 leaves.
        leaves = (self.profile, self._dndz_data, self._mag_bias_data, self._bias_data)
        return (leaves, None)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, dndz_data, mag_bias_data, bias_data = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._dndz_data = dndz_data
        obj._mag_bias_data = mag_bias_data
        obj._bias_data = bias_data
        return obj

    def update(self, profile=None, dndz=None, mag_bias=None, bias=None):
        """
        Return a new GalaxyTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : GalaxyHODProfile, optional
            New HOD profile to use for the tracer. If None, the profile is unchanged.
        dndz : array_like, optional
            New redshift distribution (z, dN/dz). If None, the distribution is unchanged.
        mag_bias : array_like, optional
            New magnification-bias slope (z, s(z)). If None, it is unchanged.
        bias : array_like, optional
            New linear galaxy bias (z, b(z)). If None, it is unchanged.

        Returns
        -------
        GalaxyTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        new_profile = profile if profile is not None else flat[0]
        new_dndz = self._prepare_z_function(dndz) if dndz is not None else flat[1]
        new_mag_bias = self._prepare_z_function(mag_bias, normalize=False) if mag_bias is not None else flat[2]
        new_bias = self._prepare_z_function(bias, normalize=False) if bias is not None else flat[3]
        return self._tree_unflatten(aux, (new_profile, new_dndz, new_mag_bias, new_bias))


    def kernel(self, cosmology, z):
        """
        Compute the galaxy kernel :math:`W_g(\\chi)` at redshift :math:`z`.

        The kernel is given by:

        .. math::

            W_g(\\chi) = \\frac{H(z)}{c} \\frac{dN}{dz}

        where :math:`dN/dz` is the normalized redshift distribution of galaxies.
        The kernel also includes a magnification-bias contribution controlled by
        ``mag_bias``.

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
        H_grid = cosmology.hubble_parameter(z) / (Const._c_ / 1e3)
        W_density = H_grid * phi_prime_g_at_z

        z_s, s_vals = self.mag_bias
        s_at_source = jnp.interp(z_g, z_s, s_vals)  # s(z) at the source (own dndz) grid, clamp-to-edge
        weight = 1.0 - 2.5 * s_at_source  # magnification weighting from the mag-bias log-slope s(z)

        cparams = cosmology._cosmo_params()
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z)
        lensing_pref = 1.5 * cparams["Omega0_m"] * (cosmology.H0 / (Const._c_ / 1e3)) ** 2 * chi_z * (1 + z)
        W_mag = -2.0 * lensing_pref \
            * self._lensing_efficiency_integral(cosmology, z, self.dndz, weight=weight)

        return jnp.squeeze(W_density + W_mag)



jax.tree_util.register_pytree_node(
    GalaxyTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GalaxyTracer._tree_unflatten(aux_data, children)
)
