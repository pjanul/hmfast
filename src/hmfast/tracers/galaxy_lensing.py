import os
import jax
import jax.numpy as jnp

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import MatterProfile, NFWMatterProfile
from hmfast.utils import Const
from hmfast.download import _get_default_data_path


jax.config.update("jax_enable_x64", True)


class GalaxyLensingTracer(Tracer):
    """
    Galaxy weak lensing tracer.

    Attributes
    ----------
    profile : MatterProfile
        Matter profile used to model the lensing signal sourced by large-scale structure.
    dndz : tuple of jnp.ndarray
        Normalized source redshift distribution stored as :math:`(z, dN/dz)`.
    ia_bias : tuple of jnp.ndarray
        Intrinsic-alignment (NLA) amplitude stored as :math:`(z, A_{IA}(z))`.
        Defaults to :math:`A_{IA}(z)\\equiv 0` (no intrinsic alignments).
    """

    _required_profile_type = MatterProfile


    def __init__(self, profile=None, dndz=None, ia_bias=None):

        super().__init__(profile=profile or NFWMatterProfile())

        if dndz is None:
            # Call _load_dndz_data from BaseTracer
            dndz_path = os.path.join(_get_default_data_path(), "auxiliary_files", "nz_source_normalized_bin4.txt")
            self.dndz = self._load_dndz_data(dndz_path)
        else:
            self.dndz = dndz

        if ia_bias is None:
            ia_bias = (jnp.array([0.0, 1.0]), jnp.array([0.0, 0.0]))
        self.ia_bias = ia_bias


    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._prepare_z_function(value)

    @property
    def ia_bias(self):
        return self._ia_bias_data

    @ia_bias.setter
    def ia_bias(self, value):
        self._ia_bias_data = self._prepare_z_function(value, normalize=False)


    # --- Begin JAX PyTree Registration ---

    def _tree_flatten(self):
        # Exactly like HOD: Profile is leaf 1, dndz array/tuple is leaf 2
        leaves = (self.profile, self._dndz_data, self._ia_bias_data)
        aux_data = None
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, dndz_data, ia_bias_data = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._dndz_data = dndz_data
        obj._ia_bias_data = ia_bias_data
        return obj

    def update(self, profile=None, dndz=None, ia_bias=None):
        """
        Return a new GalaxyLensingTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : MatterProfile, optional
            New matter profile to use for the tracer. If None, the profile is unchanged.
        dndz : array_like, optional
            New redshift distribution (z, dN/dz). If None, the distribution is unchanged.
        ia_bias : array_like, optional
            New intrinsic-alignment amplitude (z, A_IA(z)). If None, it is unchanged.

        Returns
        -------
        GalaxyLensingTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        new_profile = profile if profile is not None else flat[0]
        new_dndz = self._prepare_z_function(dndz) if dndz is not None else flat[1]
        new_ia_bias = self._prepare_z_function(ia_bias, normalize=False) if ia_bias is not None else flat[2]
        return self._tree_unflatten(aux, (new_profile, new_dndz, new_ia_bias))


    # --- End JAX PyTree Registration ---


    def kernel(self, cosmology, z):
        """
        Compute the galaxy lensing kernel :math:`W_{\\kappa_g}(\\chi)` at redshift :math:`z`.

        The kernel is given by:

        .. math::

            W_{\\kappa_g}(\\chi) = \\frac{3}{2} \\Omega_m \\left(\\frac{H_0}{c}\\right)^2 \\chi(z)\\,(1+z)\\,I_s(z)

        where :math:`\\Omega_m` is the matter density parameter,
        :math:`H_0` is the Hubble constant, :math:`c` is the speed of light,
        :math:`\\chi(z)` is the comoving distance to redshift :math:`z`,
        :math:`I_s(z)` is the lensing efficiency integral defined as

        .. math::

            I_s(z) = \\int_z^{\\infty} dz_s\\, \\frac{dN}{dz}(z_s) \\frac{\\chi(z_s) - \\chi(z)}{\\chi(z_s)}

        where :math:`\\frac{dN}{dz}(z_s)` is the normalized source redshift distribution.
        The kernel also includes an intrinsic-alignment (NLA) contribution controlled
        by ``ia_bias``.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the kernel.

        Returns
        -------
        W_kappa_g : array_like
            Galaxy lensing kernel evaluated at redshift(s) :math:`z`.
        """
        # Merge default parameters with input

        cparams = cosmology._cosmo_params()
        z = jnp.atleast_1d(z) # Ensure z is an array

        c_km_s = Const._c_ / 1e3  # Speed of light in km/s

        # Cosmological constants
        H0 = cosmology.H0  # Hubble constant in km/s/Mpc
        Omega_m = cparams["Omega0_m"]  # Matter density parameter

        # Compute comoving distance in physical Mpc.
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z)

        I_s = self._lensing_efficiency_integral(cosmology, z, self.dndz)

        # Compute the galaxy lensing kernel
        W_kappa_g = (
            (3.0 / 2.0) * Omega_m *
            (H0/c_km_s)**2 *
            chi_z * (1 + z) *
            I_s
        )

        # Intrinsic alignments (NLA model)
        z_a, A_vals = self.ia_bias
        A_IA_at_z = jnp.interp(z, z_a, A_vals)  # clamp-to-edge extrapolation (A_IA is not a density)
        D_z = cosmology.growth_factor(z)  # NaN outside the trained z-grid, even where A_IA_at_z == 0
        z_g, phi_prime_g = self.dndz
        H_grid = cosmology.hubble_parameter(z) / c_km_s
        W_density = H_grid * jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)

        rho_crit_h2_ref = cparams["Rho_crit_0"] / cparams["h"] ** 2  # strip the h^2 baked into Rho_crit_0 back out
        W_IA = -A_IA_at_z * (Const._C1_IA_ * rho_crit_h2_ref) * Omega_m / D_z * W_density

        return jnp.squeeze(W_kappa_g + W_IA)




jax.tree_util.register_pytree_node(
    GalaxyLensingTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GalaxyLensingTracer._tree_unflatten(aux_data, children)
)

       
