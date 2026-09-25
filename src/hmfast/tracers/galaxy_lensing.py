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

    The kernel has two contributions. The lensing convergence term:

    .. math::

        W_{\\kappa_g}(\\chi) = \\frac{3}{2}\\,\\Omega_m \\left(\\frac{H_0}{c}\\right)^2
        \\chi(z)\\,(1+z)\\, I_s(z),

    where :math:`I_s(z)` is the lensing efficiency integral over the source
    distribution ``dndz``:

    .. math::

        I_s(z) = \\int_z^\\infty dz_s\\, \\frac{dN}{dz}(z_s)\\,
        \\frac{\\chi(z_s)-\\chi(z)}{\\chi(z_s)}.

    An intrinsic-alignment (NLA) term, controlled by ``ia_bias``:

    .. math::

        W_g^{\\mathrm{IA}}(\\chi) = -A_{\\mathrm{IA}}(z)\\, C_1\\, \\rho_{\\mathrm{crit},0}\\,
        \\frac{\\Omega_m}{D(z)}\\, W_n(\\chi),

    where :math:`W_n(\\chi) = \\frac{H(z)}{c}\\frac{dN}{dz}(z)` is built from
    this tracer's own ``dndz`` (not necessarily the same source distribution
    as a :class:`~hmfast.tracers.galaxy.GalaxyTracer`), :math:`C_1` is the
    Hirata & Seljak (2004) normalisation constant, and :math:`\\rho_{\\mathrm{crit},0}`
    is the present-day critical density. See :meth:`kernel` for how these are
    combined and returned.

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


    def _kernel_primary(self, cosmology, z):
        """
        Weak lensing convergence term (:math:`n=0`, projected with :math:`j_\\ell`) of the galaxy lensing
        kernel:

        .. math::

            W_{\\kappa_g}(\\chi) = \\frac{3}{2} \\Omega_m \\left(\\frac{H_0}{c}\\right)^2 \\chi(z)\\,(1+z)\\,I_s(z)

        where :math:`I_s(z)` is the lensing efficiency integral over the source
        distribution ``dndz``:

        .. math::

            I_s(z) = \\int_z^\\infty dz_s\\, \\frac{dN}{dz}(z_s)\\, \\frac{\\chi(z_s)-\\chi(z)}{\\chi(z_s)}.
        """
        cparams = cosmology._cosmo_params()
        z = jnp.atleast_1d(z)

        c_km_s = Const._c_ / 1e3  # Speed of light in km/s
        H0 = cosmology.H0  # Hubble constant in km/s/Mpc
        Omega_m = cparams["Omega0_m"]  # Matter density parameter

        chi_z = cosmology.angular_diameter_distance(z) * (1 + z)
        I_s = self._lensing_efficiency_integral(cosmology, z, self.dndz)

        W_kappa_g = (
            (3.0 / 2.0) * Omega_m *
            (H0/c_km_s)**2 *
            chi_z * (1 + z) *
            I_s
        )
        return jnp.squeeze(W_kappa_g)

    def _kernel_ia(self, cosmology, z):
        """
        Intrinsic-alignment (NLA) term (:math:`n=0`, projected with :math:`j_\\ell`) of the galaxy lensing
        kernel, controlled by ``ia_bias``.
        """
        cparams = cosmology._cosmo_params()
        z = jnp.atleast_1d(z)
        c_km_s = Const._c_ / 1e3
        Omega_m = cparams["Omega0_m"]

        z_a, A_vals = self.ia_bias
        A_IA_at_z = jnp.interp(z, z_a, A_vals)  # clamp-to-edge extrapolation (A_IA is not a density)
        D_z = cosmology.growth_factor(z)  # NaN outside the trained z-grid, even where A_IA_at_z == 0
        z_g, phi_prime_g = self.dndz
        H_grid = cosmology.hubble_parameter(z) / c_km_s
        W_density = H_grid * jnp.interp(z, z_g, phi_prime_g, left=0.0, right=0.0)

        rho_crit_h2_ref = cparams["Rho_crit_0"] / cparams["h"] ** 2  # strip the h^2 baked into Rho_crit_0 back out
        W_IA = -A_IA_at_z * (Const._C1_IA_ * rho_crit_h2_ref) * Omega_m / D_z * W_density

        return jnp.squeeze(W_IA)

    def kernel(self, cosmology, z):
        """
        Radial kernel terms of the galaxy weak lensing tracer.

        Each term is a pair :math:`(W, n)`, where :math:`W(\\chi)` is a radial
        kernel and :math:`n` selects the spherical Bessel derivative
        :math:`j_\\ell^{(n)}(k\\chi)` the term is projected with in an angular
        power spectrum.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object.
        z : float or array_like
            Redshift(s) at which to evaluate the kernels.

        Returns
        -------
        list of tuple of (array_like, int)
            - :math:`(W_{\\kappa_g}, 0)`: lensing convergence term, projected with :math:`j_\\ell`.
            - :math:`(W_g^{\\mathrm{IA}}, 0)`: intrinsic-alignment term, projected with :math:`j_\\ell`; identically zero for the default ``ia_bias``.
        """
        return [
            (self._kernel_primary(cosmology, z), 0),
            (self._kernel_ia(cosmology, z), 0),
        ]


jax.tree_util.register_pytree_node(
    GalaxyLensingTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GalaxyLensingTracer._tree_unflatten(aux_data, children)
)

       
