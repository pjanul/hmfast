import os
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici, erf 

from hmfast.tracers.base_tracer import Tracer
from hmfast.download import _get_default_data_path
from hmfast.utils import Const
from hmfast.halos.profiles import MatterProfile, NFWMatterProfile
jax.config.update("jax_enable_x64", True)


class CMBLensingTracer(Tracer):
    """
    CMB weak lensing tracer.

    The kernel has a single lensing convergence term:

    .. math::

        W_{\\kappa_{\\mathrm{CMB}}}(\\chi) = \\frac{3}{2}\\,\\Omega_m
        \\left(\\frac{H_0}{c}\\right)^2 \\chi(z)\\,(1+z)\\,
        \\frac{\\chi_s - \\chi(z)}{\\chi_s}

    for :math:`\\chi(z) < \\chi_s` and zero beyond, where :math:`\\chi_s` is the comoving
    distance to the source plane: :math:`\\chi(z_{\\rm source})` if ``z_source`` is given,
    otherwise the last-scattering distance :math:`\\chi_*` from the emulator's derived
    parameters. See :meth:`kernel` for how this is returned.

    Attributes
    ----------
    profile : MatterProfile
        Matter profile used to model the CMB lensing convergence signal.
    z_source : float or None
        Source-plane redshift. If None (default), the source plane is the last-scattering
        surface, with :math:`\\chi_s = \\chi_*` and :math:`z_*` read from the emulator's derived
        parameters. Above the background emulator's redshift range (:math:`z=20`),
        :math:`\\chi(z_{\\rm source})` is NaN unless the cosmology has ``extrapolate_z=True``.
    """

    _required_profile_type = MatterProfile

    def __init__(self, profile=None, *, z_source=None):
        super().__init__(profile=profile or NFWMatterProfile())
        self.z_source = z_source

    def _tree_flatten(self):
        # z_source is a leaf (None is an empty subtree), so it can be traced and differentiated.
        leaves = (self.profile, self.z_source)
        aux_data = None
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, z_source = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj.z_source = z_source
        return obj

    def update(self, *, profile=None, z_source=None):
        """
        Return a new CMBLensingTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : MatterProfile, optional
            New matter profile to use for the tracer. If None, the profile is unchanged.
        z_source : float, optional
            New source-plane redshift. If None, z_source is unchanged.

        Returns
        -------
        CMBLensingTracer
            New tracer instance with updated attributes.
        """
        (old_profile, old_z_source), aux = self._tree_flatten()
        new_profile = profile if profile is not None else old_profile
        new_z_source = z_source if z_source is not None else old_z_source
        return self._tree_unflatten(aux, (new_profile, new_z_source))

    @staticmethod
    def _derived(cosmology, name):
        """A derived parameter of the cosmology, for the last-scattering source plane."""
        if not hasattr(cosmology, "derived_parameters"):
            raise TypeError(f"{type(cosmology).__name__} has no derived parameters, so CMBLensingTracer needs z_source.")
        return cosmology.derived_parameters()[name]

    def _z_source(self, cosmology):
        """Source-plane redshift: z_source, or the derived z_star if z_source is None."""
        return self._derived(cosmology, "z_star") if self.z_source is None else jnp.asarray(self.z_source)

    def _z_max(self, cosmology):
        """Redshift above which every kernel term vanishes: the source plane."""
        return self._z_source(cosmology)

    def _chi_source(self, cosmology):
        """Comoving distance to the source plane: chi(z_source), or the derived chi_star if z_source is None."""
        if self.z_source is None:
            return self._derived(cosmology, "chi_star")
        z_s = jnp.asarray(self.z_source)
        return cosmology.angular_diameter_distance(z_s) * (1.0 + z_s)


    def _kernel_primary(self, cosmology, z):
        """
        Compute the CMB lensing kernel :math:`W_{\\kappa_{\\mathrm{CMB}}}(\\chi)` at
        redshift :math:`z` (:math:`n=-1`, :math:`a=1`: projected with :math:`\\ell(\\ell+1)\\, j_\\ell/(k\\chi)^2`).

        The kernel is given by:

            .. math::

                W_{\\kappa_{\\mathrm{CMB}}}(\\chi) = \\frac{3}{2} \\Omega_m \\left(\\frac{H_0}{c}\\right)^2 \\chi(z)\\,(1+z)\\,\\frac{\\chi_s - \\chi(z)}{\\chi_s}

        for :math:`\\chi(z) < \\chi_s` and zero beyond, where :math:`\\chi_s` is the comoving
        distance to the source plane (:math:`\\chi_*` unless ``z_source`` is given).
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the kernel.
    
        Returns
        -------
        W_kappa_cmb : array_like
            CMB lensing kernel evaluated at redshift(s) :math:`z`.
        """
        # Merge default parameters with input
        
        cparams = cosmology._cosmo_params()
        z = jnp.atleast_1d(z)  # Ensure z is an array
        
        # Cosmological constants
        H0 = cosmology.H0    # Hubble constant in km/s/Mpc
        Omega_m = cparams["Omega0_m"]  # Matter density parameter
        c_km_s = Const._c_ / 1e3  # Speed of light in km/s
        
        # Compute comoving distances in physical Mpc.
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z)
        
        # Comoving distance to the source plane in physical Mpc.
        chi_s = self._chi_source(cosmology)

        # Compute the CMB lensing kernel
        W_kappa_cmb = (
            (3.0 / 2.0) * Omega_m *
            (H0/c_km_s)**2 *
            chi_z * (1 + z) *
            ((chi_s - chi_z) / chi_s)
        )
        # Written as chi_z > chi_s so a NaN chi_s (z_source past the emulator range) stays NaN, not 0.
        W_kappa_cmb = jnp.where(chi_z > chi_s, 0.0, W_kappa_cmb)


        return jnp.squeeze(W_kappa_cmb)

    def kernel(self, cosmology, z):
        """
        Radial kernel terms of the CMB weak lensing tracer.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object.
        z : float or array_like
            Redshift(s) at which to evaluate the kernels.

        Returns
        -------
        list of tuple of (array_like, int, int)
            One ``(W, n, a)`` per term, specifying how it is projected into :math:`C_\\ell`:
            :math:`W` is the radial kernel, :math:`n` the order of the spherical Bessel
            derivative :math:`j^{(n)}_\\ell(k\\chi)` (with :math:`n=-1` meaning
            :math:`j_\\ell(k\\chi)/(k\\chi)^2`), and :math:`a` the order of the angular derivative,
            which sets the :math:`\\ell`-dependent prefactor (:math:`1`, :math:`\\ell(\\ell+1)` or
            :math:`\\sqrt{(\\ell+2)!/(\\ell-2)!}` for :math:`a = 0, 1, 2`).

            - ``(W_κ_CMB, -1, 1)``: convergence kernel :math:`W_{\\kappa_{\\rm CMB}}` above.
        """
        return [(self._kernel_primary(cosmology, z), -1, 1)]


jax.tree_util.register_pytree_node(
    CMBLensingTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: CMBLensingTracer._tree_unflatten(aux_data, children)
)

