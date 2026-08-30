import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import numpy as np

from hmfast.halos.profiles import HaloProfile


 
class Tracer(ABC):
    """
    Parent tracer class from which other tracer classes inherit.

    Child tracers must implement :meth:`kernel`. When used in the halo model,
    they must also define a ``profile`` attribute with an appropriate profile
    object.
    """
    
    _required_profile_type = HaloProfile 

    def __init__(self, profile=None):
        """
        Initialize the tracer with a validated profile.
        """
        if profile is not None:
            self.profile = profile

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value):
        """
        Enforces type safety: prevents assigning a PressureProfile to a 
        LensingTracer, etc.
        """
        if not isinstance(value, self._required_profile_type):
            raise TypeError(
                f"{self.__class__.__name__} strictly requires a "
                f"{self._required_profile_type.__name__}. "
                f"Received: {type(value).__name__}"
            )
        self._profile = value
        


    def _load_dndz_data(self, path):
        """
        Loads dndz curves in the format (z, phi) for galaxy HOD and galaxy lensing tracers.
        """
        data = np.loadtxt(path)
        x = data[:, 0]
        y = data[:, 1]
        return (jnp.array(x), jnp.array(y))

        
    def _prepare_z_function(self, value, normalize=True):
        """
        Converts (z, phi) curves to jnp arrays, optionally normalizing phi (e.g. for dndz);
        set normalize=False for non-density curves like a magnification-bias slope or IA amplitude.
        """
        z = jnp.atleast_1d(jnp.array(value[0]))
        phi = jnp.atleast_1d(jnp.array(value[1]))
        if normalize:
            phi = phi / jnp.trapezoid(phi, x=z)
        return (z, phi)

    def _lensing_efficiency_integral(self, cosmology, z, dndz, weight=None):
        """
        Compute the lensing efficiency integral :math:`I_s(z)` at redshift :math:`z`,
        given an explicit source redshift distribution ``dndz`` (rather than a fixed
        ``self.dndz``, so any tracer can reuse this with its own source distribution).

        The integral is given by:

        .. math::

            I_s(z) = \\int_z^{\\infty} dz_s\\, w(z_s)\\, \\frac{dN}{dz}(z_s) \\frac{\\chi(z_s) - \\chi(z)}{\\chi(z_s)}

        where :math:`\\frac{dN}{dz}(z_s)` is the normalized source redshift distribution,
        :math:`\\chi(z)` is the comoving distance to redshift :math:`z`,
        :math:`\\chi(z_s)` is the comoving distance to source redshift :math:`z_s`, and
        :math:`w(z_s)` is an optional per-source weight (``weight``, defaulting to 1),
        e.g. a magnification-bias weighting evaluated at each source redshift.

        Integrates over the source redshift distribution, including only sources behind the lens.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the integral.
        dndz : tuple of jnp.ndarray
            Normalized source redshift distribution stored as :math:`(z_s, dN/dz)`.
        weight : array_like, optional
            Per-source weight :math:`w(z_s)`, same shape as ``dndz[0]``. If None, no
            weighting is applied (the plain lensing efficiency integral).

        Returns
        -------
        I_s : array_like
            Lensing efficiency integral evaluated at redshift(s) :math:`z`.
        """

        z = jnp.atleast_1d(z)

        # Load source distribution
        z_s, phi_prime_s = dndz
        if weight is not None:
            phi_prime_s = phi_prime_s * weight

        # Angular distances
        chi_z_s = cosmology.angular_diameter_distance(z_s) * (1 + z_s)
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z)

        # Reshape for broadcasting
        chi_z_s = chi_z_s[:, None]  # (N_s, 1)
        chi_z = chi_z[None, :]      # (1, N_z)

        # Lensing factor
        # A source at z_s=0 gives chi_z_s=0 (always masked out below), but an unguarded 1/chi_z_s still poisons the gradient with NaN/Inf.
        safe_chi_z_s = jnp.where(chi_z_s > 0, chi_z_s, 1.0)
        chi_diff = (chi_z_s - chi_z) / safe_chi_z_s

        # Mask: only include sources behind the lens
        mask = (z_s[:, None] > z[None, :])  # (N_s, N_z)
        chi_diff_masked = chi_diff * mask

        # Integrate over z_s using trapezoid
        I_s = jnp.trapezoid(phi_prime_s[:, None] * chi_diff_masked, x=z_s, axis=0)

        return I_s


    @abstractmethod
    def kernel(self, cosmology, z):
        """Required tracer kernel."""
        pass 
   
  