import os
import jax
import jax.numpy as jnp

from hmfast.tracers.base_tracer import Tracer
from hmfast.halos.profiles import MatterProfile, NFWMatterProfile
from hmfast.utils import Const
from hmfast.download import get_default_data_path


jax.config.update("jax_enable_x64", True)

class GalaxyLensingTracer(Tracer):
    """
    Galaxy weak lensing tracer. 
    """

    _required_profile_type = MatterProfile

    
    def __init__(self, profile=None, dndz=None):        

        super().__init__(profile=profile or NFWMatterProfile())

        if dndz is None:
            # Call _load_dndz_data from BaseTracer
            dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "nz_source_normalized_bin4.txt")
            self.dndz = self._load_dndz_data(dndz_path)
        else:
            self.dndz = dndz
            

    @property
    def dndz(self):
        return self._dndz_data

    @dndz.setter
    def dndz(self, value):
        self._dndz_data = self._normalize_dndz(value)


    # --- Begin JAX PyTree Registration ---

    def _tree_flatten(self):
        # Exactly like HOD: Profile is leaf 1, dndz array/tuple is leaf 2
        leaves = (self.profile, self._dndz_data)
        aux_data = None 
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        profile, dndz_data = leaves
        obj = cls.__new__(cls)
        obj.profile = profile
        obj._dndz_data = dndz_data
        return obj

    def update(self, profile=None, dndz=None):
        """
        Return a new GalaxyLensingTracer instance with updated attributes using PyTree logic.

        Parameters
        ----------
        profile : MatterProfile, optional
            New matter profile to use for the tracer. If None, the profile is unchanged.
        dndz : array_like, optional
            New redshift distribution (z, dN/dz). If None, the distribution is unchanged.

        Returns
        -------
        GalaxyLensingTracer
            New tracer instance with updated attributes.
        """
        flat, aux = self._tree_flatten()
        new_profile = profile if profile is not None else flat[0]
        new_dndz = dndz if dndz is not None else flat[1]
        return self._tree_unflatten(aux, (new_profile, new_dndz))


    # --- End JAX PyTree Registration ---

    
    def _I_s(self, cosmology, z):
        """
        Compute the lensing efficiency integral :math:`I_s(z)` at redshift :math:`z`.
    
        The integral is given by:
    
        .. math::
    
            I_s(z) = \\int_z^{\\infty} dz_s\\, \\frac{dN}{dz}(z_s) \\frac{\\chi(z_s) - \\chi(z)}{\\chi(z_s)}
    
        where :math:`\\frac{dN}{dz}(z_s)` is the normalized source redshift distribution,
        :math:`\\chi(z)` is the comoving distance to redshift :math:`z`, and
        :math:`\\chi(z_s)` is the comoving distance to source redshift :math:`z_s`.
    
        Integrates over the source redshift distribution, including only sources behind the lens.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object with required methods and parameters.
        z : float or array_like
            Redshift(s) at which to compute the integral.
    
        Returns
        -------
        I_s : array_like
            Lensing efficiency integral evaluated at redshift(s) :math:`z`.
        """
        
        z = jnp.atleast_1d(z)
        h = cosmology.H0 / 100
        
        # Load source distribution       
        z_s, phi_prime_s = self.dndz
        
        # Angular distances
        chi_z_s = cosmology.angular_diameter_distance(z_s) * (1 + z_s) 
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z) 
    
        # Reshape for broadcasting
        chi_z_s = chi_z_s[:, None]  # (N_s, 1)
        chi_z = chi_z[None, :]      # (1, N_z)
    
        # Lensing factor
        chi_diff = (chi_z_s - chi_z) / chi_z_s
    
        # Mask: only include sources behind the lens
        mask = (z_s[:, None] > z[None, :])  # (N_s, N_z)
        chi_diff_masked = chi_diff * mask
    
        # Integrate over z_s using trapezoid
        I_s = jnp.trapezoid(phi_prime_s[:, None] * chi_diff_masked, x=z_s, axis=0)
    
        return I_s


    def kernel(self, cosmology, z):
        """
        Compute the galaxy lensing kernel :math:`W_{\\kappa_g}(z)` at redshift :math:`z`.
    
        The kernel is given by:
    
        .. math::
    
            W_{\\kappa_g}(z) = \\frac{3}{2} \\Omega_m \\left(\\frac{H_0}{c}\\right)^2 \\frac{(1+z)}{\\chi(z)} I_s(z)
    
        where :math:`\\Omega_m` is the matter density parameter,
        :math:`H_0` is the Hubble constant, :math:`c` is the speed of light,
        :math:`\\chi(z)` is the comoving distance to redshift :math:`z`, and
        :math:`I_s(z)` is the lensing efficiency integral defined as
    
        .. math::
    
            I_s(z) = \\int_z^{\\infty} dz_s\\, \\frac{dN}{dz}(z_s) \\frac{\\chi(z_s) - \\chi(z)}{\\chi(z_s)}
    
        where :math:`\\frac{dN}{dz}(z_s)` is the normalized source redshift distribution.
    
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
        h = H0 / 100
        Omega_m = cparams["Omega0_m"]  # Matter density parameter

        # Compute comoving distance and Hubble parameter
        chi_z = cosmology.angular_diameter_distance(z) * (1 + z) * h # Comoving distance in Mpc/h
        H_z = cosmology.hubble_parameter(z)   # Hubble parameter in km/s/Mpc
    
        I_s = self._I_s(cosmology, z) 
    
        # Compute the CMB lensing kernel
        W_kappa_g =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            I_s 
        ) 
    
        return W_kappa_g 




jax.tree_util.register_pytree_node(
    GalaxyLensingTracer,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GalaxyLensingTracer._tree_unflatten(aux_data, children)
)

       
