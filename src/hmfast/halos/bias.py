import jax
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod

from hmfast.halos.massdef import MassDefinition


class HaloBias(ABC):
    """
    Parent halo bias class from which halo bias models inherit.

    Child classes must implement :meth:`halo_bias`.
    """
    @abstractmethod
    def halo_bias(self, cosmology, m, z, mass_definition=None, convert_masses=False, order=1):
        """Required halo bias evaluator."""
        pass

class T10HaloBias(HaloBias):
    """
    Halo bias model from `Tinker et al. (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T/abstract>`_.

    This class implements the large-scale halo bias relation as a function of
    peak height :math:`\\nu` and redshift, calibrated for the
    :math:`200\\mathrm{m}` halo definition.
    """

    def __init__(self):
        pass


    @partial(jax.jit, static_argnums=(0,))
    def _b1_nu(self, nu, delta_c, delta_mean):
        """
        Compute the first-order halo bias :math:`b_1(\\nu)` following
        Tinker et al. (2010).
    
        Parameters
        ----------
        nu : array-like
            Peak height :math:`\nu = \delta_c / \sigma(M, z)`.
        delta_c : float or array-like
            Spherical-collapse threshold.
        delta_mean : float or array-like
            Halo overdensity :math:`\\Delta`.
    
        Returns
        -------
        b1 : array-like
            First-order halo bias values.
        """
        y = jnp.log10(delta_mean)
        
        # Tinker (2010) parameters
        A  = jnp.array(1.0 + 0.24 * y * jnp.exp(-(4.0 / y) ** 4))
        a  = jnp.array(0.44 * y - 0.88)
        B  = jnp.array(0.183)
        b_ = jnp.array(1.5)
        C  = jnp.array((0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y) ** 4)))
        c  = jnp.array(2.4)

        nu_a = jnp.power(nu, a)
        first = A * (nu_a / (nu_a + delta_c ** a))
        b_nu = 1.0 - first + B * jnp.power(nu, b_) + C * jnp.power(nu, c)
    
        return b_nu


    @partial(jax.jit, static_argnums=(0,))
    def _b2_nu(self, nu, delta_c, z):
        """
        Compute the second-order halo bias :math:`b_2(\\nu)` following
        Tinker et al. (2010).
    
        Parameters
        ----------
        nu : array-like
            Squared peak height :math:`\nu = (\delta_c / \sigma(M, z))^2`.
        delta_c : float or array-like
            Spherical-collapse threshold.
        z : float or array-like
            Redshift(s).
    
        Returns
        -------
        b2 : array-like
            Second-order halo bias values.
        """

        z = jnp.atleast_1d(z)
        
        # Base parameters followed by redshift exponents
        alpha0, beta0, gamma0, eta0, phi0 = 0.368, 0.589, 0.864, -0.243, -0.729
        alpha_z, beta_z, gamma_z, eta_z, phi_z = 0.0, 0.2, -0.01, 0.27, -0.08

        # Compute z-dependent parameters
        alpha = alpha0 * (1 + z)**alpha_z
        beta  = beta0  * (1 + z)**beta_z
        gamma = gamma0 * (1 + z)**gamma_z
        eta   = eta0   * (1 + z)**eta_z
        phi   = phi0   * (1 + z)**phi_z


        a = -phi
        b = beta**2
        c = gamma
        d = eta + 0.5

        a2 = -17/21
        

        eps1 = (c * nu - 2 * d) / delta_c
        eps2 = (c * nu * (c * nu - 4 * d - 1) + 2 * d * (2 * d - 1)) /  delta_c**2
        
        E1 = - 2 * a / (delta_c * ((b * nu)**(-a) + 1))
        E2 = E1  * (-2 * a + 2 * c * nu - 4 * d + 1) / delta_c

        b2_nu = 2 * (1 + a2) * (eps1 + E1) + eps2 + E2

        return b2_nu


    @partial(jax.jit, static_argnums=(0, 5, 6))
    def halo_bias(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="mean"), convert_masses=False, order=1):
        """
        Compute the halo bias for a given order.
        
        The first-order (linear) and second-order (quadratic) halo bias are given by:
        
        .. math::
        
            b_1(\\nu) = 1 - A \\frac{\\nu^a}{\\nu^a + \\delta_c^a} + B \\nu^b + C \\nu^c
        
            b_2(\\nu) = 2(1 + a^2)(\\epsilon_1 + E_1) + \\epsilon_2 + E_2
        
        where
        
        - :math:`\\nu = \\delta_c / \\sigma(M)` is the peak height,
        - :math:`\\delta_c \\approx 1.686` is the critical density for collapse,
        - :math:`A, a, B, b, C, c` are given in `Tinker et al. (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T/abstract>`_, Table 2.
        - :math:`\\epsilon_1, E_1, \\epsilon_2, E_2` are given in `Hoffmann et al. (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.450.1674H/abstract>`_, Table 5.
        
        Please refer to the original paper for the parameter values and full expressions.
        
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the bias.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.
        mass_definition : MassDefinition, optional
            Halo mass definition at which to evaluate the bias. Defaults to
            the native :math:`200\\mathrm{m}` calibration definition.
        convert_masses : bool, optional
            Mass conversions are applied if ``convert_masses`` is set to
            ``True``.
        order : int, optional
            Bias order to evaluate. Supported values are ``1`` and ``2``.
        
        Returns
        -------
        float or array-like
            Dimensionless halo bias values of the requested order, with shape
            :math:`(N_m, N_z)`, where singleton dimensions get squeezed
            before return.
        """
       
       
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        sigma_M = jnp.transpose(jnp.reshape(cosmology.sigma_m(m, z), (len(m), len(z))))
        zz = jnp.broadcast_to(z[:, None], sigma_M.shape)

        # Handle delta values
        delta_numeric = mass_definition._delta_numeric(cosmology, z)
        delta_mean = mass_definition._convert_reference(
            cosmology,
            z,
            delta_numeric,
            from_ref=mass_definition.reference,
            to_ref='mean',
        )
        
        # Ensure delta_mean is 1D before indexing
        delta_mean = jnp.atleast_1d(delta_mean)
        delta_mean_2d = delta_mean[:, None] 
        
        # Broadcast to (nz, nm)
        delta_mean_broad = jnp.broadcast_to(delta_mean_2d, sigma_M.shape)
        delta_c = jnp.atleast_1d(cosmology.delta_c(z, prescription="EdS"))[:, None]

        if order == 1: 
            nu = delta_c / sigma_M
            return jnp.squeeze(self._b1_nu(nu, delta_c, delta_mean_broad).T)
        elif order == 2:
            nu = (delta_c / sigma_M)**2
            return jnp.squeeze(self._b2_nu(nu, delta_c, zz).T)
        else:
            raise ValueError("order must be either 1 or 2")



