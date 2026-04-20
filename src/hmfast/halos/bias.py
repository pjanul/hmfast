import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial
from abc import ABC, abstractmethod
from mcfit import TophatVar


class HaloBias(ABC):
    """
    Abstract base class for halo bias models.

    Subclasses implement large-scale halo bias relations evaluated on a
    mass-redshift grid.
    """
    @abstractmethod
    def halo_bias(self, halo_model, m, z, order=1):
        """
        Evaluate the halo bias of the requested order.

        Parameters
        ----------
        halo_model : HaloModel
            Halo-model instance supplying the cosmology, mass definition, and
            any mass-conversion settings needed to evaluate the requested bias.
        m : array-like
            Halo masses at which to evaluate the bias.
        z : array-like
            Redshifts at which to evaluate the bias.
        order : int, optional
            Bias order to evaluate.

        Returns
        -------
        array-like
            Halo bias values with shape :math:`(N_M, N_z)`.
        """
        pass


    @partial(jax.jit, static_argnums=(0,))
    def _compute_sigma_grid(self, halo_model):
        """
        Compute the interpolation grid for :math:`\\sigma(M, z)`.

        Returns
        -------
        ln_x : array_like
            :math:`\\ln(1+z)` grid.
        ln_M : array_like
            :math:`\\ln M` grid.
        sigma_grid : array_like
            :math:`\\sigma(M, z)` values.
        """
        
        z_grid = halo_model.cosmology._z_grid_pk()
        cparams = halo_model.cosmology._cosmo_params()
        h = cparams["h"]
    
        # Power spectra for all redshifts, shape: (n_k, n_z)
        pk_grid = jax.vmap(lambda zp: halo_model.cosmology.pk(zp, linear=True)[1].flatten())(z_grid).T
    
        # Compute σ²(R, z) and dσ²/dR using TophatVar
        R_grid, var = jax.vmap(halo_model._tophat_instance, in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()  # shape: (n_R,)
    
        # Compute dσ²/dR for each z, output shape: (n_z, n_R)
        dvar_grid = jax.vmap(lambda v: jnp.gradient(v, R_grid), in_axes=0)(var)
    
        # Compute σ(R, z)
        ln_sigma_grid = 0.5 * jnp.log(var)
        sigma_grid = jnp.exp(ln_sigma_grid)
    
        # # Mass grid, shape: (n_R,)
        rho_crit_0 = cparams["Rho_crit_0"]
        Omega0_cb = cparams['Omega0_cb']
        M_grid = 4.0 * jnp.pi / 3.0 * Omega0_cb * rho_crit_0 * (R_grid ** 3) * h ** 3
    
        # Grids for interpolation
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
    
        return ln_x, ln_M, sigma_grid
        

   

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
    def _b1_nu(self, sigmas, z, delta_mean):
        """
        Compute the first-order halo bias :math:`b_1(\\nu)` following
        Tinker et al. (2010).
    
        Parameters
        ----------
        sigmas : array-like
            Variance of the linear density field :math:`\\sigma(R, z)`.
        z : float or array-like
            Redshift(s).
        delta_mean : float or array-like
            Halo overdensity :math:`\\Delta`.
    
        Returns
        -------
        b1 : array-like
            First-order halo bias values.
        """
        y = jnp.log10(delta_mean)
        delta_c = 1.686  # the critical overdensity (slightly redshift-dependent in LCDM), so this is approximate
        
        # Tinker (2010) parameters
        A  = jnp.array(1.0 + 0.24 * y * jnp.exp(-(4.0 / y) ** 4))
        a  = jnp.array(0.44 * y - 0.88)
        B  = jnp.array(0.183)
        b_ = jnp.array(1.5)
        C  = jnp.array((0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y) ** 4)))
        c  = jnp.array(2.4)
    
        nu = delta_c / sigmas
        nu_a = jnp.power(nu, a)
        first = A * (nu_a / (nu_a + delta_c ** a))
        b_nu = 1.0 - first + B * jnp.power(nu, b_) + C * jnp.power(nu, c)
    
        return b_nu


    @partial(jax.jit, static_argnums=(0,))
    def _b2_nu(self, sigmas, z, delta_mean):
        """
        Compute the second-order halo bias :math:`b_2(\\nu)` following
        Tinker et al. (2010).
    
        Parameters
        ----------
        sigmas : array-like
            Variance of the linear density field :math:`\\sigma(R, z)`.
        z : float or array-like
            Redshift(s).
        delta_mean : float or array-like
            Halo overdensity :math:`\\Delta`.
    
        Returns
        -------
        b2 : array-like
            Second-order halo bias values.
        """

        delta_c =  1.686
        nu = (delta_c / sigmas)**2

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


    @partial(jax.jit, static_argnums=(0,4))
    def halo_bias(self, halo_model, m, z, order=1):
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
        halo_model : HaloModel
            Halo-model instance supplying the cosmology and mass definition
            used to evaluate the bias.
        m : array-like
            Halo mass grid.
        z : array-like
            Redshift grid.
        order : int, optional
            Bias order to evaluate. Supported values are ``1`` and ``2``.
        
        Returns
        -------
        array-like
            Halo bias values of the requested order, shape ``(len(m), len(z))``.
        """
       
       
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        ln_x_grid, ln_M_grid, sigma_grid = self._compute_sigma_grid(halo_model)

        # Create the interpolator, the meshgrid, and then stack the points
        _sigma_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), jnp.log(sigma_grid)) 
        zz, mm = jnp.meshgrid(z, m, indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        sigma_M = jnp.exp(_sigma_interp(pts))

        # Handle delta values
        delta_numeric = halo_model.mass_definition._delta_numeric(halo_model.cosmology, z)
        delta_mean = halo_model.mass_definition._convert_reference(
            halo_model.cosmology,
            z,
            delta_numeric,
            from_ref=halo_model.mass_definition.reference,
            to_ref='mean',
        )
        
        # Ensure delta_mean is 1D before indexing
        delta_mean = jnp.atleast_1d(delta_mean)
        delta_mean_2d = delta_mean[:, None] 
        
        # Broadcast to (nz, nm)
        delta_mean_broad = jnp.broadcast_to(delta_mean_2d, sigma_M.shape)

        if order == 1: 
            return self._b1_nu(sigma_M, zz, delta_mean_broad).T
        elif order == 2:
            return self._b2_nu(sigma_M, zz, delta_mean_broad).T
        else:
            raise ValueError("order must be either 1 or 2")



