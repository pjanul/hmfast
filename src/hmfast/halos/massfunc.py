import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial
from abc import ABC, abstractmethod

from hmfast.halos.mass_definition import MassDefinition


class HaloMass(ABC):
    """
    Abstract base class for halo mass function models.

    Subclasses must implement the public :math:`dn/d\\ln M` evaluator.
    """

    @abstractmethod
    def dndlnm(self, cosmology, m, z, mass_definition=None, convert_masses=False):
        """
        Evaluate :math:`dn/d\\ln M` on a mass-redshift grid.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the halo mass function.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.
        mass_definition : MassDefinition, optional
            Halo mass definition at which to evaluate the halo mass
            function. If omitted, subclasses default to their native
            calibration mass definition.
        convert_masses : bool, optional
            Whether to convert from the native calibration mass definition
            when required.

        Returns
        -------
        array-like
            Halo mass function values :math:`dn/d\\ln M` in comoving
            :math:`\\mathrm{Mpc}^{-3}`.
        """
        pass


class T08HaloMass(HaloMass):
    """
    Halo mass function from `Tinker et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJ...688..709T/abstract>`_.

    Calibrated for spherical-overdensity halo masses. 
    In this implementation, the fitting coefficients are interpolated over the
    tabulated overdensity grid spanning :math:`\\Delta_\\mathrm{m} = 200`
    to :math:`3200`.
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _f_sigma(self, cosmology, sigma, z, mass_definition=MassDefinition(delta=200, reference="mean")):
        """
        Evaluate the internal Tinker et al. (2008) fitting function.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the fitting function.
        sigma : jnp.ndarray
            Root-mean-square linear density fluctuation
            :math:`\\sigma(R, z)`.
        z : float or jnp.ndarray
            Redshift(s) corresponding to ``sigma``.
        mass_definition : MassDefinition, optional
            Halo mass definition used when evaluating the fitting function.
        
    
        Returns
        -------
        f_sigma : jnp.ndarray
            Values of the internal fitting function with shape matching
            ``sigma``.
        """
        
        # Overdensity threshold converted to log scale
        delta_numeric = mass_definition._delta_numeric(cosmology, z)
        delta_mean = mass_definition._convert_reference(
            cosmology,
            z,
            delta_numeric,
            from_ref=mass_definition.reference,
            to_ref='mean',
        )
        delta_mean = jnp.log10(delta_mean)
        
        # Define parameters as JAX arrays
        delta_mean_tab = jnp.log10(jnp.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]))
        A_tab = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        aa_tab = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        b_tab = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        c_tab = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
    
        # Linear interpolation using jnp.interp
        Ap = jnp.interp(delta_mean, delta_mean_tab, A_tab) * (1 + z) ** -0.14
        a = jnp.interp(delta_mean, delta_mean_tab, aa_tab) * (1 + z) ** -0.06
        b = jnp.interp(delta_mean, delta_mean_tab, b_tab) * (1 + z) ** -jnp.power(10, -jnp.power(0.75 / jnp.log10(jnp.power(10, delta_mean) / 75), 1.2))
        c = jnp.interp(delta_mean, delta_mean_tab, c_tab)
        
        # Calculate final result f(σ)
        f_sigma = 0.5 * Ap[:, None] * (jnp.power(sigma / b[:, None], -a[:, None]) + 1) * jnp.exp(-c[:, None] / sigma**2)
        return f_sigma


    @partial(jax.jit, static_argnums=(0,))
    def dndlnm(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="mean"), convert_masses=False):
        """
        Compute the halo mass function :math:`dn/d\\ln M`.
    
        The halo mass function gives the comoving number density of halos per logarithmic mass interval:
    
        .. math::
    
            \\frac{dn}{d\\ln M} = f(\\sigma) \\frac{\\rho_{m,0}}{M} \\left| \\frac{d\\ln \\sigma^{-1}}{d\\ln M} \\right|

        In this model,

        .. math::

            f(\\sigma) = 0.5 A \\left[\\left(\\frac{\\sigma}{b}\\right)^{-a} + 1\\right]
            \\exp\\left(-\\frac{c}{\\sigma^2}\\right),
    
        where :math:`f(\\sigma)` is the Tinker et al. (2008) fitting function,
        calibrated over a tabulated overdensity grid spanning
        :math:`\\Delta_\\mathrm{m} = 200` to :math:`3200`,
        :math:`A`, :math:`a`, :math:`b`, and :math:`c` are redshift-dependent
        fitting parameters, and :math:`\\sigma(M)` is the variance of the
        density field smoothed on the mass scale :math:`M`.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the halo mass function.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.
        mass_definition : MassDefinition, optional
            Halo mass definition at which to evaluate the halo mass
            function. Defaults to the native :math:`200\\mathrm{m}`
            calibration definition.
        convert_masses : bool, optional
            Mass conversions are applied if ``convert_masses`` is set to
            ``True``.
    
        Returns
        -------
        dndlnM : float or array-like
            Halo mass function values :math:`dn/d\\ln M` in comoving
            :math:`\\mathrm{Mpc}^{-3}`, with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
       
        
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        ln_x_grid, ln_M_grid, sigma_grid = cosmology._compute_sigma_grid()
        cparams = cosmology._cosmo_params()
        h = cparams["h"]
        z_grid = jnp.exp(ln_x_grid) - 1.0

        hmf_grid = self._f_sigma(cosmology, sigma_grid, z_grid, mass_definition=mass_definition)

        R_grid = jnp.exp(ln_M_grid / 3.0) / ((4.0 * jnp.pi / 3.0) * cparams['Omega0_cb'] * cparams["Rho_crit_0"])**(1.0 / 3.0)
        var_grid = sigma_grid**2
        dvar_grid = jax.vmap(lambda v: jnp.gradient(v, R_grid), in_axes=0)(var_grid)
        dlnnu_dlnR_grid = -dvar_grid * R_grid / var_grid
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)

        # Create the interpolator, the meshgrid, and then stack the points
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), dn_dlnM_grid)
        mm, zz = jnp.meshgrid(m, z, indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        
        return jnp.squeeze(_hmf_interp(pts))




class T10HaloMass(HaloMass):
    """
    Halo mass function from `Tinker et al. (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T/abstract>`_.

    Calibrated for 200m mass definition.
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _f_sigma(self, cosmology, sigma, z):
        """
        Evaluate the internal Tinker et al. (2010) fitting function.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the fitting function.
        sigma : jnp.ndarray
            Root-mean-square linear density fluctuation
            :math:`\\sigma(R, z)`.
    
        Returns
        -------
        f_nu : jnp.ndarray
            Values of the dimensionless fitting function with shape matching
            ``sigma``.
        """
        delta_c = 1.686
        log_nu = 2.0 * jnp.log(delta_c) - 2.0 * jnp.log(sigma)
        nu = jnp.exp(log_nu)
        
        # Base parameters
        alpha0, beta0, gamma0, eta0, phi0 = 0.368, 0.589, 0.864, -0.243, -0.729
        # Redshift exponents
        alpha_z, beta_z, gamma_z, eta_z, phi_z = 0.0, 0.2, -0.01, 0.27, -0.08

        # Compute z-dependent parameters
        alpha = alpha0 * (1 + z)**alpha_z
        beta  = beta0  * (1 + z)**beta_z
        gamma = gamma0 * (1 + z)**gamma_z
        eta   = eta0   * (1 + z)**eta_z
        phi   = phi0   * (1 + z)**phi_z

        # Reshape for broadcasting
        alpha = alpha[:, None]
        beta  = beta[:, None]
        gamma = gamma[:, None]
        eta   = eta[:, None]
        phi   = phi[:, None]

        beta_term = (beta ** 2 * nu) ** (-phi)  # (beta^2 * nu)^(-phi)
        eta_term  = nu ** eta
        exp_term  = jnp.exp(-gamma * nu / 2)


        f_nu = 0.5 * alpha * (1 + beta_term) * eta_term * exp_term * jnp.sqrt(nu)
        return f_nu

    @partial(jax.jit, static_argnums=(0, 5))
    def dndlnm(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="mean"), convert_masses=False):
        """
        Compute the halo mass function :math:`dn/d\\ln M`.
    
        The halo mass function gives the comoving number density of halos per logarithmic mass interval:
    
        .. math::
    
            \\frac{dn}{d\\ln M} = f(\\sigma) \\frac{\\rho_{m,0}}{M} \\left| \\frac{d\\ln \\sigma^{-1}}{d\\ln M} \\right|

        .. math::

            f(\\nu) = 0.5 \\alpha \\left[1 + (\\beta^2 \\nu)^{-\\phi}\\right]
            \\nu^{\\eta} \\exp\\left(-\\frac{\\gamma \\nu}{2}\\right) \\sqrt{\\nu},
    
        where :math:`\\nu = \\delta_c^2 / \\sigma^2(M)` with
        :math:`\\delta_c = 1.686`, :math:`f(\\nu)` is the fitting function,
        :math:`\\alpha`, :math:`\\beta`, :math:`\\gamma`, :math:`\\eta`, and
        :math:`\\phi` are redshift-dependent fitting parameters,
        :math:`\\rho_{m,0}` is the present-day mean matter density,
        :math:`M` is the halo mass, and :math:`\\sigma(M)` is the variance of
        the density field smoothed on mass scale :math:`M`.
    
        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the halo mass function.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.
        mass_definition : MassDefinition, optional
            Halo mass definition at which to evaluate the halo mass
            function. Defaults to the native :math:`200\\mathrm{m}`
            calibration definition.
        convert_masses : bool, optional
            Mass conversions are applied if ``convert_masses`` is set to
            ``True``.
    
        Returns
        -------
        dndlnM : float or array-like
            Halo mass function values :math:`dn/d\\ln M` in comoving
            :math:`\\mathrm{Mpc}^{-3}`, with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        native_mass_definition = MassDefinition(delta=200, reference="mean")
        key = (mass_definition.delta, mass_definition.reference)
        native_key = (native_mass_definition.delta, native_mass_definition.reference)
        if key != native_key:
            if not convert_masses:
                raise ValueError(f"Mass definition {key} incompatible with the selected halo mass function.")
            raise NotImplementedError("Mass conversion for T10HaloMass is not implemented.")

        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        ln_x_grid, ln_M_grid, sigma_grid = cosmology._compute_sigma_grid()
        cparams = cosmology._cosmo_params()
        h = cparams["h"]
        z_grid = jnp.exp(ln_x_grid) - 1.0

        hmf_grid = self._f_sigma(cosmology, sigma_grid, z_grid)

        R_grid = jnp.exp(ln_M_grid / 3.0) / ((4.0 * jnp.pi / 3.0) * cparams['Omega0_cb'] * cparams["Rho_crit_0"])**(1.0 / 3.0)
        var_grid = sigma_grid**2
        dvar_grid = jax.vmap(lambda v: jnp.gradient(v, R_grid), in_axes=0)(var_grid)
        dlnnu_dlnR_grid = -dvar_grid * R_grid / var_grid
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)

        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), dn_dlnM_grid)
        mm, zz = jnp.meshgrid(m, z, indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)

        return jnp.squeeze(_hmf_interp(pts))





class SubHaloMass(ABC):
    """
    Abstract base class for subhalo mass function models.
    """
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def dndlnmu(self, cosmology, m_host, m_sub):
        """
        Compute the subhalo abundance per logarithmic mass ratio.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology supplied for cosmology-dependent subhalo mass functions.
            The built-in implementations currently depend only on the mass
            ratio and are agnostic of mass definition.
        m_host : float or array_like
            Host halo mass in physical :math:`M_\\odot`.
        m_sub : float or array_like
            Subhalo mass in physical :math:`M_\\odot`.

        Returns
        -------
        array-like
            Dimensionless subhalo abundance :math:`dN/d\\ln \\mu`, i.e.
            number of subhalos per host per logarithmic mass ratio, with shape
            broadcast from ``m_host`` and ``m_sub``, where singleton
            dimensions get squeezed before return.
        """
        pass

class TW10SubHaloMass(SubHaloMass):
    """
    Subhalo mass function from `Tinker & Wetzel (2010) <https://ui.adsabs.harvard.edu/abs/2010ApJ...719...88T/abstract>`_.

    Valid for all host halo masses.
    """
    def __init__(self):
        pass
    
    @partial(jax.jit, static_argnums=(0,))
    def dndlnmu(self, cosmology, m_host, m_sub):
        """
        Compute the Tinker and Wetzel (2010) subhalo mass function.

        .. math::

            \\frac{dN}{d\\ln \\mu} = 0.30 \\mu^{-0.7} \\exp(-9.9 \\mu^{2.5})

        where :math:`\\mu = M_{\\rm sub} / M_{\\rm host}`.
    
        Parameters
        ----------
        cosmology : Cosmology
            This implementation uses only
            :math:`\\mu = M_{\\rm sub} / M_{\\rm host}` and is agnostic of
            mass definition.
        m_host : float or array_like
            Host halo mass in physical :math:`M_\\odot`.
        m_sub : float or array_like
            Subhalo mass in physical :math:`M_\\odot`.
    
        Returns
        -------
        dN_dlnmu : float or array_like
            Dimensionless number of subhalos per host per :math:`d\\ln \\mu`,
            with shape broadcast from ``m_host`` and ``m_sub``, where
            singleton dimensions get squeezed before return.
        """
        mu = m_sub / m_host
        dN_dlnmu = 0.30 * mu ** (-0.7) * jnp.exp(-9.9 * mu ** 2.5)
        return jnp.squeeze(dN_dlnmu)



class JvdB14SubHaloMass(SubHaloMass):
    """
    Subhalo mass function from `Jiang & van den Bosch (2014) <https://ui.adsabs.harvard.edu/abs/2014MNRAS.440..193J/abstract>`_.

    Valid for all host halo masses.
    """
    def __init__(self):
        # Jiang & van den Bosch (2014) parameters
        self.gamma1 = 0.13
        self.alpha1 = -0.83
        self.gamma2 = 1.33
        self.alpha2 = -0.02
        self.beta = 5.67
        self.zeta = 1.19

    @partial(jax.jit, static_argnums=(0,))
    def dndlnmu(self, cosmology, m_host, m_sub):
        """
        Compute the Jiang and van den Bosch (2014) subhalo mass function.

        .. math::

            \\frac{dN}{d\\ln \\mu} =
            (\\gamma_1 \\mu^{\\alpha_1} + \\gamma_2 \\mu^{\\alpha_2})
            \\exp(-\\beta \\mu^{\\zeta})

        where :math:`\\mu = m_{\\rm sub} / m_{\\rm host}` and
        :math:`(\\gamma_1, \\alpha_1, \\gamma_2, \\alpha_2, \\beta, \\zeta)`
        :math:`= (0.13, -0.83, 1.33, -0.02, 5.67, 1.19)` are fitting parameters.
    
        Parameters
        ----------
        cosmology : Cosmology
            This implementation uses only
            :math:`\\mu = M_{\\rm sub} / M_{\\rm host}` and is agnostic of
            mass definition.
        m_host : float or array_like
            Host halo mass in physical :math:`M_\\odot`.
        m_sub : float or array_like
            Subhalo mass in physical :math:`M_\\odot`.
    
        Returns
        -------
        dN_dlnmu : float or array_like
            Dimensionless number of subhalos per host per :math:`d\\ln \\mu`,
            with shape broadcast from ``m_host`` and ``m_sub``, where
            singleton dimensions get squeezed before return.
        """
        
        mu = m_sub / m_host
        dN_dlnmu = (self.gamma1 * mu**self.alpha1 + self.gamma2 * mu**self.alpha2) * \
                jnp.exp(-self.beta * mu**self.zeta)
        return jnp.squeeze(dN_dlnmu)



