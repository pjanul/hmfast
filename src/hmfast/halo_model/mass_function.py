import jax
import jax.numpy as jnp
from functools import partial


class T08HaloMass:
    """
    Tinker et al. (2008) halo mass function.

    Computes the differential mass function dn/dlnσ for given
    variance σ(R) over a range of redshifts.

    Parameters
    ----------
    sigmas : jnp.ndarray
        Variance of the linear density field σ(R, z), shape (n_R, n_z) or (n_R,)
    z : float or jnp.ndarray
        Redshift(s) corresponding to sigmas
    delta_mean : float or jnp.ndarray
        Halo overdensity Δ (e.g., 200, 500, 1600). Can be scalar or shape (n_z,)

    Returns
    -------
    f_sigma : jnp.ndarray
        Halo mass function values, shape matching sigmas
        (dn/dlnσ) in units consistent with Tinker et al. (2008)
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def f_sigma(self, sigmas, z, delta_mean):
        # Convert delta_mean to log scale
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
        f_sigma = 0.5 * Ap[:, None] * (jnp.power(sigmas / b[:, None], -a[:, None]) + 1) * jnp.exp(-c[:, None] / sigmas**2)
        return f_sigma




class T10HaloMass:
    """
    Tinker et al. (2010) halo mass function f(ν, z).
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def f_sigma(self, sigmas, z, delta_mean):
        """
        Tinker et al. (2010) halo mass function f(nu, z)
        nu : peak-height (delta_c / sigma)
        z  : redshift
        Returns f(nu,z) shape (n_z, n_nu)
        """
        delta_mean = jnp.log10(delta_mean)
        delta_c = 1.686 
        log_nu = 2.0 * jnp.log(delta_c) - 2.0 * jnp.log(sigmas)
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




class TW10SubHaloMass:
    """
    Tinker & Wetzel (2010) subhalo mass function (Eq. 2).

    Parameters
    ----------
    M_host : float or array_like
        Host halo mass [Msun]
    M_sub : float or array_like
        Subhalo mass [Msun]

    Returns
    -------
    dN_dlnmu : float or array_like
        Number of subhalos per host per dln(mu)
    """
    def __init__(self):
        pass
    
    def dndlnmu(self, M_host, M_sub):
        mu = M_sub / M_host
        dN_dlnmu = 0.30 * mu ** (-0.7) * jnp.exp(-9.9 * mu ** 2.5)
        return dN_dlnmu



class JvB14SubHaloMass:
    """
    Jiang & van den Bosch (2014) subhalo mass function (Eq. 21).

    Parameters
    ----------
    M_host : float or array_like
        Host halo mass [Msun]
    M_sub : float or array_like
        Subhalo mass [Msun]

    Returns
    -------
    dN_dlnmu : float or array_like
        Number of subhalos per host per dln(mu)
    """
    def __init__(self):
        # Jiang & van den Bosch (2014) parameters
        self.gamma1 = 0.13
        self.alpha1 = -0.83
        self.gamma2 = 1.33
        self.alpha2 = -0.02
        self.beta = 5.67
        self.zeta = 1.19

    def dndlnmu(self, M_host, M_sub):
        """
        Compute the subhalo mass function per host halo per ln(mu),
        where mu = M_sub / M_host.
        """
        
        mu = M_sub / M_host
        dN_dlnmu = (self.gamma1 * mu**self.alpha1 + self.gamma2 * mu**self.alpha2) * \
                    jnp.exp(-self.beta * mu**self.zeta)
        return dN_dlnmu



