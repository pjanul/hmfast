import jax
import jax.numpy as jnp


@jax.jit
def hmf_T08(sigmas, z, delta_mean):
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


@jax.jit
def hbf_T10(sigmas, z, delta_mean):
    """
    Tinker et al. (2010) large-scale linear bias, JAX-friendly.

    Parameters
    ----------
    sigmas : jnp.ndarray
        sigma(R,z) or sigma(M,z), shape (nM, n_z)
    z : scalar or array_like
        Redshift(s) (kept for API compatibility)
    delta_mean : scalar or array_like
        Halo overdensity Δ, shape (n_z,) or scalar

    Returns
    -------
    b_nu : jnp.ndarray
        Halo bias, shape same as sigmas
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


def shmf_TW10(M_host, M_sub):
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
   
    mu = M_sub / M_host
    dN_dlnmu = 0.30 * mu ** (-0.7) * jnp.exp(-9.9 * mu ** 2.5)
    return dN_dlnmu




def c_D08(z, m, D=None, A=5.71, B=-0.084, C=-0.47, M_pivot=2e12):
    """
    Duffy et al. (2008) mass-concentration relation for c200_c.
    A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
    """
    return A * (m / M_pivot)**B * (1 + z)**C


def c_SC14(z, m):
    """
    Sanchez-Conde & Prada (2014) concentration-mass relation for c200_c.
    Coefficients are found below Equation 1, https://arxiv.org/pdf/1312.1729
    """
    
    # Coefficients from Eq. 1
    c_array = jnp.array([37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7])
    logM = jnp.log10(m)
    powers = jnp.array([logM**i for i in range(6)])
    poly = jnp.sum(c_array[:, None] * powers, axis=0)
    c200_c = poly * (1 + z) ** -1
    return c200_c


def c_B13(D, m):
    """
    Bhattacharya et al. (2013) mass-concentration relation for c200_c.
    Obtained from Table 2, https://arxiv.org/pdf/1112.5479
    D here is the growth factor D(z).
    """
    # Use the nu as defined in the B13 paper and pivot mass in Msun/h
    nu = (1.12 * (m / 5e13)**0.3 + 0.53) / D
    c200_c = (D**0.54) * 5.9 * nu**(-0.35)
    return c200_c


def c_DM14(z, m):
    """
    Dutton & Macciò (2014) mass-concentration relation for c_vir.

    """
    a = 0.537 + (1.025 - 0.537) * ((1 + z) ** -0.718) ** 1.08  # exponent on (1+z)
    b = -0.097 + 0.024 * z
    c_vir = 10 ** (a + b * jnp.log10(m / 1e12))
    return c_vir
   

