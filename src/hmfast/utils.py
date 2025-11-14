"""
Utility functions for cosmological calculations.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any
from functools import partial
jax.config.update("jax_enable_x64", True)


def interpolate_tracer(z, m, tracer, ell_eval):
    """
    Interpolate u_ell values onto a uniform ell grid for multiple m values. 
    """

    ells, u_ells = tracer.compute_y_ell(z, m)  # compute_y_ell only works for tSZ and should be replaced with a general u_ell tracer function

    # Interpolator function for a single m
    def interpolate_single(ell, u_ell):
        interpolator = jscipy.interpolate.RegularGridInterpolator((ell,), u_ell, method='linear', bounds_error=False, fill_value=None)
        return interpolator(ell_eval)

    # Vectorize the interpolation across all m and interpolate
    u_ell_eval = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)(ells, u_ells)

    return ell_eval, u_ell_eval

    

def trapezoid(y, x=None, dx=1.0, axis=0):
    """
    Trapezoidal rule integration in JAX
        
    Parameters
    ----------
    y : jnp.ndarray
        Values to integrate
    x : jnp.ndarray, optional
        Integration variable
    dx : float
        Grid spacing if x not provided
    axis : int
        Integration axis
        
    Returns
    -------
    jnp.ndarray
        Integrated result
    """
    # Use trapezoidal rule for stability in JAX
    if x is not None:
        return jnp.trapezoid(y, x, axis=axis)
    else:
        return jnp.trapezoid(y, dx=dx, axis=axis)



# Legacy code. Can probably be erased in future versions.
'''
# Predefine some physical constants as JAX arrays (SI units)
c = jnp.array(299792458.0)                    # m/s
G = jnp.array(6.67430e-11)                    # m^3 kg^-1 s^-2
sigma_B = jnp.array(5.670374419e-8)           # W⋅m⁻²⋅K⁻⁴
pi = jnp.array(jnp.pi)
Mpc_over_m = jnp.array(3.0856775814913673e22)  # m in a Mpc
M_sun = jnp.array(1.98847e30)                 # kg


#from tszpower import classy_sz as csz

cosmo_params_class_sz = {
'omega_b': 0.02246576,
'omega_cdm':  0.282064,
'H0': 68.0, # use H0 because this is what is used by the emulators and to avoid any ambiguity when comparing with camb.
'tau_reio': 0.0544,
'ln10^{10}A_s':3.035173309489548,
'n_s': 0.965
}
    


#@staticmethod
def get_all_relevant_params(params: dict) -> dict:
    """
    Compute all relevant cosmological parameters from input base parameters.
    Returns the input dictionary updated with derived parameters.
    """

    h = params["H0"] / 100.0
    Omega_b = params["omega_b"] / h**2
    Omega_cdm = params["omega_cdm"] / h**2

    Omega0_g = (4.0 * sigma_B / c * params["T_cmb"]**4) / (3.0 * c**2 * 1e10 * h**2 / (Mpc_over_m**2) / (8.0 * pi * G))

    Omega0_ur = params["N_ur"] * 7/8 * (4/11)**(4/3) * Omega0_g
    Omega0_ncdm = params["deg_ncdm"] * params["m_ncdm"] / (93.14 * h**2)

    Omega_Lambda = 1.0 - Omega0_g - Omega_b - Omega_cdm - Omega0_ncdm - Omega0_ur

    Omega0_m = Omega_cdm + Omega_b + Omega0_ncdm
    Omega0_r = Omega0_g + Omega0_ur
    Omega0_m_nonu = Omega0_m - Omega0_ncdm
    Omega0_cb = Omega0_m_nonu 

    H0_SI = params["H0"] * 1e3 / c
    Rho_crit_0 = (3.0 / (8.0 * pi * G * M_sun) * Mpc_over_m * c**2 * H0_SI**2 / h**2)   

    return {
        **params, "h": h, "Omega_b": Omega_b, "Omega_cdm": Omega_cdm, "Omega0_g": Omega0_g, "Omega0_ur": Omega0_ur,
        "Omega0_ncdm": Omega0_ncdm, "Omega_Lambda": Omega_Lambda, "Omega0_m": Omega0_m, "Omega0_r": Omega0_r,
        "Omega0_m_nonu": Omega0_m_nonu, "Omega0_cb": Omega0_cb, "Rho_crit_0": Rho_crit_0,
    }

    
class cosmology_utils:
    """Utility functions for cosmological calculations.""" 
        

    def hubble_parameter(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute Hubble parameter H(z).
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            H(z) in units of H0
        """
        h = cosmology.get('h', 0.68)
        Hz = None #csz.get_hubble_at_z(z, params_values_dict = cosmo_params_class_sz) 
        
        return Hz
    
   
    def comoving_distance(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute comoving distance to redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Comoving distance in Mpc/h
        """
        h = cosmology.get('h', 0.68)
        d_A = cosmology_utils.angular_distance(z, cosmology = cosmo_params_class_sz)*h
        d_c = d_A * (1 + z)
        
        return d_c

    def angular_distance(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute angular diameter distance to redshift z.

        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters

        Returns
        -------
        float
            Angular diameter distance in Mpc/h
        """
        # Approximate for flat LCDM
        h = cosmology.get('h', 0.68)
        d_A = None #csz.get_angular_distance_at_z(z,params_values_dict = cosmo_params_class_sz)
        return d_A
    
   
    def dVdzdOmega(z: float, cosmology: Dict[str, float]) -> float:
        """
        Comoving volume element per unit redshift and solid angle.

        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters

        Returns
        -------
        float
            dV / dz / dOmega in (Mpc/h)^3 / sr
        """
        h = cosmology.get('h', 0.68)
        dAz = cosmology_utils.angular_distance(z,cosmology = cosmology) * h
        Hz = cosmology_utils.hubble_parameter(z,cosmology = cosmology) / h # in Mpc^(-1) h

        return (1 + z)**2 * dAz**2 / Hz
    
    
    def critical_density(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute critical density at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Critical density in Msun h^2 / Mpc^3

                    rho_crit = (3./(8.*pi*G*M_sun))*jnp.pow(Mpc_over_m,1)*jnp.pow(c,2)*self.pow(H,2)/self.pow(params_values['h'],2)
        """
        h = cosmology.get('h', 0.68)
        H_z = cosmology_utils.hubble_parameter(z, cosmology) 
      
        rho_crit = (3./ (8. * pi * G * M_sun) ) * Mpc_over_m * c**2 * H_z**2  / h**2
        
        return rho_crit 
    
    
    def virial_radius(z: float, M: float, Delta: float, cosmology: Dict[str, float], reference: int = 0) -> float:
        """
        Compute virial radius for a halo of mass M.
        
        Parameters
        ----------
        z : float
            Redshift
        M : float
            Halo mass in Msun/h
        Delta : float
            Overdensity factor, e.g., 500, 200
        cosmology : dict
            Cosmological parameters
        reference : int
            The matter density used as a reference
            0 = critical density reference
            1 = mean matter density reference
            
        Returns
        -------
        float
            Virial radius in Mpc/h
        """
        h = cosmology.get('h', 0.68)
        Omega_m = cosmology.get('Omega_m', 0.3153)
        rho_crit = cosmology_utils.critical_density(z, cosmology) 
        rho_mean = Omega_m * rho_crit * (1.0 + z) ** 3

        rho_ref = jnp.where(reference == 0, rho_crit, rho_mean)

        # R_vir = (3 M / (4 π Δ ρ_ref))^(1/3); multiply by h to get Mpc/h
        return (3.0 * M / (4.0 * jnp.pi * Delta * rho_ref)) ** (1.0 / 3.0) 
        

    def overdensity_threshold(z: float, Delta: float, cosmology: Dict[str, float], reference: int = 0) -> float:
        """
        Compute the overdensity Δ(z) for halo definitions.

        Parameters
        ----------
        Delta : float
            Overdensity threshold (e.g., 500)
        z : float
            Redshift 
        cosmology : dict
            Cosmological parameters
        reference : int
            The matter density used as a reference
            0 = critical density reference
            1 = mean matter density reference
        

        Returns
        -------
        float
            Mean overdensity 
        """

        params = get_all_relevant_params(cosmology)
            
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        delta_mean = Delta / Omega_m_z
        return delta_mean


def simpson(y, *, x=None, dx=1.0, axis=-1):
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd
        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)
            h = jnp.asarray([dx, dx], dtype=jnp.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))
                diffs = jnp.float64(jnp.diff(x, axis=axis))
                h = [jnp.squeeze(diffs[hm2], axis=axis),
                     jnp.squeeze(diffs[hm1], axis=axis)]
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = jnp.true_divide(
                num,
                den,
            )
            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = jnp.true_divide(
                num,
                den,
            )
            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = jnp.true_divide(
                num,
                den,
            )
            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result

    
def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))
    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = jnp.true_divide(h0, h1)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - jnp.true_divide(1.0, h0divh1)) +
                          y[slice1] * (hsum *
                                       jnp.true_divide(hsum, hprod)) +
                          y[slice2] * (2.0 - h0divh1))
        result = jnp.sum(tmp, axis=axis)
    return result
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

'''


        