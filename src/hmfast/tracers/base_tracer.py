import jax
import jax.numpy as jnp
import functools
import mcfit
from abc import ABC, abstractmethod
import jax.scipy as jscipy
from jax.scipy.special import sici, erf

from hmfast.defaults import merge_with_defaults



class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """
    def __init__(self, x, nu=0.5):
        #self._x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), num=x_npoints)
        self._hankel = mcfit.Hankel(x, nu=nu, lowring=True, backend='jax')
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


 
class BaseTracer(ABC):
    """
    Abstract base class for cosmological tracers.
    All tracers to inherit from this class, which forces them to have certain callable functions (e.g. get_u_ell() )
    """
    
    def __init__(self, params):
        """
        Initialize the radial grid and Hankel transform.
        """


    def u_k_hankel(self, m, z, params=None):
        """
        Hankel-transform a 3D halo/tracer profile to u_ell for halo model use.
    
        Parameters
        ----------
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        k : array_like, optional
            k values over which the hankel transform will be evaluated. 
            If None, the transform's natural k grid will be output.
            If not None, the transform will be inteprolated to match this k
        params : dict, optional
            Parameter dictionary

        Returns ell, u_ell_m
    
        """

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params=params)
        x = self.x

        h = params['H0']/100
       
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m(m_val):
            profile = self.profile(self.x, m_val, z, params=params)
            return x**0.5 * profile * W_x
            
        hankel_integrand = jax.vmap(single_m)(m)
        k_native, u_k_native = self.hankel.transform(hankel_integrand)

 
        return k_native, u_k_native


    
    def u_k_matter(self, k, m, z, params=None):
        """
        Calculate u^m(k, M, z) via the analytic method using a provided array of k.

         Parameters
        ----------
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        k : array_like
            k values at which to evaluate u_k.
        params : dict, optional
            Parameter dictionary
        
        Returns k, u_k_m
        -------
        u_k_m : array
            Fourier-space halo profile
        """
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
    
        m = jnp.atleast_1d(m)
        h = cparams["h"]
    
        # Concentration and halo radius
        delta = self.halo_model.delta
        c_delta = self.halo_model.c_delta(m, z, params=params)
        r_delta = self.halo_model.r_delta(m, z, delta, params=params)
        lambda_val = 1.0 #params.get("lambda_HOD", 1.0)
    
        k = jnp.atleast_1d(k)
    
        # Broadcast arrays
        k_mat = k[None, :]                        # (1, N_ell)
        r_mat = r_delta[:, None]                  # (N_m, 1)
        c_mat = jnp.atleast_1d(c_delta)[:, None]  # (N_m, 1)
    
        # Si/Ci terms
        q = k_mat * r_mat / c_mat * (1 + z)
        q_scaled = (1 + lambda_val * c_mat) * q
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
    
        # NFW normalization
        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x / (1 + x))
        f_nfw_val = f_nfw(lambda_val * c_mat)
    
        # Fourier-space profile
        u_k_m = (jnp.cos(q) * (Ci_q_scaled - Ci_q)
                   + jnp.sin(q) * (Si_q_scaled - Si_q)
                   - jnp.sin(lambda_val * c_mat * q) / q_scaled) * f_nfw_val 

    
        return k, u_k_m


    
   
    @abstractmethod
    def kernel(self, z, params=None):
        """
        Compute the tracer's radial kernel W(z). All child classes must have a version of this function implemented.
        """
        pass 
   
    @abstractmethod
    def u_k(self, k, m, z, moment=1, params=None):
        """
        Compute the tracer's profile u_k(z,m,k). All child classes must have a version of this function implemented.
        """
        pass 

    

   
  