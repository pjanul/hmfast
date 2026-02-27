import jax
import jax.numpy as jnp
import functools
import mcfit
from abc import ABC, abstractmethod
from hmfast.defaults import merge_with_defaults
from jax.scipy.special import sici, erf 



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


    def u_ell_hankel(self, z, m, x, params=None):
        """
        Hankel-transform a 3D halo/tracer profile to u_ell for halo model use.
    
        Parameters
        ----------
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        x : array_like
            Radial grid.
        params : dict, optional
            Parameter dictionary

        Returns ell, u_ell_m
    
        """

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params=params)

        h = params['H0']/100
        delta = self.halo_model.delta 
        d_A = self.halo_model.emulator.angular_diameter_distance(z, params=params) * h
        r_delta = self.halo_model.r_delta(z, m, delta, params=params) 
        ell_delta = d_A / r_delta
       
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m(m_val):
            profile = self.profile(z, m_val, params=params)
            return x**0.5 * profile * W_x
            
        hankel_integrand = jax.vmap(single_m)(m)
        k, u_k = self.hankel.transform(hankel_integrand)
        u_ell = u_k * jnp.sqrt(jnp.pi / (2 * k[None, :]))
        ell = k[None, :] * ell_delta[:, None] 

        return ell, u_ell

    
    def u_ell_analytic(self, z, m, ell=jnp.geomspace(2, 1e5, 100), params=None):
        """
        Calculate u_ell^m(z, M) via the analytic method (Kusiak et al. 2023),
        using a provided array of multipoles `ell`.

         Parameters
        ----------
        z : float or array_like
            Redshift(s).
        m : float or array_like
            Halo mass(es).
        ell : array_like
            Angular multipoles at which to evaluate u_ell.
        params : dict, optional
            Parameter dictionary
        
        Returns ell, u_ell_m
        -------
        u_ell_m : array
            Fourier-space halo profile
        """
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
    
        m = jnp.atleast_1d(m)
        h = cparams["h"]
    
        # Concentration and halo radius
        delta = self.halo_model.delta
        c_delta = self.halo_model.c_delta(z, m, params=params)
        r_delta = self.halo_model.r_delta(z, m, delta, params=params)
        lambda_val = 1.0 #params.get("lambda_HOD", 1.0)
    
        # Convert ell to k
        chi = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1.0 + z) * h
        ell = jnp.atleast_1d(ell)
        k = (ell + 0.5) / chi

        ell = jnp.broadcast_to(ell[None, :], (m.shape[0], k.shape[0]))
    
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
        u_ell_m = (jnp.cos(q) * (Ci_q_scaled - Ci_q)
                   + jnp.sin(q) * (Si_q_scaled - Si_q)
                   - jnp.sin(lambda_val * c_mat * q) / q_scaled) * f_nfw_val 

    
        return ell, u_ell_m

   
    @abstractmethod
    def get_u_ell(self, z, m, moment=1, params=None):
        """
        Compute u_ell(M,z). All child classes must have a version of this function implemented.
        """
        pass 

   
  