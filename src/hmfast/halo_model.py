"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from hmfast.halo_fits import MF_T08, BF_T10
from hmfast.utils import interpolate_tracer
from hmfast.emulator_eval import CosmoEmulator, PkEmulator
from mcfit import TophatVar
jax.config.update("jax_enable_x64", True)


class HaloModel:
    """
    A differentiable halo model implementation using JAX.
    
    This class provides methods for computing halo model predictions
    with automatic differentiation capabilities.
    """
    
    def __init__(self, cosmo_emulator, pk_emulator, params, mass_model = MF_T08, bias_model = BF_T10):
        """
        Initialize the halo model.
        
        Parameters
        ----------
        params : dict, optional
            Cosmological parameters. 
        mass_model : function, default MF_T08 (i.e. the halo mass function model from Tinker et al 2008)
            Mass function to use.
        bias_model : function, default BF_T10 (i.e. the halo bias function model from Tinker et al 2010)
            Bias function to use.
        """
        

        self.cosmo_emulator = cosmo_emulator
        self.pk_emulator = pk_emulator
        self.mass_model = mass_model
        self.bias_model = bias_model
        self.params = self.cosmo_emulator.get_all_relevant_params(params)

        # Create TophatVar instance once
        _, dummy_k = pk_emulator.get_pk_at_z(1., params=self.params, linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)

        # Precompute sigma grid and HMF grid
        self.R_grid, self.sigma_grid = self._compute_sigma_grid()
        self.dndlnm_grid = self._compute_hmf_grid()
        
        self.z_grid = self.cosmo_emulator.z_grid()
        
        # Compute the interpolation functions so they can be rapidly accessed
        self._hmf_interp = jscipy.interpolate.RegularGridInterpolator((jnp.log(1. + self.z_grid), jnp.log(self.M_grid)), jnp.log(self.dndlnm_grid))
        self._sigma_interp = jscipy.interpolate.RegularGridInterpolator((self.M_grid, self.z_grid), self.sigma_grid)


    def _compute_sigma_grid(self):
        """
        Compute σ(R, z) for use in halo mass function and bias.
        Returns:
            R : array_like, radius grid
            sigma : array_like, σ(R, z) values
        """
        params = self.params
        z_grid = self.cosmo_emulator.z_grid()
    
        # Power spectra for all redshifts
        P = jax.vmap(lambda zp: self.pk_emulator.get_pk_at_z(zp, params=params, linear=True)[0].flatten())(z_grid).T
    
        # Compute σ²(R, z)
        R, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(None, 0))(P)
        sigma = jnp.sqrt(var)
    
        return R, sigma


    def _compute_hmf_grid(self):
        """
        Compute halo mass function grid dndlnm(R, z)
        """
    
        params = self.params
        h, delta = params['h'], params['delta']
    
        # Get sigma(R, z) and radius grid
        R_grid, sigma_grid = self._compute_sigma_grid()  # shape: (n_R,), (n_R, n_z)
        z_grid = self.cosmo_emulator.z_grid()
    
        # Compute derivative dσ²/dR using TophatVar
        _, ks = self.pk_emulator.get_pk_at_z(1.0, params=params, linear=True)
        P = jax.vmap(lambda zp: self.pk_emulator.get_pk_at_z(zp, params=params, linear=True)[0].flatten())(z_grid).T
    
        # dσ²/dR grid
        dvar_grid = jax.vmap(lambda pks_col: 
                             TophatVar(ks, lowring=True, backend='jax', deriv=1)(pks_col * ks, extrap=True)[1],
                             in_axes=1)(P)  # shape: (n_z, n_R)
    
        # Compute overdensity threshold and then the HMF
        delta_mean = self.cosmo_emulator.get_delta_mean_from_delta_crit_at_z(delta, z_grid, params=params)
        hmf_grid = self.mass_model(sigma_grid, z_grid, delta_mean)
    
        # Mass grid
        rho_crit_0 = params["Rho_crit_0"] / h**2
        omega0_cb = (params['omega_cdm'] + params['omega_b']) / h**2
        M_grid = 4.0 * jnp.pi / 3.0 * omega0_cb * rho_crit_0 * (R_grid**3) * h**3
        self.M_grid = M_grid

        
        # Compute derivative dlnν/dlnR and convert to dndlnm
        dlnnudlnR_grid = -dvar_grid * R_grid / sigma_grid**2
        dndlnm_grid = dlnnudlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)
    
        return dndlnm_grid
        
    #@partial(jax.jit, static_argnums=(0,))
    def mass_function(self, z: float, M: float) -> jnp.ndarray:
        """
        Compute the halo mass function.
        
        Parameters
        ----------
        z : float
            Redshift
        M : jnp.ndarray
            Halo mass array [Msun/h]
        Returns
        -------
        jnp.ndarray
            Mass function dn/dM [h^3/Mpc^3/Msun]
        """

        hmf_interp = self._hmf_interp
        return jnp.exp(hmf_interp((jnp.log(1.+z), jnp.log(M))))
    
    #@partial(jax.jit, static_argnums=(0,))
    def bias_function(self, z: float, M: float) -> jnp.ndarray:
        """
        Compute the halo bias function.
        
        Parameters
        ----------
        z : float
            Redshift
        M : jnp.ndarray
            Halo mass array [Msun/h]
            
        Returns
        -------
        jnp.ndarray
            Halo bias
        """

        delta = self.params["delta"]
        sigma_interp = self._sigma_interp
        z_array = jnp.full_like(M, z) if jnp.ndim(z) == 0 else z
        points = jnp.stack([M, z_array], axis=-1)
    
        sigma_M = sigma_interp(points)
        delta_mean = self.cosmo_emulator.get_delta_mean_from_delta_crit_at_z(delta, 1, params=self.params)
            
        return self.bias_model(sigma_M, z, delta_mean)


    def get_integral_grid(self, tracer):
        """
        Compute the integrand grid for C_yy: shape (n_z, n_m, n_ell)
        """
    
        # Initiate parameters and set up grids
        z_min, z_max, z_npoints = self.params['z_min'], self.params['z_max'], self.params['z_npoints']
        M_min, M_max, M_npoints = self.params['M_min'], self.params['M_max'], self.params['M_npoints']
        h = self.params["H0"]/100

        # Compute grids
        z_grid = jnp.geomspace(z_min, z_max, z_npoints)
        m_grid = jnp.geomspace(M_min, M_max, M_npoints)
        ell_grid = self.get_ell_grid()

        # Vectorize u_ell interpolation and mass function over z, and also compute dVdzdOmega
        u_ell_grid = jax.vmap(lambda zp: interpolate_tracer(zp, m_grid, tracer, ell_grid)[1])(z_grid)
        dndlnm_grid = jax.vmap(lambda zp: self.mass_function(zp, m_grid))(z_grid)
        comov_vol = self.cosmo_emulator.get_dVdzdOmega_at_z(z_grid, params=self.params)

        # Expand grids to align with the shape of `result`
        dndlnm_grid_expanded = dndlnm_grid[:, :, None]  # Shape becomes (100, 100, 1)
        comov_vol_expanded = comov_vol[:, None, None]  # Shape becomes (100, 1, 1)
    
        # Perform element-wise multiplication
        result = u_ell_grid**2 * dndlnm_grid_expanded * comov_vol_expanded  # Shape becomes (dim_z, dim_m, dim_ell)
        
        return result

    def get_ell_grid(self, lmin=2.0, lmax=1.0e4, dlogell=0.05):
        """
        Generate ell grid with specified parameters.
        
        Parameters
        ----------
        lmin : float
            Minimum multipole
        lmax : float  
            Maximum multipole
        dlogell : float
            Logarithmic spacing in ell
            
        Returns
        -------
        jnp.ndarray
            Array of multipole values
        """
        log10_lmin = jnp.log10(lmin)
        log10_lmax = jnp.log10(lmax)
        num_points = int((log10_lmax - log10_lmin) / dlogell) + 1
        return jnp.logspace(log10_lmin, log10_lmax, num=num_points)

    

    def C_ell_1h(self, tracer, params = None):
        """
        Compute the integrated one halo C_yy term over z and m for each ell.
        """
    
        integrand = self.get_integral_grid(tracer) # shape is (dim_z, dim_m, dim_ell) 
        z_min, z_max, z_npoints, M_min, M_max, M_npoints = (self.params[k] for k in ('z_min', 'z_max', 'z_npoints', 'M_min', 'M_max', 'M_npoints')) 
        
        # Define grids:
        z_grid = jnp.geomspace(z_min, z_max, z_npoints)
        m_grid = jnp.geomspace(M_min, M_max, M_npoints)
        logm_grid = jnp.log(m_grid)
    
        # Calculate uniform spacings
        dx_m = logm_grid[1] - logm_grid[0]
        dx_z = z_grid[1] - z_grid[0]
    
        # Function to integrate a single ell slice
        def integrate_single_ell(integrand_slice):
            # integrate over m first, then over z
            partial_m = jnp.trapezoid(integrand_slice, x=logm_grid, dx=dx_m, axis=1)  # shape (n_z,)
            return jnp.trapezoid(partial_m, x=z_grid, dx=dx_z, axis=0)                  # scalar
    
        # Apply vectorized integration along ell axis (axis=2)
        C_yy = jax.vmap(integrate_single_ell, in_axes=2)(integrand)       
                
        return C_yy  
