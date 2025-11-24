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
        self.params = params 
        self._cache_params = None

        # Create TophatVar instance once
        _, dummy_k = pk_emulator.get_pk_at_z(1., params=params, linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)
        self._tophat_instance_dvar = partial(TophatVar(dummy_k, lowring=True, backend='jax', deriv=1))
        self.z_grid = self.cosmo_emulator.z_grid()
    
    
    def _compute_sigma_grid(self, params = None):
        """
        Compute σ(R, z) for use in halo mass function and bias.
        Returns:
            R : array_like, radius grid
            sigma : array_like, σ(R, z) values
        """

        z_grid = self.cosmo_emulator.z_grid()
        rparams = self.cosmo_emulator.get_all_relevant_params(params)   # Merge with defaults using get_all_relevant_params()
        h = rparams["h"]
        
        # Power spectra for all redshifts
        P = jax.vmap(lambda zp: self.pk_emulator.get_pk_at_z(zp, params=params, linear=True)[0].flatten())(z_grid).T
    
        # Compute σ²(R, z)
        R_grid, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(None, 0))(P)
        sigma_grid = jnp.sqrt(var)

         # Mass grid
        rho_crit_0 = rparams["Rho_crit_0"] / h**2
        omega0_cb = (params['omega_cdm'] + params['omega_b']) / h**2
        M_grid = 4.0 * jnp.pi / 3.0 * omega0_cb * rho_crit_0 * (R_grid**3) * h**3

        self._sigma_interp = jscipy.interpolate.RegularGridInterpolator((jnp.log(1. + z_grid), jnp.log(M_grid)), jnp.log(sigma_grid))
        return R_grid, sigma_grid, z_grid, M_grid


    def _compute_hmf_grid(self, params = None):
        """
        Compute halo mass function grid dndlnm(R, z)
        """
        pk_emulator = self.pk_emulator 
        h, delta = params['H0']/100, params['delta'] 
    
        # Get sigma(R, z) and radius grid
        R_grid, sigma_grid, z_grid, M_grid = self._compute_sigma_grid(params=params)  # shape: (n_R,), (n_R, n_z)
    
        # Compute derivative dσ²/dR using TophatVar
        _, ks = pk_emulator.get_pk_at_z(1.0, params=params, linear=True)
        P = jax.vmap(lambda zp: pk_emulator.get_pk_at_z(zp, params=params, linear=True)[0].flatten())(z_grid).T
        dvar_grid = jax.vmap(lambda pks_col: self._tophat_instance_dvar(pks_col * ks, extrap=True)[1], in_axes=1)(P) 
    
        # Compute overdensity threshold and then the HMF
        delta_mean = self.cosmo_emulator.get_delta_mean_from_delta_crit_at_z(delta, z_grid, params=params)
        hmf_grid = self.mass_model(sigma_grid, z_grid, delta_mean)
        
        # Compute derivative dlnν/dlnR and convert to dndlnm
        dlnnudlnR_grid = -dvar_grid * R_grid / sigma_grid**2
        dndlnm_grid = dlnnudlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)

        self._hmf_interp = jscipy.interpolate.RegularGridInterpolator((jnp.log(1. + z_grid), jnp.log(M_grid)), jnp.log(dndlnm_grid))
        
        return dndlnm_grid
        
    @partial(jax.jit, static_argnums=(0,))
    def mass_function(self, z: float, M: float, params = None) -> jnp.ndarray:
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

        # Compute the hmf values which sets up the interpolator
        self._compute_hmf_grid(params=params)
        hmf = jnp.exp(self._hmf_interp((jnp.log(1.+z), jnp.log(M))))
        return hmf
    
    #@partial(jax.jit, static_argnums=(0,))
    def bias_function(self, z: float, M: float, params = None) -> jnp.ndarray:
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

    
        # Compute the sigma values which sets up the interpolator
        self._compute_sigma_grid(params = params)
        sigma_M = jnp.exp(self._sigma_interp((jnp.log(1.+z), jnp.log(M))))

        # Get the delta_mean values and pass it to the bias model
        delta_mean = self.cosmo_emulator.get_delta_mean_from_delta_crit_at_z(params["delta"], z, params=params)
        
        return self.bias_model(sigma_M, z, delta_mean)


    

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


    def get_C_ell_1h(self, tracer, params = None):
        """
        Compute the 1-halo term for C_ell.
        """

        # Compute grids
        z_grid = jnp.geomspace(params['z_min'], params['z_max'], params['z_npoints'])
        m_grid = jnp.geomspace(params['M_min'], params['M_max'], params['M_npoints'])
        ell_grid = self.get_ell_grid()

        # Vectorize u_ell interpolation and mass function over z, and also compute dVdzdOmega
        u_ell_grid = jax.vmap(lambda zp: interpolate_tracer(zp, m_grid, tracer, ell_grid)[1])(z_grid)
        dndlnm_grid = jax.vmap(lambda zp: self.mass_function(zp, m_grid, params=params))(z_grid)
        comov_vol = self.cosmo_emulator.get_dVdzdOmega_at_z(z_grid, params=params)

        # Expand grids to align with the shape of `result`
        dndlnm_grid_expanded = dndlnm_grid[:, :, None]  # Shape becomes (100, 100, 1)
        comov_vol_expanded = comov_vol[:, None, None]  # Shape becomes (100, 1, 1)
    
        # Perform element-wise multiplication
        integrand = u_ell_grid**2 * dndlnm_grid_expanded * comov_vol_expanded  # Shape becomes (dim_z, dim_m, dim_ell)
        
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




        
    def get_C_ell_2h(self, tracer, params=None):
        """
        Compute the 2-halo term for C_ell.
        """
        h = params["H0"] / 100
        z_grid = jnp.geomspace(params['z_min'], params['z_max'], params['z_npoints'])
        m_grid = jnp.geomspace(params['M_min'], params['M_max'], params['M_npoints'])
        ell_grid = self.get_ell_grid()
    
        # Compute mass function and bias
        dndlnm_grid = jax.vmap(lambda z: self.mass_function(z, m_grid, params=params))(z_grid)
        bias_grid = jax.vmap(lambda z: self.bias_function(z, m_grid, params=params))(z_grid)
        u_ell_grid = jax.vmap(lambda z: interpolate_tracer(z, m_grid, tracer, ell_grid)[1])(z_grid)
        
        # Integrate over mass for each z and ell
        logm_grid = jnp.log(m_grid)
        dx_m = logm_grid[1] - logm_grid[0]
    
        # Function to integrate mass for a single z slice
        def mass_integral(z_idx):
            integrand = dndlnm_grid[z_idx][:, None] * bias_grid[z_idx][:, None] * u_ell_grid[z_idx]  # shape: (n_m, n_ell)
            return jnp.trapezoid(integrand, x=logm_grid, axis=0)  # shape: (n_ell,)
    
        integrals_z = jax.vmap(mass_integral)(jnp.arange(len(z_grid)))  # shape: (n_z, n_ell)
    
        # Square the mass integral
        mass_integral_sq = integrals_z**2  # shape: (n_z, n_ell)
    
        # Linear power spectrum at k = ell / chi(z)
        chi_z = jax.vmap(lambda z: self.cosmo_emulator.get_angular_distance_at_z(z, params=params))(z_grid) * (1 + z_grid)
        k_grid = (ell_grid[None, :] + 0.5) / chi_z[:, None]  # shape: (n_z, n_ell)

        
        def P_at_k_for_z(z_idx):
            P_z, ks = self.pk_emulator.get_pk_at_z(z_grid[z_idx], params=params, linear=True)
            # P_z and ks are 1D arrays (length n_k). Interpolate P_z at query points k_grid[z_idx].
            # jnp.interp does 1D linear interpolation; out-of-bounds values take edge values.
            return jnp.interp(k_grid[z_idx], ks, P_z)
    
        # Vectorize over z indices -> produces (n_z, n_ell)and convert Pk_lin to comoving
        P_lin_at_k = jax.vmap(P_at_k_for_z)(jnp.arange(z_grid.shape[0])) * h**3  
    
        # --- comoving volume factor (n_z,) already computed ---
        dVdz = self.cosmo_emulator.get_dVdzdOmega_at_z(z_grid, params=params)  # shape (n_z,)
    
        # Multiply: all shapes now match (n_z, n_ell)
        integrand_z = mass_integral_sq * P_lin_at_k * dVdz[:, None]  # (n_z, n_ell)
    
        # Integrate over z -> result shape (n_ell,)
        C_ell_2h = jnp.trapezoid(integrand_z, x=z_grid, axis=0)
    
        return C_ell_2h

