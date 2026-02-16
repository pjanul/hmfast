"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from hmfast.literature import hmf_T08, hbf_T10, c_D08
from hmfast.emulator import Emulator
from hmfast.defaults import merge_with_defaults
import hmfast.tracers as tracers


from mcfit import TophatVar
jax.config.update("jax_enable_x64", True)


class HaloModel:
    """
    A differentiable halo model implementation using JAX.
    
    This class provides methods for computing halo model predictions
    with automatic differentiation capabilities.
    """
    
    def __init__(self, cosmo_model=0, delta = 200, mass_model = hmf_T08, bias_model = hbf_T10, concentration_relation=c_D08):
        """
        Initialize the halo model.
        
        Parameters
        ----------
        params : dict, optional
            Cosmological parameters. 
        mass_model : function, default hmf_T08 (i.e. the halo mass function model from Tinker et al 2008)
            Mass function to use.
        bias_model : function, default hbf_T10 (i.e. the halo bias function model from Tinker et al 2010)
            Bias function to use.
        """
        
        # Load emulator and make sure the required files are loaded outside of jitted functions
        self.emulator = Emulator(cosmo_model=cosmo_model)
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")
        self.emulator._load_emulator("PKL")
        
        self.mass_model = mass_model
        self.bias_model = bias_model
        self.concentration_relation = concentration_relation
        self.delta = delta

        # Create TophatVar instance once to instantiate it
        dummy_k, _ = self.emulator.pk_matter(1., params=None, linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)
        self._tophat_instance_dvar = partial(TophatVar(dummy_k, lowring=True, backend='jax', deriv=1))
           

    def _compute_hmf_grid(self, params = None):
        """
        Compute σ(R, z) for use in halo mass function and bias.
        Returns:
            R : array_like, radius grid
            sigma : array_like, σ(R, z) values
        """

        z_grid = self.emulator._z_grid_pk
        cparams = self.emulator.get_all_cosmo_params(params)   
        h = cparams["h"]
       
        # Power spectra for all redshifts
        P = jax.vmap(lambda zp: self.emulator.pk_matter(zp, params=params, linear=True)[1].flatten())(z_grid).T
        
        # Compute σ²(R, z) and dσ²/dR using TophatVar
        ks, _ = self.emulator.pk_matter(1.0, params=params, linear=True)
        R_grid, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(None, 0))(P)
        dvar_grid = jax.vmap(lambda pks_col: self._tophat_instance_dvar(pks_col * ks, extrap=True)[1], in_axes=1)(P) 

        # Take square root as the log for numerical stability, though we need sigma_grid for the hmf/hbf calcs
        ln_sigma_grid = 0.5*jnp.log(var)    
        sigma_grid = jnp.exp(ln_sigma_grid)
        
         # Mass grid
        rho_crit_0 = cparams["Rho_crit_0"] / h**2
        omega0_cb = (params['omega_cdm'] + params['omega_b']) / h**2
        M_grid = 4.0 * jnp.pi / 3.0 * omega0_cb * rho_crit_0 * (R_grid**3) * h**3
    
        # Compute overdensity threshold and then the HMF
        delta_mean = self.emulator.delta_crit_to_mean(self.delta, z_grid, params=params)
        hmf_grid = self.mass_model(sigma_grid, z_grid, delta_mean)

        # Compute d n / d ln(M) using d ln(nu) / d ln(R) 
        dlnnu_dlnR_grid = - dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid  / (4.0 * jnp.pi * R_grid**3 * h**3)

        # Also store the z and M grids to interpolate later on
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
        
        return ln_x, ln_M, dn_dlnM_grid, sigma_grid


    @partial(jax.jit, static_argnums=(0,))
    def halo_mass_function(self, z = jnp.geomspace(0.005, 3.0, 100), m = jnp.geomspace(5e10, 3.5e15, 100), params = None) -> jnp.ndarray:
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
            Mass function dn/dlnM [h^3/Mpc^3/Msun]
        """

        # Compute the hmf values which sets up the interpolator
        params = merge_with_defaults(params)
        ln_x, ln_M, dn_dlnM_grid, _ = self._compute_hmf_grid(params=params)

        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x, ln_M), dn_dlnM_grid)
        hmf = _hmf_interp((jnp.log(1.+z), jnp.log(m)))
        return hmf


    @partial(jax.jit, static_argnums=(0,))
    def halo_bias_function(self, z = jnp.geomspace(0.005, 3.0, 100), m = jnp.geomspace(5e10, 3.5e15, 100), params = None) -> jnp.ndarray:
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
        params = merge_with_defaults(params)
        ln_x, ln_M, _, sigma_grid = self._compute_hmf_grid(params=params)

        _sigma_interp = jscipy.interpolate.RegularGridInterpolator((ln_x, ln_M), jnp.log(sigma_grid))
        sigma_M = jnp.exp(_sigma_interp((jnp.log(1.+z), jnp.log(m))))

        # Get the delta_mean values and pass it to the bias model
        delta_mean = self.emulator.delta_crit_to_mean(self.delta, z, params=params)
        
        return self.bias_model(sigma_M, z, delta_mean)


    def interpolate_tracer(self, tracer, z, m, l_eval, moment=1, params = None):
        """
        Interpolate u_l values onto a uniform l grid for multiple m values. 
        """
    
        ls, u_ls = tracer.get_u_ell(z, m, moment=moment, params=params)
            
        # Interpolator function for a single m
        def interpolate_single(l, u_l):
            interpolator = jscipy.interpolate.RegularGridInterpolator((l,), u_l, method='linear', bounds_error=False, fill_value=None)
            return interpolator(l_eval)
    
        # Vectorize the interpolation across all m and interpolate
        u_l_eval = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)(ls, u_ls)
    
        return l_eval, u_l_eval


    @partial(jax.jit, static_argnums=(0, 1))
    def cl_1h(self, tracer, 
                           z = jnp.geomspace(0.005, 3.0, 100), 
                           m = jnp.geomspace(5e10, 3.5e15, 100), 
                           l = jnp.geomspace(1e2, 3.5e3, 50),
                           params = None, 
                           kstar_damping_1h = 0.01):
        """
        Compute the 1-halo term for cl.
        """
        params = merge_with_defaults(params)
        
        # Vectorize u_l interpolation and mass function over z, and also compute the comoving volume element dVdzdOmega
        u_l_squared_grid = jax.vmap(lambda zp: self.interpolate_tracer(tracer, zp, m, l, moment=2, params=params)[1])(z)
        dndlnm_grid = jax.vmap(lambda zp: self.halo_mass_function(zp, m, params=params))(z)
        comov_vol = self.emulator.comoving_volume_element(z, params=params)

        # Expand grids to align with the shape of `result`
        dndlnm_grid_expanded = dndlnm_grid[:, :, None]  # Shape becomes (100, 100, 1)
        comov_vol_expanded = comov_vol[:, None, None]  # Shape becomes (100, 1, 1)
    
        # Perform element-wise multiplication
        integrand = u_l_squared_grid * dndlnm_grid_expanded * comov_vol_expanded  # Shape becomes (dim_z, dim_m, dim_l)

        # Handle 1h damping if requested (i.e. if kstar_damping_1h is not None)
        chi = self.emulator.angular_diameter_distance(z, params=params) * (1 + z)
        k_grid = (l[None, :] + 0.5) / chi[:, None]
        damping = jax.lax.cond(kstar_damping_1h <= 0.0, lambda _: jnp.ones_like(k_grid), lambda _: 1.0 - jnp.exp(-(k_grid / kstar_damping_1h) ** 2), operand=None)
        integrand *= damping[:, None, :]
    
        # Calculate uniform spacings
        logm_grid = jnp.log(m)
        dx_m = logm_grid[1] - logm_grid[0]
        dx_z = z[1] - z[0]
    
        # Function to integrate a single l slice
        def integrate_single_l(integrand_slice):
            # integrate over m first, then over z
            partial_m = jnp.trapezoid(integrand_slice, x=logm_grid, dx=dx_m, axis=1)  # shape (n_z,)
            return jnp.trapezoid(partial_m, x=z, dx=dx_z, axis=0)                  # scalar
    
        # Apply vectorized integration along l axis (axis=2)
        cl_1h = jax.vmap(integrate_single_l, in_axes=2)(integrand)       
                
        return cl_1h  

    @partial(jax.jit, static_argnums=(0, 1))
    def cl_2h(self, tracer,
                           z = jnp.geomspace(0.005, 3.0, 100), 
                           m = jnp.geomspace(5e10, 3.5e15, 100), 
                           l = jnp.geomspace(1e2, 3.5e3, 50), 
                           params=None):
        """
        Compute the 2-halo term for cl.
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100
    
        # Compute mass function and bias
        dndlnm_grid = jax.vmap(lambda z: self.halo_mass_function(z, m, params=params))(z)
        bias_grid = jax.vmap(lambda z: self.halo_bias_function(z, m, params=params))(z)
        u_l_grid = jax.vmap(lambda z: self.interpolate_tracer(tracer, z, m, l, moment=1, params=params)[1])(z)
        
        # Integrate over mass for each z and l
        logm_grid = jnp.log(m)
        dx_m = logm_grid[1] - logm_grid[0]
    
        # Function to integrate mass for a single z slice
        def mass_integral(z_idx):
            integrand = dndlnm_grid[z_idx][:, None] * bias_grid[z_idx][:, None] * u_l_grid[z_idx]  # shape: (n_m, n_l)
            return jnp.trapezoid(integrand, x=logm_grid, axis=0)  # shape: (n_l,)
    
        integrals_z = jax.vmap(mass_integral)(jnp.arange(len(z)))  # shape: (n_z, n_l)
    
        # Square the mass integral
        mass_integral_sq = integrals_z**2  # shape: (n_z, n_l)
    
        # Linear power spectrum at k = l / chi(z)
        chi_z = jax.vmap(lambda z: self.emulator.angular_diameter_distance(z, params=params))(z) * (1 + z)
        k_grid = (l[None, :] + 0.5) / chi_z[:, None]  # shape: (n_z, n_l)

        
        def P_at_k_for_z(z_idx):
            ks, P_z = self.emulator.pk_matter(z[z_idx], params=params, linear=True)
            # P_z and ks are 1D arrays (length n_k). Interpolate P_z at query points k_grid[z_idx].
            # jnp.interp does 1D linear interpolation; out-of-bounds values take edge values.
            return jnp.interp(k_grid[z_idx], ks, P_z)
    
        # Vectorize over z indices -> produces (n_z, n_l)and convert Pk_lin to comoving
        P_lin_at_k = jax.vmap(P_at_k_for_z)(jnp.arange(z.shape[0])) * h**3  
    
        # --- comoving volume factor (n_z,) already computed ---
        dVdz = self.emulator.comoving_volume_element(z, params=params)  # shape (n_z,)
    
        # Multiply: all shapes now match (n_z, n_l)
        integrand_z = mass_integral_sq * P_lin_at_k * dVdz[:, None]  # (n_z, n_l)
    
        # Integrate over z -> result shape (n_l,)
        cl_2h = jnp.trapezoid(integrand_z, x=z, axis=0)
    
        return cl_2h


