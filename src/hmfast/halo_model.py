"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from hmfast.literature import mf_T08, bf_T10
from hmfast.emulator_eval import Emulator
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
    
    def __init__(self, cosmo_model=0, mass_model = mf_T08, bias_model = bf_T10):
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
        
        # Load emulator and make sure the required files are loaded outside of jitted functions
        self.emulator = Emulator(cosmo_model=cosmo_model)
        self.cosmo_model = cosmo_model
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")
        self.emulator._load_emulator("PKL")
        
        self.mass_model = mass_model
        self.bias_model = bias_model

        # Create TophatVar instance once
        _, dummy_k = self.emulator.get_pk_at_z(1., params=None, linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)
        self._tophat_instance_dvar = partial(TophatVar(dummy_k, lowring=True, backend='jax', deriv=1))

    
    def create_tracer(self, tracer: str, x=None):
        """
        Create a tracer for this HaloModel using a short string name.
        Supported: "y" (tSZ), "g" (galaxies/HOD)
        """
        if tracer == "y":
            return tracers.tsz.TSZTracer(cosmo_model=self.cosmo_model, x=x)
        elif tracer == "g":
            return tracers.galaxy_hod.GalaxyHODTracer(cosmo_model=self.cosmo_model, x=x)
        else:
            raise ValueError(f"Unknown tracer '{tracer}'. Only 'y' (tSZ) and 'g' (galaxies/HOD) are supported.")
           

    def _compute_hmf_grid(self, params = None):
        """
        Compute σ(R, z) for use in halo mass function and bias.
        Returns:
            R : array_like, radius grid
            sigma : array_like, σ(R, z) values
        """

        z_grid = jnp.linspace(0.0, 8.0, 250) #self.emulator.z_grid() # May need to modify this
        cparams = self.emulator.get_all_cosmo_params(params)   # Merge with defaults using get_all_cosmo_params()
        h, delta = cparams["h"], params['delta'] 
       
        # Power spectra for all redshifts
        P = jax.vmap(lambda zp: self.emulator.get_pk_at_z(zp, params=params, linear=True)[0].flatten())(z_grid).T
        
        # Compute σ²(R, z) and dσ²/dR using TophatVar
        _, ks = self.emulator.get_pk_at_z(1.0, params=params, linear=True)
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
        delta_mean = self.emulator.get_delta_mean_from_delta_crit_at_z(delta, z_grid, params=params)
        hmf_grid = self.mass_model(sigma_grid, z_grid, delta_mean)

        # Compute d n / d ln(M) using d ln(nu) / d ln(R) 
        dlnnu_dlnR_grid = - dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid  / (4.0 * jnp.pi * R_grid**3 * h**3)

        # Also store the z and M grids to interpolate later on
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
        
        return ln_x, ln_M, dn_dlnM_grid, sigma_grid


    @partial(jax.jit, static_argnums=(0,))
    def get_hmf(self, z = jnp.geomspace(0.005, 3.0, 100), m = jnp.geomspace(5e10, 3.5e15, 100), params = None) -> jnp.ndarray:
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
        params = merge_with_defaults(params)
        ln_x, ln_M, dn_dlnM_grid, _ = self._compute_hmf_grid(params=params)

        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x, ln_M), dn_dlnM_grid)
        hmf = _hmf_interp((jnp.log(1.+z), jnp.log(m)))
        return hmf


    @partial(jax.jit, static_argnums=(0,))
    def get_hbf(self, z = jnp.geomspace(0.005, 3.0, 100), m = jnp.geomspace(5e10, 3.5e15, 100), params = None) -> jnp.ndarray:
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
        delta_mean = self.emulator.get_delta_mean_from_delta_crit_at_z(params["delta"], z, params=params)
        
        return self.bias_model(sigma_M, z, delta_mean)


    def interpolate_tracer(self, tracer, z, m, ell_eval, moment=1, params = None):
        """
        Interpolate u_ell values onto a uniform ell grid for multiple m values. 
        """
    
        ells, u_ells = tracer.get_u_ell(z, m, moment=moment, params=params)
            
        # Interpolator function for a single m
        def interpolate_single(ell, u_ell):
            interpolator = jscipy.interpolate.RegularGridInterpolator((ell,), u_ell, method='linear', bounds_error=False, fill_value=None)
            return interpolator(ell_eval)
    
        # Vectorize the interpolation across all m and interpolate
        u_ell_eval = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)(ells, u_ells)
    
        return ell_eval, u_ell_eval


    @partial(jax.jit, static_argnums=(0, 1))
    def get_C_ell_1h(self, tracer, 
                           z = jnp.geomspace(0.005, 3.0, 100), 
                           m = jnp.geomspace(5e10, 3.5e15, 100), 
                           ell = jnp.geomspace(1e2, 3.5e3, 50),  
                           params = None):
        """
        Compute the 1-halo term for C_ell.
        """
        params = merge_with_defaults(params)
        
        # Vectorize u_ell interpolation and mass function over z, and also compute dVdzdOmega
        u_ell_squared_grid = jax.vmap(lambda zp: self.interpolate_tracer(tracer, zp, m, ell, moment=2, params=params)[1])(z)
        dndlnm_grid = jax.vmap(lambda zp: self.get_hmf(zp, m, params=params))(z)
        comov_vol = self.emulator.get_dVdzdOmega_at_z(z, params=params)

        # Expand grids to align with the shape of `result`
        dndlnm_grid_expanded = dndlnm_grid[:, :, None]  # Shape becomes (100, 100, 1)
        comov_vol_expanded = comov_vol[:, None, None]  # Shape becomes (100, 1, 1)
    
        # Perform element-wise multiplication
        integrand = u_ell_squared_grid * dndlnm_grid_expanded * comov_vol_expanded  # Shape becomes (dim_z, dim_m, dim_ell)
        
        logm_grid = jnp.log(m)
    
        # Calculate uniform spacings
        dx_m = logm_grid[1] - logm_grid[0]
        dx_z = z[1] - z[0]
    
        # Function to integrate a single ell slice
        def integrate_single_ell(integrand_slice):
            # integrate over m first, then over z
            partial_m = jnp.trapezoid(integrand_slice, x=logm_grid, dx=dx_m, axis=1)  # shape (n_z,)
            return jnp.trapezoid(partial_m, x=z, dx=dx_z, axis=0)                  # scalar
    
        # Apply vectorized integration along ell axis (axis=2)
        C_ell_1h = jax.vmap(integrate_single_ell, in_axes=2)(integrand)       

        #C_ell_1h = self.apply_damping_1h(z, ell, C_ell_1h, params=params)
                
        return C_ell_1h  

    @partial(jax.jit, static_argnums=(0, 1))
    def get_C_ell_2h(self, tracer,
                           z = jnp.geomspace(0.005, 3.0, 100), 
                           m = jnp.geomspace(5e10, 3.5e15, 100), 
                           ell = jnp.geomspace(1e2, 3.5e3, 50), 
                           params=None):
        """
        Compute the 2-halo term for C_ell.
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100
    
        # Compute mass function and bias
        dndlnm_grid = jax.vmap(lambda z: self.get_hmf(z, m, params=params))(z)
        bias_grid = jax.vmap(lambda z: self.get_hbf(z, m, params=params))(z)
        u_ell_grid = jax.vmap(lambda z: self.interpolate_tracer(tracer, z, m, ell, moment=1, params=params)[1])(z)
        
        # Integrate over mass for each z and ell
        logm_grid = jnp.log(m)
        dx_m = logm_grid[1] - logm_grid[0]
    
        # Function to integrate mass for a single z slice
        def mass_integral(z_idx):
            integrand = dndlnm_grid[z_idx][:, None] * bias_grid[z_idx][:, None] * u_ell_grid[z_idx]  # shape: (n_m, n_ell)
            return jnp.trapezoid(integrand, x=logm_grid, axis=0)  # shape: (n_ell,)
    
        integrals_z = jax.vmap(mass_integral)(jnp.arange(len(z)))  # shape: (n_z, n_ell)
    
        # Square the mass integral
        mass_integral_sq = integrals_z**2  # shape: (n_z, n_ell)
    
        # Linear power spectrum at k = ell / chi(z)
        chi_z = jax.vmap(lambda z: self.emulator.get_angular_distance_at_z(z, params=params))(z) * (1 + z)
        k_grid = (ell[None, :] + 0.5) / chi_z[:, None]  # shape: (n_z, n_ell)

        
        def P_at_k_for_z(z_idx):
            P_z, ks = self.emulator.get_pk_at_z(z[z_idx], params=params, linear=True)
            # P_z and ks are 1D arrays (length n_k). Interpolate P_z at query points k_grid[z_idx].
            # jnp.interp does 1D linear interpolation; out-of-bounds values take edge values.
            return jnp.interp(k_grid[z_idx], ks, P_z)
    
        # Vectorize over z indices -> produces (n_z, n_ell)and convert Pk_lin to comoving
        P_lin_at_k = jax.vmap(P_at_k_for_z)(jnp.arange(z.shape[0])) * h**3  
    
        # --- comoving volume factor (n_z,) already computed ---
        dVdz = self.emulator.get_dVdzdOmega_at_z(z, params=params)  # shape (n_z,)
    
        # Multiply: all shapes now match (n_z, n_ell)
        integrand_z = mass_integral_sq * P_lin_at_k * dVdz[:, None]  # (n_z, n_ell)
    
        # Integrate over z -> result shape (n_ell,)
        C_ell_2h = jnp.trapezoid(integrand_z, x=z, axis=0)
    
        return C_ell_2h




    def apply_damping_1h(self, z, ell, C_ell, params=None):
        """
        Apply 1-halo damping to an angular power spectrum C_ell.
       
        """
        # Extract parameters
        params = merge_with_defaults(params)
        h = params["H0"] / 100.0
        kstar = 0.01
        
        # Compute mean comoving distance over the redshift range
        chi_z = self.emulator.get_angular_distance_at_z(z, params=params) * (1+z) * h  # in Mpc/h
      
        # Convert ell to effective k in h/Mpc
        k_eff = (ell + 0.5) / chi_z
        
        # Compute damping factor: 1 - exp(-(k*h/k*)^2)
        damping_factor = 1.0 - jnp.exp(-jnp.square(k_eff * h / kstar))
        
        # Apply damping
        return C_ell * damping_factor



    @partial(jax.jit, static_argnums=(0, 1, 2))
    def get_C_ell_1h_cross(self, tracer, tracer2=None,
                           z = jnp.geomspace(0.005, 3.0, 100), 
                           m = jnp.geomspace(5e10, 3.5e15, 100), 
                           ell= jnp.geomspace(1e2, 3.5e3, 50),  
                           params = None):
        """
        Compute the 1-halo term for C_ell.
        If tracer2 is provided, compute cross-spectrum 1-halo using u_ell^A * u_ell^B.
        If tracer2 is None, behavior is identical to before (uses moment=2 from tracer).
        """
        params = merge_with_defaults(params)
        
        # Decide whether we're computing a cross-spectrum
        is_cross = tracer2 is not None

        if is_cross:
            # get u_ell for both tracers with moment=1 and multiply
            u_ell_grid_1 = jax.vmap(lambda zp: self.interpolate_tracer(tracer, zp, m, ell, moment=1, params=params)[1])(z)
            u_ell_grid_2 = jax.vmap(lambda zp: self.interpolate_tracer(tracer2, zp, m, ell, moment=1, params=params)[1])(z)
            u1u2_grid = u_ell_grid_1 * u_ell_grid_2
        else:
            # same as original auto case: use moment=2 to get u^2 directly
            u1u2_grid = jax.vmap(lambda zp: self.interpolate_tracer(tracer, zp, m, ell, moment=2, params=params)[1])(z)

        dndlnm_grid = jax.vmap(lambda zp: self.get_hmf(zp, m, params=params))(z)
        comov_vol = self.emulator.get_dVdzdOmega_at_z(z, params=params)

        # Expand grids to align with the shape of `result`
        dndlnm_grid_expanded = dndlnm_grid[:, :, None]  # Shape becomes (n_z, n_m, 1)
        comov_vol_expanded = comov_vol[:, None, None]  # Shape becomes (n_z, 1, 1)
    
        # integrand shape: (n_z, n_m, n_ell)
        integrand = u1u2_grid * dndlnm_grid_expanded * comov_vol_expanded
        
        logm_grid = jnp.log(m)
    
        # Calculate uniform spacings
        dx_m = logm_grid[1] - logm_grid[0]
        dx_z = z[1] - z[0]
    
        # Function to integrate a single ell slice
        def integrate_single_ell(integrand_slice):
            # integrate over m first, then over z
            partial_m = jnp.trapezoid(integrand_slice, x=logm_grid, dx=dx_m, axis=1)  # shape (n_z,)
            return jnp.trapezoid(partial_m, x=z, dx=dx_z, axis=0)                  # scalar
    
        # Apply vectorized integration along ell axis (axis=2)
        C_ell_1h = jax.vmap(integrate_single_ell, in_axes=2)(integrand)       
                
        return C_ell_1h  


    @partial(jax.jit, static_argnums=(0, 1, 2))
    def get_C_ell_2h_cross(self, tracer, tracer2=None,
                            z = jnp.geomspace(0.005, 3.0, 100), 
                            m = jnp.geomspace(5e10, 3.5e15, 100), 
                            ell = jnp.geomspace(1e2, 3.5e3, 50),
                            params = None):
        """
        Compute the 2-halo term for C_ell cross-correlation.
        If tracer2 is provided, compute cross-spectrum 2-halo using
        (integral dndm b u_A) * (integral dndm b u_B) * P_lin.
        If tracer2 is None, same as before (auto-spectrum).
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100

        # Compute mass function and bias
        dndlnm_grid = jax.vmap(lambda z_: self.get_hmf(z_, m, params=params))(z)  # (n_z, n_m)
        bias_grid = jax.vmap(lambda z_: self.get_hbf(z_, m, params=params))(z)     # (n_z, n_m)

        # Interpolate u_ell for tracers
        u_ell_grid_1 = jax.vmap(lambda z_: self.interpolate_tracer(tracer, z_, m, ell, moment=1, params=params)[1])(z)  # (n_z, n_m, n_ell)
        if tracer2 is None:
            u_ell_grid_2 = u_ell_grid_1
        else:
            u_ell_grid_2 = jax.vmap(lambda z_: self.interpolate_tracer(tracer2, z_, m, ell, moment=1, params=params)[1])(z)

        logm_grid = jnp.log(m)

        # Mass integral for a single z slice
        def mass_integral_for_z(dndlnm_z, bias_z, u_ell_z):
            # dndlnm_z: (n_m,), bias_z: (n_m,), u_ell_z: (n_m, n_ell)
            integrand = dndlnm_z[:, None] * bias_z[:, None] * u_ell_z  # (n_m, n_ell)
            return jnp.trapezoid(integrand, x=logm_grid, axis=0)       # (n_ell,)

        # Vectorized mass integrals over all z
        integrals_tr1 = jax.vmap(mass_integral_for_z)(dndlnm_grid, bias_grid, u_ell_grid_1)  # (n_z, n_ell)
        integrals_tr2 = jax.vmap(mass_integral_for_z)(dndlnm_grid, bias_grid, u_ell_grid_2)  # (n_z, n_ell)

        # Multiply the two mass integrals (auto: squares; cross: product)
        mass_integral_prod = integrals_tr1 * integrals_tr2  # (n_z, n_ell)

        # Linear power spectrum at k = ell / chi(z)
        chi_z = jax.vmap(lambda z_: self.emulator.get_angular_distance_at_z(z_, params=params))(z) * (1 + z)
        k_grid = (ell[None, :] + 0.5) / chi_z[:, None]  # (n_z, n_ell)

        def P_at_k_for_z(z_idx):
            P_z, ks = self.emulator.get_pk_at_z(z[z_idx], params=params, linear=True)
            return jnp.interp(k_grid[z_idx], ks, P_z)

        P_lin_at_k = jax.vmap(P_at_k_for_z)(jnp.arange(z.shape[0])) * h**3  # (n_z, n_ell)

        # Comoving volume factor
        dVdz = self.emulator.get_dVdzdOmega_at_z(z, params=params)  # (n_z,)

        # Multiply all factors
        integrand_z = mass_integral_prod * P_lin_at_k * dVdz[:, None]  # (n_z, n_ell)

        # Integrate over z
        C_ell_2h = jnp.trapezoid(integrand_z, x=z, axis=0)  # (n_ell,)

        return C_ell_2h

