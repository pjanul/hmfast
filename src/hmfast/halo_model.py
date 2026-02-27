"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial

from hmfast.literature import hmf_T08, hbf_T10, c_D08, c_B13
from hmfast.emulator import Emulator
from hmfast.defaults import merge_with_defaults
import hmfast.tracers as tracers
from hmfast.tools.newton_root import newton_root


from mcfit import TophatVar
jax.config.update("jax_enable_x64", True)


class HaloModel:
    """
    A differentiable halo model implementation using JAX.
    
    This class provides methods for computing halo model predictions
    with automatic differentiation capabilities.
    """
    
    def __init__(self, cosmo_model=0, delta = 200, delta_ref = "critical", mass_model = hmf_T08, bias_model = hbf_T10, concentration_relation=c_D08):
        """
        Initialize the halo model.
        
        Parameters
        ----------
        params : dict, optional
            Cosmological parameters. 
        delta : 
            The overdensity criterion relative to the background density
        mass_model : function, default hmf_T08 (i.e. the halo mass function model from Tinker et al 2008)
            Mass function to use.
        bias_model : function, default hbf_T10 (i.e. the halo bias function model from Tinker et al 2010)
            Bias function to use.
        concentration_relation : function, default c_D08 (i.e. the concentration-mass relation from Duffy et al 2008)
            The concentration-mass relation
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
        self.delta_ref = delta_ref 


        # Create TophatVar instance once to instantiate it
        dummy_k, _ = self.emulator.pk_matter(1., params=None, linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)

    # Ensure that delta_ref is only ever critical or mean
    @property
    def delta_ref(self):
        return self._delta_ref
    
    @delta_ref.setter
    def delta_ref(self, value):
        if value not in ("critical", "mean"):
            raise ValueError("delta_ref must be either 'critical' or 'mean'")
        self._delta_ref = value

    
    def omega_m_z(self, z, params=None):
        """
        Compute Ω_m(z) = rho_m(z) / rho_crit(z) without neutrinos.
    
        Returns
        -------
        omega_m : float or array
            Dimensionless matter density at redshift z
        """
        params = merge_with_defaults(params)
        params = self.emulator.get_all_cosmo_params(params)
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        
        return Omega_m_z

    def convert_delta_ref(self, z, delta, from_ref='critical', to_ref='mean', params=None):
        """
        Convert overdensity between 'critical' and 'mean' definitions.
        
        Parameters
        ----------
        delta : float or array
        z : float or array
        from_ref, to_ref : {'critical', 'mean'}
        """
        if from_ref == to_ref:
            return jnp.full_like(z, delta)
            
        omega_m = self.omega_m_z(z, params=params)
        if from_ref == 'critical' and to_ref == 'mean':
            return delta / omega_m
        elif from_ref == 'mean' and to_ref == 'critical':
            return delta * omega_m
        else:
            raise ValueError("from_ref and to_ref must be 'critical' or 'mean'")

         
    def delta_conversion_function(self, z, m_new, m_old, delta_old, delta_new, c_old, params=None):
        """
        Vectorized version: works for scalar or array inputs for z and m_new/m_old.
        Returns F(m_new) = m_new / m_old - f_NFW(c_old) / f_NFW(c_old * r_new / r_old)
        """
        params = merge_with_defaults(params)
        
        r_old = self.r_delta(z, m_old, delta_old, params=params)
        r_new = self.r_delta(z, m_new, delta_new, params=params)
       
        
        def f_nfw(x):
            return jnp.log1p(x) - x / (1.0 + x)
        
        return m_old / m_new - f_nfw(c_old) / f_nfw(c_old * r_new / r_old)


    @partial(jax.jit, static_argnums=0)
    def convert_m_delta(self, z, m, delta_old, delta_new, c_old, x0=None, max_iter=20, params=None):
        """
        Solve for m_{Δ'} given m_old, delta_old, delta_new, c_old, and redshift.
        Fully vectorized: computes all combinations of z, m, and c_old.
        """
        params = merge_with_defaults(params)
        if x0 is None:
            x0 = m
    
        # Make sure 1D arrays
        z = jnp.atleast_1d(z)
        m = jnp.atleast_1d(m)
        c_old = jnp.atleast_1d(c_old)
    
        # Broadcast to common shape
        z, m, c_old, x0 = jnp.broadcast_arrays(z, m, c_old, x0)
    
        # Solve for a single set (scalar z, m, c_old, x0)
        def solve_single(z_i, m_i, c_i, x0_i):
            F = lambda m_new: self.delta_conversion_function(z_i, m_new, m_i, delta_old, delta_new, c_i, params=params)
            return newton_root(F, x0=x0_i, max_iter=max_iter)
    
        # Vectorize over all elements
        solve_vec = jax.vmap(solve_single)
        return solve_vec(z, m, c_old, x0)



    def r_delta(self, z, m, delta, params=None):
        """
        Compute the halo radius corresponding to a given mass and overdensity at redshift z.
    
        Parameters
        ----------
        z : float
            Redshift at which to compute the radius.
        m : float
            Halo mass enclosed within the overdensity radius, in the same units as used for rho_crit.
        delta : float
            Overdensity parameter relative to the critical density (e.g., 200 for M_200).
        
        params : dict, optional
            Dictionary of cosmological parameters to use when computing the critical density.
    
        Returns
        -------
        float
            Radius r_delta (e.g., R_200) within which the average density equals delta * rho_crit(z).
        """
        params = merge_with_defaults(params)
        rho_crit = self.emulator.critical_density(z, params=params)
        return (3.0 * m / (4.0 * jnp.pi * delta * rho_crit))**(1./3.)



    @partial(jax.jit, static_argnums=0)
    def c_delta(self, z, m, params=None):
        params = merge_with_defaults(params)
        return self.concentration_relation(self, z, m, params=params)


    def _compute_hmf_grid(self, params=None):
        """
        Compute σ(R, z) for use in halo mass function and bias.
    
        Returns:
            ln_x : array_like, log(1+z) grid
            ln_M : array_like, log(M) grid
            dn_dlnM_grid : array_like, dn/dlnM grid
            sigma_grid : array_like, σ(R, z) values
        """
        params = merge_with_defaults(params)
        z_grid = self.emulator._z_grid_pk
        cparams = self.emulator.get_all_cosmo_params(params)
        h = cparams["h"]
    
        # Power spectra for all redshifts, shape: (n_k, n_z)
        pk_grid = jax.vmap(lambda zp: self.emulator.pk_matter(zp, params=params, linear=True)[1].flatten())(z_grid).T
    
        # Compute σ²(R, z) and dσ²/dR using TophatVar
        R_grid, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()  # shape: (n_R,)
        # var shape: (n_z, n_R)
    
        # Compute dσ²/dR for each z, output shape: (n_z, n_R)
        dvar_grid = jax.vmap(lambda v: jnp.gradient(v, R_grid), in_axes=0)(var)
    
        # Compute σ(R, z)
        ln_sigma_grid = 0.5 * jnp.log(var)
        sigma_grid = jnp.exp(ln_sigma_grid)
    
        # Mass grid, shape: (n_R,)
        rho_crit_0 = cparams["Rho_crit_0"]
        Omega0_cb = cparams['Omega0_cb']
        M_grid = 4.0 * jnp.pi / 3.0 * Omega0_cb * rho_crit_0 * (R_grid ** 3) * h ** 3
    
        # Overdensity threshold
        delta_mean = self.convert_delta_ref(z_grid, self.delta, from_ref=self.delta_ref, to_ref='mean', params=params) 
    
        # Halo mass function grid, shape: (n_z, n_R)
        hmf_grid = self.mass_model(sigma_grid, z_grid, delta_mean)
    
        # Compute d n / d ln(M)
        dlnnu_dlnR_grid = -dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)
    
        # Grids for interpolation
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
        delta_mean = self.convert_delta_ref(z, self.delta, from_ref=self.delta_ref, to_ref='mean', params=params)
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
                           kstar_damping = 0.01):
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

        # Handle 1h damping if requested (i.e. if kstar_damping is not None)
        chi = self.emulator.angular_diameter_distance(z, params=params) * (1 + z)
        k_grid = (l[None, :] + 0.5) / chi[:, None]
        damping = jax.lax.cond(kstar_damping <= 0.0, lambda _: jnp.ones_like(k_grid), lambda _: 1.0 - jnp.exp(-(k_grid / kstar_damping) ** 2), operand=None)
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

   

    
    
