"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from mcfit import TophatVar

from hmfast.halo_model.mass_function import T08HaloMass
from hmfast.halo_model.bias import T10HaloBias
from hmfast.halo_model.concentration import D08Concentration, B13Concentration
from hmfast.emulator import Emulator
from hmfast.defaults import merge_with_defaults
from hmfast.utils import newton_root

jax.config.update("jax_enable_x64", True)


class HaloModel:
    """
    A differentiable halo model implementation using JAX.
    
    This class provides methods for computing halo model predictions
    with automatic differentiation capabilities.
    """
    
    def __init__(self, cosmo_model=0, delta = 200, delta_ref = "critical", mass_model = T08HaloMass(), bias_model = T10HaloBias(), concentration_relation=D08Concentration()):
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
        
        self._delta = None
        self._delta_ref = None
        self.delta_ref = delta_ref
        self.delta = delta


        # Create TophatVar instance once to instantiate it
        dummy_k, _ = self.emulator.pk_matter(1., params=None, linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)

    # Ensure that delta_ref is only ever critical or mean
    @property
    def delta_ref(self):
        return self._delta_ref
    
    @delta_ref.setter
    def delta_ref(self, value):
        value = str(value).lower()
        if value not in ("critical", "mean"):
            raise ValueError("delta_ref must be either 'critical' or 'mean'")
            
        # Prevent changing delta_ref if delta == "vir"
        if getattr(self, "_delta", None) == "vir" and value != "critical":
            raise ValueError("'vir' is only allowed with 'critical' delta_ref")
        self._delta_ref = value

        
    @property
    def delta(self):
        return self._delta

        
    @delta.setter
    def delta(self, value):
        if isinstance(value, str):
            value = value.lower()
            
        # If 'vir', delta_ref must be 'critical'
        if value == "vir":
            if getattr(self, "_delta_ref", None) != "critical":
                raise ValueError("'vir' is only allowed with 'critical' delta_ref")
            self._delta = value
            return

        # Otherwise, it must be numeric
        if isinstance(value, (int, float)):
            self._delta = value
            return

        raise ValueError("delta must be numeric or 'vir'")


    def delta_vir_to_crit(self, z, params=None):
        """
        Bryan & Norman (1998) virial overdensity for a flat universe.
        Returns Δ_vir relative to the critical density.
    
        Returns
        -------
        float or array
            Δ_vir(z) relative to rho_crit
        """
        omega_m = self.emulator.omega_m_z(z, params=params)
        x = omega_m - 1.0
    
        return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2

    def _delta_numeric(self, z, params=None):
        """ 
        Always return numeric delta at redshift z
        in the native reference (self.delta_ref).
        """
        if self.delta == "vir":
            if self.delta_ref != "critical":
                raise ValueError("virial overdensity only defined w.r.t. critical density")
            return self.delta_vir_to_crit(z, params=params)
    
        return self.delta


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
            
        omega_m = self.emulator.omega_m_z(z, params=params)
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
        
        r_old = self.r_delta(m_old, z, delta_old, params=params)
        r_new = self.r_delta(m_new, z, delta_new, params=params)
       
        
        def f_nfw(x):
            return jnp.log1p(x) - x / (1.0 + x)
        
        return m_old / m_new - f_nfw(c_old) / f_nfw(c_old * r_new / r_old)


    @partial(jax.jit, static_argnums=0)
    def convert_m_delta(self, m, z, delta_old, delta_new, c_old, x0=None, max_iter=20, params=None):
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



    def r_delta(self, m, z, delta, params=None):
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
        #cparams = get_all_cosmo_params(params)

        m = jnp.atleast_1d(m)[:, None]  # (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # (1, Nz)

        # Define your reference density. Default is rho_crit
        rho_ref = self.emulator.critical_density(z, params=params)

        # If the user selects vir or rho_mean, correct for this
        delta = self._delta_numeric(z, params=params)
        if self.delta_ref == "mean":
            rho_ref *= self.emulator.omega_m_z(z, params=params)
            
        return (3.0 * m / (4.0 * jnp.pi * delta * rho_ref))**(1./3.)



    @partial(jax.jit, static_argnums=0)
    def c_delta(self, m, z, params=None):
        params = merge_with_defaults(params)
        return self.concentration_relation.c_delta(self, m, z, params=params)


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
        delta_numeric = self._delta_numeric(z_grid, params=params)
        delta_mean = self.convert_delta_ref(z_grid, delta_numeric, from_ref=self.delta_ref, to_ref='mean', params=params) 
    
        # Halo mass function grid, shape: (n_z, n_R)
        hmf_grid = self.mass_model.f_sigma(sigma_grid, z_grid, delta_mean)
    
        # Compute d n / d ln(M)
        dlnnu_dlnR_grid = -dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)
    
        # Grids for interpolation
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
    
        return ln_x, ln_M, dn_dlnM_grid, sigma_grid


    @partial(jax.jit, static_argnums=(0,))
    def halo_mass_function(self, m=jnp.geomspace(5e10, 3.5e15, 100), z=jnp.geomspace(0.005, 3.0, 100), params=None) -> jnp.ndarray:
        """
        Compute the halo mass function for arbitrary m and z shapes.
        Returns: dn/dlnM with shape (len(z), len(m))
        """
        params = merge_with_defaults(params)
        ln_x_grid, ln_M_grid, dn_dlnM_grid, _ = self._compute_hmf_grid(params=params)

        # Create the interpolator
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), dn_dlnM_grid)
        
        # Create a meshgrid for the input points
        # indexing='ij' ensures output shape is (len(z), len(m))
        zz, mm = jnp.meshgrid(jnp.atleast_1d(z), jnp.atleast_1d(m), indexing='ij')
        
        # Reshape for the interpolator: (N_points, 2)
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        
        return _hmf_interp(pts).T

    @partial(jax.jit, static_argnums=(0, 3))
    def halo_bias(self, m=jnp.geomspace(5e10, 3.5e15, 100), z=jnp.geomspace(0.005, 3.0, 100), order=1, params=None) -> jnp.ndarray:
        params = merge_with_defaults(params)
        
        # Ensure inputs are at least 1D for consistent indexing
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        
        ln_x_grid, ln_M_grid, _, sigma_grid = self._compute_hmf_grid(params=params)
        _sigma_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), jnp.log(sigma_grid))
        
        zz, mm = jnp.meshgrid(z, m, indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        sigma_M = jnp.exp(_sigma_interp(pts))

        # Handle delta values
        delta_numeric = self._delta_numeric(z, params=params)
        delta_mean = self.convert_delta_ref(z, delta_numeric, from_ref=self.delta_ref, to_ref='mean', params=params)
        
        # Ensure delta_mean is 1D before indexing
        delta_mean = jnp.atleast_1d(delta_mean)
        delta_mean_2d = delta_mean[:, None] 
        
        # Broadcast to (nz, nm)
        delta_mean_broad = jnp.broadcast_to(delta_mean_2d, sigma_M.shape)

        if order == 1: 
            return self.bias_model.b1_nu(sigma_M, zz, delta_mean_broad).T
        elif order == 2:
            return self.bias_model.b2_nu(sigma_M, zz, delta_mean_broad).T
        else:
            raise ValueError("order must be either 1 or 2")
            

    def pk_1h(self, tracer, k, m, z, params=None, kstar_damping=0.01):
        # u_k: (nk, nm, nz)
        _, u_k_sq = tracer.u_k(k, m, z, moment=2, params=params)
        
        # hmf: (nz, nm) -> transpose to (nm, nz) -> add k-dim: (1, nm, nz)
        hmf = self.halo_mass_function(m, z, params=params)[None, ...]
        
        # Damping: (nk,) -> (nk, 1, 1)
        # Using jnp.where avoids the boolean tracer error without needing lax.cond
        mask = kstar_damping > 0
        d_vals = 1.0 - jnp.exp(-(k / jnp.where(mask, kstar_damping, 1.0))**2)
        damping = jnp.where(mask, d_vals, 1.0)[:, None, None]
    
        # Integrate over mass (axis 1)
        # (nk, nm, nz) * (1, nm, nz) * (nk, 1, 1) -> (nk, nm, nz)
        integrand = u_k_sq * hmf * damping
        return jnp.trapezoid(integrand, x=jnp.log(m), axis=1) # Result: (nk, nz)
        

    @partial(jax.jit, static_argnums=(0, 1))
    def cl_1h(self, tracer, 
              l=jnp.geomspace(1e2, 3.5e3, 50),
              m=jnp.geomspace(5e10, 3.5e15, 100), 
              z=jnp.geomspace(0.005, 3.0, 100), 
              params=None, 
              kstar_damping=0.01):
        """
        Compute the 1-halo term for angular Cl by mapping l to k 
        at each redshift slice within a vmap.
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100

        # Define the slice function to map l -> k for a specific z
        def get_pk_slice(zi):
            # 1. Compute chi for this specific redshift slice
            # Result: scalar chi_i
            chi_i = self.emulator.angular_diameter_distance(zi, params=params) * (1 + zi) * h
            
            # 2. Map the global l array to a local k array for this slice
            # Result: (nl,) array
            ki = (l + 0.5) / chi_i
            
            # 3. Call pk_1h with the 1D k-vector
            # .flatten() ensures we return (nl,) even if pk_1h returns (1, nl)
            pk = self.pk_1h(tracer, k=ki, m=m, z=jnp.atleast_1d(zi), 
                            params=params, kstar_damping=kstar_damping)
            return pk.flatten()

        # 4. Use vmap to build the (nz, nl) power spectrum grid
        P_1h_grid = jax.vmap(get_pk_slice)(z)

        # 5. Limber Projection
        kernel_grid = tracer.kernel(z, params=params)        # (nz,)
        comov_vol = self.emulator.comoving_volume_element(z, params=params) # (nz,)

        # Integrate over redshift (axis 0) to get (nl,)
        # Weight P_1h_grid (nz, nl) by the kernels (nz, 1)
        integrand = P_1h_grid * (comov_vol[:, None] * kernel_grid[:, None]**2)
        
        return jnp.trapezoid(integrand, x=z, axis=0)
    

    @partial(jax.jit, static_argnums=(0, 1))
    def pk_2h(self, tracer, 
              k=jnp.geomspace(1e-3, 10., 100), 
              m=jnp.geomspace(5e10, 3.5e15, 100), 
              z=jnp.geomspace(0.005, 3.0, 100), 
              params=None):
        """
        Compute the 2-halo term for the 3D power spectrum P(k, z) using 3D grids.
        """
        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params)
        h = cparams["h"]
        
        k = jnp.atleast_1d(k)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        
        # 1. Compute u_k: (nk, nm, nz)
        # moment=1 is required for the 2-halo term
        _, u_k = tracer.u_k(k/h, m, z, moment=1, params=params)
        
        # 2. Get Halo Ingredients: (nz, nm)
        dndlnm = self.halo_mass_function(m, z, params=params) 
        bias = self.halo_bias(m, z, params=params)           
        
        # 3. Alignment for 3D Integration
        # u_k is (nk, nm, nz). We need HMF and Bias as (1, nm, nz)
        # Transpose (nz, nm) -> (nm, nz) then add k-axis
        hmf_aligned = dndlnm[None, ...]
        bias_aligned = bias[None, ...]
        
        # 4. Integrate over lnM (axis 1)
        # (nk, nm, nz) * (1, nm, nz) * (1, nm, nz) -> (nk, nm, nz)
        integrand = u_k * hmf_aligned * bias_aligned
        halo_integral = jnp.trapezoid(integrand, x=jnp.log(m), axis=1) # (nk, nz)
        
        # 5. Linear Power Spectrum (Interp to k for each z)
        def P_at_k_for_z(zi):
            ks, P_z = self.emulator.pk_matter(zi, params=params, linear=True)
            # Assuming pk_matter returns in h-units, adjust if necessary
            return jnp.interp(k, ks, P_z) 
        
        # Map over z, returning (nz, nk) then Transpose to (nk, nz)
        P_lin_at_k = jax.vmap(P_at_k_for_z)(z).T * h**3
        
        # Final P_2h (nk, nz)
        return P_lin_at_k * (halo_integral**2)


    @partial(jax.jit, static_argnums=(0, 1))
    def cl_2h(self, tracer, 
              l=jnp.geomspace(1e2, 3.5e3, 50),
              m=jnp.geomspace(5e10, 3.5e15, 100),
              z=jnp.geomspace(0.005, 3.0, 100), 
              params=None):
        
        params = merge_with_defaults(params)
        h = params["H0"] / 100

        # Note: We only vmap over z. 'l' remains a fixed 1D array inside.
        def get_pk_slice(zi):
            # Calculate chi specifically for this redshift slice
            chi_i = self.emulator.angular_diameter_distance(zi, params=params) * (1 + zi) 
            # Compute k for all ls at this zi
            ki = (l + 0.5) / chi_i
            # pk_2h will return a (1, nl) or (nl,) array
            pk = self.pk_2h(tracer, k=ki, m=m, z=jnp.atleast_1d(zi), params=params) 
            return pk.flatten()
    
        # This produces (nz, nl)
        P_2h_grid = jax.vmap(get_pk_slice)(z) 
    
        kernel_grid = tracer.kernel(z, params=params)
        comov_vol = self.emulator.comoving_volume_element(z, params=params)
    
        # Align (nz, 1) weights with (nz, nl) power spectrum
        integrand = P_2h_grid * (comov_vol[:, None] * kernel_grid[:, None]**2)
        
        return jnp.trapezoid(integrand, x=z, axis=0)
    
