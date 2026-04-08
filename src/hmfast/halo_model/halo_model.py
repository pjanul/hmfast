"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from mcfit import TophatVar
from jax.tree_util import register_pytree_node_class

from hmfast.halo_model.mass_function import T08HaloMass
from hmfast.halo_model.bias import T10HaloBias
from hmfast.halo_model.mass_function import TW10SubHaloMass
from hmfast.halo_model.concentration import D08Concentration, B13Concentration
from hmfast.emulator import Emulator
from hmfast.halo_model.mass_definition import MassDefinition
from hmfast.utils import newton_root

jax.config.update("jax_enable_x64", True)


@register_pytree_node_class
class HaloModel:
    """
    A differentiable halo model implementation using JAX.
    
    This class provides methods for computing halo model predictions
    with automatic differentiation capabilities.
    """
    
    def __init__(self, emulator=Emulator(cosmo_model=0), 
                 mass_definition=MassDefinition(delta=200, reference="critical"), 
                 mass_model = T08HaloMass(), 
                 bias_model = T10HaloBias(), 
                 subhalo_mass_model = TW10SubHaloMass(),
                 concentration=D08Concentration(), 
                 hm_consistency=True, 
                 convert_masses=False):
        """
        Initialize the halo model.
        
        Parameters
        ----------
        mass_model : function, default hmf_T08 (i.e. the halo mass function model from Tinker et al 2008)
            Mass function to use.
        bias_model : function, default hbf_T10 (i.e. the halo bias function model from Tinker et al 2010)
            Bias function to use.
        concentration : function, default c_D08 (i.e. the concentration-mass relation from Duffy et al 2008)
            The concentration-mass relation
        """
        
        # Load emulator and make sure the required files are loaded outside of jitted functions (note that DER is needed for CMB lensing tracers)
        self.emulator = emulator 
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")
        self.emulator._load_emulator("PKL")
        self.emulator._load_emulator("DER")
        
        self.mass_model = mass_model
        self.bias_model = bias_model
        self.subhalo_mass_model = subhalo_mass_model
        self.concentration = concentration

        self.mass_definition = mass_definition
        self.hm_consistency = hm_consistency
        self.convert_masses = convert_masses


        # Create TophatVar instance once to instantiate it
        dummy_k, _ = self.emulator.pk(1., linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)


    def tree_flatten(self):
        # The emulator is a Pytree, so it is a child.
        # Everything else is configuration/metadata.
        children = (self.emulator,)
        aux_data = (self.mass_model, self.bias_model, self.subhalo_mass_model, self.concentration,
            self.mass_definition, self.hm_consistency, self.convert_masses, self._tophat_instance
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        emulator, = children
        obj = cls.__new__(cls)
        obj.emulator = emulator
        (obj.mass_model, obj.bias_model, obj.subhalo_mass_model, 
         obj.concentration, obj.mass_definition, obj.hm_consistency, 
         obj.convert_masses, obj._tophat_instance) = aux_data
        return obj

    
    def update(self, **kwargs):
        """
        Updates the underlying emulator parameters without re-initializing the whole class.
        """
        
        new_emulator = self.emulator.update(**kwargs)
        children, aux_data = self.tree_flatten()
        new_instance = self.tree_unflatten(aux_data, (new_emulator,))
        
        return new_instance
       


    @partial(jax.jit, static_argnums=0)
    def c_delta(self, m, z):
        return self.concentration.c_delta(self, m, z)


    #@partial(jax.jit, static_argnums=0)
    def delta_vir_to_crit(self, z):
        """
        Bryan & Norman (1998) virial overdensity for a flat universe.
        Returns Δ_vir relative to the critical density.
    
        Returns
        -------
        float or array
            Δ_vir(z) relative to rho_crit
        """
        omega_m = self.emulator.omega_m(z)
        x = omega_m - 1.0
    
        return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2

    #@partial(jax.jit, static_argnums=0)
    def _delta_numeric(self, z):
        """ 
        Always return numeric delta at redshift z
        in the native reference (self.reference).
        """
        if self.mass_definition.delta == "vir":
            if self.mass_definition.reference != "critical":
                raise ValueError("virial overdensity only defined w.r.t. critical density")
            return self.delta_vir_to_crit(z)
    
        return self.mass_definition.delta


     #@partial(jax.jit, static_argnums=(0,1))
    def convert_reference(self, z, delta, from_ref='critical', to_ref='mean'):
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
            
        omega_m = self.emulator.omega_m(z)
        if from_ref == 'critical' and to_ref == 'mean':
            return delta / omega_m
        elif from_ref == 'mean' and to_ref == 'critical':
            return delta * omega_m
        else:
            raise ValueError("from_ref and to_ref must be 'critical' or 'mean'")



    @partial(jax.jit, static_argnums=(3, 4, 6))
    def convert_m_delta(self, m, z, mass_def_old, mass_def_new, c_old=None, max_iter=20):
       
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        Nm, Nz = len(m), len(z)

        # Vectorized Delta calculation
        def get_delta_crit(mdef, z_val):
            d = jnp.where(mdef.delta == "vir", self.delta_vir_to_crit(z_val), mdef.delta)
            return jnp.where(mdef.reference == "mean", d * self.emulator.omega_m(z_val), d)

        d_old_z, d_new_z = get_delta_crit(mass_def_old, z), get_delta_crit(mass_def_new, z)
        is_same_z = jnp.isclose(d_old_z, d_new_z) & (mass_def_old.reference == mass_def_new.reference)

        # Explicitly handle the grid shapes
        mm, zz = jnp.meshgrid(m, z, indexing='ij')
        
        if c_old is None:
            c_old = self.c_delta(m, z)
        c_old = c_old[:Nm, :Nz].reshape(mm.shape)

        # First guess for the root finder is based on a power law approximation m * (Delta1 / Delta2)^0.2
        x0 = m[:, None] * (d_old_z / d_new_z)[None, :]**0.2   

        # Solver Logic
        def solve_single(m_i, c_i, x0_i, d_o, d_n, same_flag):
            f_nfw = lambda x: jnp.log1p(x) - x / (1.0 + x)
            obj = lambda m_new: m_i / m_new - f_nfw(c_i) / f_nfw(c_i * (m_new/m_i * d_o/d_n)**(1/3))
            
            return jax.lax.cond(same_flag, lambda _: m_i, 
                                lambda _: newton_root(obj, x0=x0_i, max_iter=max_iter), None)

    
        # Broadcast 1D redshift-dependent arrays to match the (Nm, Nz) mesh
        d_o_flat = jnp.broadcast_to(d_old_z[None, :], mm.shape).flatten()
        d_n_flat = jnp.broadcast_to(d_new_z[None, :], mm.shape).flatten()
        same_flat = jnp.broadcast_to(is_same_z[None, :], mm.shape).flatten()

        results = jax.vmap(solve_single)(mm.flatten(), c_old.flatten(), x0.flatten(), d_o_flat, d_n_flat, same_flat)
        
        return results.reshape(mm.shape)


   
    def r_delta(self, m, z, mass_definition=None):
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
        
       
    
        Returns
        -------
        float
            Radius r_delta (e.g., R_200) within which the average density equals delta * rho_crit(z).
        """
        
        mass_definition = self.mass_definition if mass_definition is None else mass_definition

        delta, reference = mass_definition.delta, mass_definition.reference
       
        m = jnp.atleast_1d(m)[:, None]  # (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # (1, Nz)

        # Define your reference density. Default is rho_crit
        rho_ref = self.emulator.critical_density(z)

        # If the user selects vir or rho_mean, correct for this
        if delta == "vir":
            delta = self.delta_vir_to_crit(z)
        
        if reference == "mean":
            rho_ref *= self.emulator.omega_m(z)
            
        return (3.0 * m / (4.0 * jnp.pi * delta * rho_ref))**(1./3.)


    
    @jax.jit
    def counter_terms(self, m, z):
        """
        Compute n_min, b1_min, b2_min counter terms for halo model consistency.
    
        Args:
            m: array-like, mass grid 
            z: array-like, redshift(s)
    
        Returns:
            n_min: array, shape (len(z),)
            b1_min: array, shape (len(z),)
            b2_min: array, shape (len(z),)
        """
       
        m = jnp.atleast_1d(m)
        logm = jnp.log(m)
        cparams = self.emulator.get_all_cosmo_params()
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"]   # Omega0_m without neutrinos
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # (Nm, 1)
    
        # Compute dn/dlnM and bias for each z
        dn_dlnm = self.halo_mass_function(m=m, z=z)  # (Nm, Nz)
        b1 = self.halo_bias(m=m, z=z, order=1)      # (Nm, Nz)
        b2 = self.halo_bias(m=m, z=z, order=2)      # (Nm, Nz)
    
        # Compute integrals I0, I1, I2
        I0 = jnp.trapezoid(dn_dlnm * m_over_rho_mean, x=logm, axis=0)  # (Nz,)
        I1 = jnp.trapezoid(b1 * dn_dlnm * m_over_rho_mean, x=logm, axis=0)
        I2 = jnp.trapezoid(b2 * dn_dlnm * m_over_rho_mean, x=logm, axis=0)
    
        # Apply formulas
        m_min =  m[0]
        n_min =  (1.0 - I0) * rho_mean_0 / m_min
        b1_min = (1.0 - I1) * rho_mean_0 / m_min / n_min
        b2_min = -I2 * rho_mean_0 / m_min / n_min
    
        return n_min, b1_min, b2_min

        
    @jax.jit 
    def _compute_hmf_grid(self):
        """
        Compute σ(R, z) for use in halo mass function and bias.
    
        Returns:
            ln_x : array_like, log(1+z) grid
            ln_M : array_like, log(M) grid
            dn_dlnM_grid : array_like, dn/dlnM grid
            sigma_grid : array_like, σ(R, z) values
        """
        
        z_grid = self.emulator._z_grid_pk()
        cparams = self.emulator.get_all_cosmo_params()
        h = cparams["h"]
    
        # Power spectra for all redshifts, shape: (n_k, n_z)
        pk_grid = jax.vmap(lambda zp: self.emulator.pk(zp, linear=True)[1].flatten())(z_grid).T
    
        # Compute σ²(R, z) and dσ²/dR using TophatVar
        R_grid, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()  # shape: (n_R,)
    
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
        delta_numeric = self._delta_numeric(z_grid)
        delta_mean = self.convert_reference(z_grid, delta_numeric, from_ref=self.mass_definition.reference, to_ref='mean') 
    
        # Halo mass function grid, shape: (n_z, n_R)
        hmf_grid = self.mass_model.f_sigma(sigma_grid, z_grid, delta_mean)
    
        # Compute d n / d ln(M)
        dlnnu_dlnR_grid = -dvar_grid * R_grid / jnp.exp(2. * ln_sigma_grid)
        dn_dlnM_grid = dlnnu_dlnR_grid * hmf_grid / (4.0 * jnp.pi * R_grid**3 * h**3)
    
        # Grids for interpolation
        ln_x = jnp.log(1. + z_grid)
        ln_M = jnp.log(M_grid)
    
        return ln_x, ln_M, dn_dlnM_grid, sigma_grid


    @jax.jit 
    def halo_mass_function(self, m=jnp.geomspace(5e10, 3.5e15, 100), z=jnp.geomspace(0.005, 3.0, 100)) -> jnp.ndarray:
        """
        Compute the halo mass function for arbitrary m and z shapes.
        Returns: dn/dlnM with shape (len(z), len(m))
        """
        
        ln_x_grid, ln_M_grid, dn_dlnM_grid, _ = self._compute_hmf_grid()

        # Create the interpolator, the meshgrid, and then stack the points
        _hmf_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), dn_dlnM_grid)
        mm, zz = jnp.meshgrid(jnp.atleast_1d(m), jnp.atleast_1d(z), indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        
        return _hmf_interp(pts)
        
       

    @partial(jax.jit, static_argnums=(3))
    def halo_bias(self, m=jnp.geomspace(5e10, 3.5e15, 100), z=jnp.geomspace(0.005, 3.0, 100), order=1) -> jnp.ndarray:
        
       
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        ln_x_grid, ln_M_grid, _, sigma_grid = self._compute_hmf_grid()

        # Create the interpolator, the meshgrid, and then stack the points
        _sigma_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_M_grid), jnp.log(sigma_grid)) 
        zz, mm = jnp.meshgrid(z, m, indexing='ij')
        pts = jnp.stack([jnp.log(1. + zz), jnp.log(mm)], axis=-1)
        sigma_M = jnp.exp(_sigma_interp(pts))

        # Handle delta values
        delta_numeric = self._delta_numeric(z)
        delta_mean = self.convert_reference(z, delta_numeric, from_ref=self.mass_definition.reference, to_ref='mean')
        
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
    

    @partial(jax.jit, static_argnums=(1, 2))
    def pk_1h(self, tracer1, tracer2=None, 
              k=jnp.geomspace(1e-3, 10., 100), 
              m=jnp.geomspace(5e10, 3.5e15, 100), 
              z=jnp.geomspace(0.005, 3.0, 100),  
              kstar_damping=0.01):
    
        h = self.emulator.H0 / 100
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Weights and Setup
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
        
        dndlnm = self.halo_mass_function(m, z)
        total_weights = dndlnm * w[:, None] # (Nm, Nz)
    
        is_same_tracer = (tracer2 is None) or (tracer1 == tracer2)
        tracer2 = tracer1 if tracer2 is None else tracer2

    
        # Process a single mass bin at a time and extract the uk^2 at the lowest mass for the halo model consistency term
        def process_bin(i):
            # We need the profiles for index 'i' while squaring uk if the user is doing an autocorrelation
            if is_same_tracer:
                _, uk_sq_full = tracer1.profile.u_k(self, k/h, m, z, moment=2)
                uk_sq_row = uk_sq_full[:, i, :]
            elif tracer1.profile.has_central_contribution and tracer2.profile.has_central_contribution:
                s1, c1 = tracer1.profile.sat_and_cen_contribution(self, k/h, m, z)
                s2, c2 = tracer2.profile.sat_and_cen_contribution(self, k/h, m, z)
                # Pull only row i
                uk_sq_row = s1[:, i, :] * s2[:, i, :] + s1[:, i, :] * c2[:, i, :] + s2[:, i, :] * c1[:, i, :]
            else:
                _, u1 = tracer1.profile.u_k(self, k/h, m, z, moment=1)
                _, u2 = tracer2.profile.u_k(self, k/h, m, z, moment=1)
                uk_sq_row = u1[:, i, :] * u2[:, i, :]
    
            return uk_sq_row * total_weights[i], uk_sq_row
    
        # vmap through the mass bins
        integrand_rows, all_sq_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
    
        pk1h = jnp.sum(integrand_rows, axis=0)
    
        # Apply halo model consistency correction: n_min * uk_sq_min 
        uk_sq_min = all_sq_profiles[0] 
        n_min, _, _ = self.counter_terms(m, z)
        correction = n_min[None, :] * uk_sq_min
        pk1h = pk1h + self.hm_consistency * correction
    
        # Apply damping
        mask = kstar_damping > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, kstar_damping, 1.0))**2), 1.0)
    
        return pk1h * damping[:, None]
            
       
    @partial(jax.jit, static_argnums=(1, 2))
    def cl_1h(self, tracer1, tracer2=None, 
              l=jnp.geomspace(1e2, 3.5e3, 50), 
              m=jnp.geomspace(5e10, 3.5e15, 100), 
              z=jnp.geomspace(0.005, 3.0, 100), 
              kstar_damping=0.01):
        """
        Compute the 1-halo term for angular Cl.
        """
        
        h = self.emulator.H0 / 100

        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function to map l -> k for a specific z
        def get_pk_slice(zi):
            chi_i = self.emulator.angular_diameter_distance(zi) * (1 + zi) 
            ki = (l + 0.5) / chi_i
            pk = self.pk_1h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi), kstar_damping=kstar_damping)
            return pk.flatten()

        # Get the halo model pk_1h, the kernel, and the comoving volume
        P_1h_grid = jax.vmap(get_pk_slice)(z)
        kernel1 = tracer1.kernel(self.emulator, z)  
        kernel2 = tracer2.kernel(self.emulator, z)  
        comov_vol = self.emulator.comoving_volume_element(z) 

        # Integrate over redshift
        integrand = P_1h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)
    


    @partial(jax.jit, static_argnums=(1, 2))
    def pk_2h(self, tracer1, tracer2=None, 
                k=jnp.geomspace(1e-3, 10., 100), 
                m=jnp.geomspace(5e10, 3.5e15, 100), 
                z=jnp.geomspace(0.005, 3.0, 100), 
                ):
        
        cparams = self.emulator.get_all_cosmo_params()
        h, k, m, z = cparams["h"], jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        tracer2 = tracer1 if tracer2 is None else tracer2
    
        # Weights and Ingredients
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        # Combine hmf, bias, and weights into a single (Nm, Nz) weight grid
        dndlnm = self.halo_mass_function(m, z)
        bias = self.halo_bias(m, z)
        total_weights = dndlnm * bias * w[:, None]
    
        def get_I(tracer):
            # This function processes a single index 'i' of the mass axis
            def process_bin(i):
                _, uk_full = tracer.profile.u_k(self, k/h, m, z, moment=1)
                uk_slice = uk_full[:, i, :] 
                return uk_slice * total_weights[i], uk_slice
    
            # Vmap over the indices 0...Nm-1, then integrate and pluck index 0 for hm consistency
            integrand_rows, all_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
            integral = jnp.sum(integrand_rows, axis=0)
            u_k_min = all_profiles[0] # vmap output is (Nm, Nk, Nz)
    
            n_min, b1_min, _ = self.counter_terms(m, z)
            correction = b1_min[None, :] * n_min[None, :] * u_k_min
            
            return integral + self.hm_consistency * correction
    
        # Final Power Spectrum
        I1 = get_I(tracer1)
        I2 = I1 if tracer1 == tracer2 else get_I(tracer2)
        
        P_lin = jax.vmap(lambda zi: jnp.interp(k, *self.emulator.pk(zi, linear=True)))(z).T * h**3
        
        return P_lin * I1 * I2


    @partial(jax.jit, static_argnums=(1, 2))
    def cl_2h(self, tracer1, tracer2=None,
              l=jnp.geomspace(1e2, 3.5e3, 50),
              m=jnp.geomspace(5e10, 3.5e15, 100),
              z=jnp.geomspace(0.005, 3.0, 100), 
              ):
        """
        Compute the 2-halo term for angular cross-power spectrum Cl_12.
        """
        
        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function for Limber integration
        def get_pk_slice(zi):
            # Map l to k using the Limber approximation and then get the pk_2h  
            chi_i = self.emulator.angular_diameter_distance(zi) * (1 + zi) 
            ki = (l + 0.5) / chi_i
            return self.pk_2h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi)).flatten()
    
        # Map over redshift to get P(k=l/chi, z)
        P_2h_grid = jax.vmap(get_pk_slice)(z) 
        
        # Get individual kernels
        kernel1 = tracer1.kernel(self.emulator, z)
        kernel2 = tracer2.kernel(self.emulator, z)
        
        comov_vol = self.emulator.comoving_volume_element(z)
    
        # Limber Integral: C_l = int dz P(k,z) * [W1 * W2 * dV/dz]
        integrand = P_2h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)


    
    

