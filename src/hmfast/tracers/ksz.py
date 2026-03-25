import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer#, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const
from hmfast.halo_model.profiles import NFWDensityProfile, B16DensityProfile

jax.config.update("jax_enable_x64", True)

class kSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, halo_model, profile=B16DensityProfile()):


        # Set tracer parameters
        self.profile = profile   # New way 
       

        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        self.halo_model.emulator._load_emulator("PKL")

         # Compute Pk once instantiate grids and thus avoid tracer errors
        _, _ = self.halo_model.emulator.pk_matter(1., params=None, linear=True) 
        

    
    def kernel(self, z, params=None):
        # sigmaT / m_prot in (Mpc/h)**2/(Msun/h) which is required for kSZ
        params = merge_with_defaults(params)
        sigma_T_over_m_p = (Const._sigma_T_ / Const._m_p_) / Const._Mpc_over_m_**2 * Const._M_sun_ * params["H0"] / 100
        return sigma_T_over_m_p / (1.0 + z)
        

    def u_k(self, k, m, z, moment=1, params=None):
        """
        Compute the kSZ tracer u_ell (Nk, Nm, Nz).
        Supports arbitrary input shapes for k, m, and z.
        """
        
        params = merge_with_defaults(params)
        h = params['H0'] / 100.0
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Compute r_delta and ell_delta
        delta = self.halo_model.mass_definition.delta
        r_delta = self.halo_model.r_delta(m, z, params=params)
        d_A_z = jnp.atleast_1d(self.halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        ell_delta = d_A_z[None, :] / r_delta
        
        # chi: (Nz,) -> Target ell grid: (Nk, Nz)
        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
    
        # Calculate kSZ Prefactor as (Nm, Nz)
        vrms = jnp.sqrt(self.halo_model.emulator.v_rms_squared(z, params=params))
        mu_e = 1.14
        f_free = 1.0
        prefactor = (4 * jnp.pi * r_delta**3 * f_free / mu_e * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :])
    
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        #k_native, u_k_native = self.u_k_hankel(m, z, params=params) 
        k_native, u_k_native = self.profile.u_k_hankel(self.halo_model, self.profile.x, m, z, params=params)   # New way
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None]))
        ell_native = k_native[:, None, None] * ell_delta[None, :, :]
        
        # Apply prefactor and moment logic
        u_ell_base = prefactor[None, :, :] * u_ell_native
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # 5. Vectorized Interpolation (Double vmap)
        def interp_single_column(target_x, native_x, native_y):
            return jnp.interp(target_x, native_x, native_y)
    
        # Map over Redshift (Nz) then Mass (Nm)
        vmapped_interp = jax.vmap(
            jax.vmap(interp_single_column, in_axes=(None, 1, 1), out_axes=1),
            in_axes=(1, 2, 2), out_axes=2
        )
        
        u_ell_interp = vmapped_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp
    

