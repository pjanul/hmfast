import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer#, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const
from hmfast.halo_model.profiles import GNFWPressureProfile

jax.config.update("jax_enable_x64", True)

class tSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, halo_model, profile=GNFWPressureProfile()):

        # Set tracer parameters
        self.profile = profile #self.gnfw_pressure_profile
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        

    def kernel(self, z, params=None):
        params = merge_with_defaults(params)
        h = params['H0']/100
        
        # Get electon mass in eV, Thomson cross section in cm^2, and Mpc/h in cm
        m_e = Const._m_e_ * Const._c_**2 / Const._eV_
        sigma_T = Const._sigma_T_ * 1e6
        mpc_per_h_to_cm =  Const._Mpc_over_m_ / h
        return (sigma_T / m_e) / (1+z) # Check this


    def u_k(self, k, m, z, moment=1, params=None):
        
        params = merge_with_defaults(params)
        h, B = params['H0']/100, params['B']
        delta = self.halo_model.mass_definition.delta
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        r_delta = self.halo_model.r_delta(m, z, params=params) / B**(1/3) # (Nm, Nz)
        d_A = jnp.atleast_1d(self.halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        ell_delta = d_A[None, :] / r_delta  # (Nm, Nz)
        
        Mpc_per_h_to_cm = Const._Mpc_over_m_ / h # This is actually Mpc_per_h_to_m, but the math is currently working
        prefactor = (1 + z)[None, :] * 4 * jnp.pi * r_delta * Mpc_per_h_to_cm / (ell_delta**2)  # (Nm, Nz)
        
        # Target ell grid for interpolation: (Nk, Nz)
        chi = d_A * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
        
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        k_native, u_k_native = self.profile.u_k_hankel(self.halo_model, self.profile.x, m, z, params=params)  
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None])) 
        ell_native = k_native[:, None, None] * ell_delta[None, :, :] # (Nk_native, Nm, Nz)
        
        # Apply prefactor and moment
        u_ell_base = prefactor[None, :, :] * u_ell_native # (Nk_native, Nm, Nz)
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # Interpolate over the native k-axis (axis 0) for every combination of m and z    
        def interp_at_z(ell_t, ell_n, u_n):
            return jnp.interp(ell_t, ell_n, u_n)
       
        vmap_interp = jax.vmap(
            jax.vmap(interp_at_z, in_axes=(None, 1, 1), out_axes=1), 
            in_axes=(1, 2, 2), out_axes=2
        )
        
        # Resulting shape: (Nk, Nm, Nz)
        u_ell_interp = vmap_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp
        
