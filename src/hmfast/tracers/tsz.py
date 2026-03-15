import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const

jax.config.update("jax_enable_x64", True)

class tSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, halo_model, x=None):

        # Set tracer parameters
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-5), jnp.log10(4.0), 256)
        self.hankel = HankelTransform(self.x, nu=0.5)
        self.profile = self.gnfw_pressure_profile
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        

    def gnfw_pressure_profile(self, x, m, z, params=None):
        """
        GNFW pressure profile as a function of dimensionless scaled radius x = r/r_delta.
        
        Fully vectorized: supports
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
        params = merge_with_defaults(params)
    
        # 1. Ensure all inputs are 1D
        x = jnp.atleast_1d(x)  # (Nx,)
        m = jnp.atleast_1d(m)  # (Nm,)
        z = jnp.atleast_1d(z)  # (Nz,)
    
        # 2. Reshape for broadcasting: (Nx,1,1), (1,Nm,1), (1,1,Nz)
        x_b = x[:, None, None]        # (Nx, 1, 1)
        m_b = m[None, :, None]        # (1, Nm, 1)
        z_b = z[None, None, :]        # (1, 1, Nz)
    
        # 3. Pull needed parameters
        H0, P0, alpha, beta, gamma, B = (params[k] for k in ("H0", "P0GNFW", "alphaGNFW", "betaGNFW", "gammaGNFW", "B"))
    
        # 4. Halo concentration (Nm, Nz) → expand to (1, Nm, Nz) to match x_b
        c_delta = self.halo_model.c_delta(m, z, params=params)  # (Nm, Nz)
        c_delta = c_delta[None, :, :]                            # (1, Nm, Nz)
    
        # 5. Helper variables for normalization
        h = H0 / 100.0
        c_km_s = Const._c_ / 1e3
        H = self.halo_model.emulator.hubble_parameter(z, params=params) * c_km_s  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)
    
        m_delta_tilde = (m / B)[None, :, None]  # (1, Nm, 1)
    
        C = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m_delta_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # 6. Scaled radius and GNFW formula
        scaled_x = c_delta * x_b  # (Nx, Nm, Nz)
        Pe = C * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return Pe  # shape: (Nx, Nm, Nz)
            


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
    
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        k = jnp.atleast_1d(k)
        
        Nk, Nm, Nz = len(k), len(m), len(z)
        
        h, B = params['H0']/100, params['B']
        delta = self.halo_model.delta
        
        r_delta = self.halo_model.r_delta(m, z, delta, params=params) / B**(1/3) # (Nm, Nz)
        d_A = jnp.atleast_1d(self.halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        ell_delta = d_A[None, :] / r_delta  # (Nm, Nz)
        
        Mpc_per_h_to_cm = Const._Mpc_over_m_ / h # This is actually Mpc_per_h_to_m, but the math is currently working
        prefactor = (1 + z)[None, :] * 4 * jnp.pi * r_delta * Mpc_per_h_to_cm / (ell_delta**2)  # (Nm, Nz)
        
        # Target ell grid for interpolation: (Nk, Nz)
        chi = d_A * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
        
        # Get native Hankel transform outputs
        # Assuming u_k_native is (Nk_native, Nm, Nz)
        k_native, u_k_native = self.u_k_hankel(m, z, params=params)  
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None])) 
        ell_native = k_native[:, None, None] * ell_delta[None, :, :] # (Nk_native, Nm, Nz)
        
        # Apply prefactor and moment
        u_ell_base = prefactor[None, :, :] * u_ell_native # (Nk_native, Nm, Nz)
        u_ell_val = jax.lax.select(moment == 1, u_ell_base, u_ell_base**2)
    
        # --- VECTORIZED INTERPOLATION ---
        # We want to interpolate over the native k-axis (axis 0) 
        # for every combination of m and z.
        
        def interp_at_z(ell_t, ell_n, u_n):
            # ell_t: (Nk,), ell_n: (Nk_n,), u_n: (Nk_n,)
            return jnp.interp(ell_t, ell_n, u_n)
    
        # Vmap over Nz (axis 2) and Nm (axis 1)
        # in_axes: ell_target is (Nk, Nz), so we map over axis 1
        # ell_native and u_ell_val are (Nk_native, Nm, Nz), so we map over axis 2 then 1
        vmap_interp = jax.vmap(
            jax.vmap(interp_at_z, in_axes=(None, 1, 1), out_axes=1), 
            in_axes=(1, 2, 2), out_axes=2
        )
        
        # Resulting shape: (Nk, Nm, Nz)
        u_ell_interp = vmap_interp(ell_target, ell_native, u_ell_val)
        
        return ell_target, u_ell_interp
        
