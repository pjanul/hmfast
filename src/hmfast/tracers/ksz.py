import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.utils import Const

jax.config.update("jax_enable_x64", True)

class kSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, halo_model, x=None):


        # Set tracer parameters
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-4), jnp.log10(1.0), 512)
        self.hankel = HankelTransform(self.x, nu=0.5)
        self.profile = self.b16_density_profile

        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        self.halo_model.emulator._load_emulator("PKL")

         # Compute Pk once instantiate grids and thus avoid tracer errors
        _, _ = self.halo_model.emulator.pk_matter(1., params=None, linear=True) 


    def nfw_density_profile(self, x, m, z, params=None):
        params = merge_with_defaults(params)
        
        # 1. Force 1D
        x = jnp.atleast_1d(x) # (Nx,)
        m = jnp.atleast_1d(m) # (Nm,)
        z = jnp.atleast_1d(z) # (Nz,)
    
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # 2. These return (Nm, Nz)
        r_delta = self.halo_model.r_delta(m, z, self.halo_model.delta, params=params)
        c_delta = self.halo_model.c_delta(m, z, params=params)
        r_s = r_delta / c_delta # (Nm, Nz)
        
        # 3. Calculate rho_s
        m_nfw = jnp.log(1 + c_delta) - c_delta / (1 + c_delta) # (Nm, Nz)
        rho_s = m[:, None] / (4 * jnp.pi * r_s**3 * m_nfw)    # (Nm, Nz)
        
        # 4. Final broadcast to (Nx, Nm, Nz)
        # x needs to be (Nx, 1, 1) and rho_s (1, Nm, Nz)
        rho_gas = f_b * rho_s[None, :, :] / (x[:, None, None] * (1 + x[:, None, None])**2)
        
        return rho_gas


    def b16_density_profile(self, x, m, z, params=None):
        """
        Battaglia et al. 2016 gas density profile (AGN feedback model).
        Fully vectorized to support:
            x.shape = (Nx,)
            m.shape = (Nm,)
            z.shape = (Nz,)
        Output shape: (Nx, Nm, Nz)
        """
        params = merge_with_defaults(params)
        
        # 1. Ensure 1D and setup broadcasting shapes
        x = jnp.atleast_1d(x)  # (Nx,)
        m = jnp.atleast_1d(m)  # (Nm,)
        z = jnp.atleast_1d(z)  # (Nz,)
        
        x_b = x[:, None, None]      # (Nx, 1, 1)
        m_b = m[None, :, None]      # (1, Nm, 1)
        z_b = z[None, None, :]      # (1, 1, Nz)
        
        h = params["H0"] / 100.0
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        
        # 2. Critical density and Concentration
        rho_crit_z = jnp.atleast_1d(self.halo_model.emulator.critical_density(z, params=params))
        rho_crit_z = rho_crit_z[None, None, :]  # (1, 1, Nz)
        
        c_delta = self.halo_model.c_delta(m, z, params=params) # (Nm, Nz)
        c_delta = c_delta[None, :, :]  # (1, Nm, Nz)
        
        # 3. Battaglia+16 parameters (Table 2: AGN feedback model)
        # Scaling parameters for rho0, alpha, beta
        A_rho0, A_alpha, A_beta = 4000.0, 0.88, 3.83
        
        # Mass scaling for M > 1e14
        alpha_m_rho0, alpha_m_alpha, alpha_m_beta = 0.29, -0.03, 0.04
        # Redshift scaling
        alpha_z_rho0, alpha_z_alpha, alpha_z_beta = -0.66, 0.19, -0.025
        # Concentration scaling (usually 0 in B16, but kept for completeness)
        alpha_c_rho0, alpha_c_alpha, alpha_c_beta = 0.0, 0.0, 0.0
        
        # Low-mass scaling (M < 1e14)
        alphap_m_rho0, alphap_m_alpha, alphap_m_beta = 0.29, -0.03, 0.04
        
        # 4. Mass scaling logic
        mcut = 1e14  # M_sun
        m_200c_msun = m_b / h
        mass_ratio = m_200c_msun / mcut
        
        # Use jnp.where to choose the mass scaling exponent
        am_rho0 = jnp.where(m_200c_msun > mcut, alpha_m_rho0, alphap_m_rho0)
        am_alpha = jnp.where(m_200c_msun > mcut, alpha_m_alpha, alphap_m_alpha)
        am_beta = jnp.where(m_200c_msun > mcut, alpha_m_beta, alphap_m_beta)
        
        # 5. Compute Shape Parameters (Equations A1, A2 from B16)
        # These result in shape (1, Nm, Nz)
        rho0 = A_rho0 * mass_ratio**am_rho0 * (1 + z_b)**alpha_z_rho0 * (1 + c_delta)**alpha_c_rho0
        alpha = A_alpha * mass_ratio**am_alpha * (1 + z_b)**alpha_z_alpha * (1 + c_delta)**alpha_c_alpha
        beta = A_beta * mass_ratio**am_beta * (1 + z_b)**alpha_z_beta * (1 + c_delta)**alpha_c_beta
        
        gamma = -0.2
        xc = 0.5
        
        # 6. Profile Shape Function (Nx, Nm, Nz)
        # x_b: (Nx,1,1) / rho0: (1,Nm,Nz) -> auto-broadcasts
        p_x = (x_b / xc)**gamma * (1 + (x_b / xc)**alpha)**(-(beta + gamma) / alpha)
        
        # 7. Final result: M_sun h^2 / Mpc^3 (ensure units align with your kernel/prefactor)
        rho_gas = rho0 * rho_crit_z * f_b * p_x 
        
        return rho_gas

    
        
    def kernel(self, z, params=None):
        # sigmaT / m_prot in (Mpc/h)**2/(Msun/h) which is required for kSZ
        params = merge_with_defaults(params)
        sigma_T_over_m_p = (Const._sigma_T_ / Const._m_p_) / Const._Mpc_over_m_**2 * Const._M_sun_ * params["H0"] / 100
        return sigma_T_over_m_p * 1 / (1.0 + z)
        

    def u_k(self, k, m, z, moment=1, params=None):
        """
        Compute the kSZ tracer u_ell (Nk, Nm, Nz).
        Supports arbitrary input shapes for k, m, and z.
        """
        params = merge_with_defaults(params)
    
        # 1. Force all inputs to be at least 1D
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        k = jnp.atleast_1d(k)
        
        Nk, Nm, Nz = len(k), len(m), len(z)
        h = params['H0'] / 100.0
        delta = self.halo_model.delta
    
        # 2. Compute background quantities for broadcasting
        # r_delta: (Nm, Nz), d_A: (Nz,)
        r_delta = self.halo_model.r_delta(m, z, delta, params=params)
        d_A_z = jnp.atleast_1d(self.halo_model.emulator.angular_diameter_distance(z, params=params)) * h
        
        # ell_delta: (Nm, Nz)
        ell_delta = d_A_z[None, :] / r_delta
        
        # chi: (Nz,) -> Target ell grid: (Nk, Nz)
        chi = d_A_z * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
    
        # 3. Calculate kSZ Prefactor as (Nm, Nz)
        # vrms is (Nz,)
        vrms = jnp.sqrt(self.halo_model.emulator.v_rms_squared(z, params=params))
        
        mu_e = 1.14
        f_free = 1.0
        # Prefactor shape: (Nm, Nz)
        prefactor = (4 * jnp.pi * r_delta**3 * f_free / mu_e * (1 + z)[None, :]**3 / chi[None, :]**2 * vrms[None, :])
    
        # 4. Hankel Transform and Native Grid
        # k_native: (Nk_n,), u_k_native: (Nk_n, Nm, Nz)
        k_native, u_k_native = self.u_k_hankel(m, z, params=params)
        
        # Convert native k to native ell
        # u_ell_native: (Nk_n, Nm, Nz), ell_native: (Nk_n, Nm, Nz)
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
    

