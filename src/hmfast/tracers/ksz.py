import jax
import jax.numpy as jnp
from hmfast.halo_model import HaloModel
from hmfast.emulator_eval import Emulator
from functools import partial
from hmfast.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults


class KSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, cosmo_model=0, x=None):
        
        if x is None:
            x = jnp.logspace(jnp.log10(1e-4), jnp.log10(20.0), 512)
        self.x = x
        self.hankel = HankelTransform(x, nu=0.5)

        # Load emulator and make sure the required files are loaded outside of jitted functions
        self.emulator = Emulator(cosmo_model=0)
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")
        self.emulator._load_emulator("PKL")

         # Compute Pk once instantiate grids and thus avoid tracer errors
        _, _ = self.emulator.get_pk_at_z(1., params=None, linear=True) 


    def c_Duffy2008(self, z, m, A=5.71, B=-0.084, C=-0.47, M_pivot=2e12):
        """
        Duffy et al. 2008 mass-concentration relation.
        A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
        """
        return A * (m / M_pivot)**B * (1 + z)**C


    def nfw_density_profile(self, z, m, params=None):
        params = merge_with_defaults(params)

        cparams = self.emulator.get_all_cosmo_params(params)
        delta = params["delta"]
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]   # Baryon fraction
        r_delta = self.emulator.get_r_delta_of_m_delta_at_z(delta, m, z, params=params)
    
        c_delta = self.c_Duffy2008(z, m)
        r_s = r_delta / c_delta
    
        x =  jnp.clip(self.x, 1e-8, None) 
    
        m_nfw = jnp.log(1 + c_delta) - c_delta / (1 + c_delta)
    
        rho_s = m / (4 * jnp.pi * r_s**3 * m_nfw)
        rho_gas = f_b * rho_s / (x * (1 + x)**2)
    
        return rho_gas 


    def b16_density_profile(self, z, m, params=None):
        """
        Battaglia et al. 2016 gas density profile.
        https://arxiv.org/pdf/1607.02442

        Returns rho_gas(x) where x = r/r_200c.
        
        Parameters:
        -----------
        z : float
            Redshift
        m : float
            Halo mass in M_sun/h (M_200c definition)
        params : dict, optional
            Parameter dictionary
        
        Returns:
        --------
        rho_gas : array_like
            Gas density profile in M_sun h^2 / Mpc^3
            Same shape as self.x
        """
        params = merge_with_defaults(params)
        x = self.x
        
        # Get cosmological parameters
        h = params["H0"] / 100.0
        cparams = self.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]   # Baryon fraction
        
        # Critical density at redshift z
        Hz = self.emulator.get_hubble_at_z(z, params=params)  # km/s/Mpc
        H0 = params["H0"]
        E_z = Hz / H0
        
        # Critical density in M_sun h^2 / Mpc^3
        rho_crit_0 = 2.77536627e11
        rho_crit = rho_crit_0 * E_z**2
        
        # Battaglia+16 parameters (Table 2: AGN feedback model)
        A_rho0 = params.get("B16_A_rho0", 4000.0)
        A_alpha = params.get("B16_A_alpha", 0.88)
        A_beta = params.get("B16_A_beta", 3.83)
        
        alpha_m_rho0 = params.get("B16_alpha_m_rho0", 0.29)
        alpha_m_alpha = params.get("B16_alpha_m_alpha", -0.03)
        alpha_m_beta = params.get("B16_alpha_m_beta", 0.04)
        
        alpha_z_rho0 = params.get("B16_alpha_z_rho0", -0.66)
        alpha_z_alpha = params.get("B16_alpha_z_alpha", 0.19)
        alpha_z_beta = params.get("B16_alpha_z_beta", -0.025)
        
        # Extended parameters
        mcut = params.get("B16_mcut", 1e14)  # M_sun
        alphap_m_rho0 = params.get("B16_alphap_m_rho0", 0.29)
        alphap_m_alpha = params.get("B16_alphap_m_alpha", -0.03)
        alphap_m_beta = params.get("B16_alphap_m_beta", 0.04)
        
        alpha_c_rho0 = params.get("B16_alpha_c_rho0", 0.0)
        alpha_c_alpha = params.get("B16_alpha_c_alpha", 0.0)
        alpha_c_beta = params.get("B16_alpha_c_beta", 0.0)
        
        gamma = params.get("B16_gamma", -0.2)
        xc = params.get("B16_xc", 0.5)
        c_delta = params.get("c_200c", 1.0)
        
        # Convert mass to M_sun (not M_sun/h)
        m_200c_msun = m / h
        
        # Select mass scaling exponents based on mass cut
        # Use jnp.where for JAX compatibility
        mass_ratio = m_200c_msun / mcut
        alpha_m_rho0_eff = jnp.where(m_200c_msun > mcut, alpha_m_rho0, alphap_m_rho0)
        alpha_m_alpha_eff = jnp.where(m_200c_msun > mcut, alpha_m_alpha, alphap_m_alpha)
        alpha_m_beta_eff = jnp.where(m_200c_msun > mcut, alpha_m_beta, alphap_m_beta)
        
        # Compute shape parameters (Eq. A1, A2)
        rho0 = A_rho0 * mass_ratio**alpha_m_rho0_eff * (1 + z)**alpha_z_rho0 * (1 + c_delta)**alpha_c_rho0
        alpha = A_alpha * mass_ratio**alpha_m_alpha_eff * (1 + z)**alpha_z_alpha * (1 + c_delta)**alpha_c_alpha
        beta = A_beta * mass_ratio**alpha_m_beta_eff * (1 + z)**alpha_z_beta * (1 + c_delta)**alpha_c_beta
        
        # Profile shape function (dimensionless)
        p_x = (x / xc)**gamma * (1 + (x / xc)**alpha)**(-(beta + gamma) / alpha)

        # Factor of 8.86e10 is required to give the same result as class_sz but should eventually be split into consitutuent constants/factors
        rho_gas = rho0 * rho_crit * f_b * p_x * 8.86e10  
        
        return rho_gas
    

    def _compute_r_and_ell(self, z, m, params=None):
        """
        Helper to compute r_delta and ell_delta for each halo.
        """
        params = merge_with_defaults(params)
        h, B, delta = params['H0']/100, params['B'], params['delta']
        d_A = self.emulator.get_angular_distance_at_z(z, params=params) * h
        r_delta = self.emulator.get_r_delta_of_m_delta_at_z(delta, m, z, params=params) #/ B**(1/3)
        
        ell_delta = d_A / r_delta
        return r_delta, ell_delta


   
    def get_prefactor(self, z, m, params=None):
        """
        Compute kSZ prefactor.
        """

        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params=params)

        # Get relevant quantities
        h = cparams['h'] 
        r_delta, ell_delta = self._compute_r_and_ell(z, m, params=params)
        chi = self.emulator.get_angular_distance_at_z(z, params=params) * h * (1 + z)

        # Compute the root mean squared velocity needed for the kSZ prefactor
        vrms = jnp.sqrt(self.emulator.get_vrms2_at_z(z, params=params))

        # sigmaT / m_prot in (Mpc/h)**2/(Msun/h) which is required for kSZ
        sigma_T_over_m_p = 8.305907197761162e-17 * h  

        # Get full prefactor
        a = 1.0 / (1.0 + z)
        mu_e = 1.14
        f_free = 1
        prefactor =  4 * jnp.pi * r_delta**3 * a * sigma_T_over_m_p * f_free / mu_e * (1 + z)**3 / chi**2 * vrms

        return prefactor, ell_delta


    def get_hankel_integrand(self, z, m, params=None):
        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params=params)

        h  = cparams['h'] 
        r_delta, ell_delta = self._compute_r_and_ell(z, m, params=params)
        x = self.x
        x_min = x[0]  

        # First element in x grid is the smallest, truncate at r_delta (x = r_delta/r_delta = 1)
        W_x = jnp.where((x >= x_min) & (x <= 1), 1.0, 0.0)

        def single_m(m_val):
            rho_gas = self.b16_density_profile(z, m_val, params=params)
            return x**0.5 * rho_gas * W_x
            
        return jax.vmap(single_m)(m)


    def get_u_ell(self, z, m, moment=1, params=None):
        """
        Compute either the first or second moment of the tSZ power spectrum tracer u_ell.
        For tSZ:
            1st moment:  u_ell
            2nd moment:  u_ell^2
        """
        
        params = merge_with_defaults(params)
        
        # Hankel transform
        hankel_integrand = self.get_hankel_integrand(z, m, params=params)
        k, u_k = self.hankel.transform(hankel_integrand)
        u_k *= jnp.sqrt(jnp.pi / (2 * k[None, :]))
    
        # Prefactors and ell-scaling
        prefactor, scale_factor = self.get_prefactor(z, m, params=params)
        ell = k[None, :] * scale_factor[:, None] 
        u_ell_base = prefactor[:, None] * u_k
    
        # Select moment using JAX-safe branching
        moment_funcs = [
            lambda _: u_ell_base,          # moment = 1
            lambda _: u_ell_base**2,       # moment = 2
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell


