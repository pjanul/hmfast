import jax
import jax.numpy as jnp

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults



class KSZTracer(BaseTracer):
    """
    tSZ tracer using GNFW profile.
    """
    def __init__(self, halo_model, x=None):


        # Set tracer parameters
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-4), jnp.log10(20.0), 512)
        self.hankel = HankelTransform(self.x, nu=0.5)
        self.profile = self.b16_density_profile

        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        self.halo_model.emulator._load_emulator("PKL")

         # Compute Pk once instantiate grids and thus avoid tracer errors
        _, _ = self.halo_model.emulator.pk_matter(1., params=None, linear=True) 


    def nfw_density_profile(self, z, m, params=None):
        params = merge_with_defaults(params)

        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        delta = self.halo_model.delta
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]   # Baryon fraction
        r_delta = self.halo_model.r_delta(z, m, delta, params=params)
    
        c_delta = self.halo_model.c_delta(z, m, params=params)
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
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]   # Baryon fraction
        
        # Critical density at redshift z
        rho_crit_z = self.halo_model.emulator.critical_density(z, params=params)
        
        # Battaglia+16 parameters (Table 2: AGN feedback model)
        A_rho0 = 4000.0
        A_alpha = 0.88
        A_beta = 3.83
        
        alpha_m_rho0 = 0.29
        alpha_m_alpha = -0.03
        alpha_m_beta = 0.04
        
        alpha_z_rho0 = -0.66
        alpha_z_alpha = 0.19
        alpha_z_beta = -0.025
        
        # Extended parameters
        mcut = 1e14  # M_sun
        alphap_m_rho0 = 0.29
        alphap_m_alpha = -0.03
        alphap_m_beta = 0.04
        
        alpha_c_rho0 = 0.0
        alpha_c_alpha = 0.0
        alpha_c_beta = 0.0
        
        gamma = -0.2
        xc = 0.5
        c_delta = self.halo_model.c_delta(z, m, params=params)
        
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

        # Compute final profile
        rho_gas = rho0 * rho_crit_z * f_b * p_x 
        
        return rho_gas
    
   
    def get_prefactor(self, z, m, params=None):
        """
        Compute kSZ prefactor.
        """

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params=params)

        # Get relevant quantities
        h = params['H0']/100
        d_A = self.halo_model.emulator.angular_diameter_distance(z, params=params) * h
        r_delta = self.halo_model.r_delta(z, m, self.halo_model.delta, params=params) 
        ell_delta = d_A / r_delta
        chi = self.halo_model.emulator.angular_diameter_distance(z, params=params) * h * (1 + z)

        # Compute the root mean squared velocity needed for the kSZ prefactor
        vrms = jnp.sqrt(self.halo_model.emulator.v_rms_squared(z, params=params))

        # sigmaT / m_prot in (Mpc/h)**2/(Msun/h) which is required for kSZ
        sigma_T_over_m_p = 8.305907197761162e-17 * h  

        # Get full prefactor
        a = 1.0 / (1.0 + z)
        mu_e = 1.14
        f_free = 1
        prefactor =  4 * jnp.pi * r_delta**3 * a * sigma_T_over_m_p * f_free / mu_e * (1 + z)**3 / chi**2 * vrms

        return prefactor


    def get_u_ell(self, z, m, moment=1, params=None):
        """
        Compute either the first or second moment of the tSZ power spectrum tracer u_ell.
        For tSZ:
            1st moment:  u_ell
            2nd moment:  u_ell^2
        """
        
        params = merge_with_defaults(params)
        
        # Get prefactor and perform Hankel transform from BaseTracer 
        prefactor = self.get_prefactor(z, m, params=params)
        ell, u_ell = self.u_ell_hankel(z, m, self.x, params=params)
        
        u_ell_base = prefactor[:, None] * u_ell
    
        # Select moment using JAX-safe branching
        moment_funcs = [
            lambda _: u_ell_base,          # moment = 1
            lambda _: u_ell_base**2,       # moment = 2
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell


