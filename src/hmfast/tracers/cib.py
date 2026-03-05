import jax
import jax.numpy as jnp

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.defaults import merge_with_defaults
from hmfast.halo_model.mass_function import TW10SubHaloMass
from hmfast.utils import lambertw, Const
from hmfast.download import get_default_data_path


jax.config.update("jax_enable_x64", True)


class CIBTracer(BaseTracer):
    """
    CIB lensing tracer. 

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
     x : array
        The x array used to define the radial profile over which the tracer will be evaluated
    """

    def __init__(self, halo_model, nu=100, subhalo_mass_function=TW10SubHaloMass(), cib_model="shang", s_nu=None):        

        self.nu = nu
        self.subhalo_mass_function = subhalo_mass_function # Might eventually want to move this to halo_model

        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        self.cib_model = cib_model

        if s_nu is None:
            s_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_fine.txt")
            self.s_nu = self.np.loadtxt(s_nu_path)
        else:
            self.s_nu = s_nu
        


    
    def sigma(self, z, m, nu, params=None):
        params = merge_with_defaults(params)

        M_eff_cib = params['m_eff_cib']
        sigma2_LM_cib = params['sigma2_LM_cib'] 

        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM_cib)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM_cib) )
        return Sigma_M


    def phi(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
        delta_cib = params["delta_cib"]
        z_p = params["z_plateau_cib"]

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta_cib, 1.0)

        return Phi_z


    def theta(self, z, m, nu, params=None):

       
        """Spectral energy distribution function Theta(nu,z) for CIB, analogous to class_sz."""
        params = merge_with_defaults(params)
        T0 = params["T0_cib"]
        alpha_cib = params["alpha_cib"]
        beta_cib = params["beta_cib"]
        gamma_cib = params["gamma_cib"]
    
        h = Const._h_P_  # Planck [J s]
        k_B = Const._k_B_ #1.380649e-23  # Boltzmann [J/K]
        c = Const._c_  #2.99792458e8    # speed of light [m/s]
    
        T_d_z = T0 * (1 + z) ** alpha_cib
    
        x = -(3. + beta_cib + gamma_cib) * jnp.exp(-(3. + beta_cib + gamma_cib))
        # nu0 in GHz
        nu0_GHz = 1e-9 * k_B * T_d_z / h * (3. + beta_cib + gamma_cib + lambertw(x))
        # convert all nu, nu0 to Hz for Planck
        nu_Hz   = nu * 1e9      # If input is GHz!
        nu0_Hz  = nu0_GHz * 1e9
    
        def B_nu(nu_Hz, T):
            return (2 * h * nu_Hz ** 3 / c ** 2) / (jnp.exp(h * nu_Hz / (k_B * T)) - 1)
    
        
        Theta = jnp.where(
            nu_Hz >= nu0_Hz,
            (nu_Hz / nu0_Hz) ** (-gamma_cib),
            (nu_Hz / nu0_Hz) ** beta_cib * (B_nu(nu_Hz, T_d_z) / B_nu(nu0_Hz, T_d_z))
        )
        
        return Theta

    def s_nu(self, z, m, nu, params=None):

        params = merge_with_defaults(params)
        
        chi = self.halo_model.emulator.angular_diameter_distance(z, params = params)

        L_sat_nu = self.l_sat(z, m, nu, params=params) 
        L_cen_nu = self.l_cen(z, m, nu, params=params) 
        L_nu = L_sat_nu + L_cen_nu

        S_nu = L_nu / (4 * jnp.pi * (1 + z) * chi**2)

        return S_nu

    def m_dot(self, z, m, params=None):

        params = merge_with_defaults(params)
        c_km_s = Const._c_ / 1e3
        
        E_z = self.halo_model.emulator.hubble_parameter(z, params=params) * c_km_s / params["H0"]
        
        return 46.1 * (1.0 + 1.11 * z) * E_z * (m / 1e12) ** 1.1


    def sfr_maniyar(self, z, m, params=None):
        """
        Compute Maniyar et al. CIB galaxy luminosity from halo mass and redshift.
    
        Returns
        -------
        L_gal : array
            Galaxy luminosity [Lsun] per halo
        """
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        
        # General CIB parameters
        M_eff = params["m_eff_cib"]
        sigma2_LM = params["sigma2_LM_cib"]

        # Maniyar-specific parameters
        etamax = params["maniyar_cib_etamax"]
        tau = params["maniyar_cib_tau"]
        z_c = params["maniyar_cib_zc"]
        f_sub = params["maniyar_cib_fsub"]

    
        # Halo accretion rate and baryon fraction
        Mdot = self.m_dot(z, m, params=params)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
    
        # Log-normal width with redshift evolution
        logM = jnp.log(m)
        logMeff = jnp.log(M_eff)
    
        # sigma^2 depends on whether M < M_eff or > M_eff
        sigma2_lnM = jnp.where(
            m < M_eff,
            sigma2_LM,
            (jnp.sqrt(sigma2_LM) - tau * jnp.maximum(0.0, z_c - z)),
        )
    
        # Log-normal efficiency
        sfr_c = etamax * jnp.exp(- ((logM - logMeff)**2) / (2.0 * sigma2_lnM))
    
        # Galaxy luminosity in L_sun (includes Kennicutt constant 1e10)
        sfr = 1e10 * Mdot * f_b * sfr_c

        #print("z, m, m, Mdot, f_b, logM, logMeff, sigma_lnM, sfr_c, sfr =", z, m, m, Mdot, f_b, logM, logMeff, sigma2_lnM, sfr_c, sfr)
    
        return sfr



    def l_gal(self, z, m, nu, params=None):
        params = merge_with_defaults(params)

        if self.cib_model == "shang":
      
            # Note that Theta takes nu*(1+z) for SED instead of nu
            L0 = params["L0_cib"]
            Phi = self.phi(z, m, nu, params)
            Theta = self.theta(z, m, nu, params)  
            Sigma = self.sigma(z, m, nu, params)
    
            return L0 * Phi * Sigma * Theta
            

        elif self.cib_model == "maniyar":
            #S_nu = self.s_nu(z, m, nu, params=params)  # Need a way to compute this value
            sfr = self.sfr_maniyar(z, m, params=params)
            raise NotImplementedError("Maniyar is still in progress")
            
        

        else:
            raise ValueError(f"Unknown CIB model: {self.cib_model}. Please select either 'shang' or 'maniyar'")
        


    def l_sat(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
        
        # Use a small fraction of host mass or a fixed value for min subhalo mass
        Ms_min = 1e5  # or Ms_min = 1e-4 * M_host - should eventually be a free parameter
        ngrid = 100   # Reasonable default for integration grid
    
        Ms_grid = jnp.logspace(jnp.log10(Ms_min), jnp.log10(m), ngrid)
        dlnMs = jnp.log(Ms_grid[1]/Ms_grid[0])
    
        # Subhalo mass function per dlnMs
        dN_dlnMs = self.subhalo_mass_function.dndlnmu(m, Ms_grid)
    
        # Galaxy luminosity for each subhalo mass
        L_gal = self.l_gal(z, Ms_grid, nu, params=params)
    
        # Integrate over ln(Ms)
        integrand = dN_dlnMs * L_gal
        L_sat = jnp.sum(integrand * dlnMs)
        return L_sat
        

    def l_cen(self, z, m, nu, params=None):
         params = merge_with_defaults(params)

         M_min = params["M_min_cib"]

         if self.cib_model == "maniyar":
             f_sub = params["maniyar_cib_fsub"]
             m_central = m * (1 - f_sub)
         else:
             m_central = m
             
         N_cen = jnp.where(m > M_min, 1.0, 0.0)

         # Galaxy luminosity for each subhalo mass
         L_gal = self.l_gal(z, m_central, nu, params=params)
         L_cen = N_cen * L_gal
         return L_cen

   

    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CIB tracer u_ell.
        For CIB:, 
            First moment:     W_I_nu / jnu_bar * Lc + Ls * u_ell_m
            Second moment:     W_I_nu^2 / jnu_nu^2 * [Ls^2 * u_ell_m^2 + 2 * Ls * Lc * u_ell_m]
        You cannot simply take u_ell_g**2.

        Note that  W_I_nu = a(z) * jnu_bar, so  W_I_nu / jnu_bar = a(z)
        """

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        
        h = params["H0"]/100
        chi = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) * h 
        nu_rest = self.nu * (1 + z)

        s_nu_factor = jnp.sqrt(1 + z) / ((1 + z) * chi**2) 
        
        Ls = self.l_sat(z, m, nu_rest, params=params) 
        Lc = self.l_cen(z, m, nu_rest, params=params) 
        

        # Compute u_m_ell from BaseTracer
        ell, u_m = self.u_ell_analytic(z, m, params=params)

        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"] 
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        Hz = self.halo_model.emulator.hubble_parameter(z, params=params)

        #u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _: 1 / h**2      / (4*jnp.pi)          * (Lc + Ls * u_m)                           * s_nu_factor        ,
            lambda _: 1 / h**4      / (4*jnp.pi)**2       * (Ls**2 * u_m**2 + 2 * Ls * Lc * u_m)      * s_nu_factor**2     ,
        ]


        #moment_funcs = [
        #    lambda _: 1 / h**4      / (4*jnp.pi)          * (Lc + Ls * u_m)                                                ,
        #    lambda _: 1 / h**8      / (4*jnp.pi)**2       * (Ls**2 * u_m**2 + 2 * Ls * Lc * u_m)                           ,
        #]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell

    
