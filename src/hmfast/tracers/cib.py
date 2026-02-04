import jax
import jax.numpy as jnp
from hmfast.emulator_eval import Emulator
from hmfast.base_tracer import BaseTracer, HankelTransform
from hmfast.defaults import merge_with_defaults
from hmfast.literature import c_D08, shmf_TW10
from jax.scipy.special import sici, erf 
from hmfast.tools.lambertw import lambertw



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

    def __init__(self, cosmo_model=0, nu=100e9, x=None, concentration_relation=c_D08, subhalo_mass_function=shmf_TW10):        

        self.nu = nu
        self.x = x if x is not None else jnp.logspace(jnp.log10(1e-4), jnp.log10(20.0), 512)
        self.hankel = HankelTransform(self.x, nu=0.5)
        self.concentration_relation = concentration_relation
        self.subhalo_mass_function = subhalo_mass_function

        # Load emulator and make sure the required files are loaded outside of jitted functions
        self.emulator = Emulator(cosmo_model=cosmo_model)
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")
        self.emulator._load_emulator("DER")


    
    def get_Sigma(self, z, m, nu, params=None):
        params = merge_with_defaults(params)

        M_eff_cib = params['m_eff_cib']
        sigma2_LM_cib = params['sigma2_LM_cib'] 

        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM_cib)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM_cib) )
        return Sigma_M


    def get_Phi(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
        delta_cib = params["delta_cib"]
        z_p = params["z_plateau_cib"]

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta_cib, 1.0)

        return Phi_z


    def get_Theta(self, z, m, nu, params=None):

        nu = nu * (1 + z)
        """Spectral energy distribution function Theta(nu,z) for CIB, analogous to class_sz."""
        params = merge_with_defaults(params)
        T0 = params["T0_cib"]
        alpha_cib = params["alpha_cib"]
        beta_cib = params["beta_cib"]
        gamma_cib = params["gamma_cib"]
    
        h = 6.62607015e-34  # Planck [J s]
        k_B = 1.380649e-23  # Boltzmann [J/K]
        c = 2.99792458e8    # speed of light [m/s]
    
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

        

    def get_L_gal(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
    
        L0 = params["L0_cib"]
        
        
        # Note: class_sz uses nu*(1+z) for SED
        Phi = self.get_Phi(z, m, nu, params)
        Theta = self.get_Theta(z, m, nu, params)
        Sigma = self.get_Sigma(z, m, nu, params)

        
        return L0 * Phi * Sigma * Theta
        


    def get_L_sat(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
        
        # Use a small fraction of host mass or a fixed value for min subhalo mass
        Ms_min = 1e9  # or Ms_min = 1e-4 * M_host
        ngrid = 100   # Reasonable default for integration grid
    
        Ms_grid = jnp.logspace(jnp.log10(Ms_min), jnp.log10(m), ngrid)
        dlnMs = jnp.log(Ms_grid[1]/Ms_grid[0])
    
        # Subhalo mass function per dlnMs
        dN_dlnMs = self.subhalo_mass_function(m, Ms_grid)
    
        # Galaxy luminosity for each subhalo mass
        L_gal = self.get_L_gal(z, Ms_grid, nu, params=params)
    
        # Integrate over ln(Ms)
        integrand = dN_dlnMs * L_gal
        L_sat = jnp.sum(integrand * dlnMs)
        return L_sat
        

    def get_L_cen(self, z, m, nu, params=None):
         params = merge_with_defaults(params)

         M_min = params["M_min_cib"]
         N_cen = jnp.where(m > M_min, 1.0, 0.0)

         # Galaxy luminosity for each subhalo mass
         L_gal = self.get_L_gal(z, m, nu, params=params)
         L_cen = N_cen * L_gal
         return L_cen


    def get_u_m_ell(self, z, m, params = None):
        """
        This function calculates u_ell^m(z, M) via the analytic method described in Kusiak et al (2023).
        """
        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params)

        m = jnp.atleast_1d(m) 
        h = cparams["h"]
        x = self.x

        # Concentration parameters
        delta = params["delta"]
        c_delta = self.concentration_relation(z, m)
        r_delta = self.emulator.get_r_delta_of_m_delta_at_z(delta, m, z, params=params) 
        lambda_val = params.get("lambda_HOD", 1.0) 

        # Use x grid to get l values. It may eventually make sense to not do the Hankel
        dummy_profile = jnp.ones_like(x)
        k, _ = self.hankel.transform(dummy_profile)
        chi = self.emulator.get_angular_distance_at_z(z, params=params) * (1.0 + z) * h
        ell = k * chi - 0.5
        ell = jnp.broadcast_to(ell[None, :], (m.shape[0], k.shape[0]))    # (N_m, N_k)

        # Ensure proper dimensionality of k, r_delta, c_delta
        k_mat = k[None, :]                            # (1, N_k)
        r_mat = r_delta[:, None]                       # (N_m, 1)
        c_mat = jnp.atleast_1d(c_delta)[:, None]       # (N_m, 1)

        # Convert rho_crit in  M_sun/Mpc^3 to rho_mean in (M_sun/h)/(Mpc/h^3) 
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"] / h**2   
        m_over_rho_mean = jnp.broadcast_to((m / rho_mean_0)[:, None], (m.shape[0], k.shape[0])) 

        # Get q values for the SiCi functions
        q = k_mat * r_mat / c_mat * (1+z)            # (N_m, N_k)
        q_scaled = (1 + lambda_val * c_mat) * q
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)

        # Get NFW function f_NFW(x) 
        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x/(1 + x))
        f_nfw_val = f_nfw(lambda_val * c_mat)
        
        # Compute Fourier transform via analytic formula
        u_ell_m =  (   jnp.cos(q) * (Ci_q_scaled - Ci_q) 
                    +  jnp.sin(q) * (Si_q_scaled - Si_q) 
                    -  jnp.sin(lambda_val * c_mat * q) / q_scaled ) * f_nfw_val * m_over_rho_mean
        
     
        return ell, u_ell_m



    

    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CIB tracer u_ell.
        For CIB:, 
            First moment:     W_I_nu / jnu_bar * Lc + Ls * u_ell_m
            Second moment:     W_I_nu^2 / jnu_nu^2 * [Ls^2 * u_ell_m^2 + 2 * Ls * u_ell_m]
        You cannot simply take u_ell_g**2.

        Note that  W_I_nu = a(z) * jnu_bar, so  W_I_nu / jnu_bar = a(z)
        """

        params = merge_with_defaults(params)
        Ls = self.get_L_cen(z, m, self.nu, params=params)
        Lc = self.get_L_sat(z, m, self.nu, params=params)

        Ls_nu_prime = self.get_L_cen(z, m, self.nu, params=params)
        Lc_nu_prime = self.get_L_sat(z, m, self.nu, params=params)

        
        a = 1. / (1. + z)
        
        ell, u_m = self.get_u_m_ell(z, m, params=params)

        
        moment_funcs = [
            lambda _: 1 / (4 * jnp.pi) * (Lc + Ls * u_m),
            lambda _: 1 / (4 * jnp.pi) * (Ls**2 * u_m**2 + 2 * Ls * Lc * u_m),
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell

    
