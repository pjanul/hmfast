import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import os
import numpy as np

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.halo_model.profiles import NFWMatterProfile
from hmfast.utils import lambertw, Const
from hmfast.download import get_default_data_path
from hmfast.defaults import merge_with_defaults


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

    def __init__(self, halo_model, profile=NFWMatterProfile(), nu=100, cib_model="shang", s_nu=None):        

        self.nu = nu
        self.profile = profile
        self.cib_model = cib_model
        
        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")
        

        if s_nu is None:
            s_nu_z_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_z_fine.txt")
            s_nu_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_nu_fine.txt")
            s_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_fine.txt")
            self.s_nu = (np.loadtxt(s_nu_z_path), np.loadtxt(s_nu_nu_path), np.loadtxt(s_nu_path))
        else:
            self.s_nu = s_nu


    @property
    def has_central_contribution(self):
        return True

    @property
    def cib_model(self):
        return self._cib_model

    @cib_model.setter
    def cib_model(self, value):
        value = str(value).lower()
        if value not in ("shang", "maniyar"):
            raise ValueError("cib_model must be either 'shang' or 'maniyar'")
        self._cib_model = value

    
    def sigma(self, m, params=None):
        params = merge_with_defaults(params)

        M_eff_cib = params['m_eff_cib']
        sigma2_LM_cib = params['sigma2_LM_cib'] 

        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM_cib)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM_cib) )
        return Sigma_M


    def phi(self, z, params=None):
        ''' 
        Implementation of Φ(z) = (1 + z)^(δ_CIB) for z < z_plateau, 1 for z >= z_plateau from the Shang model'''
        params = merge_with_defaults(params)
        delta_cib = params["delta_cib"]
        z_p = params["z_plateau_cib"]

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta_cib, 1.0)

        return Phi_z


    def theta(self, z, nu, params=None):
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


    def m_dot(self, m, z, params=None):
        ''' Mdot =  46.1(1 + 1.11z)E(z)(m /10^12Msun)^1.1 from the Maniyar model'''

        params = merge_with_defaults(params)
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        c_km_s = Const._c_ / 1e3
        E_z = jnp.atleast_1d(self.halo_model.emulator.hubble_parameter(z, params=params)) * c_km_s / params["H0"]
        
        return 46.1 * (1.0 + 1.11 * z[None, :]) * E_z[None, :] * (m[:, None] / 1e12) ** 1.1


    def sfr_maniyar(self, m, z, params=None):
        """
        Compute Maniyar et al. CIB galaxy luminosity from halo mass and redshift.
    
        Returns
        -------
        L_gal : array
            Galaxy luminosity [Lsun] per halo
        """

        # Gather all relevant parameters 
        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        M_eff, sigma2_LM, eta_max, tau, z_c, f_sub = (params[k] for k in ["m_eff_cib", "sigma2_LM_cib", "eta_max_cib", "tau_cib", "zc_cib", "fsub_cib"])
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # sigma^2 depends on whether M < M_eff or > M_eff
        sigma2_lnM = jnp.where(m[:, None] < M_eff,sigma2_LM, (jnp.sqrt(sigma2_LM) - tau * jnp.maximum(0.0, z_c - z[None, :]))**2,)

        # Get the halo accretion rate, baryon fraction, and also take log of relevant quantities
        Mdot = self.m_dot(m, z, params=params)
        logM = jnp.log(m)[:, None]
        logMeff = jnp.log(M_eff)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
    
        # Get SFR_c and then use that to get SFR
        sfr_c = eta_max * jnp.exp(- ((logM - logMeff)**2) / (2.0 * sigma2_lnM))
        sfr = 1e10 * Mdot * f_b * sfr_c

        return sfr

    def s_nu_maniyar(self, z, nu, params=None):
        ln_x_grid, ln_nu_grid, ln_s_nu_grid = jnp.log(1 + self.s_nu[0]), jnp.log(self.s_nu[1]), jnp.log(self.s_nu[2])
        _s_nu_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_nu_grid), ln_s_nu_grid)  
        s_nu = jnp.exp(_s_nu_interp((jnp.log(1 + z), jnp.log(nu))))
        return s_nu

        

    def l_gal(self, m, z, nu, params=None):
        params = merge_with_defaults(params)
        model_idx = {"shang": 0, "maniyar": 1}.get(self.cib_model, -1)
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        if model_idx == -1:
            raise ValueError(f"Unknown CIB model: {self.cib_model}. Please select either 'shang' or 'maniyar'")

        L_gal = jax.lax.switch(
            model_idx,
            [
                lambda: params["L0_cib"] *  jnp.atleast_1d(self.phi(z, params=params))[None, :] * 
                                            jnp.atleast_1d(self.sigma(m, params=params))[:, None] * 
                                            jnp.atleast_1d(self.theta(z, nu * (1 + z), params=params))[None, :],
                
                lambda: 4 * jnp.pi * self.s_nu_maniyar(z, nu, params=params)[None, :] * self.sfr_maniyar(m, z, params=params)
            ])
    
        return L_gal


    def l_sat(self, m, z, nu, params=None):
        params = merge_with_defaults(params)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
    
        def integrate_single_halo(m_single):
            """Perform the subhalo integration for a single host halo mass."""
            Ms_min = params["M_min_cib"]
            ngrid = 200 # 100,000 is likely overkill and will slow down JIT significantly
            
            # Determine upper bound based on model
            Ms_max = jax.lax.cond(self.cib_model == "maniyar", lambda x: x * (1 - params["fsub_cib"]), lambda x: x, m_single)
    
            # Create integration grid for this specific host mass
            Ms_grid = jnp.logspace(jnp.log10(Ms_min), jnp.log10(Ms_max), ngrid)
            dlnMs = jnp.log(Ms_grid[1] / Ms_grid[0])
            
            # Subhalo mass function (Shape: ngrid,)
            dN_dlnMs = self.halo_model.subhalo_mass_model.dndlnmu(m_single, Ms_grid)
            
            # Galaxy luminosity for subhalos
            if self.cib_model == "maniyar":
                SFR_I = self.l_gal(Ms_grid, z, nu, params=params)
                # SFR_II uses the host efficiency scaled by subhalo mass
                M_host_eff = Ms_max
                SFR_II = self.l_gal(M_host_eff, z, nu, params=params) * Ms_grid[:, None] / M_host_eff
                L_gal_grid = jnp.minimum(SFR_I, SFR_II)
            else:
                L_gal_grid = self.l_gal(Ms_grid, z, nu, params=params)
    
            # Integrate over Ms_grid (the 0th axis of L_gal_grid)
            # result shape: (Nz,)
            return jnp.sum(dN_dlnMs[:, None] * L_gal_grid * dlnMs, axis=0)
    
       
        L_sat_matrix = jax.vmap(integrate_single_halo, in_axes=(0,))(m)
        
        return L_sat_matrix

        

    def l_cen(self, m, z, nu, params=None):

        # Get required parameters
        params = merge_with_defaults(params)
        M_min, f_sub = params["M_min_cib"], params["fsub_cib"]
        m = jnp.atleast_1d(m)

        # For the Maniyar model, mass becomes m * (1 - f_sub); for Shang model it is unchanged
        m = jax.lax.cond(self.cib_model == "maniyar", lambda x: x * (1 - f_sub), lambda x: x, m)
        
        # Get N_cen and galaxy luminosity for each subhalo mass
        N_cen = jnp.where(m > M_min, 1.0, 0.0)
        L_gal = self.l_gal(m, z, nu, params=params)
    
        L_cen = N_cen[:, None] * L_gal
        return L_cen

        

    def kernel(self, z, params=None):
        params = merge_with_defaults(params)
        h = params["H0"]/100
        chi = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) * h
        s_nu_factor = 1/((1+z)*chi**2) if self.cib_model=='shang' else jnp.ones_like(z)# ones for maniyar
        
        return s_nu_factor

        

    def u_k(self, k, m, z, moment=1, params=None):
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
        h_factor = h**2 if self.cib_model=='shang' else 1
        

        # Compute the physical mass for Ls and Lc and then u_k_matter from BaseTracer
        m_physical = m/h
        Ls = self.l_sat(m_physical, z, self.nu , params=params)
        Lc = self.l_cen(m_physical, z, self.nu , params=params)
        # _, u_m = self.u_k_matter(k, m, z, params=params)    # Old way
        _, u_m = self.profile.u_k_matter(self.halo_model, k, m, z, params=params)

        moment_funcs = [
            lambda _: h_factor**1       / (4*jnp.pi)          * (Lc[None, :, :] + Ls[None, :, :] * u_m )                                           ,
            lambda _: h_factor**2       / (4*jnp.pi)**2       * (Ls[None, :, :]**2 * u_m**2 + 2 * Ls[None, :, :] * Lc[None, :, :] * u_m )          ,
        ]

        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k


    def sat_and_cen_contribution(self, k, m, z, params=None):

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        
        h = params["H0"]/100
       
        nu = self.nu 
        h_factor = h**2 if self.cib_model=='shang' else 1

        # Compute the physical mass for Ls and Lc and then u_k_matter from BaseTracer
        m_physical = m/h
        Ls = self.l_sat(m_physical, z, self.nu, params=params)
        Lc = self.l_cen(m_physical, z, self.nu , params=params)
        _, u_m = self.u_k_matter(k, m, z, params=params)

        # Compute central and satellite terms
        sat_term = h_factor**1    / (4*jnp.pi)        * (Ls[None, :, :] * u_m ) 
        cen_term = h_factor**1    / (4*jnp.pi)        * (Lc[None, :, :])       

        return sat_term, cen_term


    def j_bar_nu(self, m, z, nu, params=None):
        """
        Compute the mean comoving emissivity j_bar_nu(z) in [Lsun / Mpc^3].
        Integral of (L_cen + L_sat) over the halo mass function.
        """
        params = merge_with_defaults(params)
        h = params["H0"] / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(m_phys, z, nu, params=params) # Shape: (Nm, Nz)
        ls = self.l_sat(m_phys, z, nu, params=params) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = self.halo_model.halo_mass_function(m, z, params=params) # Shape: (Nm, Nz)
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        h_factor = h**2 if self.cib_model == 'shang' else 1.0
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)
        
        return j_bar * h_factor / (4 * jnp.pi)


    def monopole(self, m, z, nu, params=None):
        """
        Compute total CIB intensity I_nu [Jy/sr] using the line-of-sight integral.
        I_nu = integral [ dchi/dz * a(z) * j_bar_nu(z) ] dz
        """
        params = merge_with_defaults(params)
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(m, z, nu, params=params)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / self.halo_model.emulator.hubble_parameter(z, params=params)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity
    
