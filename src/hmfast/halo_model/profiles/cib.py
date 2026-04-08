import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import mcfit
import functools
from jax.scipy.special import sici, erf
from jax.tree_util import register_pytree_node_class

from hmfast.download import get_default_data_path
from hmfast.defaults import merge_with_defaults
from hmfast.utils import lambertw, Const
from hmfast.halo_model.mass_definition import MassDefinition
from hmfast.halo_model.profiles import HaloProfile


class CIBProfile(HaloProfile):
    pass



@register_pytree_node_class
class Shang12CIBProfile(CIBProfile):
    def __init__(self, nu, L0_cib=6.4e-8, alpha_cib=0.36, beta_cib=1.75, gamma_cib=1.7,
                 T0_cib=24.4, m_eff_cib=10**12.6, sigma2_LM_cib=0.5, 
                 delta_cib=3.6, z_plateau_cib=1e100, M_min_cib=10**11.5):

        self.nu = nu
        self.L0_cib, self.alpha_cib, self.beta_cib, self.gamma_cib = L0_cib, alpha_cib, beta_cib, gamma_cib
        self.T0_cib, self.m_eff_cib, self.sigma2_LM_cib = T0_cib, m_eff_cib, sigma2_LM_cib
        self.delta_cib, self.z_plateau_cib, self.M_min_cib = delta_cib, z_plateau_cib, M_min_cib

    @property
    def has_central_contribution(self):
        return True

    def tree_flatten(self):
        leaves = (self.nu, self.L0_cib, self.alpha_cib, self.beta_cib, self.gamma_cib, self.T0_cib, 
                  self.m_eff_cib, self.sigma2_LM_cib, self.delta_cib, self.z_plateau_cib, self.M_min_cib)
        return (leaves, None)
        

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves)

    def update_params(self, **kwargs):
        names = [
            'nu', 'L0_cib', 'alpha_cib', 'beta_cib', 'gamma_cib', 'T0_cib', 'm_eff_cib',
            'sigma2_LM_cib', 'delta_cib', 'z_plateau_cib', 'M_min_cib'
        ]
        # Check for typos/invalid names
        if not set(kwargs).issubset(names):
            raise ValueError(f"Invalid CIB parameter(s): {set(kwargs) - set(names)}")
    
        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


    def sigma(self, m):
        M_eff_cib, sigma2_LM_cib = self.m_eff_cib, self.sigma2_LM_cib
       
        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM_cib)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM_cib) )
        return Sigma_M


    def phi(self, z):
        ''' 
        Implementation of Φ(z) = (1 + z)^(δ_CIB) for z < z_plateau, 1 for z >= z_plateau from the Shang model'''
        
        delta_cib = self.delta_cib
        z_p = self.z_plateau_cib

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta_cib, 1.0)

        return Phi_z


    def theta(self,  z, nu):
        """Spectral energy distribution function Theta(nu,z) for CIB, analogous to class_sz."""
        
        T0, alpha_cib, beta_cib, gamma_cib = self.T0_cib, self.alpha_cib, self.beta_cib, self.gamma_cib
    
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


    def l_gal(self, halo_model, m, z, nu):
        # Shang model logic: L0 * Phi(z) * Sigma(m) * Theta(nu_eff)
        phi_z = jnp.atleast_1d(self.phi(z))[None, :]
        sigma_m = jnp.atleast_1d(self.sigma(m))[:, None]
        theta_val = jnp.atleast_1d(self.theta(z, nu * (1 + z)))[None, :]
        return self.L0_cib * phi_z * sigma_m * theta_val



    def l_sat(self, halo_model, m, z, nu):
        def integrate_single_halo(m_single):
            ms_min = self.M_min_cib
            ms_max = m_single
            ngrid = 200
            
            ms_grid = jnp.logspace(jnp.log10(ms_min), jnp.log10(ms_max), ngrid)
            dlnms = jnp.log(ms_grid[1] / ms_grid[0])
            
            # Subhalo mass function
            dn_dlnms = halo_model.subhalo_mass_model.dndlnmu(m_single, ms_grid)
            # Standard Shang luminosity
            l_gal_grid = self.l_gal(halo_model, ms_grid, z, nu)
            
            return jnp.sum(dn_dlnms[:, None] * l_gal_grid * dlnms, axis=0)

        return jax.vmap(integrate_single_halo)(m)


     
    def l_cen(self, halo_model, m, z, nu):
        # Shang: Central mass is the full halo mass
        n_cen = jnp.where(m > self.M_min_cib, 1.0, 0.0)
        l_gal = self.l_gal(halo_model, m, z, nu)
        return n_cen[:, None] * l_gal


     
    def j_bar_nu(self, halo_model, m, z, nu):
        """
        Compute the mean comoving emissivity j_bar_nu(z) in [Lsun / Mpc^3].
        Integral of (L_cen + L_sat) over the halo mass function.
        """
        
        h = halo_model.emulator.H0 / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        ls = self.l_sat(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = halo_model.halo_mass_function(m, z) # Shape: (Nm, Nz)

        # Correct for Maniyar if needed
        chi = halo_model.emulator.angular_diameter_distance(z) * (1 + z) 
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)

        # Add the consistency counter-term (correction for unbound mass) if hm_consistency is True
        j_bar = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z)[0] * lc[0], lambda x: x, j_bar)
        
        return j_bar * h**3 / (4 * jnp.pi) 


    def monopole(self, halo_model, m, z, nu):
        """
        Compute total CIB intensity I_nu [Jy/sr] using the line-of-sight integral.
        I_nu = integral [ dchi/dz * a(z) * j_bar_nu(z) ] dz
        """
       
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(halo_model, m, z, nu)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / halo_model.emulator.hubble_parameter(z)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity


    def sat_and_cen_contribution(self, halo_model, k, m, z):

        
        cparams = halo_model.emulator.get_all_cosmo_params()
        nu = self.nu
        h = cparams["h"]
       
        #nu = self.nu 
        chi = halo_model.emulator.angular_diameter_distance(z) * (1 + z) 

        # Compute the physical mass for ls and lc and then u_k_matter from BaseTracer
        m_physical = m/h
        ls = self.l_sat(halo_model, m_physical, z, nu)
        lc = self.l_cen(halo_model, m_physical, z, nu)

        # Apply flux cut if flux cut is not None
        #mask = ((ls + lc) / (4 * jnp.pi * (1 + z) * chi**2) * 1e3 > self.flux_cut) 
        #lc, ls = jax.lax.cond(self.flux_cut is not None, lambda _: (jnp.where(mask, 0.0, lc), jnp.where(mask, 0.0, ls)), lambda _: (lc, ls), operand=None)

        _, u_m = self.u_k_matter(halo_model, k, m, z)

        # Compute central and satellite terms
        sat_term =  1  / (4*jnp.pi)    *   (ls[None, :, :] * u_m ) 
        cen_term =  1  / (4*jnp.pi)    *   (lc[None, :, :])       

        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1):
        """ 
        Compute either the first or second moment of the CIB tracer.
        Refactored to use sat_and_cen_contribution to avoid redundant math.
        """
        # Get the individual components (scaled correctly by h_factors and 4pi)
        
        sat_term, cen_term = self.sat_and_cen_contribution(halo_model, k, m, z)

        moment_funcs = [
            lambda _: cen_term + sat_term,                         # prefactor * (lc[None, :, :] + ls[None, :, :] * u_m ) 
            lambda _: sat_term**2 + 2 * sat_term * cen_term,       # prefactor * (ls[None, :, :]**2 * u_m**2 + 2 * ls[None, :, :] * lc[None, :, :] * u_m ) 
        ]

        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k


        



@register_pytree_node_class
class Maniyar21CIBProfile(CIBProfile):
    def __init__(self, nu, eta_max_cib=0.4028, zc_cib=1.5, tau_cib=1.204, fsub_cib=0.134, 
                 M_min_cib=10**11.5, m_eff_cib=10**12.6, sigma2_LM_cib=0.5, s_nu_data=None):
        self.nu = nu
        self.eta_max_cib, self.zc_cib, self.tau_cib, self.fsub_cib = eta_max_cib, zc_cib, tau_cib, fsub_cib
        self.M_min_cib, self.m_eff_cib, self.sigma2_LM_cib = M_min_cib, m_eff_cib, sigma2_LM_cib
        self.s_nu_data = s_nu_data # Passed from Tracer


        if s_nu_data is None:
            s_nu_z_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_z_fine.txt")
            s_nu_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_nu_fine.txt")
            s_nu_path = os.path.join(get_default_data_path(), "auxiliary_files", "filtered_snu_planck_fine.txt")
            self.s_nu_data = (np.loadtxt(s_nu_z_path), np.loadtxt(s_nu_nu_path), np.loadtxt(s_nu_path))
        else:
            self.s_nu_data = s_nu_data

    @property
    def has_central_contribution(self):
        return True
        
    def tree_flatten(self):
        leaves = (self.nu, self.eta_max_cib, self.zc_cib, self.tau_cib, self.fsub_cib, 
                  self.M_min_cib, self.m_eff_cib, self.sigma2_LM_cib)
        aux = self.s_nu_data
        return (leaves, aux)

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        return cls(*leaves, s_nu_data=aux)


    def update_params(self, **kwargs):
        names = ['nu', 'eta_max_cib', 'zc_cib', 'tau_cib', 'fsub_cib', 'M_min_cib', 'm_eff_cib', 'sigma2_LM_cib']
        # Check for typos/invalid names
        if not set(kwargs).issubset(names):
            raise ValueError(f"Invalid CIB parameter(s): {set(kwargs) - set(names)}")
    
        leaves, treedef = jax.tree_util.tree_flatten(self)
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    
    def m_dot(self, halo_model, m, z):
        ''' Mdot =  46.1(1 + 1.11z)E(z)(m /10^12Msun)^1.1 from the Maniyar model'''

        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        c_km_s = Const._c_ / 1e3
        
        E_z = jnp.atleast_1d(halo_model.emulator.hubble_parameter(z)) * c_km_s / halo_model.emulator.H0
        
        return 46.1 * (1.0 + 1.11 * z[None, :]) * E_z[None, :] * (m[:, None] / 1e12) ** 1.1


    def sfr_maniyar(self, halo_model, m, z):
        """
        Compute Maniyar et al. CIB galaxy luminosity from halo mass and redshift.
    
        Returns
        -------
        L_gal : array
            Galaxy luminosity [lsun] per halo
        """

        # Gather all relevant parameters 
        
        cparams = halo_model.emulator.get_all_cosmo_params()
        M_eff, sigma2_LM, eta_max, tau, z_c, f_sub = self.m_eff_cib, self.sigma2_LM_cib, self.eta_max_cib, self.tau_cib, self.zc_cib, self.fsub_cib 
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # sigma^2 depends on whether M < M_eff or > M_eff
        sigma2_lnM = jnp.where(m[:, None] < M_eff,sigma2_LM, (jnp.sqrt(sigma2_LM) - tau * jnp.maximum(0.0, z_c - z[None, :]))**2,)

        # Get the halo accretion rate, baryon fraction, and also take log of relevant quantities
        Mdot = self.m_dot(halo_model, m, z)
        logM = jnp.log(m)[:, None]
        logMeff = jnp.log(M_eff)
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
    
        # Get SFR_c and then use that to get SFR
        sfr_c = eta_max * jnp.exp(- ((logM - logMeff)**2) / (2.0 * sigma2_lnM))
        sfr = 1e10 * Mdot * f_b * sfr_c

        return sfr

    def s_nu_maniyar(self, z, nu):
        ln_x_grid, ln_nu_grid, ln_s_nu_grid = jnp.log(1 + self.s_nu_data[0]), jnp.log(self.s_nu_data[1]), jnp.log(self.s_nu_data[2])
        _s_nu_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_nu_grid), ln_s_nu_grid)  
        s_nu = jnp.exp(_s_nu_interp((jnp.log(1 + z), jnp.log(nu))))
        return s_nu

        

    def l_gal(self, halo_model, m, z, nu):
        # Maniyar model logic: 4pi * s_nu * SFR
        s_nu = self.s_nu_maniyar(z, nu)[None, :]
        sfr = self.sfr_maniyar(halo_model, m, z)
        return 4 * jnp.pi * s_nu * sfr



    def l_sat(self, halo_model, m, z, nu):
        def integrate_single_halo(m_single):
            ms_min = self.M_min_cib
            # Host efficiency scaling uses mass corrected by fsub
            ms_max = m_single * (1 - self.fsub_cib)
            ngrid = 200
            
            ms_grid = jnp.logspace(jnp.log10(ms_min), jnp.log10(ms_max), ngrid)
            dlnms = jnp.log(ms_grid[1] / ms_grid[0])
            
            dn_dlnms = halo_model.subhalo_mass_model.dndlnmu(m_single, ms_grid)
            
            # Maniyar Clamping Logic
            sfr_i = self.l_gal(halo_model, ms_grid, z, nu)
            sfr_ii = self.l_gal(halo_model, ms_max, z, nu) * ms_grid[:, None] / ms_max
            l_gal_grid = jnp.minimum(sfr_i, sfr_ii)
            
            return jnp.sum(dn_dlnms[:, None] * l_gal_grid * dlnms, axis=0)

        return jax.vmap(integrate_single_halo)(m)


    def l_cen(self, halo_model, m, z, nu):
        # Maniyar: Central mass is reduced by the subhalo fraction
        m_eff = m * (1 - self.fsub_cib)
        n_cen = jnp.where(m_eff > self.M_min_cib, 1.0, 0.0)
        l_gal = self.l_gal(halo_model, m_eff, z, nu)
        return n_cen[:, None] * l_gal

    
    
    def j_bar_nu(self, halo_model, m, z, nu):
        """
        Compute the mean comoving emissivity j_bar_nu(z) in [Lsun / Mpc^3].
        Integral of (L_cen + L_sat) over the halo mass function.
        """
       
        h = halo_model.emulator.H0 / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        ls = self.l_sat(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = halo_model.halo_mass_function(m, z) # Shape: (Nm, Nz)

        # Correct for Maniyar if needed
        chi = halo_model.emulator.angular_diameter_distance(z) * (1 + z) 
        maniyar_factor = (1+z) * chi**2 #if self.cib_model == 'maniyar' else 1
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)

        # Add the consistency counter-term (correction for unbound mass) if hm_consistency is True
        j_bar = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model.counter_terms(m, z)[0] * lc[0], lambda x: x, j_bar)
        
        return j_bar * h**3 / (4 * jnp.pi) * maniyar_factor


    def monopole(self, halo_model, m, z, nu):
        """
        Compute total CIB intensity I_nu [Jy/sr] using the line-of-sight integral.
        I_nu = integral [ dchi/dz * a(z) * j_bar_nu(z) ] dz
        """
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(halo_model, m, z, nu)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / halo_model.emulator.hubble_parameter(z)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity

    
    def sat_and_cen_contribution(self, halo_model, k, m, z):

        cparams = halo_model.emulator.get_all_cosmo_params()
        nu = self.nu
        h = halo_model.emulator.H0 / 100
       
        #nu = self.nu 
        chi = halo_model.emulator.angular_diameter_distance(z) * (1 + z) 

        # Compute the physical mass for ls and lc and then u_k_matter from BaseTracer
        m_physical = m/h
        ls = self.l_sat(halo_model, m_physical, z, nu)
        lc = self.l_cen(halo_model, m_physical, z, nu)

        # Apply flux cut if flux cut is not None
        #mask = ((ls + lc) / (4 * jnp.pi * (1 + z) * chi**2) * 1e3 > self.flux_cut) 
        #lc, ls = jax.lax.cond(self.flux_cut is not None, lambda _: (jnp.where(mask, 0.0, lc), jnp.where(mask, 0.0, ls)), lambda _: (lc, ls), operand=None)

        _, u_m = self.u_k_matter(halo_model, k, m, z)

        # Compute central and satellite terms
        sat_term =  1  / (4*jnp.pi)    *   (ls[None, :, :] * u_m ) 
        cen_term =  1  / (4*jnp.pi)    *   (lc[None, :, :])       

        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1):
        """ 
        Compute either the first or second moment of the CIB tracer.
        Refactored to use sat_and_cen_contribution to avoid redundant math.
        """
        # Get the individual components (scaled correctly by h_factors and 4pi)
        

        nu = self.nu
        sat_term, cen_term = self.sat_and_cen_contribution(halo_model, k, m, z)

        moment_funcs = [
            lambda _: cen_term + sat_term,                         # prefactor * (lc[None, :, :] + ls[None, :, :] * u_m ) 
            lambda _: sat_term**2 + 2 * sat_term * cen_term,       # prefactor * (ls[None, :, :]**2 * u_m**2 + 2 * ls[None, :, :] * lc[None, :, :] * u_m ) 
        ]

        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k

     
