import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import mcfit
import functools

from hmfast.download import get_default_data_path
from hmfast.utils import lambertw, Const
from hmfast.halos.profiles import HaloProfile


class CIBProfile(HaloProfile):
    pass



class S12CIBProfile(CIBProfile):
    def __init__(self, nu, L0=6.4e-8, alpha=0.36, beta=1.75, gamma=1.7,
                 T0=24.4, M_eff=10**12.6, sigma2_LM=0.5, 
                 delta=3.6, z_p=1e100, M_min=10**11.5):

        self.nu = nu
        self.L0, self.alpha, self.beta, self.gamma = L0, alpha, beta, gamma
        self.T0, self.M_eff, self.sigma2_LM = T0, M_eff, sigma2_LM
        self.delta, self.z_p, self.M_min = delta, z_p, M_min

    @property
    def has_central_contribution(self):
        return True

    def _tree_flatten(self):
        leaves = (self.nu, self.L0, self.alpha, self.beta, self.gamma, self.T0, 
                  self.M_eff, self.sigma2_LM, self.delta, self.z_p, self.M_min)
        return (leaves, None)
        

    @classmethod
    def _tree_unflatten(cls, aux, leaves):
        return cls(*leaves)

    def update(self, nu=None, L0=None, alpha=None, beta=None, gamma=None,
               T0=None, M_eff=None, sigma2_LM=None, 
               delta=None, z_p=None, M_min=None):
        """
        Return a new profile instance with updated CIB parameters.

        Parameters
        ----------
        nu : float, optional
        L0 : float, optional
        alpha : float, optional
        beta : float, optional
        gamma : float, optional
        T0 : float, optional
        M_eff : float, optional
        sigma2_LM : float, optional
        delta : float, optional
        z_p : float, optional
        M_min : float, optional

        Returns
        -------
        S12CIBProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        # Explicitly map current attributes to new leaves if not provided in kwargs
        new_leaves = (
            nu if nu is not None else self.nu,
            L0 if L0 is not None else self.L0,
            alpha if alpha is not None else self.alpha,
            beta if beta is not None else self.beta,
            gamma if gamma is not None else self.gamma,
            T0 if T0 is not None else self.T0,
            M_eff if M_eff is not None else self.M_eff,
            sigma2_LM if sigma2_LM is not None else self.sigma2_LM,
            delta if delta is not None else self.delta,
            z_p if z_p is not None else self.z_p,
            M_min if M_min is not None else self.M_min,
        )
        
        return self._tree_unflatten(treedef, new_leaves)


    def sigma(self, m):
        """
        Compute the halo-mass dependence of the Shang et al. CIB luminosity.

        The mass weighting is modeled as a log-normal function,

        .. math::

            \\Sigma(M) = \\frac{M}{\\sqrt{2 \\pi \\, \\sigma_{LM}^2}}
            \\exp\\left[
            -\\frac{\\left(\\log_{10} M - \\log_{10} M_{\\mathrm{eff}}\\right)^2}
            {2 \\sigma_{LM}^2}
            \\right].

        Parameters
        ----------
        m : float or jnp.ndarray
            Halo mass or masses.

        Returns
        -------
        float or jnp.ndarray
            Log-normal mass weighting :math:`\\Sigma(M)`.
        """
        M_eff_cib, sigma2_LM = self.M_eff, self.sigma2_LM
       
        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM) )
        return Sigma_M


    def phi(self, z):
        """
        Compute the redshift evolution factor in the Shang et al. CIB model.

        The evolution is implemented as

        .. math::

            \\Phi(z) =
            \\begin{cases}
            (1 + z)^{\\delta}, & z < z_p, \\
            1, & z \\ge z_p.
            \\end{cases}

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        float or jnp.ndarray
            Redshift evolution factor :math:`\\Phi(z)`.
        """
        
        delta = self.delta
        z_p = self.z_p

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta, 1.0)

        return Phi_z


    def theta(self,  z, nu):
        """
        Compute the Shang et al. spectral energy distribution factor.

        The frequency dependence is modeled as

        .. math::

            \\Theta(\\nu, z) =
            \\begin{cases}
            \\left(\\nu / \\nu_0\\right)^{-\\gamma}, & \\nu \\ge \\nu_0, \\
            \\left(\\nu / \\nu_0\\right)^{\\beta}
            \\dfrac{B_\\nu(\\nu, T_d)}{B_\\nu(\\nu_0, T_d)}, & \\nu < \\nu_0,
            \\end{cases}

        where :math:`T_d(z) = T_0 (1 + z)^{\\alpha}` and
        :math:`B_\\nu(\\nu, T)` is the Planck blackbody function.
        The transition frequency :math:`\\nu_0(z)` is computed from the
        continuity condition used in the implementation.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Frequency or frequencies in GHz.

        Returns
        -------
        float or jnp.ndarray
            Spectral energy distribution factor :math:`\\Theta(\\nu, z)`.
        """
        
        T0, alpha, beta, gamma = self.T0, self.alpha, self.beta, self.gamma
    
        h = Const._h_P_  # Planck [J s]
        k_B = Const._k_B_ #1.380649e-23  # Boltzmann [J/K]
        c = Const._c_  #2.99792458e8    # speed of light [m/s]
    
        T_d_z = T0 * (1 + z) ** alpha
    
        x = -(3. + beta + gamma) * jnp.exp(-(3. + beta + gamma))
        # nu0 in GHz
        nu0_GHz = 1e-9 * k_B * T_d_z / h * (3. + beta + gamma + lambertw(x))
        # convert all nu, nu0 to Hz for Planck
        nu_Hz   = nu * 1e9      # If input is GHz!
        nu0_Hz  = nu0_GHz * 1e9
    
        def B_nu(nu_Hz, T):
            return (2 * h * nu_Hz ** 3 / c ** 2) / (jnp.exp(h * nu_Hz / (k_B * T)) - 1)
    
        
        Theta = jnp.where(
            nu_Hz >= nu0_Hz,
            (nu_Hz / nu0_Hz) ** (-gamma),
            (nu_Hz / nu0_Hz) ** beta * (B_nu(nu_Hz, T_d_z) / B_nu(nu0_Hz, T_d_z))
        )
        
        return Theta


    def l_gal(self, halo_model, m, z, nu):
        """
        Compute the Shang et al. galaxy luminosity assigned to a halo.

        The luminosity is modeled as

        .. math::

            \\L_\\nu^{\\mathrm{gal}}(M, z) = L_0 \\, \\Phi(z) \\, \\Sigma(M)
            \\, \\Theta\\!\\left((1+z)\\nu, z\\right),

        where :math:`\\Phi(z)`, :math:`\\Sigma(M)`, and
        :math:`\\Theta(\\nu, z)` are given by the Shang-profile helper
        functions implemented in this class.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model. Included for interface consistency.
        m : float or jnp.ndarray
            Halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Galaxy luminosity :math:`L_\\nu^{\\mathrm{gal}}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """
        # Shang model logic: L0 * Phi(z) * Sigma(m) * Theta(nu_eff)
        phi_z = jnp.atleast_1d(self.phi(z))[None, :]
        sigma_m = jnp.atleast_1d(self.sigma(m))[:, None]
        theta_val = jnp.atleast_1d(self.theta(z, nu * (1 + z)))[None, :]
        return self.L0 * phi_z * sigma_m * theta_val



    def l_sat(self, halo_model, m, z, nu):
        """
        Compute the total satellite CIB luminosity in the Shang et al. model.

        The satellite contribution is evaluated as

        .. math::

            L_\\nu^{\\mathrm{sat}}(M, z) =
            \\int d\\ln m_s \\, \\frac{dN}{d\\ln m_s}(m_s \\mid M)
            \\, L_\\nu^{\\mathrm{gal}}(m_s, z),

        with the integral approximated numerically over the subhalo mass grid
        used in the implementation.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the subhalo mass function.
        m : float or jnp.ndarray
            Host halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Satellite luminosity :math:`L_\\nu^{\\mathrm{sat}}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """
        def integrate_single_halo(m_single):
            ms_min = self.M_min
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
        """
        Compute the central-galaxy CIB luminosity in the Shang et al. model.

        The central contribution is implemented as

        .. math::

            L_\\nu^{\\mathrm{cen}}(M, z) = N_{\\mathrm{cen}}(M)
            \\, L_\\nu^{\\mathrm{gal}}(M, z),

        where :math:`N_{\\mathrm{cen}}(M) = 1` for :math:`M > M_{\\min}` and
        zero otherwise.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model. Included for interface consistency.
        m : float or jnp.ndarray
            Halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Central luminosity :math:`L_\\nu^{\\mathrm{cen}}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """
        # Shang: Central mass is the full halo mass
        n_cen = jnp.where(m > self.M_min, 1.0, 0.0)
        l_gal = self.l_gal(halo_model, m, z, nu)
        return n_cen[:, None] * l_gal


     
    def j_bar_nu(self, halo_model, m, z, nu):
        """
        Compute the mean comoving emissivity in the Shang et al. CIB model.

        The emissivity is computed as

        .. math::

            \\bar{j}_\\nu(z) = \\frac{h^3}{4 \\pi}
            \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z)
            \\left[L_{\\nu}^{\\mathrm{cen}}(M, z) + L_{\\nu}^{\\mathrm{sat}}(M, z)\\right],

        where the luminosities are evaluated at the physical halo mass.
        If halo-model consistency is enabled, the implementation adds the
        low-mass counterterm

        .. math::

            \\Delta \\bar{j}_\\nu(z) = \\frac{h^3}{4 \\pi}
            \\, n_{\\min}(z) \\, L_{\\nu}^{\\mathrm{cen}}(M_{\\min}, z).

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and halo mass function.
        m : float or jnp.ndarray
            Halo mass grid in :math:`M_\\odot / h`.
        z : float or jnp.ndarray
            Redshift grid.
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Mean comoving emissivity :math:`\\bar{j}_\\nu(z)`.
        """
        
        h = halo_model.cosmology.H0 / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        ls = self.l_sat(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = halo_model.halo_mass_function(m, z) # Shape: (Nm, Nz)

        # Correct for Maniyar if needed
        chi = halo_model.cosmology.angular_diameter_distance(z) * (1 + z) 
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)

        # Add the consistency counter-term (correction for unbound mass) if hm_consistency is True
        j_bar = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model._counter_terms(m, z)[0] * lc[0], lambda x: x, j_bar)
        
        return j_bar * h**3 / (4 * jnp.pi) 


    def monopole(self, halo_model, m, z, nu):
        """
        Compute the CIB monopole intensity in the Shang et al. CIB model.

        The specific intensity is evaluated as

        .. math::

            I_\\nu = \\int dz \\, \\frac{d\\chi}{dz} \\, a(z) \\, \\bar{j}_\\nu(z),

        where :math:`a(z) = 1/(1+z)` and
        :math:`d\\chi / dz = 1 / H(z)` in the units used internally.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology.
        m : float or jnp.ndarray
            Halo mass grid.
        z : float or jnp.ndarray
            Redshift grid.
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        float or jnp.ndarray
            Monopole intensity :math:`I_\\nu`.
        """
       
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(halo_model, m, z, nu)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / halo_model.cosmology.hubble_parameter(z)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity


    def sat_and_cen_contribution(self, halo_model, k, m, z):

        
        cparams = halo_model.cosmology._cosmo_params()
        nu = self.nu
        h = cparams["h"]
       
        #nu = self.nu 
        chi = halo_model.cosmology.angular_diameter_distance(z) * (1 + z) 

        # Compute the physical mass for ls and lc and then u_k_matter from Tracer
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
        Compute the first or second CIB profile moment in Fourier space.

        Writing :math:`u_m(k, M, z)` for the normalized analytic Fourier
        transform of the NFW matter profile returned by
        :meth:`u_k_matter`, the implemented first moment is

        .. math::

            u_\\nu(k, M, z) = \\frac{1}{4\\pi}
            \\left[L_\\nu^{\\mathrm{cen}}(M, z)
            + L_\\nu^{\\mathrm{sat}}(M, z) \\, u_m(k, M, z)\\right].

        For ``moment=2``, this method returns

        .. math::

            u_\\nu^{(2)}(k, M, z) = \\frac{1}{(4\\pi)^2}
            \\left[L_\\nu^{\\mathrm{sat}}(M, z)^2 u_m(k, M, z)^2
            + 2 L_\\nu^{\\mathrm{sat}}(M, z) L_\\nu^{\\mathrm{cen}}(M, z)
            u_m(k, M, z)\\right].

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the matter profile and CIB luminosities.
        k : float or jnp.ndarray
            Comoving wavenumber(s).
        m : float or jnp.ndarray
            Halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        moment : int, optional
            Profile moment to return. Supported values are ``1`` and ``2``.

        Returns
        -------
        tuple
            :math:`(k, u_\\nu)` where the profile array has shape
            :math:`(N_k, N_M, N_z)`.
        """
        # Get the individual components (scaled correctly by h_factors and 4pi)
        
        sat_term, cen_term = self.sat_and_cen_contribution(halo_model, k, m, z)

        moment_funcs = [
            lambda _: cen_term + sat_term,                         # prefactor * (lc[None, :, :] + ls[None, :, :] * u_m ) 
            lambda _: sat_term**2 + 2 * sat_term * cen_term,       # prefactor * (ls[None, :, :]**2 * u_m**2 + 2 * ls[None, :, :] * lc[None, :, :] * u_m ) 
        ]

        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return k, u_k

        

jax.tree_util.register_pytree_node(
    S12CIBProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: S12CIBProfile._tree_unflatten(aux_data, children)
)
        




class M21CIBProfile(CIBProfile):
    def __init__(self, nu, eta_max=0.4028, z_c=1.5, tau=1.204, f_sub=0.134, 
                 M_min=10**11.5, M_eff=10**12.6, sigma2_LM=0.5, s_nu=None):
        self.nu = nu
        self.eta_max, self.z_c, self.tau, self.f_sub = eta_max, z_c, tau, f_sub
        self.M_min, self.M_eff, self.sigma2_LM = M_min, M_eff, sigma2_LM
        self.s_nu = s_nu # Passed from Tracer


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
        
    def _tree_flatten(self):
        leaves = (self.nu, self.eta_max, self.z_c, self.tau, self.f_sub, 
                  self.M_min, self.M_eff, self.sigma2_LM)
        aux = self.s_nu
        return (leaves, aux)

    @classmethod
    def _tree_unflatten(cls, aux, leaves):
        return cls(*leaves, s_nu=aux)


    def update(self, nu=None, eta_max=None, z_c=None, tau=None, f_sub=None, 
               M_min=None, M_eff=None, sigma2_LM=None):
        """
        Return a new profile instance with updated CIB parameters.

        Parameters
        ----------
        nu : float, optional
        eta_max : float, optional
        z_c : float, optional
        tau : float, optional
        f_sub : float, optional
        M_min : float, optional
        M_eff : float, optional
        sigma2_LM : float, optional

        Returns
        -------
        M21CIBProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        new_leaves = (
            nu if nu is not None else self.nu,
            eta_max if eta_max is not None else self.eta_max,
            z_c if z_c is not None else self.z_c,
            tau if tau is not None else self.tau,
            f_sub if f_sub is not None else self.f_sub,
            M_min if M_min is not None else self.M_min,
            M_eff if M_eff is not None else self.M_eff,
            sigma2_LM if sigma2_LM is not None else self.sigma2_LM,
        )
        
        return self._tree_unflatten(treedef, new_leaves)

    
    def m_dot(self, halo_model, m, z):
        """
        Compute the halo mass accretion rate in the Maniyar et al. model.

        The accretion rate is given by

        .. math::

            \\dot{M}(M, z) = 46.1 \\, (1 + 1.11 z) \\, E(z)
            \\left(\\frac{M}{10^{12} M_{\\odot}}\\right)^{1.1},

        where :math:`E(z) = H(z) / H_0`.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology.
        m : float or jnp.ndarray
            Halo mass or masses in :math:`M_{\odot}`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Mass accretion rate :math:`\\dot{M}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """

        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        c_km_s = Const._c_ / 1e3
        
        E_z = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z)) * c_km_s / halo_model.cosmology.H0
        
        return 46.1 * (1.0 + 1.11 * z[None, :]) * E_z[None, :] * (m[:, None] / 1e12) ** 1.1


    def sfr(self, halo_model, m, z):
        """
        Compute the star-formation rate in the Maniyar et al. model.

        The star-formation rate is modeled as

        .. math::

            \\mathrm{SFR}(M, z) = 10^{10} \\, f_b \\, \\dot{M}(M, z)
            \\, \\eta(M, z),

        where :math:`f_b = \\Omega_b / \\Omega_{m,0}` and

        .. math::

            \\eta(M, z) = \\eta_{\\max}
            \\exp\\left[
            -\\frac{\\left(\\ln M - \\ln M_{\\mathrm{eff}}\\right)^2}
            {2 \\sigma_{\\ln M}^2(z)}
            \\right].

        The width is implemented as

        .. math::

            \\sigma_{\\ln M}^2(z) =
            \\begin{cases}
            \\sigma_{LM}^2, & M < M_{\\mathrm{eff}}, \\
            \\left(\\sqrt{\\sigma_{LM}^2} - \\tau \\max(0, z_c - z)\\right)^2,
            & M \\ge M_{\\mathrm{eff}}.
            \\end{cases}

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology.
        m : float or jnp.ndarray
            Halo mass or masses in :math:`M_{\odot}`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Star-formation rate with shape :math:`(N_M, N_z)`.
        """

        # Gather all relevant parameters 
        
        cparams = halo_model.cosmology._cosmo_params()
        M_eff, sigma2_LM, eta_max, tau, z_c, f_sub = self.M_eff, self.sigma2_LM, self.eta_max, self.tau, self.z_c, self.f_sub 
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

    def _s_nu_interp(self, z, nu):
        ln_x_grid, ln_nu_grid, ln_s_nu_grid = jnp.log(1 + self.s_nu[0]), jnp.log(self.s_nu[1]), jnp.log(self.s_nu[2])
        _s_nu_interp = jscipy.interpolate.RegularGridInterpolator((ln_x_grid, ln_nu_grid), ln_s_nu_grid)  
        s_nu = jnp.exp(_s_nu_interp((jnp.log(1 + z), jnp.log(nu))))
        return s_nu

        

    def l_gal(self, halo_model, m, z, nu):
        """
        Compute the Maniyar et al. galaxy luminosity assigned to a halo.

        The luminosity is modeled as

        .. math::

            L_\\nu^{\\mathrm{gal}}(M, z) = 4\\pi \\, s_\\nu(z, \\nu)
            \\, \\mathrm{SFR}(M, z),

        where :math:`s_\\nu(z, \\nu)` is obtained by interpolation from the
        tabulated spectral template used in the implementation.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model. Included for interface consistency.
        m : float or jnp.ndarray
            Halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Galaxy luminosity :math:`L_\\nu^{\\mathrm{gal}}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """
        # Maniyar model logic: 4pi * s_nu * SFR
        s_nu = self._s_nu_interp(z, nu)[None, :]
        sfr = self.sfr(halo_model, m, z)
        return 4 * jnp.pi * s_nu * sfr



    def l_sat(self, halo_model, m, z, nu):
        """
        Compute the total satellite CIB luminosity in the Maniyar et al. model.

        The satellite contribution is evaluated as

        .. math::

            L_\\nu^{\\mathrm{sat}}(M, z) =
            \\int d\\ln m_s \\, \\frac{dN}{d\\ln m_s}(m_s \\mid M)
            \\, \\min\\!\\left[
            L_\\nu^{\\mathrm{gal}}(m_s, z),
            L_\\nu^{\\mathrm{gal}}\\!\\left(M_{\\max}, z\\right)
            \\frac{m_s}{M_{\\max}}
            \\right],

        where :math:`M_{\\max} = M (1-f_{\\mathrm{sub}})`.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the subhalo mass function.
        m : float or jnp.ndarray
            Host halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Satellite luminosity :math:`L_\\nu^{\\mathrm{sat}}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """
        def integrate_single_halo(m_single):
            ms_min = self.M_min
            # Host efficiency scaling uses mass corrected by fsub
            ms_max = m_single * (1 - self.f_sub)
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
        """
        Compute the central-galaxy CIB luminosity in the Maniyar et al. model.

        The central contribution is implemented as

        .. math::

            L_\\nu^{\\mathrm{cen}}(M, z) = N_{\\mathrm{cen}}(M)
            \\, L_\\nu^{\\mathrm{gal}}\\!\\left(M(1-f_{\\mathrm{sub}}), z\\right),

        where :math:`N_{\\mathrm{cen}}(M) = 1` for
        :math:`M(1-f_{\\mathrm{sub}}) > M_{\\min}` and zero otherwise.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model. Included for interface consistency.
        m : float or jnp.ndarray
            Halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Central luminosity :math:`L_\\nu^{\\mathrm{cen}}(M, z)` with shape
            :math:`(N_M, N_z)`.
        """
        # Maniyar: Central mass is reduced by the subhalo fraction
        m_eff = m * (1 - self.f_sub)
        n_cen = jnp.where(m_eff > self.M_min, 1.0, 0.0)
        l_gal = self.l_gal(halo_model, m_eff, z, nu)
        return n_cen[:, None] * l_gal

    
    
    def j_bar_nu(self, halo_model, m, z, nu):
        """
        Compute the mean comoving emissivity in the Maniyar et al. CIB model.

        The emissivity is computed as

        .. math::

            \\bar{j}_\\nu(z) = \\frac{h^3}{4 \\pi}
            \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z)
            \\left[L_{\\nu}^{\\mathrm{cen}}(M, z) + L_{\\nu}^{\\mathrm{sat}}(M, z)\\right],

        where the luminosities are evaluated at the physical halo mass.
        If halo-model consistency is enabled, the implementation adds the
        low-mass counterterm

        .. math::

            \\Delta \\bar{j}_\\nu(z) = \\frac{h^3}{4 \\pi}
            \\, n_{\\min}(z) \\, L_{\\nu}^{\\mathrm{cen}}(M_{\\min}, z).

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and halo mass function.
        m : float or jnp.ndarray
            Halo mass grid in :math:`M_\\odot / h`.
        z : float or jnp.ndarray
            Redshift grid.
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        jnp.ndarray
            Mean comoving emissivity :math:`\\bar{j}_\\nu(z)`.
        """
       
        h = halo_model.cosmology.H0 / 100

        # Get the luminosities (ensure physical mass if needed)
        m_phys = m / h
        lc = self.l_cen(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        ls = self.l_sat(halo_model, m_phys, z, nu) # Shape: (Nm, Nz)
        
        # Get the halo mass function dn/dlnm 
        dndlnm = halo_model.halo_mass_function(m, z) # Shape: (Nm, Nz)

        # Correct for Maniyar if needed
        chi = halo_model.cosmology.angular_diameter_distance(z) * (1 + z) 
        maniyar_factor = (1+z) * chi**2 #if self.cib_model == 'maniyar' else 1
        
        # Integrate: j_bar = integral [dn/dlnm * (L_c + L_s)] dlnm
        integrand = dndlnm * (lc + ls)
        j_bar = jnp.trapezoid(integrand, x=jnp.log(m), axis=0)

        # Add the consistency counter-term (correction for unbound mass) if hm_consistency is True
        j_bar = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model._counter_terms(m, z)[0] * lc[0], lambda x: x, j_bar)
        
        return j_bar * h**3 / (4 * jnp.pi) * maniyar_factor


    def monopole(self, halo_model, m, z, nu):
        """
        Compute the CIB monopole intensity in the Maniyar et al. CIB model.

        The specific intensity is evaluated as

        .. math::

            I_\\nu = \\int dz \\, \\frac{d\\chi}{dz} \\, a(z) \\, \\bar{j}_\\nu(z),

        where :math:`a(z) = 1/(1+z)` and
        :math:`d\\chi / dz = 1 / H(z)` in the units used internally.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology.
        m : float or jnp.ndarray
            Halo mass grid.
        z : float or jnp.ndarray
            Redshift grid.
        nu : float or jnp.ndarray
            Observed frequency or frequencies in GHz.

        Returns
        -------
        float or jnp.ndarray
            Monopole intensity :math:`I_\\nu`.
        """
    
        # Get the mean comoving emissivity (Shape: Nz)
        j_bar = self.j_bar_nu(halo_model, m, z, nu)
        
        # dchi/dz = c / H(z), a(z) = 1/(1+z)
        dchi_dz = 1.0 / halo_model.cosmology.hubble_parameter(z)
        a = 1.0 / (1.0 + z)
        
        # Final Integral over redshift
        integrand = dchi_dz * a * j_bar
        intensity = jnp.trapezoid(integrand, x=z) 
        
        return intensity

    
    def sat_and_cen_contribution(self, halo_model, k, m, z):

        cparams = halo_model.cosmology._cosmo_params()
        nu = self.nu
        h = halo_model.cosmology.H0 / 100
       
        #nu = self.nu 
        chi = halo_model.cosmology.angular_diameter_distance(z) * (1 + z) 

        # Compute the physical mass for ls and lc and then u_k_matter from Tracer
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
        Compute the first or second CIB profile moment in Fourier space.

        Writing :math:`u_m(k, M, z)` for the normalized analytic Fourier
        transform of the NFW matter profile returned by
        :meth:`u_k_matter`, the implemented first moment is

        .. math::

            u_\\nu(k, M, z) = \\frac{1}{4\\pi}
            \\left[L_\\nu^{\\mathrm{cen}}(M, z)
            + L_\\nu^{\\mathrm{sat}}(M, z) \\, u_m(k, M, z)\\right].

        For ``moment=2``, this method returns

        .. math::

            u_\\nu^{(2)}(k, M, z) = \\frac{1}{(4\\pi)^2}
            \\left[L_\\nu^{\\mathrm{sat}}(M, z)^2 u_m(k, M, z)^2
            + 2 L_\\nu^{\\mathrm{sat}}(M, z) L_\\nu^{\\mathrm{cen}}(M, z)
            u_m(k, M, z)\\right].

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the matter profile and CIB luminosities.
        k : float or jnp.ndarray
            Comoving wavenumber(s).
        m : float or jnp.ndarray
            Halo mass or masses.
        z : float or jnp.ndarray
            Redshift(s).
        moment : int, optional
            Profile moment to return. Supported values are ``1`` and ``2``.

        Returns
        -------
        tuple
            :math:`(k, u_\\nu)` where the profile array has shape
            :math:`(N_k, N_M, N_z)`.
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


jax.tree_util.register_pytree_node(
    M21CIBProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: M21CIBProfile._tree_unflatten(aux_data, children)
)

