import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from hmfast.base_tracer import BaseTracer, HankelTransform
#from hmfast.utils import sici # for now, we will implement our own sici until JAX fixes the bug
from jax.scipy.special import sici # eventually we will want to import this, but there is a bug in the JAX version
from jax.scipy.special import erf  # Use jax-enabled math for special functions
import numpy as np
_eps = 1e-30

jax.config.update("jax_enable_x64", True)
dndz_data = jnp.array(np.loadtxt("../../data/hmfast_data/normalised_dndz_cosmos_0.txt"))




class GalaxyHODTracer(BaseTracer):
    """
    Galaxy HOD tracer implementing central + satellite occupation and
    NFW satellites. Implements the formalism described in Kusiak et al (2003)
    Link to paper: https://arxiv.org/pdf/2203.12583

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
    params : dict
        Dictionary of parameters. 
        Relevant keys:
          - M_min_HOD, sigma_log10M_HOD, M0_HOD, M1_prime_HOD, alpha_s_HOD
    """

    def __init__(self, emulator, halo_model, params):        
        self.params = params
        x_min, x_max, x_npoints = params['x_min'], params['x_max'], params['x_npoints']
        self.x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), x_npoints)
        self.hankel = HankelTransform(x_min=x_min, x_max=x_max, x_npoints=x_npoints, nu=0.5)
        self.r_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), x_npoints)
        self.emulator = emulator # cosmology emulator
        self.halo_model = halo_model


    def get_N_centrals(self, m, params = None):
        """Mean central occupation: shape = M.shape"""
        M_min = params["M_min_HOD"]
        sigma = params["sigma_log10M_HOD"]
        x = (jnp.log10(m) - jnp.log10(M_min)) / sigma
        return 0.5 * (1.0 + erf(x))

    
    def get_N_satellites(self, m, params = None):
        """Mean satellite occupation: shape = M.shape"""
        M0 = params["M0_HOD"]
        M1p = params["M1_prime_HOD"]
        alpha = params["alpha_s_HOD"]

        # power law only above M0 and use jnp.where to keep differentiability
        pow_term = jnp.maximum((m - M0) / M1p, 0.0)**alpha
        N_c = self.get_N_centrals(m, params = params)
        return  N_c * pow_term
        

    def get_ng_bar_at_z(self, z, params = None):
        """
        Compute comoving galaxy number density ng(z) = ∫ dlnM [dn/dlnM] [Nc+Ns].
        halo_model: HaloModel instance
        tracer: GalaxyHODTracer instance (provides HOD via helper funcs)
        z: scalar redshift
        params: parameter dict (used to construct the m_grid etc.)
        """
        # mass grid (same convention used in HaloModel.get_C_ell_*)
        m_grid = jnp.geomspace(params['M_min'], params['M_max'], params['M_npoints'])
        logm = jnp.log(m_grid)
        z = jnp.atleast_1d(z)

        Nc = self.get_N_centrals(m_grid, params=params)
        Ns = self.get_N_satellites(m_grid, params=params)
        Ntot = Nc + Ns
    
        def ng_bar_single(z_single):
            dndlnm = self.halo_model.mass_function(z_single, m_grid, params=params)  # shape (n_m,)
            integrand = dndlnm * Ntot
            return jnp.trapezoid(integrand, x=logm)
    
        # vectorize over z
        return jax.vmap(ng_bar_single)(z)


    def get_wg_at_z_alt(self, z, params=None):
        """
        Compute Wg(z) = [H(z) / c] * [phi'_g(z) / chi(z)^2] following Eq. (13)
        of the paper https://arxiv.org/pdf/2203.12583, 
        where phi'_g(z) = (1 / N_tot) * dN_g/dz and
        dN_g/dz = ng(z) * dV/dz/dOmega.

        """

        z_grid = jnp.geomspace(params['z_min'], params['z_max'], params['z_npoints'])

        # Compute ng(z) on the grid using the existing get_ng_bar_at_z
        ng_grid = self.get_ng_bar_at_z(z_grid, params=params)  # shape (n_z,)

        # Compute dNdz given that dN/dz/dOmega = ng(z) * dV/dz/dOmega
        dVdz_grid = self.emulator.get_dVdzdOmega_at_z(z_grid, params=params)  # shape (n_z,)
        dNdz = ng_grid * dVdz_grid

        # Total number N_tot = ∫ dN/dz dz (per steradian)
        N_tot = jnp.trapezoid(dNdz, x=z_grid)
        N_tot_safe = jnp.maximum(N_tot, _eps)

        # Normalized galaxy distribution phi'_g(z) = dN/dz / N_tot
        phi_prime = dNdz / N_tot_safe

        # Get H(z) and chi(z). No need to divide H by c since the emulator outputs 1/Mpc
        H_grid = self.emulator.get_hubble_at_z(z_grid, params=params)  # 1/Mpc
        chi_grid = self.emulator.get_angular_distance_at_z(z_grid, params=params) * (1.0 + z_grid)  # Mpc comov

        # Assemble Wg on the grid
        chi2_safe = jnp.maximum(chi_grid ** 2, _eps)
        Wg_grid = H_grid * (phi_prime / chi2_safe)

        # Interpolate Wg_grid to requested z
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
        Wg_q = jnp.interp(zq, z_grid, Wg_grid, left=0.0, right=0.0)

        z = dndz_data[:, 0]          # first column: redshifts
        phi_prime = dndz_data[:, 1]  # second column: phi_prime

        return Wg_q
        

    def get_wg_at_z(self, z, params=None):
        """
        Return Wg_grid at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        """
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
    
        # Extract precomputed phi_prime
        z_data = dndz_data[:, 0]
        phi_prime_data = dndz_data[:, 1]
    
        # Interpolate phi_prime to requested z
        phi_prime_at_z = jnp.interp(zq, z_data, phi_prime_data, left=0.0, right=0.0)
    
        H_grid = self.emulator.get_hubble_at_z(zq, params=params)  # 1/Mpc
        chi_grid = self.emulator.get_angular_distance_at_z(zq, params=params) * (1.0 + zq)  # Mpc comov

        # Assemble Wg on the grid
        Wg_grid = H_grid * (phi_prime_at_z / chi_grid**2)
        return Wg_grid


    def c_Duffy2008(self, z, m, A=5.71, B=-0.084, C=-0.47, M_pivot=2e12):
        """
        Duffy et al. 2008 mass-concentration relation.
        A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
        """
        return A * (m / M_pivot)**B * (1 + z)**C

     
    def compute_u_m_ell_alt(self, z, m, params = None):
        """
        This function calculates u_ell^m(z, M) via the analytic method described in Kusiak et al (2023).
        As of November 2025, the jax.scipy.special.sici functions are not well behaved for large inputs.
        As a result, we will shelve this method for now until the next stable JAX release.
        """

        m = jnp.atleast_1d(m) 

        ell_min = params.get("ell_min", 1e2)
        ell_max = params.get("ell_max", 3.5e3) 

        c_200c = self.c_Duffy2008(z, m)
        r_200c = self.emulator.get_r_delta_of_m_delta_at_z(200, m, z, params=params) 
        lambda_val = params.get("lambda_HOD", 1.0) 

        
        chi = self.emulator.get_angular_distance_at_z(z, params=params) * (1.0 + z) 
        ell_row = jnp.logspace(jnp.log10(ell_min), jnp.log10(ell_max), 100)

        ell = jnp.broadcast_to(ell_row[None, :], (m.shape[0], 100))           # (N_m, N_k)
        k = (ell_row + 0.5) / chi   # physical k
  
        k_mat = k[None, :]                            # (1, N_k)
        r_mat = r_200c[:, None]                       # (N_m, 1)
        c_mat = jnp.atleast_1d(c_200c)[:, None]       # (N_m, 1)
                
        q = k_mat * r_mat / c_mat            # (N_m, N_k)
        q_scaled = (1 + lambda_val * c_mat) * q

        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x/(1 + x))
        f_nfw_val = f_nfw(lambda_val * c_mat)

        
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        
        u_ell_m = (    jnp.cos(q) * (Ci_q_scaled - Ci_q) 
                    +  jnp.sin(q) * (Si_q_scaled - Si_q) 
                    -  jnp.sin(lambda_val * c_mat * q) / q_scaled ) * f_nfw_val 
        
        #u_ell_m = jnp.clip(u_ell_m, 0.0, 1.0)
        return ell, u_ell_m




    def compute_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the galaxy HOD tracer u_ell.
        For galaxy HOD:, 
            First moment:     W_g / ng_bar * [Nc + Ns * u_ell_m]
            Second moment:    W_g^2 / ng_bar^2 * [Ns^2 * u_ell_m^2 + 2 * Ns * u_ell_m]
        You cannot simply take u_ell_g**2.
        """
        Ns = self.get_N_satellites(m, params=params)
        Nc = self.get_N_centrals(m, params=params)
        ng = self.get_ng_bar_at_z(z, params=params) * (params["H0"]/100)**3
        W  = self.get_wg_at_z(z, params=params)
        ell, u_m = self.compute_u_m_ell(z, m, params=params)
    
        moment_funcs = [
            lambda _: (W/ng)[:, None] * (Nc[:, None] + Ns[:, None] * u_m),
            lambda _: (W**2/ng**2)[:, None] * (Ns[:, None]**2 * u_m**2 + 2*Ns[:, None] * u_m),
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell
  
        

    # This code is for the Hankel transform. We may be able to remove it once JAX resolved the bugs in SiCi  
    def _nfw_profile(self, r, r_s, rho_s, r_out):
        x = r / r_s
        rho = rho_s / (x * (1 + x)**2)
        return jnp.where(r <= r_out, rho, 0.0)

    def _rho_s_from_M(self, M, r_s, c):
        return M / (4.0 * jnp.pi * r_s**3 * (jnp.log1p(c) - c / (1 + c)) + 1e-30)

    def compute_u_m_ell(self, z, m_array, lambda_val=1.0, params=None):
        m_array = jnp.atleast_1d(m_array)
    
        # compute halo quantities
        r200c = self.emulator.get_r_delta_of_m_delta_at_z(200, m_array, z, params=params)
        c200c = self.c_Duffy2008(z, m_array)
        r_s = r200c / c200c
        r_out = lambda_val * r200c
    
        # avoid zero radius
        r_min = 1e-4
        r_grid = jnp.clip(self.r_grid, a_min=r_min)
    
        rho_s = jax.vmap(self._rho_s_from_M)(m_array, r_s, jnp.full_like(m_array, c200c))
    
        # integrand for Hankel
        def integrand_single(r_s_i, rho_s_i, r_out_i):
            rho_r = self._nfw_profile(r_grid, r_s_i, rho_s_i, r_out_i)
            return 4 * jnp.pi * r_grid**2 * rho_r
    
        integrand = jax.vmap(integrand_single)(r_s, rho_s, r_out)
    
        # Hankel transform
        k, u_k = self.hankel.transform(integrand)
        u_k *= jnp.sqrt(jnp.pi / (2 * k[None, :]))
        
        # normalize so u(k→0) = 1
        u_k /= u_k[:, 0:1]
    
        # scale to ell
        chi = self.emulator.get_angular_distance_at_z(z, params=params) * (1+z) * params["H0"]/100
        ell = k[None, :] * chi
        N_m = m_array.shape[0]
        ell = jnp.broadcast_to(ell, (N_m, k.shape[0]))
    
        return ell, u_k 
    

    

