import os
import jax
import jax.numpy as jnp
from typing import Dict, Union
from hmfast.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path

jax.config.update("jax_enable_x64", True)


_COSMO_MODELS = {
    0: {"suffix": "v1", "subdir": "lcdm"},
    1: {"suffix": "mnu_v1", "subdir": "mnu"},
    2: {"suffix": "neff_v1", "subdir": "neff"},
    3: {"suffix": "w_v1", "subdir": "wcdm"},
    4: {"suffix": "v1", "subdir": "ede"},
    5: {"suffix": "v1", "subdir": "mnu-3states"},
    6: {"suffix": "v2", "subdir": "ede"},
}


class Emulator:
    """
    Unified, JAX-native, lazily-loaded emulator interface.
    """

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def __init__(self, cosmo_model: int = 0):
        self.cosmo_model = cosmo_model
        self._model_info = _COSMO_MODELS[cosmo_model]

        self._base_path = os.path.join(
            get_default_data_path(),
            self._model_info["subdir"],
        )

        # atomic emulator cache
        self._emu = {}
        # cached grids / derived constants
        self._z_grid_bg = jnp.linspace(0.0, 20.0, 5000, dtype=jnp.float64)    # z grid for background quantities such as H(z), d_A(z), sigma8(z)
        self._z_grid_pk = jnp.linspace(0.0, 5.0, 1000, dtype=jnp.float64)     # z grid for Pk(z)
        self._k_grid = None
        self._pk_power_fac = None
        self._setup_pk_grid()                  

    # ------------------------------------------------------------------
    # atomic lazy loader (Python-side only)
    # ------------------------------------------------------------------

    def _load_emulator(self, key: str):
        if key in self._emu:
            return self._emu[key]
    
        key_map = {
            "DAZ":  ("growth-and-distances", EmulatorLoader),
            "HZ":   ("growth-and-distances", EmulatorLoader),
            "S8Z":  ("growth-and-distances", EmulatorLoader),
            "PKL":  ("PK", EmulatorLoader),
            "PKNL": ("PK", EmulatorLoader),
            "TT":   ("TTTEEE", EmulatorLoader),
            "EE":   ("TTTEEE", EmulatorLoader),
            "TE":   ("TTTEEE", EmulatorLoaderPCA),
            "PP":   ("PP", EmulatorLoader),
            "DER":  ("derived-parameters", EmulatorLoader),
        }
    
        try:
            subdir, loader_cls = key_map[key]
        except KeyError:
            raise KeyError(f"Unknown emulator key: {key}")
    
        self._emu[key] = loader_cls(os.path.join(self._base_path, subdir, f"{key}_{self._model_info['suffix']}"))
        return self._emu[key]

    # ------------------------------------------------------------------
    # shared grids (lazy, cached)
    # ------------------------------------------------------------------

    def _setup_pk_grid(self):
        if self._k_grid is not None:
            return

        is_ede_v2 = (self.cosmo_model == 6)

        self.cp_ndspl_k = 1 if is_ede_v2 else 10
        self.cp_nk      = 1000 if is_ede_v2 else 5000

        emu = self._load_emulator("PKL")
        n_k = len(emu.modes)

        k_min = 5e-4 if is_ede_v2 else 1e-4
        k_max = 10.0 if is_ede_v2 else 50.0

        self._k_grid = jnp.geomspace(k_min, k_max, n_k, dtype=jnp.float64)

        if is_ede_v2:
            self._pk_power_fac = self._k_grid ** (-3)
        else:
            ls = jnp.arange(2,self.cp_nk+2)[::self.cp_ndspl_k] # jan 10 ndspl
            self._pk_power_fac= (ls*(ls+1.)/2./jnp.pi)**-1

    # ------------------------------------------------------------------
    # JAX-safe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interp_z(z, z_grid, values):
        z = jnp.atleast_1d(z)
        out = jnp.interp(z, z_grid, values, left=jnp.nan, right=jnp.nan)
        return out[0] if out.shape[0] == 1 else out


    # ------------------------------------------------------------------
    # Cosmology
    # ------------------------------------------------------------------

    def hubble_parameter(self, z, params=None):
        """
        Get Hubble parameter at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Hubble parameter(s) in Mpc^(-1)
        """
        
        params = merge_with_defaults(params)
        emu = self._load_emulator("HZ")
        preds = 10.0 ** emu.predictions(params)
        return self._interp_z(z, self._z_grid_bg, preds)

    def angular_diameter_distance(self, z, params=None):
        """
        Get angular diameter distance at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Angular diameter distance(s) in Mpc
        """
        
        params = merge_with_defaults(params)
        emu = self._load_emulator("DAZ")
        preds = emu.predictions(params)

        if self.cosmo_model == 6:
            preds = 10.0 ** preds
            preds = jnp.insert(preds, 0, 0.0)

        return self._interp_z(z, self._z_grid_bg, preds)

    def sigma8(self, z, params=None):
        """
        Get sigma8 at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            sigma8 value(s)
        """
        params = merge_with_defaults(params)
        emu = self._load_emulator("S8Z")
        preds = emu.predictions(params)
        return self._interp_z(z, self._z_grid_bg, preds)


    def get_all_cosmo_params(self, params = None):
        """
        Get all relevant cosmological parameters.
                
        Parameters
        ----------
        params : dict, optional
            Input cosmological parameters
            
        Returns
        -------
        dict
            Dictionary with all derived cosmological parameters
        """
    
        p = merge_with_defaults(params)
        
        # Derive additional parameters following classy_szfast.py:308-320
        p['h'] = p['H0'] / 100.0
        p['Omega_b'] = p['omega_b'] / p['h']**2
        p['Omega_cdm'] = p['omega_cdm'] / p['h']**2
        
        # Radiation density 
        sigma_B = 5.6704004737209545e-08  # Stefan-Boltzmann constant
        p['Omega0_g'] = (4.0 * sigma_B / 2.99792458e10 * (p['T_cmb']**4)) / (3.0 * 2.99792458e10**2 * 1.0e10 * p['h']**2 / 8.262056120185e-10 / 8.0 / jnp.pi / 6.67430e-11)
        p['Omega0_ur'] = p['N_ur'] * 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * p['Omega0_g']
        p['Omega0_ncdm'] = p['deg_ncdm'] * p['m_ncdm'] / (93.14 * p['h']**2)
        p['Omega_Lambda'] = 1.0 - p['Omega0_g'] - p['Omega_b'] - p['Omega_cdm'] - p['Omega0_ncdm'] - p['Omega0_ur']
        p['Omega0_m'] = p['Omega_cdm'] + p['Omega_b'] + p['Omega0_ncdm']
        p['Omega0_r'] = p['Omega0_ur'] + p['Omega0_g']
        p['Omega0_m_nonu'] = p['Omega0_m'] - p['Omega0_ncdm']
        p['Omega0_cb'] = p['Omega0_m_nonu']
        
        # Critical density (corrected to match standard cosmological value)
        # Using standard value: ρ_crit = 2.78e11 h^2 Msun/h per (Mpc/h)^3
        p['Rho_crit_0'] = 2.77528234822e11 * p['h']**2  
        
        return p


    def critical_density(self, z, params=None):
        """
        Get critical density at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Critical density(s) in (Msun/h) / (Mpc/h)^3
        """
        
        params = merge_with_defaults(params)
        # Get Hubble parameter    
        H_z = self.hubble_parameter(z, params)
        h = (params["H0"]/100)
        
        # Convert to critical density
        # rho_crit = 3 H^2 / (8 pi G) * Mpc_over_m * c**2 
        rho_crit_factor = (3./ (8. * jnp.pi * (6.67430e-11) * 1.98847e30)) * 3.0856775814913673e22 * (299792458.0)**2 
        
        return rho_crit_factor * (H_z/h)**2 


    def growth_factor(self, z, params=None):
        """
        Linear growth factor D(z), normalized to D(0)=1.
        """
        
        params = merge_with_defaults(params)
        z = jnp.atleast_1d(z)
    
        k0 = 1e-2  # reference wavenumber
        pk0_grid = jax.vmap(lambda zp: jnp.interp(k0, *self.pk_matter(zp, params=params, linear=True)))(self._z_grid_pk)
        D_grid = jnp.sqrt(pk0_grid / jnp.interp(k0, *self.pk_matter(0.0, params=params, linear=True)))
    
        return jnp.interp(z, self._z_grid_pk, D_grid)


    def growth_rate(self, z, params=None):
        """
        Return the linear growth rate f(z) = d ln D / d ln a.
        """
        
        params = merge_with_defaults(params)
        z = jnp.atleast_1d(z)
        
        D_grid = self.growth_factor(self._z_grid_pk, params=params)
        a_grid = 1.0 / (1.0 + self._z_grid_pk)
        f_grid = jnp.gradient(jnp.log(D_grid), jnp.log(a_grid))
        
        return jnp.interp(z, self._z_grid_pk, f_grid)



    def v_rms_squared(self, z, params=None):
        """
        v_rms^2(z) from linear growth factor and matter power spectrum.
        """
        
        z = jnp.atleast_1d(z)
        k_grid = jnp.geomspace(1e-5, 1e1, 1000)
    
        # P(k, z) on the pk grid
        P_grid = jax.vmap(lambda zp: jnp.interp(k_grid, *self.pk_matter(zp, params=params, linear=True)))(self._z_grid_pk)
    
        a_grid = 1.0 / (1.0 + self._z_grid_pk)
        H_grid = self.hubble_parameter(self._z_grid_pk, params=params)
        f_grid = self.growth_rate(self._z_grid_pk, params=params)
    
        W_grid = f_grid * a_grid * H_grid
        integrand = (W_grid[:, None]**2 / 3) * P_grid * k_grid / (2 * jnp.pi**2)
        vrms2_grid = jax.scipy.integrate.trapezoid(integrand, x=jnp.log(k_grid), axis=1)
    
        return jnp.interp(z, self._z_grid_pk, vrms2_grid)

        
    def delta_crit_to_mean(self, delta_crit, z, params=None):
        """
        Convert critical density to mean density at given redshifts.
        
        For Δ_crit = 200, we typically get Δ_mean ≈ 200 / Ω_m(z)
        """
        
        params = self.get_all_cosmo_params(params)
                
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        delta_mean = delta_crit / Omega_m_z
        return delta_mean

    def delta_vir_to_crit(self, z, params=None):
        """
        Bryan & Norman (1998) virial overdensity with respect to critical density.
        Returns Δ_vir,c at redshift z.
        """
        params = self.get_all_cosmo_params(params)
                
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos

        
        x = Omega_m_z - 1.0
        delta_vir = 18 * jnp.pi**2 + 82 * x - 39 * x**2
        
        return delta_vir
        

    def r_delta(self, z, m, delta, params=None):
        """
        Compute the halo radius corresponding to a given mass and overdensity at redshift z.
    
        Parameters
        ----------
        z : float
            Redshift at which to compute the radius.
        m : float
            Halo mass enclosed within the overdensity radius, in the same units as used for rho_crit.
        delta : float
            Overdensity parameter relative to the critical density (e.g., 200 for M_200).
        
        params : dict, optional
            Dictionary of cosmological parameters to use when computing the critical density.
    
        Returns
        -------
        float
            Radius r_delta (e.g., R_200) within which the average density equals delta * rho_crit(z).
        """
        rho_crit = self.critical_density(z,params=params)
        return (3.0 * m / (4.0 * jnp.pi * delta * rho_crit))**(1./3.)


    def comoving_volume_element(self, z, params=None):
        """
        Comoving volume element per unit redshift and solid angle.

        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters

        Returns
        -------
        float
            dV / dz / dOmega in (Mpc/h)^3 / sr
        """
        cparams = self.get_all_cosmo_params(params)
        h = cparams["h"]
        dAz = self.angular_diameter_distance(z, params=params) * h
        Hz = self.hubble_parameter(z, params=params) / h  # in Mpc^(-1) h

        return (1 + z)**2 * dAz**2 / Hz
   

    # ------------------------------------------------------------------
    # Matter power spectra
    # ------------------------------------------------------------------

    def pk_matter(self, z, params=None, linear=True):
        """
        Get the matter power spectrum at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params : dict
            Cosmological parameters
        linear : bool
            True for linear matter power spectrum, False for nonlinear matter power spectrum
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Power spectrum and k array
        """
        
        params = merge_with_defaults(params)
        params["z_pk_save_nonclass"] = jnp.atleast_1d(z)[0]

        key = "PKL" if linear else "PKNL"
        emu = self._load_emulator(key)

        pk_log = emu.predictions(params)
        pk = 10.0 ** pk_log * self._pk_power_fac

        return self._k_grid, pk

    # ------------------------------------------------------------------
    # CMB
    # ------------------------------------------------------------------

    def cmb_dls(self, params=None, lmax=10000):
        params = merge_with_defaults(params)

        tt = self._load_emulator("TT").ten_to_predictions(params)
        ee = self._load_emulator("EE").ten_to_predictions(params)
        te = self._load_emulator("TE").predictions(params)
        pp = self._load_emulator("PP").ten_to_predictions(params)

        n = min(len(tt), len(ee), len(te), len(pp), lmax - 1)
        ell = jnp.arange(2, n + 2)

        return {
            "ell": ell,
            "tt": tt[:n],
            "ee": ee[:n],
            "te": te[:n],
            "pp": pp[:n] / (2 * jnp.pi),
        }

        
    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------

    def derived_parameters(self, params=None):
        params = merge_with_defaults(params)
        emu = self._load_emulator("DER")
        preds = emu.ten_to_predictions(params)

        names = [  '100*theta_s',
                   'sigma8',
                   'YHe',
                   'z_reio',
                   'Neff',
                   'tau_rec',  # conformal time at which the visibility reaches its maximum (= recombination time)
                   'z_rec', # z at which the visibility reaches its maximum (= recombination redshift)
                   'rs_rec', # comoving sound horizon at recombination in Mpc
                   'chi_rec', # comoving distance to recombination in Mpc
                   'tau_star', # conformal time at which photon optical depth crosses one
                   'z_star', # redshift at which photon optical depth crosses one, i.e., last scattering surface
                   'rs_star', # comoving sound horizon at z_star in Mpc
                   'chi_star', # comoving distance to the last scattering surface in Mpc
                   'rs_drag'] # comoving sound horizon at baryon drag in Mpc

        
        out = {n: preds[i] for i, n in enumerate(names) if i < len(preds)}
        out["h"] = params["H0"] / 100.0
        out["Omega_m"] = (params["omega_b"] + params["omega_cdm"]) / out["h"]**2

        return out


