"""
EDE-v2 cosmological emulator for hmfast.

This module provides fast JAX-compatible emulated cosmological calculations
specifically for the EDE-v2 (Early Dark Energy version 2) model.
"""

import os
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Union
from functools import partial
from .emulator_load import EmulatorLoader, EmulatorLoaderPCA 

ede_version = "v1" # hard coded for now but should become a free parameter in future versions

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class BaseEmulator:
    """ 
    A parent emulator class which contains the general functions that are needed across all emulator types. 
    This class will have children (e.g. growth-and-distances, PK, derived-parameters, TTTEEE, PP) that will inherit its general functionality.
    """
    
    @staticmethod
    def get_default_params():
        default_params = {
            'fEDE': 0.001,
            'tau_reio': 0.054,
            'H0': 67.66,
            'ln10^{10}A_s': 3.047,
            'omega_b': 0.02242,
            'omega_cdm': 0.11933,
            'n_s': 0.9665,
            'log10z_c': 3.562,
            'thetai_scf': 2.83,
            'r': 0.,
            'N_ur': 0.00441,  # For Neff = 3.044
            'N_ncdm': 1,
            'deg_ncdm': 3,
            'm_ncdm': 0.02,
            'T_cmb': 2.7255
        }

        return default_params

   
    def __init__(self, params=None):
        self.params = self._merge_with_defaults(params)
        

    def _load_emulators(self, subfolder: str, emulator_dict: Dict[str, str]):
        if not os.path.exists(self.emulator_path):
            raise FileNotFoundError(f"Emulator directory not found: {self.emulator_path}")
        
        self._emulators = {}
        for key, filename in emulator_dict.items():
            path = os.path.join(self.emulator_path, subfolder, filename)
            self._emulators[key] = EmulatorLoader(path)
        
        print(f"✓ Successfully loaded {len(self._emulators)} emulators from {self.emulator_path}")

    def _merge_with_defaults(self, params):
        merged = self.get_default_params().copy()
        if params:
            merged.update(params)
        return merged

    def get_all_relevant_params(self, params: Dict[str, Union[float, jnp.ndarray]] = None) -> Dict[str, float]:
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
        if params is None:
            params = {}
        
        p = self._merge_with_defaults(params)
        
        # Derive additional parameters following classy_szfast.py:308-320
        p['h'] = p['H0'] / 100.0
        p['Omega_b'] = p['omega_b'] / p['h']**2
        p['Omega_cdm'] = p['omega_cdm'] / p['h']**2
        
        # Radiation density (following classy_szfast calculation)
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

    def validate_parameters(self, params_dict: Dict[str, Union[float, jnp.ndarray]]) -> bool:
        """
        Validate that parameters are within emulator training ranges.
        
        Parameters
        ----------
        params_dict : dict
            Parameters to validate
            
        Returns
        -------
        bool
            True if parameters are valid
        """
        # Define valid ranges for EDE-v2 parameters
        valid_ranges = {
            'fEDE': (0.0, 0.15),
            'H0': (50.0, 90.0),
            'omega_b': (0.015, 0.035),
            'omega_cdm': (0.08, 0.2),
            'ln10^{10}A_s': (2.5, 3.5),
            'n_s': (0.85, 1.15),
            'tau_reio': (0.02, 0.15),
            'log10z_c': (3.0, 4.5),
            'thetai_scf': (1.0, 5.0),
        }
        
        for param, value in params_dict.items():
            if param in valid_ranges:
                min_val, max_val = valid_ranges[param]
                val = float(value)
                if not (min_val <= val <= max_val):
                    return False
        
        return True
    
        
   

class CosmoEmulator(BaseEmulator):
    """
    Cosmological emulator with JAX compatibility.

    This class inherits all of BaseEmulator's functionality but is specialised for cosmological calculations,
    such as H(z), d_A(z), sigma8(z), and other quantities derived from them.
    """
    
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the cosmology emulator.
        
        Parameters
        ----------
        data_path : str, optional
            Path to emulator data directory. If None, uses environment variable.
        """

        # Emulator metadata
        EMULATOR_DICT = {'DAZ': f'DAZ_{ede_version}', 'HZ': f'HZ_{ede_version}', 'S8Z': f'S8Z_{ede_version}'}
   
        # Get data path using same logic as classy_szfast
        self.data_path = data_path
        self.emulator_path = os.path.join(self.data_path, f'ede_{ede_version}_numpy')
        
        # Initialize emulator storage
        self._emulators = {}
        self._load_emulators("growth-and-distances", EMULATOR_DICT)
        
        # Set up interpolation grids using actual emulator modes
        self._setup_interpolation_grids_post_load()
        
    
        
    def _setup_interpolation_grids_post_load(self):
        """Set up interpolation grids after emulators are loaded."""
        
        # Get z-grid from DAZ emulator (z=1 to z=4999), HZ emulator, and S8Z emulator
        self.daz_z_grid = jnp.array(self._emulators['DAZ'].modes, dtype=jnp.float64)
        self.hz_z_grid = jnp.array(self._emulators['HZ'].modes, dtype=jnp.float64) 
        self.s8z_z_grid = jnp.array(self._emulators['S8Z'].modes, dtype=jnp.float64)

        # Get z-grid from DAZ emulator following classy_szfast.py:271-272
        self.cp_z_interp_zmax = 20.0
        self.cp_z_interp = jnp.linspace(0.0, self.cp_z_interp_zmax, 5000, dtype=jnp.float64)

    
    def _interpolate_z_dependent(self, 
                                z_requested: Union[float, jnp.ndarray], 
                                predictions: jnp.ndarray, 
                                z_grid: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate z-dependent quantities to requested redshifts.
        
        Parameters
        ----------
        z_requested : float or jnp.ndarray
            Requested redshift(s)
        predictions : jnp.ndarray
            Emulator predictions on z_grid
        z_grid : jnp.ndarray
            Redshift grid used for predictions
            
        Returns
        -------
        jnp.ndarray
            Interpolated values
        """
        # Ensure z_requested is an array
        z_req = jnp.atleast_1d(z_requested)
        
        # Perform linear interpolation
        result = jnp.interp(z_req, z_grid, predictions, left=jnp.nan, right=jnp.nan)
        
        # Return scalar if input was scalar
        if jnp.ndim(z_requested) == 0:
            return result[0]
        return result
        
    
    def get_hubble_at_z(self, 
                       z: Union[float, jnp.ndarray], 
                       params: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
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
            Hubble parameter(s) in km/s/Mpc
        """
        merged_params = self._merge_with_defaults(params)
        
        # Get predictions on full grid. 
        hz_predictions = 10**self._emulators['HZ'].predictions(merged_params)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, hz_predictions, self.cp_z_interp)
        

    def get_angular_distance_at_z(self, 
                                  z: Union[float, jnp.ndarray], 
                                  params: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
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
        merged_params = self._merge_with_defaults(params)
        
        # Get predictions on full grid
        da_predictions = self._emulators['DAZ'].predictions(merged_params)

        # If we're dealing with EDE-v2, we need to convert from log and add an extra element to the array due to size differences in the emulators
        if ede_version == "v2":
            da_predictions = 10.0**da_predictions  
            da_predictions = jnp.insert(da_predictions, 0, 0.0)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, da_predictions, self.cp_z_interp)


    def get_sigma8_at_z(self, 
                       z: Union[float, jnp.ndarray], 
                       params: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
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
        merged_params = self._merge_with_defaults(params)
        
        # Get predictions on full grid
        s8_predictions = self._emulators['S8Z'].predictions(merged_params)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, s8_predictions, self.cp_z_interp)


    def get_rho_crit_at_z(self, 
                         z: Union[float, jnp.ndarray], 
                         params: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
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
        # Get Hubble parameter
        
        H_z = self.get_hubble_at_z(z, params)
        h = (params["H0"]/100)
        
        # Convert to critical density
        # rho_crit = 3 H^2 / (8 pi G) * Mpc_over_m * c**2 
        rho_crit_factor = (3./ (8. * jnp.pi * (6.67430e-11) * 1.98847e30)) * 3.0856775814913673e22 * (299792458.0)**2 
        
        return rho_crit_factor * (H_z/h)**2 
    
    
    def get_delta_mean_from_delta_crit_at_z(self, delta_crit, z, params=None):
        """
        Convert critical density to mean density at given redshifts.
        
        For Δ_crit = 200, we typically get Δ_mean ≈ 200 * Ω_m(z) / Ω_m(0)
        """
        params = self.get_all_relevant_params(params)
                
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        delta_mean = delta_crit / Omega_m_z
        return delta_mean
        

    def get_r_delta_of_m_delta_at_z(self, delta, m_delta, z, params=None):
        """
        Compute the halo radius corresponding to a given mass and overdensity at redshift z.
    
        Parameters
        ----------
        delta : float
            Overdensity parameter relative to the critical density (e.g., 200 for M_200).
        m_delta : float
            Halo mass enclosed within the overdensity radius, in the same units as used for rho_crit.
        z : float
            Redshift at which to compute the radius.
        params : dict, optional
            Dictionary of cosmological parameters to use when computing the critical density.
    
        Returns
        -------
        float
            Radius r_delta (e.g., R_200) within which the average density equals delta * rho_crit(z).
        """
        
        rho_crit = self.get_rho_crit_at_z(z,params=params)
        return (3.0 * m_delta / (4.0 * jnp.pi * delta * rho_crit))**(1./3.)


    def get_dVdzdOmega_at_z(self, z, params=None):
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
        rparams = self.get_all_relevant_params(params)
        h = rparams["h"]
        dAz = self.get_angular_distance_at_z(z, params=params) * h
        Hz = self.get_hubble_at_z(z, params=params) / h  # in Mpc^(-1) h

        return (1 + z)**2 * dAz**2 / Hz
   

    def z_grid(self) -> jnp.ndarray:
        """
        Return the redshift grid used by emulators (following classy_sz interface).
        
        Returns
        -------
        jnp.ndarray
            Redshift grid from 0 to 20
        """
        return self.cp_z_interp
   


class PkEmulator(BaseEmulator):
    """
    Cosmological emulator with JAX compatibility.
    
    Provides fast emulated predictions for matter power spectra P(k)
    using the CosmoPower emulators.
    """
     
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the P(k) emulator.
        
        Parameters
        ----------
        data_path : str, optional
            Path to emulator data directory. If None, uses environment variable.
        """
        # Emulator metadata
        EMULATOR_DICT = {'PKNL': f'PKNL_{ede_version}', 'PKL': f'PKL_{ede_version}'}
        
        # Get data path 
        self.data_path = data_path
        self.emulator_path = os.path.join(self.data_path, f'ede_{ede_version}_numpy')
        
        # Initialize emulator storage
        self._emulators = {}
        self._load_emulators("PK", EMULATOR_DICT)
        
        # Set up interpolation grids using actual emulator modes
        self._setup_interpolation_grids_post_load()
        
   

    def _setup_interpolation_grids_post_load(self):

        is_v2 = (ede_version == "v2")

        self.cp_ndspl_k = 1 if is_v2 else 10
        self.cp_nk      = 1000 if is_v2 else 5000
       
        # Original modes are just indices, need to construct actual k values
        n_k = len(self._emulators['PKL'].modes)  # Get number of k points from emulator
        k_min = 5e-4 if is_v2 else 1e-4
        k_max = 10.0 if is_v2 else 50.0       
        self.k_grid = jnp.geomspace(k_min, k_max, n_k, dtype=jnp.float64)
        
        # P(k) scaling factor 
        if is_v2:
            self.pk_power_fac = self.k_grid**(-3)  # k^(-_interpolate_z_dependent3) factor
        else:
            ls = jnp.arange(2,self.cp_nk+2)[::self.cp_ndspl_k] # jan 10 ndspl
            self.pk_power_fac= (ls*(ls+1.)/2./jnp.pi)**-1
        
        # Maximum redshift for power spectrum grid
        self.pk_grid_zmax = 4999.0


    def get_pk_at_z(self, 
                    z: Union[float, jnp.ndarray], 
                    params: Dict[str, Union[float, jnp.ndarray]], 
                    linear=True) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get linear power spectrum at redshift z.
        
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
        
        merged_params = self._merge_with_defaults(params)
        
        # Add redshift to parameters for prediction
        z_val = jnp.atleast_1d(z)[0] if hasattr(z, '__len__') else z
        merged_params['z_pk_save_nonclass'] = z_val  # Remove float() conversion for JAX compatibility

        #if z_val > self.pk_grid_zmax:   # this leads to a boolean tracer error, so we'll need to find a different way to enforce this
        #    raise ValueError(f"Redshift z={z_val:.3f} exceeds maximum training redshift z_max={self.pk_grid_zmax:.1f}")
     
        # Direct prediction returns log10(P(k))
        key = 'PKL' if linear else 'PKNL'
        pk_log = self._emulators[key].predictions(merged_params)
       
        # Convert to linear scale and apply scaling factor 
        pk = 10.0**pk_log * self.pk_power_fac
        
        return pk, self.k_grid
    
   
    