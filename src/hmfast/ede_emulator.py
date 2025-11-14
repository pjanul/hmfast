"""
EDE-v2 cosmological emulator for hmfast.

This module provides fast JAX-compatible emulated cosmological calculations
specifically for the EDE-v2 (Early Dark Energy version 2) model, removing
the dependency on classy_szfast while maintaining the same interface.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Union
from functools import partial
from .load_emulator import EmulatorLoader, EmulatorLoaderPCA 

ede_version = "v1" # hard coded for now but should become a free parameter in future versions

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class EDEEmulator:
    """
    EDE-v2 cosmological emulator with JAX compatibility.
    
    Provides fast emulated predictions for cosmological quantities
    using the EDE-v2 (Early Dark Energy v2) model emulators.
    """
    
    # EDE-v2 emulator metadata
    EMULATOR_DICT = {
        'PKNL': f'PKNL_{ede_version}',
        'PKL': f'PKL_{ede_version}',
        'DAZ': f'DAZ_{ede_version}',
        'HZ': f'HZ_{ede_version}',
        'S8Z': f'S8Z_{ede_version}'
    }
    
    # Default EDE-v2 parameters
    DEFAULT_PARAMS = {
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
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the EDE-v2 emulator.
        
        Parameters
        ----------
        data_path : str, optional
            Path to emulator data directory. If None, uses environment variable.
        """
        # Get data path using same logic as classy_szfast
        self.data_path = self._get_data_path(data_path)
        self.emulator_path = os.path.join(self.data_path, f'ede_{ede_version}_numpy')
        
        # Initialize emulator storage
        self._emulators = {}
        self._load_emulators()
        
        # Set up interpolation grids using actual emulator modes
        self._setup_interpolation_grids_post_load()
        
        # Initialize HMF grid cache to avoid recomputing
        self._hmf_cache = {}
    
    def _get_data_path(self, provided_path: Optional[str] = None) -> str:
        """
        Get the data path using the same logic as classy_szfast.
        
        Parameters
        ----------
        provided_path : str, optional
            User-provided path
            
        Returns
        -------
        str
            Path to data directory
        """
        if provided_path is not None:
            if os.path.exists(provided_path):
                return provided_path
            else:
                raise ValueError(f"Provided data path does not exist: {provided_path}")
        
        # Check environment variable (same as classy_szfast)
        env_path = os.getenv('PATH_TO_CLASS_SZ_DATA')
        
        if env_path is not None:
            # Handle case where env var might end with class_sz_data_directory
            if env_path.endswith("class_sz_data_directory"):
                base_path = env_path
            else:
                base_path = os.path.join(env_path, "class_sz_data_directory")
                
            if os.path.exists(base_path):
                return base_path
            else:
                print(f"Warning: PATH_TO_CLASS_SZ_DATA points to non-existent directory: {base_path}")
        
        # Fall back to default location (same as get_cosmopower_emus)
        home_dir = os.path.expanduser("~")
        default_path = os.path.join(home_dir, "class_sz_data_directory")
        
        if os.path.exists(default_path):
            return default_path
        
        # Last resort: check common locations
        common_paths = [
            "/usr/local/share/class_sz_data",
            os.path.join(os.getcwd(), "..", "class_sz_data"),
            os.path.join(os.getcwd(), "class_sz_data"),
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, f'ede-{ede_version}')):
                return path
        
        raise ValueError(
            f"Could not find emulator data directory. Tried:\n"
            f"  - Provided path: {provided_path}\n"
            f"  - Environment variable PATH_TO_CLASS_SZ_DATA: {env_path}\n"
            f"  - Default: {default_path}\n"
            f"  - Common locations: {common_paths}\n"
            f"Please set PATH_TO_CLASS_SZ_DATA or provide data_path explicitly."
        )
    
    def _load_emulators(self):
        """Load all EDE-v2 emulators."""
        if not os.path.exists(self.emulator_path):
            self._print_directory_info()
            raise FileNotFoundError(
                f"EDE emulator directory not found: {self.emulator_path}\n"
                f"Please ensure this directory exists in your data path."
            )
        
        try:
          
            # Load power spectrum emulators
            self._emulators['PKNL'] = EmulatorLoader(
                os.path.join(self.emulator_path, 'PK', self.EMULATOR_DICT['PKNL'])
            )
            self._emulators['PKL'] = EmulatorLoader(
                os.path.join(self.emulator_path, 'PK', self.EMULATOR_DICT['PKL'])
            )
            self._emulators['HZ'] = EmulatorLoader(
                os.path.join(self.emulator_path, 'growth-and-distances', self.EMULATOR_DICT['HZ'])
            )
            # Load distance and growth emulators
            self._emulators['DAZ'] = EmulatorLoader(
                os.path.join(self.emulator_path, 'growth-and-distances', self.EMULATOR_DICT['DAZ'])
            )
            self._emulators['S8Z'] = EmulatorLoader(
                os.path.join(self.emulator_path, 'growth-and-distances', self.EMULATOR_DICT['S8Z'])
            )
            
            print(f"✓ Successfully loaded {len(self._emulators)} EDE emulators from {self.emulator_path}")
            
        except Exception as e:
            self._print_directory_info()
            raise RuntimeError(f"Failed to load EDE emulators: {e}")
    
    def _print_directory_info(self):
        """Print helpful information about the data directory structure."""
        print(f"\nDEBUG: Data path information:")
        print(f"  Base data path: {self.data_path}")
        print(f"  EDE-{ede_version} path: {self.emulator_path}")
        print(f"  Base path exists: {os.path.exists(self.data_path)}")
        print(f"  EDE-{ede_version} path exists: {os.path.exists(self.emulator_path)}")
        
        if os.path.exists(self.data_path):
            print(f"  Contents of {self.data_path}:")
            try:
                for item in sorted(os.listdir(self.data_path)):
                    item_path = os.path.join(self.data_path, item)
                    item_type = "dir" if os.path.isdir(item_path) else "file"
                    print(f"    - {item} ({item_type})")
            except PermissionError:
                print("    [Permission denied]")
        
        if os.path.exists(self.emulator_path):
            print(f"  Contents of {self.emulator_path}:")
            try:
                for item in sorted(os.listdir(self.emulator_path)):
                    item_path = os.path.join(self.emulator_path, item)
                    item_type = "dir" if os.path.isdir(item_path) else "file"
                    print(f"    - {item} ({item_type})")
            except PermissionError:
                print("    [Permission denied]")
        
        print(f"\n  Expected structure:")
        print(f"  {self.emulator_path}/")
        for subdir, files in [
            ('TTTEEE', ['TT_v2.npz', 'TE_v2.npz', 'EE_v2.npz']),
            ('PP', ['PP_v2.npz']),
            ('PK', ['PKNL_v2.npz', 'PKL_v2.npz']),
            ('derived-parameters', ['DER_v2.npz']),
            ('growth-and-distances', ['DAZ_v2.npz', 'HZ_v2.npz', 'S8Z_v2.npz'])
        ]:
            print(f"    ├── {subdir}/")
            for file in files:
                print(f"    │   └── {file}")
        print()
    
    def _setup_interpolation_grids(self):
        """Set up interpolation grids for z-dependent quantities."""
        # Get actual z-grids from emulators after loading
        pass  # Will be called after _load_emulators()
        
    def _setup_interpolation_grids_post_load(self):
        """Set up interpolation grids after emulators are loaded."""
        # Get z-grid from DAZ emulator (z=1 to z=4999)
        self.daz_z_grid = jnp.array(self._emulators['DAZ'].modes, dtype=jnp.float64)
        
        # Get z-grid from HZ emulator
        self.hz_z_grid = jnp.array(self._emulators['HZ'].modes, dtype=jnp.float64)
        
        # Get z-grid from S8Z emulator  
        self.s8z_z_grid = jnp.array(self._emulators['S8Z'].modes, dtype=jnp.float64)

        
        if ede_version == "v2":

            self.cp_ndspl_k = 1
            self.cp_nk = 1000
        
        else:
        
            self.cp_ndspl_k = 10
            self.cp_nk = 5000
    
        # Get z-grid from DAZ emulator following classy_szfast.py:271-272
        # cp_z_interp = self.linspace(0.,self.cp_z_interp_zmax,5000)
        self.cp_z_interp_zmax = 20.0
        self.cp_z_interp = jnp.linspace(0.0, self.cp_z_interp_zmax, 5000, dtype=jnp.float64)

            
        
        # k grid for power spectra (EDE-v2 range from classy_szfast)
        # Original modes are just indices, need to construct actual k values
        n_k = len(self._emulators['PKL'].modes)  # Get number of k points from emulator

        k_min = 1e-4  # h/Mpc, from classy_szfast EDE-v1 settings
        k_max = 50.0  # h/Mpc, from classy_szfast EDE-v1 settings  

        
        if ede_version == "v2":
            k_min = 5e-4  # h/Mpc, from classy_szfast EDE-v2 settings
            k_max = 10.0  # h/Mpc, from classy_szfast EDE-v2 settings  

       
        self.k_grid = jnp.geomspace(k_min, k_max, n_k, dtype=jnp.float64)
        
        # P(k) scaling factor for EDE-v2 (from classy_szfast line 261)
        if ede_version == "v2":
            self.pk_power_fac = self.k_grid**(-3)  # k^(-3) factor
        else:
            ls = jnp.arange(2,self.cp_nk+2)[::self.cp_ndspl_k] # jan 10 ndspl
            dls = ls*(ls+1.)/2./jnp.pi
            self.pk_power_fac= (dls)**-1
        
        # Maximum redshift for power spectrum grid
        self.pk_grid_zmax = 4999.0
    
    def _merge_with_defaults(self, params_dict: Dict[str, Union[float, jnp.ndarray]]) -> Dict[str, Union[float, jnp.ndarray]]:
        """Merge user parameters with defaults."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(params_dict)
        return merged
    
    @partial(jax.jit, static_argnums=(0,))
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
    
    
    def get_rho_crit_at_z(self, 
                         z: Union[float, jnp.ndarray], 
                         params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get critical density at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Critical density(s) in (Msun/h) / (Mpc/h)^3
        """
        # Get Hubble parameter
        
        H_z = self.get_hubble_at_z(z, params_values_dict)
        h = (params_values_dict["H0"]/100)
        
        # Convert to critical density
        # rho_crit = 3 H^2 / (8 pi G)
        # Using units: rho_crit_over_h2_in_GeV_per_cm3 = 1.0537e-5
        rho_crit_factor = (3./ (8. * jnp.pi * (6.67430e-11) * 1.98847e30)) * 3.0856775814913673e22 * (299792458.0)**2 # (3./ (8. * pi * G * M_sun) ) * Mpc_over_m * c**2 
        
        return rho_crit_factor * (H_z/h)**2 
    
    def get_pkl_at_z(self, 
                    z: Union[float, jnp.ndarray], 
                    params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get linear power spectrum at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Linear power spectrum and k array
        """
        
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Add redshift to parameters for prediction
        z_val = jnp.atleast_1d(z)[0] if hasattr(z, '__len__') else z
        merged_params['z_pk_save_nonclass'] = z_val  # Remove float() conversion for JAX compatibility
        
        # Check if redshift is within training range (avoid boolean conversion in JAX)
        # if z_val > self.pk_grid_zmax:
        #     raise ValueError(f"Redshift z={z_val:.3f} exceeds maximum training redshift z_max={self.pk_grid_zmax:.1f}")
        
        # Direct prediction returns log10(P(k))
        pkl_log = self._emulators['PKL'].predictions(merged_params)
        # Convert to linear scale and apply scaling factor like classy_szfast
        pkl = 10.0**pkl_log * self.pk_power_fac
        
        return pkl, self.k_grid
    
    # Alias to match classy_szfast naming convention
    def calculate_pkl_at_z(self, 
                          z_asked: Union[float, jnp.ndarray],
                          params_values_dict: Dict[str, Union[float, jnp.ndarray]] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Alias for get_pkl_at_z() to match classy_szfast naming convention.
        
        Parameters
        ----------
        z_asked : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Linear power spectrum and k array
        """
        if params_values_dict is None:
            raise ValueError("params_values_dict is required")
        return self.get_pkl_at_z(z_asked, params_values_dict)
    
    def get_pknl_at_z(self, 
                     z: Union[float, jnp.ndarray], 
                     params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get nonlinear power spectrum at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Nonlinear power spectrum and k array
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        z_val = jnp.atleast_1d(z)[0] if hasattr(z, '__len__') else z
        merged_params['z_pk_save_nonclass'] = float(z_val)
        
        # Check if redshift is within training range
        if z_val > self.pk_grid_zmax:
            raise ValueError(f"Redshift z={z_val:.3f} exceeds maximum training redshift z_max={self.pk_grid_zmax:.1f}")
        
        # Direct prediction returns log10(P_nl(k))
        pknl_log = self._emulators['PKNL'].predictions(merged_params)
        # Convert to linear scale and apply scaling factor like classy_szfast
        pknl = 10.0**pknl_log * self.pk_power_fac
        return pknl, self.k_grid
    
    def calculate_chi(self, **params_values_dict):
        """
        Calculate comoving distance (chi) following classy_szfast.py:962-1013
        
        This is the internal method that sets up chi_interp following the 
        classy_szfast pattern exactly.
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get D_A predictions from emulator (classy_szfast.py:997)
        daz_log_predictions = self._emulators['DAZ'].predictions(merged_params)

        # For EDE-v2, insert D_A(z=0) = 0 at beginning (classy_szfast.py:998). For ede-v1, this must be commented out
        if ede_version == "v2":
            cp_predicted_da = 10.0**daz_log_predictions
            cp_predicted_da = jnp.insert(cp_predicted_da, 0, 0.0)

        else: 
            cp_predicted_da = daz_log_predictions
         
        # Convert D_A to chi by multiplying by (1+z) (classy_szfast.py:1007)
        chi_values = cp_predicted_da * (1.0 + self.cp_z_interp)
        
        # Create interpolation function (classy_szfast.py:1005-1013)
        def chi_interp(z_requested):
            return jnp.interp(z_requested, self.cp_z_interp, chi_values, left=jnp.nan, right=jnp.nan)
        
        self.chi_interp = chi_interp

    def calculate_hubble(self, **params_values_dict):
        """
        Calculate Hubble parameter H(z) following classy_szfast.py:915-960
        
        This method sets up hz_interp following the classy_szfast pattern exactly.
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get H(z) predictions from emulator (classy_szfast.py:936,948)
        hz_log_predictions = self._emulators['HZ'].predictions(merged_params)
        cp_predicted_hubble = 10.0**hz_log_predictions
        
        # Create interpolation function (classy_szfast.py:942-943, 952-960)
        def hz_interp(z_requested):
            return jnp.interp(z_requested, self.cp_z_interp, cp_predicted_hubble, left=jnp.nan, right=jnp.nan)
        
        self.hz_interp = hz_interp

    def get_angular_distance_at_z(self, 
                                 z: Union[float, jnp.ndarray], 
                                 params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get angular diameter distance D_A(z) following classy_szfast.py:336
        
        D_A(z) = chi(z) / (1+z) where chi is the comoving distance
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Angular diameter distance D_A(z) in Mpc/h
        """
        # Calculate comoving distance interpolator
        self.calculate_chi(**params_values_dict)
        
        # Return D_A = chi/(1+z) following classy_szfast.py:336
        return self.chi_interp(z) / (1.0 + z)

    def get_hubble_at_z(self, 
                       z: Union[float, jnp.ndarray], 
                       params_values_dict: Dict[str, Union[float, jnp.ndarray]],
                       units: str = "1/Mpc") -> jnp.ndarray:
        """
        Get Hubble parameter H(z) following classy_szfast.py:1075-1079
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
        units : str
            Units for H(z): "1/Mpc" or "km/s/Mpc"
            
        Returns
        -------
        jnp.ndarray
            Hubble parameter H(z)
        """
        # Calculate Hubble interpolator
        self.calculate_hubble(**params_values_dict)
        
        # Unit conversion factors from classy_szfast.py:20
        # H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": Const.c_km_s}
        H_units_conv_factor = {"1/Mpc": 1.0, "km/s/Mpc": 299792.458}
        
        # Return H(z) with unit conversion (classy_szfast.py:1077)
        return self.hz_interp(z) * H_units_conv_factor[units] 

    def Hubble(self, z: Union[float, jnp.ndarray]) -> jnp.ndarray:
        """
        JAX-compatible Hubble parameter method following classy_sz notebook pattern.
        
        This method mimics the classy_sz.Hubble(z) interface for drop-in compatibility
        with JAX workflows. Uses the most recent parameter set from last calculation.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s) 
            
        Returns
        -------
        jnp.ndarray
            Hubble parameter H(z) in 1/Mpc units (following classy_sz convention)
        """
        if not hasattr(self, 'hz_interp'):
            raise RuntimeError("Hubble interpolator not initialized. Call get_hubble_at_z() first.")
        
        # Use 1/Mpc units like classy_sz.Hubble() method
        return self.hz_interp(z)
    
    def get_all_relevant_params(self, params_values_dict: Dict[str, Union[float, jnp.ndarray]] = None) -> Dict[str, float]:
        """
        Get all relevant cosmological parameters following classy_szfast pattern.
        
        Following classy_szfast.py:302-321 for parameter derivation.
        
        Parameters
        ----------
        params_values_dict : dict, optional
            Input cosmological parameters
            
        Returns
        -------
        dict
            Dictionary with all derived cosmological parameters
        """
        if params_values_dict is None:
            params_values_dict = {}
        
        params_values = params_values_dict.copy()
        
        # Default parameters (from emulator training)
        defaults = {
            'omega_b': 0.02242,
            'omega_cdm': 0.11933,
            'H0': 67.66,
            'tau_reio': 0.0561,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.9665,
            'N_ur': 2.0328,
            'deg_ncdm': 1.0,
            'm_ncdm': 0.06,
            'T_cmb': 2.7255,
        }
        
        # Update with defaults
        for key, value in defaults.items():
            if key not in params_values:
                params_values[key] = value
        
        # Derive additional parameters following classy_szfast.py:308-320
        params_values['h'] = params_values['H0'] / 100.0
        params_values['Omega_b'] = params_values['omega_b'] / params_values['h']**2
        params_values['Omega_cdm'] = params_values['omega_cdm'] / params_values['h']**2
        
        # Radiation density (following classy_szfast calculation)
        sigma_B = 5.6704004737209545e-08  # Stefan-Boltzmann constant
        params_values['Omega0_g'] = (4.0 * sigma_B / 2.99792458e10 * (params_values['T_cmb']**4)) / (3.0 * 2.99792458e10**2 * 1.0e10 * params_values['h']**2 / 8.262056120185e-10 / 8.0 / jnp.pi / 6.67430e-11)
        params_values['Omega0_ur'] = params_values['N_ur'] * 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * params_values['Omega0_g']
        params_values['Omega0_ncdm'] = params_values['deg_ncdm'] * params_values['m_ncdm'] / (93.14 * params_values['h']**2)
        params_values['Omega_Lambda'] = 1.0 - params_values['Omega0_g'] - params_values['Omega_b'] - params_values['Omega_cdm'] - params_values['Omega0_ncdm'] - params_values['Omega0_ur']
        params_values['Omega0_m'] = params_values['Omega_cdm'] + params_values['Omega_b'] + params_values['Omega0_ncdm']
        params_values['Omega0_r'] = params_values['Omega0_ur'] + params_values['Omega0_g']
        params_values['Omega0_m_nonu'] = params_values['Omega0_m'] - params_values['Omega0_ncdm']
        params_values['Omega0_cb'] = params_values['Omega0_m_nonu']
        
        # Critical density (corrected to match standard cosmological value)
        # Using standard value: ρ_crit = 2.78e11 h^2 Msun/h per (Mpc/h)^3
        params_values['Rho_crit_0'] = 2.77528234822e11 * params_values['h']**2  
        
        return params_values
    
    
    def get_delta_mean_from_delta_crit_at_z(self, delta_crit, z, params_values_dict=None):
        """
        Convert critical density to mean density at given redshifts.
        Following classy_szfast pattern.
        
        For Δ_crit = 200, we typically get Δ_mean ≈ 200 * Ω_m(z) / Ω_m(0)
        """
        params = self.get_all_relevant_params(params_values_dict)
                
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        delta_mean = delta_crit / Omega_m_z
        return delta_mean

    def get_r_delta_of_m_delta_at_z(self, delta, m_delta, z, params_values_dict=None):
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
        params_values_dict : dict, optional
            Dictionary of cosmological parameters to use when computing the critical density.
    
        Returns
        -------
        float
            Radius r_delta (e.g., R_200) within which the average density equals delta * rho_crit(z).
        """
        
        rho_crit = self.get_rho_crit_at_z(z,params_values_dict=params_values_dict)
        return (3.0 * m_delta / (4.0 * jnp.pi * delta * rho_crit))**(1./3.)


    def dVdzdOmega(self, z, params_values_dict=None):
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
        rparams = self.get_all_relevant_params(params_values_dict)
        h = rparams["h"]
        dAz = self.get_angular_distance_at_z(z, params_values_dict=params_values_dict) * h
        Hz = self.get_hubble_at_z(z, params_values_dict=params_values_dict) / h  # in Mpc^(-1) h

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

    
    # Alias to match classy_szfast naming convention  
    def calculate_pknl_at_z(self,
                           z_asked: Union[float, jnp.ndarray], 
                           params_values_dict: Dict[str, Union[float, jnp.ndarray]] = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Alias for get_pknl_at_z() to match classy_szfast naming convention.
        
        Parameters
        ----------
        z_asked : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Nonlinear power spectrum and k array
        """
        if params_values_dict is None:
            raise ValueError("params_values_dict is required")
        return self.get_pknl_at_z(z_asked, params_values_dict)
    
    def get_sigma8_at_z(self, 
                       z: Union[float, jnp.ndarray], 
                       params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get sigma8 at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            sigma8 value(s)
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get predictions on full grid
        s8_predictions = self._emulators['S8Z'].predictions(merged_params)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, s8_predictions, self.s8z_z_grid)
    
    
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