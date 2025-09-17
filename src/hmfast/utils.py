"""
Utility functions for cosmological calculations.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any
from functools import partial


class cosmology_utils:
    """Utility functions for cosmological calculations."""
    
    @staticmethod
    @jax.jit
    def hubble_parameter(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute Hubble parameter H(z).
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            H(z) in units of H0
        """
        Omega_m = cosmology.get('Omega_m', 0.3153)
        Omega_L = 1.0 - Omega_m  # Flat universe assumption
        
        return jnp.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    
    @staticmethod
    @jax.jit
    def comoving_distance(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute comoving distance to redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Comoving distance in Mpc/h
        """
        # Simple approximation - in practice would integrate
        h = cosmology.get('h', 0.6736)
        c_km_s = 299792.458  # km/s
        H0 = 100 * h  # km/s/Mpc
        
        # Approximate integral for flat LCDM
        Omega_m = cosmology.get('Omega_m', 0.3153)
        Ez_inv_approx = 1.0 / jnp.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
        
        return c_km_s / H0 * z * Ez_inv_approx
    
    @staticmethod
    @jax.jit
    def growth_factor(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute linear growth factor D(z).
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Growth factor normalized to 1 at z=0
        """
        # Approximate growth factor for flat LCDM
        Omega_m = cosmology.get('Omega_m', 0.3153)
        a = 1.0 / (1.0 + z)
        
        # Carroll, Press & Turner 1992 approximation
        omega_a = Omega_m / (Omega_m + (1 - Omega_m) * a**3)
        
        growth = 2.5 * omega_a / (omega_a**(4./7.) - (1 - omega_a) + 
                                  (1 + omega_a/2.) * (1 + (1 - omega_a)/70.))
        
        # Normalize to z=0
        omega_0 = Omega_m
        growth_0 = 2.5 * omega_0 / (omega_0**(4./7.) - (1 - omega_0) + 
                                     (1 + omega_0/2.) * (1 + (1 - omega_0)/70.))
        
        return growth / growth_0 * a
    
    @staticmethod
    @jax.jit
    def sigma8_z(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute sigma_8 at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            sigma_8(z)
        """
        sigma8_0 = cosmology.get('sigma_8', 0.8111)
        D_z = cosmology_utils.growth_factor(z, cosmology)
        
        return sigma8_0 * D_z
    
    @staticmethod
    @jax.jit
    def critical_density(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute critical density at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Critical density in Msun h^2 / Mpc^3
        """
        h = cosmology.get('h', 0.6736)
        H_z = cosmology_utils.hubble_parameter(z, cosmology) * 100 * h  # km/s/Mpc
        
        # Critical density in kg/m^3
        G = 6.67430e-11  # m^3/kg/s^2
        H_z_SI = H_z * 1e3 / (3.086e22)  # 1/s
        rho_crit = 3 * H_z_SI**2 / (8 * jnp.pi * G)  # kg/m^3
        
        # Convert to Msun h^2 / Mpc^3
        Msun = 1.989e30  # kg
        Mpc = 3.086e22  # m
        
        return rho_crit * (Mpc**3 / Msun) * h**2
    
    @staticmethod
    @jax.jit  
    def virial_radius(M: float, z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute virial radius for a halo of mass M.
        
        Parameters
        ----------
        M : float
            Halo mass in Msun/h
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Virial radius in Mpc/h
        """
        Delta_vir = 200.0  # Typical overdensity definition
        rho_crit = cosmology_utils.critical_density(z, cosmology)
        Omega_m = cosmology.get('Omega_m', 0.3153)
        rho_m = Omega_m * rho_crit * (1 + z)**3
        
        # R_vir = (3M / (4π * Delta_vir * rho_m))^(1/3)
        return (3 * M / (4 * jnp.pi * Delta_vir * rho_m))**(1./3.)