"""
Tests for emulator functionality.
"""

import pytest
import jax.numpy as jnp
from hmfast import HaloEmulator


class TestHaloEmulator:
    """Test halo emulator functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.emulator = HaloEmulator()
        self.k = jnp.logspace(-3, 2, 50)
        self.z = 0.5
        self.cosmology = {
            'Omega_m': 0.3,
            'sigma_8': 0.8,
            'h': 0.7,
            'n_s': 1.0
        }
        
    def test_initialization(self):
        """Test emulator initialization."""
        assert 'Omega_m' in self.emulator.parameter_ranges
        assert 'sigma_8' in self.emulator.parameter_ranges
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        valid_params = {
            'Omega_m': 0.3,
            'sigma_8': 0.8,
            'h': 0.7,
            'n_s': 1.0
        }
        invalid_params = {
            'Omega_m': 0.9,  # Out of range
            'sigma_8': 0.8,
            'h': 0.7,
            'n_s': 1.0
        }
        
        assert self.emulator.validate_parameters(valid_params)
        assert not self.emulator.validate_parameters(invalid_params)
        
    def test_power_spectrum_prediction(self):
        """Test power spectrum prediction."""
        prediction = self.emulator.predict_power_spectrum(
            self.k, self.z, self.cosmology
        )
        
        assert prediction.shape == self.k.shape
        assert jnp.all(jnp.isfinite(prediction))
        
    def test_mass_function_prediction(self):
        """Test mass function prediction."""
        M = jnp.logspace(10, 16, 20)
        prediction = self.emulator.predict_mass_function(
            M, self.z, self.cosmology
        )
        
        assert prediction.shape == M.shape
        assert jnp.all(jnp.isfinite(prediction))