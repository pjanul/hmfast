"""
Tests for halo model functionality.
"""

import pytest
import jax.numpy as jnp
from hmfast import HaloModel


class TestHaloModel:
    """Test halo model calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.halo_model = HaloModel()
        self.k = jnp.logspace(-3, 2, 50)  # h/Mpc
        self.z = 0.5
        
    def test_initialization(self):
        """Test halo model initialization."""
        assert self.halo_model.cosmology['h'] == 0.6736
        assert self.halo_model.mass_function == "tinker08"
        assert self.halo_model.bias_function == "tinker10"
        
    def test_power_spectrum_shapes(self):
        """Test power spectrum output shapes."""
        p1h = self.halo_model.power_spectrum_1halo(self.k, self.z)
        p2h = self.halo_model.power_spectrum_2halo(self.k, self.z)
        ptot = self.halo_model.power_spectrum_total(self.k, self.z)
        
        assert p1h.shape == self.k.shape
        assert p2h.shape == self.k.shape
        assert ptot.shape == self.k.shape
        
    def test_mass_function_shape(self):
        """Test mass function output shape."""
        M = jnp.logspace(10, 16, 20)  # Msun/h
        mf = self.halo_model.mass_function(M, self.z)
        
        assert mf.shape == M.shape
        
    def test_bias_function_shape(self):
        """Test bias function output shape."""
        M = jnp.logspace(10, 16, 20)  # Msun/h
        bias = self.halo_model.bias_function(M, self.z)
        
        assert bias.shape == M.shape