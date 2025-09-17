"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.1.0"
__author__ = "Boris"
__email__ = "boris@example.com"

from .halo_model import HaloModel
from .emulator import HaloEmulator
from .utils import cosmology_utils

__all__ = ["HaloModel", "HaloEmulator", "cosmology_utils"]