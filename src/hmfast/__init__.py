"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.1.0"
__author__ = "Patrick Janulewicz, Licong Xu, Boris Bolliet"
__email__ = "pj407@cam.ac.uk"

from .halo_model import HaloModel
from .utils import interpolate_tracer
from .tracers.tsz import gnfw_pressure_profile, TSZTracer
from .load_emulator import EmulatorLoader, EmulatorLoaderPCA
from .ede_emulator import EDEEmulator

__all__ = ["HaloModel", "EDEEmulator", "EDEEmulator"]