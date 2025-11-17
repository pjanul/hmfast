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
from .emulator_load import EmulatorLoader, EmulatorLoaderPCA
from .emulator_eval import CosmoEmulator, PkEmulator

__all__ = ["HaloModel", "EDEEmulator", "TSZTracer"]