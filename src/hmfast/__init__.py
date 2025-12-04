"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.1.0"
__author__ = "Patrick Janulewicz, Licong Xu, Boris Bolliet"
__email__ = "pj407@cam.ac.uk"

from .halo_model import HaloModel
from .emulator_load import EmulatorLoader, EmulatorLoaderPCA
from .emulator_eval import Emulator
from .tracers.tsz import TSZTracer
from .tracers.galaxy_hod import GalaxyHODTracer
from .download import download_emulators


__all__ = ["HaloModel", "Emulator", "TSZTracer", "GalaxyHODTracer"]