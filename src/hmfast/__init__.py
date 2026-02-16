"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.1.0"
__author__ = "Patrick Janulewicz, Licong Xu, Boris Bolliet"
__email__ = "pj407@cam.ac.uk"


from .download import download_emulators

download_emulators(models=["lcdm", "ede-v2"], skip_existing=True)


from .halo_model import HaloModel
from .emulator_load import EmulatorLoader, EmulatorLoaderPCA
from .emulator import Emulator
from .tracers.tsz import TSZTracer
from .tracers.ksz import KSZTracer
from .tracers.cmb_lensing import CMBLensingTracer
from .tracers.galaxy_lensing import GalaxyLensingTracer
from .tracers.galaxy_hod import GalaxyHODTracer
from .tracers.cib import CIBTracer




__all__ = ["HaloModel", "Emulator", "TSZTracer", "GalaxyHODTracer"]