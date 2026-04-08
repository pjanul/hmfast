import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import numpy as np

from hmfast.halo_model.profiles import HaloProfile


 
class BaseTracer(ABC):
    """
    Abstract base class for cosmological tracers.
    All tracers to inherit from this class, which forces them to have certain callable functions (e.g. get_u_ell() )
    """
    
    _required_profile_type = HaloProfile 

    def __init__(self, profile=None):
        """
        Initialize the tracer with a validated profile.
        """
        if profile is not None:
            self.profile = profile

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value):
        """
        Enforces type safety: prevents assigning a PressureProfile to a 
        LensingTracer, etc.
        """
        if not isinstance(value, self._required_profile_type):
            raise TypeError(
                f"{self.__class__.__name__} strictly requires a "
                f"{self._required_profile_type.__name__}. "
                f"Received: {type(value).__name__}"
            )
        self._profile = value
        


    def _load_dndz_data(self, path):
        """
        Loads dndz curves in the format (z, phi) for galaxy HOD and galaxy lensing tracers.
        """
        data = np.loadtxt(path)
        x = data[:, 0]
        y = data[:, 1]
        return (jnp.array(x), jnp.array(y))

        
    def _normalize_dndz(self, value):
        """
        Normalizes dndz curves in the format (z, phi) for galaxy HOD and galaxy lensing tracers if needed.
        """
        z = jnp.atleast_1d(jnp.array(value[0]))
        phi = jnp.atleast_1d(jnp.array(value[1]))
        norm = jnp.trapezoid(phi, x=z)
        return (z, phi / norm)


    @abstractmethod
    def kernel(self, emulator, z):
        """
        Compute the tracer's radial kernel W(z). All child classes must have a version of this function implemented.
        """
        pass 
   
  