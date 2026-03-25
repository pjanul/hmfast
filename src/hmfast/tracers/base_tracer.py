import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici
import functools
import mcfit
from abc import ABC, abstractmethod
import numpy as np


from hmfast.defaults import merge_with_defaults


 
class BaseTracer(ABC):
    """
    Abstract base class for cosmological tracers.
    All tracers to inherit from this class, which forces them to have certain callable functions (e.g. get_u_ell() )
    """
    
    def __init__(self, params):
        """
        Initialize the radial grid and Hankel transform.
        """

    @property
    def has_central_contribution(self):
        """ 
        Indicates whether the tracer has a contribution from central terms, such as:
        
            - HOD, which has profile = N_sat * u_k + N_sat 
            - CIB, which has profile = L_sat * u_k + L_sat * L_cen

        For most tracers, profile = prefactor * u_k, meaning that this will be set to False.
        """
        return False


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
    def kernel(self, z, params=None):
        """
        Compute the tracer's radial kernel W(z). All child classes must have a version of this function implemented.
        """
        pass 
   
    @abstractmethod
    def u_k(self, k, m, z, moment=1, params=None):
        """
        Compute the tracer's profile u_k(k, m, z). All child classes must have a version of this function implemented.
        """
        pass 

    

   
  