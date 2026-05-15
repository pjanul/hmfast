import jax
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod

from hmfast.halos.massdef import MassDefinition



class Concentration(ABC):
    """
    Abstract base class for all concentration-mass relations.
    All subclasses must implement the c_delta method.
    """
    @abstractmethod
    @partial(jax.jit, static_argnums=(0, 4))
    def c_delta(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="critical")):
        """
        Compute the concentration parameter :math:`c_\\Delta`.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the concentration relation.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.
        mass_definition : MassDefinition, optional
            Target halo mass definition. Defaults to
            ``MassDefinition(delta=200, reference="critical")``.

        Returns
        -------
        float or array-like
            Concentration values with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        pass


        
class ConstantConcentration(Concentration):
    """
    Constant concentration-mass relation.

    Agnostic to the choice of mass definition.

    The concentration parameter :math:`c_\\Delta` is fixed to a user-specified value for all halos.
    """
    def __init__(self, c):
        self.c = c
        pass

    @partial(jax.jit, static_argnums=(0, 4))
    def c_delta(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="critical")):
        """
        Returns a constant value for the concentration parameter, broadcast to the shape of the input masses and redshifts.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the concentration relation.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.
        mass_definition : MassDefinition, optional
            Target halo mass definition. Included for API consistency.

        Returns
        -------
        float or array-like
            Concentration values with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        return jnp.squeeze(jnp.broadcast_to(self.c, (len(jnp.atleast_1d(m)), len(jnp.atleast_1d(z)))))



class D08Concentration(Concentration):
    """
    Concentration-mass relation from `Duffy et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D/abstract>`_.

    The fitted relation is

    .. math::

        c_\\Delta(M, z) = A \\left(\\frac{M}{M_\\mathrm{pivot}}\\right)^B (1+z)^C

    where :math:`A`, :math:`B`, :math:`C`, and :math:`M_\\mathrm{pivot}` are fit parameters.

    Calibrated for 200c, 200m, and virial mass definitions.
    """
    def __init__(self):
        pass


    @partial(jax.jit, static_argnums=(0, 4))
    def c_delta(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="critical")):
        """
        Compute the concentration parameter.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the concentration relation.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.
        mass_definition : MassDefinition, optional
            Target halo mass definition. Defaults to
            ``MassDefinition(delta=200, reference="critical")``.

        Returns
        -------
        float or array-like
            Concentration values with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = cosmology.H0 / 100.0
        m_internal = m * h
        mdef = mass_definition

        # Parameter Lookup Table
        coeffs = {
            (200, "critical"):       (5.71, -0.084, -0.47, 2e12),
            (200, "mean"):           (10.14, -0.081, -1.01, 2e12),
            ("vir", "critical"):     (7.85, -0.081, -0.71, 2e12),
        }
        
        key = (mdef.delta, mdef.reference) 

        if key not in coeffs:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        A, B, C, M_pivot = coeffs[key]
        return jnp.squeeze(A * (m_internal[:, None] / M_pivot)**B * (1 + z[None, :])**C)





class B13Concentration(Concentration):
    """
    Concentration-mass relation from `Bhattacharya et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...766...32B/abstract>`_.

    The fitted relation is

    .. math::

        c_\\Delta(M, z) = A D(z)^B \\nu^C

    where :math:`D(z)` is the linear growth factor and
    :math:`\\nu(M, z) = \\frac{\\delta_c}{\\sigma(M, z)}`, with
    :math:`\\delta_c \\approx 1.686` and :math:`\\sigma(M, z)` the linear-theory
    variance of the density field smoothed on the mass scale :math:`M`.

    Calibrated for 200c, 200m, and virial mass definitions.
    """
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0, 4))
    def c_delta(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="critical")):
        """
        Compute the concentration parameter.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the concentration relation.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.
        mass_definition : MassDefinition, optional
            Target halo mass definition. Defaults to
            ``MassDefinition(delta=200, reference="critical")``.

        Returns
        -------
        float or array-like
            Concentration values with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        mdef = mass_definition
        
        # Parameter Lookup Table
        coeffs = {
            (200, "critical"):    (5.9, 0.54, -0.35),
            (200, "mean"):        (9.0, 1.15, -0.29),
            ("vir", "critical"):  (7.7, 0.9,  -0.29),
        }

        key = (mdef.delta, mdef.reference)
        D = jnp.atleast_1d(cosmology.growth_factor(z))

        def compute_c(masses, redshifts, A, B, C):
            masses = jnp.asarray(masses)
            redshifts = jnp.atleast_1d(redshifts)
            sigma_m = jnp.reshape(cosmology.sigma_m(masses, redshifts), (len(masses), len(redshifts)))
            delta_c = jnp.atleast_1d(cosmology.delta_c(redshifts, prescription="EdS"))[None, :]
            nu = delta_c / sigma_m
            return A * D[None, :]**B * nu**C

        if key not in coeffs:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        A, B, C = coeffs[key]
        return jnp.squeeze(compute_c(m, z, A, B, C))




class SC14Concentration(Concentration):
    """
    Concentration-mass relation from `Sanchez-Conde & Prada (2014) <https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.2271S/abstract>`_.

    The fitted relation is

    .. math::

        c_{200c}(M, z) = \\sum_{i=0}^5 a_i [\\log_{10}(M)]^i \\times (1+z)^{-1}

    where the coefficients :math:`a_i` are from Eq. 1 of the paper.

    Calibrated for 200c mass definition.
    """
    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0, 4))
    def c_delta(self, cosmology, m, z, mass_definition=MassDefinition(delta=200, reference="critical")):
        """
        Compute the concentration parameter.

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology used to evaluate the concentration relation.
        m : array-like
            Halo masses in physical :math:`M_\\odot`.
        z : array-like
            Redshifts.
        mass_definition : MassDefinition, optional
            Target halo mass definition. Defaults to
            ``MassDefinition(delta=200, reference="critical")``.

        Returns
        -------
        float or array-like
            Concentration values with shape :math:`(N_m, N_z)`, where
            singleton dimensions get squeezed before return.
        """
        
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = cosmology.H0 / 100.0
        m_internal = m * h
        mdef = mass_definition

        # 1. Parameter Table (SC14 is only natively defined for 200c)
        # Coefficients in descending order for jnp.polyval: [p5, p4, p3, p2, p1, p0]
        coeffs = {
            (200, "critical"): jnp.array([5.32e-7, -2.89237e-5, 3.66e-4, 1.636e-2, -1.5093, 37.5153])
        }

        key = (mdef.delta, mdef.reference)

        def compute_c(m_val, z_val, p_coeffs):
            # Eq 1: c(M, z=0) = sum(a_i * (log10 M)^i)
            # Then scaled by (1+z)^-1
            logM = jnp.log10(m_val)
            c_z0 = jnp.polyval(p_coeffs, logM)
            return c_z0 * (1 + z_val)**-1

        if key not in coeffs:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        return jnp.squeeze(compute_c(m_internal[:, None], z[None, :], coeffs[key]))






        