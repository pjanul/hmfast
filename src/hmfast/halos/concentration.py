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




class K11Concentration(Concentration):
    """
    Concentration-mass relation from `Klypin et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...740..102K/abstract>`_.

    The fitted relation is

    .. math::

        c_\\mathrm{vir}(M, z) = c_0(z)
        \\left(\\frac{M}{10^{12} \\; M_\\odot / h}\\right)^{-0.075}
        \\left[1 + \\left(\\frac{M}{M_0(z)}\\right)^{0.26}\\right]

    where :math:`c_0(z)` and :math:`M_0(z)` are interpolated from the values in
    Table 3 of the reference.

    Calibrated for the virial mass definition only.
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

        key = (mdef.delta, mdef.reference)
        if key != ("vir", "critical"):
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        z_tab = jnp.array([0.0, 0.31578947, 0.63157895, 0.94736842, 1.26315789, 1.57894737, 1.89473684, 2.21052632, 2.52631579, 2.84210526, 3.15789474, 3.47368421, 3.78947368, 4.10526316, 4.42105263, 4.73684211, 5.05263158, 5.36842105, 5.68421053, 6.0])
        c0_tab = jnp.array([9.6, 7.89848895, 6.57388797, 5.59198421, 4.82413741, 4.2543651, 3.80201899, 3.4341066, 3.15047911, 2.92643281, 2.74396076, 2.60306296, 2.50373941, 2.44412709, 2.40000661, 2.36392585, 2.33588481, 2.31588348, 2.30392188, 2.3])
        lnM0_tab = jnp.array([45.46291469, 41.51644832, 38.29554435, 35.80033605, 34.03132449, 32.96668373, 32.12518764, 31.309971, 30.52126833, 29.79801323, 29.16858499, 28.6329836, 28.19120908, 27.84158345, 27.56229323, 27.34662656, 27.19458346, 27.10616391, 27.08136792, 27.12019549])

        c0 = jnp.interp(z, z_tab, c0_tab)
        m0 = jnp.exp(jnp.interp(z, z_tab, lnM0_tab))
        return jnp.squeeze(
            c0[None, :]
            * (m_internal[:, None] / 1e12) ** (-0.075)
            * (1.0 + (m_internal[:, None] / m0[None, :]) ** 0.26)
        )






        