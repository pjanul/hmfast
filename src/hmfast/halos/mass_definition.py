import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial

from hmfast.utils import newton_root

class MassDefinition:
    """
    Mass definition for halos specified by an overdensity threshold and a reference density.

    For example, :math:`M_{200c}` corresponds to
    ``MassDefinition(delta=200, reference="critical")``, while
    :math:`M_{200m}` corresponds to
    ``MassDefinition(delta=200, reference="mean")``. The special value
    ``delta='vir'`` denotes the redshift-dependent virial overdensity and can
    only be used with ``reference='critical'``.

    Attributes
    ----------
    delta : int, float, or str
        Overdensity threshold used to define the halo boundary. This can be a numeric value such as ``200`` or ``500``, or the string ``'vir'`` for the redshift-dependent virial overdensity. The value ``'vir'`` is only valid with ``reference='critical'``.
    reference : str
        Reference density associated with ``delta``, either ``'critical'`` or ``'mean'``.

    Raises
    ------
    ValueError
        If an invalid combination of `delta` and `reference` is provided, or if either
        parameter is set to an unsupported value.
    """

    def __init__(self, delta=200, reference="critical"):
        self._delta = None
        self._reference = None
        self.reference = reference
        self.delta = delta
        
    # Ensure that reference is only ever critical or mean
    @property
    def reference(self):
        return self._reference
    
    @reference.setter
    def reference(self, value):
        value = str(value).lower()
        if value not in ("critical", "mean"):
            raise ValueError("reference must be either 'critical' or 'mean'")
            
        # Prevent changing reference if delta == "vir"
        if getattr(self, "_delta", None) == "vir" and value != "critical":
            raise ValueError("'vir' is only allowed with 'critical' reference")
        self._reference = value

        
    @property
    def delta(self):
        return self._delta

        
    @delta.setter
    def delta(self, value):
        if isinstance(value, str):
            value = value.lower()
            
        # If 'vir', reference must be 'critical'
        if value == "vir":
            if getattr(self, "_reference", None) != "critical":
                raise ValueError("'vir' is only allowed with 'critical' reference")
            self._delta = value
            return

        # Otherwise, it must be numeric
        if isinstance(value, (int, float)):
            self._delta = value
            return

        raise ValueError("delta must be numeric or 'vir'")


    def _tree_flatten(self):
        # delta can be a tracer (numeric) or a static string ('vir')
        # reference is always a static string for critical/mean
        children = () 
        aux_data = (self.delta, self.reference) 
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    def _delta_vir_to_crit(self, cosmology, z):
        """
        Compute the virial overdensity with respect to the critical density.
        """
        omega_m = cosmology.omega_m(z)
        x = omega_m - 1.0
        return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2

    def r_delta(self, cosmology, m, z):
        """
        Compute the halo radius :math:`r_\\Delta` associated with a halo mass.

        .. math::

            r_\\Delta = \\left[\\frac{3M}{4\\pi \\Delta \\rho_{\\mathrm{ref}}(z)}\\right]^{1/3}

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object used to evaluate the reference density.
        m : float or array-like
            Halo mass enclosed within the overdensity radius.
        z : float or array-like
            Redshift at which to compute the radius.

        Returns
        -------
        float or array-like
            Radius :math:`r_\\Delta` within which the mean enclosed density is
            :math:`\\Delta \\rho_{\\mathrm{ref}}(z)`.
        """
        delta, reference = self.delta, self.reference

        m = jnp.atleast_1d(m)[:, None]
        z = jnp.atleast_1d(z)[None, :]

        rho_ref = cosmology.critical_density(z)

        if delta == "vir":
            delta = self._delta_vir_to_crit(cosmology, z)

        if reference == "mean":
            rho_ref *= cosmology.omega_m(z)

        return (3.0 * m / (4.0 * jnp.pi * delta * rho_ref)) ** (1.0 / 3.0)


jax.tree_util.register_pytree_node(
    MassDefinition,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: MassDefinition._tree_unflatten(aux_data, children)
)


