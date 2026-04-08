import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial
from jax.tree_util import register_pytree_node_class

from hmfast.emulator import Emulator
from hmfast.utils import newton_root

@register_pytree_node_class
class MassDefinition:

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


    def tree_flatten(self):
        # delta can be a tracer (numeric) or a static string ('vir')
        # reference is always a static string for critical/mean
        children = () 
        aux_data = (self.delta, self.reference) 
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)





