Custom extensions guide
=======================

This guide explains how to add custom extension points to HMFast: tracers, halo-model
ingredients (mass functions, bias, concentration relations), and halo profiles.

The reference for each parent class is included below.

Tracer
------

A user-defined tracer should inherit from
``hmfast.tracers.Tracer``. At minimum it must implement the
``kernel(self, cosmology, z)`` method and provide a compatible ``profile``
attribute (an instance of a subclass of ``HaloProfile``) when used in the
halo model.

.. autoclass:: hmfast.tracers.Tracer
   :members:
   :undoc-members:
   :inherited-members:

Profiles
--------

All halo profiles inherit from ``hmfast.halos.profiles.HaloProfile``.
Child classes must implement ``real(self, halo_model, r, m, z)`` and
``fourier(self, halo_model, k, m, z)``. Use the helper methods on the base
class for common transforms (Hankel, NFW helpers, etc.).

.. autoclass:: hmfast.halos.profiles.HaloProfile
   :members:
   :undoc-members:
   :inherited-members:

Halo-model ingredients
----------------------

The halo model is composed from several interchangeable components. The main
parent classes are documented below; implementations must provide the methods
declared in their base classes.

Concentration relations

.. autoclass:: hmfast.halos.concentration.Concentration
   :members:
   :undoc-members:
   :inherited-members:

Halo mass function

.. autoclass:: hmfast.halos.massfunc.HaloMassFunction
   :members:
   :undoc-members:
   :inherited-members:

Halo bias

.. autoclass:: hmfast.halos.bias.HaloBias
   :members:
   :undoc-members:
   :inherited-members:

Subhalo mass function

.. autoclass:: hmfast.halos.massfunc.SubHaloMassFunction
   :members:
   :undoc-members:
   :inherited-members:


JIT / autodiff compatibility (Pytrees)
------------------------------------

To take full advantage of JAX `jit` and `grad` for user-defined components,
implement your classes as JAX pytrees. A pytree lets JAX traverse a Python
object's arrays as part of the computation graph while treating configuration
metadata as static Python values.

