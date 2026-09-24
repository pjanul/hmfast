
Custom extensions guide
=======================

This short guide points to the API pages for the parent classes that users
can subclass to provide custom ingredients (tracers, profiles, and halo-model
components).

The following list shows some of the parent classes you can implement:

- **Tracer**: see the Tracer API documentation: :doc:`/api/tracers`.
- **Halo profiles**: prefer one of the profile parent classes (examples
  include MatterProfile, CIBProfile, GalaxyHODProfile, PressureProfile,
  DensityProfile): :doc:`/api/halos/profiles`.
- **Halo mass function**: :doc:`/api/halos/massfunc`.
- **Halo bias**: :doc:`/api/halos/bias`.
- **Concentration relations**: :doc:`/api/halos/concentration`.
- **Subhalo mass function**: :doc:`/api/halos/massfunc` (see subhalo classes).

For JAX `jit`/autodiff compatibility implement your classes as JAX pytrees
so JAX can traverse array children while treating configuration as static.

For full API details and method signatures consult the linked API pages above.

Example
-------

Minimal working example showing how to supply toy halo-model ingredients.
Not physical — only intended as a tiny runnable example users can adapt::

  import jax.numpy as jnp
  from jax.tree_util import register_pytree_node_class
  from hmfast.halos import HaloModel
  from hmfast.halos.massfunc import HaloMassFunction, SubHaloMassFunction
  from hmfast.halos.bias import HaloBias
  from hmfast.halos.concentration import Concentration
  from hmfast.halos.profiles.matter import MatterProfile
  from hmfast.tracers.base_tracer import Tracer
  from hmfast.stats import Pk, cl_hm

  # Grids used for the example (mass, multipole; z is passed as a (min, max) range)
  m_grid = jnp.geomspace(1e10, 1e15, 105)
  l_grid = jnp.geomspace(1, 1e3, 100)
  z_range = (0.05, 2.0)
  n_z = 32

  # --- Toy implementations of halo-model building blocks ---
  #
  # Each is registered as a (trivial, stateless) JAX pytree so it can be passed
  # into a jitted function such as cl_hm; see "Pytrees & differentiability"
  # below for a version that carries a differentiable parameter.

  @register_pytree_node_class
  class NewHaloMassFunction(HaloMassFunction):
    """Toy halo mass function: returns ones on (Nm, Nz) grid."""
    def dndlnm(self, cosmology, m, z, mass_def=None):
      m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.ones((len(m), len(z)))
    def tree_flatten(self):
      return (), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls()

  @register_pytree_node_class
  class NewSubHaloMassFunction(SubHaloMassFunction):
    """Toy subhalo mass function: shape matches m_sub input."""
    def dndlnmu(self, cosmology, m_host, m_sub):
      return jnp.ones_like(m_sub)
    def tree_flatten(self):
      return (), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls()

  @register_pytree_node_class
  class NewHaloBias(HaloBias):
    """Toy halo bias: returns ones on (Nm, Nz) grid (supports order arg)."""
    def bias(self, cosmology, m, z, mass_def=None, order=1):
      m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.ones((len(m), len(z)))
    def tree_flatten(self):
      return (), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls()

  @register_pytree_node_class
  class NewConcentration(Concentration):
    """Toy concentration: constant ones on (Nm, Nz)."""
    def c_delta(self, cosmology, m, z, mass_def=None):
      m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.ones((len(m), len(z)))
    def tree_flatten(self):
      return (), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls()

  @register_pytree_node_class
  class NewMatterProfile(MatterProfile):
    """Toy matter profile: minimal broadcasting implementations. Note that we arbitrarily select a MatterProfile for this example, but it could be any type of profile.

    - `real`: returns an array shaped (Nr, Nm, Nz) (here we emulate a mass-dependent field).
    - `fourier`: returns an array shaped (Nk, Nm, Nz) by broadcasting k and m.
    """
    def real(self, halo_model, r, m, z):
      r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.squeeze(jnp.broadcast_to(1.0, (len(r), len(m), len(z))))

    def fourier(self, halo_model, k, m, z):
      k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.squeeze(jnp.broadcast_to(1.0, (len(k), len(m), len(z))))

    def tree_flatten(self):
      return (), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      return cls()

  @register_pytree_node_class
  class NewTracer(Tracer):
    """Simple tracer carrying a profile and a trivial kernel."""
    def __init__(self, profile):
      super().__init__(profile=profile)
    def kernel(self, cosmology, z):
      # trivial der_bessel=0 (density-type) kernel term, weight of 1 at every z
      return [(jnp.ones_like(z), 0)]
    def tree_flatten(self):
      return (self.profile,), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      (profile,) = children
      return cls(profile=profile)

  # --- Instantiate toy ingredients and run a halo-model call ---

  tracer1 = NewTracer(profile=NewMatterProfile())

  hm = HaloModel(
    halo_mass_function=NewHaloMassFunction(),
    halo_bias=NewHaloBias(),
    subhalo_mass_function=NewSubHaloMassFunction(),
    concentration=NewConcentration(),
  )

  pk_calc = Pk()

  # Compute a tiny toy halo-model cl (1-halo + 2-halo). Second tracer None => autocorrelation of tracer1.
  cl = cl_hm(pk_calc, hm, tracer1, None, l_grid, z_range, n_z)

  print("cl shape:", cl.shape)   # should be (N_ell,)
  print("cl (toy values):", cl)


Pytrees & differentiability
---------------------------

To make user-supplied classes compatible with JAX `jit` and `grad`, register
them as pytrees so JAX can traverse numeric children while treating
configuration as static. The snippet below shows a minimal `NewHaloMassFunction`
whose scalar `amplitude` is a differentiable parameter (we compute a simple
gradient immediately after the class).

::

  import jax
  import jax.numpy as jnp
  from jax.tree_util import register_pytree_node_class
  from hmfast.halos.massfunc import HaloMassFunction

  # small grids used for the test
  m = jnp.geomspace(1e10, 1e12, 5)
  z = jnp.geomspace(0.1, 1.0, 4)

  @register_pytree_node_class
  class NewHaloMassFunction(HaloMassFunction):
    def __init__(self, amplitude):
      self.amplitude = jnp.array(amplitude)

    def tree_flatten(self):
      return ((self.amplitude,), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
      (amplitude,) = children
      return cls(amplitude)

    def dndlnm(self, cosmology, m_in, z_in, mass_def=None):
      m_in, z_in = jnp.atleast_1d(m_in), jnp.atleast_1d(z_in)
      return jnp.broadcast_to(self.amplitude, (len(m_in), len(z_in)))

  # single-line gradient of the sum of the HMF w.r.t. amplitude at amplitude=0.5
  g = jax.grad(lambda a: jnp.sum(NewHaloMassFunction(a).dndlnm(None, m, z)))(0.5)
  print(g)

See the API pages for full method signatures and optional behaviors.

