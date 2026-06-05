
Custom extensions guide
=======================

This short guide points to the API pages for the parent classes that users
can subclass to provide custom ingredients (tracers, profiles, and halo-model
components).

The following list shows some of the parent classes you can implement:

- **Tracer**: see the Tracer API documentation: :doc:`api/tracers`.
- **Halo profiles**: prefer one of the profile parent classes (examples
  include MatterProfile, CIBProfile, GalaxyHODProfile, PressureProfile,
  DensityProfile): :doc:`api/halos/profiles`.
- **Halo mass function**: :doc:`api/halos/massfunc`.
- **Halo bias**: :doc:`api/halos/bias`.
- **Concentration relations**: :doc:`api/halos/concentration`.
- **Subhalo mass function**: :doc:`api/halos/massfunc` (see subhalo classes).

For JAX `jit`/autodiff compatibility implement your classes as JAX pytrees
so JAX can traverse array children while treating configuration as static.

For full API details and method signatures consult the linked API pages above.

Example
-------

Minimal working example showing how to supply toy halo-model ingredients.
Not physical — only intended as a tiny runnable example users can adapt::

  import jax.numpy as jnp
  from hmfast.halos import HaloModel
  from hmfast.halos.massfunc import HaloMassFunction, SubHaloMassFunction
  from hmfast.halos.bias import HaloBias
  from hmfast.halos.concentration import Concentration
  from hmfast.halos.profiles.matter import MatterProfile
  from hmfast.tracers.base_tracer import Tracer

  # Grids used for the example (mass, redshift, multipole)
  m_grid = jnp.geomspace(1e10, 1e15, 105)
  z_grid = jnp.geomspace(0.05, 2, 95)
  l_grid = jnp.geomspace(1, 1e3, 100)

  # --- Toy implementations of halo-model building blocks ---

  class NewHaloMassFunction(HaloMassFunction):
    """Toy halo mass function: returns ones on (Nm, Nz) grid."""
    def dndlnm(self, cosmology, m, z, mass_definition=None, convert_masses=False):
      m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.ones((len(m), len(z)))

  class NewSubHaloMassFunction(SubHaloMassFunction):
    """Toy subhalo mass function: shape matches m_sub input."""
    def dndlnmu(self, cosmology, m_host, m_sub):
      return jnp.ones_like(m_sub)

  class NewHaloBias(HaloBias):
    """Toy halo bias: returns ones on (Nm, Nz) grid (supports order arg)."""
    def halo_bias(self, cosmology, m, z, mass_definition=None, convert_masses=False, order=1):
      m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.ones((len(m), len(z)))

  class NewConcentration(Concentration):
    """Toy concentration: constant ones on (Nm, Nz)."""
    def c_delta(self, cosmology, m, z, mass_definition=None):
      m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
      return jnp.ones((len(m), len(z)))

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

  class NewTracer(Tracer):
    """Simple tracer carrying a profile and a trivial kernel."""
    def __init__(self, profile):
      super().__init__(profile=profile)
    def kernel(self, cosmology, z):
      # trivial kernel (ones) matching z shape
      return jnp.ones_like(z)

  # --- Instantiate toy ingredients and run a halo-model call ---

  tracer1 = NewTracer(profile=NewMatterProfile())

  hm = HaloModel(
    halo_mass_function=NewHaloMassFunction(),
    halo_bias=NewHaloBias(),
    subhalo_mass_function=NewSubHaloMassFunction(),
    concentration=NewConcentration(),
  )

  # Compute a tiny toy 1-halo + 2-halo cl. Second tracer None => autocorrelation of tracer1.
  cl = hm.cl_1h(tracer1, None, l_grid, m_grid, z_grid) + hm.cl_2h(tracer1, None, l_grid, m_grid, z_grid)

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

    def dndlnm(self, cosmology, m_in, z_in, mass_definition=None, convert_masses=False):
      m_in, z_in = jnp.atleast_1d(m_in), jnp.atleast_1d(z_in)
      return jnp.broadcast_to(self.amplitude, (len(m_in), len(z_in)))

  # single-line gradient of the sum of the HMF w.r.t. amplitude at amplitude=0.5
  g = jax.grad(lambda a: jnp.sum(NewHaloMassFunction(a).dndlnm(None, m, z)))(0.5)
  print(g)

See the API pages for full method signatures and optional behaviors.

