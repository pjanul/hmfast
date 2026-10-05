
Custom extensions guide
=======================

This short guide points to the API pages for the parent classes that users
can subclass or build to provide custom ingredients (cosmologies, tracers,
profiles, and halo-model components).

The following list shows some of the parent classes you can implement:

- **Cosmology**: computes :math:`H(z)`, :math:`D_A(z)` and :math:`P(k, z)`
  from functions you supply: :doc:`/api/cosmology`.
- **Tracer**: see the Tracer API documentation: :doc:`/api/tracers`.
- **Halo profiles**: prefer one of the profile parent classes (examples
  include MatterProfile, CIBProfile, GalaxyHODProfile, PressureProfile,
  DensityProfile): :doc:`/api/halos/profiles`.
- **Halo mass function**: :doc:`/api/halos/massfunc`.
- **Halo bias**: :doc:`/api/halos/bias`.
- **Concentration relations**: :doc:`/api/halos/concentration`.
- **Subhalo mass function**: :doc:`/api/halos/massfunc` (see subhalo classes).

For JAX `jit`/autodiff compatibility implement your classes as JAX pytrees
so JAX can traverse array children while treating configuration as static
(``Cosmology`` is the exception; see `Pytrees & differentiability`_).

A ``Cosmology`` takes :math:`H(z)` and :math:`P(k, z)` from functions, and
everything else in ``hmfast`` (distances, growth, :math:`\sigma(M)`, the halo
model, statistics) is built on them. ``hmfast`` provides two ready-made
cosmologies, ``CosmoPowerCosmology`` (emulators) and ``AnalyticCosmology``
(analytic formulae); to test a model ``hmfast`` does not support, pass your own
functions to ``Cosmology`` itself.

Each function takes ``p``, the dict of the cosmology's parameter values, as its
last argument and must be JAX-traceable; :class:`~hmfast.cosmology.Cosmology`
lists the functions, their signatures and which are optional. ``p`` holds the
standard parameters (``H0``, ``omega_b``, ``omega_cdm``, ...), which also set the
densities, plus any further keyword passed to ``Cosmology``. The functions are
traced on construction, so a parameter name they read that does not exist raises
a ``TypeError`` straight away. The example below builds one from scratch.

For full API details and method signatures consult the linked API pages above.

Example
-------

Minimal working example showing how to supply toy halo-model ingredients.
Not physical — only intended as a tiny runnable example users can adapt::

  import jax.numpy as jnp
  from jax.tree_util import register_pytree_node_class
  from hmfast.cosmology import Cosmology
  from hmfast.halos import HaloModel
  from hmfast.halos.massfunc import HaloMassFunction, SubHaloMassFunction
  from hmfast.halos.bias import HaloBias
  from hmfast.halos.concentration import Concentration
  from hmfast.halos.profiles.matter import MatterProfile
  from hmfast.tracers.base_tracer import Tracer
  from hmfast.stats import Pk, cl

  # Grids used for the example (mass, multipole; z is passed as a (min, max) range)
  m_grid = jnp.geomspace(1e10, 1e15, 105)
  l_grid = jnp.geomspace(1, 1e3, 100)
  z_range = (0.05, 2.0)
  n_z = 32

  # --- Toy cosmology: flat LCDM expansion and a toy linear P(k) ---

  def hubble(z, p):
    om = (p["omega_b"] + p["omega_cdm"]) / (p["H0"] / 100) ** 2
    return p["H0"] * jnp.sqrt(om * (1 + z) ** 3 + 1 - om)

  def pk_linear(k, z, p):
    return p["amp"] * jnp.outer(1e4 * k / (1 + (k / 0.02) ** 3), 1 / (1 + z) ** 2)

  cosmo = Cosmology(hubble, pk_linear, amp=1.0)  # amp is a new parameter; the standard ones take their defaults

  # --- Toy implementations of halo-model building blocks ---
  #
  # Each is registered as a (trivial, stateless) JAX pytree so it can be passed
  # into a jitted function such as cl; see "Pytrees & differentiability"
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
      # one (W, der_bessel, der_angles) term: density-type (j_l, no angular prefactor), weight of 1 at every z
      return [(jnp.ones_like(z), 0, 0)]
    def tree_flatten(self):
      return (self.profile,), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
      (profile,) = children
      return cls(profile=profile)

  # --- Instantiate toy ingredients and run a halo-model call ---

  tracer1 = NewTracer(profile=NewMatterProfile())

  hm = HaloModel(
    cosmology=cosmo,
    halo_mass_function=NewHaloMassFunction(),
    halo_bias=NewHaloBias(),
    subhalo_mass_function=NewSubHaloMassFunction(),
    concentration=NewConcentration(),
  )

  pk_calc = Pk()

  # Compute a tiny toy halo-model cl (1-halo + 2-halo); with no second tracer this is the autocorrelation of tracer1.
  cl_toy = cl(pk_calc, hm, l_grid, tracer1, z_range=z_range, n_z=n_z)

  print("cl shape:", cl_toy.shape)   # should be (N_ell,)
  print("cl (toy values):", cl_toy)


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

A ``Cosmology`` needs no registration: its functions are static and its parameters
are its leaves, so a gradient goes through ``update``::

  g = jax.grad(lambda a: cosmo.update(amp=a).pk(0.1, 1.0))(1.0)

Functions are compared by identity under ``jit``, so define them once and reuse
them; building a ``Cosmology`` from new function objects compiles anew.

See the API pages for full method signatures and optional behaviors.

