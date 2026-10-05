
Custom extensions guide
=======================

This short guide points to the API pages for the parent classes that users
can subclass or build to provide custom ingredients (cosmology engines, tracers,
profiles, and halo-model components).

The following list shows some of the parent classes you can implement:

- **Cosmology engine**: computes :math:`H(z)`, :math:`D_A(z)` and :math:`P(k, z)`
  for a ``Cosmology``: :doc:`/api/cosmology`.
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
(engines are the exception; see `Pytrees & differentiability`_).

A ``Cosmology`` takes :math:`H(z)`, :math:`D_A(z)` and :math:`P(k, z)` from its
engine, and everything else in ``hmfast`` (growth, :math:`\sigma(M)`, the halo
model, statistics) is built on them. An engine is just a set of functions, so you
can define your own cosmology if you want. ``hmfast`` already provides working
engines for the cosmopower emulators (``EmulatorEngine``) and for an analytic
cosmology (``AnalyticEngine``), but to test a model ``hmfast`` does not support
you can swap any of their functions with ``Engine.replace``, for example
interchanging the linear and nonlinear spectra::

  from hmfast.cosmology import Cosmology, EmulatorEngine, AnalyticEngine

  emu = EmulatorEngine("lcdm:v1")
  linear_only = emu.replace(pk_nonlinear=emu.pk_linear)  # emulated PKL wherever PKNL is used
  mixed = emu.replace(pk_linear=AnalyticEngine().pk_linear,  # analytic P(k) on the emulated background
                      params=(*emu.params, "T_cmb", "w0"))  # parameters the analytic P(k) also reads
  cosmo = Cosmology(linear_only, H0=67.4)

``engine.params`` names the parameters an engine reads. ``Cosmology`` supplies
the defaults of those it knows (``H0``, ``omega_b``, ...), and any new one must
be passed to it. Each function takes ``p``, a dict of the parameter values, and
must be JAX-traceable; :class:`~hmfast.cosmology.Engine` lists the functions,
their signatures and which are optional. The example below builds an engine
from scratch.

For full API details and method signatures consult the linked API pages above.

Example
-------

Minimal working example showing how to supply toy halo-model ingredients.
Not physical — only intended as a tiny runnable example users can adapt::

  import jax.numpy as jnp
  from jax.tree_util import register_pytree_node_class
  from hmfast.cosmology import Cosmology, Engine
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

  # --- Toy cosmology engine: flat LCDM expansion and a toy linear P(k) ---

  engine = Engine(
    params=("H0", "Omega_m"),
    hubble_parameter=lambda z, p: p["H0"] * jnp.sqrt(p["Omega_m"] * (1 + z) ** 3 + 1 - p["Omega_m"]),
    pk_linear=lambda k, z, p: jnp.outer(1e4 * k / (1 + (k / 0.02) ** 3), 1 / (1 + z) ** 2),
    densities=lambda p: {"Omega0_m": p["Omega_m"], "Omega0_cb": p["Omega_m"], "Omega0_b": 0.05},
  )
  cosmo = Cosmology(engine, Omega_m=0.31)  # H0 takes Cosmology's default; Omega_m has none, so it is required

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

Engines need no registration: their functions are static and their parameters
are the leaves of the ``Cosmology``, so a gradient goes through ``update``::

  g = jax.grad(lambda om: cosmo.update(Omega_m=om).hubble_parameter(1.0))(0.31)

A new engine compiles anew under ``jit``, so create it once and reuse it.

See the API pages for full method signatures and optional behaviors.

