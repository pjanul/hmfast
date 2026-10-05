
Custom extensions guide
=======================

This short guide points to the API pages for the parent classes that users
can subclass to provide custom ingredients (cosmology engines, tracers, profiles,
and halo-model components).

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
  from hmfast.stats import Pk, cl

  # Grids used for the example (mass, multipole; z is passed as a (min, max) range)
  m_grid = jnp.geomspace(1e10, 1e15, 105)
  l_grid = jnp.geomspace(1, 1e3, 100)
  z_range = (0.05, 2.0)
  n_z = 32

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

See the API pages for full method signatures and optional behaviors.


Cosmology engines
-----------------

A ``Cosmology`` holds the cosmological parameters; its engine computes
:math:`H(z)`, :math:`D_A(z)` and :math:`P(k, z)` from them, and ``hmfast``
derives everything else (growth, :math:`\sigma(M)`, halo model, statistics).
Two engines are built in: ``EmulatorEngine`` calls the emulators (within their
training ranges), and ``AnalyticEngine`` uses analytic formulae with the
Eisenstein & Hu (1998) transfer function and halofit (for any parameter values).
``CombinedEngine`` takes the background from one engine and :math:`P(k, z)` from another::

  from hmfast.cosmology import Cosmology, EmulatorEngine, AnalyticEngine, CombinedEngine

  cosmo_emu = Cosmology(EmulatorEngine("lcdm:v1"), H0=67.4)
  cosmo_ana = Cosmology(AnalyticEngine(), H0=110.0, omega_cdm=0.30, w0=-0.7)
  cosmo_mix = Cosmology(CombinedEngine(background=EmulatorEngine("wcdm:v1"), power=AnalyticEngine()), w0=-0.9)

The engine decides which parameters exist. ``print(engine)`` lists the ones it
takes (``engine.params``, with defaults) and the values it fixes (``engine.fixed``)::

  >>> print(EmulatorEngine("wcdm:v1"))
  EmulatorEngine('wcdm:v1', pknl_mode='hmcode')
    free : H0=68, omega_cdm=0.12, omega_b=0.0224658, A_s=2.1053e-09, n_s=0.965, tau=0.0544, w0=-1
    fixed: m_ncdm=0.06, N_ur=3.046, T_cmb=2.7255, deg_ncdm=1

Parameters are passed to ``Cosmology`` as keywords, read as attributes
(``cosmo.H0``) and changed with ``update``; setting a fixed or unknown one
raises a ``TypeError``. A ``CombinedEngine`` fixes anything either engine fixes,
so both always see the same cosmology.

For your own engine, subclass ``Engine`` (or a built-in engine). Every
``Cosmology`` method ``X`` that uses the engine calls ``engine.compute_X`` with the
same arguments plus ``p``, a dict of the parameters, fixed values and densities;
``Cosmology`` then masks values outside the engine's domain and extrapolates.
``compute_hubble_parameter(z, p)`` and ``compute_pk(k, z, p, linear=True)`` are
required; ``super().compute_pk(k, z, p, linear=False)`` is halofit on your linear
spectrum, and :math:`D_A(z)` defaults to the integral of :math:`c/H`.
``compute_sigma8``, ``compute_cl_cmb``, ``compute_derived_parameters``,
``compute_densities`` and ``in_bounds`` are optional. New parameters go in
``params``; the example below adds a running of the spectral index to the
analytic engine::

  import jax
  import jax.numpy as jnp
  from hmfast.cosmology import Cosmology, AnalyticEngine

  class RunningEngine(AnalyticEngine):
    """Analytic LCDM with a running spectral index."""
    params = {**AnalyticEngine.params, "n_run": 0.0}

    def compute_pk(self, k, z, p, linear=True):
      if not linear:
        return super().compute_pk(k, z, p, linear=False)  # halofit on the spectrum below
      running = jnp.exp(0.5 * p["n_run"] * jnp.log(k / 0.05) ** 2)
      return super().compute_pk(k, z, p) * running[:, None]

  engine = RunningEngine()
  cosmo = Cosmology(engine, H0=67.4, n_run=-0.01)
  g = jax.grad(lambda a: cosmo.update(n_run=a).sigma8(0.0))(-0.01)

To fix a parameter instead, declare it in ``fixed`` (for example
``fixed = {**AnalyticEngine.fixed, "w0": -1.0}``); a name is never in both.

Engines are static under ``jit``. Built-in engines with the same settings share
compiled code; a custom engine shares it only with itself unless it defines
``_key()`` to return the settings that change its output, so create one and reuse it.

