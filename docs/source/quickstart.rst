Quickstart
==========

``hmfast`` is organized around three main pieces:

- ``Cosmology`` provides emulator-backed background and power-spectrum quantities.
- ``HaloModel`` combines the cosmology with halo ingredients such as the mass function, bias, concentration, and mass definition.
- ``Tracer`` classes pair a kernel with a halo profile so you can build projected observables such as :math:`C_\ell`.

Two conventions are worth keeping in mind from the start:

- Public inputs and outputs use physical units such as :math:`\mathrm{Mpc}`, :math:`M_\odot`, and :math:`\mathrm{Mpc}^{-1}`, not :math:`h^{-1}\mathrm{Mpc}` or :math:`M_\odot/h`.
- When a class exposes an ``update()`` method, use it to create modified copies instead of mutating attributes in place. This avoids unnecessary JIT recompilation and gives massive speedups. 

Minimal example
---------------

The snippet below shows three core tasks: reading the Hubble parameter, evaluating the Tinker et al. (2008) halo mass function, and computing a halo-model angular power spectrum.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from hmfast.cosmology import Cosmology
   from hmfast.halos import HaloModel
   from hmfast.halos.massdef import MassDefinition
   from hmfast.halos.massfunc import T08HaloMassFunction
   from hmfast.tracers import CMBLensingTracer, tSZTracer

   cosmo = Cosmology(emulator_set="lcdm:v1")
   cosmo = cosmo.update(H0=67.4)
   m_200c = MassDefinition(200, "critical")
   hmf_t08 = T08HaloMassFunction()
   halo_model = HaloModel(cosmology=cosmo, mass_definition=m_200c, halo_mass_function=hmf_t08)

   z_grid = jnp.linspace(0.05, 3.0, 64)
   m_grid = jnp.geomspace(1e12, 1e15, 64)
   l_grid = jnp.arange(100, 1100, 100)

   # Compute hubble parameter H(z)
   hubble = cosmo.hubble_parameter(z_grid)

   # Compute HMF dn/dlnM
   dndlnm = hmf_t08.dndlnm(cosmo, m_grid, 0.5, mass_definition=m_200c)

   y = tSZTracer()
   kappa_cmb = CMBLensingTracer()

   # Compute tSZ x CMB lensing cross-correlation
   cl_yk = halo_model.cl_1h(y, kappa_cmb, l_grid, m_grid, z_grid)
   cl_yk += halo_model.cl_2h(y, kappa_cmb, l_grid, m_grid, z_grid)

   # Example gradients with respect to H0
   grad_hubble = jax.grad(lambda H0: jnp.sum(cosmo.update(H0=H0).hubble_parameter(z_grid)))(67.4)
   grad_dndlnm = jax.grad(lambda H0: jnp.sum(hmf_t08.dndlnm(cosmo.update(H0=H0), m_grid, 0.5, mass_definition=m_200c)))(67.4)
   grad_cl_yk = jax.grad(lambda H0: jnp.sum(HaloModel(cosmology=cosmo.update(H0=H0), mass_definition=m_200c, halo_mass_function=hmf_t08).cl_1h(y, kappa_cmb, l_grid, m_grid, z_grid) + HaloModel(cosmology=cosmo.update(H0=H0), mass_definition=m_200c, halo_mass_function=hmf_t08).cl_2h(y, kappa_cmb, l_grid, m_grid, z_grid)))(67.4)

``hubble`` has units of :math:`\mathrm{km} \, \mathrm{s}^{-1} \, \mathrm{Mpc}^{-1}`, ``dndlnm`` is evaluated for physical halo masses in :math:`M_\odot`, and ``cl_yk`` is the tSZ-CMB lensing cross-spectrum built from the specified halo-model ingredients.