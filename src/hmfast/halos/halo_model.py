"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Callable
from functools import partial

from hmfast.halos.massfunc import T08HaloMassFunction, TW10SubHaloMassFunction
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import D08Concentration, B13Concentration
from hmfast.halos.massdef import MassDefinition
from hmfast.cosmology import Cosmology
from hmfast.utils import gauss_legendre_nodes_weights

jax.config.update("jax_enable_x64", True)


class HaloModel:
    """
    Differentiable halo model.

    Provides halo-model predictions for arbitrary tracers using a configurable
    cosmology, halo mass function, halo bias model, concentration relation,
    and subhalo mass function.

    Attributes
    ----------
    cosmology : Cosmology
        Cosmology object supplying background, growth, and matter power spectra quantities.
        Defaults to ``Cosmology(emulator_set="lcdm:v1")``.
    mass_def : MassDefinition
        Native spherical-overdensity mass definition used throughout the halo model.
        Defaults to 200c, ``MassDefinition(delta=200, reference="critical")``.
    halo_mass_function : HaloMassFunction
        Halo mass function model used to compute :math:`dn / d\\ln M`.
        Defaults to ``T08HaloMassFunction()``.
    halo_bias : HaloBias
        Halo bias model used for large-scale halo bias predictions.
        Defaults to ``T10HaloBias()``.
    subhalo_mass_function : SubHaloMassFunction
        Subhalo mass function model used in observables with satellite or subhalo contributions.
        Defaults to ``TW10SubHaloMassFunction()``.
    concentration : Concentration
        Halo concentration relation used to map halo mass and redshift to concentration.
        Defaults to ``D08Concentration()``.
    hm_consistency : bool
        Flag controlling whether halo-model consistency counterterms are applied.
        Defaults to ``True``.
    m_range : tuple
        ``(m_min, m_max)`` in :math:`M_\\odot` spanning all mass integrals.
        Defaults to ``(1e10, 1e15)``.
    n_m : int
        Number of Gauss-Legendre mass-integral nodes (static: changing it
        triggers recompilation; sweeping ``m_range`` alone does not).
        Defaults to ``100``.
    """

    def __init__(self,
                 cosmology=None,
                 mass_def=None,
                 halo_mass_function=None,
                 halo_bias=None,
                 subhalo_mass_function=None,
                 concentration=None,
                 hm_consistency=True,
                 m_range=(1e10, 1e15), n_m=100):
        """Initialize the halo model."""

        # None-sentinel: avoids building these (some expensive, some cache-holding) at import time.
        cosmology = cosmology if cosmology is not None else Cosmology(emulator_set="lcdm:v1")
        mass_def = mass_def if mass_def is not None else MassDefinition(delta=200, reference="critical")
        halo_mass_function = halo_mass_function if halo_mass_function is not None else T08HaloMassFunction()
        halo_bias = halo_bias if halo_bias is not None else T10HaloBias()
        subhalo_mass_function = subhalo_mass_function if subhalo_mass_function is not None else TW10SubHaloMassFunction()
        concentration = concentration if concentration is not None else D08Concentration()

        # Load cosmology and make sure the required files are loaded outside of jitted functions (note that DER is needed for CMB lensing tracers)
        self.cosmology = cosmology
        self.cosmology._load_emulator("DAZ")
        self.cosmology._load_emulator("HZ")
        self.cosmology._load_emulator("PKL")
        self.cosmology._load_emulator("DER")

        self.halo_mass_function = halo_mass_function
        self.halo_bias = halo_bias
        self.subhalo_mass_function = subhalo_mass_function
        self.concentration = concentration

        self.mass_def = mass_def
        self.hm_consistency = hm_consistency
        self.m_range = (jnp.asarray(m_range[0]), jnp.asarray(m_range[1]))
        self.n_m = int(n_m)


    def _tree_flatten(self):
        # Cosmology and m_range are JAX arrays / pytrees — children (dynamic: sweeping
        # them does not retrace). Everything else, including n_m, is static aux_data.
        children = (self.cosmology, self.m_range)
        aux_data = (self.halo_mass_function, self.halo_bias, self.subhalo_mass_function, self.concentration,
            self.mass_def, self.hm_consistency, self.n_m
        )
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        cosmology, m_range = children
        obj = cls.__new__(cls)
        obj.cosmology = cosmology
        obj.m_range = m_range
        (obj.halo_mass_function, obj.halo_bias, obj.subhalo_mass_function,
         obj.concentration, obj.mass_def, obj.hm_consistency, obj.n_m) = aux_data
        return obj

    def update(self, cosmology=None, halo_mass_function=None, halo_bias=None, subhalo_mass_function=None, concentration=None, mass_def=None,
               hm_consistency=None, m_range=None, n_m=None):
        """
        Return a new HaloModel instance with updated components.

        Parameters
        ----------
        cosmology, halo_mass_function, halo_bias, subhalo_mass_function, concentration, mass_def, hm_consistency, m_range, n_m : optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

        Returns
        -------
        HaloModel
            New halo-model instance with updated attributes.
        """
        # Flatten current state
        children, aux_data = self._tree_flatten()
        # Unpack
        cosmo_child, m_range0 = children
        (
            halo_mass_function0, halo_bias0, subhalo_mass_function0, concentration0,
            mass_def0, hm_consistency0, n_m0
        ) = aux_data

        # Update only provided components
        new_cosmo = cosmology if cosmology is not None else cosmo_child
        new_m_range = (jnp.asarray(m_range[0]), jnp.asarray(m_range[1])) if m_range is not None else m_range0
        new_n_m = int(n_m) if n_m is not None else n_m0
        new_halo_mass_function = halo_mass_function if halo_mass_function is not None else halo_mass_function0
        new_halo_bias = halo_bias if halo_bias is not None else halo_bias0
        new_subhalo_mass_function = subhalo_mass_function if subhalo_mass_function is not None else subhalo_mass_function0
        new_concentration = concentration if concentration is not None else concentration0
        new_mass_def = mass_def if mass_def is not None else mass_def0
        new_hm_consistency = hm_consistency if hm_consistency is not None else hm_consistency0

        new_aux_data = (
            new_halo_mass_function, new_halo_bias, new_subhalo_mass_function, new_concentration,
            new_mass_def, new_hm_consistency, new_n_m
        )
        # Use _tree_unflatten to create the new instance efficiently
        return self._tree_unflatten(new_aux_data, (new_cosmo, new_m_range))
       
    @jax.jit
    def _counter_terms(self, z):
        """
        Compute :math:`n_{\\min}`, :math:`b_{1,\\min}`, and :math:`b_{2,\\min}` counter terms for halo model consistency.

        Parameters
        ----------
        z : array-like
            Redshift(s).

        Returns
        -------
        n_min : array
            Minimum number density.
        b1_min : array
            Minimum linear bias.
        b2_min : array
            Minimum quadratic bias.
        """

        logm, gl_w = gauss_legendre_nodes_weights(jnp.log(self.m_range[0]), jnp.log(self.m_range[1]), self.n_m)
        m = jnp.exp(logm)
        z = jnp.atleast_1d(z)
        cparams = self.cosmology._cosmo_params()
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"]
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # (Nm, 1)
    
    
        # Public HMF and bias interfaces use physical masses.
        dn_dlnm = jnp.reshape(self.halo_mass_function.dndlnm(self.cosmology, m, z, self.mass_def), (len(m), len(z)))
        b1 = jnp.reshape(self.halo_bias.bias(self.cosmology, m, z, self.mass_def, order=1), (len(m), len(z)))
        b2 = jnp.reshape(self.halo_bias.bias(self.cosmology, m, z, self.mass_def, order=2), (len(m), len(z)))
    
        # Compute integrals I0, I1, I2 via Gauss-Legendre quadrature in ln(M)
        I0 = jnp.sum(dn_dlnm * m_over_rho_mean * gl_w[:, None], axis=0)  # (Nz,)
        I1 = jnp.sum(b1 * dn_dlnm * m_over_rho_mean * gl_w[:, None], axis=0)
        I2 = jnp.sum(b2 * dn_dlnm * m_over_rho_mean * gl_w[:, None], axis=0)
    
        # Apply formulas
        m_min =  m[0]
        n_min =  (1.0 - I0) * rho_mean_0 / m_min
        b1_min = (1.0 - I1) * rho_mean_0 / m_min / n_min
        b2_min = -I2 * rho_mean_0 / m_min / n_min
    
        return n_min, b1_min, b2_min


    @partial(jax.jit, static_argnums=(4,))
    def _I(self, profile, k, z, bias_order=1):
        """
        Generalised halo model mass integral.

        .. math::

            I^\\beta(k, z) = \\int \\frac{dn}{d\\ln M}\\, b_\\beta(M, z)\\,
            u(k \\mid M, z)\\, d\\ln M

        where :math:`b_\\beta` is the :math:`\\beta`-th order bias
        (:math:`b_0 = 1`, :math:`b_1` linear bias, :math:`b_2` quadratic bias)
        and :math:`u(k \\mid M, z)` is the Fourier-space tracer profile.

        This integral is the fundamental building block for the 2h power spectrum
        (``bias_order=1``) and all three bispectrum terms.

        The halo-model consistency counterterm is included:
        a point mass at the minimum grid mass contributes ``n_min * b_beta_min * u(k, m_min)``.

        Parameters
        ----------
        profile : HaloProfile
            Halo profile.
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        bias_order : int, default 1
            Bias order ``beta``. Accepted values: ``0`` (unweighted), ``1`` (linear bias),
            ``2`` (quadratic bias).

        Returns
        -------
        array
            Integral with shape :math:`(N_k, N_z)`, where singleton dimensions are squeezed.
        """
        k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
        logm, w = gauss_legendre_nodes_weights(jnp.log(self.m_range[0]), jnp.log(self.m_range[1]), self.n_m)
        m = jnp.exp(logm)

        dndlnm = jnp.reshape(
            self.halo_mass_function.dndlnm(self.cosmology, m, z, self.mass_def),
            (len(m), len(z)),
        )

        if bias_order == 0:
            bias_w = jnp.ones((len(m), len(z)))
        elif bias_order == 1:
            bias_w = jnp.reshape(
                self.halo_bias.bias(self.cosmology, m, z, self.mass_def, order=1),
                (len(m), len(z)),
            )
        elif bias_order == 2:
            bias_w = jnp.reshape(
                self.halo_bias.bias(self.cosmology, m, z, self.mass_def, order=2),
                (len(m), len(z)),
            )

        total_weights = dndlnm * bias_w * w[:, None]  # (Nm, Nz)

        uk = jnp.reshape(profile.fourier(self, k, m, z), (len(k), len(m), len(z)))  # (Nk, Nm, Nz)
        integral = jnp.sum(uk * total_weights[None, :, :], axis=1)  # (Nk, Nz)

        u_k_min = uk[:, 0, :]  # profile at m_grid[0] (Nk, Nz)
        n_min, b1_min, b2_min = self._counter_terms(z)

        if bias_order == 0:
            correction = n_min[None, :] * u_k_min
        elif bias_order == 1:
            correction = n_min[None, :] * b1_min[None, :] * u_k_min
        elif bias_order == 2:
            correction = n_min[None, :] * b2_min[None, :] * u_k_min

        return jnp.squeeze(integral + self.hm_consistency * correction)


jax.tree_util.register_pytree_node(
    HaloModel,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: HaloModel._tree_unflatten(aux_data, children)
)
