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
from hmfast.halos.profiles.profiles_2pt import _fourier_2pt

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
    mass_def : MassDefinition
        Native spherical-overdensity mass definition used throughout the halo model.
    halo_mass_function : HaloMassFunction
        Halo mass function model used to compute :math:`dn / d\\ln M`.
    halo_bias : HaloBias
        Halo bias model used for large-scale halo bias predictions.
    subhalo_mass_function : SubHaloMassFunction
        Subhalo mass function model used in observables with satellite or subhalo contributions.
    concentration : Concentration
        Halo concentration relation used to map halo mass and redshift to concentration.
    hm_consistency : bool
        Flag controlling whether halo-model consistency counterterms are applied.
    convert_masses : bool
        Flag controlling whether profile-specific native mass definitions are converted automatically.
    m_grid : array
        Log-spaced halo mass grid in :math:`M_\\odot` used for all mass integrals.
    """

    def __init__(self,
                 cosmology=Cosmology(emulator_set="lcdm:v1"),
                 mass_def=MassDefinition(delta=200, reference="critical"),
                 halo_mass_function=T08HaloMassFunction(),
                 halo_bias=T10HaloBias(),
                 subhalo_mass_function=TW10SubHaloMassFunction(),
                 concentration=D08Concentration(),
                 hm_consistency=True,
                 convert_masses=False,
                 m_grid=None):
        """Initialize the halo model."""

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
        self.convert_masses = convert_masses
        self.m_grid = jnp.sort(m_grid if m_grid is not None else jnp.geomspace(1e10, 1e15, 100))


    def _tree_flatten(self):
        # Cosmology and m_grid are JAX arrays / pytrees — children.
        # Everything else is configuration/metadata — aux_data.
        children = (self.cosmology, self.m_grid)
        aux_data = (self.halo_mass_function, self.halo_bias, self.subhalo_mass_function, self.concentration,
            self.mass_def, self.hm_consistency, self.convert_masses
        )
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        cosmology, m_grid = children
        obj = cls.__new__(cls)
        obj.cosmology = cosmology
        obj.m_grid = m_grid
        (obj.halo_mass_function, obj.halo_bias, obj.subhalo_mass_function,
         obj.concentration, obj.mass_def, obj.hm_consistency,
         obj.convert_masses) = aux_data
        return obj

    def update(self, cosmology=None, halo_mass_function=None, halo_bias=None, subhalo_mass_function=None, concentration=None, mass_def=None,
               hm_consistency=None, convert_masses=None, m_grid=None):
        """
        Return a new HaloModel instance with updated components.

        Parameters
        ----------
        cosmology, halo_mass_function, halo_bias, subhalo_mass_function, concentration, mass_def, hm_consistency, convert_masses, m_grid : optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

        Returns
        -------
        HaloModel
            New halo-model instance with updated attributes.
        """
        # Flatten current state
        children, aux_data = self._tree_flatten()
        # Unpack
        cosmo_child, m_grid0 = children
        (
            halo_mass_function0, halo_bias0, subhalo_mass_function0, concentration0,
            mass_def0, hm_consistency0, convert_masses0
        ) = aux_data

        # Update only provided components
        new_cosmo = cosmology if cosmology is not None else cosmo_child
        new_m_grid = jnp.sort(m_grid) if m_grid is not None else m_grid0
        new_halo_mass_function = halo_mass_function if halo_mass_function is not None else halo_mass_function0
        new_halo_bias = halo_bias if halo_bias is not None else halo_bias0
        new_subhalo_mass_function = subhalo_mass_function if subhalo_mass_function is not None else subhalo_mass_function0
        new_concentration = concentration if concentration is not None else concentration0
        new_mass_def = mass_def if mass_def is not None else mass_def0
        new_hm_consistency = hm_consistency if hm_consistency is not None else hm_consistency0
        new_convert_masses = convert_masses if convert_masses is not None else convert_masses0

        new_aux_data = (
            new_halo_mass_function, new_halo_bias, new_subhalo_mass_function, new_concentration,
            new_mass_def, new_hm_consistency, new_convert_masses
        )
        # Use _tree_unflatten to create the new instance efficiently
        return self._tree_unflatten(new_aux_data, (new_cosmo, new_m_grid))
       
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

        m = self.m_grid
        z = jnp.atleast_1d(z)
        cparams = self.cosmology._cosmo_params()
        logm = jnp.log(m)
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"]
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # (Nm, 1)
    
    
        # Public HMF and bias interfaces use physical masses.
        dn_dlnm = jnp.reshape(self.halo_mass_function.dndlnm(self.cosmology, m, z, self.mass_def, self.convert_masses), (len(m), len(z)))
        b1 = jnp.reshape(self.halo_bias.bias(self.cosmology, m, z, self.mass_def, self.convert_masses, 1), (len(m), len(z)))
        b2 = jnp.reshape(self.halo_bias.bias(self.cosmology, m, z, self.mass_def, self.convert_masses, 2), (len(m), len(z)))
    
        # Compute integrals I0, I1, I2
        I0 = jnp.trapezoid(dn_dlnm * m_over_rho_mean, x=logm, axis=0)  # (Nz,)
        I1 = jnp.trapezoid(b1 * dn_dlnm * m_over_rho_mean, x=logm, axis=0)
        I2 = jnp.trapezoid(b2 * dn_dlnm * m_over_rho_mean, x=logm, axis=0)
    
        # Apply formulas
        m_min =  m[0]
        n_min =  (1.0 - I0) * rho_mean_0 / m_min
        b1_min = (1.0 - I1) * rho_mean_0 / m_min / n_min
        b2_min = -I2 * rho_mean_0 / m_min / n_min
    
        return n_min, b1_min, b2_min


    @partial(jax.jit, static_argnums=(1, 4))
    def _I(self, profile, k, z, bias_order=1):
        """
        Generalised halo model mass integral.

        .. math::

            I^{(\\beta)}(k, z) = \\int \\frac{dn}{d\\ln M}\\, b_\\beta(M, z)\\,
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
        k, m, z = jnp.atleast_1d(k), self.m_grid, jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(
            self.halo_mass_function.dndlnm(self.cosmology, m, z, self.mass_def, self.convert_masses),
            (len(m), len(z)),
        )

        if bias_order == 0:
            bias_w = jnp.ones((len(m), len(z)))
        elif bias_order == 1:
            bias_w = jnp.reshape(
                self.halo_bias.bias(self.cosmology, m, z, self.mass_def, self.convert_masses, order=1),
                (len(m), len(z)),
            )
        elif bias_order == 2:
            bias_w = jnp.reshape(
                self.halo_bias.bias(self.cosmology, m, z, self.mass_def, self.convert_masses, order=2),
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

    @partial(jax.jit, static_argnums=(1, 2))
    def pk_1h(self, profile1, profile2, k, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M} \\, u_1(k, M, z) u_2(k, M, z)

        where :math:`dn/d\\ln M` is the halo mass function
        and :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile.
        The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None
            Second halo profile object (if None, uses profile1).
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for the low-k suppression factor.

        Returns
        -------
        pk_1h : array
            1-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """

        k, m, z = jnp.atleast_1d(k), self.m_grid, jnp.atleast_1d(z)
        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Setup
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(self.halo_mass_function.dndlnm(self.cosmology, m, z, self.mass_def, self.convert_masses), (len(m), len(z)))
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        # Process a single mass bin at a time and extract the uk^2 at the lowest mass for the halo model consistency term
        def process_bin(i):
            pair_kernel = _fourier_2pt(self, profile1, profile2, k, m, z)
            pair_kernel = jnp.reshape(pair_kernel, (len(k), len(m), len(z)))
            uk_sq_row = pair_kernel[:, i, :]

            return uk_sq_row * total_weights[i], uk_sq_row

        # vmap through the mass bins
        integrand_rows, all_sq_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))

        pk1h = jnp.sum(integrand_rows, axis=0)

        # Apply halo model consistency correction: n_min * uk_sq_min
        uk_sq_min = all_sq_profiles[0]
        n_min, _, _ = self._counter_terms(z)
        correction = n_min[None, :] * uk_sq_min
        pk1h = pk1h + self.hm_consistency * correction

        # Apply damping
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, k_damp, 1.0))**2), 1.0)

        return jnp.squeeze(pk1h * damping[:, None])
            
       
    @partial(jax.jit, static_argnums=(1, 2))
    def cl_1h(self, tracer1, tracer2, l, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the angular power spectrum
        :math:`C_\\ell^{1h}`.

        The Limber-projected spectrum is obtained by integrating the 1-halo
        3D power spectrum against the tracer kernels and the comoving volume
        element. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` passed through to :meth:`pk_1h`.

        Returns
        -------
        cl_1h : array
            Dimensionless 1-halo angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """

        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function to map l -> k for a specific z
        def get_pk_slice(zi):
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1 + zi)
            ki = (l + 0.5) / chi_i
            pk = self.pk_1h(tracer1.profile, tracer2.profile, k=ki, z=jnp.atleast_1d(zi), k_damp=k_damp)
            return pk.flatten()

        # Get the halo model pk_1h, the kernels, and the Limber weight c/(H chi^2)
        P_1h_grid = jax.vmap(get_pk_slice)(z)
        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        chi = self.cosmology.angular_diameter_distance(z) * (1.0 + z)
        limber_weight = self.cosmology.comoving_volume_element(z) / chi**4

        # Limber integral: C_ell = int dz (c/H chi^2) W1 W2 P
        integrand = P_1h_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))
    


    @partial(jax.jit, static_argnums=(1, 2))
    def pk_2h(self, profile1, profile2, k, z):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1(k, z) \\, I_2(k, z)

        with

        .. math::

            I_i(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z) \\, b(M, z) \\, u_i(k \\mid M, z),

        where :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile,
        :math:`dn/d\\ln M` is the halo mass function, and :math:`b(M, z)` is the
        linear halo bias. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None
            Second halo profile object (if None, uses profile1).
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        pk_2h : array
            2-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """

        k, m, z = jnp.atleast_1d(k), self.m_grid, jnp.atleast_1d(z)

        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Ingredients
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        # Combine hmf, bias, and weights into a single (Nm, Nz) weight grid
        dndlnm = jnp.reshape(self.halo_mass_function.dndlnm(self.cosmology, m, z, self.mass_def, self.convert_masses), (len(m), len(z)))
        bias = jnp.reshape(self.halo_bias.bias(self.cosmology, m, z, self.mass_def, self.convert_masses), (len(m), len(z)))
        total_weights = dndlnm * bias * w[:, None]

        def get_I(profile):
            # This function processes a single index 'i' of the mass axis
            def process_bin(i):
                uk_full = jnp.reshape(profile.fourier(self, k, m, z), (len(k), len(m), len(z)))
                uk_slice = uk_full[:, i, :]
                return uk_slice * total_weights[i], uk_slice

            # Vmap over the indices 0...Nm-1, then integrate and pluck index 0 for hm consistency
            integrand_rows, all_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
            integral = jnp.sum(integrand_rows, axis=0)
            u_k_min = all_profiles[0]  # vmap output is (Nm, Nk, Nz)

            n_min, b1_min, _ = self._counter_terms(z)
            correction = b1_min[None, :] * n_min[None, :] * u_k_min

            return integral + self.hm_consistency * correction
    
        # Final Power Spectrum
        I1 = get_I(profile1)
        I2 = I1 if profile1 is profile2 else get_I(profile2)

        P_lin = self.cosmology.pk(k, z, linear=True)
        # Ensure P_lin has shape (N_k, N_z)
        P_lin = jnp.reshape(P_lin, (len(k), -1))

        return jnp.squeeze(P_lin * I1 * I2)


    @partial(jax.jit, static_argnums=(1, 2))
    def cl_2h(self, tracer1, tracer2, l, z):
        """
        Compute the 2-halo contribution to the angular power spectrum
        :math:`C_\\ell^{2h}`.

        The Limber-projected spectrum is obtained by integrating the 2-halo
        3D power spectrum against the tracer kernels and the comoving volume
        element. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.

        Returns
        -------
        cl_2h : array
            Dimensionless 2-halo angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function for Limber integration
        def get_pk_slice(zi):
            # Map l to k using the Limber approximation and then get the pk_2h
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1 + zi)
            ki = (l + 0.5) / chi_i
            return self.pk_2h(tracer1.profile, tracer2.profile, k=ki, z=jnp.atleast_1d(zi)).flatten()
    
        # Map over redshift to get P(k=l/chi, z)
        P_2h_grid = jax.vmap(get_pk_slice)(z) 
        
        # Get individual kernels and the Limber weight c/(H chi^2)
        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        chi = self.cosmology.angular_diameter_distance(z) * (1.0 + z)
        limber_weight = self.cosmology.comoving_volume_element(z) / chi**4

        # Limber integral: C_ell = int dz (c/H chi^2) W1 W2 P
        integrand = P_2h_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))


jax.tree_util.register_pytree_node(
    HaloModel,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: HaloModel._tree_unflatten(aux_data, children)
)
