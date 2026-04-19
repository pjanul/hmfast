"""
Core halo model implementation using JAX for differentiability.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any, Optional, Callable
from functools import partial
from mcfit import TophatVar

from hmfast.halos.massfunc import T08HaloMass, TW10SubHaloMass
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import D08Concentration, B13Concentration
from hmfast.halos.mass_definition import MassDefinition
from hmfast.cosmology import Cosmology
from hmfast.utils import newton_root

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
    mass_definition : MassDefinition
        Native spherical-overdensity mass definition used throughout the halo model.
    halo_mass_function : HaloMass
        Halo mass function model used to compute :math:`dn / d\ln M`.
    halo_bias : HaloBias
        Halo bias model used for large-scale halo bias predictions.
    subhalo_mass_function : SubHaloMass
        Subhalo mass function model used in observables with satellite or subhalo contributions.
    concentration : Concentration
        Halo concentration relation used to map halo mass and redshift to concentration.
    hm_consistency : bool
        Flag controlling whether halo-model consistency counterterms are applied.
    convert_masses : bool
        Flag controlling whether profile-specific native mass definitions are converted automatically.
    """
   

    
    def __init__(self, 
                 cosmology=Cosmology(cosmo_model=0), 
                 mass_definition=MassDefinition(delta=200, reference="critical"), 
                 halo_mass_function=T08HaloMass(), 
                 halo_bias=T10HaloBias(), 
                 subhalo_mass_function=TW10SubHaloMass(),
                 concentration=D08Concentration(), 
                 hm_consistency=True, 
                 convert_masses=False):
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

        self.mass_definition = mass_definition
        self.hm_consistency = hm_consistency
        self.convert_masses = convert_masses


        # Create TophatVar instance once to instantiate it
        dummy_k, _ = self.cosmology.pk(1., linear=True)
        self._tophat_instance = partial(TophatVar(dummy_k, lowring=True, backend='jax'), extrap=True)


    def _tree_flatten(self):
        # The cosmology is a Pytree, so it is a child.
        # Everything else is configuration/metadata.
        children = (self.cosmology,)
        aux_data = (self.halo_mass_function, self.halo_bias, self.subhalo_mass_function, self.concentration,
            self.mass_definition, self.hm_consistency, self.convert_masses, self._tophat_instance
        )
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        cosmology, = children
        obj = cls.__new__(cls)
        obj.cosmology = cosmology
        (obj.halo_mass_function, obj.halo_bias, obj.subhalo_mass_function, 
         obj.concentration, obj.mass_definition, obj.hm_consistency, 
         obj.convert_masses, obj._tophat_instance) = aux_data
        return obj

   

    def update(self, cosmology=None, halo_mass_function=None, halo_bias=None, subhalo_mass_function=None, concentration=None, mass_definition=None, 
               hm_consistency=None, convert_masses=None):
        """
        Return a new HaloModel instance with updated components.

        Parameters
        ----------
        cosmology, halo_mass_function, halo_bias, subhalo_mass_function, concentration, mass_definition, hm_consistency, convert_masses : optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

        Returns
        -------
        HaloModel
            New halo-model instance with updated attributes.
        """
        # Flatten current state
        children, aux_data = self._tree_flatten()
        # Unpack
        (cosmo_child,) = children
        (
            halo_mass_function0, halo_bias0, subhalo_mass_function0, concentration0,
            mass_definition0, hm_consistency0, convert_masses0, tophat_instance0
        ) = aux_data
    
        # Update only provided components
        new_cosmo = cosmology if cosmology is not None else cosmo_child
        new_halo_mass_function = halo_mass_function if halo_mass_function is not None else halo_mass_function0
        new_halo_bias = halo_bias if halo_bias is not None else halo_bias0
        new_subhalo_mass_function = subhalo_mass_function if subhalo_mass_function is not None else subhalo_mass_function0
        new_concentration = concentration if concentration is not None else concentration0
        new_mass_definition = mass_definition if mass_definition is not None else mass_definition0
        new_hm_consistency = hm_consistency if hm_consistency is not None else hm_consistency0
        new_convert_masses = convert_masses if convert_masses is not None else convert_masses0
    
        # Reuse the existing tophat instance (or update if needed)
        new_aux_data = (
            new_halo_mass_function, new_halo_bias, new_subhalo_mass_function, new_concentration,
            new_mass_definition, new_hm_consistency, new_convert_masses, tophat_instance0
        )
        # Use _tree_unflatten to create the new instance efficiently
        return self._tree_unflatten(new_aux_data, (new_cosmo,))
       

    #@partial(jax.jit, static_argnums=0)
    def _delta_vir_to_crit(self, z):
        """
        Compute the virial overdensity with respect to the critical density,
        :math:`\\Delta_{\\mathrm{vir}}(z)`,
        using the Bryan & Norman (1998) fitting formula for a flat universe.

        The formula is:

        .. math::

            \\Delta_{\\mathrm{vir}}(z) = 18\\pi^2 + 82x - 39x^2

        where :math:`x = \\Omega_m(z) - 1`.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to compute the virial overdensity.

        Returns
        -------
        delta_vir : float or array-like
            Virial overdensity relative to the critical density at redshift :math:`z`.
        """
        omega_m = self.cosmology.omega_m(z)
        x = omega_m - 1.0
    
        return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2

    #@partial(jax.jit, static_argnums=0)
    def _delta_numeric(self, z):
        """ 
        Always return numeric delta at redshift z
        in the native reference (self.reference).
        """
        if self.mass_definition.delta == "vir":
            if self.mass_definition.reference != "critical":
                raise ValueError("virial overdensity only defined w.r.t. critical density")
            return self._delta_vir_to_crit(z)
    
        return self.mass_definition.delta


     #@partial(jax.jit, static_argnums=(0,1))
    def _convert_reference(self, z, delta, from_ref='critical', to_ref='mean'):
        """
        Convert overdensity between 'critical' and 'mean' definitions.
        
        Parameters
        ----------
        delta : float or array
        z : float or array
        from_ref, to_ref : {'critical', 'mean'}
        """
        if from_ref == to_ref:
            return jnp.full_like(z, delta)
            
        omega_m = self.cosmology.omega_m(z)
        if from_ref == 'critical' and to_ref == 'mean':
            return delta / omega_m
        elif from_ref == 'mean' and to_ref == 'critical':
            return delta * omega_m
        else:
            raise ValueError("from_ref and to_ref must be 'critical' or 'mean'")


    @partial(jax.jit, static_argnums=(3, 4, 6))
    def convert_m_delta(self, m, z, mass_def_old, mass_def_new, c_old=None, max_iter=20):
        """
        Convert halo masses between two spherical-overdensity definitions.
        
        The overdensity threshold is changed from :math:`\\Delta` to
        :math:`\\Delta'` by solving for the corresponding mass
        :math:`M_{\\Delta'}`,
        
        .. math::
        
            \\frac{M_{\\Delta}}{M_{\\Delta'}} =
            \\frac{f_\\mathrm{NFW}(c_{\\Delta})}
            {f_\\mathrm{NFW}\\left(
            c_{\\Delta} \\, \\frac{r_{\\Delta'}}{r_{\\Delta}}
            \\right)},
        
        where :math:`f_\\mathrm{NFW}(x) = \\ln(1+x) - x/(1+x)` and
        :math:`r_\\Delta = \\left[3 M_\\Delta / (4 \\pi \\, \\Delta \\, \\rho_\\mathrm{ref}(z))\\right]^{1/3}`.
        Here :math:`r_{\\Delta'}` is defined analogously using
        :math:`M_{\\Delta'}` and :math:`\\Delta'`.
        
        Reference-density conversions are performed through
        
        .. math::
        
            \\Delta_{\\mathrm{crit}} = \\Delta_{\\mathrm{mean}} \\, \\Omega_m(z),
        
        and the virial overdensity relative to the critical density is defined by
        
        .. math::
        
            \\Delta_{\\mathrm{vir}}(z) = 18\\pi^2 + 82x - 39x^2,
            \\qquad x = \\Omega_m(z) - 1.
        
        Parameters
        ----------
        m : array-like
            Halo mass in the original definition, :math:`M_{\\Delta}`.
        z : array-like
            Redshift(s).
        mass_def_old : MassDefinition
            Original mass definition specifying :math:`\\Delta` and its reference
            density.
        mass_def_new : MassDefinition
            Target mass definition specifying :math:`\\Delta'` and its reference
            density.
        c_old : array-like, optional
            Halo concentration :math:`c_{\\Delta}` in the original definition. If
            ``None``, it is computed automatically.
        max_iter : int, optional
            Maximum number of root-finder iterations.
        
        Returns
        -------
        array-like
            Halo mass in the target definition, :math:`M_{\\Delta'}`, with shape
            :math:`(N_M, N_z)`.
        """
       
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        Nm, Nz = len(m), len(z)

        # Vectorized Delta calculation
        def get_delta_crit(mdef, z_val):
            d = jnp.where(mdef.delta == "vir", self._delta_vir_to_crit(z_val), mdef.delta)
            return jnp.where(mdef.reference == "mean", d * self.cosmology.omega_m(z_val), d)

        d_old_z, d_new_z = get_delta_crit(mass_def_old, z), get_delta_crit(mass_def_new, z)
        is_same_z = jnp.isclose(d_old_z, d_new_z) & (mass_def_old.reference == mass_def_new.reference)

        # Explicitly handle the grid shapes
        mm, zz = jnp.meshgrid(m, z, indexing='ij')
        
        if c_old is None:
            c_old = self.concentration.c_delta(self, m, z)
        c_old = c_old[:Nm, :Nz].reshape(mm.shape)

        # First guess for the root finder is based on a power law approximation m * (Delta1 / Delta2)^0.2
        x0 = m[:, None] * (d_old_z / d_new_z)[None, :]**0.2   

        # Solver Logic
        def solve_single(m_i, c_i, x0_i, d_o, d_n, same_flag):
            f_nfw = lambda x: jnp.log1p(x) - x / (1.0 + x)
            obj = lambda m_new: m_i / m_new - f_nfw(c_i) / f_nfw(c_i * (m_new/m_i * d_o/d_n)**(1/3))
            
            return jax.lax.cond(same_flag, lambda _: m_i, 
                                lambda _: newton_root(obj, x0=x0_i, max_iter=max_iter), None)

    
        # Broadcast 1D redshift-dependent arrays to match the (Nm, Nz) mesh
        d_o_flat = jnp.broadcast_to(d_old_z[None, :], mm.shape).flatten()
        d_n_flat = jnp.broadcast_to(d_new_z[None, :], mm.shape).flatten()
        same_flat = jnp.broadcast_to(is_same_z[None, :], mm.shape).flatten()

        results = jax.vmap(solve_single)(mm.flatten(), c_old.flatten(), x0.flatten(), d_o_flat, d_n_flat, same_flat)
        
        return results.reshape(mm.shape)


   
    def r_delta(self, m, z, mass_definition=None):
        """
        Compute the halo radius :math:`r_\\Delta` associated with a halo mass.

        .. math::

            r_\\Delta = \\left[\\frac{3M}{4\\pi \\Delta \\rho_{\\mathrm{ref}}(z)}\\right]^{1/3}

        Parameters
        ----------
        m : float
            Halo mass enclosed within the overdensity radius.
        z : float
            Redshift at which to compute the radius.
        mass_definition : MassDefinition, optional
            Mass definition (default: self.mass_definition).

        Returns
        -------
        float
            Radius :math:`r_\\Delta` within which the mean enclosed density is
            :math:`\\Delta \\rho_{\\mathrm{ref}}(z)`.
        """
        
        mass_definition = self.mass_definition if mass_definition is None else mass_definition

        delta, reference = mass_definition.delta, mass_definition.reference
       
        m = jnp.atleast_1d(m)[:, None]  # (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # (1, Nz)

        # Define your reference density. Default is rho_crit
        rho_ref = self.cosmology.critical_density(z)

        # If the user selects vir or rho_mean, correct for this
        if delta == "vir":
            delta = self._delta_vir_to_crit(z)
        
        if reference == "mean":
            rho_ref *= self.cosmology.omega_m(z)
            
        return (3.0 * m / (4.0 * jnp.pi * delta * rho_ref))**(1./3.)


    
    @jax.jit
    def _counter_terms(self, m, z):
        """
        Compute :math:`n_{\\min}`, :math:`b_{1,\\min}`, and :math:`b_{2,\\min}` counter terms for halo model consistency.

        Parameters
        ----------
        m : array-like
            Mass grid.
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
       
        m = jnp.atleast_1d(m)
        logm = jnp.log(m)
        cparams = self.cosmology._cosmo_params()
        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_cb"]   # Omega0_m without neutrinos
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # (Nm, 1)
    
        # Compute dn/dlnM and bias for each z
        dn_dlnm = self.halo_mass_function.halo_mass_function(self, m=m, z=z)  # (Nm, Nz)
        b1 = self.halo_bias.halo_bias(self, m=m, z=z, order=1)      # (Nm, Nz)
        b2 = self.halo_bias.halo_bias(self, m=m, z=z, order=2)      # (Nm, Nz)
    
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


    @partial(jax.jit, static_argnums=(1, 2))
    def pk_1h(self, tracer1, tracer2, k, m, z,  k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M} \\, u_1(k, M, z) u_2(k, M, z)

        where :math:`dn/d\\ln M` is the halo mass function 
        and :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        k : array-like
            Wavenumber grid.
        m : array-like
            Mass grid.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping scale for small-scale power.

        Returns
        -------
        pk_1h : array-like
            1-halo power spectrum, shape (len(k), len(z)).
        """
    
        h = self.cosmology.H0 / 100
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        
        # Weights and Setup
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5
        
        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)
        total_weights = dndlnm * w[:, None] # (Nm, Nz)
    
        is_same_tracer = (tracer2 is None) or (tracer1 == tracer2)
        tracer2 = tracer1 if tracer2 is None else tracer2

    
        # Process a single mass bin at a time and extract the uk^2 at the lowest mass for the halo model consistency term
        def process_bin(i):
            # We need the profiles for index 'i' while squaring uk if the user is doing an autocorrelation
            if is_same_tracer:
                _, uk_sq_full = tracer1.profile.u_k(self, k/h, m, z, moment=2)
                uk_sq_row = uk_sq_full[:, i, :]
            elif tracer1.profile.has_central_contribution and tracer2.profile.has_central_contribution:
                s1, c1 = tracer1.profile._sat_and_cen_contribution(self, k/h, m, z)
                s2, c2 = tracer2.profile._sat_and_cen_contribution(self, k/h, m, z)
                # Pull only row i
                uk_sq_row = s1[:, i, :] * s2[:, i, :] + s1[:, i, :] * c2[:, i, :] + s2[:, i, :] * c1[:, i, :]
            else:
                _, u1 = tracer1.profile.u_k(self, k/h, m, z, moment=1)
                _, u2 = tracer2.profile.u_k(self, k/h, m, z, moment=1)
                uk_sq_row = u1[:, i, :] * u2[:, i, :]
    
            return uk_sq_row * total_weights[i], uk_sq_row
    
        # vmap through the mass bins
        integrand_rows, all_sq_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
    
        pk1h = jnp.sum(integrand_rows, axis=0)
    
        # Apply halo model consistency correction: n_min * uk_sq_min 
        uk_sq_min = all_sq_profiles[0] 
        n_min, _, _ = self._counter_terms(m, z)
        correction = n_min[None, :] * uk_sq_min
        pk1h = pk1h + self.hm_consistency * correction
    
        # Apply damping
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, k_damp, 1.0))**2), 1.0)
    
        return pk1h * damping[:, None]
            
       
    @partial(jax.jit, static_argnums=(1, 2))
    def cl_1h(self, tracer1, tracer2, l, m, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the angular power spectrum
        :math:`C_\\ell^{1h}`.

        The Limber-projected spectrum is obtained by integrating the 1-halo
        3D power spectrum against the tracer kernels and comoving volume element.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        m : array-like
            Mass grid.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping scale for small-scale power.

        Returns
        -------
        cl_1h : array-like
            1-halo angular power spectrum, shape (len(l),).
        """
        
        
        h = self.cosmology.H0 / 100

        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function to map l -> k for a specific z
        def get_pk_slice(zi):
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1 + zi) 
            ki = (l + 0.5) / chi_i
            pk = self.pk_1h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi), k_damp=k_damp)
            return pk.flatten()

        # Get the halo model pk_1h, the kernel, and the comoving volume
        P_1h_grid = jax.vmap(get_pk_slice)(z)
        kernel1 = tracer1.kernel(self.cosmology, z)  
        kernel2 = tracer2.kernel(self.cosmology, z)  
        comov_vol = self.cosmology.comoving_volume_element(z) 

        # Integrate over redshift
        integrand = P_1h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)
    


    @partial(jax.jit, static_argnums=(1, 2))
    def pk_2h(self, tracer1, tracer2, k, m, z):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::
        
            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1(k, z) \\, I_2(k, z)
        
        with
        
        .. math::
        
            I_i(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z) \\, b(M, z) \\, u_i(k \\mid M, z),
        
        where :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile,
        :math:`dn/d\\ln M` is the halo mass function, and :math:`b(M, z)` is the
        linear halo bias.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        k : array-like
            Wavenumber grid.
        m : array-like
            Mass grid.
        z : array-like
            Redshift grid.

        Returns
        -------
        pk_2h : array-like
            2-halo power spectrum, shape (len(k), len(z)).
        """
        
        cparams = self.cosmology._cosmo_params()
        h, k, m, z = cparams["h"], jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        tracer2 = tracer1 if tracer2 is None else tracer2
    
        # Weights and Ingredients
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        # Combine hmf, bias, and weights into a single (Nm, Nz) weight grid
        dndlnm = self.halo_mass_function.halo_mass_function(self, m, z)
        bias = self.halo_bias.halo_bias(self, m, z)
        total_weights = dndlnm * bias * w[:, None]
    
        def get_I(tracer):
            # This function processes a single index 'i' of the mass axis
            def process_bin(i):
                _, uk_full = tracer.profile.u_k(self, k/h, m, z, moment=1)
                uk_slice = uk_full[:, i, :] 
                return uk_slice * total_weights[i], uk_slice
    
            # Vmap over the indices 0...Nm-1, then integrate and pluck index 0 for hm consistency
            integrand_rows, all_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
            integral = jnp.sum(integrand_rows, axis=0)
            u_k_min = all_profiles[0] # vmap output is (Nm, Nk, Nz)
    
            n_min, b1_min, _ = self._counter_terms(m, z)
            correction = b1_min[None, :] * n_min[None, :] * u_k_min
            
            return integral + self.hm_consistency * correction
    
        # Final Power Spectrum
        I1 = get_I(tracer1)
        I2 = I1 if tracer1 == tracer2 else get_I(tracer2)
        
        P_lin = jax.vmap(lambda zi: jnp.interp(k, *self.cosmology.pk(zi, linear=True)))(z).T * h**3
        
        return P_lin * I1 * I2


    @partial(jax.jit, static_argnums=(1, 2))
    def cl_2h(self, tracer1, tracer2, l, m, z):
        """
        Compute the 2-halo contribution to the angular power spectrum
        :math:`C_\\ell^{2h}`.

        The Limber-projected spectrum is obtained by integrating the 2-halo
        3D power spectrum against the tracer kernels and comoving volume element.

        Parameters
        ----------
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        m : array-like
            Mass grid.
        z : array-like
            Redshift grid.

        Returns
        -------
        cl_2h : array-like
            2-halo angular power spectrum, shape (len(l),).
        """
       
        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function for Limber integration
        def get_pk_slice(zi):
            # Map l to k using the Limber approximation and then get the pk_2h  
            chi_i = self.cosmology.angular_diameter_distance(zi) * (1 + zi) 
            ki = (l + 0.5) / chi_i
            return self.pk_2h(tracer1, tracer2, k=ki, m=m, z=jnp.atleast_1d(zi)).flatten()
    
        # Map over redshift to get P(k=l/chi, z)
        P_2h_grid = jax.vmap(get_pk_slice)(z) 
        
        # Get individual kernels
        kernel1 = tracer1.kernel(self.cosmology, z)
        kernel2 = tracer2.kernel(self.cosmology, z)
        
        comov_vol = self.cosmology.comoving_volume_element(z)
    
        # Limber Integral: C_l = int dz P(k,z) * [W1 * W2 * dV/dz]
        integrand = P_2h_grid * (comov_vol[:, None] * kernel1[:, None] * kernel2[:, None])
        
        return jnp.trapezoid(integrand, x=z, axis=0)


jax.tree_util.register_pytree_node(
    HaloModel,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: HaloModel._tree_unflatten(aux_data, children)
)
