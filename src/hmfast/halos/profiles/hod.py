import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
import functools
from jax.scipy.special import erf

from hmfast.download import get_default_data_path
from hmfast.halos.profiles import HaloProfile


class GalaxyHODProfile(HaloProfile):
    pass



class StandardGalaxyHODProfile(GalaxyHODProfile):
    """
    Standard Galaxy HOD profile.

    Attributes
    ----------
    sigma_log10M : float
        Scatter in :math:`\log_{10} M` for the central-galaxy occupation threshold.
    alpha_s : float
        Power-law slope of the satellite occupation.
    M1_prime : float
        Characteristic satellite mass scale.
    M_min : float
        Minimum halo mass for central occupation.
    M0 : float
        Satellite cutoff mass.
    """

    def __init__(self, sigma_log10M=0.68, alpha_s=1.30, M1_prime=10**12.7, M_min=10**11.8, M0=0.0):        
        
        self.sigma_log10M, self.alpha_s, self.M1_prime, self.M_min, self.M0  = sigma_log10M, alpha_s, M1_prime, M_min, M0

    @property
    def has_central_contribution(self):
        return True
    
  
    # --- JAX PyTree Registration ---

    def _tree_flatten(self):
        # Dynamic leaves (JAX will track these for gradients/jit) and static metadata (changes will trigger a recompile)
        leaves = (self.sigma_log10M, self.alpha_s, self.M1_prime, self.M_min, self.M0)
        return (leaves, None)


    @classmethod
    def _tree_unflatten(cls, aux, leaves):
        return cls(*leaves)

    def update(self, sigma_log10M=None, alpha_s=None, M1_prime=None, M_min=None, M0=None):
        """
        Return a new profile instance with updated HOD parameters.

        Parameters
        ----------
        sigma_log10M, alpha_s, M1_prime, M_min, M0 : float, optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.

        Returns
        -------
        StandardGalaxyHODProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        # Use existing values if the new ones are None
        new_leaves = (
            sigma_log10M if sigma_log10M is not None else self.sigma_log10M,
            alpha_s if alpha_s is not None else self.alpha_s,
            M1_prime if M1_prime is not None else self.M1_prime,
            M_min if M_min is not None else self.M_min,
            M0 if M0 is not None else self.M0,
        )
        
        return self._tree_unflatten(treedef, new_leaves)

    # --- Physics Implementations ---

    def n_cen(self, m):
        """
        Expected number of central galaxies in a halo of mass m.
    
        The value is given by:
    
            .. math::
    
                N_\\mathrm{cen}(m) = \\frac{1}{2} \\left[1 + \\mathrm{erf}\\left(\\frac{\\log_{10} m - \\log_{10} M_\\mathrm{min}}{\\sigma_{\\log_{10} M}}\\right)\\right]
    
        Parameters
        ----------
        m : array-like
            Halo mass.
    
        Returns
        -------
        n_cen : array-like
            Expected number of central galaxies per halo.
        """
        # Using attributes directly as they are now JAX-traced leaves
        x = (jnp.log10(m) - jnp.log10(self.M_min)) / self.sigma_log10M
        return 0.5 * (1.0 + erf(x))

    def n_sat(self, m):
        """
        Expected number of satellite galaxies in a halo of mass m.
    
        The value is given by:
    
            .. math::
    
                N_\\mathrm{sat}(m) = H(m - M_0) \\, N_\\mathrm{cen}(m) \\, \\left(\\frac{m - M_0}{M_1'}\\right)^{\\alpha_s}

    
        where the term in parentheses is set to zero if negative.
    
        Parameters
        ----------
        m : array-like
            Halo mass.
    
        Returns
        -------
        n_sat : array-like
            Expected number of satellite galaxies per halo.
        """
        pow_term = jnp.maximum((m - self.M0) / self.M1_prime, 0.0)**self.alpha_s
        return self.n_cen(m) * pow_term

    def ng_bar(self, halo_model, m, z):
        """
        Comoving mean galaxy number density at redshift z.
    
        The value is given by:
    
            .. math::
    
                \\bar{n}_g(z) = \\langle N_\\mathrm{cen} + N_\\mathrm{sat} \\rangle_n
    
        where the average is over the halo mass function.
    
        Parameters
        ----------
        halo_model : HaloModel
            The parent halo model instance.
        m : array-like
            Halo mass grid.
        z : array-like
            Redshift grid.
    
        Returns
        -------
        ng : array-like
            Mean galaxy number density as a function of redshift.
        """
       
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = halo_model.mass_model.halo_mass_function(halo_model, m, z)
        ng_val = jnp.trapezoid(dndlnm * Ntot[:, None], x=logm, axis=0)

        # HM Consistency check
        return jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model._counter_terms(m, z)[0] * Ntot[0], lambda x: x, ng_val)

    def galaxy_bias(self, halo_model, m, z):
        """
        Large-scale galaxy bias at redshift z.
    
        The value is given by:
    
            .. math::
    
                b_g(z) = \\frac{1}{\\bar{n}_g(z)} \\langle b^{(1)} (N_\\mathrm{cen} + N_\\mathrm{sat}) \\rangle_n
    
        where $b^{(1)}$ is the first-order halo bias and the average is over the halo mass function.
    
        Parameters
        ----------
        halo_model : HaloModel
            The parent halo model instance.
        m : array-like
            Halo mass grid.
        z : array-like
            Redshift grid.
    
        Returns
        -------
        bias : array-like
            Large-scale galaxy bias as a function of redshift.
        """
       
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Ntot = self.n_cen(m) + self.n_sat(m)
        dndlnm = halo_model.mass_model.halo_mass_function(halo_model, m, z)
        bh = halo_model.halo_bias(m, z, order=1)
        ng = self.ng_bar(halo_model, m, z)

        bg_num = jnp.trapezoid(dndlnm * bh * Ntot[:, None], x=logm, axis=0)
        bg_num = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model._counter_terms(m, z)[1] * Ntot[0], lambda x: x, bg_num)
        return bg_num / ng


    def _sat_and_cen_contribution(self, halo_model, k, m, z):
        """ 
        Compute either the first or second moment of the galaxy HOD tracer u_ell.
        For galaxy HOD:, 
            First moment:     W_g / ng_bar * [Nc + Ns * u_ell_m]
            Second moment:    W_g^2 / ng_bar^2 * [Ns^2 * u_ell_m^2 + 2 * Ns * u_ell_m]
        You cannot simply take u_ell_g**2.
        """

       
        Ns = self.n_sat(m)
        Nc = self.n_cen(m)
        ng = self.ng_bar(halo_model, m, z) * (halo_model.cosmology.H0 / 100)**3

        _, u_m = self._u_k_matter(halo_model, k, m, z)  

        sat_term = (1/ng) * (Ns[None, :, None] * u_m)
        cen_term = (1/ng) * (Nc[None, :, None]**0)
    
        return sat_term, cen_term


    def u_k(self, halo_model, k, m, z, moment=1):
        """
        Fourier-space moment of the galaxy HOD profile.
    
        Computes the first or second moment of the galaxy HOD Fourier-space profile:
    
            .. math::
    
                u_k^{(1)}(k, m, z) = \\frac{N_\\mathrm{cen}(m) + N_\\mathrm{sat}(m) \\, u_m(k, m, z)}{\\bar{n}_g(z)}
    
            .. math::
    
                u_k^{(2)}(k, m, z) = \\frac{N_\\mathrm{sat}^2(m) \\, u_m^2(k, m, z) + 2 N_\\mathrm{sat}(m) \\, u_m(k, m, z)}{\\bar{n}_g^2(z)}
    
        where $u_m(k, m, z)$ is the normalized matter profile in Fourier space.
    
        Parameters
        ----------
        halo_model : HaloModel
            The parent halo model instance.
        k : array-like
            Wavenumber grid.
        m : array-like
            Halo mass grid.
        z : array-like
            Redshift grid.
        moment : int, optional
            Moment to compute (1 for first, 2 for second). Default is 1.
    
        Returns
        -------
        k : array-like
            Wavenumber grid.
        u_k : array-like
            Fourier-space profile moment.
        """
       
        Ns = self.n_sat(m)
        Nc = self.n_cen(m)
        ng = self.ng_bar(halo_model, m, z) * (halo_model.cosmology.H0 / 100)**3

        _, u_m = self._u_k_matter(halo_model, k, m, z)
    
        moment_funcs = [
            
            lambda _: (1/ng) * (Nc[None, :, None] + Ns[None, :, None] * u_m),
            lambda _: (1/ng**2) * (Ns[None, :, None]**2 * u_m**2 + 2 * Ns[None, :, None] * u_m),
        ]
    
        u_k = jax.lax.switch(moment - 1, moment_funcs, None)
        return k, u_k
        

jax.tree_util.register_pytree_node(
    StandardGalaxyHODProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: StandardGalaxyHODProfile._tree_unflatten(aux_data, children)
)