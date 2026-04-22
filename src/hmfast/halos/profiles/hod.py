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
    General halo occupation distribution (HOD) profile following
    `Zheng et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJ...667..760Z/abstract>`_
    and
    `Zehavi et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...736...59Z/abstract>`_.

    In this model, the real-space galaxy profile is written as

    .. math::

        u_r(r, m, z) = \\frac{1}{\\bar{n}_g(z)}
        \\left[N_{\\mathrm{cen}}(m) + N_{\\mathrm{sat}}(m) \\, u_{\\mathrm{sat}}(r, m, z)\\right]
        \\tag{1}

    where :math:`u_{\\mathrm{sat}}(r, m, z)` is taken to be the NFW satellite
    profile. Central galaxies are naturally assumed to live at the halo center,
    so their real-space density profile is a Dirac delta function, while
    satellite galaxies are assumed to be randomly distributed according to an
    NFW-like radial profile.

    The occupation functions are

    .. math::

        N_{\\mathrm{cen}}(m) = \\frac{1}{2} \\left[1 + \\mathrm{erf}\\left(
        \\frac{\\log_{10} m - \\log_{10} M_{\\mathrm{min}}}{\\sigma_{\\log_{10} M}}
        \\right)\\right]
        \\tag{2}

    .. math::

        N_{\\mathrm{sat}}(m) = H(m - M_0) \\, N_{\\mathrm{cen}}(m)
        \\, \\left(\\frac{m - M_0}{M_1'}\\right)^{\\alpha_s}
        \\tag{3}

    with the power-law term set to zero when :math:`m < M_0`.

    The mean comoving galaxy number density is

    .. math::

        \\bar{n}_g(z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z)
        \\left[N_{\\mathrm{cen}}(M) + N_{\\mathrm{sat}}(M)\\right]
        \\tag{4}

    where :math:`dn / d\\ln M` is the halo model's halo mass function, and the
    large-scale galaxy bias is

    .. math::

        b_g(z) = \\frac{1}{\\bar{n}_g(z)} \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z)
        \\, b^{(1)}_h(M, z) \\left[N_{\\mathrm{cen}}(M) + N_{\\mathrm{sat}}(M)\\right]
        \\tag{5}

    Here :math:`b_h^{(1)}` is the halo model's first-order halo bias.

    Attributes
    ----------
    sigma_log10M : float
        Scatter parameter :math:`\\sigma_{\\log_{10} M}` controlling the width
        of the central-galaxy occupation threshold.
    alpha_s : float
        Power-law slope :math:`\\alpha_s` of the satellite occupation.
    M1_prime : float
        Characteristic satellite mass scale :math:`M_1'` entering the
        normalization of :math:`N_{\\mathrm{sat}}`.
    M_min : float
        Central-occupation threshold mass :math:`M_{\\mathrm{min}}`.
    M0 : float
        Satellite cutoff mass :math:`M_0` below which the satellite occupation
        vanishes.
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
        Expected number of central galaxies in a halo of mass ``m``.

        See Eq. (2) for the explicit form of
        :math:`N_{\\mathrm{cen}}(m)`.

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
        Expected number of satellite galaxies in a halo of mass ``m``.

        See Eq. (3) for the explicit form of
        :math:`N_{\\mathrm{sat}}(m)`.

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
        Comoving mean galaxy number density at redshift ``z``.

        See Eq. (4) for :math:`\\bar{n}_g(z)`.

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
        dndlnm = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        ng_val = jnp.trapezoid(dndlnm * Ntot[:, None], x=logm, axis=0)

        # HM Consistency check
        return jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model._counter_terms(m, z)[0] * Ntot[0], lambda x: x, ng_val)

    def galaxy_bias(self, halo_model, m, z):
        """
        Large-scale galaxy bias at redshift ``z``.

        See Eq. (5) for :math:`b_g(z)`.

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
        dndlnm = halo_model.halo_mass_function.halo_mass_function(halo_model, m, z)
        bh = halo_model.halo_bias.halo_bias(halo_model, m, z, order=1)
        ng = self.ng_bar(halo_model, m, z)

        bg_num = jnp.trapezoid(dndlnm * bh * Ntot[:, None], x=logm, axis=0)
        bg_num = jax.lax.cond(halo_model.hm_consistency, lambda x: x + halo_model._counter_terms(m, z)[1] * Ntot[0], lambda x: x, bg_num)
        return bg_num / ng


    def _sat_and_cen_contribution(self, halo_model, k, m, z):
        """ 
        Compute the satellite and central pieces of the galaxy HOD tracer.
        """

        h = halo_model.cosmology.H0 / 100.0
        m_internal = m * h

        Ns = self.n_sat(m_internal)
        Nc = self.n_cen(m_internal)
        ng = self.ng_bar(halo_model, m_internal, z) * h**3

        _, u_m = self._u_k_matter(halo_model, k, m, z)  

        sat_term = (1/ng) * (Ns[None, :, None] * u_m)
        cen_term = (1/ng) * (Nc[None, :, None]**0)
    
        return sat_term, cen_term


    def u_r(self, halo_model, r, m, z):
        """
        Real-space galaxy HOD profile.

        This evaluates Eq. (1), with
        :math:`u_{\\mathrm{sat}}` identified with the NFW satellite profile.

        Parameters
        ----------
        halo_model : HaloModel
            The parent halo model instance.
        r : float or jnp.ndarray
            Physical radius or radii in the same units as :math:`r_\\Delta`.
        m : float or jnp.ndarray
            Halo mass grid in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift grid.

        Returns
        -------
        jnp.ndarray
            Real-space profile with shape :math:`(N_r, N_M, N_z)`.
        """
        r = jnp.atleast_1d(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        h = halo_model.cosmology.H0 / 100.0
        m_internal = m * h

        Ns = self.n_sat(m_internal)
        Nc = self.n_cen(m_internal)
        ng = self.ng_bar(halo_model, m_internal, z) * h**3

        u_m = self._u_r_matter(halo_model, r, m, z)

        return (1 / ng[None, None, :]) * (Nc[None, :, None] + Ns[None, :, None] * u_m)


    def u_k(self, halo_model, k, m, z):
        """
        Fourier-space galaxy HOD profile.

        This is the Fourier-space analogue of Eq. (1),
        with the satellite term traced by the NFW matter profile in Fourier
        space using the analytic Fourier transform of :math:`u_{\\mathrm{sat}}`.
    
        Parameters
        ----------
        halo_model : HaloModel
            The parent halo model instance.
        k : array-like
            Wavenumber grid.
        m : array-like
            Halo mass grid in physical :math:`M_\\odot`.
        z : array-like
            Redshift grid.

        Returns
        -------
        k : array-like
            Wavenumber grid.
        u_k : array-like
            Fourier-space profile.
        """
       
        h = halo_model.cosmology.H0 / 100.0
        m_internal = m * h

        Ns = self.n_sat(m_internal)
        Nc = self.n_cen(m_internal)
        ng = self.ng_bar(halo_model, m_internal, z) * h**3

        _, u_m = self._u_k_matter(halo_model, k, m, z)
    
        u_k = (1/ng) * (Nc[None, :, None] + Ns[None, :, None] * u_m)
        return u_k
        

jax.tree_util.register_pytree_node(
    StandardGalaxyHODProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: StandardGalaxyHODProfile._tree_unflatten(aux_data, children)
)