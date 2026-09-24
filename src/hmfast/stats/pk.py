import functools
from functools import partial

import jax
import jax.numpy as jnp
import mcfit

from hmfast.halos.profiles.profiles_2pt import _fourier_2pt
from hmfast.utils import gauss_legendre_nodes_weights
from . import cl as _cl

# -------------------------
# Halo model power spectrum
# -------------------------

class Pk:
    """
    Halo model power spectrum.

    .. math::

        P(k, z) = P_{1h} + P_{2h}

    where the two terms are built from the generalised halo-model mass
    integral

    .. math::

        I_\\mu^\\beta(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\,|\\, M, z)

    where :math:`\\mu` is the number of profiles/wavenumbers in the
    product, :math:`b_\\beta` is the :math:`\\beta`-th order halo bias
    (:math:`b_0 = 1` unweighted, :math:`b_1` linear), and :math:`u_i` are
    the Fourier-space profiles (first moments). See :meth:`pk_1h` and
    :meth:`pk_2h` for how each term is assembled from
    :math:`I_\\mu^\\beta`.

    Attributes
    ----------
    include_1h : bool
        Whether :meth:`pk_tot` includes the 1-halo term.
    include_2h : bool
        Whether :meth:`pk_tot` includes the 2-halo term.
    k_damp : float
        Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for :meth:`pk_1h`'s
        low-k suppression factor.
    alpha_smooth : float
        HMcode-style exponent smoothing the transition between the 1-halo
        and 2-halo regimes in :meth:`pk_tot`; ``1.0`` recovers a plain sum.
    """

    def __init__(self, include_1h=True, include_2h=True, k_damp=0.01, alpha_smooth=1.0):
        """
        Parameters
        ----------
        include_1h : bool, default True
            Whether :meth:`pk_tot` includes the 1-halo term.
        include_2h : bool, default True
            Whether :meth:`pk_tot` includes the 2-halo term.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for :meth:`pk_1h`'s
            low-k suppression factor.
        alpha_smooth : float, default 1.0
            HMcode-style exponent smoothing the transition between the
            1-halo and 2-halo regimes in :meth:`pk_tot`; ``1.0`` recovers a
            plain sum.
        """
        self.include_1h = include_1h
        self.include_2h = include_2h
        self.k_damp = jnp.asarray(k_damp)
        self.alpha_smooth = jnp.asarray(alpha_smooth)

    def _tree_flatten(self):
        return (self.k_damp, self.alpha_smooth), (self.include_1h, self.include_2h)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.k_damp, obj.alpha_smooth = children
        obj.include_1h, obj.include_2h = aux_data
        return obj

    def _p2xi(self, halo_model):
        """Build the P2xi FFTLog transform (:class:`mcfit.P2xi`) on the halo model's
        own cosmology's tabulated wavenumber grid (:meth:`Cosmology._pk_grid`), rather
        than an independent FFTLog grid -- mcfit needs this instantiated before it can
        be used in a jitted function, so this returns an already-jitted callable."""
        k, _ = halo_model.cosmology._pk_grid()
        return jax.jit(functools.partial(
            mcfit.P2xi(k, lowring=True, backend='jax'),
            axis=0, extrap=False,
        ))

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def pk_1h(self, halo_model, k, z, profile1, profile2=None):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = I_2^0(k, k, z)

        where :math:`I_2^0` is the unweighted (:math:`\\beta=0`) pair
        mass integral :math:`I_\\mu^\\beta` with :math:`\\mu=2`,
        evaluated with both profiles at the same wavenumber :math:`k`.
        The mass integral is performed over :attr:`m_range`/:attr:`n_m`.

        Parameters
        ----------
        halo_model : HaloModel
        k : float or jnp.ndarray
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.

        Returns
        -------
        pk_1h : array
            1-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Setup
        logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
        m = jnp.exp(logm)

        dndlnm = jnp.reshape(hm.halo_mass_function.dndlnm(hm.cosmology, m, z, hm.mass_def), (len(m), len(z)))
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        # Process a single mass bin at a time and extract the uk^2 at the lowest mass for the halo model consistency term
        def process_bin(i):
            pair_kernel = _fourier_2pt(hm, profile1, profile2, k, m, z)
            pair_kernel = jnp.reshape(pair_kernel, (len(k), len(m), len(z)))
            uk_sq_row = pair_kernel[:, i, :]

            return uk_sq_row * total_weights[i], uk_sq_row

        # vmap through the mass bins
        integrand_rows, all_sq_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))

        pk1h = jnp.sum(integrand_rows, axis=0)

        # Apply halo model consistency correction: n_min * uk_sq_min
        uk_sq_min = all_sq_profiles[0]
        n_min, _, _ = hm._counter_terms(z)
        correction = n_min[None, :] * uk_sq_min
        pk1h = pk1h + hm.hm_consistency * correction

        # Apply damping
        mask = self.k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, self.k_damp, 1.0))**2), 1.0)

        return jnp.squeeze(pk1h * damping[:, None])

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def pk_2h(self, halo_model, k, z, profile1, profile2=None):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1^1(k, z) \\, I_1^1(k, z)

        where :math:`I_1^1` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^\\beta` with
        :math:`\\mu=1`, evaluated once per profile at wavenumber
        :math:`k`. The mass integral is performed over :attr:`m_range`/:attr:`n_m`.

        Parameters
        ----------
        halo_model : HaloModel
        k : float or jnp.ndarray
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.

        Returns
        -------
        pk_2h : array
            2-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)

        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Ingredients
        logm, w = gauss_legendre_nodes_weights(jnp.log(hm.m_range[0]), jnp.log(hm.m_range[1]), hm.n_m)
        m = jnp.exp(logm)

        # Combine hmf, bias, and weights into a single (Nm, Nz) weight grid
        dndlnm = jnp.reshape(hm.halo_mass_function.dndlnm(hm.cosmology, m, z, hm.mass_def), (len(m), len(z)))
        bias = jnp.reshape(hm.halo_bias.bias(hm.cosmology, m, z, hm.mass_def), (len(m), len(z)))
        total_weights = dndlnm * bias * w[:, None]

        def get_I(profile):
            # This function processes a single index 'i' of the mass axis
            def process_bin(i):
                uk_full = jnp.reshape(profile.fourier(hm, k, m, z), (len(k), len(m), len(z)))
                uk_slice = uk_full[:, i, :]
                return uk_slice * total_weights[i], uk_slice

            # Vmap over the indices 0...Nm-1, then integrate and pluck index 0 for hm consistency
            integrand_rows, all_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
            integral = jnp.sum(integrand_rows, axis=0)
            u_k_min = all_profiles[0]  # vmap output is (Nm, Nk, Nz)

            n_min, b1_min, _ = hm._counter_terms(z)
            correction = b1_min[None, :] * n_min[None, :] * u_k_min

            return integral + hm.hm_consistency * correction

        # Final Power Spectrum
        I1 = get_I(profile1)
        I2 = I1 if profile1 is profile2 else get_I(profile2)

        P_lin = hm.cosmology.pk(k, z, linear=True)
        # Ensure P_lin has shape (N_k, N_z)
        P_lin = jnp.reshape(P_lin, (len(k), -1))

        return jnp.squeeze(P_lin * I1 * I2)

    # ------------------------------------------------------------------
    # Combined 1-halo + 2-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def pk_tot(self, halo_model, k, z, profile1, profile2=None):
        """
        Combine the 1-halo and 2-halo terms, with an HMcode-style smoothed
        transition between the two regimes.

        .. math::

            P(k, z) = \\left[ P_{1h}(k, z)^\\alpha + P_{2h}(k, z)^\\alpha \\right]^{1/\\alpha}

        where :math:`\\alpha` is :attr:`alpha_smooth`; :math:`\\alpha=1`
        recovers the plain sum :math:`P_{1h} + P_{2h}`. A term excluded via
        :attr:`include_1h`/:attr:`include_2h` is set to zero before combining,
        and since :math:`0^\\alpha = 0` for :math:`\\alpha > 0`, this reduces
        to the other term alone with no separate branch needed.

        Parameters
        ----------
        halo_model : HaloModel
        k : float or jnp.ndarray
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.

        Returns
        -------
        pk_tot : array
            Combined power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        p1h = self.pk_1h(halo_model, k, z, profile1, profile2) if self.include_1h else 0.0
        p2h = self.pk_2h(halo_model, k, z, profile1, profile2) if self.include_2h else 0.0
        a = self.alpha_smooth

        return (p1h**a + p2h**a) ** (1.0 / a)

    # ------------------------------------------------------------------
    # Correlation function (FFTLog transform of pk_tot)
    # ------------------------------------------------------------------

    def xi_hm(self, halo_model, r, z, profile1, profile2=None):
        """
        Halo-model 3D correlation function, combining the 1-halo and 2-halo
        terms according to :attr:`include_1h`/:attr:`include_2h` (and
        :attr:`alpha_smooth`).

        .. math::

            \\xi(r, z) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\, P(k, z)\\, j_0(kr)

        obtained by an FFTLog transform (:class:`mcfit.P2xi`) of
        :meth:`pk_tot`, tabulated on ``halo_model``'s own cosmology's native
        log-spaced ``k`` grid (:meth:`Cosmology._pk_grid`) and interpolated
        onto the requested ``r``. Transforming the already-combined
        :meth:`pk_tot` (rather than transforming each term and summing) is
        what makes this correct when :attr:`alpha_smooth` differs from 1,
        since the FFTLog transform is linear but :meth:`pk_tot`'s combination
        need not be.

        Parameters
        ----------
        halo_model : HaloModel
        r : float or jnp.ndarray
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to ``halo_model``'s cosmology's own
            ``k`` grid; values of ``r`` too close to that range's edges are
            affected by FFTLog ringing.
        z : float or jnp.ndarray
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.

        Returns
        -------
        xi_hm : array
            Halo-model correlation function (dimensionless), with shape
            :math:`(N_r, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

        k, _ = halo_model.cosmology._pk_grid()
        pk = self.pk_tot(halo_model, k, z, profile1, profile2)
        pk = jnp.reshape(pk, (len(k), len(z)))

        r_native, xi_native = self._p2xi(halo_model)(pk)
        ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

        # Linear in xi against ln r rather than log-log
        def interp_col(xi_col):
            return jnp.interp(ln_r, ln_r_native, xi_col)

        xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
        return jnp.squeeze(xi)

    # ------------------------------------------------------------------
    # Angular power spectrum (halo model; Limber, with optional non-Limber 2-halo)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(6,), static_argnames=("l_limber", "n_fft", "n_interp", "bias", "window"))
    def cl_hm(self, halo_model, tracer1, tracer2, l, z_range, n_z, l_limber=0.0,
              z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
        """
        Halo-model angular power spectrum :math:`C_\\ell`, combining the
        1-halo and 2-halo contributions according to :attr:`include_1h`/
        :attr:`include_2h`.

        By default (``l_limber=0.0``) this uses the Limber approximation
        everywhere, which maps each multipole to a wavenumber,
        :math:`k = (\\ell + 1/2)/\\chi`, and sums whichever of the 1-halo/
        2-halo 3D power spectra are selected (the mass integral is performed
        over :attr:`m_range`/:attr:`n_m`). Below `l_limber`, the 2-halo term
        instead uses an exact projection using a SwiftCl-style (`Reymond et
        al. 2025 <https://arxiv.org/abs/2505.22718>`_) FFTLog decomposition,
        built on the exact decomposition of the 2-halo power spectrum

        .. math::

            P_{2h}(k; z_1, z_2) = P_{\\rm lin}(k, z_{\\rm fid})\\,
            D(k, z_1)\\, D(k, z_2),

        where :math:`D(k, z)` is the growth factor. This avoids the double
        line-of-sight integral over oscillatory spherical Bessel functions
        an exact projection would otherwise require. The 1-halo term has no
        non-Limber treatment (it only matters at high :math:`\\ell`, where
        Limber is already accurate), so below `l_limber` only the 2-halo
        term (if :attr:`include_2h`) contributes; a 1-halo-only ``Pk``
        (``include_2h=False``) therefore returns zero below `l_limber`.

        A tracer with an RSD term (e.g. ``GalaxyTracer(rsd=True)``) is
        supported in the Limber branch: its ``der_bessel=2`` kernel term is
        projected via an extended-Limber recipe (see
        :func:`hmfast.stats.cl.cl_limber`), adding roughly 1.7x the cost of
        this function for a pair where at least one tracer has
        ``rsd=True``, and no added cost otherwise.

        Parameters
        ----------
        halo_model : HaloModel
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid. Must be concrete (not a value being traced by
            JAX), since the low/high `l_limber` split is a data-dependent
            shape decision.
        z_range : tuple
            ``(z_min, z_max)`` spanning the redshift integration grid (Limber
            branch) and the internal FFTLog chi grid (non-Limber branch).
        n_z : int
            Number of redshift nodes (static: changing it triggers
            recompilation; sweeping ``z_range`` alone does not).
        l_limber : float, default 0.0
            Multipole threshold: below `l_limber`, the exact non-Limber
            calculation is used for the 2-halo term; at or above it, the
            Limber approximation is used. The default, 0.0, uses Limber
            everywhere.
        z_fid : float, default 0.0
            Used only by the non-Limber calculation. Fiducial redshift at
            which the power spectrum is evaluated in the decomposition
            above.
        n_fft : int or None, default None
            Used only by the non-Limber calculation. Number of FFTLog
            nodes; defaults to ``n_z``.
        n_interp : int, default 200
            Used only by the non-Limber calculation. Number of wavenumber
            points at which its most expensive step is evaluated, before
            interpolating onto the cosmology's own tabulated
            power-spectrum grid.
        bias : float, default 0.1
            Used only by the non-Limber calculation. FFTLog de-trending
            exponent (must be less than 1); the default is robust across
            tracer types and rarely needs changing.
        window : float, default 0.2
            Used only by the non-Limber calculation. Fraction of
            high-frequency FFTLog modes smoothly anti-aliased to suppress
            edge/periodicity ringing; the default is robust across tracer
            types.

        Returns
        -------
        cl_hm : array
            Dimensionless halo-model angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2
        return _cl._dispatch_by_ell(
            l, l_limber,
            lambda l_low: (
                _cl._cl_2h_nonlimber(halo_model, tracer1, tracer2, l_low, z_range, n_z, z_fid=z_fid,
                                      n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)
                if self.include_2h else jnp.zeros_like(jnp.atleast_1d(l_low))
            ),
            lambda l_high: _cl._cl_limber(self, halo_model, tracer1, tracer2, l_high, z_range, n_z,
                                           include_1h=self.include_1h, include_2h=self.include_2h),
        )

    # ------------------------------------------------------------------
    # Angular power spectrum (linear bias)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(6,), static_argnames=("linear", "l_limber", "n_fft", "n_interp", "bias", "window"))
    def cl_lin(self, cosmology, tracer1, tracer2, l, z_range, n_z, linear=True,
               l_limber=0.0, z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
        """
        Angular power spectrum for linearly-biased or unbiased tracers, with
        no halo-model mass integral. Companion to
        :meth:`cl_hm`: uses each tracer's own scalar/array bias
        (from a ``bias`` attribute, e.g. ``GalaxyTracer``; tracers without
        one, e.g. CMB/galaxy lensing, are treated as unbiased) in place of
        the halo-model mass integral, and the raw matter power spectrum in
        place of :meth:`pk_1h`/:meth:`pk_2h`. Takes a ``Cosmology`` directly,
        unlike :meth:`cl_hm`, since no halo-model mass integral
        is performed. Below ``l_limber``, uses an exact SwiftCl-style FFTLog
        projection (mirroring ``cl_hm``'s non-Limber 2-halo branch); at or above it,
        uses the Limber approximation (the default, ``l_limber=0.0``, uses
        Limber everywhere).

        Every ``der_bessel=0`` kernel term (e.g. a ``GalaxyTracer``'s density and
        magnification-bias terms) is summed before multiplying by the tracer's bias.
        An RSD (``der_bessel=2``) term is supported in both branches: below
        ``l_limber`` via the exact FFTLog projection, at or above it via an
        extended-Limber recipe (see :func:`hmfast.stats.cl._cl_linear_limber`).

        Parameters
        ----------
        cosmology : Cosmology
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid. Must be concrete (not a value being traced by JAX),
            since the low/high `l_limber` split is a data-dependent shape
            decision (matches `cl_hm`).
        z_range : tuple
            ``(z_min, z_max)`` spanning the redshift integration grid (Limber
            branch) and the internal FFTLog chi grid (non-Limber branch).
        n_z : int
            Number of redshift nodes (static: changing it triggers
            recompilation; sweeping ``z_range`` alone does not).
        linear : bool, default True
            If True, use the linear matter power spectrum; if False, use the
            nonlinear power spectrum.
        l_limber : float, default 0.0
            Multipole threshold: below `l_limber`, the exact non-Limber
            calculation is used; at or above it, the Limber approximation is
            used. The default, 0.0, uses Limber everywhere.
        z_fid : float, default 0.0
            Used only by the non-Limber calculation. Fiducial redshift at which
            the power spectrum is evaluated in the separable D(k,z) ansatz.
        n_fft : int or None, default None
            Used only by the non-Limber calculation. Number of FFTLog nodes;
            defaults to ``len(z)``.
        n_interp : int, default 200
            Used only by the non-Limber calculation. Number of wavenumber
            points at which its most expensive step is evaluated, before
            interpolating onto the cosmology's own tabulated power-spectrum grid.
        bias : float, default 0.1
            Used only by the non-Limber calculation. FFTLog de-trending
            exponent (must be less than 1); unrelated to a tracer's own bias.
        window : float, default 0.2
            Used only by the non-Limber calculation. Fraction of high-frequency
            FFTLog modes smoothly anti-aliased to suppress edge/periodicity
            ringing.

        Returns
        -------
        cl_lin : array
            Dimensionless linearly-biased angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        return _cl._dispatch_by_ell(
            l, l_limber,
            lambda l_low: _cl._cl_linear_nonlimber(cosmology, tracer1, tracer2, l_low, z_range, n_z, linear=linear,
                                                    z_fid=z_fid, n_fft=n_fft, n_interp=n_interp,
                                                    bias=bias, window=window),
            lambda l_high: _cl._cl_linear_limber(cosmology, tracer1, tracer2, l_high, z_range, n_z, linear=linear),
        )


jax.tree_util.register_pytree_node(
    Pk,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: Pk._tree_unflatten(aux_data, children)
)
