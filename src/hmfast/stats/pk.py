import functools

import jax
import jax.numpy as jnp
import mcfit

from hmfast.halos.profiles.profiles_2pt import _fourier_2pt
from .cl_linear import _cl_linear_limber, _cl_linear_nonlimber
from .nonlimber import _cl_2h_nonlimber


def _raise_if_rsd_in_limber(tracer1, tracer2, l_limber):
    """No Limber correction is implemented for a der_bessel!=0 (e.g. RSD) kernel term in
    ``cl_linear``'s Limber branch (``_cl_linear_limber``), so any such term routed through it
    would silently be dropped rather than projected; raise instead of returning a silently
    incomplete result. (``Pk._cl_limber``, used by ``cl_1h``/``cl_2h``, does not need this guard:
    it projects a der_bessel=2 term directly via CCL's extended-Limber recipe -- see
    ``_effective_kernel_limber`` below.)"""
    for t in (tracer1, tracer2):
        if getattr(t, "rsd", False):
            raise ValueError(
                f"{type(t).__name__} has rsd=True, but l_limber={l_limber} routes some multipoles "
                "through cl_linear's Limber approximation, which has no way to project the "
                "der_bessel=2 RSD term. Raise l_limber to cover your full l range with the exact "
                "non-Limber branch instead, or set rsd=False if you don't need RSD. (This "
                "restriction is specific to cl_linear -- cl_1h/cl_2h support RSD under Limber.)"
            )


def _effective_kernel_limber(pk_obj, hm, tracer, cosmology, z, chi, l, z_b, k_damp,
                              include_1h, include_2h, z_dense, chi_dense, P_grid,
                              profile1, profile2):
    """Reduce one tracer's ``kernel()`` terms to a single effective per-``(z, l)`` Limber kernel.

    A ``der_bessel=0`` term (density, magnification bias, IA, lensing/CMB-lensing convergence --
    anything folded into ``.kernel()``) uses the existing direct substitution, unchanged.

    A ``der_bessel=2`` term (RSD) needs more: there is no simple local substitution for a
    j_l''(k*chi)-weighted term. This follows CCL's own "extended Limber" treatment instead
    (Chisari et al. 2019, Sec. 2.4.1; ``transfer_limber_single`` in CCL's ``src/ccl_cls.c``):
    the spherical-Bessel recursion rewrites j_l'' as a combination of *ordinary* j_l and
    j_{l+1}, each of which has its own perfectly standard Limber peak (at k*chi=l+1/2 and
    k*chi=l+3/2 respectively). Concretely: evaluate this tracer's own weight, and the pair's
    shared halo-model P(k), a *second* time at the same wavenumber k_l=(l+1/2)/chi but the
    *shifted* comoving distance chi_l'=chi_l*(l+3/2)/(l+1/2) (equivalently redshift z'), then
    combine algebraically with CCL's exact coefficients. Verified against CCL's own
    ``has_rsd=True`` Limber C_ell to ~0.1% relative precision.

    Only ever reaches the ``der_bessel=2`` branch for a tracer with ``rsd=True`` (see the
    ``needs_extended`` gate in ``_cl_limber``); every other tracer takes the fast, ``(Nz,)``-
    shaped path below with no added cost at all -- identical output and cost to a plain
    ``tracer.kernel(cosmology, z)`` call.
    """
    terms = tracer.kernel(cosmology, z)
    if all(der_bessel == 0 for _, der_bessel in terms):
        total = jnp.zeros_like(jnp.atleast_1d(z))
        for weight, _ in terms:
            total = total + jnp.atleast_1d(weight)
        return total

    l = jnp.atleast_1d(l)
    lp1h, lp3h = l + 0.5, l + 1.5
    chi_lp = chi[:, None] * (lp3h / lp1h)[None, :]  # (Nz, Nl): same k_l, shifted chi
    z_lp_raw = jnp.interp(chi_lp.reshape(-1), chi_dense, z_dense).reshape(chi_lp.shape)

    # Same z_b clip/growth-rescale treatment _cl_limber already applies to the primary z (the
    # emulator's trained Pk grid ends at z_b; a shifted z' pushed past it is handled the same way)
    # -- except the shift itself can push z' past z_b *even when the caller's own z-grid never
    # would have* (at low l, the shift factor (l+1.5)/(l+0.5) is large), so unlike the primary z
    # (where a caller pairs a wide grid with an extrapolate_z=True cosmology by construction),
    # this can hit a cosmology with extrapolate_z=False. growth_factor then returns NaN rather
    # than extrapolating -- fall back to no growth correction (ratio=1, matching the flat z_b
    # freeze already applied to the Pk evaluation below) instead of propagating that NaN.
    in_bounds_lp = z_lp_raw <= z_b
    z_lp = jnp.where(in_bounds_lp, z_lp_raw, z_b)
    growth_ratio_sq_lp = jnp.nan_to_num(
        jnp.where(in_bounds_lp, 1.0, (cosmology.growth_factor(z_lp_raw) / cosmology.growth_factor(z_b)) ** 2),
        nan=1.0,
    )

    k_l = lp1h[None, :] / chi[:, None]  # (Nz, Nl) -- the SAME k as P_grid, at the shifted z'

    def _pk_pair(k_i, z_i):
        zi = jnp.atleast_1d(z_i)
        p = 0.0
        if include_1h:
            p = p + pk_obj.pk_1h(hm, jnp.atleast_1d(k_i), zi, profile1, profile2, k_damp=k_damp)
        if include_2h:
            p = p + pk_obj.pk_2h(hm, jnp.atleast_1d(k_i), zi, profile1, profile2)
        return jnp.squeeze(p)

    # The one extra cost of this whole mechanism: a second, elementwise (not outer-product) P(k)
    # evaluation over the full (Nz, Nl) grid, at the shifted (k_l, z') pairs -- P_grid (passed in)
    # already *is* the unshifted P(k_l, z) this needs for the ratio below, so it isn't recomputed.
    P_lp = jax.vmap(_pk_pair)(k_l.reshape(-1), z_lp.reshape(-1)).reshape(k_l.shape) * growth_ratio_sq_lp
    pk_ratio = jnp.abs(P_lp / P_grid)
    sqell = jnp.sqrt(lp1h[None, :] * pk_ratio / lp3h[None, :])

    terms_lp = tracer.kernel(cosmology, z_lp.reshape(-1))

    total = jnp.zeros((z.shape[0], l.shape[0]))
    for weight, der_bessel in terms:
        weight = jnp.broadcast_to(jnp.atleast_1d(weight)[:, None], total.shape)
        if der_bessel == 0:
            total = total + weight
        elif der_bessel == 2:
            weight_lp = None
            for w2, db2 in terms_lp:
                if db2 == 2:
                    weight_lp = jnp.reshape(w2, k_l.shape)
                    break
            total = total + (
                sqell * 2 * weight_lp / lp3h[None, :]
                - (0.25 + 2 * l[None, :]) * weight / (lp1h[None, :] ** 2)
            )
        else:
            raise NotImplementedError(
                f"{type(tracer).__name__}.kernel() has a der_bessel={der_bessel} entry; "
                "Pk._cl_limber only knows how to project der_bessel in {0, 2}."
            )
    return total


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
    """

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

    def pk_1h(self, halo_model, k, z, profile1, profile2=None, k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = I_2^0(k, k, z)

        where :math:`I_2^0` is the unweighted (:math:`\\beta=0`) pair
        mass integral :math:`I_\\mu^\\beta` with :math:`\\mu=2`,
        evaluated with both profiles at the same wavenumber :math:`k`.
        The mass integral is performed over :attr:`m_grid`.

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
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for the low-k suppression factor.

        Returns
        -------
        pk_1h : array
            1-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        k, m, z = jnp.atleast_1d(k), hm.m_grid, jnp.atleast_1d(z)
        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Setup
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

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
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, k_damp, 1.0))**2), 1.0)

        return jnp.squeeze(pk1h * damping[:, None])

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def pk_2h(self, halo_model, k, z, profile1, profile2=None):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1^1(k, z) \\, I_1^1(k, z)

        where :math:`I_1^1` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^\\beta` with
        :math:`\\mu=1`, evaluated once per profile at wavenumber
        :math:`k`. The mass integral is performed over :attr:`m_grid`.

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
        k, m, z = jnp.atleast_1d(k), hm.m_grid, jnp.atleast_1d(z)

        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Ingredients
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

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
    # Correlation function (FFTLog transform of pk_1h / pk_2h)
    # ------------------------------------------------------------------

    def xi_1h(self, halo_model, r, z, profile1, profile2=None, k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D correlation function.

        .. math::

            \\xi_{1h}(r, z) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\,
            P_{1h}(k, z)\\, j_0(kr)

        obtained by an FFTLog transform (:class:`mcfit.P2xi`) of
        :meth:`pk_1h`, tabulated on ``halo_model``'s own cosmology's native
        log-spaced ``k`` grid (:meth:`Cosmology._pk_grid`) and interpolated
        onto the requested ``r``.

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
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}`, passed through
            to :meth:`pk_1h`.

        Returns
        -------
        xi_1h : array
            1-halo correlation function (dimensionless), with shape
            :math:`(N_r, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

        k, _ = halo_model.cosmology._pk_grid()
        pk = self.pk_1h(halo_model, k, z, profile1, profile2, k_damp=k_damp)
        pk = jnp.reshape(pk, (len(k), len(z)))

        r_native, xi_native = self._p2xi(halo_model)(pk)
        ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

        # Linear in xi against ln r rather than log-log
        def interp_col(xi_col):
            return jnp.interp(ln_r, ln_r_native, xi_col)

        xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
        return jnp.squeeze(xi)

    def xi_2h(self, halo_model, r, z, profile1, profile2=None):
        """
        Compute the 2-halo contribution to the 3D correlation function.

        .. math::

            \\xi_{2h}(r, z) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\,
            P_{2h}(k, z)\\, j_0(kr)

        obtained by an FFTLog transform (:class:`mcfit.P2xi`) of
        :meth:`pk_2h`, tabulated on ``halo_model``'s own cosmology's native
        log-spaced ``k`` grid (:meth:`Cosmology._pk_grid`) and interpolated
        onto the requested ``r``.

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
        xi_2h : array
            2-halo correlation function (dimensionless), with shape
            :math:`(N_r, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

        k, _ = halo_model.cosmology._pk_grid()
        pk = self.pk_2h(halo_model, k, z, profile1, profile2)
        pk = jnp.reshape(pk, (len(k), len(z)))

        r_native, xi_native = self._p2xi(halo_model)(pk)
        ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

        # Linear in xi against ln r rather than log-log
        def interp_col(xi_col):
            return jnp.interp(ln_r, ln_r_native, xi_col)

        xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
        return jnp.squeeze(xi)

    # ------------------------------------------------------------------
    # Angular power spectrum (Limber projection)
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("include_1h", "include_2h"))
    def _cl_limber(self, halo_model, tracer1, tracer2, l, z, include_1h=False, include_2h=True, k_damp=0.01):
        """
        Limber C_ell for either or both halo terms; shared implementation
        behind :meth:`cl_1h` (``include_1h=True, include_2h=False``) and
        the Limber branch of :meth:`cl_2h` (its default, and its
        ``l >= l_limber`` branch). ``l`` may be traced. Jitted with
        ``self`` static (``Pk`` isn't a registered JAX pytree), so
        repeated calls on the *same* ``Pk`` instance reuse the cached
        compilation.

        Supports a tracer with an RSD (``der_bessel=2``) ``kernel()`` entry (e.g.
        ``GalaxyTracer(rsd=True)``) via CCL's extended-Limber recipe -- see
        ``_effective_kernel_limber``. This adds one extra (Nz, Nl)-shaped P(k) evaluation
        (~1.7x the cost of this function, empirically) *only* for a tracer pair where at least
        one side has ``rsd=True``; every other pair takes the original, unchanged code path.
        """
        hm = halo_model
        cosmology = hm.cosmology
        tracer2 = tracer1 if tracer2 is None else tracer2
        z = jnp.atleast_1d(z)
        l = jnp.atleast_1d(l)
        z_b = cosmology._z_grid_pk()[-1]

        # growth_factor is evaluated on the full z array, not per-scalar inside vmap, to match pk_1h/pk_2h exactly.
        in_bounds = z <= z_b
        growth_ratio_sq = jnp.where(in_bounds, 1.0, (cosmology.growth_factor(z) / cosmology.growth_factor(z_b)) ** 2)
        z_eval = jnp.where(in_bounds, z, z_b)

        def get_pk_slice(zi, zi_eval):
            chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            ki = (l + 0.5) / chi_i
            zi_eval = jnp.atleast_1d(zi_eval)

            p = 0.0
            if include_1h:
                p = p + self.pk_1h(hm, ki, zi_eval, tracer1.profile, tracer2.profile, k_damp=k_damp)
            if include_2h:
                p = p + self.pk_2h(hm, ki, zi_eval, tracer1.profile, tracer2.profile)
            return jnp.atleast_1d(p).flatten()

        P_grid = jax.vmap(get_pk_slice)(z, z_eval) * growth_ratio_sq[:, None]
        chi = cosmology.angular_diameter_distance(z) * (1.0 + z)

        # Only build the dense chi(z) table (needed to invert the RSD term's shifted chi_l' back
        # to a redshift) when a tracer actually carries an RSD term -- keeps every other pair at
        # today's cost exactly (see _effective_kernel_limber's own fast path for the other half
        # of that guarantee).
        needs_extended = getattr(tracer1, "rsd", False) or getattr(tracer2, "rsd", False)
        if needs_extended:
            z_dense = cosmology._z_grid_bg()
            chi_dense = cosmology.angular_diameter_distance(z_dense) * (1.0 + z_dense)
        else:
            z_dense = chi_dense = None

        kernel1 = _effective_kernel_limber(self, hm, tracer1, cosmology, z, chi, l, z_b, k_damp,
                                            include_1h, include_2h, z_dense, chi_dense, P_grid,
                                            tracer1.profile, tracer2.profile)
        kernel2 = _effective_kernel_limber(self, hm, tracer2, cosmology, z, chi, l, z_b, k_damp,
                                            include_1h, include_2h, z_dense, chi_dense, P_grid,
                                            tracer1.profile, tracer2.profile)
        kernel1 = kernel1[:, None] if kernel1.ndim == 1 else kernel1
        kernel2 = kernel2[:, None] if kernel2.ndim == 1 else kernel2

        limber_weight = cosmology.comoving_volume_element(z) / chi**4
        integrand = P_grid * limber_weight[:, None] * kernel1 * kernel2
        return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))

    def cl_1h(self, halo_model, tracer1, tracer2, l, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the angular power spectrum
        :math:`C_\\ell^{1h}` via the Limber approximation, which maps each
        multipole to a wavenumber, :math:`k = (\\ell + 1/2)/\\chi`, and
        integrates the 1-halo 3D power spectrum against the tracer kernels
        (the mass integral is performed over :attr:`m_grid`). A tracer with
        an RSD term (e.g. ``GalaxyTracer(rsd=True)``) is supported here via
        CCL's extended-Limber treatment of its ``der_bessel=2`` kernel term
        (see :func:`_effective_kernel_limber`). No non-Limber treatment is
        offered here, since the 1-halo term only matters at high
        :math:`\\ell`, where Limber is already accurate.

        Parameters
        ----------
        halo_model : HaloModel
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : float or jnp.ndarray
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
        return self._cl_limber(halo_model, tracer1, tracer2, l, z,
                                include_1h=True, include_2h=False, k_damp=k_damp)

    # ------------------------------------------------------------------
    # Angular power spectrum (2-halo term; Limber, with optional non-Limber)
    # ------------------------------------------------------------------

    def cl_2h(self, halo_model, tracer1, tracer2, l, z, l_limber=0.0,
              z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
        """
        Compute the 2-halo contribution to the angular power spectrum
        :math:`C_\\ell^{2h}`. By default this uses the Limber approximation,
        which maps each multipole to a wavenumber,
        :math:`k = (\\ell + 1/2)/\\chi`, at each comoving distance
        :math:`\\chi` along the tracer kernels (the mass integral is
        performed over :attr:`m_grid`). Below `l_limber`, it instead
        performs an exact projection using a SwiftCl-style (`Reymond et al.
        2025 <https://arxiv.org/abs/2505.22718>`_) FFTLog decomposition,
        built on the exact decomposition of the 2-halo power spectrum

        .. math::

            P_{2h}(k; z_1, z_2) = P_{\\rm lin}(k, z_{\\rm fid})\\,
            D(k, z_1)\\, D(k, z_2),

        where :math:`D(k, z)` is the growth factor. This avoids the
        double line-of-sight integral over oscillatory spherical Bessel
        functions an exact projection would otherwise require. Only the
        2-halo term is treated here, since the 1-halo
        term only matters at high :math:`\\ell`, where Limber
        (:meth:`cl_1h`) is already accurate.

        A tracer with an RSD term (e.g. ``GalaxyTracer(rsd=True)``) is
        supported in the Limber branch (unlike ``cl_linear``'s Limber
        branch, which still requires ``rsd=False``): its ``der_bessel=2``
        kernel term is projected via CCL's own extended-Limber recipe (see
        :func:`_effective_kernel_limber`), adding roughly 1.7x the cost of
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
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.
        l_limber : float, default 0.0
            Multipole threshold: below `l_limber`, the exact non-Limber
            calculation is used; at or above it, the Limber approximation
            is used. The default, 0.0, uses Limber everywhere.
        z_fid : float, default 0.0
            Used only by the non-Limber calculation. Fiducial redshift at
            which the power spectrum is evaluated in the decomposition
            above.
        n_fft : int or None, default None
            Used only by the non-Limber calculation. Number of FFTLog
            nodes; defaults to ``len(z)``.
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
        cl_2h : array
            Dimensionless 2-halo angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed
            before return.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
        l_vals = [float(x) for x in l_arr]
        idx_low = [i for i, li in enumerate(l_vals) if li < l_limber]
        idx_high = [i for i, li in enumerate(l_vals) if li >= l_limber]

        result = jnp.zeros(len(l_vals), dtype=jnp.float64)
        if idx_low:
            low_idx = jnp.array(idx_low)
            result = result.at[low_idx].set(jnp.atleast_1d(
                _cl_2h_nonlimber(halo_model, tracer1, tracer2, l_arr[low_idx], z,
                                 z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)))
        if idx_high:
            high_idx = jnp.array(idx_high)
            result = result.at[high_idx].set(jnp.atleast_1d(
                self._cl_limber(halo_model, tracer1, tracer2, l_arr[high_idx], z, include_2h=True)))
        return jnp.squeeze(result)

    # ------------------------------------------------------------------
    # Angular power spectrum (linear bias)
    # ------------------------------------------------------------------

    def cl_linear(self, cosmology, tracer1, tracer2, l, z, linear=True,
                  l_limber=0.0, z_fid=0.0, n_fft=None, n_interp=200, bias=0.1, window=0.2):
        """
        Angular power spectrum for linearly-biased or unbiased tracers, with
        no halo-model mass integral. Companion to
        :meth:`cl_1h`/:meth:`cl_2h`: uses each tracer's own scalar/array bias
        (from a ``bias`` attribute, e.g. ``GalaxyTracer``; tracers without
        one, e.g. CMB/galaxy lensing, are treated as unbiased) in place of
        the halo-model mass integral, and the raw matter power spectrum in
        place of :meth:`pk_1h`/:meth:`pk_2h`. Takes a ``Cosmology`` directly,
        unlike :meth:`cl_1h`/:meth:`cl_2h`, since no halo-model mass integral
        is performed. Below ``l_limber``, uses an exact SwiftCl-style FFTLog
        projection (mirroring ``cl_2h``'s non-Limber branch); at or above it,
        uses the Limber approximation (the default, ``l_limber=0.0``, uses
        Limber everywhere).

        Note: for ``GalaxyTracer``, ``kernel()`` bundles a density term and a
        magnification-bias term; multiplying the whole kernel by its bias is
        exact only when magnification bias is off (the tracer's default
        ``mag_bias`` slope ``s=0.4`` exactly zeroes that term).

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
            decision (matches `cl_2h`).
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift (Limber branch) and the range of the
            internal FFTLog chi grid (non-Limber branch).
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
        cl_linear : array
            Dimensionless linearly-biased angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
        l_vals = [float(x) for x in l_arr]
        idx_low = [i for i, li in enumerate(l_vals) if li < l_limber]
        idx_high = [i for i, li in enumerate(l_vals) if li >= l_limber]

        result = jnp.zeros(len(l_vals), dtype=jnp.float64)
        if idx_low:
            low_idx = jnp.array(idx_low)
            result = result.at[low_idx].set(jnp.atleast_1d(
                _cl_linear_nonlimber(cosmology, tracer1, tracer2, l_arr[low_idx], z,
                                      linear=linear,
                                      z_fid=z_fid, n_fft=n_fft, n_interp=n_interp, bias=bias, window=window)))
        if idx_high:
            _raise_if_rsd_in_limber(tracer1, tracer2, l_limber)
            high_idx = jnp.array(idx_high)
            result = result.at[high_idx].set(jnp.atleast_1d(
                _cl_linear_limber(cosmology, tracer1, tracer2, l_arr[high_idx], z,
                                   linear=linear)))
        return jnp.squeeze(result)
