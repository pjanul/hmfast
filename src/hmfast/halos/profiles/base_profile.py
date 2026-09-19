import functools
import os
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import mcfit
import numpy as np
from jax.scipy.special import sici
from jax.tree_util import register_pytree_node_class


class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """

    def __init__(self, x, nu=0.5):

        # Kept so a profile's pytree aux can carry this object alone, never the raw grid.
        self.x = x
        self._hankel = mcfit.Hankel(x, nu=nu, lowring=True, backend="jax")
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


class HaloProfile(ABC):
    """
    Grandparent halo profile class from which all halo profile classes inherit.

    Child profile classes must implement :meth:`real` and :meth:`fourier`.
    """

    @abstractmethod
    def real(self, halo_model, r, m, z):
        pass

    @abstractmethod
    def fourier(self, halo_model, k, m, z):
        pass

    def _u_k_hankel(self, halo_model, x, r, m, z):
        """
        Hankel-transform a real-space profile sampled on a dimensionless grid.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model passed through to ``real``.
        x : jnp.ndarray
            Dimensionless transform grid.
        r : jnp.ndarray
            Comoving radius grid with shape :math:`(N_x, N_m, N_z)`.
        m : float or jnp.ndarray
            Halo mass(es).
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Native Hankel wavenumbers and transformed profile values with shape
            :math:`(N_k, N_m, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        x = jnp.atleast_1d(x)
        r = jnp.asarray(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)
        # m may be z-independent (N_m,) or already z-dependent (N_m, N_z); normalize to the latter.
        if m.ndim == 1:
            m = jnp.broadcast_to(m[:, None], (m.shape[0], z.shape[0]))
        W_x = jnp.where((x >= x[0]) & (x <= x[-1]), 1.0, 0.0)

        def single_m_z(r_vals, m_val, z_val):
            profile = jnp.squeeze(self.real(halo_model, r_vals, m_val, z_val))
            return profile * x**0.5 * W_x

        hankel_integrand = jax.vmap(
            jax.vmap(single_m_z, in_axes=(1, 0, 0), out_axes=0),
            in_axes=(1, 0, None),
            out_axes=0,
        )(r, m, z)

        k_native, u_k_native = self._hankel.transform(hankel_integrand)
        u_k_native = jnp.swapaxes(u_k_native, 2, 0)
        u_k_native = jnp.swapaxes(u_k_native, 2, 1)

        return k_native, jnp.squeeze(u_k_native)

    def _fourier_via_hankel_transform(self, halo_model, k, m, z, r_scale):
        """
        Generic Fourier-space transform via a Hankel transform of ``self.real()``,
        interpolated against the dimensionless wavenumber :math:`q = k\\,r_{\\rm scale}\\,(1+z)`,
        with an analytic :math:`q \\to 0` anchor from direct real-space integration.

        Parameters
        ----------
        halo_model : HaloModel
            Passed through to ``self.real()`` and ``self._u_k_hankel()``.
        k, m, z : jnp.ndarray
            Already ``atleast_1d``'d wavenumber, mass, and redshift grids.
        r_scale : jnp.ndarray
            Characteristic radius (e.g. :math:`r_\\Delta`, :math:`r_s`, :math:`r_{\\rm vir}`)
            with shape :math:`(N_m, N_z)`, defining both the real-space sampling grid
            and the q-space rescaling.

        Returns
        -------
        jnp.ndarray
            Fourier-space profile with shape :math:`(N_k, N_m, N_z)`, where singleton
            dimensions get squeezed before return.
        """
        r = self.x_grid[:, None, None] * r_scale[None, :, :] * (1.0 + z[None, None, :])
        real_profile = jnp.reshape(self.real(halo_model, r, m, z), (len(self.x_grid), len(m), len(z)))

        k_native, u_k_native = self._u_k_hankel(halo_model, self.x_grid, r, m, z)
        u_k_native = jnp.reshape(u_k_native, (len(k_native), len(m), len(z)))

        q_native = jnp.broadcast_to(k_native[:, None, None], (len(k_native), len(m), len(z)))
        q_target = k[:, None, None] * r_scale[None, :, :] * (1.0 + z[None, None, :])
        prefactor = 4.0 * jnp.pi * r_scale**3 * (1.0 + z)[None, :] ** 3

        u_k_val = prefactor[None, :, :] * u_k_native * jnp.sqrt(jnp.pi / (2.0 * q_native))
        u_k_zero = prefactor * jnp.trapezoid(self.x_grid[:, None, None] ** 2 * real_profile, x=self.x_grid, axis=0)

        q_native = jnp.concatenate([jnp.zeros((1, len(m), len(z))), q_native], axis=0)
        u_k_val = jnp.concatenate([u_k_zero[None, :, :], u_k_val], axis=0)

        def interp_at_col(q_t, q_n, u_n):
            return jnp.interp(jnp.log(q_t), jnp.log(q_n[1:]), u_n[1:], left=u_n[0])

        q_target_cols = jnp.transpose(q_target, (1, 2, 0))
        q_native_cols = jnp.transpose(q_native, (1, 2, 0))
        u_k_cols = jnp.transpose(u_k_val, (1, 2, 0))

        vmap_interp = jax.vmap(jax.vmap(interp_at_col, in_axes=(0, 0, 0), out_axes=0), in_axes=(0, 0, 0), out_axes=0)

        u_interp = vmap_interp(q_target_cols, q_native_cols, u_k_cols)
        return jnp.squeeze(jnp.transpose(u_interp, (2, 0, 1)))

    def _u_r_nfw(self, halo_model, r, m, z):
        """
        Calculate the normalized real-space NFW matter profile.

        This is the real-space analogue of ``_u_k_nfw`` and returns the
        unit-mass NFW profile sampled on a radial grid. The profile is
        truncated at :math:`r_\Delta`, matching the standard finite-mass NFW
        convention assumed by the Fourier-space helper.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the concentration relation and mass definition.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass(es) in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Normalized real-space profile with shape :math:`(N_r, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """
        r = jnp.atleast_1d(r)
        m = jnp.atleast_1d(m)
        z = jnp.atleast_1d(z)

        c_delta = jnp.reshape(
            halo_model.concentration.c_delta(
                halo_model.cosmology,
                m,
                z,
                mass_def=halo_model.mass_def,
            ),
            (len(m), len(z)),
        )
        r_delta = jnp.reshape(
            halo_model.mass_def.r_delta(halo_model.cosmology, m, z), (len(m), len(z))
        )
        r_s = r_delta * (1.0 + z[None, :]) / c_delta

        f_nfw = 1.0 / (jnp.log1p(c_delta) - c_delta / (1.0 + c_delta))
        x = r[:, None, None] / r_s[None, :, :]
        prefactor = 1.0 / (4.0 * jnp.pi * r_s**3)
        c_delta_b = c_delta[None, :, :]
        # NaN-safe: a bare `x <= c_delta_b` would silently zero out NaN instead of propagating it.
        inside_halo = jnp.isnan(x) | jnp.isnan(c_delta_b) | (x <= c_delta_b)

        profile = prefactor[None, :, :] * f_nfw[None, :, :] / (x * (1.0 + x) ** 2)
        return jnp.squeeze(jnp.where(inside_halo, profile, 0.0))

    def _u_k_nfw(self, halo_model, k, m, z):
        """
        Calculate :math:`u^m(k, M, z)` for wavenumbers in :math:`\\mathrm{Mpc}^{-1}`
        supporting independent dimensions for ``k``, ``m``, and ``z``.

        Returns
        -------
        jnp.ndarray
            Fourier-space matter profile with shape :math:`(N_k, N_m, N_z)`,
            where singleton dimensions get squeezed before return.
        """

        # Ensure all inputs are 1D arrays
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)

        # Get c_delta and r_delta
        c_delta = jnp.reshape(
            halo_model.concentration.c_delta(
                halo_model.cosmology,
                m,
                z,
                mass_def=halo_model.mass_def,
            ),
            (len(m), len(z)),
        )
        r_delta = jnp.reshape(
            halo_model.mass_def.r_delta(halo_model.cosmology, m, z), (len(m), len(z))
        )
        lambda_val = 1.0

        # Compute analytical profile q terms with shape: (N_k, N_m, N_z)
        q = (
            k[:, None, None]
            * r_delta[None, :, :]
            / c_delta[None, :, :]
            * (1 + z[None, None, :])
        )
        q_scaled = (1 + lambda_val * c_delta[None, :, :]) * q

        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)

        # NFW normalization
        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x / (1 + x))
        f_nfw_val = f_nfw(lambda_val * c_delta)
        f_nfw_val = f_nfw_val[None, :, :]

        # Fourier-space profile calculation
        u_k_m = (
            jnp.cos(q) * (Ci_q_scaled - Ci_q)
            + jnp.sin(q) * (Si_q_scaled - Si_q)
            - jnp.sin(lambda_val * c_delta[None, :, :] * q) / q_scaled
        ) * f_nfw_val

        return k, jnp.squeeze(u_k_m)
