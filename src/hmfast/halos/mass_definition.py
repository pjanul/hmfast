import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial

from hmfast.utils import newton_root

class MassDefinition:
    """
    Mass definition for halos specified by an overdensity threshold and a reference density.

    For example, :math:`M_{200c}` corresponds to
    ``MassDefinition(delta=200, reference="critical")``, while
    :math:`M_{200m}` corresponds to
    ``MassDefinition(delta=200, reference="mean")``. The special value
    ``delta='vir'`` denotes the redshift-dependent virial overdensity and can
    only be used with ``reference='critical'``.

    Attributes
    ----------
    delta : int, float, or str
        Overdensity threshold used to define the halo boundary. This can be a numeric value such as ``200`` or ``500``, or the string ``'vir'`` for the redshift-dependent virial overdensity. The value ``'vir'`` is only valid with ``reference='critical'``.
    reference : str
        Reference density associated with ``delta``, either ``'critical'`` or ``'mean'``.

    Raises
    ------
    ValueError
        If an invalid combination of `delta` and `reference` is provided, or if either
        parameter is set to an unsupported value.
    """

    def __init__(self, delta=200, reference="critical"):
        self._delta = None
        self._reference = None
        self.reference = reference
        self.delta = delta
        
    # Ensure that reference is only ever critical or mean
    @property
    def reference(self):
        return self._reference
    
    @reference.setter
    def reference(self, value):
        value = str(value).lower()
        if value not in ("critical", "mean"):
            raise ValueError("reference must be either 'critical' or 'mean'")
            
        # Prevent changing reference if delta == "vir"
        if getattr(self, "_delta", None) == "vir" and value != "critical":
            raise ValueError("'vir' is only allowed with 'critical' reference")
        self._reference = value

        
    @property
    def delta(self):
        return self._delta

        
    @delta.setter
    def delta(self, value):
        if isinstance(value, str):
            value = value.lower()
            
        # If 'vir', reference must be 'critical'
        if value == "vir":
            if getattr(self, "_reference", None) != "critical":
                raise ValueError("'vir' is only allowed with 'critical' reference")
            self._delta = value
            return

        # Otherwise, it must be numeric
        if isinstance(value, (int, float)):
            self._delta = value
            return

        raise ValueError("delta must be numeric or 'vir'")


    def _tree_flatten(self):
        # delta can be a tracer (numeric) or a static string ('vir')
        # reference is always a static string for critical/mean
        children = () 
        aux_data = (self.delta, self.reference) 
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*aux_data)

    def _delta_vir_to_crit(self, cosmology, z):
        """
        Compute the virial overdensity with respect to the critical density.
        """
        omega_m = cosmology.omega_m(z)
        x = omega_m - 1.0
        return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2

    def _delta_numeric(self, cosmology, z):
        """
        Return the numeric overdensity threshold at redshift ``z``.
        """
        if self.delta == "vir":
            if self.reference != "critical":
                raise ValueError("virial overdensity only defined w.r.t. critical density")
            return self._delta_vir_to_crit(cosmology, z)

        return self.delta

    def _convert_reference(self, cosmology, z, delta, from_ref="critical", to_ref="mean"):
        """
        Convert an overdensity threshold between critical and mean references.
        """
        z = jnp.asarray(z)
        delta = jnp.asarray(delta)

        if from_ref == to_ref:
            return jnp.broadcast_to(delta, z.shape)

        omega_m = cosmology.omega_m(z)
        if from_ref == "critical" and to_ref == "mean":
            return delta / omega_m
        if from_ref == "mean" and to_ref == "critical":
            return delta * omega_m

        raise ValueError("from_ref and to_ref must be 'critical' or 'mean'")

    @partial(jax.jit, static_argnums=(0,))
    def r_delta(self, cosmology, m, z):
        """
        Compute the halo radius :math:`r_\\Delta` associated with a halo mass.

        .. math::

            r_\\Delta = \\left[\\frac{3M}{4\\pi \\Delta \\rho_{\\mathrm{ref}}(z)}\\right]^{1/3}

        Parameters
        ----------
        cosmology : Cosmology
            Cosmology object used to evaluate the reference density.
        m : float or array-like
            Halo mass enclosed within the overdensity radius, in
            :math:`M_\\odot`.
        z : float or array-like
            Redshift at which to compute the radius.

        Returns
        -------
        float or array-like
            Radius :math:`r_\\Delta` within which the mean enclosed density is
            :math:`\\Delta \\rho_{\\mathrm{ref}}(z)`, in physical :math:`\\mathrm{Mpc}`.
            With shape :math:`(N_m, N_z)`, where singleton dimensions get
            squeezed before return.
        """
        delta, reference = self.delta, self.reference

        m = jnp.asarray(m)
        z = jnp.asarray(z)

        if m.ndim <= 1 and z.ndim <= 1:
            m, z = jnp.atleast_1d(m)[:, None], jnp.atleast_1d(z)[None, :]

        rho_ref = cosmology.critical_density(z)

        if delta == "vir":
            delta = self._delta_vir_to_crit(cosmology, z)

        if reference == "mean":
            rho_ref *= cosmology.omega_m(z)

        return jnp.squeeze((3.0 * m / (4.0 * jnp.pi * delta * rho_ref)) ** (1.0 / 3.0))


jax.tree_util.register_pytree_node(
    MassDefinition,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: MassDefinition._tree_unflatten(aux_data, children)
)


@partial(jax.jit, static_argnums=(3, 4, 6))
def convert_m_delta(cosmology, m, z, mass_def_old, mass_def_new, c_old, max_iter=20):
    """
    Convert halo masses between two spherical-overdensity definitions.

    The conversion assumes an NFW profile and requires the input concentration
    :math:`c_{\\Delta}` for the original mass definition.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object used to evaluate :math:`\\Omega_m(z)` for reference-density
        conversions and virial overdensities.
    m : array-like
        Halo mass in the original definition, :math:`M_{\\Delta}`, in
        physical :math:`M_\\odot`.
    z : array-like
        Redshift(s).
    mass_def_old : MassDefinition
        Original mass definition specifying :math:`\\Delta` and its reference density.
    mass_def_new : MassDefinition
        Target mass definition specifying :math:`\\Delta'` and its reference density.
    c_old : array-like
        Halo concentration :math:`c_{\\Delta}` in the original definition,
        evaluated for the input halo masses in physical :math:`M_\\odot`.
    max_iter : int, optional
        Maximum number of root-finder iterations.

    Returns
    -------
    float or array-like
        Halo mass in the target definition, :math:`M_{\\Delta'}`, in physical
        :math:`M_\\odot`, with shape :math:`(N_m, N_z)`, where singleton
        dimensions get squeezed before return.
    """
    m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
    c_old = jnp.atleast_2d(c_old)
    nm, nz = len(m), len(z)

    def get_delta_crit(mdef, z_val):
        delta = mdef._delta_numeric(cosmology, z_val)
        return mdef._convert_reference(cosmology, z_val, delta, from_ref=mdef.reference, to_ref="critical")

    d_old_z = get_delta_crit(mass_def_old, z)
    d_new_z = get_delta_crit(mass_def_new, z)
    is_same_z = jnp.isclose(d_old_z, d_new_z) & (mass_def_old.reference == mass_def_new.reference)

    mm, zz = jnp.meshgrid(m, z, indexing='ij')
    c_old = c_old[:nm, :nz].reshape(mm.shape)
    x0 = m[:, None] * (d_old_z / d_new_z)[None, :] ** 0.2

    def solve_single(m_i, c_i, x0_i, d_o, d_n, same_flag):
        f_nfw = lambda x: jnp.log1p(x) - x / (1.0 + x)
        obj = lambda m_new: m_i / m_new - f_nfw(c_i) / f_nfw(c_i * (m_new / m_i * d_o / d_n) ** (1 / 3))

        return jax.lax.cond(
            same_flag,
            lambda _: m_i,
            lambda _: newton_root(obj, x0=x0_i, max_iter=max_iter),
            None,
        )

    d_o_flat = jnp.broadcast_to(d_old_z[None, :], mm.shape).flatten()
    d_n_flat = jnp.broadcast_to(d_new_z[None, :], mm.shape).flatten()
    same_flat = jnp.broadcast_to(is_same_z[None, :], mm.shape).flatten()

    results = jax.vmap(solve_single)(mm.flatten(), c_old.flatten(), x0.flatten(), d_o_flat, d_n_flat, same_flat)
    return jnp.squeeze(results.reshape(mm.shape))


