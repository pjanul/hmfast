import jax
import jax.numpy as jnp
from functools import partial

from hmfast.halo_model.mass_definition import MassDefinition



class ConstantConcentration:
    """
    Constant concentration-mass relation, with the value of c_delta being specified in the parameters.
    """
    def __init__(self, c):
        self.c = c
        pass

    def c_delta(self, halo_model, m, z):
        return jnp.broadcast_to(self.c, (len(jnp.atleast_1d(m)), len(jnp.atleast_1d(z))))



class D08Concentration:
    """
    Duffy et al. (2008) mass-concentration relation.
    A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
    """
    def __init__(self):
        pass


    def c_delta(self, halo_model, m, z):
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        mdef = halo_model.mass_definition

        # Parameter Lookup Table
        coeffs = {
            (200, "critical"):       (5.71, -0.084, -0.47, 2e12),
            (200, "mean"):           (10.14, -0.081, -1.01, 2e12),
            ("vir", "critical"):     (7.85, -0.081, -0.71, 2e12),
        }
        
        # Determine if we have a direct match or need conversion
        key = (mdef.delta, mdef.reference) 
        
        if key in coeffs:
            A, B, C, M_pivot = coeffs[key]
            return A * (m[:, None] / M_pivot)**B * (1 + z[None, :])**C

        if not halo_model.convert_masses:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        # Conversion Logic (Native 200c)
        A, B, C, M_pivot = coeffs[(200, "critical")]
        native_def = MassDefinition(200, "critical")

        c_seed = A * (m[:, None] / M_pivot)**B * (1 + z[None, :])**C
        m_200c = halo_model.convert_m_delta(m, z, mass_def_old=mdef, mass_def_new=native_def, c_old=c_seed)
        
        # Compute r_s from native 200c mesh
        c_200c = A * (m_200c / M_pivot)**B * (1 + z[None, :])**C
        r_200c = jax.vmap(lambda mc, zi: halo_model.r_delta(mc, zi, native_def), (1, 0))(m_200c, z).T
        
        # Final Target Radius / r_s
        r_target = halo_model.r_delta(m, z, mdef)
        return (r_target * c_200c / r_200c).reshape(len(m), len(z))





class B13Concentration:
    """
    Bhattacharya et al. (2013) mass-concentration relation.
    Parameters A, B, C from Table 2 (arXiv:1112.5479).
    """
    def __init__(self):
        pass

    def c_delta(self, halo_model, m, z):
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        mdef = halo_model.mass_definition
        
        # Parameter Lookup Table
        coeffs = {
            (200, "critical"):    (5.9, 0.54, -0.35),
            (200, "mean"):        (9.0, 1.15, -0.29),
            ("vir", "critical"):  (7.7, 0.9,  -0.29),
        }

        key = (mdef.delta, mdef.reference)
        D = halo_model.emulator.growth_factor(z) # Shape (Nz,)

        # Get concentration for a given mass-redshift pair and a set of parameters
        def compute_c(m_val, z_val, D_val, A, B, C):
            nu = (1.12 * (m_val / 5e13)**0.3 + 0.53) / D_val
            return A * D_val**B * nu**C

        # Direct Match Case
        if key in coeffs:
            A, B, C = coeffs[key]
            return compute_c(m[:, None], z[None, :], D[None, :], A, B, C)

        # Conversion Fallback
        if not halo_model.convert_masses:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        # Use 200c as native reference for conversion
        A, B, C = coeffs[(200, "critical")]
        native_def = MassDefinition(200, "critical")

        # c_seed for the solver
        c_seed = compute_c(m[:, None], z[None, :], D[None, :], A, B, C)
        
        m_native = halo_model.convert_m_delta(m, z, mass_def_old=mdef, mass_def_new=native_def, c_old=c_seed)
        
        # Re-compute concentration and scale radius at native definition
        c_native = compute_c(m_native, z[None, :], D[None, :], A, B, C)
        
        r_native = jax.vmap(lambda mc, zi: halo_model.r_delta(mc, zi, mass_definition=native_def), in_axes=(1, 0))(m_native, z).T
        r_s = r_native / c_native

        # Final Target Radius / r_s
        r_target = halo_model.r_delta(m, z, mass_definition=mdef)
        return (r_target / r_s).reshape(len(m), len(z))




class SC14Concentration:
    """
    Sanchez-Conde & Prada (2014) concentration-mass relation.
    Coefficients from Eq. 1 (arXiv:1312.1729). Note: calibrated for 200c.
    """
    def __init__(self):
        pass

    def c_delta(self, halo_model, m, z):
        m, z = jnp.atleast_1d(m), jnp.atleast_1d(z)
        mdef = halo_model.mass_definition

        # 1. Parameter Table (SC14 is only natively defined for 200c)
        # Coefficients in descending order for jnp.polyval: [p5, p4, p3, p2, p1, p0]
        coeffs = {
            (200, "critical"): jnp.array([5.32e-7, -2.89237e-5, 3.66e-4, 1.636e-2, -1.5093, 37.5153])
        }

        key = (mdef.delta, mdef.reference)

        def compute_c(m_val, z_val, p_coeffs):
            # Eq 1: c(M, z=0) = sum(a_i * (log10 M)^i)
            # Then scaled by (1+z)^-1
            logM = jnp.log10(m_val)
            c_z0 = jnp.polyval(p_coeffs, logM)
            return c_z0 * (1 + z_val)**-1

        # Direct Match Case
        if key in coeffs:
            return compute_c(m[:, None], z[None, :], coeffs[key])

        # Conversion Fallback
        if not halo_model.convert_masses:
            raise ValueError(f"Mass definition {key} incompatible with the selected concentration-mass relation.")

        # Use 200c as native reference
        native_coeffs = coeffs[(200, "critical")]
        native_def = MassDefinition(200, "critical")

        # c_seed for the solver
        c_seed = compute_c(m[:, None], z[None, :], native_coeffs)
        
        m_native = halo_model.convert_m_delta(m, z, mass_def_old=mdef, mass_def_new=native_def, c_old=c_seed)
        
        # Re-compute concentration and radii at native definition
        c_native = compute_c(m_native, z[None, :], native_coeffs)
        
        r_native = jax.vmap(lambda mc, zi: halo_model.r_delta(mc, zi, native_def), in_axes=(1, 0))(m_native, z).T
        
        r_s = r_native / c_native

        # Final Target Radius / r_s
        r_target = halo_model.r_delta(m, z, mdef)
        return (r_target / r_s).reshape(len(m), len(z))






        