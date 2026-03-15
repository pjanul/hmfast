import jax
import jax.numpy as jnp
from functools import partial



class ConstantConcentration:
    """
    Constant concentration-mass relation, with the value of c_delta being specified in the parameters.
    """
    def __init__(self, c):
        self.c = c
        pass

    def c_delta(self, halo_model, m, z, params):

        m = jnp.atleast_1d(m)[:, None]  # shape (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # shape (1, Nz)
        
        return jnp.full((m.shape[0], z.shape[1]), self.c)



class D08Concentration:
    """
    Duffy et al. (2008) mass-concentration relation.
    A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
    """
    def __init__(self):
        pass


    def c_delta(self, halo_model, m, z, params):

        m = jnp.atleast_1d(m)[:, None]  # shape (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # shape (1, Nz)
        
        # Probably a prettier way of doing this
        if halo_model.delta == 200 and halo_model.delta_ref == "critical":
            A, B, C, M_pivot = 5.71, -0.084, -0.47, 2e12
        elif halo_model.delta == 200 and halo_model.delta_ref == "mean":
            A, B, C, M_pivot = 10.14, -0.081, -1.01, 2e12
        elif halo_model.delta == "vir":
            A, B, C, M_pivot = 7.85, -0.081, -0.71, 2e12
        else:
            raise ValueError("The c-M relation c_D08 is incompatible with the chosen definiton of delta. You must select from the following: '200c', '200m', 'vir'.")
    
        return A * (m / M_pivot)**B * (1 + z)**C



class B13Concentration:
    """
    Bhattacharya et al. (2013) mass-concentration relation for c200_c.
    Obtained from Table 2, https://arxiv.org/pdf/1112.5479
    D here is the growth factor D(z).
    """


    def __init__(self):
        pass


    def c_delta(self, halo_model, m, z, params):

        m = jnp.atleast_1d(m)[:, None]  # shape (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # shape (1, Nz)
        
        # Use the nu as defined in the B13 paper and pivot mass in Msun/h
        D = halo_model.emulator.growth_factor(z, params=params)
    
        # Probably a prettier way of doing this
        if halo_model.delta == 200 and halo_model.delta_ref == "critical":
            A, B, C = 5.9, 0.54, -0.35
        elif halo_model.delta == 200 and halo_model.delta_ref == "mean":
            A, B, C = 9.0, 1.15, -0.29
        elif halo_model.delta == "vir":
            A, B, C = 7.7, 0.9, -0.29
        else:
            raise ValueError("The c-M relation c_B13 is incompatible with the chosen definiton of delta. You must select from the following: '200c', 200m', 'vir'.")
    
        
        nu = (1.12 * (m / 5e13)**0.3 + 0.53) / D
        c_delta = A * D**B * nu**C
        return c_delta



class SC14Concentration:
    """
    Sanchez-Conde & Prada (2014) concentration-mass relation for c200_c.
    Coefficients are found below Equation 1, https://arxiv.org/pdf/1312.1729
    """
    def __init__(self):
        pass

    
    def c_delta(self, halo_model, m, z, params=None):

        m = jnp.atleast_1d(m)[:, None]  # shape (Nm, 1)
        z = jnp.atleast_1d(z)[None, :]  # shape (1, Nz)
        
        # Coefficients from Eq. 1
        if halo_model.delta == 200 and halo_model.delta_ref == "critical":
            c_array = jnp.array([37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7])
            logM = jnp.log10(m)
            powers = jnp.stack([logM**i for i in range(6)], axis=0)  # (6, Nm, 1)
            c_array_reshaped = c_array[:, None, None]                 # (6, 1, 1)
            poly = jnp.sum(c_array_reshaped * powers, axis=0)        # (Nm, 1)
            c_delta = poly * (1 + z) ** -1                            # (Nm, Nz)
    
        else: 
            raise ValueError("The c-M relation c_SC14 is incompatible with the chosen definiton of delta. You must select from the following: '200c'.")
        return c_delta
        