import jax.numpy as jnp

DEFAULT_PARAMS = {

    # Emulator parameters.
    'omega_b': 0.02246576,
    'omega_cdm': 0.12,
    'H0': 68.0,
    'tau_reio': 0.0544,
    'ln10^{10}A_s': 3.035173309489548,
    'n_s': 0.965,
    'fEDE': 0.1,
    'log10z_c': 3.5,
    'thetai_scf': jnp.pi/2,
    'r': 0.01,
    'm_ncdm': 0.06,
    'N_ur': 3.046,
    'w0_fld': -0.95,           # only for wcdm models

    # Other cosmological parameters
    'T_cmb': 2.7255,
    'deg_ncdm': 1,
    'delta': 500,
    'B': 1.4,

    
    # tSZ tracer-specific parameters
    'P0GNFW': 8.130,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,
    'gammaGNFW': 0.3292,
    'c500': 1.156,

    
    # HOD tracer-specific parameters
    'sigma_log10M_HOD': 0.68,
    'alpha_s_HOD':    1.30,
    'M1_prime_HOD': 10**12.7, # msun/h
    'M_min_HOD': 10**11.8, # msun/h
    'M0_HOD' :0,
    
}


def merge_with_defaults(user_params=None):
    """
    Merge user-supplied parameters with the global defaults.
    """
    merged = DEFAULT_PARAMS.copy()
    if user_params:
        merged.update(user_params)
    return merged


