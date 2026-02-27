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
    'c_delta': 1.156,

    # Other cosmological parameters
    'T_cmb': 2.7255,
    'deg_ncdm': 1,
    'B': 1.4,

    
    # tSZ tracer-specific parameters
    'P0GNFW': 8.130,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,
    'gammaGNFW': 0.3292,

    
    # HOD tracer-specific parameters
    'sigma_log10M_HOD': 0.68,
    'alpha_s_HOD':    1.30,
    'M1_prime_HOD': 10**12.7, # msun/h
    'M_min_HOD': 10**11.8, # msun/h
    'M0_HOD' :0,



     # CIB tracer-specific parameters
    'L0_cib': 6.4e-8,             # Normalisation of L − M relation in [Jy MPc2/Msun]
    'alpha_cib': 0.36,            # redshift evolution of dust temperature
    'beta_cib': 1.75,             # emissivity index of sed
    'gamma_cib': 1.7,             # Power law index of SED at high frequency
    'T0_cib': 24.4,               # dust temperature today in Kelvins
    'm_eff_cib': 10**12.6,        # Most efficient halo mass in Msun
    'sigma2_LM_cib': 0.5,         # Size of of halo masses sourcing CIB emission
    'delta_cib': 3.6,             # Redshift evolution of L − M normalisation
    'z_plateau_cib': 1e100,       # see 5.2.1 of https://arxiv.org/pdf/1208.5049.pdf
    'M_min_cib': 10**11.5,

    
}


def merge_with_defaults(user_params=None):
    """
    Merge user-supplied parameters with the global defaults.
    """
    merged = DEFAULT_PARAMS.copy()
    if user_params:
        merged.update(user_params)
    return merged


