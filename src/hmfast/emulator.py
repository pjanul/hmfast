import os
import jax
import jax.numpy as jnp
from typing import Dict, Union
from hmfast.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path
from hmfast.utils import Const
from jax.tree_util import register_pytree_node_class
from functools import partial

jax.config.update("jax_enable_x64", True)


_COSMO_MODELS = {
    0: {"suffix": "v1", "subdir": "lcdm"},
    1: {"suffix": "mnu_v1", "subdir": "mnu"},
    2: {"suffix": "neff_v1", "subdir": "neff"},
    3: {"suffix": "w_v1", "subdir": "wcdm"},
    4: {"suffix": "v1", "subdir": "ede"},
    5: {"suffix": "v1", "subdir": "mnu-3states"},
    6: {"suffix": "v2", "subdir": "ede"},
}



@register_pytree_node_class
class Emulator:
    def __init__(self, cosmo_model=0, 
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, ln1e10A_s=3.035173309489548, n_s=0.965, tau_reio=0.0544,      # LCDM
                 m_ncdm=0.06, N_ur=3.046, w0_fld=-0.95,                                                                     # wCDM, Neff, MNU
                 fEDE=0.1, log10z_c=3.5, thetai_scf=jnp.pi/2, r=0.01,                                                       # EDE
                 T_cmb=2.7255, deg_ncdm=1.0,                                                                                # Non-emulator 
        ):
        
        # Static Metadata
        self.cosmo_model = cosmo_model
        self._emu = {}  # This will be treated as static

        # Cosmological params (leaves) to be changed without recompiling jit
        self.H0, self.omega_cdm, self.omega_b, self.ln1e10A_s, self.n_s, self.tau_reio = H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio
        self.m_ncdm, self.N_ur, self.w0_fld = m_ncdm, N_ur, w0_fld
        self.fEDE, self.log10z_c, self.thetai_scf, self.r = fEDE, log10z_c, thetai_scf, r
        self.T_cmb, self.deg_ncdm = T_cmb, deg_ncdm


    # ------------------------------------------------------------------
    # PyTree registration
    # ------------------------------------------------------------------

    def tree_flatten(self):
        # 1. Children: Only the 15 numerical parameters JAX should "see"
        children = (
            self.H0, self.omega_cdm, self.omega_b, self.ln1e10A_s, self.n_s, self.tau_reio,
            self.m_ncdm, self.N_ur, self.w0_fld, 
            self.fEDE, self.log10z_c, self.thetai_scf, self.r,
            self.T_cmb, self.deg_ncdm
        )
        # 2. Aux data: Static metadata (Model ID and the Cache dictionary)
        aux_data = (self.cosmo_model, self._emu)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Reconstruct using the static metadata
        cosmo_model, _emu = aux_data
        
        # We bypass __init__ to avoid re-triggering the Loader logic
        obj = cls.__new__(cls)
        obj.cosmo_model = cosmo_model
        obj._emu = _emu 
        
        # Assign the 15 parameter children to the object
        (obj.H0, obj.omega_cdm, obj.omega_b, obj.ln1e10A_s, obj.n_s, obj.tau_reio,
         obj.m_ncdm, obj.N_ur, obj.w0_fld, 
         obj.fEDE, obj.log10z_c, obj.thetai_scf, obj.r,
         obj.T_cmb, obj.deg_ncdm) = children
        
        return obj
    
    def update_params(self, **kwargs):
        names = [
            'H0', 'omega_cdm', 'omega_b', 'ln1e10A_s', 'n_s', 'tau_reio',
            'm_ncdm', 'N_ur', 'w0_fld', 
            'fEDE', 'log10z_c', 'thetai_scf', 'r',
            'T_cmb', 'deg_ncdm'
        ]
        
        invalid = set(kwargs) - set(names)
        if invalid:
            raise ValueError(f"Invalid Emulator parameter(s): {invalid}")
    
        # Flatten the current instance
        leaves, treedef = jax.tree_util.tree_flatten(self)
        
        # Update the parameter list
        new_leaves = [kwargs.get(name, val) for name, val in zip(names, leaves)]
        
        # Unflatten returns a new instance with the SAME aux_data (the cache)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
        
    # ------------------------------------------------------------------
    # atomic lazy loader (Python-side only)
    # ------------------------------------------------------------------

    def _base_path(self):
        return os.path.join(get_default_data_path(),_COSMO_MODELS[self.cosmo_model]["subdir"])
                         

    def _load_emulator(self, key: str):
        if key in self._emu:
            return self._emu[key]
    
        key_map = {
            "DAZ":  ("growth-and-distances", EmulatorLoader),
            "HZ":   ("growth-and-distances", EmulatorLoader),
            "S8Z":  ("growth-and-distances", EmulatorLoader),
            "PKL":  ("PK", EmulatorLoader),
            "PKNL": ("PK", EmulatorLoader),
            "TT":   ("TTTEEE", EmulatorLoader),
            "EE":   ("TTTEEE", EmulatorLoader),
            "TE":   ("TTTEEE", EmulatorLoaderPCA),
            "PP":   ("PP", EmulatorLoader),
            "DER":  ("derived-parameters", EmulatorLoader),
        }
    
        try:
            subdir, loader_cls = key_map[key]
        except KeyError:
            raise KeyError(f"Unknown emulator key: {key}")
    
        self._emu[key] = loader_cls(os.path.join(self._base_path(), subdir, f"{key}_{_COSMO_MODELS[self.cosmo_model]['suffix']}"))
        return self._emu[key]


    def _to_dict(self):
        """
        Converts the class attributes into a dictionary format 
        required by the underlying emulator predictions.
        """
        return {
            'H0': self.H0,
            'omega_cdm': self.omega_cdm,
            'omega_b': self.omega_b,
            'ln10^{10}A_s': self.ln1e10A_s,  # Mapping your attribute to emulator key
            'n_s': self.n_s,
            'tau_reio': self.tau_reio,
            'm_ncdm': self.m_ncdm,
            'N_ur': self.N_ur,
            'w0_fld': self.w0_fld,
            'fEDE': self.fEDE,
            'log10z_c': self.log10z_c,
            'thetai_scf': self.thetai_scf,
            'r': self.r,
            'T_cmb': self.T_cmb,
            'deg_ncdm': self.deg_ncdm
        }
                       

    # ------------------------------------------------------------------
    # shared grids 
    # ------------------------------------------------------------------

    def _z_grid_bg(self):
        return jnp.linspace(0.0, 20.0, 5000, dtype=jnp.float64) 

    def _z_grid_pk(self):
        z_max = jnp.where(self.cosmo_model == 6, 20.0, 5.0)
        return jnp.linspace(0.0, z_max, 100, dtype=jnp.float64)     # z grid for Pk(z)

    def _pk_grid(self):
        is_ede_v2 = (self.cosmo_model == 6)
        k_min = 5e-4 if is_ede_v2 else 1e-4
        k_max = 10.0 if is_ede_v2 else 50.0

        n_downsample_k = 1 if is_ede_v2 else 10
        n_k            = 1000 if is_ede_v2 else 5000
        _k_grid = jnp.geomspace(k_min, k_max, n_k, dtype=jnp.float64)[::n_downsample_k]

        if is_ede_v2:
            _pk_power_fac = _k_grid ** (-3)
        else:
            ls = jnp.arange(2,n_k+2)[::n_downsample_k] 
            _pk_power_fac = (ls*(ls+1.)/2./jnp.pi)**-1

        return _k_grid, _pk_power_fac


    # ------------------------------------------------------------------
    # JAX-safe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interp_z(z, z_grid, values):
        z = jnp.atleast_1d(z)
        out = jnp.interp(z, z_grid, values, left=jnp.nan, right=jnp.nan)
        return out[0] if out.shape[0] == 1 else out


    # ------------------------------------------------------------------
    # Cosmology
    # ------------------------------------------------------------------

    @jax.jit
    def hubble_parameter(self, z):
        """
        Get Hubble parameter at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        
        Returns
        -------
        jnp.ndarray
            Hubble parameter(s) in Mpc^(-1)
        """
        params = self._to_dict()
        emu = self._load_emulator("HZ")
        preds = 10.0 ** emu.predictions(params)
        return self._interp_z(z, self._z_grid_bg(), preds)

    @jax.jit
    def angular_diameter_distance(self, z):
        """
        Get angular diameter distance at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
            
        Returns
        -------
        jnp.ndarray
            Angular diameter distance(s) in Mpc
        """
        
        params = self._to_dict()
        emu = self._load_emulator("DAZ")
        preds = emu.predictions(params)

        if self.cosmo_model == 6:
            preds = 10.0 ** preds
            preds = jnp.insert(preds, 0, 0.0)

        return self._interp_z(z, self._z_grid_bg(), preds)

    def sigma8(self, z):
        """
        Get sigma8 at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
            
        Returns
        -------
        jnp.ndarray
            sigma8 value(s)
        """

        params = self._to_dict()
        emu = self._load_emulator("S8Z")
        preds = emu.predictions(params)
        return self._interp_z(z, self._z_grid_bg(), preds)


    def get_all_cosmo_params(self):
        """
        Get all relevant cosmological parameters.
                
        Returns
        -------
        dict
            Dictionary with all derived cosmological parameters
        """
    
        p = self._to_dict()

        c, G, M_sun, sigma_B, Mpc_over_m = Const._c_, Const._G_, Const._M_sun_, Const._sigma_B_, Const._Mpc_over_m_

        # From user-defined parameters (or defaults if none are defined)
        p['h'] = p['H0']/100.
        p['Omega_b'] = p['omega_b'] / p['h']**2.
        p['Omega_cdm'] = p['omega_cdm'] / p['h']**2.
        
        # More cosmological params
        p['Omega0_g'] = (4. * sigma_B / c * p['T_cmb']**4.) / (3.0 * c**2 * 1e10 * p['h']**2 / Mpc_over_m**2 /8.0 / jnp.pi / G)
        p['Omega0_ur'] = p['N_ur']* 7.0/8.0 * (4.0/11.0)**(4.0/3.0) * p['Omega0_g']
        p['Omega0_ncdm'] = p['deg_ncdm'] * p['m_ncdm'] / (93.14 * p['h']**2) ## valid only in standard cases, default T_ncdm etc
        p['Omega_Lambda'] = 1. - p['Omega0_g'] - p['Omega_b'] - p['Omega_cdm'] - p['Omega0_ncdm'] - p['Omega0_ur']
        p['Omega0_m'] = p['Omega_cdm'] + p['Omega_b'] + p['Omega0_ncdm']
        p['Omega0_r'] = p['Omega0_ur']+p['Omega0_g']
        p['Omega0_m_nonu'] = p['Omega0_m'] - p['Omega0_ncdm']
        p['Omega0_cb'] = p['Omega0_m_nonu'] 

        # Critical density
        H0 = p['H0'] / (c / 1e3) # Convert to H0 over c (c being in km/s)
        p['Rho_crit_0'] = (3.0 / (8.0 * jnp.pi * G * M_sun)) * Mpc_over_m * c**2 * H0**2 / p['h']**2
        
        return p

    
    def critical_density(self, z):
        """
        Get critical density at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
            
        Returns
        -------
        jnp.ndarray
            Critical density(s) in (Msun/h) / (Mpc/h)^3
        """
        
        
        # Get Hubble parameter    
        H_z = self.hubble_parameter(z)
        h = self.H0/100
        
        # Get critical density rho_crit = 3 H^2 / (8 pi G) * Mpc_over_m * c**2 
        c, G, M_sun, sigma_B, Mpc_over_m = Const._c_, Const._G_, Const._M_sun_, Const._sigma_B_, Const._Mpc_over_m_
        rho_crit_factor = (3.0 / (8.0 * jnp.pi * G * M_sun)) * Mpc_over_m * c**2 
        
        return rho_crit_factor * (H_z/h)**2 
        

    def omega_m(self, z):
        """
        Compute Ω_m(z) = rho_m(z) / rho_crit(z) without neutrinos.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
    
        Returns
        -------
        omega_m : float or array
            Dimensionless matter density at redshift z
        """
       
        params = self.get_all_cosmo_params()
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        
        return Omega_m_z


    def growth_factor(self, z):
        """
        Linear growth factor D(z), normalized to D(0)=1.
        """
        
        z = jnp.atleast_1d(z)
    
        k0 = 1e-2  # reference wavenumber
        z_grid_pk = self._z_grid_pk()
        pk0_grid = jax.vmap(lambda zp: jnp.interp(k0, *self.pk_matter(zp, linear=True)))(z_grid_pk)
        D_grid = jnp.sqrt(pk0_grid / jnp.interp(k0, *self.pk_matter(0.0, linear=True)))
    
        return jnp.interp(z, z_grid_pk, D_grid)


    def growth_rate(self, z):
        """
        Return the linear growth rate f(z) = d ln D / d ln a.
        """
        
        z = jnp.atleast_1d(z)

        z_grid_pk = self._z_grid_pk()
        D_grid = self.growth_factor(z_grid_pk)
        a_grid = 1.0 / (1.0 + z_grid_pk)
        f_grid = jnp.gradient(jnp.log(D_grid), jnp.log(a_grid))
        
        return jnp.interp(z, z_grid_pk, f_grid)



    def v_rms_squared(self, z):
        """
        v_rms^2(z) from linear growth factor and matter power spectrum.
        """
        
        z = jnp.atleast_1d(z)
        k_grid = jnp.geomspace(1e-5, 1e1, 1000)
        z_grid_pk = self._z_grid_pk()
    
        # P(k, z) on the pk grid
        P_grid = jax.vmap(lambda zp: jnp.interp(k_grid, *self.pk_matter(zp, linear=True)))(z_grid_pk)
    
        a_grid = 1.0 / (1.0 + z_grid_pk)
        H_grid = self.hubble_parameter(z_grid_pk)
        f_grid = self.growth_rate(z_grid_pk)
    
        W_grid = f_grid * a_grid * H_grid
        integrand = (W_grid[:, None]**2 / 3) * P_grid * k_grid / (2 * jnp.pi**2)
        vrms2_grid = jax.scipy.integrate.trapezoid(integrand, x=jnp.log(k_grid), axis=1)
    
        return jnp.interp(z, z_grid_pk, vrms2_grid)



    def comoving_volume_element(self, z):
        """
        Comoving volume element per unit redshift and solid angle.

        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters

        Returns
        -------
        float
            dV / dz / dOmega in (Mpc/h)^3 / sr
        """
        
        h = self.H0 / 100
        dAz = self.angular_diameter_distance(z) * h
        Hz = self.hubble_parameter(z) / h  # in Mpc^(-1) h

        return (1 + z)**2 * dAz**2 / Hz
   

    # ------------------------------------------------------------------
    # Matter power spectra
    # ------------------------------------------------------------------

    #@partial(jax.jit, static_argnums=(2,))
    def pk_matter(self, z, linear=True):
        """
        Get the matter power spectrum at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        linear : bool
            True for linear matter power spectrum, False for nonlinear matter power spectrum
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Power spectrum and k array
        """
        params = self._to_dict()
        params["z_pk_save_nonclass"] = jnp.atleast_1d(z)[0]

        key = "PKL" if linear else "PKNL"
        emu = self._load_emulator(key)
        k_grid, pk_power_fac = self._pk_grid()
        pk_log = emu.predictions(params)
        pk = 10.0 ** pk_log * pk_power_fac

        return k_grid, pk

    # ------------------------------------------------------------------
    # CMB
    # ------------------------------------------------------------------

    def cmb_dls(self, lmax=10000):
        
        params = self._to_dict()

        tt = self._load_emulator("TT").ten_to_predictions(params)
        ee = self._load_emulator("EE").ten_to_predictions(params)
        te = self._load_emulator("TE").predictions(params)
        pp = self._load_emulator("PP").ten_to_predictions(params)

        n = min(len(tt), len(ee), len(te), len(pp), lmax - 1)
        ell = jnp.arange(2, n + 2)

        return {
            "ell": ell,
            "tt": tt[:n],
            "ee": ee[:n],
            "te": te[:n],
            "pp": pp[:n] / (2 * jnp.pi),
        }

        
    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------

    def derived_parameters(self):
        params = self._to_dict()
        emu = self._load_emulator("DER")
        preds = emu.ten_to_predictions(params)

        names = [  '100*theta_s',
                   'sigma8',
                   'YHe',
                   'z_reio',
                   'Neff',
                   'tau_rec',  # conformal time at which the visibility reaches its maximum (= recombination time)
                   'z_rec', # z at which the visibility reaches its maximum (= recombination redshift)
                   'rs_rec', # comoving sound horizon at recombination in Mpc
                   'chi_rec', # comoving distance to recombination in Mpc
                   'tau_star', # conformal time at which photon optical depth crosses one
                   'z_star', # redshift at which photon optical depth crosses one, i.e., last scattering surface
                   'rs_star', # comoving sound horizon at z_star in Mpc
                   'chi_star', # comoving distance to the last scattering surface in Mpc
                   'rs_drag'] # comoving sound horizon at baryon drag in Mpc

        
        out = {n: preds[i] for i, n in enumerate(names) if i < len(preds)}
        out["h"] = params["H0"] / 100.0
        out["Omega_m"] = (params["omega_b"] + params["omega_cdm"]) / out["h"]**2

        return out


