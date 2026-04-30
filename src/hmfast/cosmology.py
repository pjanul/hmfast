import os
import jax
import jax.numpy as jnp
from typing import Dict, Union
from mcfit import TophatVar
from hmfast.emulator_load import EmulatorLoader, EmulatorLoaderPCA
from hmfast.download import get_default_data_path
from hmfast.utils import Const
from functools import partial

jax.config.update("jax_enable_x64", True)


_COSMO_MODELS = {
    "lcdm:v1": {"suffix": "v1", "subdir": "lcdm"},
    "mnu:v1": {"suffix": "mnu_v1", "subdir": "mnu"},
    "neff:v1": {"suffix": "neff_v1", "subdir": "neff"},
    "wcdm:v1": {"suffix": "w_v1", "subdir": "wcdm"},
    "ede:v1": {"suffix": "v1", "subdir": "ede"},
    "mnu-3states:v1": {"suffix": "v1", "subdir": "mnu-3states"},
    "ede:v2": {"suffix": "v2", "subdir": "ede"},
}



class Cosmology:
    """
    Cosmology model and emulator interface.

    Provides access to cosmological parameters and emulator-based predictions for distances, Hubble parameter, power spectra, CMB spectra, and derived parameters.

    Attributes
    ----------
    emulator_set : str
        Emulator-set identifier selecting the corresponding emulator set.
        Allowed values are ``"lcdm:v1"``, ``"mnu:v1"``, ``"neff:v1"``,
        ``"wcdm:v1"``, ``"ede:v1"``, ``"mnu-3states:v1"``, and ``"ede:v2"``.
    H0 : float
        Hubble constant at :math:`z = 0` in units of
        :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}`.
    omega_cdm : float
        Physical cold dark matter density :math:`\\Omega_{\\mathrm{cdm}} h^2`.
    omega_b : float
        Physical baryon density :math:`\\Omega_b h^2`.
    ln1e10A_s : float
        Log-amplitude of the primordial scalar power spectrum,
        :math:`\\ln(10^{10} A_s)`.
    n_s : float
        Scalar spectral index of primordial perturbations.
    tau_reio : float
        Optical depth to reionization.
    m_ncdm : float
        Total non-cold dark matter mass, used if a massive-neutrino cosmological model is selected.
    N_ur : float
        Effective number of ultra-relativistic species, used if a model with additional radiation degrees of freedom is selected.
    w0_fld : float
        Present-day dark energy equation-of-state parameter :math:`w_0`,
        used if a cosmological model with dark energy equation-of-state
        parameter :math:`w_0` is selected.
    fEDE : float
        Maximum fractional contribution of early dark energy, used if an early dark energy cosmological model is selected.
    log10z_c : float
        Base-10 logarithm of the critical redshift for the early dark energy
        transition, used if an early dark energy cosmological model is
        selected.
    thetai_scf : float
        Initial scalar field displacement for the early dark energy model, in radians, used if an early dark energy cosmological model is selected.
    r : float
        Tensor-to-scalar ratio, used if a cosmological model including primordial tensors is selected.
    T_cmb : float
        CMB temperature today in Kelvin, used when non-emulator background quantities require it.
    deg_ncdm : float
        Degeneracy factor for the non-cold dark matter species, used if a massive-neutrino cosmological model is selected.
    """
    def __init__(self, emulator_set="lcdm:v1", 
                 H0=68.0, omega_cdm=0.12, omega_b=0.02246576, ln1e10A_s=3.035173309489548, n_s=0.965, tau_reio=0.0544,      # LCDM
                 m_ncdm=0.06, N_ur=3.046, w0_fld=-0.95,                                                                     # wCDM, Neff, MNU
                 fEDE=0.1, log10z_c=3.5, thetai_scf=jnp.pi/2, r=0.01,                                                       # EDE
                 T_cmb=2.7255, deg_ncdm=1.0,                                                                                # Non-emulator 
        ):
        
        # Static Metadata
        if emulator_set not in _COSMO_MODELS:
            allowed_models = ", ".join(f'"{model}"' for model in _COSMO_MODELS)
            raise ValueError(
                f"Unknown emulator_set {emulator_set!r}. Allowed values are: {allowed_models}."
            )
        self.emulator_set = emulator_set
        self._emu = {}  # This will be treated as static
        # Eagerly load these emulators to keep Python-side loader state out of jitted paths and avoid JAX tracer errors.
        for key in ("S8Z", "HZ", "DAZ", "PKL", "PKNL"):
            self._load_emulator(key)
        self._tophat_instance = partial(TophatVar(self._pk_grid()[0], lowring=True, backend='jax'), extrap=True)

        # Cosmological params (leaves) to be changed without recompiling jit
        self.H0, self.omega_cdm, self.omega_b, self.ln1e10A_s, self.n_s, self.tau_reio = H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio
        self.m_ncdm, self.N_ur, self.w0_fld = m_ncdm, N_ur, w0_fld
        self.fEDE, self.log10z_c, self.thetai_scf, self.r = fEDE, log10z_c, thetai_scf, r
        self.T_cmb, self.deg_ncdm = T_cmb, deg_ncdm


    # ------------------------------------------------------------------
    # PyTree registration
    # ------------------------------------------------------------------

    def _tree_flatten(self):
        # 1. Children: Only the 15 numerical parameters JAX should "see"
        children = (
            self.H0, self.omega_cdm, self.omega_b, self.ln1e10A_s, self.n_s, self.tau_reio,
            self.m_ncdm, self.N_ur, self.w0_fld, 
            self.fEDE, self.log10z_c, self.thetai_scf, self.r,
            self.T_cmb, self.deg_ncdm
        )
        # 2. Aux data: Static metadata and cached helper objects.
        aux_data = (self.emulator_set, self._emu, self._tophat_instance)
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # Reconstruct using the static metadata
        emulator_set, _emu, _tophat_instance = aux_data
        
        # We bypass __init__ to avoid re-triggering the Loader logic
        obj = cls.__new__(cls)
        obj.emulator_set = emulator_set
        obj._emu = _emu 
        obj._tophat_instance = _tophat_instance
        
        # Assign the 15 parameter children to the object
        (obj.H0, obj.omega_cdm, obj.omega_b, obj.ln1e10A_s, obj.n_s, obj.tau_reio,
         obj.m_ncdm, obj.N_ur, obj.w0_fld, 
         obj.fEDE, obj.log10z_c, obj.thetai_scf, obj.r,
         obj.T_cmb, obj.deg_ncdm) = children
        
        return obj
    
    def update(self, H0=None, omega_cdm=None, omega_b=None, ln1e10A_s=None, n_s=None,
        tau_reio=None, m_ncdm=None, N_ur=None, w0_fld=None, fEDE=None, log10z_c=None,
        thetai_scf=None, r=None, T_cmb=None, deg_ncdm=None):
        """
        Return a new Cosmology instance with updated parameters.

        Each parameter defaults to None. Only those not None are updated.

        Parameters
        ----------
        H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio, m_ncdm, N_ur, w0_fld, fEDE, log10z_c, thetai_scf, r, T_cmb, deg_ncdm : float or None
            Cosmological parameters to update.

        Returns
        -------
        Cosmology
            New instance with updated parameters.
        """
        # Flatten the current instance to get aux_data (static metadata)
        leaves, aux_data = self._tree_flatten()
        names = [
            'H0', 'omega_cdm', 'omega_b', 'ln1e10A_s', 'n_s', 'tau_reio',
            'm_ncdm', 'N_ur', 'w0_fld',
            'fEDE', 'log10z_c', 'thetai_scf', 'r',
            'T_cmb', 'deg_ncdm'
        ]
        values = [
            H0, omega_cdm, omega_b, ln1e10A_s, n_s, tau_reio,
            m_ncdm, N_ur, w0_fld,
            fEDE, log10z_c, thetai_scf, r,
            T_cmb, deg_ncdm
        ]
        # Only update values that are not None
        new_leaves = [v if v is not None else old for v, old in zip(values, leaves)]
        return self._tree_unflatten(aux_data, new_leaves)
            
    # ------------------------------------------------------------------
    # atomic lazy loader (Python-side only)
    # ------------------------------------------------------------------

    def _base_path(self):
        return os.path.join(get_default_data_path(),_COSMO_MODELS[self.emulator_set]["subdir"])
                         

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
            "BB":   ("BB", EmulatorLoader),
            "DER":  ("derived-parameters", EmulatorLoader),
        }
    
        try:
            subdir, loader_cls = key_map[key]
        except KeyError:
            raise KeyError(f"Unknown key: {key}")
    
        self._emu[key] = loader_cls(os.path.join(self._base_path(), subdir, f"{key}_{_COSMO_MODELS[self.emulator_set]['suffix']}"))
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
            'ln10^{10}A_s': self.ln1e10A_s,  # Mapping attribute to what the emulator expects (ln10^{10}A_s is not a valid variable name)
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
        z_max = jnp.where(self.emulator_set == "ede:v2", 20.0, 5.0)
        return jnp.linspace(0.0, z_max, 100, dtype=jnp.float64)     # z grid for Pk(z)

    def _pk_grid(self):
        is_ede_v2 = (self.emulator_set == "ede:v2")
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

    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_sigma_grid(self):
        """
        Compute the interpolation grid for :math:`\\sigma(M, z)`.

        The interpolation mass grid returned here is in physical
        :math:`M_\\odot`.

        Returns
        -------
        ln_x : array_like
            :math:`\\ln(1+z)` grid.
        ln_M : array_like
            :math:`\\ln M` grid.
        sigma_grid : array_like
            :math:`\\sigma(M, z)` values.
        """

        z_grid = self._z_grid_pk()
        cparams = self._cosmo_params()
        h = cparams["h"]

        # Power spectra for all redshifts, shape: (n_k, n_z)
        pk_grid = jax.vmap(lambda zp: self.pk(zp, linear=True)[1].flatten())(z_grid).T * h**3

        # Compute σ²(R, z) using the cached top-hat helper.
        R_grid, var = jax.vmap(self._tophat_instance, in_axes=1, out_axes=(0, 0))(pk_grid)
        R_grid = R_grid[0].flatten()

        # Compute σ(R, z)
        sigma_grid = jnp.exp(0.5 * jnp.log(var))

        # Mass grid, shape: (n_R,)
        rho_crit_0 = cparams["Rho_crit_0"]
        Omega0_cb = cparams['Omega0_cb']
        M_grid = 4.0 * jnp.pi / 3.0 * Omega0_cb * rho_crit_0 * (R_grid ** 3)

        ln_x = jnp.log1p(z_grid)
        ln_M = jnp.log(M_grid)

        return ln_x, ln_M, sigma_grid


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
        Get Hubble parameter :math:`H(z)` at redshift :math:`z` from the emulator.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Hubble parameter(s) in :math:`\\mathrm{km} \\, \\mathrm{s}^{-1} \\, \\mathrm{Mpc}^{-1}`
        """
        
        params = self._to_dict()
        emu = self._load_emulator("HZ")
        preds = 10.0 ** emu.predictions(params) * (Const._c_ / 1e3)
        return self._interp_z(z, self._z_grid_bg(), preds)

    @jax.jit
    def angular_diameter_distance(self, z):
        """
        Get angular diameter distance :math:`D_A(z)` at redshift :math:`z` from the emulator.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Angular diameter distance(s) in :math:`\\mathrm{Mpc}`.
        """
        
        params = self._to_dict()
        emu = self._load_emulator("DAZ")
        preds = emu.predictions(params)

        if self.emulator_set == "ede:v2":
            preds = 10.0 ** preds
            preds = jnp.insert(preds, 0, 0.0)

        return self._interp_z(z, self._z_grid_bg(), preds)

    @jax.jit
    def sigma8(self, z):
        """
        Get :math:`\\sigma_8(z)` at redshift :math:`z` from the emulator.

        :math:`\\sigma_8(z)` is the dimensionless root-mean-square linear
        matter fluctuation amplitude in spheres of radius
        :math:`8 \\, \\mathrm{Mpc}/h`.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless :math:`\\sigma_8` value(s)
        """


        params = self._to_dict()
        emu = self._load_emulator("S8Z")
        preds = emu.predictions(params)
        return self._interp_z(z, self._z_grid_bg(), preds)

    @jax.jit
    def _cosmo_params(self):
        """
        Get the input cosmological parameters together with derived background quantities.
    
        Returns
        -------
        dict
            Dictionary containing the emulator input parameters and the following
            derived quantities:
    
            - ``h``: Dimensionless Hubble parameter, :math:`h = H_0 / 100`
            - ``Omega_b``: Present-day baryon density parameter
            - ``Omega_cdm``: Present-day cold dark matter density parameter
            - ``Omega0_g``: Present-day photon density parameter
            - ``Omega0_ur``: Present-day ultra-relativistic density parameter
            - ``Omega0_ncdm``: Present-day massive neutrino density parameter
            - ``Omega_Lambda``: Present-day dark energy density parameter
            - ``Omega0_m``: Present-day total matter density parameter
            - ``Omega0_r``: Present-day total radiation density parameter
            - ``Omega0_m_nonu``: Present-day matter density parameter excluding
              massive neutrinos
            - ``Omega0_cb``: Present-day CDM+baryon density parameter
            - ``Rho_crit_0``: Present-day critical density in
                            :math:`M_\\odot \\, \\mathrm{Mpc}^{-3}`
    
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
        p['Rho_crit_0'] = (3.0 / (8.0 * jnp.pi * G * M_sun)) * Mpc_over_m * c**2 * H0**2
        
        return p

    @jax.jit
    def critical_density(self, z):
        """
        Get critical density :math:`\\rho_{\\mathrm{crit}}(z)` at redshift :math:`z`.

        .. math::

            \\rho_{\\mathrm{crit}}(z) = \\frac{3 H(z)^2}{8 \\pi G}

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Critical density in :math:`M_\\odot \\, \\mathrm{Mpc}^{-3}`
        """
        
        # Get Hubble parameter    
        H_z = self.hubble_parameter(z)
        
        # Convert H(z) from km/s/Mpc to s^-1 inside the prefactor.
        G, M_sun, Mpc_over_m = Const._G_, Const._M_sun_, Const._Mpc_over_m_
        rho_crit_factor = (3.0 / (8.0 * jnp.pi * G * M_sun)) * (1e6 * Mpc_over_m)
        
        return rho_crit_factor * H_z**2 
        
    @jax.jit
    def omega_m(self, z):
        """
        Matter density parameter excluding neutrinos.
    
        .. math::
    
            \\Omega_m(z) = \\frac{\\Omega_{m,\\mathrm{no\\nu},0}(1+z)^3}{\\Omega_{m,0}(1+z)^3 + \\Omega_{\\Lambda,0} + \\Omega_{r,0}(1+z)^4}
    
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
    
        Returns
        -------
        float or jnp.ndarray
            Dimensionless matter density at redshift :math:`z`
        """
       
        params = self._cosmo_params()
        om0, om0_nonu, or0, ol0 = params['Omega0_m'], params['Omega0_m_nonu'], params['Omega0_r'], params['Omega_Lambda']
        Omega_m_z = om0_nonu * (1. + z)**3. / (om0 * (1. + z)**3. + ol0 + or0 * (1. + z)**4.) # omega_matter without neutrinos
        
        return Omega_m_z

    @jax.jit
    def growth_factor(self, z):
        """
        Linear growth factor :math:`D(z)`, normalized to :math:`D(0)=1`.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless linear growth factor at :math:`z`, with shape
            :math:`(N_z,)`, where singleton dimensions get squeezed before
            return.
        """
        
        z = jnp.atleast_1d(z)
    
        h = self.H0 / 100.0
        k0 = 1e-2  # legacy reference wavenumber in h Mpc^-1
        z_grid_pk = self._z_grid_pk()
        pk0_grid = jax.vmap(lambda zp: jnp.interp(h * k0, *self.pk(zp, linear=True)))(z_grid_pk)
        D_grid = jnp.sqrt(pk0_grid / jnp.interp(h * k0, *self.pk(0.0, linear=True)))
    
        return jnp.squeeze(jnp.interp(z, z_grid_pk, D_grid))

    @jax.jit
    def growth_rate(self, z):
        """
        Linear growth rate

        .. math::

            f(z) = \\frac{d \\ln D}{d \\ln a}

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless linear growth rate at :math:`z`, with shape
            :math:`(N_z,)`, where singleton dimensions get squeezed before
            return.
        """
        
        z = jnp.atleast_1d(z)

        z_grid_pk = self._z_grid_pk()
        D_grid = self.growth_factor(z_grid_pk)
        a_grid = 1.0 / (1.0 + z_grid_pk)
        f_grid = jnp.gradient(jnp.log(D_grid), jnp.log(a_grid))
        
        return jnp.squeeze(jnp.interp(z, z_grid_pk, f_grid))

    @jax.jit
    def velocity_dispersion(self, z):
        """
        Compute the dimensionless velocity dispersion

        .. math::

            \\frac{1}{3} \\frac{v_\\mathrm{rms}^2}{c^2}

        from the linear growth factor and matter power spectrum.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)

        Returns
        -------
        jnp.ndarray
            Dimensionless velocity dispersion at :math:`z`, equal to
            :math:`\\frac{1}{3} \\frac{v_\\mathrm{rms}^2}{c^2}`, with shape
            :math:`(N_z,)`, where singleton dimensions get squeezed before
            return.
        """
        
        z = jnp.atleast_1d(z)
        h = self.H0 / 100.0
        c_km_s = Const._c_ / 1e3
        k_grid = jnp.geomspace(1e-5, 1e1, 1000)
        z_grid_pk = self._z_grid_pk()
    
        # Reconstruct the legacy linear spectrum values so this derived
        # quantity remains numerically unchanged.
        P_grid = jax.vmap(lambda zp: jnp.interp(h * k_grid, *self.pk(zp, linear=True)))(z_grid_pk) * h**3
    
        a_grid = 1.0 / (1.0 + z_grid_pk)
        H_grid = self.hubble_parameter(z_grid_pk)
        f_grid = self.growth_rate(z_grid_pk)
    
        W_grid = f_grid * a_grid * H_grid / c_km_s
        integrand = (W_grid[:, None]**2 / 3) * P_grid * k_grid / (2 * jnp.pi**2)
        velocity_dispersion_grid = jax.scipy.integrate.trapezoid(integrand, x=jnp.log(k_grid), axis=1)
    
        return jnp.squeeze(jnp.interp(z, z_grid_pk, velocity_dispersion_grid))

    @jax.jit
    def comoving_volume_element(self, z):
        """
        Comoving volume element per unit redshift and solid angle.
    
        .. math::
    
            \\frac{dV}{dz\\,d\\Omega} = \\frac{(1+z)^2\\, D_A(z)^2 \\, c}{H(z)}
    
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
    
        Returns
        -------
        float or jnp.ndarray
            :math:`\\frac{dV}{dz\\,d\\Omega}` in :math:`\\mathrm{Mpc}^3 \\, \\mathrm{sr}^{-1}`
        """

        dAz = self.angular_diameter_distance(z)
        Hz = self.hubble_parameter(z)

        return (1 + z)**2 * dAz**2 * (Const._c_ / 1e3) / Hz
   

    # ------------------------------------------------------------------
    # Matter power spectra
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(2,))
    def pk(self, z, linear=True):
        """
        Get the matter power spectrum :math:`P(k, z)` from the emulator.

        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        linear : bool
            True for linear :math:`P(k)`, False for nonlinear :math:`P(k)`

        Returns
        -------
        tuple
            :math:`(k, P(k))`, with :math:`k` in physical
            :math:`\\mathrm{Mpc}^{-1}` and :math:`P(k)` in physical
            :math:`\\mathrm{Mpc}^3`.
        """
        params = self._to_dict()
        params["z_pk_save_nonclass"] = jnp.atleast_1d(z)[0]
        h = self.H0 / 100.0

        key = "PKL" if linear else "PKNL"
        emu = self._load_emulator(key)
        k_grid, pk_power_fac = self._pk_grid()
        pk_log = emu.predictions(params)
        pk = 10.0 ** pk_log * pk_power_fac / h**3

        return h * k_grid, pk

    # ------------------------------------------------------------------
    # CMB
    # ------------------------------------------------------------------

    @jax.jit
    def cl_tt(self):
        """
        Get CMB temperature power spectrum :math:`C_\\ell^{TT}` from the emulator.

        Returns
        -------
        tuple
            :math:`(\\ell, C_\\ell^{TT})`
        """
        params = self._to_dict()
        preds = self._load_emulator("TT").ten_to_predictions(params)
        ell = jnp.arange(2, len(preds) + 2)
        return ell, preds

    @jax.jit
    def cl_ee(self):
        """
        Get CMB :math:`E`-mode polarization power spectrum :math:`C_\\ell^{EE}` from the emulator.

        Returns
        -------
        tuple
            :math:`(\\ell, C_\\ell^{EE})`
        """
        params = self._to_dict()
        preds = self._load_emulator("EE").ten_to_predictions(params)
        ell = jnp.arange(2, len(preds) + 2)
        return ell, preds

    @jax.jit
    def cl_te(self):
        """
        Get CMB temperature-:math:`E`-mode cross power spectrum :math:`C_\\ell^{TE}` from the emulator.

        Returns
        -------
        tuple
            :math:`(\\ell, C_\\ell^{TE})`
        """
        params = self._to_dict()
        preds = self._load_emulator("TE").predictions(params)
        ell = jnp.arange(2, len(preds) + 2)
        return ell, preds

    @jax.jit
    def cl_pp(self):
        """
        Get CMB lensing potential power spectrum :math:`C_\\ell^{\\phi\\phi}` from the emulator.

        Returns
        -------
        tuple
            :math:`(\\ell, C_\\ell^{\\phi\\phi})`
        """
        params = self._to_dict()
        preds = self._load_emulator("PP").ten_to_predictions(params)
        ell = jnp.arange(2, len(preds) + 2)
        # Apply the 1/(2pi) normalization used in your original code
        return ell, preds / (2 * jnp.pi)

    # def cl_bb(self):
    #     if self.emulator_set != "ede:v2": 
    #         raise ValueError("This function is only implemented for EDE-v2 emulators.")
    #     params = self._to_dict()
    #     preds = self._load_emulator("BB").ten_to_predictions(params)
    #     ell, n = self._get_ell_and_n(preds, lmax)
    #     return ell, preds[:n]
        
    # ------------------------------------------------------------------
    # Derived parameters
    # ------------------------------------------------------------------

    @jax.jit
    def derived_parameters(self):
        """
        Get derived cosmological parameters from the emulator.
    
        Returns
        -------
        dict
            Dictionary of derived parameters with the following keys:
    
            - '100*theta_s' : Sound horizon angle (in units of 1/100 radians)
            - 'sigma8' : Dimensionless RMS linear matter fluctuation in 8 Mpc/h spheres
            - 'YHe' : Primordial helium fraction
            - 'z_reio' : Redshift of reionization
            - 'Neff' : Effective number of relativistic species
            - 'tau_rec' : Conformal time at recombination (maximum visibility)
            - 'z_rec' : Redshift at recombination (maximum visibility)
            - 'rs_rec' : Comoving sound horizon at recombination [Mpc]
            - 'chi_rec' : Comoving distance to recombination [Mpc]
            - 'tau_star' : Conformal time at last scattering (optical depth = 1)
            - 'z_star' : Redshift at last scattering (optical depth = 1)
            - 'rs_star' : Comoving sound horizon at last scattering [Mpc]
            - 'chi_star' : Comoving distance to last scattering [Mpc]
            - 'rs_drag' : Comoving sound horizon at baryon drag [Mpc]
        """
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
        
        return out



jax.tree_util.register_pytree_node(
    Cosmology,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: Cosmology._tree_unflatten(aux_data, children)
)