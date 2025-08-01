"""
Simple API wrapper for the multiphase galactic wind model.

This provides a clean interface while preserving all the original physics and units.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os

# Import necessary items from core physics
from .core_physics import (
    setup_cloud_powerlaw_distribution, Wind_Evo, Hot_Wind_Evo,
    create_cold_wind_event, wind_negative, create_all_clouds_frozen_event,
    create_cloud_density_low_event
)
from .constants import *
from .config import WindConfig, get_default_config


class WindModel:
    """
    Simple wrapper for the multiphase galactic wind model.
    
    All units are preserved from the original code:
    - Masses in solar masses (Msun)
    - Velocities in km/s  
    - Distances in kpc
    - Densities in cm^-3
    - Temperatures in K
    """
    
    def __init__(self, 
                 # Galaxy properties
                 v_circ=150.0,         # km/s, circular velocity
                 redshift=0.0,         
                 
                 # Wind launch properties
                 SFR=20.0,             # Msun/yr, star formation rate
                 eta_M=0.1,            # hot phase mass loading
                 eta_M_cold=1.0,       # cold phase mass loading
                 eta_E=1.0,            # energy loading
                 
                 # Sonic point properties
                 r_star_kpc=0.3,       # kpc, sonic radius
                 Z_star=1.0,           # solar metallicity
                 
                 # Cloud properties
                 cloud_mass_range=(1, 1e5),      # Msun, min and max cloud mass
                 cloud_alpha=2.0,                # power law slope
                 N_cloud_species=10,             # number of cloud mass bins
                 T_cl=1e4,                       # K, cloud temperature
                 
                 # Solver settings
                 r_max_kpc=100.0,      # kpc, maximum radius
                 rtol=1e-8,
                 atol=1e-10,
                 
                 # Configuration
                 config=None,          # WindConfig instance
                 **config_kwargs):
        """
        Initialize the wind model with galaxy and wind parameters.
        
        Parameters
        ----------
        config : WindConfig, optional
            Configuration object with model parameters. If None, uses defaults.
        **config_kwargs : dict
            Additional parameters to override in the configuration.
            
        Examples
        --------
        Use default configuration:
        >>> model = WindModel(SFR=10.0)
        
        Use custom configuration:
        >>> config = WindConfig(f_turb0=0.2, drag_coeff=0.3)
        >>> model = WindModel(SFR=10.0, config=config)
        
        Override specific parameters:
        >>> model = WindModel(SFR=10.0, f_turb0=0.2, drag_coeff=0.3)
        """
        
        # Set up configuration
        if config is None:
            # Add redshift to config_kwargs if not already there
            if 'redshift' not in config_kwargs:
                config_kwargs['redshift'] = redshift
            config = WindConfig(**config_kwargs)
        else:
            # Override any parameters passed as kwargs
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            # Also set redshift on config if provided
            if redshift != config.redshift:
                config.redshift = redshift
        self.config = config
        
        # Store parameters
        self.v_circ = v_circ
        self.redshift = redshift
        self.SFR = SFR
        self.eta_M = eta_M
        self.eta_M_cold = eta_M_cold
        self.eta_E = eta_E
        self.r_star_kpc = r_star_kpc
        self.Z_star = Z_star
        self.T_cl = T_cl
        self.r_max_kpc = r_max_kpc
        self.rtol = rtol
        self.atol = atol
        
        # Set up cloud distribution
        log_M_cloud_min = np.log10(cloud_mass_range[0])
        log_M_cloud_max = np.log10(cloud_mass_range[1])
        self.N_cloud_species = N_cloud_species
        
        # Use the original function to set up clouds
        self.M_cloud0, self.eta_M_cold_array, self.Mdot_cold0, self.Ndot_cloud0 = \
            setup_cloud_powerlaw_distribution(
                log_M_cloud_min, log_M_cloud_max, N_cloud_species,
                alpha_cloud=cloud_alpha, eta_M_cold_tot=eta_M_cold, SFR=SFR*Msun/yr
            )
        
        # Calculate sonic point conditions from physics
        self._calculate_sonic_point_conditions()
        
        # Load cooling table
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.cooling_table_path = os.path.join(data_dir, 'Lambda_tab_redshifts.npz')
    
    def _calculate_sonic_point_conditions(self):
        """
        Calculate sonic point conditions from energy and mass injection rates.
        
        This follows the analytic solution for spherical winds at the sonic point
        where Mach = 1 + epsilon. The conditions are uniquely determined by the
        energy and mass injection rates.
        """
        # Convert to CGS units
        SFR_cgs = self.SFR * Msun / yr
        r_star = self.r_star_kpc * kpc
        
        # Mass and energy injection rates
        Mdot = self.eta_M * SFR_cgs  # g/s
        Edot = self.eta_E * (self.config.E_SN / self.config.mstar / Msun) * SFR_cgs  # erg/s
        
        # Sonic point Mach number
        Mach0 = 1.0 + self.config.epsilon
        gamma = 5.0/3.0  # Adiabatic index
        
        # Solve for sonic point velocity from energy conservation
        # v0 = sqrt(Edot/Mdot) * (1/((gamma-1)*Mach0) + 1/2)^(-1/2)
        v0 = np.sqrt(Edot/Mdot) * (1.0/((gamma-1)*Mach0) + 0.5)**(-0.5)
        
        # Density from mass flux conservation
        rho0 = Mdot / (self.config.Omwind * r_star**2 * v0)
        
        # Pressure from Mach number definition
        P0 = rho0 * v0**2 / (Mach0**2 * gamma)
        
        # Convert to convenient units
        self.v_star = v0 / 1e5  # km/s
        self.n_star = rho0 / (self.config.mu * mp)  # cm^-3
        self.T_star = P0 / (rho0 / (self.config.mu * mp)) / kb  # K
        
        # Store for later use
        self.rho_star = rho0
        self.P_star = P0
        
    def run(self):
        """
        Run the wind model and return a Solution object.
        """
        # Convert units to CGS as expected by the core physics
        r_star = self.r_star_kpc * kpc
        v_star_cgs = self.v_star * 1e5  # km/s to cm/s
        v_circ_cgs = self.v_circ * 1e5
        
        # Use pre-calculated sonic point values
        rho_star = self.rho_star
        P_star = self.P_star
        
        # Set up initial state vector
        # State: [v_wind, rho_wind, Pressure, rhoZ_wind, 
        #         M_cloud_1, ..., M_cloud_N,
        #         v_cloud_1, ..., v_cloud_N,
        #         Z_cloud_1, ..., Z_cloud_N]
        y0 = np.zeros(4 + 3*self.N_cloud_species)
        y0[0] = v_star_cgs
        y0[1] = rho_star
        y0[2] = P_star
        y0[3] = rho_star * self.Z_star  # rhoZ_wind
        y0[4:4+self.N_cloud_species] = self.M_cloud0 * Msun  # Convert to grams
        y0[4+self.N_cloud_species:4+2*self.N_cloud_species] = 100.0 * 1e5  # v_cloud array, 100 km/s in cm/s
        y0[4+2*self.N_cloud_species:] = self.Z_star  # Z_cloud array
        
        # Integration span
        r_span = [r_star, self.r_max_kpc * kpc]
        
        # Calculate source term parameters
        # Energy and mass injection rates
        Edot = self.eta_E * (self.SFR * Msun/yr) * 0.5 * v_circ_cgs**2  # erg/s
        Mdot = self.eta_M * (self.SFR * Msun/yr)  # g/s
        
        # Source volume (sphere of radius r_star)
        r0 = r_star  # injection radius same as sonic radius
        source_volume = 4./3. * np.pi * r0**3
        
        # Volume-averaged source terms
        Edot_per_Vol = Edot / source_volume  # erg/s/cm^3
        Mdot_per_Vol = Mdot / source_volume  # g/s/cm^3
        
        # Parameters for Wind_Evo
        # Use injection parameters from config
        injection_radius = self.config.cold_cloud_injection_radial_extent
        injection_power = self.config.cold_cloud_injection_radial_power
        
        # Pre-calculate cooling interpolator for efficiency
        from .cooling import get_cooling_interpolator
        cooling_interpolator = get_cooling_interpolator(
            self.config.mu, self.config.metallicity, self.config.redshift
        )
        
        # Extended params tuple including source terms and cooling interpolator
        params = (v_circ_cgs, self.Ndot_cloud0, self.T_cl, 
                  injection_radius, injection_power, self.config.to_dict(),
                  r0, Edot_per_Vol, Mdot_per_Vol, cooling_interpolator)
        
        # Create event functions with proper parameters
        cold_wind = create_cold_wind_event(self.T_cl, self.config.mu)
        all_clouds_frozen = create_all_clouds_frozen_event(self.config.M_cloud_min)
        cloud_density_low = create_cloud_density_low_event(
            self.Ndot_cloud0, injection_radius, injection_power,
            self.config.Omwind, self.config.M_cloud_min
        )
        
        # Run the integration
        sol = solve_ivp(
            lambda r, y: Wind_Evo(r, y, params),
            r_span, y0,
            rtol=self.rtol, atol=self.atol,
            dense_output=True,
            events=[cold_wind, wind_negative, all_clouds_frozen, cloud_density_low]
        )
        
        # Also run hot-only solution for comparison
        y0_hot = y0[:3].copy()
        # Include source terms for hot wind
        params_hot = (v_circ_cgs, True, r0, Edot_per_Vol, Mdot_per_Vol)
        
        sol_hot = solve_ivp(
            lambda r, y: Hot_Wind_Evo(r, y, params_hot),
            r_span, y0_hot,
            rtol=self.rtol, atol=self.atol,
            dense_output=True,
            events=[wind_negative]
        )
        
        return Solution(sol, sol_hot, self)


class Solution:
    """
    Container for wind model solution with convenient access to results.
    """
    
    def __init__(self, sol, sol_hot, model):
        self.sol = sol
        self.sol_hot = sol_hot
        self.model = model
        
        # Extract solution arrays
        self.r = sol.t / kpc  # Convert to kpc
        self.v = sol.y[0] / 1e5  # Convert to km/s
        self.rho = sol.y[1]
        self.P = sol.y[2]
        self.rhoZ = sol.y[3]
        self.Z = self.rhoZ / self.rho  # metallicity
        self.M_clouds = sol.y[4:4+model.N_cloud_species] / Msun  # Convert to Msun
        self.v_cl = sol.y[4+model.N_cloud_species:4+2*model.N_cloud_species] / 1e5  # km/s
        self.Z_cl = sol.y[4+2*model.N_cloud_species:]
        
        # Hot-only solution
        self.r_hot = sol_hot.t / kpc
        self.v_hot = sol_hot.y[0] / 1e5
        self.rho_hot = sol_hot.y[1]
        self.P_hot = sol_hot.y[2]
        
        # Derived quantities
        mu = model.config.mu
        self.n = self.rho / (mu * mp)  # number density
        self.T = self.P / (self.rho / (mu * mp)) / kb  # temperature
        self.n_hot = self.rho_hot / (mu * mp)
        self.T_hot = self.P_hot / (self.rho_hot / (mu * mp)) / kb
        
        # Mass fluxes
        self.Mdot = 4 * np.pi * sol.t**2 * self.rho * sol.y[0] / (Msun/yr)
        self.Mdot_hot = 4 * np.pi * sol_hot.t**2 * self.rho_hot * sol_hot.y[0] / (Msun/yr)
        
        # Total cloud mass
        self.M_cloud_tot = np.sum(self.M_clouds, axis=0)
        
    def interpolate(self, r_eval):
        """
        Interpolate solution at specific radii (in kpc).
        """
        r_eval_cgs = np.atleast_1d(r_eval) * kpc
        return self.sol.sol(r_eval_cgs)
    
    @property
    def v_at_10kpc(self):
        """Velocity at 10 kpc in km/s."""
        if self.r[-1] >= 10:
            return np.interp(10, self.r, self.v)
        else:
            return np.nan
    
    @property
    def mass_loading_at_10kpc(self):
        """Mass loading factor at 10 kpc."""
        if self.r[-1] >= 10:
            Mdot_10kpc = np.interp(10, self.r, self.Mdot)
            return Mdot_10kpc / self.model.SFR
        else:
            return np.nan
    
    def calculate_velocity_distribution(self, **kwargs):
        """
        Calculate the velocity distribution dN/dv.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments passed to observables.calculate_velocity_distribution
            
        Returns
        -------
        v_cloud : array
            Cloud velocities [km/s by default]
        dN_dv : array
            Velocity distribution
        """
        from .observables import calculate_velocity_distribution
        return calculate_velocity_distribution(self, **kwargs)
    
    def calculate_velocity_moments(self, **kwargs):
        """
        Calculate velocity distribution and its moments.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments for velocity distribution calculation
            
        Returns
        -------
        moments : dict
            Dictionary with mean, dispersion, etc.
        """
        from .observables import calculate_velocity_moments
        v_cloud, dN_dv = self.calculate_velocity_distribution(**kwargs)
        return calculate_velocity_moments(v_cloud, dN_dv)
    
    def calculate_column_density_distribution(self, **kwargs):
        """
        Calculate column density distribution dN/dv in cm^-2 / (km/s).
        
        Parameters
        ----------
        **kwargs : dict
            Arguments passed to observables.calculate_column_density_distribution
            
        Returns
        -------
        v_cloud : array
            Cloud velocities [km/s]
        dN_dv_column : array
            Column density distribution [cm^-2 / (km/s)]
        """
        from .observables import calculate_column_density_distribution
        return calculate_column_density_distribution(self, **kwargs)
    
    def calculate_column_density_by_species(self, **kwargs):
        """
        Calculate column density distribution for each cloud species.
        
        Parameters
        ----------
        **kwargs : dict
            Arguments passed to observables.calculate_column_density_by_species
            
        Returns
        -------
        v_cloud : array
            Cloud velocities [km/s]
        dN_dv_species : dict
            Dictionary with 'total', 'species' list, and 'M_cloud0'
        """
        from .observables import calculate_column_density_by_species
        return calculate_column_density_by_species(self, **kwargs)