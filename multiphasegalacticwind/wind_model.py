"""
Simple API wrapper for the multiphase galactic wind model.

This provides a clean interface while preserving all the original physics and units.
"""

import numpy as np
from scipy.integrate import solve_ivp
import os

# Import everything from the core physics module
from .core_physics import *


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
                 
                 # Initial conditions at sonic point
                 r_star_kpc=0.3,       # kpc, sonic radius
                 n_star=0.1,           # cm^-3, hot phase density
                 v_star=200.0,         # km/s, initial velocity
                 T_star=5e6,           # K, hot phase temperature
                 Z_star=1.0,           # solar metallicity
                 
                 # Cloud properties
                 cloud_mass_range=(1, 1e5),      # Msun, min and max cloud mass
                 cloud_alpha=2.0,                # power law slope
                 N_cloud_species=10,             # number of cloud mass bins
                 T_cl=1e4,                       # K, cloud temperature
                 
                 # Solver settings
                 r_max_kpc=100.0,      # kpc, maximum radius
                 rtol=1e-8,
                 atol=1e-10):
        """
        Initialize the wind model with galaxy and wind parameters.
        """
        
        # Store parameters
        self.v_circ = v_circ
        self.redshift = redshift
        self.SFR = SFR
        self.eta_M = eta_M
        self.eta_M_cold = eta_M_cold
        self.eta_E = eta_E
        self.r_star_kpc = r_star_kpc
        self.n_star = n_star
        self.v_star = v_star
        self.T_star = T_star
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
        
        # Load cooling table
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.cooling_table_path = os.path.join(data_dir, 'Lambda_tab_redshifts.npz')
        
    def run(self):
        """
        Run the wind model and return a Solution object.
        """
        # Convert units to CGS as expected by the core physics
        r_star = self.r_star_kpc * kpc
        v_star_cgs = self.v_star * 1e5  # km/s to cm/s
        v_circ_cgs = self.v_circ * 1e5
        
        # Calculate derived quantities
        rho_star = self.n_star * mu_mol * mp
        P_star = rho_star * k_B * self.T_star / (mu_mol * mp)
        
        # Set up initial state vector
        # State: [rho, v, P, M_cl_1, ..., M_cl_N, v_cl, Z_cl]
        y0 = np.zeros(3 + self.N_cloud_species + 2)
        y0[0] = rho_star
        y0[1] = v_star_cgs
        y0[2] = P_star
        y0[3:3+self.N_cloud_species] = self.M_cloud0
        y0[3+self.N_cloud_species] = v_star_cgs  # v_cl
        y0[4+self.N_cloud_species] = self.Z_star  # Z_cl
        
        # Integration span
        r_span = [r_star, self.r_max_kpc * kpc]
        
        # Parameters tuple for the ODE system
        params = (self.SFR*Msun/yr, self.eta_M, self.eta_E, v_circ_cgs, 
                  self.N_cloud_species, self.Mdot_cold0, self.Ndot_cloud0,
                  self.T_cl, self.redshift, self.cooling_table_path)
        
        # Run the integration
        sol = solve_ivp(
            lambda r, y: Wind_Evo(r, y, params),
            r_span, y0,
            rtol=self.rtol, atol=self.atol,
            dense_output=True,
            events=[T_eq_Tcl, v_zero, no_clouds]
        )
        
        # Also run hot-only solution for comparison
        y0_hot = y0[:3].copy()
        params_hot = (self.SFR*Msun/yr, self.eta_M, self.eta_E, v_circ_cgs,
                      self.redshift, self.cooling_table_path)
        
        sol_hot = solve_ivp(
            lambda r, y: Hot_Wind_Evo(r, y, params_hot),
            r_span, y0_hot,
            rtol=self.rtol, atol=self.atol,
            dense_output=True,
            events=[T_eq_Tcl_hot, v_zero_hot]
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
        self.rho = sol.y[0]
        self.v = sol.y[1] / 1e5  # Convert to km/s
        self.P = sol.y[2]
        self.M_clouds = sol.y[3:3+model.N_cloud_species]
        self.v_cl = sol.y[3+model.N_cloud_species] / 1e5  # km/s
        self.Z_cl = sol.y[4+model.N_cloud_species]
        
        # Hot-only solution
        self.r_hot = sol_hot.t / kpc
        self.rho_hot = sol_hot.y[0]
        self.v_hot = sol_hot.y[1] / 1e5
        self.P_hot = sol_hot.y[2]
        
        # Derived quantities
        self.n = self.rho / (mu_mol * mp)  # number density
        self.T = self.P / (self.rho / (mu_mol * mp)) / k_B  # temperature
        self.n_hot = self.rho_hot / (mu_mol * mp)
        self.T_hot = self.P_hot / (self.rho_hot / (mu_mol * mp)) / k_B
        
        # Mass fluxes
        self.Mdot = 4 * np.pi * sol.t**2 * self.rho * sol.y[1] / (Msun/yr)
        self.Mdot_hot = 4 * np.pi * sol_hot.t**2 * self.rho_hot * sol_hot.y[1] / (Msun/yr)
        
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