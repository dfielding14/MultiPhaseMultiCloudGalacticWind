"""
Physical analysis helper functions for multiphase galactic wind models.

This module contains helper functions for analyzing wind solutions,
calculating derived quantities, and diagnostic metrics.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
from scipy import interpolate

from .constants import *
from .config import WindConfig
from .cooling import get_tcool_min_interpolators, tcool_P


def Field_Length(state: np.ndarray, config: Optional[WindConfig] = None) -> float:
    """
    Calculate the Field length for thermal conduction.
    
    This is the characteristic length scale where thermal conduction
    balances cooling in the hot wind.
    
    Parameters
    ----------
    state : array
        State vector [v, rho, P, rhoZ, ...]
    config : WindConfig, optional
        Configuration object. If None, uses default values.
        
    Returns
    -------
    L_Field : float
        Field length in cm
    """
    if config is None:
        config = WindConfig()
        
    rho_wind = state[1]
    Pressure = state[2]
    rhoZ_wind = state[3]
    Z_wind = rhoZ_wind / rho_wind
    T_wind = Pressure / kb / (rho_wind / (config.mu * mp))
    
    # Spitzer thermal conductivity
    f_spitzer = 1.0  # Spitzer suppression factor
    kappa = 5.0e-7 * T_wind**2.5  # erg cm^-1 s^-1 K^-1
    
    # Get minimum cooling time interpolator
    _, tcool_min_P = get_tcool_min_interpolators()
    edot_cool = 1.5 * Pressure / tcool_min_P((Pressure/kb, Z_wind/Z_solar))
    
    return np.sqrt(f_spitzer * kappa * T_wind / edot_cool)


def Field_Length_mix(state: np.ndarray, config: Optional[WindConfig] = None) -> float:
    """
    Calculate the Field length for the mixed temperature layer.
    
    This uses the geometric mean temperature between hot and cold phases.
    
    Parameters
    ----------
    state : array
        State vector [v, rho, P, rhoZ, ...]
    config : WindConfig, optional
        Configuration object. If None, uses default values.
        
    Returns
    -------
    L_Field_mix : float
        Field length for mixed layer in cm
    """
    if config is None:
        config = WindConfig()
        
    rho_wind = state[1]
    Pressure = state[2]
    rhoZ_wind = state[3]
    Z_wind = rhoZ_wind / rho_wind
    T_wind = Pressure / kb / (rho_wind / (config.mu * mp))
    
    # Mixed temperature (geometric mean with cloud temperature)
    T_cl = config.T_cl  # Use config cloud temperature
    T_mix = np.sqrt(T_wind * T_cl)
    
    # Mixed metallicity
    Z_cloud = config.Z_cloud_over_Z_solar * Z_solar  # Default cloud metallicity
    Z_mix = np.sqrt(Z_wind * Z_cloud)
    
    # Spitzer thermal conductivity at mixed temperature
    f_spitzer = 1.0
    kappa = 5.0e-7 * T_mix**2.5
    
    # tcool_P expects pressure in P/k_B [K cm^-3]
    edot_cool = 1.5 * Pressure / tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, config.redshift, config.mu)
    
    return np.sqrt(f_spitzer * kappa * T_mix / np.abs(edot_cool))


def cloud_radius(r: float, state: np.ndarray, config: Optional[WindConfig] = None,
                 N_cloud_species: Optional[int] = None) -> np.ndarray:
    """
    Calculate cloud radii from their masses and densities.
    
    Assumes pressure equilibrium between hot and cold phases.
    
    Parameters
    ----------
    r : float
        Current radius in cm
    state : array
        State vector [v, rho, P, rhoZ, M_cloud1, ..., v_cloud1, ..., Z_cloud1, ...]
    config : WindConfig, optional
        Configuration object. If None, uses default values.
    N_cloud_species : int, optional
        Number of cloud species. If None, determined from state vector.
        
    Returns
    -------
    r_cloud : array
        Cloud radii in cm
    """
    if config is None:
        config = WindConfig()
        
    # Determine number of cloud species from state vector
    if N_cloud_species is None:
        N_cloud_species = (len(state) - 4) // 3
        
    # Extract cloud properties
    M_cloud = state[4:4+N_cloud_species]
    if N_cloud_species == 1:
        M_cloud = np.atleast_1d(M_cloud)
        
    # Get pressure from state
    Pressure = state[2]  # dyne/cm^2
    
    # Calculate cloud density from pressure equilibrium
    # P = n * k * T, so rho = (mu * mp) * n = (mu * mp) * P / (k * T)
    T_cloud = config.T_cl  # K
    rho_cloud = (config.mu * mp) * Pressure / (kb * T_cloud)  # g/cm^3
    
    # Guard against unphysical pressure/temperature states.
    if rho_cloud <= 0:
        return np.zeros_like(M_cloud)

    # Cloud radius from mass and density (only for positive masses).
    r_cloud = np.zeros_like(M_cloud, dtype=float)
    valid = M_cloud > 0
    r_cloud[valid] = (M_cloud[valid] / (4*np.pi/3. * rho_cloud))**(1/3.)
    
    return r_cloud


def cloud_ksi(r: float, state: np.ndarray, config: Optional[WindConfig] = None,
              N_cloud_species: Optional[int] = None) -> np.ndarray:
    """
    Calculate the cloud mixing parameter ksi.
    
    This parameter controls the strength of mixing/mass transfer between
    hot and cold phases.
    
    Parameters
    ----------
    r : float
        Current radius in cm
    state : array
        State vector
    config : WindConfig, optional
        Configuration object
    N_cloud_species : int, optional
        Number of cloud species
        
    Returns
    -------
    ksi : array
        Mixing parameter for each cloud species
    """
    if config is None:
        config = WindConfig()
        
    # Determine number of cloud species
    if N_cloud_species is None:
        N_cloud_species = (len(state) - 4) // 3
        
    # Extract state variables
    v_wind = state[0]
    rho_wind = state[1]
    Pressure = state[2]
    rhoZ_wind = state[3]
    M_cloud = state[4:4+N_cloud_species]
    v_cloud = state[4+N_cloud_species:4+2*N_cloud_species]
    Z_cloud = state[4+2*N_cloud_species:]
    
    if N_cloud_species == 1:
        M_cloud = np.atleast_1d(M_cloud)
        v_cloud = np.atleast_1d(v_cloud)
        Z_cloud = np.atleast_1d(Z_cloud)
    
    # Calculate derived quantities
    Z_wind = rhoZ_wind / rho_wind
    T_wind = Pressure / kb / (rho_wind / (config.mu * mp))
    T_cl = config.T_cl  # Cloud temperature from config
    
    # Cloud properties
    r_cl = cloud_radius(r, state, config, N_cloud_species)
    chi = T_cl / T_wind
    v_rel = v_wind - v_cloud
    
    # Turbulent velocity
    v_turb = config.f_turb0 * np.abs(v_rel) * chi**config.TurbulentVelocityChiPower
    
    # Mixed layer properties
    T_mix = np.sqrt(T_wind * T_cl)
    Z_mix = np.sqrt(Z_wind * Z_cloud)
    
    # tcool_P expects pressure in P/k_B [K cm^-3]
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, config.redshift, config.mu)
    if np.isscalar(t_cool_layer):
        t_cool_layer = np.full_like(M_cloud, t_cool_layer)
    
    # Handle negative cooling times
    t_cool_layer = np.where(t_cool_layer < 0, 1e10*Myr, t_cool_layer)
    
    # Calculate ksi
    ksi = r_cl / (np.maximum(v_turb, 1e-10) * t_cool_layer)
    
    # Set ksi to zero for destroyed clouds
    ksi = np.where(M_cloud > config.M_cloud_min, ksi, 0.0)
    
    return ksi


def Cooling_and_Acceleration(r: float, state: np.ndarray, 
                             config: Optional[WindConfig] = None) -> Dict[str, Any]:
    """
    Calculate cooling rates and acceleration terms.
    
    Parameters
    ----------
    r : float
        Current radius in cm
    state : array
        State vector
    config : WindConfig, optional
        Configuration object
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - edot_cool: cooling rate (erg/s/cm^3)
        - t_cool: cooling time (s)
        - dv_dr_grav: gravitational acceleration (cm/s/cm)
        - dv_dr_press: pressure gradient acceleration (cm/s/cm)
        - M_sonic: Mach number
    """
    if config is None:
        config = WindConfig()
        
    # Extract state
    v_wind = state[0]
    rho_wind = state[1]
    Pressure = state[2]
    rhoZ_wind = state[3]
    
    # Derived quantities
    Z_wind = rhoZ_wind / rho_wind
    T_wind = Pressure / kb / (rho_wind / (config.mu * mp))
    cs_sq = gamma * Pressure / rho_wind
    M_sonic = v_wind / np.sqrt(cs_sq)
    
    # Cooling rate and time
    # tcool_P expects pressure in P/k_B [K cm^-3]
    t_cool = tcool_P(T_wind, Pressure/kb, Z_wind/Z_solar, config.redshift, config.mu)
    edot_cool = 1.5 * Pressure / t_cool if t_cool > 0 else 0.0
    
    # Acceleration terms (simplified - assuming isothermal potential)
    v_circ = 150.0 * km  # Default circular velocity
    dv_dr_grav = -(v_circ/v_wind)**2 * v_wind / r
    dv_dr_press = -cs_sq / (rho_wind * v_wind) * (-2 * rho_wind / r)  # Assuming drho/dr ~ -2*rho/r
    
    return {
        'edot_cool': edot_cool,
        't_cool': t_cool,
        'dv_dr_grav': dv_dr_grav,
        'dv_dr_press': dv_dr_press,
        'M_sonic': M_sonic,
        'T_wind': T_wind,
        'cs': np.sqrt(cs_sq)
    }


def Gradient_Components(r: float, state: np.ndarray, 
                        config: Optional[WindConfig] = None) -> Dict[str, np.ndarray]:
    """
    Calculate individual components of the gradient terms.
    
    This is useful for debugging and understanding which physical
    processes dominate the evolution.
    
    Parameters
    ----------
    r : float
        Current radius in cm
    state : array
        State vector
    config : WindConfig, optional
        Configuration object
        
    Returns
    -------
    components : dict
        Dictionary containing gradient components for each variable
    """
    raise NotImplementedError(
        "Gradient_Components is not implemented yet. "
        "Returning placeholder zeros is intentionally disabled to avoid "
        "physically misleading diagnostics."
    )


def calculate_cloud_moments(r: float, state: np.ndarray, 
                           config: Optional[WindConfig] = None,
                           N_cloud_species: Optional[int] = None,
                           Ndot_cloud0: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate moments of the cloud distribution.
    
    Parameters
    ----------
    r : float
        Current radius in cm
    state : array
        State vector
    config : WindConfig, optional
        Configuration object
    N_cloud_species : int, optional
        Number of cloud species
    Ndot_cloud0 : array, optional
        Cloud injection rates [1/s] for each species
        
    Returns
    -------
    moments : dict
        Dictionary containing:
        - Mdot_cloud_tot: Total cloud mass flux (g/s)
        - v_cloud_avg: Mass-weighted average cloud velocity (cm/s)
        - v_cloud_disp: Velocity dispersion (cm/s)
        - N_cloud_tot: Total number of clouds
        - Z_cloud_mass_avg: Mass-weighted average cloud metallicity
    """
    if config is None:
        config = WindConfig()
        
    # Determine number of cloud species
    if N_cloud_species is None:
        N_cloud_species = (len(state) - 4) // 3
        
    # Extract cloud properties
    M_cloud = state[4:4+N_cloud_species]
    v_cloud = state[4+N_cloud_species:4+2*N_cloud_species]
    Z_cloud = state[4+2*N_cloud_species:]
    
    if N_cloud_species == 1:
        M_cloud = np.atleast_1d(M_cloud)
        v_cloud = np.atleast_1d(v_cloud)
        Z_cloud = np.atleast_1d(Z_cloud)
    
    # Mask for existing clouds
    cloud_exists = M_cloud > config.M_cloud_min
    
    # Calculate actual cloud number density if injection rates provided
    if Ndot_cloud0 is not None:
        # Get injection parameters
        r0 = r  # Assuming we're past the injection region for simplicity
        injection_radius = config.cold_cloud_injection_radial_extent_frac * r0
        injection_power = config.cold_cloud_injection_radial_power
        
        # Calculate injection function
        r_kpc = r / kpc
        injection_radius_kpc = injection_radius / kpc
        if r_kpc < injection_radius_kpc:
            injection_function = (r_kpc / injection_radius_kpc)**injection_power
        else:
            injection_function = 1.0
            
        # Number conservation: n_cloud = Ndot * f_inj / (Omega * r^2 * v)
        number_density_cloud = np.zeros_like(M_cloud)
        for i in range(N_cloud_species):
            if cloud_exists[i] and v_cloud[i] > 0:
                number_density_cloud[i] = (Ndot_cloud0[i] * injection_function / 
                                          (config.Omwind * r**2 * v_cloud[i]))
    else:
        # Without injection rates, we can't calculate actual density
        # Use a more reasonable estimate based on typical values
        # Typical cloud densities are ~10^-8 to 10^-6 cm^-3 at 10 kpc
        typical_density = 1e-7 * (kpc / r)**2  # Scale with r^-2
        number_density_cloud = np.where(cloud_exists, typical_density, 0.0)
    
    # Mass flux
    Mdot_cloud = config.Omwind * r**2 * number_density_cloud * M_cloud * v_cloud
    
    # Calculate moments
    moments = {}
    
    # Total cloud mass flux
    moments['Mdot_cloud_tot'] = np.sum(Mdot_cloud[cloud_exists])
    
    # Number of active clouds
    moments['N_cloud_tot'] = np.sum(cloud_exists)
    
    if moments['Mdot_cloud_tot'] > 0:
        # Mass-weighted average velocity
        moments['v_cloud_avg'] = np.sum(Mdot_cloud * v_cloud * cloud_exists) / moments['Mdot_cloud_tot']
        
        # Velocity dispersion
        v_diff_sq = (v_cloud - moments['v_cloud_avg'])**2
        moments['v_cloud_disp'] = np.sqrt(
            np.sum(Mdot_cloud * v_diff_sq * cloud_exists) / moments['Mdot_cloud_tot']
        )
        
        # Mass-weighted metallicity
        moments['Z_cloud_mass_avg'] = np.sum(Mdot_cloud * Z_cloud * cloud_exists) / moments['Mdot_cloud_tot']
    else:
        moments['v_cloud_avg'] = 0.0
        moments['v_cloud_disp'] = 0.0
        moments['Z_cloud_mass_avg'] = 0.0
    
    return moments


def get_cloud_mass_spectrum(M_cloud: np.ndarray, 
                           number_density_cloud: np.ndarray,
                           M_cloud_min: float,
                           mass_bins: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the cloud mass spectrum dN/dlogM.
    
    Parameters
    ----------
    M_cloud : array
        Cloud masses in grams
    number_density_cloud : array
        Number density of each cloud species in cm^-3
    M_cloud_min : float
        Minimum cloud mass threshold in grams
    mass_bins : array, optional
        Mass bin edges for spectrum calculation
        
    Returns
    -------
    M_bins : array
        Mass bin centers in grams
    dN_dlogM : array
        Number per logarithmic mass interval in cm^-3
    """
    # Mask for existing clouds
    cloud_exists = M_cloud > M_cloud_min
    
    if mass_bins is None:
        # Create logarithmic mass bins
        if np.any(cloud_exists):
            M_min = np.min(M_cloud[cloud_exists])
            M_max = np.max(M_cloud[cloud_exists])
        else:
            M_min = M_cloud_min
            M_max = M_cloud_min * 1e4
        mass_bins = np.logspace(np.log10(M_min), np.log10(M_max), 50)
    
    # Bin centers
    M_bins = np.sqrt(mass_bins[:-1] * mass_bins[1:])
    
    # Calculate spectrum
    dN_dlogM = np.zeros(len(M_bins))
    
    for i in range(len(M_bins)):
        # Find clouds in this mass bin
        in_bin = ((M_cloud >= mass_bins[i]) & (M_cloud < mass_bins[i+1]) & cloud_exists)
        # Sum number density in bin
        if np.any(in_bin):
            dN_dlogM[i] = np.sum(number_density_cloud[in_bin])
    
    # Normalize by bin width in log space
    dlogM = np.diff(np.log10(mass_bins))
    dN_dlogM = dN_dlogM / dlogM
    
    return M_bins, dN_dlogM
