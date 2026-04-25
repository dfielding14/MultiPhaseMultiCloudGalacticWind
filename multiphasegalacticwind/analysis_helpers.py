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
from .topaz_cooling import load_cooling_table, lambda_total_cgs, tcool_P_topaz

_FOUR_PI_OVER_THREE = 4.0 * np.pi / 3.0


def _interp_profile_at_radius(
    radius_cm: np.ndarray,
    values: np.ndarray,
    radius_new_cm: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a 1D or species-by-radius profile onto a new radius grid."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.interp(radius_new_cm, radius_cm, arr)
    if arr.ndim == 2:
        if arr.shape[0] == 0:
            return np.zeros((0, radius_new_cm.size), dtype=float)
        return np.vstack([np.interp(radius_new_cm, radius_cm, row) for row in arr])
    raise ValueError(f"Expected 1D or 2D profile array, got shape={arr.shape}")


def _restrict_profiles_to_radius_window(
    radius_cm: np.ndarray,
    profiles: Dict[str, np.ndarray],
    r_min_cm: float,
    r_max_cm: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Restrict profiles to a radius window and insert exact boundary points.

    Parameters
    ----------
    radius_cm : array
        Monotonic radius grid in cm.
    profiles : dict
        Mapping from profile names to 1D or 2D arrays sampled on ``radius_cm``.
    r_min_cm, r_max_cm : float
        Radius limits in cm.
    """
    radius_cm = np.asarray(radius_cm, dtype=float)
    if radius_cm.ndim != 1 or radius_cm.size < 2:
        raise ValueError("radius grid must be a 1D array with at least 2 points")
    if not np.all(np.diff(radius_cm) > 0.0):
        raise ValueError("radius grid must be strictly increasing")
    if r_min_cm > r_max_cm:
        raise ValueError("r_min_kpc must be <= r_max_kpc")
    tol = 1e-10 * max(1.0, abs(radius_cm[0]), abs(radius_cm[-1]))
    if r_min_cm < radius_cm[0] - tol or r_max_cm > radius_cm[-1] + tol:
        raise ValueError(
            "Requested radius window is outside the solved domain: "
            f"[{r_min_cm / kpc:.6g}, {r_max_cm / kpc:.6g}] kpc not in "
            f"[{radius_cm[0] / kpc:.6g}, {radius_cm[-1] / kpc:.6g}] kpc."
        )
    r_min_cm = float(np.clip(r_min_cm, radius_cm[0], radius_cm[-1]))
    r_max_cm = float(np.clip(r_max_cm, radius_cm[0], radius_cm[-1]))

    if abs(r_max_cm - r_min_cm) <= tol:
        radius_window = np.array([float(r_min_cm)], dtype=float)
    else:
        interior = radius_cm[(radius_cm > r_min_cm) & (radius_cm < r_max_cm)]
        radius_window = np.concatenate(([r_min_cm], interior, [r_max_cm]))

    profiles_window = {}
    for key, profile in profiles.items():
        profiles_window[key] = _interp_profile_at_radius(radius_cm, profile, radius_window)
    return radius_window, profiles_window


def _cumulative_trapezoid(radius_cm: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Return cumulative trapezoidal integral of ``profile`` over ``radius_cm``."""
    radius_cm = np.asarray(radius_cm, dtype=float)
    profile = np.asarray(profile, dtype=float)
    cumulative = np.zeros_like(profile)
    if profile.size > 1:
        cumulative[1:] = np.cumsum(0.5 * (profile[1:] + profile[:-1]) * np.diff(radius_cm))
    return cumulative


def calculate_radiative_cooling_losses(
    solution: Any,
    r_min_kpc: Optional[float] = None,
    r_max_kpc: Optional[float] = None,
    include_interface: bool = True,
) -> Dict[str, Any]:
    """
    Compute radiative cooling-loss magnitudes from a solved wind trajectory.

    This diagnostic is post-processing only and does not modify ODE evolution.
    All returned losses are positive magnitudes.

    Parameters
    ----------
    solution : Solution
        Output from ``WindModel.run()``.
    r_min_kpc, r_max_kpc : float, optional
        Radius window in kpc. Defaults to the full solved domain.
    include_interface : bool, optional
        If True, include interface cooling from cloud-growth energy loss.

    Returns
    -------
    losses : dict
        Dictionary with emissivity profiles, luminosity profiles, and integrals.
    """
    model = solution.model
    config = model.config
    if config.cooling_backend != "topaz":
        raise ValueError(
            "calculate_radiative_cooling_losses currently supports only cooling_backend='topaz'. "
            f"Got {config.cooling_backend!r}."
        )

    radius_full_cm = np.asarray(solution.sol.t, dtype=float)
    state = np.asarray(solution.sol.y, dtype=float)
    n_species = int(model.N_cloud_species)
    expected_rows = 4 + 3 * n_species
    if state.shape[0] != expected_rows:
        raise ValueError(
            f"State vector has {state.shape[0]} rows, expected {expected_rows} for {n_species} species."
        )

    if r_min_kpc is None:
        r_min_cm = float(radius_full_cm[0])
    else:
        r_min_cm = float(r_min_kpc) * kpc

    if r_max_kpc is None:
        r_max_cm = float(radius_full_cm[-1])
    else:
        r_max_cm = float(r_max_kpc) * kpc

    v_wind = state[0]
    rho_wind = state[1]
    pressure = state[2]
    rhoz_wind = state[3]

    safe_rho_wind = np.maximum(rho_wind, 1e-60)
    hot_valid = (rho_wind > 0.0) & (pressure > 0.0)
    temperature_hot = np.where(
        hot_valid,
        (pressure / kb) * (config.mu * mp / safe_rho_wind),
        1.0,
    )

    topaz_table = load_cooling_table(config.topaz_cooling_table_path)
    lambda_hot = np.asarray(
        lambda_total_cgs(
            temperature_hot,
            config.Z_hot_over_Z_solar,
            table=topaz_table,
        ),
        dtype=float,
    )
    cooling_factor = max(float(config.Cooling_Factor), 0.0)
    q_hot_full = cooling_factor * (safe_rho_wind / (muH * mp)) ** 2 * lambda_hot
    q_hot_full = np.where(hot_valid & np.isfinite(q_hot_full) & (q_hot_full > 0.0), q_hot_full, 0.0)

    if include_interface:
        M_cloud = state[4 : 4 + n_species]
        v_cloud = state[4 + n_species : 4 + 2 * n_species]
        Z_cloud = state[4 + 2 * n_species : 4 + 3 * n_species]

        safe_r = np.maximum(radius_full_cm, 1e-30)
        rho_cloud = pressure * (config.mu * mp) / (kb * config.T_cl)
        chi_safe = np.maximum(rho_cloud / safe_rho_wind, 1e-30)

        z_wind = rhoz_wind / safe_rho_wind
        t_wind = (pressure / kb) * (config.mu * mp / safe_rho_wind)
        t_mix = np.sqrt(np.maximum(t_wind[None, :] * config.T_cl, 1e-30))
        z_mix = np.sqrt(np.maximum(z_wind[None, :] * Z_cloud, 0.0))

        t_cool_layer = np.asarray(
            tcool_P_topaz(
                t_mix,
                pressure / kb,
                z_mix / Z_solar,
                config.mu,
                table=topaz_table,
            ),
            dtype=float,
        )
        t_cool_layer = np.broadcast_to(t_cool_layer, (n_species, radius_full_cm.size)).copy()
        t_cool_layer = np.where(t_cool_layer < 0.0, 1e10 * Myr, t_cool_layer)

        r0 = model.r_star_kpc * kpc
        injection_radius = config.cold_cloud_injection_radial_extent_frac * r0
        injection_factor = np.where(
            radius_full_cm < injection_radius,
            np.power(np.maximum(safe_r / max(injection_radius, 1e-30), 0.0), config.cold_cloud_injection_radial_power),
            1.0,
        )
        Ndot_cloud = np.asarray(model.Ndot_cloud0, dtype=float)[:, None] * injection_factor[None, :]

        v_cloud_floor = max(config.v_cloud_min * 1e5, 1e-10)
        v_cloud_safe = np.maximum(v_cloud, v_cloud_floor)
        number_density_cloud = Ndot_cloud / (config.Omwind * safe_r[None, :] * safe_r[None, :] * v_cloud_safe)

        rho_cloud_safe = np.maximum(rho_cloud, 1e-60)
        r_cloud = np.where(
            M_cloud > 0.0,
            np.power(np.maximum(M_cloud / (_FOUR_PI_OVER_THREE * rho_cloud_safe[None, :]), 0.0), 1.0 / 3.0),
            0.0,
        )
        r_cloud_safe = np.where(r_cloud > 0.0, r_cloud, np.inf)

        v_rel = v_wind[None, :] - v_cloud
        v_turb = (
            config.f_turb0
            * np.abs(v_rel)
            * np.power(chi_safe[None, :], config.TurbulentVelocityChiPower)
        )
        ksi = r_cloud / (np.maximum(v_turb, 1e-10) * np.maximum(t_cool_layer, 1e-30))
        area_boost = config.geometric_factor * np.power(chi_safe[None, :], config.CoolingAreaChiPower)
        ksi_factor = np.where(
            ksi < 1.0,
            np.power(np.maximum(ksi, 0.0), 0.5),
            np.power(np.maximum(ksi, 0.0), 0.25),
        )
        cloud_active = M_cloud > config.M_cloud_min
        Mdot_grow = np.where(
            cloud_active,
            config.Mdot_coefficient
            * 3.0
            * M_cloud
            * v_turb
            * area_boost
            / (r_cloud_safe * chi_safe[None, :])
            * ksi_factor,
            0.0,
        )

        cs_sq_wind = gamma * pressure / safe_rho_wind
        cs_cl_sq = gamma * kb * config.T_cl / (config.mu * mp)
        delta_e = (cs_sq_wind[None, :] - cs_cl_sq) / (gamma - 1.0) + 0.5 * v_rel * v_rel
        delta_e = np.maximum(delta_e, 0.0)

        q_interface_species_full = number_density_cloud * Mdot_grow * delta_e
        q_interface_species_full = np.where(
            np.isfinite(q_interface_species_full) & (q_interface_species_full > 0.0),
            q_interface_species_full,
            0.0,
        )
        q_interface_full = np.sum(q_interface_species_full, axis=0)
    else:
        q_interface_species_full = np.zeros((n_species, radius_full_cm.size), dtype=float)
        q_interface_full = np.zeros(radius_full_cm.size, dtype=float)

    q_total_full = q_hot_full + q_interface_full

    profiles_full = {
        "q_hot_cgs": q_hot_full,
        "q_interface_cgs": q_interface_full,
        "q_total_cgs": q_total_full,
        "q_interface_species_cgs": q_interface_species_full,
    }
    radius_window_cm, profiles_window = _restrict_profiles_to_radius_window(
        radius_full_cm,
        profiles_full,
        r_min_cm=r_min_cm,
        r_max_cm=r_max_cm,
    )

    q_hot = profiles_window["q_hot_cgs"]
    q_interface = profiles_window["q_interface_cgs"]
    q_total = profiles_window["q_total_cgs"]
    q_interface_species = profiles_window["q_interface_species_cgs"]

    dLdr_hot = q_hot * config.Omwind * radius_window_cm * radius_window_cm
    dLdr_interface = q_interface * config.Omwind * radius_window_cm * radius_window_cm
    dLdr_total = q_total * config.Omwind * radius_window_cm * radius_window_cm

    L_hot_cumulative = _cumulative_trapezoid(radius_window_cm, dLdr_hot)
    L_interface_cumulative = _cumulative_trapezoid(radius_window_cm, dLdr_interface)
    L_total_cumulative = _cumulative_trapezoid(radius_window_cm, dLdr_total)

    L_hot = float(L_hot_cumulative[-1])
    L_interface = float(L_interface_cumulative[-1])
    L_total = float(L_total_cumulative[-1])

    return {
        "r_kpc": radius_window_cm / kpc,
        "q_hot_cgs": q_hot,
        "q_interface_cgs": q_interface,
        "q_total_cgs": q_total,
        "q_interface_species_cgs": q_interface_species,
        "dLdr_hot_cgs": dLdr_hot,
        "dLdr_interface_cgs": dLdr_interface,
        "dLdr_total_cgs": dLdr_total,
        "L_hot_cgs": L_hot,
        "L_interface_cgs": L_interface,
        "L_total_cgs": L_total,
        "L_hot_cumulative_cgs": L_hot_cumulative,
        "L_interface_cumulative_cgs": L_interface_cumulative,
        "L_total_cumulative_cgs": L_total_cumulative,
    }


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
    chi = T_wind / T_cl
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


def Cooling_and_Acceleration(
    r: float,
    state: np.ndarray,
    config: Optional[WindConfig] = None,
    *,
    v_circ_kms: float = 150.0,
) -> Dict[str, Any]:
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
    v_circ_kms : float, optional
        Circular velocity in km/s for the isothermal gravitational potential.
        
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
    v_circ = float(v_circ_kms) * km
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
