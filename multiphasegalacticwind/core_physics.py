"""
Multiphase Wind Evolution - Multicloud Version

This module extends the single-cloud wind evolution model to handle multiple cloud species
with a power-law mass distribution: dN/dM ∝ M^-α

Based on Fielding & Bryan "The Structure of Multiphase Galactic Winds"
"""

import numpy as np
# Import physical constants
from .constants import *
# Import cooling functions
from .cooling import tcool_P

def setup_cloud_powerlaw_distribution(log_M_cloud_min, log_M_cloud_max, N_cloud_species,
                                     alpha_cloud=2.0, eta_M_cold_tot=1.0, SFR=1.0):
    """
    Set up a power-law distribution of cloud masses: dN/dM ∝ M^-α

    Parameters:
    -----------
    log_M_cloud_min : float
        Log10 of minimum cloud mass in solar masses
    log_M_cloud_max : float
        Log10 of maximum cloud mass in solar masses
    N_cloud_species : int
        Number of cloud mass bins
    alpha_cloud : float
        Power-law exponent (default=2.0)
    eta_M_cold_tot : float
        Total cold phase mass loading factor
    SFR : float
        Star formation rate in Msun/yr

    Returns:
    --------
    M_cloud0 : array
        Initial cloud masses for each species (grams)
    eta_M_cold : array
        Mass loading factor for each cloud species
    Mdot_cold0 : array
        Mass flux for each cloud species (g/s)
    Ndot_cloud0 : array
        Number flux for each cloud species (1/s)
    """
    # Generate logarithmically spaced cloud mass bin edges
    M_cloud0 = np.logspace(log_M_cloud_min, log_M_cloud_max, N_cloud_species) * Msun

    # Power-law distribution: dN/dM ∝ M^-α
    # For each bin, we need dN/dlogM ∝ M^(1-α)
    dN_dlogM = M_cloud0**(1 - alpha_cloud)

    if N_cloud_species == 1:
        # Special case: single cloud mass
        eta_M_cold = np.array([eta_M_cold_tot])
    else:
        # Width of each logarithmic bin
        dlogM = np.diff(np.log10(M_cloud0))[0]

        # Number of clouds in each bin (relative)
        N_rel = dN_dlogM * dlogM

        # Mass in each bin: M * dN
        M_in_bin = M_cloud0 * N_rel

        # Normalize to get eta_M_cold for each species
        eta_M_cold = eta_M_cold_tot * M_in_bin / np.sum(M_in_bin)

    # Mass flux for each species
    Mdot_cold0 = eta_M_cold * SFR

    # Number flux for each species
    Ndot_cloud0 = Mdot_cold0 / M_cloud0

    return M_cloud0, eta_M_cold, Mdot_cold0, Ndot_cloud0


def Wind_Evo(r, state, params):
    """
    Compute wind evolution derivatives - multicloud version
    Matches the original multicloud.py formulation exactly

    State vector format:
    [v_wind, rho_wind, Pressure, rhoZ_wind,
     M_cloud_1, ..., M_cloud_N,
     v_cloud_1, ..., v_cloud_N,
     Z_cloud_1, ..., Z_cloud_N]

    Parameters:
    params : tuple
        (v_circ, Ndot_cloud0, T_cloud, injection_radius, injection_power, config_dict, r0, Edot_per_Vol, Mdot_per_Vol)
        - v_circ : circular velocity [cm/s]
        - Ndot_cloud0 : cloud injection rates [1/s]
        - T_cloud : cloud temperature [K]
        - injection_radius : cold cloud injection extent [cm]
        - injection_power : cold cloud injection power law
        - config_dict : dict with model parameters
        - r0 : source region radius [cm]
        - Edot_per_Vol : energy injection rate per volume [erg/s/cm^3]
        - Mdot_per_Vol : mass injection rate per volume [g/s/cm^3]
    """

    # Unpack parameters
    if len(params) != 10:
        raise ValueError(f"Wind_Evo requires 10 parameters, got {len(params)}")

    v_circ, Ndot_cloud0, T_cloud, injection_radius, injection_power, config_dict, r0, Edot_per_Vol, Mdot_per_Vol, Lambda_P_rho = params

    # Extract config values
    M_cloud_min = config_dict['M_cloud_min']
    CoolingAreaChiPower = config_dict['CoolingAreaChiPower']
    ColdTurbulenceChiPower = config_dict['ColdTurbulenceChiPower']
    TurbulentVelocityChiPower = config_dict['TurbulentVelocityChiPower']
    geometric_factor = config_dict['geometric_factor']
    Mdot_coefficient = config_dict['Mdot_coefficient']
    Cooling_Factor = config_dict['Cooling_Factor']
    drag_coeff = config_dict['drag_coeff']
    f_turb0 = config_dict['f_turb0']
    Omwind = config_dict['Omwind']
    mu = config_dict['mu']
    metallicity = config_dict.get('metallicity', 1.0)
    redshift = config_dict.get('redshift', 0.0)

    # Use pre-calculated cooling interpolator from params
    # Lambda_P_rho is now params[9]

    # Determine N_cloud_species from state vector length
    # state has: 4 wind vars + 3*N_cloud_species cloud vars
    N_cloud_species = (len(state) - 4) // 3

    # Unpack state vector
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4:4+N_cloud_species]
    v_cloud    = state[4+N_cloud_species:4+2*N_cloud_species]
    Z_cloud    = state[-N_cloud_species:]

    # Early safeguard: if pressure or density becomes unphysical, stop integration
    if Pressure <= 0 or rho_wind <= 0 or v_wind <= 0:
        return np.zeros(4 + 3*N_cloud_species)

    # Ensure arrays for single cloud case
    if N_cloud_species == 1:
        M_cloud = np.atleast_1d(M_cloud)
        v_cloud = np.atleast_1d(v_cloud)
        Z_cloud = np.atleast_1d(Z_cloud)

    # Wind properties
    cs_sq_wind   = gamma * Pressure / rho_wind
    Mach_sq_wind = v_wind**2 / cs_sq_wind
    Z_wind       = rhoZ_wind / rho_wind
    vc           = v_circ  # Simple isothermal potential
    Phir         = v_circ**2 * np.log(r)
    vBsq_wind    = 0.5 * v_wind**2 + (gamma/(gamma-1)) * Pressure/rho_wind + Phir

    # Cloud properties with injection cutoff
    Ndot_cloud = Ndot_cloud0 * np.where(r < injection_radius,
                                        (r/injection_radius)**injection_power,
                                        1.0)

    number_density_cloud = Ndot_cloud / (Omwind * v_cloud * r**2)
    cs_cl_sq = gamma * kb * T_cloud / (mu * mp)
    vBsq_cl = 0.5 * v_cloud**2 + (gamma/(gamma-1)) * cs_cl_sq + Phir

    # Cloud transfer rates
    rho_cloud = Pressure * (mu*mp) / (kb*T_cloud)  # Pressure equilibrium
    chi = rho_cloud / rho_wind

    # Safeguard against negative chi (would cause NaN in power operations)
    if chi <= 0:
        return np.zeros(4 + 3*N_cloud_species)

    # Compute radii only for physically valid cloud masses to avoid
    # invalid fractional powers during transient integration states.
    r_cloud = np.zeros_like(M_cloud, dtype=float)
    valid_radius = M_cloud > 0
    r_cloud[valid_radius] = (M_cloud[valid_radius] / (4*np.pi/3. * rho_cloud))**(1/3.)
    r_cloud_safe = np.where(r_cloud > 0, r_cloud, np.inf)
    v_rel = v_wind - v_cloud
    v_turb = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind = Pressure/kb * (mu*mp/rho_wind)
    T_mix = np.sqrt(T_wind * T_cloud)
    Z_mix = np.sqrt(Z_wind * Z_cloud)

    # Cooling time with proper handling
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, 0.0, mu)
    if np.isscalar(t_cool_layer):
        t_cool_layer = np.full_like(M_cloud, t_cool_layer)
    t_cool_layer = np.where(t_cool_layer < 0, 1e10*Myr, t_cool_layer)

    # Add small epsilon to prevent division by zero when v_turb = 0
    ksi = r_cloud / (np.maximum(v_turb, 1e-10) * t_cool_layer)
    AreaBoost = geometric_factor * chi**CoolingAreaChiPower
    v_turb_cold = v_turb * chi**ColdTurbulenceChiPower

    # Mass transfer rates (Mdot_loss is negative!)
    # Only calculate for clouds above minimum mass
    cloud_active = M_cloud > M_cloud_min
    Mdot_grow = np.where(
        cloud_active,
        Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud_safe * chi) *
        np.where(ksi < 1, ksi**0.5, ksi**0.25),
        0
    )
    Mdot_loss = np.where(
        cloud_active,
        Mdot_coefficient * 3.0 * -M_cloud * v_turb_cold / r_cloud_safe,
        0
    )
    Mdot_cloud = Mdot_grow + Mdot_loss

    # Galaxy source terms (SN feedback)
    if r < r0:
        Mdot_SN = Mdot_per_Vol
        Edot_SN = Edot_per_Vol
    else:
        Mdot_SN = 0.0
        Edot_SN = 0.0

    # Density source (galaxy + clouds)
    drhodt = Mdot_SN - 1.0 * np.sum(number_density_cloud * Mdot_cloud)

    # Momentum source
    # Fix: Use v_rel * |v_rel| to preserve sign (drag opposes relative motion)
    p_dot_ram = 0.5 * drag_coeff * rho_wind * np.pi * v_rel * np.abs(v_rel) * r_cloud**2
    p_dot_transfer = v_wind*Mdot_grow + v_cloud*Mdot_loss
    dpdt = -1.0 * np.sum(number_density_cloud * (p_dot_transfer + p_dot_ram))

    # Energy source (galaxy + clouds + cooling)
    e_dot_cool = 0.0 if (Cooling_Factor == 0) else -(rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure, rho_wind))
    e_dot_transfer = vBsq_wind*Mdot_grow + vBsq_cl*Mdot_loss
    dedt = Edot_SN - 1.0 * np.sum(number_density_cloud * (e_dot_transfer + p_dot_ram*v_wind)) + e_dot_cool

    # Metallicity source
    drhoZdt = -1.0 * np.sum(number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients with regularization near sonic point
    # The factor (1 - 1/M²) causes a singularity at M = 1
    # Apply regularization when close to sonic
    sonic_regularization_width = 0.01  # Width of regularization region
    
    # Check if we're near the sonic point
    if np.abs(Mach_sq_wind - 1.0) < sonic_regularization_width:
        # Near sonic point - use L'Hôpital's rule or Taylor expansion
        # For small ε where M² = 1 + ε, (1 - 1/M²) ≈ ε/(1+ε) ≈ ε
        epsilon = Mach_sq_wind - 1.0
        if np.abs(epsilon) < 1e-10:
            epsilon = 1e-10 * np.sign(epsilon) if epsilon != 0 else 1e-10
        denominator = epsilon
    else:
        # Far from sonic - use standard formula
        denominator = 1.0 - (1.0/Mach_sq_wind)
    
    # Apply denominator with regularization
    dv_dr    = (v_wind/r)/denominator * ( 2.0/Mach_sq_wind - (vc/v_wind)**2
                - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. - gamma * dpdt/v_wind + (gamma-1)*dedt/v_wind**2 - (gamma-1) * (Phir/v_wind**2)*drhodt))
    drho_dr  = (rho_wind/r)/denominator * ( -2.0 + (vc/v_wind)**2
                + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. - gamma * dpdt/v_wind + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind + (gamma-1) * (Phir/v_wind**2)*drhodt))
    drhoZ_dr = ((rhoZ_wind/r)/denominator * ( -2.0 + (vc/v_wind)**2
                + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. - gamma * dpdt/v_wind + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind + (gamma-1) * (Phir/v_wind**2)*drhodt))
                + (rhoZ_wind/r)*(1/(rho_wind*v_wind/r))*((drhoZdt/Z_wind)-drhodt))
    dP_dr    = (Pressure/r)*gamma/denominator * ( -2.0 + (vc/v_wind)**2
                + 1/(rho_wind*v_wind/r) * (drhodt + drhodt * (gamma-1)/2.*Mach_sq_wind * (1 - 2.0 * Phir/v_wind**2) - dpdt/v_wind + (gamma-1)*Mach_sq_wind*(dedt-v_wind*dpdt)/v_wind**2))

    # Cloud gradients - set all to 0 for clouds below minimum mass
    dM_cloud_dr = np.where(cloud_active, Mdot_cloud / v_cloud, 0)

    dv_cloud_dr = np.where(cloud_active,
                          (p_dot_ram + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud),
                          0)

    dZ_cloud_dr = np.where(cloud_active,
                          (Z_wind - Z_cloud) * Mdot_grow / (M_cloud * v_cloud),
                          0)

    # Return derivatives
    if N_cloud_species == 1:
        derivatives = np.r_[dv_dr, drho_dr, dP_dr, drhoZ_dr, dM_cloud_dr[0], dv_cloud_dr[0], dZ_cloud_dr[0]]
    else:
        derivatives = np.concatenate([
            [dv_dr, drho_dr, dP_dr, drhoZ_dr],
            dM_cloud_dr,
            dv_cloud_dr,
            dZ_cloud_dr
        ])

    # If any derivatives are NaN, return zeros to stop integration gracefully
    if np.any(np.isnan(derivatives)):
        return np.zeros_like(derivatives)

    return derivatives


def Hot_Wind_Evo(r, state, params):
    """
    Compute hot-only wind evolution derivatives.

    This is used for comparison with the multiphase solution. By default,
    no source terms are included (pure adiabatic wind).

    Parameters:
    params : tuple
        (v_circ,) - circular velocity [cm/s]
        Optional: (v_circ, include_source_terms, r0, Edot_per_Vol, Mdot_per_Vol)
    """
    # Unpack parameters
    v_circ = params[0]

    # Check if source terms are requested (for special cases)
    if len(params) > 1:
        include_source_terms = params[1]
        r0 = params[2]
        Edot_per_Vol = params[3]
        Mdot_per_Vol = params[4]
    else:
        include_source_terms = False
        r0 = 0
        Edot_per_Vol = 0
        Mdot_per_Vol = 0

    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    vc           = v_circ  # Simple isothermal potential
    Phir         = v_circ**2 * np.log(r)
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy (only if requested)
    if include_source_terms:
        Edot_SN = Edot_per_Vol * np.where(r<r0, 1.0, 0.0)
        Mdot_SN = Mdot_per_Vol * np.where(r<r0, 1.0, 0.0)
    else:
        Edot_SN = 0.0
        Mdot_SN = 0.0

    # density
    drhodt          = Mdot_SN

    # momentum
    dpdt            = 0

    # energy
    dedt            = Edot_SN

    # Regularize near sonic point
    sonic_denom = 1.0-(1.0/Mach_sq_wind)
    epsilon = 1e-5
    if abs(sonic_denom) < epsilon:
        sonic_denom = np.sign(sonic_denom) * epsilon

    dv_dr    = (v_wind/r)/sonic_denom * ( 2.0/Mach_sq_wind - (vc/v_wind)**2 - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. + (gamma-1)*dedt/v_wind**2))
    drho_dr  = (rho_wind/r)/sonic_denom * ( -2.0 + (vc/v_wind)**2 + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind))
    dP_dr    = (Pressure/r)*gamma/sonic_denom * ( -2.0 + (vc/v_wind)**2 + 1/(rho_wind*v_wind/r) * (drhodt + drhodt * (gamma-1)/2.*Mach_sq_wind + (gamma-1)*Mach_sq_wind*dedt/v_wind**2))

    return np.r_[dv_dr, drho_dr, dP_dr]


# ----------------------------------------------------------------------------
# Event detection functions for galactic wind integration
#
# All events follow the factory pattern: create_X_event(params) returns an
# event function compatible with scipy.integrate.solve_ivp
#
# Return value convention:
# - Positive: event condition not met
# - Zero/Negative: event triggered
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Flow Condition Events - Monitor wind flow properties
# ----------------------------------------------------------------------------

def create_supersonic_event(params):
    """Create supersonic event function with captured parameters.

    Triggers when flow transitions to supersonic (Mach > 1 + tolerance).

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo containing:
        (v_circ, Ndot_cloud0, T_cloud, injection_radius, injection_power,
         config_dict, r0, Edot_per_Vol, Mdot_per_Vol, Lambda_P_rho)

    Returns
    -------
    supersonic : function
        Event function that returns negative when Mach > 1 + tolerance
    """
    config_dict = params[5]
    sonic_tolerance = config_dict['sonic_transition_tolerance']

    def supersonic(r, state):
        v_wind = state[0]
        rho_wind = state[1]
        P_wind = state[2]
        cs_sq = gamma * P_wind / rho_wind
        mach = v_wind / np.sqrt(cs_sq)
        return mach - (1.0 + sonic_tolerance)

    supersonic.terminal = True
    supersonic.direction = 0  # Detect crossing in either direction
    return supersonic


def create_subsonic_event(params):
    """Create subsonic event function with captured parameters.

    Triggers when flow transitions to subsonic (Mach < 1 - tolerance).

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo

    Returns
    -------
    subsonic : function
        Event function that returns negative when Mach < 1 - tolerance
    """
    config_dict = params[5]
    sonic_tolerance = config_dict['sonic_transition_tolerance']

    def subsonic(r, state):
        v_wind = state[0]
        rho_wind = state[1]
        P_wind = state[2]
        cs_sq = gamma * P_wind / rho_wind
        mach = v_wind / np.sqrt(cs_sq)
        return mach - (1.0 - sonic_tolerance)

    subsonic.terminal = True
    subsonic.direction = 0
    return subsonic


def create_wind_negative_event(params):
    """Create wind_negative event function.

    Triggers when wind velocity becomes negative (unphysical).

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo

    Returns
    -------
    wind_negative : function
        Event function that returns negative when v_wind < 0
    """
    def wind_negative(r, state):
        return state[0]  # v_wind

    wind_negative.terminal = True
    wind_negative.direction = -1  # Only trigger when crossing from positive to negative
    return wind_negative


def create_cold_wind_event(params):
    """Create cold_wind event function with captured parameters.

    Triggers when hot wind temperature drops to near cloud temperature.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo

    Returns
    -------
    cold_wind : function
        Event function that returns negative when T_wind ~ T_cloud
    """
    T_cloud = params[2]
    config_dict = params[5]
    mu = config_dict['mu']
    sonic_tolerance = config_dict['sonic_transition_tolerance']

    def cold_wind(r, state):
        rho_wind = state[1]
        P_wind = state[2]
        # Calculate wind sound speed and cloud sound speed
        cs_wind_sq = gamma * P_wind / rho_wind
        cs_cloud_sq = gamma * kb * T_cloud / (mu * mp)
        # Trigger when wind sound speed approaches cloud sound speed
        return np.sqrt(cs_wind_sq / cs_cloud_sq) - (1.0 + sonic_tolerance)

    cold_wind.terminal = True
    return cold_wind

# ----------------------------------------------------------------------------
# Cloud Property Events - Monitor cloud evolution
# ----------------------------------------------------------------------------

def create_cloud_density_low_event(params, density_threshold=1e-50):
    """Create cloud_density_low event function with captured parameters.

    Triggers when cloud number density drops below threshold.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo
    density_threshold : float, optional
        Minimum cloud number density (cm^-3). Default: 1e-50

    Returns
    -------
    cloud_density_low : function
        Event function that returns negative when density < threshold
    """
    # Extract needed parameters
    Ndot_cloud0 = params[1]
    injection_radius = params[3]
    injection_power = params[4]
    config_dict = params[5]
    Omwind = config_dict['Omwind']
    M_cloud_min = config_dict['M_cloud_min']

    def cloud_density_low(r, state):
        # Determine N_cloud_species from state vector
        N_cloud_species = (len(state) - 4) // 3

        v_cloud = state[4+N_cloud_species:4+2*N_cloud_species]
        M_cloud = state[4:4+N_cloud_species]

        # Ensure arrays
        if N_cloud_species == 1:
            v_cloud = np.atleast_1d(v_cloud)
            M_cloud = np.atleast_1d(M_cloud)
            Ndot_cloud0_arr = np.atleast_1d(Ndot_cloud0)
        else:
            Ndot_cloud0_arr = Ndot_cloud0

        # Calculate cloud number densities
        Ndot_cloud = Ndot_cloud0_arr * np.where(r < injection_radius,
                                               (r/injection_radius)**injection_power,
                                               1.0)
        number_density_cloud = Ndot_cloud / (Omwind * v_cloud * r**2)

        # Only check active clouds
        active_clouds = M_cloud > M_cloud_min
        if np.any(active_clouds):
            min_density = np.min(number_density_cloud[active_clouds])
            return min_density - density_threshold
        return 1.0

    cloud_density_low.terminal = True
    cloud_density_low.direction = -1
    return cloud_density_low

def create_all_clouds_frozen_event(params):
    """Create all_clouds_frozen event function with captured parameters.

    Triggers when all clouds drop below minimum mass threshold.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo

    Returns
    -------
    all_clouds_frozen : function
        Event function that returns negative when all M_cloud < M_cloud_min
    """
    config_dict = params[5]
    M_cloud_min = config_dict['M_cloud_min']

    def all_clouds_frozen(r, state):
        # Determine N_cloud_species from state vector
        N_cloud_species = (len(state) - 4) // 3
        M_cloud = state[4:4+N_cloud_species]
        if N_cloud_species == 1:
            M_cloud = np.atleast_1d(M_cloud)
        return np.max(M_cloud) - M_cloud_min

    all_clouds_frozen.terminal = True
    all_clouds_frozen.direction = -1
    return all_clouds_frozen

def create_cloud_velocity_low_event(params):
    """Create cloud_velocity_low event function with captured parameters.

    Triggers when any cloud velocity drops below minimum threshold.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo

    Returns
    -------
    cloud_velocity_low : function
        Event function that returns negative when any v_cloud < v_cloud_min
    """
    config_dict = params[5]
    v_cloud_min_cgs = config_dict['v_cloud_min'] * 1e5  # Convert km/s to cm/s

    def cloud_velocity_low(r, state):
        # Determine N_cloud_species from state vector
        N_cloud_species = (len(state) - 4) // 3
        v_cloud = state[4+N_cloud_species:4+2*N_cloud_species]
        if N_cloud_species == 1:
            v_cloud = np.atleast_1d(v_cloud)
        # Return the difference between minimum cloud velocity and the threshold
        # Negative when any cloud is below threshold
        return np.min(v_cloud) - v_cloud_min_cgs

    cloud_velocity_low.terminal = True
    cloud_velocity_low.direction = -1
    return cloud_velocity_low


# ----------------------------------------------------------------------------
# Unphysical State Detection Events
# ----------------------------------------------------------------------------

def create_nan_state_event(params):
    """Create event to detect NaN in any state variable.

    Terminates integration if any state variable becomes NaN, indicating
    the solution has become unphysical and the integration is stuck.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo (included for consistency)

    Returns
    -------
    nan_state : function
        Event function that triggers when any state variable is NaN
    """
    def nan_state(r, state):
        """Check if any state variable has become NaN."""
        # Check if any element of the state vector is NaN
        if np.any(np.isnan(state)):
            return -1.0  # Trigger event
        return 1.0  # OK

    nan_state.terminal = True
    nan_state.direction = -1
    return nan_state


def create_negative_pressure_event(params):
    """Create event to detect negative pressure.

    Terminates integration if pressure becomes negative, which is unphysical.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo (included for consistency)

    Returns
    -------
    negative_pressure : function
        Event function that triggers when pressure <= 0
    """
    def negative_pressure(r, state):
        """Check if pressure has become negative."""
        # Extract pressure (index 2)
        P = state[2]
        # Return negative when P <= 0 (event triggers when crossing zero)
        if P <= 0:
            return -1.0
        # Check if pressure is getting very small
        if P < 1e-20:
            return P - 1e-20
        return 1.0  # OK

    negative_pressure.terminal = True
    negative_pressure.direction = -1
    return negative_pressure


def create_negative_density_event(params):
    """Create event to detect negative density.

    Terminates integration if hot gas density becomes negative, which is unphysical.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo (included for consistency)

    Returns
    -------
    negative_density : function
        Event function that triggers when rho <= 0
    """
    def negative_density(r, state):
        """Check if density has become negative."""
        # Extract density (index 1)
        rho = state[1]
        # Return negative when rho <= 0
        if rho <= 0:
            return -1.0
        # Check if density is getting very small
        if rho < 1e-30:
            return rho - 1e-30
        return 1.0  # OK

    negative_density.terminal = True
    negative_density.direction = -1
    return negative_density


# ----------------------------------------------------------------------------
# Integration Health Events - Detect stuck integration
# ----------------------------------------------------------------------------

def create_step_size_event(params, min_relative_step=1e-8, n_small_steps=100):
    """Create event to detect when integration is stuck taking tiny steps.

    Terminates integration if it takes too many steps without making progress,
    indicating the solver is stuck on numerical stiffness.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo (included for consistency)
    min_relative_step : float, optional
        Minimum relative step size Δr/r before considering stuck (default: 1e-8)
    n_small_steps : int, optional
        Number of consecutive small steps before terminating (default: 100)

    Returns
    -------
    step_size_event : function
        Event function that triggers when integration is stuck
    """
    import time
    from .constants import kpc

    # Use a class to properly encapsulate state for each instance
    class StepSizeMonitor:
        def __init__(self):
            self.last_r = None
            self.small_step_count = 0
            self.total_steps = 0
            self.start_time = None
            self.last_check_r = None

        def __call__(self, r, y):
            """Check if integration is making progress."""
            self.total_steps += 1

            # Initialize timer on first call
            if self.start_time is None:
                self.start_time = time.time()
                self.last_check_r = r
                return 1.0  # OK on first call

            # Check progress every 5 seconds
            elapsed = time.time() - self.start_time
            if elapsed > 5.0:
                # Check how far we've progressed
                progress = abs(r - self.last_check_r) / kpc
                if progress < 0.001:  # Less than 1 pc progress in 5 seconds
                    # We're stuck
                    return 0.0  # Trigger termination
                # Reset for next check
                self.start_time = time.time()
                self.last_check_r = r

            # Always return positive (no termination) unless stuck
            return 1.0

    # Create a new instance for this event
    step_size_event = StepSizeMonitor()
    step_size_event.terminal = True
    step_size_event.direction = 0  # No direction checking
    return step_size_event

# ----------------------------------------------------------------------------
# Progress Tracking Events - Non-terminating monitoring
# ----------------------------------------------------------------------------

def create_progress_event(params, r_start, r_interval, progress_callback, r_max):
    """Create progress reporting event function.

    Non-terminating event that reports integration progress at regular intervals.

    Parameters
    ----------
    params : tuple
        Full parameter tuple passed to Wind_Evo (included for consistency)
    r_start : float
        Starting radius in cm
    r_interval : float
        Radius interval between progress reports in cm
    progress_callback : callable
        Function to call with (r_current, r_max, n_steps)
    r_max : float
        Maximum radius in cm

    Returns
    -------
    progress_event : function
        Non-terminating event function for progress tracking
    """
    # Track progress state
    progress_state = {
        'last_r': r_start,
        'n_steps': 0,
        'next_r': r_start + r_interval
    }

    def progress_event(r, state):
        progress_state['n_steps'] += 1

        # Check if we've passed the next reporting radius
        if r >= progress_state['next_r']:
            if progress_callback is not None:
                progress_callback(r, r_max, progress_state['n_steps'])
            progress_state['next_r'] += r_interval

        # Return positive value (event never triggers)
        return 1.0

    progress_event.terminal = False
    progress_event.direction = 0  # No direction checking
    return progress_event
