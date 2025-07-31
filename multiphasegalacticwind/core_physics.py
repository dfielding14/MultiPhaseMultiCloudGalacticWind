"""
Multiphase Wind Evolution - Multicloud Version

This module extends the single-cloud wind evolution model to handle multiple cloud species
with a power-law mass distribution: dN/dM ∝ M^-α

Based on Fielding & Bryan "The Structure of Multiphase Galactic Winds"
"""

import numpy as np
import glob
from scipy import integrate, interpolate
import h5py
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy import optimize
import cmasher as cmr

# Import physical constants
from .constants import *
import time

# Import cooling functions
from .cooling import (
    tcool_P,
    get_tcool_min_interpolators
)

# Cooling functions are now imported from cooling.py module

# Numerical tolerance for event detection
epsilon = 1e-1


def Field_Length(state, f_spitzer=1, mu=0.62):
    """Calculate the Field length for thermal conduction."""
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    T_wind       = Pressure/kb/(rho_wind/(mu*mp))
    kappa        = 5.0e-7 * T_wind**2.5
    # Get minimum cooling time interpolator
    _, tcool_min_P = get_tcool_min_interpolators()
    edot_cool    = 1.5*Pressure / tcool_min_P((Pressure/kb,Z_wind/Z_solar))   
    return np.sqrt(f_spitzer*kappa*T_wind / edot_cool)


def Field_Length_mix(state, T_cloud, f_spitzer=1, mu=0.62):
    """Calculate the Field length for the mixed layer."""
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    T_wind       = Pressure/kb/(rho_wind/(mu*mp))
    Z_cloud      = state[6]

    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, 0.0, mu)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    kappa = 5.0e-7 * T_wind**2.5
    edot_cool = 1.5*Pressure / t_cool_layer   
    return np.sqrt(f_spitzer*kappa*T_wind / (edot_cool))


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
        Initial cloud masses for each species (Msun)
    eta_M_cold : array
        Mass loading factor for each cloud species
    Mdot_cold0 : array
        Mass flux for each cloud species (Msun/yr)
    Ndot_cloud0 : array
        Number flux for each cloud species (1/yr)
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

def cloud_radius(r, state, N_cloud_species=1):
    v_wind       = state[0]
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    
    if N_cloud_species == 1:
        # Single cloud case for backward compatibility
        M_cloud      = state[4]
        v_cloud      = state[5]
        Z_cloud      = state[6]
    else:
        # Multicloud case
        M_cloud      = state[4:4+N_cloud_species]
        v_cloud      = state[4+N_cloud_species:4+2*N_cloud_species]
        Z_cloud      = state[-N_cloud_species:]
    
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    return r_cloud

def cloud_ksi(r, state, N_cloud_species=1):
    v_wind       = state[0]
    rho_wind     = state[1]
    Pressure     = state[2]
    rhoZ_wind    = state[3]
    Z_wind       = rhoZ_wind/rho_wind
    
    if N_cloud_species == 1:
        # Single cloud case for backward compatibility
        M_cloud      = state[4]
        v_cloud      = state[5]
        Z_cloud      = state[6]
    else:
        # Multicloud case
        M_cloud      = state[4:4+N_cloud_species]
        v_cloud      = state[4+N_cloud_species:4+2*N_cloud_species]
        Z_cloud      = state[-N_cloud_species:]
        
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    
    # Handle Z_mix for arrays
    if N_cloud_species == 1:
        Z_mix = (Z_wind*Z_cloud)**0.5
        t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, 0.0, mu)[()] 
    else:
        # For multiple clouds, we need to handle each species
        t_cool_layer = np.zeros_like(Z_cloud)
        for i in range(N_cloud_species):
            Z_mix_i = (Z_wind*Z_cloud[i])**0.5
            t_cool_layer[i] = tcool_P(T_mix, Pressure/kb, Z_mix_i/Z_solar, 0.0, mu)[()] 
    
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    return ksi




def Cooling_and_Acceleration(r, state):
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0  # Simple isothermal potential
    Phir         = v_circ0**2 * np.log(r) 
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)

    # cloud properties
    Ndot_cloud              = Ndot_cloud0 * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kb*T_cloud/(mu*mp)
    vBsq_cl                 = 0.5 * v_cloud**2 + (gamma / (gamma-1)) * cs_cl_sq + Phir

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, 0.0, mu)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = geometric_factor * chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 *  M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * -M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow + Mdot_loss, 0)

    # density
    drhodt      = -1.0 * (number_density_cloud * Mdot_cloud)
    drhodt_plus     = -1.0 * (number_density_cloud * Mdot_loss)
    drhodt_minus    = -1.0 * (number_density_cloud * Mdot_grow) 

    # momentum
    p_dot_drag      = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    p_dot_transfer  = v_wind*Mdot_grow + v_cloud*Mdot_loss
    dpdt            = -1.0 * (number_density_cloud * (p_dot_transfer + p_dot_drag))
    dpdt_p          = -1.0 * (number_density_cloud * v_cloud*Mdot_loss)
    dpdt_m          = -1.0 * (number_density_cloud * v_wind*Mdot_grow)
    dpdt_drag       = -1.0 * (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool        = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure,rho_wind))
    e_dot_transfer    = vBsq_wind*Mdot_grow + vBsq_cl*Mdot_loss
    dedt              = -1.0 * (number_density_cloud * (e_dot_transfer + p_dot_drag*v_cloud)) - e_dot_cool
    dv_cloud_dr       = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud) * np.where(M_cloud>M_cloud_min, 1, 0)
    return number_density_cloud * e_dot_transfer, e_dot_cool, dv_cloud_dr


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
    r_cloud = (M_cloud / (4*np.pi/3. * rho_cloud))**(1/3.)
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
    Mdot_grow = np.where(cloud_active,
                        Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where(ksi < 1, ksi**0.5, ksi**0.25),
                        0)
    Mdot_loss = np.where(cloud_active,
                        Mdot_coefficient * 3.0 * -M_cloud * v_turb_cold / r_cloud,
                        0)
    Mdot_cloud = Mdot_grow + Mdot_loss

    # Density source
    drhodt = -1.0 * np.sum(number_density_cloud * Mdot_cloud)
    
    # Momentum source
    p_dot_ram = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2
    p_dot_transfer = v_wind*Mdot_grow + v_cloud*Mdot_loss
    dpdt = -1.0 * np.sum(number_density_cloud * (p_dot_transfer + p_dot_ram))
    
    # Energy source
    e_dot_cool = 0.0 if (Cooling_Factor == 0) else -(rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure, rho_wind))
    e_dot_transfer = vBsq_wind*Mdot_grow + vBsq_cl*Mdot_loss
    dedt = -1.0 * np.sum(number_density_cloud * (e_dot_transfer + p_dot_ram*v_wind)) + e_dot_cool
    
    # Metallicity source
    drhoZdt = -1.0 * np.sum(number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients
    dv_dr    = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( 2.0/Mach_sq_wind - (vc/v_wind)**2 
                - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. - gamma * dpdt/v_wind + (gamma-1)*dedt/v_wind**2 - (gamma-1) * (Phir/v_wind**2)*drhodt)) 
    drho_dr  = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + (vc/v_wind)**2 
                + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. - gamma * dpdt/v_wind + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind + (gamma-1) * (Phir/v_wind**2)*drhodt)) 
    drhoZ_dr = ((rhoZ_wind/r)/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + (vc/v_wind)**2 
                + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. - gamma * dpdt/v_wind + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind + (gamma-1) * (Phir/v_wind**2)*drhodt))
                + (rhoZ_wind/r)*(1/(rho_wind*v_wind/r))*((drhoZdt/Z_wind)-drhodt))
    dP_dr    = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * ( -2.0 + (vc/v_wind)**2 
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
        return np.r_[dv_dr, drho_dr, dP_dr, drhoZ_dr, dM_cloud_dr[0], dv_cloud_dr[0], dZ_cloud_dr[0]]
    else:
        return np.concatenate([
            [dv_dr, drho_dr, dP_dr, drhoZ_dr],
            dM_cloud_dr,
            dv_cloud_dr,
            dZ_cloud_dr
        ])

def Gradient_Components(r, state):
    v_wind     = state[0]
    rho_wind   = state[1]
    Pressure   = state[2]
    rhoZ_wind  = state[3]
    M_cloud    = state[4]
    v_cloud    = state[5]
    Z_cloud    = state[6]

    # wind properties
    cs_sq_wind   = (gamma*Pressure/rho_wind)
    Mach_sq_wind = (v_wind**2 / cs_sq_wind)
    Z_wind       = rhoZ_wind/rho_wind
    vc           = v_circ0  # Simple isothermal potential
    Phir         = v_circ0**2 * np.log(r) 
    # vc           = v_circ0 * np.where(r<r0, (r/r0)**2, 1.0)
    # Phir         = v_circ0**2 * np.where(r<r0, (1/3.) * (r/r0)**3, np.log(r)) 
    vBsq_wind    = 0.5 * v_wind**2 + (gamma / (gamma-1)) * Pressure/rho_wind + Phir

    # source term from inside galaxy
    Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)
    Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0) #* np.where(r<r0, 1.0, 0.0)

    # cloud properties
    Ndot_cloud              = Ndot_cloud0 * np.where(r<cold_cloud_injection_radial_extent, (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud    = Ndot_cloud/(Omwind * v_cloud * r**2)
    cs_cl_sq                = gamma * kb*T_cloud/(mu*mp)
    vBsq_cl                 = 0.5 * v_cloud**2 + (gamma / (gamma-1)) * cs_cl_sq + Phir

    # cloud transfer rates
    rho_cloud    = Pressure * (mu*mp) / (kb*T_cloud) # cloud in pressure equilibrium
    chi          = rho_cloud / rho_wind
    r_cloud      = (M_cloud / ( 4*np.pi/3. * rho_cloud))**(1/3.) 
    v_rel        = (v_wind-v_cloud)
    v_turb       = f_turb0 * v_rel * chi**TurbulentVelocityChiPower
    T_wind       = Pressure/kb * (mu*mp/rho_wind)
    T_mix        = (T_wind*T_cloud)**0.5
    Z_mix        = (Z_wind*Z_cloud)**0.5
    t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, 0.0, mu)[()] 
    t_cool_layer = np.where(t_cool_layer<0, 1e10*Myr, t_cool_layer)
    ksi          = r_cloud / (v_turb * t_cool_layer)
    AreaBoost    = geometric_factor * chi**CoolingAreaChiPower
    v_turb_cold  = v_turb * chi**ColdTurbulenceChiPower
    Mdot_grow    = Mdot_coefficient * 3.0 * M_cloud * v_turb * AreaBoost / (r_cloud * chi) * np.where( ksi < 1, ksi**0.5, ksi**0.25 )
    Mdot_loss    = Mdot_coefficient * 3.0 * -M_cloud * v_turb_cold / r_cloud 
    Mdot_cloud   = np.where(M_cloud > M_cloud_min, Mdot_grow + Mdot_loss, 0)

    # density
    drhodt       = -1.0 * (number_density_cloud * Mdot_cloud)  # Net change (negative)
    drhodt_plus  = (number_density_cloud * (-Mdot_loss))       # Mass added to wind (positive)
    drhodt_minus = (number_density_cloud * Mdot_grow)          # Mass removed from wind (positive) 

    # momentum
    p_dot_drag   = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2 * np.where(M_cloud>M_cloud_min, 1, 0)
    dpdt_drag    = (number_density_cloud * p_dot_drag)
    
    # energy
    e_dot_cool   = 0.0 if (Cooling_Factor==0) else (rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure,rho_wind))

    # metallicity
    drhoZdt         = -1.0 * (number_density_cloud * (Z_wind*Mdot_grow + Z_cloud*Mdot_loss))

    # wind gradients
    # velocity
    dv_dr       = 2/Mach_sq_wind
    dv_dr      += - (vc/v_wind)**2
    dv_dr      +=  drhodt_minus/(rho_wind*v_wind/r) *  (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr      += -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr      += (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr      += -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr      += -dpdt_drag/(rho_wind*v_wind**2/r)
    # Regularize near sonic point
    sonic_denom = 1.0-(1.0/Mach_sq_wind)
    epsilon = 1e-5
    if abs(sonic_denom) < epsilon:
        sonic_denom = np.sign(sonic_denom) * epsilon
    dv_dr      *= (v_wind/r)/sonic_denom
    
    dv_dr_1      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * 2/Mach_sq_wind
    dv_dr_2      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * - (vc/v_wind)**2
    dv_dr_3      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) *  drhodt_minus/(rho_wind*v_wind/r) *  (1/Mach_sq_wind)
    dv_dr_4      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * (1/Mach_sq_wind)
    dv_dr_5      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dv_dr_6      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    dv_dr_7      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    dv_dr_8      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * (gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    dv_dr_9      = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -(gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    dv_dr_10     = (v_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -dpdt_drag/(rho_wind*v_wind**2/r)

    # density
    drho_dr       = -2
    drho_dr      += (vc/v_wind)**2
    drho_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r)
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    drho_dr      *= (rho_wind/r)/sonic_denom

    drho_dr_1      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -2
    drho_dr_2      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * (vc/v_wind)**2
    drho_dr_3      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -drhodt_minus/(rho_wind*v_wind/r)
    drho_dr_4      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r)
    drho_dr_5      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    drho_dr_6      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.*(v_rel/v_wind)**2
    drho_dr_7      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/v_wind**2)
    drho_dr_8      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * -(gamma-1)*e_dot_cool/(rho_wind*v_wind**3/r)
    drho_dr_9      = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind**3/r)
    drho_dr_10     = (rho_wind/r)/(1.0-(1.0/Mach_sq_wind)) * dpdt_drag/(rho_wind*v_wind**2/r)
    
    # pressure
    dP_dr       = -2
    dP_dr      += (vc/v_wind)**2
    dP_dr      += -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr      += drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr      += -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr      += dpdt_drag/(rho_wind*v_wind**2/r)
    dP_dr      *= (Pressure/r)*gamma/sonic_denom

    dP_dr_1       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * -2
    dP_dr_2       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * (vc/v_wind)**2
    dP_dr_3       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * -drhodt_minus/(rho_wind*v_wind/r)
    dP_dr_4       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r)
    dP_dr_5       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * v_rel/v_wind
    dP_dr_6       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (gamma-1)/2.* (v_rel**2 / cs_sq_wind)
    dP_dr_7       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind - cs_cl_sq)/cs_sq_wind)
    dP_dr_8       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * -(gamma-1)*e_dot_cool/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr_9       = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * (gamma-1)*dpdt_drag*v_rel/(rho_wind*v_wind*cs_sq_wind/r)
    dP_dr_10      = (Pressure/r)*gamma/(1.0-(1.0/Mach_sq_wind)) * dpdt_drag/(rho_wind*v_wind**2/r)

    # entropy
    K = (Pressure/kb) / (rho_wind/(mu*mp))**gamma
    dK_dr   = (K/r) * (gamma * drhodt_plus/(rho_wind*v_wind/r) * ((gamma-1)/2. * (v_rel**2/cs_sq_wind) - (cs_sq_wind-cs_cl_sq)/cs_sq_wind) - (r/v_wind) * (e_dot_cool)/Pressure * (gamma-1) + (r/v_wind) * dpdt_drag*v_rel/Pressure * (gamma-1))  
    dK_dr_1 = (K/r) * (gamma * drhodt_plus/(rho_wind*v_wind/r) * ((gamma-1)/2. * (v_rel**2/cs_sq_wind)))
    dK_dr_2 = (K/r) * (gamma * drhodt_plus/(rho_wind*v_wind/r) * (-(cs_sq_wind-cs_cl_sq)/cs_sq_wind))
    dK_dr_3 = (K/r) * (-(r/v_wind) * (e_dot_cool)/Pressure * (gamma-1))
    dK_dr_4 = (K/r) * ((r/v_wind) * dpdt_drag*v_rel/Pressure * (gamma-1))

    # cloud gradients
    dM_cloud_dr   = Mdot_cloud/v_cloud* np.where(M_cloud>M_cloud_min, 1, 0)
    dM_cloud_dr_1 = Mdot_grow/v_cloud * np.where(M_cloud>M_cloud_min, 1, 0)
    dM_cloud_dr_2 = -Mdot_loss/v_cloud * np.where(M_cloud>M_cloud_min, 1, 0)

    dv_cloud_dr   = (p_dot_drag + v_rel*Mdot_grow - M_cloud * vc**2/r) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)
    dv_cloud_dr_1 = (v_rel*Mdot_grow) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)
    dv_cloud_dr_2 = (p_dot_drag) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)
    dv_cloud_dr_3 = (-M_cloud * vc**2/r) / (M_cloud * v_cloud)* np.where(M_cloud>M_cloud_min, 1, 0)

    return [[dv_dr,dv_dr_1,dv_dr_2,dv_dr_3,dv_dr_4,dv_dr_5,dv_dr_6,dv_dr_7,dv_dr_8,dv_dr_9,dv_dr_10],
            [drho_dr,drho_dr_1,drho_dr_2,drho_dr_3,drho_dr_4,drho_dr_5,drho_dr_6,drho_dr_7,drho_dr_8,drho_dr_9,drho_dr_10],
            [dP_dr,dP_dr_1,dP_dr_2,dP_dr_3,dP_dr_4,dP_dr_5,dP_dr_6,dP_dr_7,dP_dr_8,dP_dr_9,dP_dr_10],
            [dK_dr,dK_dr_1,dK_dr_2,dK_dr_3,dK_dr_4],
            [dM_cloud_dr,dM_cloud_dr_1,dM_cloud_dr_2],
            [dv_cloud_dr,dv_cloud_dr_1,dv_cloud_dr_2,dv_cloud_dr_3]]


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
    
    dv_dr    = (v_wind/r)/sonic_denom * ( 2.0/Mach_sq_wind - 1/(rho_wind*v_wind/r) * (drhodt*(gamma+1)/2. + (gamma-1)*dedt/v_wind**2)) 
    drho_dr  = (rho_wind/r)/sonic_denom * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt*(gamma+3)/2. + (gamma-1)*dedt/v_wind**2 - drhodt/Mach_sq_wind)) 
    dP_dr    = (Pressure/r)*gamma/sonic_denom * ( -2.0 + 1/(rho_wind*v_wind/r) * (drhodt + drhodt * (gamma-1)/2.*Mach_sq_wind + (gamma-1)*Mach_sq_wind*dedt/v_wind**2))

    return np.r_[dv_dr, drho_dr, dP_dr]


def supersonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 + epsilon)

supersonic.terminal = True

def subsonic(r,z):
    return z[0]/np.sqrt(gamma*z[2]/z[1]) - (1.0 - epsilon)

subsonic.terminal = True

def create_cold_wind_event(T_cloud, mu):
    """Create cold_wind event function with captured parameters."""
    def cold_wind(r, z):
        return np.sqrt(gamma*z[2]/z[1])/np.sqrt(gamma*kb*T_cloud/(mu*mp)) - (1.0 + epsilon)
    cold_wind.terminal = True
    return cold_wind


def cloud_stop(r,z):
    return z[5] - 10e5

cloud_stop.terminal = True

# Additional termination condition functions for multicloud
def supersonic(r, state):
    """Detect transition to supersonic flow"""
    v_wind = state[0]
    rho_wind = state[1]
    P_wind = state[2]
    return v_wind/np.sqrt(gamma*P_wind/rho_wind) - (1.0 + 0.1)
supersonic.terminal = True

def subsonic(r, state):
    """Detect transition to subsonic flow"""
    v_wind = state[0]
    rho_wind = state[1]
    P_wind = state[2]
    return v_wind/np.sqrt(gamma*P_wind/rho_wind) - (1.0 - 0.1)
subsonic.terminal = True

def wind_negative(r, state):
    """Terminate if wind velocity goes negative"""
    return state[0]
wind_negative.terminal = True
wind_negative.direction = -1

def create_cloud_density_low_event(Ndot_cloud0, cold_cloud_injection_radial_extent, 
                                  cold_cloud_injection_radial_power, Omwind, M_cloud_min,
                                  density_threshold=1e-50):
    """Create cloud_density_low event function with captured parameters."""
    def cloud_density_low(r, state):
        """Terminate if cloud number density gets too low"""
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
        Ndot_cloud = Ndot_cloud0_arr * np.where(r < cold_cloud_injection_radial_extent,
                                               (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power,
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

def create_all_clouds_frozen_event(M_cloud_min):
    """Create all_clouds_frozen event function with captured parameters."""
    def all_clouds_frozen(r, state):
        """Terminate if all clouds drop below minimum mass"""
        # Determine N_cloud_species from state vector
        N_cloud_species = (len(state) - 4) // 3
        M_cloud = state[4:4+N_cloud_species]
        if N_cloud_species == 1:
            M_cloud = np.atleast_1d(M_cloud)
        return np.max(M_cloud) - M_cloud_min
    
    all_clouds_frozen.terminal = True
    all_clouds_frozen.direction = -1
    return all_clouds_frozen

def calculate_cloud_moments(r, state):
    """
    Calculate mass, momentum, and energy-weighted moments of the cloud population
    
    Parameters:
    -----------
    r : float
        Radius
    state : array
        State vector
        
    Returns:
    --------
    moments : dict
        Dictionary containing various moment calculations
    """
    # Determine N_cloud_species from state vector
    N_cloud_species = (len(state) - 4) // 3
    
    # Extract state variables
    v_wind = state[0]
    rho_wind = state[1]
    Pressure = state[2]
    
    if N_cloud_species == 1:
        M_cloud = np.array([state[4]])
        v_cloud = np.array([state[5]])
        Z_cloud = np.array([state[6]])
    else:
        M_cloud = state[4:4+N_cloud_species]
        v_cloud = state[4+N_cloud_species:4+2*N_cloud_species]
        Z_cloud = state[-N_cloud_species:]
    
    # Get Ndot_cloud0 from global scope
    Ndot_cloud0 = globals().get('Ndot_cloud0', np.ones(N_cloud_species) * 1e-5 / yr)
    if N_cloud_species == 1:
        Ndot_cloud0 = np.atleast_1d(Ndot_cloud0)
    
    # Calculate cloud properties
    Ndot_cloud = Ndot_cloud0 * np.where(r<cold_cloud_injection_radial_extent, 
                                        (r/cold_cloud_injection_radial_extent)**cold_cloud_injection_radial_power, 1.0)
    number_density_cloud = Ndot_cloud/(Omwind * v_cloud * r**2)
    
    # Mask for existing clouds
    cloud_exists = M_cloud > M_cloud_min
    
    # Calculate moments
    moments = {}
    
    # Total mass flux
    Mdot_cloud = number_density_cloud * M_cloud * v_cloud
    moments['Mdot_cloud_tot'] = np.sum(Mdot_cloud * cloud_exists)
    
    # Mass-weighted average velocity
    if moments['Mdot_cloud_tot'] > 0:
        moments['v_cloud_mass_avg'] = np.sum(Mdot_cloud * v_cloud * cloud_exists) / moments['Mdot_cloud_tot']
    else:
        moments['v_cloud_mass_avg'] = 0.0
    
    # Mass-weighted velocity dispersion
    if moments['Mdot_cloud_tot'] > 0:
        v_cloud_sq_avg = np.sum(Mdot_cloud * v_cloud**2 * cloud_exists) / moments['Mdot_cloud_tot']
        moments['sigma_v_cloud'] = np.sqrt(v_cloud_sq_avg - moments['v_cloud_mass_avg']**2)
    else:
        moments['sigma_v_cloud'] = 0.0
    
    # Number-weighted average mass
    N_cloud_tot = np.sum(number_density_cloud * cloud_exists)
    if N_cloud_tot > 0:
        moments['M_cloud_avg'] = np.sum(number_density_cloud * M_cloud * cloud_exists) / N_cloud_tot
    else:
        moments['M_cloud_avg'] = 0.0
    
    # Cloud fraction by mass
    rho_cloud = Pressure * (mu*mp) / (kb*T_cloud)
    volume_cloud = M_cloud / rho_cloud
    moments['f_cloud_volume'] = np.sum(number_density_cloud * volume_cloud * cloud_exists)
    
    # Mass-weighted metallicity
    if moments['Mdot_cloud_tot'] > 0:
        moments['Z_cloud_mass_avg'] = np.sum(Mdot_cloud * Z_cloud * cloud_exists) / moments['Mdot_cloud_tot']
    else:
        moments['Z_cloud_mass_avg'] = 0.0
    
    return moments

def get_cloud_mass_spectrum(M_cloud, number_density_cloud, M_cloud_min, mass_bins=None):
    """
    Calculate the cloud mass spectrum dN/dlogM
    
    Parameters:
    -----------
    M_cloud : array
        Cloud masses
    number_density_cloud : array
        Number density of each cloud species
    M_cloud_min : float
        Minimum cloud mass threshold
    mass_bins : array, optional
        Mass bin edges for spectrum calculation
        
    Returns:
    --------
    M_bins : array
        Mass bin centers
    dN_dlogM : array
        Number per logarithmic mass interval
    """
    # Mask for existing clouds
    cloud_exists = M_cloud > M_cloud_min
    
    if mass_bins is None:
        # Create logarithmic mass bins
        M_min = np.min(M_cloud[cloud_exists]) if np.any(cloud_exists) else M_cloud_min
        M_max = np.max(M_cloud[cloud_exists]) if np.any(cloud_exists) else M_cloud_min * 1e4
        mass_bins = np.logspace(np.log10(M_min), np.log10(M_max), 50)
    
    # Bin centers
    M_bins = np.sqrt(mass_bins[:-1] * mass_bins[1:])
    
    # Calculate spectrum
    dN_dlogM = np.zeros(len(M_bins))
    
    for i in range(len(M_bins)):
        # Find clouds in this mass bin
        in_bin = ((M_cloud >= mass_bins[i]) & (M_cloud < mass_bins[i+1]) & cloud_exists)
        # Sum number density in bin
        dN_dlogM[i] = np.sum(number_density_cloud[in_bin])
    
    # Normalize by bin width in log space
    dlogM = np.diff(np.log10(mass_bins))
    dN_dlogM /= dlogM
    
    return M_bins, dN_dlogM
