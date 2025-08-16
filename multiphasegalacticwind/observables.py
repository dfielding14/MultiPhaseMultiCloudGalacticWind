"""
Functions for calculating observables from wind solutions for comparison with data.

This module provides functions to calculate velocity distributions (dN/dv) and their
moments, which are useful for comparing model predictions with observations.
"""

import numpy as np
from .constants import mp, kb, kpc, Msun
from .config import get_default_config


def calculate_cloud_density(solution, cloud_index=None, 
                           injection_radius_kpc=0.3, 
                           injection_power=6.0):
    """
    Calculate the number density of clouds as a function of radius.
    
    Parameters
    ----------
    solution : Solution object
        The wind solution from WindModel.run()
    cloud_index : int, optional
        Index of specific cloud species. If None, sum over all species.
    injection_radius_kpc : float
        Radius below which cloud injection is enhanced [kpc]
    injection_power : float
        Power law index for cloud injection profile
        
    Returns
    -------
    cloud_density : array
        Number density of clouds [cm^-3]
    """
    r = solution.sol.t  # radius in cm
    r_kpc = r / kpc
    
    # Get cloud masses and velocities
    if cloud_index is not None:
        M_cloud = solution.M_clouds[cloud_index]
        Ndot_cloud = solution.model.Ndot_cloud0[cloud_index]
        v_cloud = solution.sol.y[4 + solution.model.N_cloud_species + cloud_index]  # cm/s
    else:
        # For all species combined, sum the number densities
        Ndot_cloud = solution.model.Ndot_cloud0  # array of all species
        v_cloud = solution.sol.y[4 + solution.model.N_cloud_species:4 + 2*solution.model.N_cloud_species]  # all velocities
    
    # Cloud injection profile
    injection_profile = np.where(r_kpc < injection_radius_kpc,
                               (r_kpc / injection_radius_kpc)**injection_power,
                               1.0)
    
    # Calculate cloud number density
    # Number conservation: Ndot = Omega * r^2 * v * n
    Omwind = solution.model.config.Omwind
    
    if cloud_index is not None:
        # Single species
        cloud_density = (Ndot_cloud * injection_profile / 
                        (Omwind * r**2 * v_cloud))
    else:
        # Sum over all species
        cloud_density = np.zeros_like(r)
        for i in range(solution.model.N_cloud_species):
            cloud_density += (Ndot_cloud[i] * injection_profile / 
                            (Omwind * r**2 * v_cloud[i]))
    
    return cloud_density



def calculate_velocity_moments(v_cloud, dN_dv, max_order=3):
    """
    Calculate moments of the velocity distribution.
    
    Parameters
    ----------
    v_cloud : array
        Cloud velocities
    dN_dv : array
        Velocity distribution
    max_order : int
        Maximum moment order to calculate
        
    Returns
    -------
    moments : dict
        Dictionary containing:
        - 'raw': Raw moments (0th through max_order)
        - 'mean': Mean velocity
        - 'dispersion': Velocity dispersion
        - 'skewness': Skewness (if max_order >= 3)
    """
    # Calculate raw moments
    raw_moments = []
    for n in range(max_order + 1):
        moment = np.trapz(dN_dv * v_cloud**n, v_cloud)
        raw_moments.append(moment)
    
    # Calculate central moments
    zeroth = raw_moments[0]
    results = {'raw': raw_moments}
    
    if zeroth > 0:
        # Mean velocity
        mean_v = raw_moments[1] / zeroth
        results['mean'] = mean_v
        
        if max_order >= 2:
            # Velocity dispersion
            var = raw_moments[2] / zeroth - mean_v**2
            results['dispersion'] = np.sqrt(max(0, var))
            
            if max_order >= 3 and var > 0:
                # Skewness
                third_central = raw_moments[3] / zeroth - 3 * mean_v * raw_moments[2] / zeroth + 2 * mean_v**3
                results['skewness'] = third_central / var**1.5
    
    return results


def calculate_column_density_distribution(solution, cloud_index=None,
                                        r_min_kpc=0.05, r_max_kpc=100.0,
                                        injection_radius_kpc=0.3,
                                        injection_power=6.0):
    """
    Calculate dN/dv in column density units [cm^-2 / (km/s)].
    
    This is suitable for comparison with absorption line observations.
    Uses the gradient method: dN/dv = (dN/dr) / (dv/dr)
    
    Parameters
    ----------
    solution : Solution object
        The wind solution from WindModel.run()
    cloud_index : int, optional
        Index of specific cloud species. If None, sum over all species.
    r_min_kpc : float
        Minimum radius to include [kpc]
    r_max_kpc : float
        Maximum radius to include [kpc]
    injection_radius_kpc : float
        Radius below which cloud injection is enhanced [kpc]
    injection_power : float
        Power law index for cloud injection profile
        
    Returns
    -------
    v_cloud : array
        Cloud velocities [km/s]
    dN_dv_column : array
        Column density distribution [cm^-2 / (km/s)]
    """
    # Get radius array
    r = solution.sol.t  # cm
    r_kpc = r / kpc
    
    # Find indices for radius range
    mask = (r_kpc >= r_min_kpc) & (r_kpc <= r_max_kpc)
    r_use = r[mask]
    
    # Calculate path length through each shell (assuming spherical geometry)
    path_length = 2 * r_use  # diameter
    
    if cloud_index is not None:
        # Single cloud species
        v_cloud = solution.sol.y[4 + solution.model.N_cloud_species + cloud_index, mask] / 1e5  # km/s
        M_cloud = solution.sol.y[4 + cloud_index, mask]
        
        # Skip if cloud is destroyed
        if np.max(M_cloud) < solution.model.config.M_cloud_min:
            return np.array([0.0]), np.array([0.0])
        
        # Cloud number density
        cloud_density = calculate_cloud_density(solution, cloud_index,
                                              injection_radius_kpc, injection_power)
        n_cloud = cloud_density[mask]
        
        # Gas density (n_H) in cold phase
        rho_cold = n_cloud * M_cloud
        n_H_cold = rho_cold / (solution.model.config.mu * mp)
        
        # Column density: N = integral of n_H * dl
        N_column = n_H_cold * path_length
        
        # Calculate gradients
        dN_dr = np.gradient(N_column, r_use)  # cm^-2 per cm
        dv_dr = np.gradient(v_cloud * 1e5, r_use)  # (cm/s) per cm
        
        # Calculate dN/dv using chain rule
        # dN/dv = (dN/dr) / (dv/dr) with proper units
        # Result should be in cm^-2 per (km/s)
        dN_dv_column = np.zeros_like(v_cloud)
        valid = np.abs(dv_dr) > 1e-20  # avoid division by zero (use small threshold for cgs units)
        dN_dv_column[valid] = dN_dr[valid] / dv_dr[valid] * 1e5  # convert to per km/s
        
    else:
        # Sum contributions from all cloud species
        # First collect all velocities to determine output grid
        all_velocities = []
        for i in range(solution.model.N_cloud_species):
            v_cl_i = solution.sol.y[4 + solution.model.N_cloud_species + i, mask] / 1e5  # km/s
            M_cl_i = solution.sol.y[4 + i, mask]
            if np.max(M_cl_i) >= solution.model.config.M_cloud_min:
                all_velocities.extend(v_cl_i)
        
        if len(all_velocities) == 0:
            return np.array([0.0]), np.array([0.0])
        
        # Use velocity grid from first surviving species for output
        # (all species have same velocity at each radius)
        v_cloud = solution.sol.y[4 + solution.model.N_cloud_species + 0, mask] / 1e5
        dN_dv_column = np.zeros_like(v_cloud)
        
        # Calculate dN/dv for each species and sum
        for i in range(solution.model.N_cloud_species):
            v_cl_i = solution.sol.y[4 + solution.model.N_cloud_species + i, mask] / 1e5  # km/s
            M_cl_i = solution.sol.y[4 + i, mask]
            
            # Skip if cloud is destroyed
            if np.max(M_cl_i) < solution.model.config.M_cloud_min:
                continue
            
            # Cloud density for this species
            cloud_density_i = calculate_cloud_density(solution, i,
                                                    injection_radius_kpc, injection_power)
            n_cl_i = cloud_density_i[mask]
            
            # Gas density for this species
            rho_i = n_cl_i * M_cl_i
            n_H_i = rho_i / (solution.model.config.mu * mp)
            
            # Column density for this species
            N_i = n_H_i * path_length
            
            # Gradients for this species
            dN_dr_i = np.gradient(N_i, r_use)  # cm^-2 per cm
            dv_dr_i = np.gradient(v_cl_i * 1e5, r_use)  # (cm/s) per cm
            
            # Add this species' contribution to total dN/dv
            valid = np.abs(dv_dr_i) > 1e-20  # small threshold for cgs units
            dN_dv_i = np.zeros_like(v_cl_i)
            dN_dv_i[valid] = dN_dr_i[valid] / dv_dr_i[valid] * 1e5  # convert to per km/s
            
            dN_dv_column += dN_dv_i
    
    # Take absolute value (physical column density is positive)
    dN_dv_column = np.abs(dN_dv_column)
    
    return v_cloud, dN_dv_column


def calculate_column_density_by_species(solution, r_min_kpc=0.05, r_max_kpc=100.0,
                                      injection_radius_kpc=0.3,
                                      injection_power=6.0):
    """
    Calculate dN/dv for each cloud species separately.
    
    This is useful for understanding which cloud masses contribute to different
    parts of the velocity distribution.
    
    Parameters
    ----------
    solution : Solution object
        The wind solution from WindModel.run()
    r_min_kpc : float
        Minimum radius to include [kpc]
    r_max_kpc : float
        Maximum radius to include [kpc]
    injection_radius_kpc : float
        Radius below which cloud injection is enhanced [kpc]
    injection_power : float
        Power law index for cloud injection profile
        
    Returns
    -------
    v_cloud : array
        Cloud velocities [km/s]
    dN_dv_species : dict
        Dictionary with keys:
        - 'total': Total dN/dv from all species [cm^-2 / (km/s)]
        - 'species': List of dN/dv for each species [cm^-2 / (km/s)]
        - 'M_cloud0': Initial cloud masses for each species [Msun]
    """
    # Calculate total
    v_cloud, dN_dv_total = calculate_column_density_distribution(
        solution, cloud_index=None,
        r_min_kpc=r_min_kpc, r_max_kpc=r_max_kpc,
        injection_radius_kpc=injection_radius_kpc,
        injection_power=injection_power
    )
    
    # Calculate for each species
    dN_dv_list = []
    for i in range(solution.model.N_cloud_species):
        _, dN_dv_i = calculate_column_density_distribution(
            solution, cloud_index=i,
            r_min_kpc=r_min_kpc, r_max_kpc=r_max_kpc,
            injection_radius_kpc=injection_radius_kpc,
            injection_power=injection_power
        )
        dN_dv_list.append(dN_dv_i)
    
    return v_cloud, {
        'total': dN_dv_total,
        'species': dN_dv_list,
        'M_cloud0': solution.model.M_cloud0 / Msun
    }


def calculate_mass_weighted_velocity(solution, r_eval_kpc=10.0):
    """
    Calculate mass-weighted average velocity at a given radius.
    
    This is often more relevant for observations than number-weighted velocity.
    
    Parameters
    ----------
    solution : Solution object
        The wind solution from WindModel.run()
    r_eval_kpc : float or array
        Radius(ii) at which to evaluate [kpc]
        
    Returns
    -------
    v_mass_weighted : float or array
        Mass-weighted velocity [km/s]
    """
    r_eval_kpc = np.atleast_1d(r_eval_kpc)
    v_mass_weighted = np.zeros_like(r_eval_kpc)
    
    for i, r_kpc in enumerate(r_eval_kpc):
        if r_kpc <= solution.r[-1]:
            # Hot phase contribution
            v_hot = np.interp(r_kpc, solution.r, solution.v)
            rho_hot = np.interp(r_kpc, solution.r, solution.rho)
            
            # Cold phase contribution
            v_cold = np.interp(r_kpc, solution.r, solution.v_cl)
            M_cloud_tot = np.interp(r_kpc, solution.r, solution.M_cloud_tot)
            
            # Mass flux contributions
            Mdot_hot = 4 * np.pi * (r_kpc * kpc)**2 * rho_hot * v_hot * 1e5
            Mdot_cold = 4 * np.pi * (r_kpc * kpc)**2 * rho_hot * v_cold * 1e5 * \
                       M_cloud_tot / np.sum(solution.model.M_cloud0)
            
            # Mass-weighted average
            v_mass_weighted[i] = (Mdot_hot * v_hot + Mdot_cold * v_cold) / (Mdot_hot + Mdot_cold)
    
    return v_mass_weighted[0] if len(r_eval_kpc) == 1 else v_mass_weighted