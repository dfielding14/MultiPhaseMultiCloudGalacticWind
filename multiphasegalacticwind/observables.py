"""
Functions for calculating observables from wind solutions for comparison with data.

This module provides functions to calculate velocity distributions (dN/dv) and their
moments, which are useful for comparing model predictions with observations.
"""

import numpy as np
from .constants import mp, kpc, Msun
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
    else:
        M_cloud = solution.M_cloud_tot
        Ndot_cloud = np.sum(solution.model.Ndot_cloud0)
    
    v_cloud = solution.sol.y[3 + solution.model.N_cloud_species]  # cm/s
    
    # Cloud injection profile
    injection_profile = np.where(r_kpc < injection_radius_kpc,
                               (r_kpc / injection_radius_kpc)**injection_power,
                               1.0)
    
    # Calculate cloud density
    # Number conservation: Ndot = Omega * r^2 * v * n
    Omwind = solution.model.config.Omwind
    cloud_density = (Ndot_cloud * injection_profile * M_cloud / 
                    (Omwind * r**2 * v_cloud))
    
    return cloud_density


def calculate_velocity_distribution(solution, cloud_index=None,
                                  r_min_kpc=0.05, r_max_kpc=10.0,
                                  velocity_units='km/s',
                                  injection_radius_kpc=0.3,
                                  injection_power=6.0):
    """
    Calculate dN/dv - the velocity distribution of clouds.
    
    Uses the chain rule: dN/dv = (dN/dr) / (dv/dr)
    
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
    velocity_units : str
        Units for velocity: 'km/s' or 'cm/s'
    injection_radius_kpc : float
        Radius below which cloud injection is enhanced [kpc]
    injection_power : float
        Power law index for cloud injection profile
        
    Returns
    -------
    v_cloud : array
        Cloud velocities [km/s or cm/s]
    dN_dv : array
        Velocity distribution [number per velocity unit]
    """
    # Get radius array
    r = solution.sol.t  # cm
    r_kpc = r / kpc
    
    # Find indices for radius range
    mask = (r_kpc >= r_min_kpc) & (r_kpc <= r_max_kpc)
    r_use = r[mask]
    
    # Get cloud density
    cloud_density = calculate_cloud_density(solution, cloud_index,
                                          injection_radius_kpc, injection_power)
    cloud_density_use = cloud_density[mask]
    
    # Convert to number density
    # Get mu from model config
    mu = solution.model.config.mu
    n_cloud = cloud_density_use / (mu * mp)
    
    # Get cloud velocity
    if cloud_index is None:
        # Average over all cloud species weighted by density
        v_cloud = np.zeros(np.sum(mask))
        total_density = np.zeros_like(v_cloud)
        for i in range(solution.model.N_cloud_species):
            v_cl_i = solution.sol.y[4 + solution.model.N_cloud_species + i, mask]  # cm/s
            density_i = cloud_density[i, mask] if cloud_density.ndim > 1 else cloud_density[mask]
            v_cloud += v_cl_i * density_i
            total_density += density_i
        v_cloud = np.where(total_density > 0, v_cloud / total_density, 0)
    else:
        # Single cloud species
        v_cloud = solution.sol.y[4 + solution.model.N_cloud_species + cloud_index, mask]  # cm/s
    
    # Calculate velocity gradient
    dv_dr = np.gradient(v_cloud, r_use)
    
    # Apply chain rule: dN/dv = (dN/dr) / (dv/dr)
    # Avoid division by zero
    dN_dv = np.where(np.abs(dv_dr) > 1e-10, n_cloud / np.abs(dv_dr), 0)
    
    # Convert units if needed
    if velocity_units == 'km/s':
        v_cloud = v_cloud / 1e5  # cm/s to km/s
        dN_dv = dN_dv * 1e5  # adjust distribution
    
    return v_cloud, dN_dv


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
                                        injection_power=6.0,
                                        path_length_method='diameter'):
    """
    Calculate dN/dv in column density units [cm^-2 / (km/s)].
    
    This is suitable for comparison with absorption line observations.
    
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
    path_length_method : str
        Method for calculating path length:
        - 'diameter': Use 2r (path through center)
        - 'shell': Use shell thickness dr
        
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
    
    # Get cloud density
    cloud_density = calculate_cloud_density(solution, cloud_index,
                                          injection_radius_kpc, injection_power)
    cloud_density_use = cloud_density[mask]
    
    # Convert to number density
    # Get mu from model config
    mu = solution.model.config.mu
    n_cloud = cloud_density_use / (mu * mp)
    
    # Get cloud velocity
    v_cloud = solution.sol.y[3 + solution.model.N_cloud_species, mask] / 1e5  # km/s
    
    # Calculate path length through each shell
    if path_length_method == 'diameter':
        # Path length is approximately the diameter
        path_length = 2 * r_use
    elif path_length_method == 'shell':
        # Path length is the shell thickness
        dr = np.gradient(r_use)
        path_length = np.abs(dr)
    else:
        raise ValueError(f"Unknown path_length_method: {path_length_method}")
    
    # Calculate column density per radius interval
    dN_dr = n_cloud * path_length
    
    # Calculate velocity gradient
    dv_dr = np.gradient(v_cloud * 1e5, r_use)  # Convert back to cm/s for gradient
    
    # Apply chain rule: dN/dv = dN/dr / (dv/dr)
    # Result is in cm^-2 / (cm/s), convert to cm^-2 / (km/s)
    dN_dv_column = np.where(np.abs(dv_dr) > 1e-10, 
                           dN_dr / np.abs(dv_dr) * 1e5,  # multiply by 1e5 for km/s units
                           0)
    
    return v_cloud, dN_dv_column


def calculate_column_density_by_species(solution, r_min_kpc=0.05, r_max_kpc=100.0,
                                      injection_radius_kpc=0.3,
                                      injection_power=6.0,
                                      path_length_method='diameter'):
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
    path_length_method : str
        Method for calculating path length
        
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
        injection_power=injection_power,
        path_length_method=path_length_method
    )
    
    # Calculate for each species
    dN_dv_list = []
    for i in range(solution.model.N_cloud_species):
        _, dN_dv_i = calculate_column_density_distribution(
            solution, cloud_index=i,
            r_min_kpc=r_min_kpc, r_max_kpc=r_max_kpc,
            injection_radius_kpc=injection_radius_kpc,
            injection_power=injection_power,
            path_length_method=path_length_method
        )
        dN_dv_list.append(dN_dv_i)
    
    return v_cloud, {
        'total': dN_dv_total,
        'species': dN_dv_list,
        'M_cloud0': solution.model.M_cloud0
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