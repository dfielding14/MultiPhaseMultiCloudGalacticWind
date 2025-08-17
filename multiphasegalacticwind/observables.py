"""
Functions for calculating observables from wind solutions for comparison with data.

This module provides functions to calculate velocity distributions (dN/dv) and their
moments, which are useful for comparing model predictions with observations.
"""

import numpy as np
from .constants import mp, kb, kpc, Msun, km
from .config import get_default_config

# Mean molecular weight for ionized cold gas (following Xinfeng_data)
mu_cool = 1.4  # Mean atomic mass per proton for ionized gas


def calculate_cloud_density(solution, cloud_index=None, 
                           injection_radius_kpc=None, 
                           injection_power=None):
    """
    Calculate the number density of clouds as a function of radius.
    
    Parameters
    ----------
    solution : Solution object
        The wind solution from WindModel.run()
    cloud_index : int, optional
        Index of specific cloud species. If None, sum over all species.
    injection_radius_kpc : float, optional
        Radius below which cloud injection is enhanced [kpc].
        If None, uses model's injection radius from config.
    injection_power : float, optional
        Power law index for cloud injection profile.
        If None, uses model's injection power from config.
        
    Returns
    -------
    cloud_density : array
        Number density of clouds [cm^-3]
    """
    # Get injection parameters from model if not provided
    if injection_radius_kpc is None:
        r0_kpc = solution.model.r_star_kpc
        injection_radius_kpc = solution.model.config.cold_cloud_injection_radial_extent_frac * r0_kpc
    if injection_power is None:
        injection_power = solution.model.config.cold_cloud_injection_radial_power
    
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
                                        injection_radius_kpc=None,
                                        injection_power=None):
    """
    Calculate dN/dv in column density units [cm^-2 / (km/s)].
    
    Following the Xinfeng_data approach for consistency with previous work.
    This uses direct gradient method without path length multiplication.
    
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
    injection_radius_kpc : float, optional
        Radius below which cloud injection is enhanced [kpc].
        If None, uses model's injection radius from config.
    injection_power : float, optional
        Power law index for cloud injection profile.
        If None, uses model's injection power from config.
        
    Returns
    -------
    v_cloud : array
        Cloud velocities [km/s]
    dN_dv_column : array
        Column density distribution [cm^-2 / (km/s)]
    """
    # Get radius array in cm
    r = solution.sol.t  # cm
    r_kpc = r / kpc
    
    # Find indices for radius range
    mask = (r_kpc >= r_min_kpc) & (r_kpc <= r_max_kpc)
    r_use = r[mask]
    r_kpc_use = r_kpc[mask]
    
    # Get injection parameters from model if not provided
    if injection_radius_kpc is None:
        r0_kpc = solution.model.r_star_kpc
        injection_radius_kpc = solution.model.config.cold_cloud_injection_radial_extent_frac * r0_kpc
    if injection_power is None:
        injection_power = solution.model.config.cold_cloud_injection_radial_power
    
    # Get injection function (following Xinfeng_data line 249)
    injection_function = np.where(r_kpc_use < injection_radius_kpc,
                                 (r_kpc_use / injection_radius_kpc)**injection_power,
                                 1.0)
    
    # Get wind parameters
    Omwind = solution.model.config.Omwind
    
    if cloud_index is not None:
        # Single cloud species
        v_cloud_cms = solution.sol.y[4 + solution.model.N_cloud_species + cloud_index, mask]  # cm/s
        M_cloud = solution.sol.y[4 + cloud_index, mask]  # grams
        Ndot_cloud = solution.model.Ndot_cloud0[cloud_index]  # number/s
        
        # Skip if cloud is destroyed
        if np.max(M_cloud) < solution.model.config.M_cloud_min:
            return np.array([0.0]), np.array([0.0])
        
        # Calculate cloud mass density (following Xinfeng_data line 453)
        # rho = Ndot * M * injection / (Omega * r^2 * v)
        cloud_density = (Ndot_cloud * M_cloud * injection_function / 
                        (Omwind * r_use**2 * v_cloud_cms))
        
        # Convert to hydrogen number density using mu_cool
        n_H = cloud_density / (mu_cool * mp)
        
        # Calculate velocity gradient
        grad_v = np.gradient(v_cloud_cms, r_use)  # (cm/s) per cm
        
        # Calculate dN/dv (following Xinfeng_data line 470)
        dN_dv_column = np.zeros_like(v_cloud_cms)
        
        # Check for positive gradient (normal case)
        if np.min(grad_v) > 0:
            # Avoid exact zeros
            grad_v = np.where(grad_v == 0, 1e-30, grad_v)
            dN_dv_column = n_H / grad_v  # cm^-2 per (cm/s)
            
        else:
            # Fallback method when gradient becomes negative (Xinfeng_data line 487)
            # This rebins the data - we'll use a simpler interpolation approach
            dr = np.gradient(r_use)
            dN = n_H * dr  # column density increment
            
            # Sort by velocity and accumulate
            sort_idx = np.argsort(v_cloud_cms)
            v_sorted = v_cloud_cms[sort_idx]
            dN_sorted = dN[sort_idx]
            
            # Create velocity bins
            n_bins = max(10, len(v_cloud_cms) // 8)
            v_bins = np.linspace(v_sorted.min(), v_sorted.max(), n_bins)
            v_centers = 0.5 * (v_bins[:-1] + v_bins[1:])
            
            # Bin the column density
            dN_binned, _ = np.histogram(v_sorted, bins=v_bins, weights=dN_sorted)
            dv_bins = np.diff(v_bins)
            dN_dv_binned = dN_binned / dv_bins
            
            # Interpolate back to original velocity grid
            from scipy.interpolate import interp1d
            f_interp = interp1d(v_centers, dN_dv_binned, 
                              kind='linear', fill_value=0, bounds_error=False)
            dN_dv_column = f_interp(v_cloud_cms)
        
        # Convert velocity to km/s
        v_cloud_kms = v_cloud_cms / 1e5
        # Convert dN/dv from per (cm/s) to per (km/s)
        dN_dv_column = dN_dv_column * 1e5
        
    else:
        # Sum contributions from all cloud species
        # Get velocity from first species as reference (all species have same v at each r)
        v_cloud_cms = solution.sol.y[4 + solution.model.N_cloud_species, mask]  # cm/s
        v_cloud_kms = v_cloud_cms / 1e5
        dN_dv_column = np.zeros_like(v_cloud_cms)
        
        # Calculate dN/dv for each species and sum
        for i in range(solution.model.N_cloud_species):
            M_cl_i = solution.sol.y[4 + i, mask]  # grams
            
            # Skip if cloud is destroyed
            if np.max(M_cl_i) < solution.model.config.M_cloud_min:
                continue
                
            v_cl_i_cms = solution.sol.y[4 + solution.model.N_cloud_species + i, mask]  # cm/s
            Ndot_i = solution.model.Ndot_cloud0[i]
            
            # Calculate mass density for this species
            cloud_density_i = (Ndot_i * M_cl_i * injection_function /
                             (Omwind * r_use**2 * v_cl_i_cms))
            
            # Convert to hydrogen number density
            n_H_i = cloud_density_i / (mu_cool * mp)
            
            # Calculate gradient
            grad_v_i = np.gradient(v_cl_i_cms, r_use)
            
            # Calculate dN/dv for this species
            if np.min(grad_v_i) > 0:
                grad_v_i = np.where(grad_v_i == 0, 1e-30, grad_v_i)
                dN_dv_i = n_H_i / grad_v_i * 1e5  # Convert to per (km/s)
            else:
                # Fallback for negative gradients
                dr = np.gradient(r_use)
                dN_i = n_H_i * dr
                # Simple average for this species
                dN_dv_i = np.sum(dN_i) / (v_cl_i_cms.max() - v_cl_i_cms.min()) * 1e5
                dN_dv_i = np.ones_like(v_cl_i_cms) * dN_dv_i
            
            dN_dv_column += dN_dv_i
    
    # Take absolute value (physical column density is positive)
    dN_dv_column = np.abs(dN_dv_column)
    
    return v_cloud_kms, dN_dv_column


def calculate_column_density_by_species(solution, r_min_kpc=0.05, r_max_kpc=100.0,
                                      injection_radius_kpc=None,
                                      injection_power=None):
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
    injection_radius_kpc : float, optional
        Radius below which cloud injection is enhanced [kpc].
        If None, uses model's injection radius from config.
    injection_power : float, optional
        Power law index for cloud injection profile.
        If None, uses model's injection power from config.
        
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