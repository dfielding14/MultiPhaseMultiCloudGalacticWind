"""
Functions for calculating observables from wind solutions for comparison with data.

This module provides functions to calculate velocity distributions (dN/dv) and their
moments, which are useful for comparing model predictions with observations.
"""

import numpy as np
from .constants import mp, kpc, Msun, yr

# Mean molecular weight for ionized cold gas (following Xinfeng_data)
mu_cool = 1.4  # Mean atomic mass per proton for ionized gas


def _interpolate_distribution(v_source, y_source, v_target):
    """
    Linearly interpolate a distribution onto a target velocity grid.

    Inputs may be unsorted and can contain duplicate velocities.
    """
    v_source = np.asarray(v_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)
    v_target = np.asarray(v_target, dtype=float)

    valid = np.isfinite(v_source) & np.isfinite(y_source)
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(v_target, dtype=float)

    v_valid = v_source[valid]
    y_valid = y_source[valid]
    order = np.argsort(v_valid)
    v_sorted = v_valid[order]
    y_sorted = y_valid[order]

    v_unique, inv = np.unique(v_sorted, return_inverse=True)
    if v_unique.size < 2:
        return np.zeros_like(v_target, dtype=float)

    y_accum = np.zeros_like(v_unique, dtype=float)
    counts = np.zeros_like(v_unique, dtype=float)
    np.add.at(y_accum, inv, y_sorted)
    np.add.at(counts, inv, 1.0)
    y_unique = y_accum / np.maximum(counts, 1.0)

    y_target = np.interp(v_target, v_unique, y_unique, left=0.0, right=0.0)
    return np.where(np.isfinite(y_target), y_target, 0.0)


def _calculate_species_column_density_distribution(
    solution, cloud_index, mask, r_use, injection_function, Omwind
):
    """
    Compute dN/dv for one cloud species on its native velocity grid.
    """
    v_cloud_cms = solution.sol.y[4 + solution.model.N_cloud_species + cloud_index, mask]
    M_cloud = solution.sol.y[4 + cloud_index, mask]  # grams
    Ndot_cloud = solution.model.Ndot_cloud0[cloud_index]  # number / s

    # Exclude extinct-cloud tail points from the mapping.
    alive = (M_cloud >= solution.model.config.M_cloud_min) & (v_cloud_cms > 0.0)
    if np.count_nonzero(alive) < 2:
        return np.array([0.0]), np.array([0.0])

    v_cloud_cms = v_cloud_cms[alive]
    M_cloud = M_cloud[alive]
    r_alive = r_use[alive]
    injection_alive = injection_function[alive]

    denom = Omwind * r_alive**2 * v_cloud_cms
    cloud_density = np.divide(
        Ndot_cloud * M_cloud * injection_alive,
        denom,
        out=np.zeros_like(M_cloud, dtype=float),
        where=denom > 0.0,
    )
    n_H = np.where(np.isfinite(cloud_density), cloud_density / (mu_cool * mp), 0.0)

    grad_v = np.gradient(v_cloud_cms, r_alive)  # (cm/s) per cm
    finite_grad = grad_v[np.isfinite(grad_v)]
    if finite_grad.size < 2:
        return np.array([0.0]), np.array([0.0])

    # Treat tiny negative gradients as numerical noise.
    grad_scale = np.max(np.abs(finite_grad))
    grad_tol = max(1e-30, 1e-10 * grad_scale)

    if np.min(finite_grad) >= -grad_tol:
        grad_safe = np.where(grad_v > grad_tol, grad_v, grad_tol)
        dN_dv_cgs = np.divide(
            n_H, grad_safe,
            out=np.zeros_like(n_H),
            where=grad_safe > 0,
        )
    else:
        # Robust fallback for non-monotonic velocity histories.
        dr = np.gradient(r_alive)
        dN = np.where(np.isfinite(n_H * dr), n_H * dr, 0.0)

        sort_idx = np.argsort(v_cloud_cms)
        v_sorted = v_cloud_cms[sort_idx]
        dN_sorted = dN[sort_idx]
        if np.allclose(v_sorted, v_sorted[0]):
            return np.array([0.0]), np.array([0.0])

        n_bins = max(10, len(v_sorted) // 8)
        v_bins = np.linspace(v_sorted.min(), v_sorted.max(), n_bins)
        dv_bins = np.diff(v_bins)
        if np.count_nonzero(dv_bins > 0) == 0:
            return np.array([0.0]), np.array([0.0])

        dN_binned, _ = np.histogram(v_sorted, bins=v_bins, weights=dN_sorted)
        dN_dv_binned = np.divide(
            dN_binned, dv_bins,
            out=np.zeros_like(dN_binned, dtype=float),
            where=dv_bins > 0,
        )
        v_centers = 0.5 * (v_bins[:-1] + v_bins[1:])
        dN_dv_cgs = _interpolate_distribution(v_centers, dN_dv_binned, v_cloud_cms)

    v_cloud_kms = v_cloud_cms / 1e5
    dN_dv_kms = np.abs(np.where(np.isfinite(dN_dv_cgs), dN_dv_cgs, 0.0) * 1e5)
    return v_cloud_kms, dN_dv_kms


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
        denom = Omwind * r**2 * v_cloud
        cloud_density = np.divide(
            Ndot_cloud * injection_profile,
            denom,
            out=np.zeros_like(r, dtype=float),
            where=denom > 0.0,
        )
    else:
        # Sum over all species using vectorized broadcasting.
        denom = Omwind * r[np.newaxis, :]**2 * v_cloud
        cloud_density = np.sum(
            np.divide(
                Ndot_cloud[:, np.newaxis] * injection_profile[np.newaxis, :],
                denom,
                out=np.zeros_like(denom, dtype=float),
                where=denom > 0.0,
            ),
            axis=0,
        )
    
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
        moment = np.trapezoid(dN_dv * v_cloud**n, v_cloud)
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
    if r_use.size < 2:
        return np.array([0.0]), np.array([0.0])
    
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
    
    Omwind = solution.model.config.Omwind

    if cloud_index is not None:
        return _calculate_species_column_density_distribution(
            solution, cloud_index, mask, r_use, injection_function, Omwind
        )

    species_distributions = []
    v_min = np.inf
    v_max = -np.inf
    for i in range(solution.model.N_cloud_species):
        v_i, dN_dv_i = _calculate_species_column_density_distribution(
            solution, i, mask, r_use, injection_function, Omwind
        )
        if v_i.size < 2 or np.max(dN_dv_i) <= 0:
            continue
        species_distributions.append((v_i, dN_dv_i))
        v_min = min(v_min, np.min(v_i))
        v_max = max(v_max, np.max(v_i))

    if not species_distributions:
        return np.array([0.0]), np.array([0.0])
    if v_max <= v_min:
        return np.array([0.0]), np.array([0.0])

    n_v_points = np.count_nonzero(mask)
    v_cloud_kms = np.linspace(v_min, v_max, n_v_points)
    dN_dv_total = np.zeros_like(v_cloud_kms)
    for v_i, dN_dv_i in species_distributions:
        dN_dv_total += _interpolate_distribution(v_i, dN_dv_i, v_cloud_kms)

    return v_cloud_kms, np.abs(dN_dv_total)


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
    # Get radius array in cm
    r = solution.sol.t
    r_kpc = r / kpc

    # Find indices for radius range
    mask = (r_kpc >= r_min_kpc) & (r_kpc <= r_max_kpc)
    r_use = r[mask]
    r_kpc_use = r_kpc[mask]
    if r_use.size < 2:
        zeros = np.array([0.0])
        return zeros, {
            'total': zeros.copy(),
            'species': [zeros.copy() for _ in range(solution.model.N_cloud_species)],
            'M_cloud0': solution.model.M_cloud0 / Msun,
        }

    # Get injection parameters from model if not provided
    if injection_radius_kpc is None:
        r0_kpc = solution.model.r_star_kpc
        injection_radius_kpc = solution.model.config.cold_cloud_injection_radial_extent_frac * r0_kpc
    if injection_power is None:
        injection_power = solution.model.config.cold_cloud_injection_radial_power

    injection_function = np.where(
        r_kpc_use < injection_radius_kpc,
        (r_kpc_use / injection_radius_kpc)**injection_power,
        1.0,
    )
    Omwind = solution.model.config.Omwind

    species_native = []
    v_min = np.inf
    v_max = -np.inf
    for i in range(solution.model.N_cloud_species):
        v_i, dN_dv_i = _calculate_species_column_density_distribution(
            solution, i, mask, r_use, injection_function, Omwind
        )
        species_native.append((v_i, dN_dv_i))
        if v_i.size >= 2 and np.max(dN_dv_i) > 0:
            v_min = min(v_min, np.min(v_i))
            v_max = max(v_max, np.max(v_i))

    if not np.isfinite(v_min) or not np.isfinite(v_max):
        zeros = np.array([0.0])
        return zeros, {
            'total': zeros.copy(),
            'species': [zeros.copy() for _ in range(solution.model.N_cloud_species)],
            'M_cloud0': solution.model.M_cloud0 / Msun,
        }
    if v_max <= v_min:
        zeros = np.array([0.0])
        return zeros, {
            'total': zeros.copy(),
            'species': [zeros.copy() for _ in range(solution.model.N_cloud_species)],
            'M_cloud0': solution.model.M_cloud0 / Msun,
        }

    n_v_points = np.count_nonzero(mask)
    v_cloud = np.linspace(v_min, v_max, n_v_points)
    dN_dv_list = []
    for v_i, dN_dv_i in species_native:
        dN_dv_list.append(_interpolate_distribution(v_i, dN_dv_i, v_cloud))

    dN_dv_total = np.sum(np.vstack(dN_dv_list), axis=0)
    return v_cloud, {
        'total': np.abs(dN_dv_total),
        'species': [np.abs(arr) for arr in dN_dv_list],
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
    r_eval_kpc = np.atleast_1d(r_eval_kpc).astype(float)
    v_mass_weighted = np.full_like(r_eval_kpc, np.nan)
    v_cloud_species = np.atleast_2d(solution.v_cl)
    M_cloud_species = np.atleast_2d(solution.M_clouds)
    Ndot_cloud0 = np.atleast_1d(solution.model.Ndot_cloud0)

    injection_radius_kpc = solution.model.config.cold_cloud_injection_radial_extent_frac * solution.model.r_star_kpc
    injection_power = solution.model.config.cold_cloud_injection_radial_power
    
    for i, r_kpc in enumerate(r_eval_kpc):
        if r_kpc <= solution.r[-1]:
            # Hot phase contribution
            v_hot = np.interp(r_kpc, solution.r, solution.v)
            Mdot_hot = np.interp(r_kpc, solution.r, solution.Mdot)  # Msun/yr

            if r_kpc < injection_radius_kpc:
                injection_factor = (r_kpc / injection_radius_kpc)**injection_power
            else:
                injection_factor = 1.0

            Mdot_cold = 0.0
            v_cold_numerator = 0.0
            for j in range(solution.model.N_cloud_species):
                if j >= v_cloud_species.shape[0] or j >= M_cloud_species.shape[0] or j >= Ndot_cloud0.size:
                    continue

                v_cl_j = np.interp(r_kpc, solution.r, v_cloud_species[j])  # km/s
                M_cl_j = np.interp(r_kpc, solution.r, M_cloud_species[j]) * Msun  # g
                Mdot_cl_j = Ndot_cloud0[j] * injection_factor * M_cl_j / (Msun/yr)  # Msun/yr

                if np.isfinite(Mdot_cl_j) and Mdot_cl_j > 0:
                    Mdot_cold += Mdot_cl_j
                    v_cold_numerator += Mdot_cl_j * v_cl_j

            v_cold = v_cold_numerator / Mdot_cold if Mdot_cold > 0 else 0.0
            Mdot_total = Mdot_hot + Mdot_cold
            if Mdot_total > 0:
                v_mass_weighted[i] = (Mdot_hot * v_hot + Mdot_cold * v_cold) / Mdot_total
    
    return v_mass_weighted[0] if len(r_eval_kpc) == 1 else v_mass_weighted
