#!/usr/bin/env python
# coding: utf-8

# # Cloud Species Discretization Comparison
# 
# This notebook compares wind models with different numbers of cloud species to understand how the discretization affects the wind solution and observables.
# 
# We compare models with 1, 3, 6, 11, and 21 cloud species, where:
# - For N>1: Cloud masses follow a power-law distribution from 10 to 10^6 Msun
# - For N=1: Uses the geometric mean of the distribution (~3162 Msun)

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.constants import Msun, yr
from multiphasegalacticwind.observables import (
    calculate_column_density_distribution,
    calculate_velocity_moments
)

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


# ## Define Base Model Parameters
# 
# These parameters are kept constant across all models, only varying the number of cloud species.

# In[34]:


# Base parameters for all models
base_params = {
    # Galaxy properties
    'v_circ': 150.0,           # km/s, circular velocity (Milky Way-like)
    'redshift': 0.0,

    # Wind launch properties
    'SFR': 20.0,               # Msun/yr, star formation rate
    'eta_M': 0.1,              # hot phase mass loading
    'eta_M_cold': 0.3,         # cold phase mass loading
    'eta_E': 1.0,              # energy loading

    # Sonic point
    'r_star_kpc': 0.3,         # kpc (= 300 pc)

    # Cloud properties
    'cloud_mass_range': (10, 1e6),    # Msun
    'cloud_alpha': 2.0,               # power law slope dN/dM ∝ M^-α
    'T_cl': 1e4,                      # K, cloud temperature
    'v_cloud_init': 20.0,             # km/s, initial cloud velocity
    'M_cloud_min': 1e-1*Msun,         # Msun, minimum cloud mass

    # Integration settings
    'r_max_kpc': 30.0,         # kpc, maximum radius
    'rtol': 1e-6,              # relaxed tolerance for efficiency
    'atol': 1e-8,
}

# Number of cloud species to compare
n_species_list = [1, 3, 6, 11, 21]

# For single cloud case, use geometric mean of the distribution
M_min, M_max = base_params['cloud_mass_range']
M_geometric_mean = np.sqrt(M_min * M_max)

print("Base model parameters:")
print(f"  SFR = {base_params['SFR']} Msun/yr")
print(f"  eta_M = {base_params['eta_M']}")
print(f"  eta_M,cold = {base_params['eta_M_cold']}")
print(f"  eta_E = {base_params['eta_E']}")
print(f"  r_sonic = {base_params['r_star_kpc']} kpc = {base_params['r_star_kpc'] * 1000} pc")
print(f"  Cloud mass range = [{M_min}, {M_max}] Msun")
print(f"  Geometric mean cloud mass = {M_geometric_mean:.1f} Msun")


# ## Run Wind Models with Different Cloud Discretizations

# In[35]:


# Store models and solutions
models = {}
solutions = {}

# Track which models succeeded for plotting
successful_species = []

# Run models with different numbers of cloud species
for n_species in n_species_list:
    print(f"\n{'='*40}")
    print(f"Running model with {n_species} cloud species...")

    # Copy base parameters
    params = base_params.copy()
    params['N_cloud_species'] = n_species

    # For single cloud, use geometric mean mass
    if n_species == 1:
        params['cloud_mass_range'] = (M_geometric_mean, M_geometric_mean)
        print(f"  Using single cloud mass = {M_geometric_mean:.1f} Msun")

    # Create and run model
    models[n_species] = WindModel(**params)
    solutions[n_species] = models[n_species].run()

    # Print summary
    if solutions[n_species].sol.status == -1:
        print(f"  ⚠ Integration terminated early at {solutions[n_species].r[-1]:.2f} kpc")
        print(f"    {solutions[n_species].sol.message}")
    else:
        print(f"  Integration successful!")
        print(f"  Final radius reached: {solutions[n_species].r[-1]:.1f} kpc")
        print(f"  Final wind velocity: {solutions[n_species].v[-1]:.1f} km/s")
        print(f"  Mass loading at 10 kpc: {solutions[n_species].mass_loading_at_10kpc:.2f}")
        successful_species.append(n_species)

print("\n" + "=" * 40)
if len(successful_species) == len(n_species_list):
    print("All models completed successfully!")
else:
    print(f"Completed {len(successful_species)} of {len(n_species_list)} models successfully")
    if len(successful_species) < len(n_species_list):
        print(f"Failed models: {[n for n in n_species_list if n not in successful_species]}")
        print("Note: Failed models will be excluded from plots")


# ## Compare Wind Solutions
# 
# Plot the wind velocity, mass flux, and cloud masses for each model side by side.

# In[ ]:


# Create figure with subplots for wind solutions
# Use only successful runs for plotting
plot_species = successful_species if 'successful_species' in locals() else n_species_list
fig1 = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig1.add_gridspec(3, len(plot_species), height_ratios=[1, 1, 1])

# Color scheme for different models
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(plot_species)))

# Plot wind solutions for each model
for idx, n_species in enumerate(plot_species):
    solution = solutions[n_species]

    # Velocity panel
    ax1 = fig1.add_subplot(gs[0, idx])
    ax1.loglog(solution.r, solution.v, 'k-', lw=1.5, label='Hot')

    # Plot cloud velocities if multiple species
    if n_species > 1:
        for i in range(n_species):
            # Mask where clouds don't exist
            v_cl_masked = np.ma.masked_where(
                solution.M_clouds[i] < solution.model.config.M_cloud_min / Msun,
                solution.v_cl[i]
            )
            ax1.loglog(solution.r, v_cl_masked, '-', lw=0.5, alpha=0.5)
    else:
        # Single cloud
        v_cl_masked = np.ma.masked_where(
            solution.M_clouds[0] < solution.model.config.M_cloud_min / Msun,
            solution.v_cl[0]
        )
        ax1.loglog(solution.r, v_cl_masked, 'b--', lw=1, label='Cloud')

    ax1.set_xlim(0.3, 30)
    ax1.set_ylim(30, 3500)
    ax1.set_title(f'N = {n_species}', fontsize=10)
    ax1.legend(fontsize=7, frameon=False, loc='best')

    if idx == 0:
        ax1.set_ylabel(r'$v$ [km/s]')
    else:
        ax1.set_yticklabels([])

    ax1.tick_params(labelbottom=False)

    # Mass flux panel
    ax2 = fig1.add_subplot(gs[1, idx])
    ax2.loglog(solution.r, solution.Mdot/solution.model.SFR, 'k-', lw=1.5, label='Total')

    # Calculate and plot cloud mass flux
    Mdot_cl_total = np.zeros_like(solution.r)
    injection_radius = solution.model.config.cold_cloud_injection_radial_extent
    injection_power = solution.model.config.cold_cloud_injection_radial_power

    for i in range(n_species):
        r_cgs = solution.sol.t  # radius in cm
        injection_function = np.where(r_cgs < injection_radius,
                                     (r_cgs/injection_radius)**injection_power,
                                     1.0)
        Ndot_cloud_i = solution.model.Ndot_cloud0[i] * injection_function
        Mdot_cl_i = Ndot_cloud_i * solution.M_clouds[i] * Msun / (Msun/yr) / solution.model.SFR

        # Mask where clouds don't exist
        Mdot_cl_i_masked = np.ma.masked_where(
            solution.M_clouds[i] < solution.model.config.M_cloud_min / Msun,
            Mdot_cl_i
        )
        Mdot_cl_total += np.ma.filled(Mdot_cl_i_masked, 0)

    Mdot_cl_total_masked = np.ma.masked_where(Mdot_cl_total <= 0, Mdot_cl_total)
    ax2.loglog(solution.r, Mdot_cl_total_masked, 'b--', lw=1, label='Clouds')

    ax2.set_xlim(0.3, 30)
    ax2.set_ylim(0.01, 3)
    ax2.legend(fontsize=7, frameon=False, loc='upper left')

    if idx == 0:
        ax2.set_ylabel(r'$\dot{M}/{{\rm SFR}}$')
    else:
        ax2.set_yticklabels([])

    ax2.tick_params(labelbottom=False)

    # Cloud mass panel
    ax3 = fig1.add_subplot(gs[2, idx])

    for i in range(n_species):
        M_cl_i = np.ma.masked_where(
            solution.M_clouds[i] < solution.model.config.M_cloud_min / Msun,
            solution.M_clouds[i]
        )
        if n_species <= 11:
            ax3.loglog(solution.r, M_cl_i, '-', lw=0.8, alpha=0.7)
        else:
            # For many species, use thinner lines
            ax3.loglog(solution.r, M_cl_i, '-', lw=0.3, alpha=0.5)

    ax3.set_xlim(0.3, 30)
    ax3.set_ylim(1e-1, 1e7)
    ax3.set_xlabel(r'$r$ [kpc]')

    if idx == 0:
        ax3.set_ylabel(r'$M_{{\rm cl}}$ [$M_\odot$]')
    else:
        ax3.set_yticklabels([])

fig1.suptitle('Wind Solution Comparison: Effect of Cloud Species Discretization', fontsize=12)
plt.show()


# ## Use Built-in Plotting Functions
# 
# Show how the built-in plotting functions handle different numbers of cloud species.

# In[ ]:


# Plot using built-in plotting function for a few cases
from multiphasegalacticwind.plotting import plot_wind_solution

# Only plot models that succeeded
if 1 in successful_species:
    fig_1, axes_1 = plot_wind_solution(solutions[1], show_hot_only=False, figsize=(4, 7))
    fig_1.suptitle('1 Cloud Species', fontsize=10)

if 3 in successful_species:
    fig_3, axes_3 = plot_wind_solution(solutions[3], show_hot_only=False, figsize=(4, 7))
    fig_3.suptitle('3 Cloud Species', fontsize=10)
else:
    print("Note: 3 cloud species model failed - skipping plot")

if 11 in successful_species:
    fig_11, axes_11 = plot_wind_solution(solutions[11], show_hot_only=False, figsize=(4, 7))
    fig_11.suptitle('11 Cloud Species', fontsize=10)

plt.show()


# ## Compare Column Density Distributions
# 
# The column density distribution dN/dv is a key observable that can be compared with absorption line studies.

# In[ ]:


# Create figure for column density distributions
fig2, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()

# Plot column density for each model (only successful ones)
plot_species = successful_species if 'successful_species' in locals() else n_species_list
for idx, n_species in enumerate(plot_species):
    ax = axes[idx]
    solution = solutions[n_species]

    # Calculate column density distribution
    v_cloud, dN_dv = calculate_column_density_distribution(solution)

    # Plot with log-log scale
    if np.any(dN_dv > 0):
        ax.loglog(v_cloud, dN_dv, 'k-', lw=1.5)
        ax.set_xlim(10, 2000)
        ax.set_ylim(1e16, 1e22)
    ax.set_title(f'N = {n_species} species', fontsize=10)
    ax.set_xlabel(r'$v$ [km/s]')

    if idx % 3 == 0:
        ax.set_ylabel(r'$dN/dv$ [cm$^{{-2}}$ (km/s)$^{{-1}}$]')

    # Add velocity moments
    moments = calculate_velocity_moments(v_cloud, dN_dv)
    if moments['raw'][0] > 0:
        mean_v = moments.get('mean', 0)
        disp_v = moments.get('dispersion', 0)

        # Add text annotation
        ax.text(0.95, 0.95,
                f'$\\langle v \\rangle = {mean_v:.0f}$ km/s\n' +
                f'$\\sigma_v = {disp_v:.0f}$ km/s',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=8)

# Hide unused subplots
for idx in range(len(plot_species), len(axes)):
    axes[idx].set_visible(False)

fig2.suptitle('Column Density Distribution Comparison', fontsize=12)
plt.show()


# ## Show Column Density with Species Breakdown
# 
# For the 11-species model, show individual cloud contributions.

# In[ ]:


# Show detailed column density for 11-species model with log-log scale
from multiphasegalacticwind.observables import calculate_column_density_by_species

# Only plot if 11-species model succeeded
if 11 in successful_species:
    fig, ax = plt.subplots(figsize=(7, 5))

    # Calculate column density by species
    v_cloud, dN_dv_data = calculate_column_density_by_species(solutions[11])

    # Plot total
    ax.loglog(v_cloud, dN_dv_data['total'], 'k-', lw=2, label='Total')

    # Plot individual species with transparency
    for i, dN_dv_species in enumerate(dN_dv_data['species']):
        if np.any(dN_dv_species > 0):
            # Only label first few species to avoid crowding
            if i < 5:
                mass_str = f'{dN_dv_data["M_cloud0"][i]:.1e}'
                label = f'M0={mass_str} Msun'
            else:
                label = None
            ax.loglog(v_cloud, dN_dv_species, '-', alpha=0.6, lw=0.8, label=label)

    ax.set_xlim(10, 2000)
    ax.set_ylim(1e14, 1e22)
    ax.set_xlabel(r'$v$ [km/s]')
    ax.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km/s)$^{-1}$]')
    ax.set_title('Column Density Distribution - 11 Cloud Species (log-log)')
    ax.legend(loc='lower left', fontsize=7, frameon=True, ncol=1)
    # ax.grid(True, alpha=0.3, which='both')
    plt.show()
else:
    print("Note: 11 cloud species model did not complete successfully - skipping detailed plot")


# ## Summary Metrics Comparison
# 
# Compare key metrics across different discretizations.

# In[ ]:


# Create summary comparison plot
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

# Compare mass loading at 10 kpc (only successful runs)
plot_species = successful_species if 'successful_species' in locals() else n_species_list
mass_loadings = []
for n in plot_species:
    ml = solutions[n].mass_loading_at_10kpc
    if not np.isnan(ml):
        mass_loadings.append(ml)
    else:
        mass_loadings.append(None)

# Filter out None values
valid_indices = [i for i, ml in enumerate(mass_loadings) if ml is not None]
valid_species = [plot_species[i] for i in valid_indices]
valid_mass_loadings = [mass_loadings[i] for i in valid_indices]

if valid_mass_loadings:
    ax1.plot(valid_species, valid_mass_loadings, 'o-', color='darkblue', lw=2, ms=8)
ax1.axhline(y=base_params['eta_M'] + base_params['eta_M_cold'],
           color='gray', linestyle='--', label='Initial total')
ax1.set_xlabel('Number of Cloud Species')
ax1.set_ylabel(r'$\eta(10\,{{\rm kpc}})$')
ax1.set_title('Mass Loading at 10 kpc')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Compare mean velocities
mean_velocities = []
dispersions = []

for n_species in plot_species:
    solution = solutions[n_species]
    try:
        v_cloud, dN_dv = calculate_column_density_distribution(solution)
        moments = calculate_velocity_moments(v_cloud, dN_dv)
        mean_velocities.append(moments.get('mean', 0))
        dispersions.append(moments.get('dispersion', 0))
    except:
        mean_velocities.append(None)
        dispersions.append(None)

# Filter out None values
valid_indices = [i for i, mv in enumerate(mean_velocities) if mv is not None]
valid_species = [plot_species[i] for i in valid_indices]
valid_mean_velocities = [mean_velocities[i] for i in valid_indices]
valid_dispersions = [dispersions[i] for i in valid_indices]

if valid_mean_velocities:
    ax2.errorbar(valid_species, valid_mean_velocities, yerr=valid_dispersions,
                fmt='o-', color='darkred', lw=2, ms=8, capsize=5)
ax2.set_xlabel('Number of Cloud Species')
ax2.set_ylabel(r'$\langle v \rangle$ [km/s]')
ax2.set_title('Mean Cloud Velocity')
ax2.grid(True, alpha=0.3)

fig3.suptitle('Key Metrics vs Number of Cloud Species', fontsize=12)
plt.show()

# Print summary table
print("\nSummary Table:")
print("=" * 60)
print(f"{'N_species':>10} | {'η(10kpc)':>10} | {'<v> [km/s]':>12} | {'σ_v [km/s]':>12}")
print("-" * 60)
for i, n in enumerate(plot_species):
    ml = mass_loadings[i] if i < len(mass_loadings) else None
    mv = mean_velocities[i] if i < len(mean_velocities) else None
    dv = dispersions[i] if i < len(dispersions) else None
    
    if ml is not None and mv is not None:
        print(f"{n:>10} | {ml:>10.2f} | {mv:>12.0f} | {dv:>12.0f}")
    else:
        print(f"{n:>10} | {'FAILED':>10} | {'---':>12} | {'---':>12}")
print("=" * 60)

if valid_mass_loadings and valid_mean_velocities:
    print(f"\nKey findings (from successful runs):")
    print(f"  Mass loading at 10 kpc ranges from {min(valid_mass_loadings):.2f} to {max(valid_mass_loadings):.2f}")
    print(f"  Mean cloud velocity ranges from {min(valid_mean_velocities):.0f} to {max(valid_mean_velocities):.0f} km/s")
    print(f"  Velocity dispersion ranges from {min(valid_dispersions):.0f} to {max(valid_dispersions):.0f} km/s")


# ## Conclusions
# 
# This comparison shows how the number of cloud species affects:
# 
# 1. **Wind Evolution**: More cloud species provide smoother transitions in cloud mass evolution
# 2. **Mass Loading**: The total mass loading is relatively insensitive to discretization
# 3. **Column Density**: The velocity distribution shape changes with discretization
# 4. **Mean Velocity**: Higher discretization captures the full velocity range better
# 
# For most applications, 6-11 cloud species provides a good balance between accuracy and computational efficiency.
