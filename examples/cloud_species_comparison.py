"""
Cloud Species Comparison Example

This example compares wind models with different numbers of cloud species
to understand how the discretization affects the wind solution and observables.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind import plot_wind_solution, plot_column_density_distribution
from multiphasegalacticwind.constants import Msun

# Base parameters for all models
base_params = {
    # Galaxy properties
    'v_circ': 150.0,           # km/s, circular velocity (Milky Way-like)
    'redshift': 0.0,
    
    # Wind launch properties
    'SFR': 20.0,               # Msun/yr, star formation rate
    'eta_M': 0.2,              # hot phase mass loading (increased for stability)
    'eta_M_cold': 0.2,         # cold phase mass loading (balanced with hot)
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

print("=" * 60)
print("Cloud Species Comparison")
print("=" * 60)
print(f"\nBase model parameters:")
print(f"  SFR = {base_params['SFR']} Msun/yr")
print(f"  eta_M = {base_params['eta_M']}")
print(f"  eta_M,cold = {base_params['eta_M_cold']}")
print(f"  eta_E = {base_params['eta_E']}")
print(f"  r_sonic = {base_params['r_star_kpc']} kpc = {base_params['r_star_kpc'] * 1000} pc")
print(f"  Cloud mass range = [{M_min}, {M_max}] Msun")
print(f"  Geometric mean cloud mass = {M_geometric_mean:.1f} Msun")

# Store models and solutions
models = {}
solutions = {}

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
    print(f"  Integration successful!")
    print(f"  Final radius reached: {solutions[n_species].r[-1]:.1f} kpc")
    print(f"  Final wind velocity: {solutions[n_species].v[-1]:.1f} km/s")
    print(f"  Mass loading at 10 kpc: {solutions[n_species].mass_loading_at_10kpc:.2f}")

print("\n" + "=" * 60)
print("Creating comparison plots...")

# Create figure with subplots for wind solutions
fig1 = plt.figure(figsize=(14, 10), constrained_layout=True)
gs = fig1.add_gridspec(3, len(n_species_list), height_ratios=[1, 1, 1])

# Color scheme for different models
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_species_list)))

# Plot wind solutions for each model
for idx, n_species in enumerate(n_species_list):
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
    ax1.set_ylim(30, 1500)
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
    from multiphasegalacticwind.constants import yr
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
        ax2.set_ylabel(r'$\dot{M}/{\rm SFR}$')
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
        ax3.set_ylabel(r'$M_{\rm cl}$ [$M_\odot$]')
    else:
        ax3.set_yticklabels([])

fig1.suptitle('Wind Solution Comparison: Effect of Cloud Species Discretization', fontsize=12)
plt.savefig('cloud_species_wind_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Create figure for column density distributions
fig2, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()

# Plot column density for each model
for idx, n_species in enumerate(n_species_list):
    ax = axes[idx]
    solution = solutions[n_species]
    
    # Calculate column density distribution
    from multiphasegalacticwind.observables import calculate_column_density_distribution
    v_cloud, dN_dv = calculate_column_density_distribution(solution)
    
    # Plot with log-log scale
    if np.any(dN_dv > 0):
        ax.loglog(v_cloud, dN_dv, 'k-', lw=1.5)
        ax.set_xlim(10, 2000)
        ax.set_ylim(1e16, 1e22)
    ax.set_title(f'N = {n_species} species', fontsize=10)
    ax.set_xlabel(r'$v$ [km/s]')
    
    if idx % 3 == 0:
        ax.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km/s)$^{-1}$]')
    
    # Add velocity moments
    from multiphasegalacticwind.observables import calculate_velocity_moments
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

# Hide the last subplot if not needed
axes[-1].set_visible(False)

fig2.suptitle('Column Density Distribution Comparison', fontsize=12)
plt.savefig('cloud_species_column_density_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Create summary comparison plot
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

# Compare mass loading at 10 kpc
mass_loadings = [solutions[n].mass_loading_at_10kpc for n in n_species_list]
ax1.plot(n_species_list, mass_loadings, 'o-', color='darkblue', lw=2, ms=8)
ax1.axhline(y=base_params['eta_M'] + base_params['eta_M_cold'], 
           color='gray', linestyle='--', label=f'Initial total ({base_params["eta_M"] + base_params["eta_M_cold"]:.1f})')
ax1.set_xlabel('Number of Cloud Species')
ax1.set_ylabel(r'$\eta(10\,{\rm kpc})$')
ax1.set_title('Mass Loading at 10 kpc')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Compare mean velocities
mean_velocities = []
dispersions = []

for n_species in n_species_list:
    solution = solutions[n_species]
    v_cloud, dN_dv = calculate_column_density_distribution(solution)
    moments = calculate_velocity_moments(v_cloud, dN_dv)
    mean_velocities.append(moments.get('mean', 0))
    dispersions.append(moments.get('dispersion', 0))

ax2.errorbar(n_species_list, mean_velocities, yerr=dispersions, 
            fmt='o-', color='darkred', lw=2, ms=8, capsize=5)
ax2.set_xlabel('Number of Cloud Species')
ax2.set_ylabel(r'$\langle v \rangle$ [km/s]')
ax2.set_title('Mean Cloud Velocity')
ax2.grid(True, alpha=0.3)

fig3.suptitle('Key Metrics vs Number of Cloud Species', fontsize=12)
plt.savefig('cloud_species_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nComparison complete!")
print("\nKey findings:")
print(f"  Mass loading at 10 kpc ranges from {min(mass_loadings):.2f} to {max(mass_loadings):.2f}")
print(f"  Mean cloud velocity ranges from {min(mean_velocities):.0f} to {max(mean_velocities):.0f} km/s")
print("\nPlots saved as:")
print("  - cloud_species_wind_comparison.png")
print("  - cloud_species_column_density_comparison.png")
print("  - cloud_species_metrics_comparison.png")