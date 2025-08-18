#!/usr/bin/env python
"""
Generate example plots for documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.plotting import plot_wind_solution
import os

# Create output directory
os.makedirs('docs/examples/plots', exist_ok=True)

# Set up plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'figure.dpi': 100,
    'lines.linewidth': 2
})

print("Generating documentation plots...")

# ============================================================================
# 1. Basic Wind Solution
# ============================================================================
print("1. Generating wind solution plot...")
# Use more balanced parameters to avoid numerical issues
model = WindModel(SFR=10.0, v_circ=200.0, eta_M=0.2, eta_M_cold=0.5, eta_E=1.0)
solution = model.run()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Velocity
ax = axes[0, 0]
ax.plot(solution.r, solution.v, 'b-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.set_title('Wind Velocity')
ax.grid(True, alpha=0.3)

# Density
ax = axes[0, 1]
ax.loglog(solution.r, solution.rho, 'r-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Density [g/cm³]')
ax.set_title('Wind Density')
ax.grid(True, alpha=0.3)

# Temperature
ax = axes[1, 0]
ax.loglog(solution.r, solution.T, 'g-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Wind Temperature')
ax.grid(True, alpha=0.3)

# Mach number (calculate from velocity and sound speed)
ax = axes[1, 1]
# Sound speed: cs = sqrt(gamma * P / rho) = sqrt(gamma * kb * T / (mu * mp))
gamma = 5.0/3.0
kb = 1.38e-16  # erg/K
mp = 1.67e-24  # g
mu = 0.62
cs = np.sqrt(gamma * kb * solution.T / (mu * mp))  # cm/s
cs_km_s = cs / 1e5  # Convert to km/s
Mach = solution.v / cs_km_s
ax.plot(solution.r, Mach, 'm-', linewidth=2)
ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='Sonic')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Mach Number')
ax.set_title('Mach Number')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Multiphase Wind Solution (SFR=10 M☉/yr)', fontsize=16)
plt.tight_layout()
plt.savefig('docs/examples/plots/wind_solution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Radial Profiles
# ============================================================================
print("2. Generating radial profiles plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Velocity with acceleration
ax = axes[0, 0]
ax.plot(solution.r, solution.v, 'b-', linewidth=2, label='Velocity')
ax2 = ax.twinx()
dv_dr = np.gradient(solution.v, solution.r)
ax2.plot(solution.r[1:], dv_dr[1:], 'r--', alpha=0.6, label='Acceleration')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]', color='b')
ax2.set_ylabel('dv/dr [km/s/kpc]', color='r')
ax.set_title('Velocity and Acceleration')
ax.grid(True, alpha=0.3)

# Mass flux
ax = axes[0, 1]
ax.loglog(solution.r, solution.Mdot, 'g-', linewidth=2)
ax.axhline(model.SFR, color='k', linestyle='--', alpha=0.5, label=f'SFR={model.SFR}')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Mass Flux [M☉/yr]')
ax.set_title('Mass Flux')
ax.legend()
ax.grid(True, alpha=0.3)

# Pressure
ax = axes[1, 0]
ax.loglog(solution.r, solution.P, 'c-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Pressure [dyne/cm²]')
ax.set_title('Thermal Pressure')
ax.grid(True, alpha=0.3)

# Energy flux
ax = axes[1, 1]
energy_flux = 0.5 * solution.Mdot * 1.989e33 * (solution.v * 1e5)**2 / 3.15e7
ax.loglog(solution.r, energy_flux, 'm-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Energy Flux [erg/s]')
ax.set_title('Kinetic Energy Flux')
ax.grid(True, alpha=0.3)

plt.suptitle('Wind Radial Profiles', fontsize=16)
plt.tight_layout()
plt.savefig('docs/examples/plots/radial_profiles.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Cloud Evolution
# ============================================================================
print("3. Generating cloud evolution plot...")
# Use balanced parameters to avoid numerical issues
model = WindModel(SFR=10.0, v_circ=200.0, eta_M=0.2, eta_M_cold=0.5, eta_E=1.0, N_cloud_species=7)
solution = model.run()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cloud velocities
ax = axes[0]
ax.plot(solution.r, solution.v, 'k-', linewidth=3, label='Hot wind', alpha=0.8)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, model.N_cloud_species))
for i in range(model.N_cloud_species):
    ax.plot(solution.r, solution.v_cl[i, :], '--', 
            color=colors[i], alpha=0.7, linewidth=1.5,
            label=f'Cloud {i+1}' if i < 3 else None)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.set_title('Cloud Velocity Evolution')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)

# Cloud masses
ax = axes[1]
for i in range(model.N_cloud_species):
    ax.loglog(solution.r, solution.M_clouds[i, :], 
              color=colors[i], alpha=0.7, linewidth=1.5)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Cloud Mass [M☉]')
ax.set_title('Cloud Mass Evolution')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.1, 50)

plt.suptitle('Cloud Population Evolution', fontsize=16)
plt.tight_layout()
plt.savefig('docs/examples/plots/cloud_evolution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. Column Density Distribution
# ============================================================================
print("4. Generating column density plot...")
v_cloud, dN_dv = solution.calculate_column_density_distribution()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear scale
ax = axes[0]
ax.plot(v_cloud, dN_dv/1e13, 'b-', linewidth=2)
ax.fill_between(v_cloud, dN_dv/1e13, alpha=0.3)
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('dN/dv [10¹³ cm⁻² / (km/s)]')
ax.set_title('Column Density Distribution')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 600)

# Log scale with moments
ax = axes[1]
ax.semilogy(v_cloud, dN_dv, 'b-', linewidth=2)
moments = solution.calculate_velocity_moments()
ax.axvline(moments['mean'], color='r', linestyle='--', alpha=0.7, 
           label=f"Mean: {moments['mean']:.0f} km/s")
ax.axvspan(moments['mean'] - moments['dispersion'], 
           moments['mean'] + moments['dispersion'],
           alpha=0.2, color='r', label=f"σ: {moments['dispersion']:.0f} km/s")
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('dN/dv [cm⁻² / (km/s)]')
ax.set_title('Column Density with Moments')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 600)

plt.suptitle('Observable Column Density', fontsize=16)
plt.tight_layout()
plt.savefig('docs/examples/plots/column_density.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. Parameter Study
# ============================================================================
print("5. Generating parameter study plot...")
eta_E_values = np.logspace(-0.5, 0.5, 6)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(eta_E_values)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, eta_E in enumerate(eta_E_values):
    model = WindModel(SFR=10.0, v_circ=200.0, eta_M=0.2, eta_M_cold=0.3, eta_E=eta_E, rtol=1e-6)
    solution = model.run()
    
    # Velocity profiles
    axes[0].plot(solution.r, solution.v, color=colors[i], 
                linewidth=2, label=f'η_E = {eta_E:.2f}')
    
    # Column densities
    v_cloud, dN_dv = solution.calculate_column_density_distribution()
    axes[1].semilogy(v_cloud, dN_dv, color=colors[i], 
                    linewidth=2, label=f'η_E = {eta_E:.2f}')

axes[0].set_xlabel('Radius [kpc]')
axes[0].set_ylabel('Velocity [km/s]')
axes[0].set_title('Velocity vs Energy Loading')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 50)

axes[1].set_xlabel('Velocity [km/s]')
axes[1].set_ylabel('dN/dv [cm⁻² / (km/s)]')
axes[1].set_title('Column Density vs Energy Loading')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 700)

plt.suptitle('Parameter Study: Energy Loading Effects', fontsize=16)
plt.tight_layout()
plt.savefig('docs/examples/plots/parameter_study.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. Model vs Observations (Mock)
# ============================================================================
print("6. Generating model vs observations plot...")

# Create mock observational data
np.random.seed(42)
v_obs = np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300])
N_true = np.interp(v_obs, v_cloud, dN_dv)
N_obs = N_true * (1 + 0.15 * np.random.randn(len(v_obs)))  # Add 15% noise
N_err = 0.1 * N_obs  # 10% errors

fig, ax = plt.subplots(figsize=(10, 6))

# Data points
ax.errorbar(v_obs, N_obs/1e13, yerr=N_err/1e13, 
           fmt='ko', capsize=5, markersize=8, label='Mock Observations')

# Model
ax.plot(v_cloud, dN_dv/1e13, 'b-', linewidth=2, alpha=0.7, label='Model')

# Confidence band (simplified)
ax.fill_between(v_cloud, 0.8*dN_dv/1e13, 1.2*dN_dv/1e13, 
                alpha=0.2, color='b', label='20% Uncertainty')

ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('dN/dv [10¹³ cm⁻² / (km/s)]')
ax.set_title('Model vs Mock Observations')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-300, 400)

plt.tight_layout()
plt.savefig('docs/examples/plots/model_vs_obs.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. MCMC Corner Plot (Mock)
# ============================================================================
print("7. Generating mock MCMC corner plot...")

# Generate mock MCMC samples with correlations
n_samples = 5000
mean = [-0.7, -0.3, -0.2]  # log10([0.2, 0.5, 0.6])
cov = [[0.04, 0.02, 0.01],
       [0.02, 0.05, -0.01],
       [0.01, -0.01, 0.03]]
samples = np.random.multivariate_normal(mean, cov, n_samples)

# Create simplified corner plot
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
labels = [r'$\log \eta_M$', r'$\log \eta_{M,cold}$', r'$\log \eta_E$']

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        
        if i == j:
            # Diagonal: histograms
            ax.hist(samples[:, i], bins=30, color='blue', alpha=0.7)
            if i == 2:
                ax.set_xlabel(labels[i])
            if i == 0:
                ax.set_title(labels[i])
        elif i > j:
            # Lower triangle: 2D histograms
            ax.hist2d(samples[:, j], samples[:, i], bins=30, cmap='Blues')
            if i == 2:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
        else:
            # Upper triangle: empty
            ax.axis('off')

plt.suptitle('Mock MCMC Parameter Posteriors', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('docs/examples/plots/mcmc_corner.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. Fit Quality Plot
# ============================================================================
print("8. Generating fit quality plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals
ax = axes[0, 0]
residuals = (N_obs - N_true) / N_err
ax.scatter(v_obs, residuals, c='red', s=50, alpha=0.7)
ax.axhline(0, color='k', linestyle='-', alpha=0.5)
ax.axhspan(-1, 1, alpha=0.2, color='gray', label='1σ')
ax.axhspan(-2, 2, alpha=0.1, color='gray', label='2σ')
ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('Residuals (σ)')
ax.set_title('Fit Residuals')
ax.legend()
ax.grid(True, alpha=0.3)

# Chi-squared distribution
ax = axes[0, 1]
chi2_values = ((N_obs - N_true) / N_err)**2
ax.bar(range(len(chi2_values)), chi2_values, color='blue', alpha=0.7)
ax.axhline(1, color='r', linestyle='--', label='χ²=1')
ax.set_xlabel('Data Point')
ax.set_ylabel('χ² Contribution')
ax.set_title(f'χ² Distribution (Total: {np.sum(chi2_values):.1f})')
ax.legend()
ax.grid(True, alpha=0.3)

# QQ plot
ax = axes[1, 0]
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot')
ax.grid(True, alpha=0.3)

# Parameter constraints
ax = axes[1, 1]
# Mock confidence ellipses
from matplotlib.patches import Ellipse
ax.add_patch(Ellipse((0.2, 0.5), 0.15, 0.3, angle=30, 
                     alpha=0.3, color='blue', label='68% CL'))
ax.add_patch(Ellipse((0.2, 0.5), 0.25, 0.5, angle=30, 
                     alpha=0.2, color='blue', label='95% CL'))
ax.scatter([0.2], [0.5], c='red', s=100, marker='*', label='Best fit')
ax.set_xlabel(r'$\eta_M$')
ax.set_ylabel(r'$\eta_{M,cold}$')
ax.set_title('Parameter Constraints')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.0)

plt.suptitle('Fit Quality Assessment', fontsize=16)
plt.tight_layout()
plt.savefig('docs/examples/plots/fit_quality.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll plots generated successfully!")
print("Plots saved to: docs/examples/plots/")
print("\nGenerated plots:")
for plot in ['wind_solution.png', 'radial_profiles.png', 'cloud_evolution.png',
             'column_density.png', 'parameter_study.png', 'model_vs_obs.png',
             'mcmc_corner.png', 'fit_quality.png']:
    print(f"  - {plot}")