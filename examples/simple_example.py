#!/usr/bin/env python
"""
Simple working example of the multiphase galactic wind model.

This example shows the basic usage in just a few lines of code.
"""

from multiphasegalacticwind import (WindModel, plot_wind_solution, 
                                   plot_velocity_distribution,
                                   plot_column_density_distribution)

# Create and run a wind model
model = WindModel(SFR=20.0, eta_M=0.1, eta_M_cold=1.0)
solution = model.run()

# Print key results
print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")

# Calculate velocity distribution and moments
moments = solution.calculate_velocity_moments()
print(f"\nVelocity distribution:")
print(f"  Mean: {moments['mean']:.1f} km/s")
print(f"  Dispersion: {moments['dispersion']:.1f} km/s")

# Calculate column density
v_cloud, dN_dv = solution.calculate_column_density_distribution()
print(f"\nColumn density distribution:")
print(f"  Max dN/dv: {dN_dv.max():.2e} cm^-2 / (km/s)")

# Create plots
fig1, axes = plot_wind_solution(solution)
fig1.savefig('wind_profiles.pdf')

fig2, ax = plot_velocity_distribution(solution)
fig2.savefig('velocity_distribution.pdf')

fig3, ax = plot_column_density_distribution(solution)
fig3.savefig('column_density.pdf')

print("\nPlots saved as PDF files")