#!/usr/bin/env python
"""
Simple working example of the multiphase galactic wind model.

This example shows the basic usage in just a few lines of code.
"""

from multiphasegalacticwind import (WindModel, plot_wind_solution, 
                                   plot_column_density_distribution)

# Create and run a wind model
print("Creating wind model...")
model = WindModel(
    SFR=10.0,           # Msun/yr
    eta_M=0.1,          # hot mass loading
    eta_M_cold=0.1,     # cold mass loading
    r_max_kpc=50.0,     # limit integration to 50 kpc for speed
    rtol=1e-6           # relaxed tolerance for speed
)

print("Running wind integration...")
solution = model.run()
print("Integration complete!")

# Print key results
print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")

# Calculate velocity distribution and moments
moments = solution.calculate_velocity_moments()
print(f"\nVelocity distribution:")
if 'mean' in moments:
    print(f"  Mean: {moments['mean']:.1f} km/s")
    print(f"  Dispersion: {moments['dispersion']:.1f} km/s")
else:
    print("  (Velocity moments not available)")

# Calculate column density
v_cloud, dN_dv = solution.calculate_column_density_distribution()
print(f"\nColumn density distribution:")
print(f"  Max dN/dv: {dN_dv.max():.2e} cm^-2 / (km/s)")

# Create plots
fig1, axes = plot_wind_solution(solution)
fig1.savefig('wind_profiles.pdf')

# Plot column density distribution with velocity moments
fig2, ax = plot_column_density_distribution(solution, show_moments=True)
fig2.savefig('column_density_distribution.pdf')

print("\nPlots saved as PDF files")

# Close all figures to prevent hanging
import matplotlib
matplotlib.pyplot.close('all')