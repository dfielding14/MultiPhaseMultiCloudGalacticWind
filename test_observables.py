#!/usr/bin/env python
"""
Test the velocity distribution and observables functionality.
"""

import sys
sys.path.insert(0, '.')

from multiphasegalacticwind import WindModel
import numpy as np

print("Testing velocity distribution functionality...")

# Create a simple model
model = WindModel(
    SFR=10.0,
    eta_M=0.1,
    eta_M_cold=0.5,
    v_circ=150.0,
    N_cloud_species=5  # Fewer for quick test
)

print("\nRunning wind model...")
solution = model.run()

# Test velocity distribution calculation
print("\nCalculating velocity distribution...")
v_cloud, dN_dv = solution.calculate_velocity_distribution(
    r_min_kpc=0.1,
    r_max_kpc=10.0,
    velocity_units='km/s'
)

print(f"  Velocity range: {v_cloud.min():.1f} - {v_cloud.max():.1f} km/s")
print(f"  dN/dv shape: {dN_dv.shape}")
print(f"  Total number (integral): {np.trapz(dN_dv, v_cloud):.2e}")

# Test moment calculation
print("\nCalculating velocity moments...")
moments = solution.calculate_velocity_moments()

print(f"  Zeroth moment (total): {moments['raw'][0]:.2e}")
if moments['raw'][0] > 0:
    print(f"  Mean velocity: {moments['mean']:.1f} km/s")
    print(f"  Velocity dispersion: {moments['dispersion']:.1f} km/s")
    if 'skewness' in moments:
        print(f"  Skewness: {moments['skewness']:.2f}")

# Test individual cloud species
print("\nTesting individual cloud species...")
for i in range(model.N_cloud_species):
    v_i, dN_dv_i = solution.calculate_velocity_distribution(cloud_index=i)
    if np.any(dN_dv_i > 0):
        print(f"  Species {i} (M_cl = {model.M_cloud0[i]:.1e} Msun): " + 
              f"v range = {v_i[dN_dv_i>0].min():.0f}-{v_i[dN_dv_i>0].max():.0f} km/s")

# Test mass-weighted velocity
print("\nTesting mass-weighted velocity...")
from multiphasegalacticwind import calculate_mass_weighted_velocity
v_mass = calculate_mass_weighted_velocity(solution, [1.0, 5.0, 10.0])
for r, v in zip([1.0, 5.0, 10.0], v_mass):
    print(f"  At {r} kpc: {v:.1f} km/s")

# Sanity checks
print("\nRunning sanity checks...")
checks_passed = True

# Check 1: Velocity should be positive
if np.all(v_cloud >= 0):
    print("  ✓ All velocities are non-negative")
else:
    print("  ✗ Found negative velocities!")
    checks_passed = False

# Check 2: dN/dv should be non-negative
if np.all(dN_dv >= 0):
    print("  ✓ dN/dv is non-negative")
else:
    print("  ✗ Found negative dN/dv!")
    checks_passed = False

# Check 3: Mean velocity should be reasonable
if 0 < moments.get('mean', 0) < 2000:  # km/s
    print("  ✓ Mean velocity is reasonable")
else:
    print("  ✗ Mean velocity is unreasonable!")
    checks_passed = False

# Check 4: Dispersion should be positive
if moments.get('dispersion', 0) > 0:
    print("  ✓ Velocity dispersion is positive")
else:
    print("  ✗ Velocity dispersion is not positive!")
    checks_passed = False

if checks_passed:
    print("\nAll tests passed! ✓")
else:
    print("\nSome tests failed! ✗")
    sys.exit(1)