#!/usr/bin/env python
"""
Quick test to verify the package works correctly.
"""

import sys
sys.path.insert(0, '.')

from multiphasegalacticwind import WindModel

print("Testing multiphasegalacticwind package...")

# Create a simple model
model = WindModel(
    SFR=10.0,
    eta_M=0.1,
    eta_M_cold=0.5,
    v_circ=150.0,
    N_cloud_species=5  # Fewer for quick test
)

print("Model created successfully")
print(f"  SFR = {model.SFR} Msun/yr")
print(f"  eta_M = {model.eta_M}")
print(f"  eta_M_cold = {model.eta_M_cold}")
print(f"  Number of cloud species = {model.N_cloud_species}")

# Run the model
print("\nRunning model...")
try:
    solution = model.run()
    print("Model ran successfully!")
    print(f"  Final radius: {solution.r[-1]:.1f} kpc")
    print(f"  Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
    print(f"  Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")
except Exception as e:
    print(f"Error running model: {e}")
    sys.exit(1)

print("\nPackage test passed!")