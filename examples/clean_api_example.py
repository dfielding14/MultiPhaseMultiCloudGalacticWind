#!/usr/bin/env python
"""
Example showing the clean API with proper configuration management.

This demonstrates best practices for using the multiphase galactic wind package.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from multiphasegalacticwind import WindModel, WindConfig

# Example 1: Using defaults
print("Example 1: Default configuration")
model = WindModel(SFR=10.0, eta_M_cold=0.1, N_cloud_species=3)
print(f"Configuration:")
print(f"  f_turb0 = {model.config.f_turb0}")
print(f"  drag_coeff = {model.config.drag_coeff}")
print(f"  mu = {model.config.mu}")
print(f"  metallicity = {model.config.metallicity}")
print()

# Example 2: Custom configuration for low-metallicity gas
print("Example 2: Low metallicity configuration")
low_Z_config = WindConfig(
    mu=0.59,                    # Lower mean molecular weight
    metallicity=0.1,            # 0.1 solar
    f_turb0=0.05,              # Less turbulent mixing
    drag_coeff=0.3             # Less drag
)

model_lowZ = WindModel(
    SFR=5.0,                   # Lower SFR
    eta_M_cold=0.5,            # But higher cold gas fraction
    N_cloud_species=5,
    config=low_Z_config
)

print(f"Low-Z Configuration:")
print(f"  mu = {model_lowZ.config.mu}")
print(f"  metallicity = {model_lowZ.config.metallicity}")
print(f"  f_turb0 = {model_lowZ.config.f_turb0}")
print()

# Example 3: High-redshift galaxy
print("Example 3: High-redshift galaxy")
model_highz = WindModel(
    SFR=50.0,
    eta_M=0.2,
    eta_M_cold=2.0,
    v_circ=200.0,              # Higher circular velocity
    redshift=2.0,              # This should affect cooling
    mu=0.61,                   # Slightly different composition
    metallicity=0.3,           # Sub-solar
    half_opening_angle=np.pi/3 # Narrower outflow
)

print(f"High-z Configuration:")
print(f"  redshift = {model_highz.config.redshift}")
print(f"  Omwind = {model_highz.config.Omwind/(4*np.pi):.3f} × 4π")
print(f"  metallicity = {model_highz.config.metallicity}")
print()

# Example 4: Showing all parameters
print("Example 4: All configurable parameters")
print("-" * 50)
for key, value in model.config.to_dict().items():
    if isinstance(value, float) and abs(value) > 1e10:
        print(f"{key:.<35} {value:.2e}")
    else:
        print(f"{key:.<35} {value}")

# Show that parameters affect the physics
print("\nPhysical implications:")
print(f"Default opening angle: Ω = {model.config.Omwind/(4*np.pi):.3f} × 4π")
print(f"High-z opening angle:  Ω = {model_highz.config.Omwind/(4*np.pi):.3f} × 4π")
print(f"Mass flux scaling: ∝ Ω × r² × ρ × v")
print(f"Smaller Ω → higher densities for same mass flux")