#!/usr/bin/env python
"""
Example showing how to use custom configuration parameters with WindModel.

This demonstrates the three ways to customize model parameters:
1. Creating a WindConfig instance
2. Passing parameters as keyword arguments
3. Modifying defaults globally
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.config import WindConfig

# Example 1: Using default configuration
print("Example 1: Default configuration")
model1 = WindModel(SFR=10.0, eta_M_cold=0.1, N_cloud_species=3, r_max_kpc=10.0)
print(f"Default f_turb0: {model1.config.f_turb0}")
print(f"Default drag_coeff: {model1.config.drag_coeff}")
print(f"Default opening angle: {model1.config.half_opening_angle:.2f} rad")
print()

# Example 2: Creating a custom configuration
print("Example 2: Custom WindConfig")
custom_config = WindConfig(
    f_turb0=0.2,                    # Double the turbulent velocity factor
    drag_coeff=0.3,                 # Reduce drag coefficient
    CoolingAreaChiPower=0.7,        # Modify cooling area scaling
    half_opening_angle=np.pi/3      # 60 degree opening angle
)
model2 = WindModel(SFR=10.0, eta_M_cold=0.1, N_cloud_species=3, 
                   r_max_kpc=10.0, config=custom_config)
print(f"Custom f_turb0: {model2.config.f_turb0}")
print(f"Custom drag_coeff: {model2.config.drag_coeff}")
print(f"Custom opening angle: {model2.config.half_opening_angle:.2f} rad")
print(f"Solid angle factor: {model2.config.Omwind/(4*np.pi):.3f}")
print()

# Example 3: Overriding specific parameters with kwargs
print("Example 3: Override with kwargs")
model3 = WindModel(SFR=10.0, eta_M_cold=0.1, N_cloud_species=3, 
                   r_max_kpc=10.0,
                   f_turb0=0.05,               # Halve the turbulent velocity
                   M_cloud_min=0.1*1.989e33)   # 0.1 Msun minimum cloud mass
print(f"Override f_turb0: {model3.config.f_turb0}")
print(f"Override M_cloud_min: {model3.config.M_cloud_min/1.989e33:.1f} Msun")
print()

# Example 4: Modifying cloud injection parameters
print("Example 4: Custom cloud injection")
model4 = WindModel(SFR=10.0, eta_M_cold=0.1, N_cloud_species=3,
                   r_max_kpc=10.0,
                   cold_cloud_injection_radial_power=4,  # Shallower injection profile
                   cold_cloud_injection_radial_extent=0.5*3.086e21)  # 0.5 kpc extent
print(f"Injection power: {model4.config.cold_cloud_injection_radial_power}")
print(f"Injection extent: {model4.config.cold_cloud_injection_radial_extent/3.086e21:.2f} kpc")
print()

# Run one model to show it works
print("Running model with custom configuration...")
try:
    solution = model2.run()
    print("Integration successful!")
    print(f"Final radius: {solution.r[-1]:.1f} kpc")
    print(f"Final velocity: {solution.v[-1]:.1f} km/s")
except Exception as e:
    print(f"Integration failed: {e}")

# Show all configurable parameters
print("\nAll configurable parameters:")
example_config = WindConfig()
for key, value in example_config.to_dict().items():
    if isinstance(value, float) and value > 1e10:
        print(f"  {key}: {value:.2e}")
    else:
        print(f"  {key}: {value}")