# Migration Guide: Old Code to New API

This guide helps users migrate from the legacy script-based approach to the new package API.

## Overview of Changes

The code has been refactored from standalone scripts into a proper Python package with:
- Clean API via `WindModel` class
- Configuration management via `WindConfig`
- Separated concerns (physics, cooling, plotting)
- No global variables or import-time execution
- Sonic point conditions calculated from physics (not user inputs)

## Quick Comparison

### Old Way (Legacy Scripts)
```python
# From Multiphase_Wind_Evolution_Multicloud.py
mu_mol = 0.62
k_B = 1.38e-16
f_turb0 = 0.1
# ... many globals ...

# Set parameters directly
eta_M = 0.1
eta_M_cold = 1.0
SFR = 20 * Msun/yr

# Run simulation with complex setup
sol = solve_ivp(Wind_Evo, ...)
```

### New Way (Package API)
```python
from multiphasegalacticwind import WindModel, WindConfig

# Configure physics parameters
config = WindConfig(mu=0.62, f_turb0=0.1)

# Create and run model
model = WindModel(SFR=20.0, eta_M=0.1, eta_M_cold=1.0, config=config)
solution = model.run()
```

## Detailed Migration Steps

### 1. Import Changes

**Old:**
```python
from Multiphase_Wind_Evolution_Multicloud import *
import numpy as np
import matplotlib.pyplot as plt
```

**New:**
```python
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind import plot_wind_solution, plot_column_density_distribution
import numpy as np
```

### 2. Parameter Setup

**Old:** Parameters were set as module-level variables
```python
# Thermodynamics
mu_mol = 0.62
k_B = 1.38e-16
gamma = 5./3.

# Mixing parameters
f_turb0 = 0.1
drag_coeff = 0.5

# Cooling
metallicity = 10**-0.5
```

**New:** Parameters are managed by WindConfig
```python
config = WindConfig(
    mu=0.62,
    f_turb0=0.1,
    drag_coeff=0.5,
    metallicity=10**-0.5
)
```

### 3. Model Creation

**Old:** Manual setup of initial conditions
```python
# Set up cloud distribution
M_cloud0, eta_M_cold_array, Mdot_cold0, Ndot_cloud0 = \
    setup_cloud_powerlaw_distribution(
        log_M_cloud_min, log_M_cloud_max, N_cloud_species,
        alpha_cloud=2.0, eta_M_cold_tot=1.0, SFR=20*Msun/yr
    )

# Create initial state vector manually
y0 = np.zeros(4 + 3*N_cloud_species)
y0[0] = v_star * 1e5  # Convert to CGS
# ... etc ...
```

**New:** Simplified model creation
```python
model = WindModel(
    SFR=20.0,
    eta_M=0.1,
    eta_M_cold=1.0,
    cloud_mass_range=(1, 1e5),
    cloud_alpha=2.0,
    N_cloud_species=10,
    config=config
)
```

### 4. Running Simulations

**Old:** Direct solve_ivp call with complex setup
```python
sol = solve_ivp(
    lambda s, y: Wind_Evo(s, y, params),
    [r_star, r_max], y0,
    rtol=1e-8, atol=1e-10,
    dense_output=True,
    events=[cold_wind, wind_negative, all_clouds_frozen]
)
```

**New:** Simple run method
```python
solution = model.run()
```

### 5. Accessing Results

**Old:** Manual extraction from solution
```python
r = sol.t / kpc
v_wind = sol.y[0] / 1e5  # Convert from CGS
rho_wind = sol.y[1]
# Calculate derived quantities
T_wind = P / (rho_wind / (mu_mol * m_p)) / k_B
```

**New:** Pre-computed attributes
```python
r = solution.r          # Already in kpc
v = solution.v          # Already in km/s
T = solution.T          # Temperature in K
n = solution.n          # Number density in cm^-3
```

### 6. Plotting

**Old:** Manual plotting code
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].loglog(r, v_wind)
axes[0,0].set_xlabel('r [kpc]')
axes[0,0].set_ylabel('v [km/s]')
# ... many more lines ...
```

**New:** Built-in plotting functions
```python
fig, axes = plot_wind_solution(solution)
fig, ax = plot_column_density_distribution(solution)
```

## Common Migration Issues

### Issue 1: Variable Name Changes
- `mu_mol` → `mu`
- `k_B` → `kb` (imported from constants)
- `m_p` → `mp` (imported from constants)

### Issue 2: Unit Conversions
The new API handles unit conversions automatically:
- Distances: Input in kpc, converted to cm internally
- Velocities: Input in km/s, converted to cm/s internally
- Masses: Input in Msun, converted to g internally

### Issue 3: Missing Globals
All parameters must now be explicit:
```python
# Old: relied on global f_turb0
# New: must specify in config
config = WindConfig(f_turb0=0.1)
```

### Issue 4: Cooling Table Location
The cooling table is now in the package data directory:
```python
# Old: Lambda_tab_redshifts.npz in working directory
# New: multiphasegalacticwind/data/Lambda_tab_redshifts.npz
```

### Issue 5: Sonic Point Conditions
The sonic point conditions (n_star, v_star, T_star) are now calculated from physics:
```python
# Old: Manual specification (often inconsistent)
n_star = 0.1
v_star = 200.0
T_star = 5e6

# New: Automatically calculated from energy/mass conservation
model = WindModel(SFR=20.0, eta_M=0.1, eta_E=1.0)
# n_star, v_star, T_star are calculated internally
print(f"Sonic point: v={model.v_star:.0f} km/s, n={model.n_star:.2f} cm^-3, T={model.T_star:.1e} K")
```

## Complete Example Migration

### Old Script
```python
# Old approach
from Multiphase_Wind_Evolution_Multicloud import *

# Set parameters
SFR = 20 * Msun/yr
eta_M = 0.1
eta_M_cold = 1.0
f_turb0 = 0.1
metallicity = 10**-0.5

# Run simulation
# ... 100+ lines of setup code ...
sol = solve_ivp(Wind_Evo, ...)

# Plot results
plt.figure()
plt.loglog(sol.t/kpc, sol.y[0]/1e5)
plt.xlabel('r [kpc]')
plt.ylabel('v [km/s]')
plt.show()
```

### New Package
```python
# New approach
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig, plot_wind_solution

# Configure and run
config = WindConfig(f_turb0=0.1, metallicity=10**-0.5)
model = WindModel(SFR=20.0, eta_M=0.1, eta_M_cold=1.0, config=config)
solution = model.run()

# Plot results
fig, axes = plot_wind_solution(solution)
plt.show()

# Access key results
print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")
```

## Advanced Features

### Custom Configuration
```python
# Modify multiple parameters
config = WindConfig(
    mu=0.7,                    # Different molecular weight
    f_turb0=0.2,              # Enhanced mixing
    drag_coeff=1.0,           # Higher drag
    Cooling_Factor=0.5,       # Reduced cooling
    TurbulentVelocityChiPower=0.25  # Modified scaling
)
```

### Observables
```python
# Column density distribution
v_cloud, dN_dv_col = solution.calculate_column_density_distribution()

# Velocity moments
moments = solution.calculate_velocity_moments()
```

## Tips for Migration

1. **Start Simple**: Begin with default parameters and add complexity
2. **Check Units**: The new API handles conversions, don't convert twice
3. **Use Config**: Put all physics parameters in WindConfig
4. **Leverage Built-ins**: Use provided plotting and analysis functions
5. **Prefer Canonical Names**: Some legacy aliases exist (for example `eta_M_cold_tot`), but use the canonical API names in new code

## Need Help?

- See `examples/` directory for working examples
- Check `docs/windconfig_parameters.md` for parameter documentation
- Review the README for basic usage
- Open an issue on GitHub for specific problems
