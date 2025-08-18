# WindModel API Reference

## Overview

`WindModel` is the main class for running multiphase galactic wind simulations. It provides a high-level interface to configure, run, and analyze wind models.

## Class Definition

```python
class WindModel:
    """
    Main class for running multiphase galactic wind simulations.
    
    Parameters
    ----------
    SFR : float, optional
        Star formation rate [Msun/yr]. Default: 1.0
    v_circ : float, optional
        Circular velocity [km/s]. Default: 200.0
    eta_M : float, optional
        Hot mass loading factor. Default: 0.1
    eta_M_cold : float, optional
        Cold mass loading factor. Default: 1.0
    eta_E : float, optional
        Energy loading factor. Default: 1.0
    config : WindConfig, optional
        Configuration object with all parameters
    r_max_kpc : float, optional
        Maximum integration radius [kpc]. Default: 100.0
    rtol : float, optional
        Relative tolerance for ODE solver. Default: 1e-8
    atol : float, optional
        Absolute tolerance for ODE solver. Default: 1e-10
    """
```

## Constructor

### Basic Usage

```python
from multiphasegalacticwind import WindModel

# Simple model with default parameters
model = WindModel(SFR=10.0)

# Specify key parameters
model = WindModel(
    SFR=10.0,          # Star formation rate [Msun/yr]
    v_circ=200.0,      # Circular velocity [km/s]
    eta_M=0.1,         # Hot mass loading
    eta_M_cold=1.0,    # Cold mass loading
    eta_E=1.0          # Energy loading
)
```

### Advanced Configuration

```python
from multiphasegalacticwind import WindConfig, WindModel

# Create custom configuration
config = WindConfig(
    f_turb0=0.2,        # Turbulent velocity fraction
    drag_coeff=0.3,     # Cloud drag coefficient
    T_cl=5e3,           # Cloud temperature [K]
    N_cloud_species=10  # Number of cloud species
)

# Use with WindModel
model = WindModel(config=config, SFR=10.0)
```

## Methods

### `run()`

Run the wind simulation to completion.

```python
def run(self, progress_callback=None):
    """
    Run the wind simulation.
    
    Parameters
    ----------
    progress_callback : callable, optional
        Function called with (r, state) during integration
        
    Returns
    -------
    WindSolution
        Solution object containing results
    """
```

**Example:**

```python
# Basic run
solution = model.run()

# With progress monitoring
def monitor(r, state):
    print(f"r = {r:.1f} kpc, v = {state[0]:.1f} km/s")
    
solution = model.run(progress_callback=monitor)
```

### `run_hot_only()`

Run simulation with hot wind only (no clouds).

```python
def run_hot_only(self):
    """
    Run hot-only wind simulation for comparison.
    
    Returns
    -------
    WindSolution
        Solution object for hot-only wind
    """
```

**Example:**

```python
# Compare with and without clouds
hot_solution = model.run_hot_only()
full_solution = model.run()

print(f"Hot-only terminal velocity: {hot_solution.v_terminal:.1f} km/s")
print(f"Full model terminal velocity: {full_solution.v_terminal:.1f} km/s")
```

### `get_parameters()`

Get dictionary of all model parameters.

```python
def get_parameters(self):
    """
    Get dictionary of all model parameters.
    
    Returns
    -------
    dict
        All parameters including derived quantities
    """
```

**Example:**

```python
params = model.get_parameters()
print(f"Sonic radius: {params['r0']:.2f} kpc")
print(f"Energy injection: {params['Edot']:.2e} erg/s")
```

## WindSolution Object

The `run()` method returns a `WindSolution` object with the following attributes:

### Core Attributes

```python
solution.r          # Radius array [kpc]
solution.v          # Hot wind velocity [km/s]  
solution.rho        # Hot wind density [g/cm^3]
solution.P          # Pressure [dyne/cm^2]
solution.T          # Temperature [K]
solution.Z          # Metallicity [solar]
solution.M_cloud    # Cloud masses [Msun] (N_species × N_radius)
solution.v_cloud    # Cloud velocities [km/s] (N_species × N_radius)
solution.Z_cloud    # Cloud metallicities [solar] (N_species × N_radius)
```

### Derived Properties

```python
solution.Mach                  # Mach number
solution.mass_flux             # Mass flux [Msun/yr]
solution.momentum_flux         # Momentum flux [dyne]
solution.energy_flux           # Energy flux [erg/s]
solution.v_terminal            # Terminal velocity [km/s]
solution.v_at_10kpc           # Velocity at 10 kpc [km/s]
solution.mass_loading_at_10kpc # Mass loading at 10 kpc
```

### Observable Methods

```python
# Calculate column density distribution
v_cloud, dN_dv = solution.calculate_column_density_distribution()

# Get velocity moments
moments = solution.calculate_velocity_moments()
print(f"Mean: {moments['mean']:.1f} km/s")
print(f"Dispersion: {moments['dispersion']:.1f} km/s")

# Per-species column densities
v_cloud, dN_dv_species = solution.calculate_column_density_by_species()
```

## Parameter Validation

WindModel performs comprehensive parameter validation:

```python
try:
    model = WindModel(SFR=-1.0)  # Invalid!
except ValueError as e:
    print(f"Error: {e}")
    # Error: SFR must be positive, got -1.0
```

Validated parameters include:
- `SFR > 0` (star formation rate)
- `v_circ > 0` (circular velocity)
- `0 < eta_M < 100` (mass loading)
- `0 < eta_E < 100` (energy loading)
- `r_max_kpc > 1` (maximum radius)
- `rtol > 0, atol > 0` (solver tolerances)

## Performance Tips

### For MCMC/Parameter Studies

Use relaxed tolerances for faster integration:

```python
model = WindModel(
    SFR=10.0,
    rtol=1e-6,   # Relaxed relative tolerance
    atol=1e-8    # Relaxed absolute tolerance
)
```

### For Production Runs

Use strict tolerances for accuracy:

```python
model = WindModel(
    SFR=10.0,
    rtol=1e-10,  # Strict relative tolerance
    atol=1e-12   # Strict absolute tolerance
)
```

### Reduce Cloud Species

Fewer species speeds up integration:

```python
config = WindConfig(N_cloud_species=5)  # Only 5 species
model = WindModel(config=config, SFR=10.0)
```

## Common Patterns

### Parameter Study

```python
import numpy as np

sfr_values = np.logspace(-1, 2, 20)  # 0.1 to 100 Msun/yr
results = []

for sfr in sfr_values:
    model = WindModel(SFR=sfr, rtol=1e-6)
    solution = model.run()
    results.append({
        'SFR': sfr,
        'v_terminal': solution.v_terminal,
        'mass_loading': solution.mass_loading_at_10kpc
    })
```

### Comparison Study

```python
# Compare different mass loadings
eta_values = [0.01, 0.1, 1.0]
solutions = {}

for eta in eta_values:
    model = WindModel(SFR=10.0, eta_M=eta)
    solutions[eta] = model.run()
    
# Plot comparison
import matplotlib.pyplot as plt
for eta, sol in solutions.items():
    plt.plot(sol.r, sol.v, label=f'η_M = {eta}')
plt.xlabel('r [kpc]')
plt.ylabel('v [km/s]')
plt.legend()
```

## Error Handling

```python
# Check for integration failures
solution = model.run()

if solution.sol.status != 0:
    print(f"Integration terminated: {solution.sol.message}")
    print(f"Last radius: {solution.r[-1]:.1f} kpc")
    
# Common termination reasons:
# - "A termination event occurred" (hit sonic point)
# - "Reached maximum radius" (r_max)
# - "Negative velocity detected" (wind stalls)
```

## See Also

- [WindConfig](config.md) - Configuration options
- [WindSolution](solution.md) - Solution object details
- [Examples](../getting_started/examples.md) - Complete examples