# Quick Start Guide

## Basic Usage

### 1. Create a Wind Model

```python
from multiphasegalacticwind import WindModel

# Milky Way-like galaxy
model = WindModel(
    SFR=1.0,           # Star formation rate [Msun/yr]
    v_circ=200.0,      # Circular velocity [km/s]
    eta_M=0.1,         # Hot mass loading
    eta_M_cold=1.0,    # Cold mass loading
    eta_E=1.0          # Energy loading
)
```

### 2. Run the Simulation

```python
# Integrate to 100 kpc
solution = model.run()

# Access results
print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")
```

### 3. Visualize Results

```python
from multiphasegalacticwind.plotting import plot_wind_solution
import matplotlib.pyplot as plt

# Create standard plot
fig, axes = plot_wind_solution(solution)
plt.show()
```

## Common Use Cases

### Starburst Galaxy

```python
model = WindModel(
    SFR=100.0,         # High SFR
    v_circ=150.0,      # Smaller galaxy
    eta_E=3.0,         # Strong energy loading
    r_max_kpc=200      # Extend to larger radius
)
```

### Parameter Study

```python
import numpy as np

sfr_values = [0.1, 1.0, 10.0, 100.0]
solutions = []

for sfr in sfr_values:
    model = WindModel(SFR=sfr)
    solution = model.run()
    solutions.append(solution)
    print(f"SFR={sfr}: v(10kpc)={solution.v_at_10kpc:.0f} km/s")
```

### Custom Configuration

```python
from multiphasegalacticwind import WindConfig, WindModel

# Customize TRML parameters
config = WindConfig(
    f_turb0=0.2,       # Stronger turbulence
    drag_coeff=0.3,    # Lower drag
    T_cl=5e3           # Cooler clouds
)

model = WindModel(config=config, SFR=10.0)
```

## Observables

### Column Density Distribution

```python
# Calculate dN/dv
v_cloud, dN_dv = solution.calculate_column_density_distribution()

# Get velocity moments
moments = solution.calculate_velocity_moments()
print(f"Mean velocity: {moments['mean']:.1f} km/s")
print(f"Velocity dispersion: {moments['dispersion']:.1f} km/s")
```

### Multi-Species Analysis

```python
# Get per-species column densities
v_cloud, dN_dv_species = solution.calculate_column_density_by_species()

# Plot each species
import matplotlib.pyplot as plt
for i, dN_dv_i in enumerate(dN_dv_species['species']):
    plt.plot(v_cloud, dN_dv_i, label=f'M={dN_dv_species["M_cloud0"][i]:.1e} Msun')
plt.xlabel('v [km/s]')
plt.ylabel('dN/dv [cm^-2 / (km/s)]')
plt.legend()
```

## Tips

1. **Start Simple**: Use default parameters first
2. **Check Units**: All inputs use convenient units (Msun/yr, km/s, kpc)
3. **Validate Results**: Compare hot-only vs full solution
4. **Monitor Events**: Check `solution.sol.message` for termination reason

## Next Steps

- Read the [Physics Overview](../physics/overview.md)
- Explore [Examples](examples.md)
- See [Parameter Guide](../guide/parameters.md)