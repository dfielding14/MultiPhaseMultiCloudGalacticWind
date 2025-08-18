# Parameter Selection Guide

## Overview

Choosing appropriate parameters is crucial for realistic wind simulations. This guide provides recommendations based on galaxy type and observational constraints.

## Key Parameter Categories

### 1. Loading Parameters (η)

These control the efficiency of mass and energy injection:

| Parameter | Symbol | Typical Range | Impact |
|-----------|--------|---------------|---------|
| Hot mass loading | η_M | 0.01 - 1.0 | Hot wind density |
| Cold mass loading | η_M_cold | 0.1 - 10.0 | Cloud population |
| Energy loading | η_E | 0.1 - 3.0 | Wind velocity |

**Guidelines**:
- **Dwarf galaxies**: Higher loading (η_M ~ 0.3, η_E ~ 1-3)
- **Massive galaxies**: Lower loading (η_M ~ 0.01-0.1, η_E ~ 0.1-1)
- **Starbursts**: Enhanced cold loading (η_M_cold ~ 3-10)

### 2. Galaxy Properties

| Parameter | Symbol | Units | How to Estimate |
|-----------|--------|-------|-----------------|
| Star formation rate | SFR | Msun/yr | UV/IR luminosity |
| Circular velocity | v_circ | km/s | Rotation curve, σ_stars |
| Scale radius | r_0 | kpc | 0.1-0.2 × R_eff |

**Circular Velocity Estimation**:
```python
# From stellar velocity dispersion
v_circ ≈ √2 * σ_stars

# From galaxy mass
v_circ ≈ √(G * M_gal / R_eff)

# From rotation curve
v_circ = v_rot(R_flat)
```

### 3. Cloud Distribution

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| N_cloud_species | 10 | 5-20 | Balance accuracy vs speed |
| M_cloud_min | 10 Msun | 1-100 | Smallest resolved clouds |
| M_cloud_max | 1e6 Msun | 1e5-1e7 | GMC scale |
| cloud_alpha | 2.0 | 1.5-2.5 | Observed distribution |

**Choosing N_cloud_species**:
- **Quick exploration**: 5 species
- **Standard runs**: 10 species
- **Publication quality**: 15-20 species

### 4. TRML Physics

| Parameter | Symbol | Default | Range | Physical Meaning |
|-----------|--------|---------|-------|------------------|
| Turbulent fraction | f_turb0 | 0.1 | 0.05-0.3 | v_turb/v_rel ratio |
| Drag coefficient | C_d | 0.475 | 0.3-1.0 | Ram pressure efficiency |
| Cloud temperature | T_cl | 1e4 K | 5e3-2e4 | Photoionization equilibrium |

**Turbulence Guidelines**:
- **Smooth winds**: f_turb0 ~ 0.05
- **Standard mixing**: f_turb0 ~ 0.1
- **Enhanced mixing**: f_turb0 ~ 0.2

## Parameter Sets by Galaxy Type

### Milky Way-like Galaxies

```python
config = WindConfig(
    # Standard TRML
    f_turb0=0.1,
    drag_coeff=0.475,
    T_cl=1e4,
    # Standard distribution
    N_cloud_species=10,
    cloud_alpha=2.0
)

model = WindModel(
    config=config,
    SFR=1.0,           # Msun/yr
    v_circ=200.0,      # km/s
    eta_M=0.1,         # Low hot loading
    eta_M_cold=1.0,    # Moderate cold loading
    eta_E=1.0          # Moderate energy
)
```

### Dwarf Galaxies

```python
config = WindConfig(
    # Enhanced mixing in shallow potential
    f_turb0=0.15,
    drag_coeff=0.3,    # Less drag
)

model = WindModel(
    config=config,
    SFR=0.1,           # Low SFR
    v_circ=50.0,       # Small galaxy
    eta_M=0.3,         # Higher loading
    eta_M_cold=3.0,    # More cold gas
    eta_E=3.0          # Strong feedback
)
```

### Starburst Galaxies

```python
config = WindConfig(
    # Strong turbulence
    f_turb0=0.2,
    # More cloud species for complex distribution
    N_cloud_species=15,
    M_cloud_max=1e7    # Larger clouds
)

model = WindModel(
    config=config,
    SFR=100.0,         # Intense star formation
    v_circ=150.0,
    eta_M=0.2,
    eta_M_cold=5.0,    # Massive cold outflow
    eta_E=2.0,
    r_max_kpc=200      # Extended wind
)
```

## Numerical Parameters

### Integration Tolerances

| Purpose | rtol | atol | Speed | Accuracy |
|---------|------|------|-------|----------|
| Quick exploration | 1e-6 | 1e-8 | Fast | Good |
| Standard runs | 1e-8 | 1e-10 | Medium | Excellent |
| Publication | 1e-10 | 1e-12 | Slow | Best |

### Convergence Testing

```python
# Test numerical convergence
tolerances = [(1e-6, 1e-8), (1e-8, 1e-10), (1e-10, 1e-12)]
results = []

for rtol, atol in tolerances:
    model = WindModel(SFR=10.0, rtol=rtol, atol=atol)
    solution = model.run()
    results.append(solution.v_terminal)
    
# Check convergence
convergence = np.diff(results) / results[:-1]
print(f"Relative changes: {convergence}")
```

## Parameter Studies

### Scanning Parameter Space

```python
import numpy as np
from itertools import product

# Define parameter grid
eta_M_values = np.logspace(-2, 0, 5)      # 0.01 to 1
eta_E_values = np.logspace(-1, 0.5, 5)    # 0.1 to ~3

# Grid search
results = {}
for eta_M, eta_E in product(eta_M_values, eta_E_values):
    try:
        model = WindModel(
            SFR=10.0,
            eta_M=eta_M,
            eta_E=eta_E,
            eta_M_cold=1.0,
            rtol=1e-6  # Fast for parameter study
        )
        solution = model.run()
        results[(eta_M, eta_E)] = {
            'v_terminal': solution.v_terminal,
            'mass_flux_10kpc': solution.mass_flux[solution.r <= 10][-1]
        }
    except:
        results[(eta_M, eta_E)] = None
```

### Latin Hypercube Sampling

```python
from scipy.stats import qmc

# Define parameter bounds (log space)
bounds = [
    [-2, 0],    # log10(eta_M): 0.01 to 1
    [-1, 1],    # log10(eta_M_cold): 0.1 to 10
    [-1, 0.5]   # log10(eta_E): 0.1 to ~3
]

# Generate samples
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=100)
scaled = qmc.scale(sample, bounds[0], bounds[1])

# Convert from log space
parameters = 10**scaled
```

## Observational Constraints

### From UV Absorption Lines

```python
# Typical ion velocities constrain eta_E
v_CII_observed = 200  # km/s
v_SiII_observed = 150  # km/s

# Energy loading from velocity
# v_terminal ~ 3 * v_circ * sqrt(eta_E)
eta_E_estimated = (v_CII_observed / (3 * v_circ))**2
```

### From Column Densities

```python
# Column densities constrain mass loading
N_H_observed = 1e20  # cm^-2

# Very rough scaling
# N_H ~ eta_M * eta_M_cold * SFR * ...
# Requires full model for accurate constraint
```

## Common Issues

### Wind Fails to Launch
- **Symptom**: Integration stops early
- **Solutions**:
  - Increase η_E (more energy)
  - Decrease η_M_cold (less cloud drag)
  - Check v_circ is correct

### Unrealistic Velocities
- **Symptom**: v_terminal > 1000 km/s
- **Solutions**:
  - Decrease η_E
  - Increase η_M (more mass to accelerate)
  - Add more clouds (η_M_cold)

### Numerical Instabilities
- **Symptom**: Oscillations, NaN values
- **Solutions**:
  - Tighten tolerances (smaller rtol/atol)
  - Increase sonic_point_tolerance
  - Reduce f_turb0 if very high

## Best Practices

1. **Start with defaults**: Use standard parameters first
2. **Validate against hot-only**: Compare with/without clouds
3. **Check physical scales**: Ensure r_0 < 0.2 × galaxy size
4. **Monitor convergence**: Test numerical parameters
5. **Use observational constraints**: Match velocities/columns when available

## See Also

- [WindConfig API](../api/config.md) - Configuration details
- [Examples](../getting_started/examples.md) - Working examples
- [Troubleshooting](troubleshooting.md) - Common problems