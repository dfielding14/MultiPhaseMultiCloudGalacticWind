# Cooling Physics

## Overview

Radiative cooling is a critical process in multiphase winds, affecting both the hot wind and turbulent mixing layers. This implementation uses tabulated cooling functions from Wiersma et al. (2009).

## Cooling Function

### Definition

The cooling rate per unit volume:

$$\mathcal{L} = n_H^2 \Lambda(T, Z)$$

where:
- $n_H$ = hydrogen number density [cm⁻³]
- $\Lambda(T, Z)$ = cooling function [erg cm³/s]
- $T$ = temperature [K]
- $Z$ = metallicity [solar units]

### Cooling Time

The cooling timescale:

$$t_{cool} = \frac{3 k_B T}{2 n_H \Lambda(T, Z)}$$

This represents the time for gas to radiate its thermal energy.

## Implementation

### Cooling Tables

The code uses pre-computed tables from Wiersma+09:

```python
# Tables location
data/z_0.000.hdf5  # Primordial
data/z_0.020.hdf5  # Solar metallicity

# Temperature range
T_min = 1e4 K
T_max = 1e9 K

# Density range
n_H_min = 1e-8 cm^-3
n_H_max = 1e4 cm^-3
```

### Interpolation

Cooling rates are interpolated in (T, n_H) space:

```python
from multiphasegalacticwind.cooling import cooling_rate

# Calculate cooling rate
Lambda = cooling_rate(T=1e6, n_H=0.01, Z=1.0)

# Calculate cooling time
t_cool = cooling_time(T=1e6, n_H=0.01, Z=1.0)
```

## Physical Processes

### Temperature Regimes

| Temperature [K] | Dominant Process | Typical Coolant |
|----------------|------------------|-----------------|
| 10⁴ - 10⁵ | Collisional excitation | H I, metals |
| 10⁵ - 10⁶ | Collisional ionization | He II, metals |
| 10⁶ - 10⁷ | Metal lines | O VII, O VIII, Fe |
| 10⁷ - 10⁸ | Bremsstrahlung | Free-free |
| > 10⁸ | Compton | Electron scattering |

### Metallicity Dependence

Cooling efficiency strongly depends on metallicity:

$$\Lambda(T, Z) \approx \Lambda_0(T) + Z \cdot \Lambda_Z(T)$$

where:
- $\Lambda_0$ = primordial cooling
- $\Lambda_Z$ = metal cooling contribution

## TRML Cooling

### Mixing Layer Temperature

The mixing layer has intermediate temperature:

$$T_{mix} = \sqrt{T_{hot} \cdot T_{cloud}}$$

For typical values:
- $T_{hot} = 10^6$ K
- $T_{cloud} = 10^4$ K
- $T_{mix} = 10^5$ K

### Enhanced Cooling

The mixing layer cools efficiently due to:
1. **Optimal temperature**: Peak cooling at $10^5 - 10^6$ K
2. **High density**: $\rho_{mix} = \sqrt{\rho_{hot} \cdot \rho_{cloud}}$
3. **Turbulent mixing**: Continual replenishment

## Cooling in the Code

### Hot Wind Cooling

From `core_physics.py`:

```python
# Cooling rate per volume
e_dot_cool = -(rho_wind/(muH*mp))**2 * Lambda_P_rho((Pressure, rho_wind))

# Only applied if Cooling_Factor > 0
if Cooling_Factor == 0:
    e_dot_cool = 0
```

### TRML Cooling Time

```python
# Cooling time in mixing layer
t_cool_layer = tcool_P(T_mix, Pressure/kb, Z_mix/Z_solar, 0.0, mu)

# Safeguard against negative cooling times
t_cool_layer = np.where(t_cool_layer < 0, 1e10*Myr, t_cool_layer)
```

### Cloud Growth Rate

Cooling affects cloud mass growth:

```python
# Dimensionless cooling parameter
ksi = r_cloud / (v_turb * t_cool_layer)

# Mass growth rate depends on cooling
if ksi < 1:  # Cooling-dominated
    Mdot_grow ∝ ksi^0.5
else:        # Turbulence-dominated
    Mdot_grow ∝ ksi^0.25
```

## Cooling Regimes

### Fast Cooling (ξ < 1)

When cooling is efficient:
- Rapid cloud growth
- Efficient momentum transfer
- Lower terminal velocities

### Slow Cooling (ξ > 1)

When cooling is inefficient:
- Slow cloud growth
- Cloud destruction dominates
- Higher terminal velocities

## Numerical Considerations

### Table Resolution

The cooling tables have finite resolution:
- 200 temperature points (log-spaced)
- 200 density points (log-spaced)
- Linear interpolation in log-log space

### Performance

Cooling calculations are optimized:
- Tables loaded lazily on first use
- Interpolators cached after creation
- Vectorized for multiple clouds

### Edge Cases

Special handling for:
- $T < 10^4$ K: Assume photoionization equilibrium
- $T > 10^9$ K: Extrapolate or use Compton
- Very low densities: Floor at minimum table value

## Configuration

### Disable Cooling

For testing, cooling can be disabled:

```python
config = WindConfig(Cooling_Factor=0)  # No cooling
```

### Metallicity Scaling

Adjust metallicity:

```python
config = WindConfig(metallicity=0.3)  # Sub-solar
config = WindConfig(metallicity=2.0)  # Super-solar
```

## Observational Implications

### Column Densities

Cooling affects observable ions:
- Efficient cooling → more cold gas → higher N_H
- Inefficient cooling → less cold gas → lower N_H

### Velocity Distributions

Cooling impacts kinematics:
- Strong cooling → massive clouds → lower velocities
- Weak cooling → light clouds → higher velocities

## Cooling Diagnostics

### Calculate Cooling Time

```python
from multiphasegalacticwind.cooling import tcool_P

# At specific conditions
T = 1e6  # K
P_over_kB = 1000  # K/cm^3
Z = 1.0  # Solar

t_cool = tcool_P(T, P_over_kB, Z, redshift=0, mu=0.62)
print(f"Cooling time: {t_cool/Myr:.1f} Myr")
```

### Cooling Curve

```python
import numpy as np
import matplotlib.pyplot as plt

T_range = np.logspace(4, 8, 100)
Lambda = np.array([cooling_rate(T, 0.01, 1.0) for T in T_range])

plt.loglog(T_range, Lambda)
plt.xlabel('T [K]')
plt.ylabel('Λ [erg cm³/s]')
plt.title('Cooling Function')
```

## References

- Wiersma, Schaye, & Smith (2009, MNRAS 393, 99)
- Fielding & Bryan (2024, ApJ)

## See Also

- [TRML Physics](trml.md) - Mixing layer details
- [Core Physics](../api/core_physics.md) - Implementation
- [Configuration](../api/config.md) - Cooling parameters