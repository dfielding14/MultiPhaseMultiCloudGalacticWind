# Turbulent Radiative Mixing Layer (TRML) Physics

## Overview

The Turbulent Radiative Mixing Layer (TRML) is the critical interface between hot wind and cold clouds where mass, momentum, and energy exchange occurs.

## Key Physics

### Turbulent Velocity

The turbulent velocity at the cloud surface:

$$v_{turb} = f_{turb} \times |v_{hot} - v_{cloud}|$$

where:
- $f_{turb} \sim 0.1-0.2$ is the turbulent velocity fraction
- $v_{hot} - v_{cloud}$ is the relative velocity

### Mixing Layer Properties

**Temperature**: Geometric mean of hot and cold
$$T_{mix} = \sqrt{T_{hot} \times T_{cloud}} \sim 10^5 \text{ K}$$

**Density**: Pressure equilibrium
$$\rho_{mix} = \sqrt{\rho_{hot} \times \rho_{cloud}}$$

**Metallicity**: Mass-weighted average
$$Z_{mix} = \sqrt{Z_{hot} \times Z_{cloud}}$$

## Mass Transfer

### Cloud Growth (Cooling)

When the mixing layer cools efficiently:
$$\dot{M}_{grow} = 4\pi R_{cloud}^2 \rho_{mix} v_{turb} \times f(ξ)$$

where the cooling factor:
$$ξ = \frac{R_{cloud}}{v_{turb} \times t_{cool}}$$

- $ξ < 1$: Cooling-dominated → $f(ξ) \propto ξ^{1/2}$
- $ξ > 1$: Turbulence-dominated → $f(ξ) \propto ξ^{1/4}$

### Cloud Destruction (Shredding)

Turbulent shredding removes mass:
$$\dot{M}_{loss} = -\frac{3 M_{cloud} v_{turb,cold}}{R_{cloud}}$$

where $v_{turb,cold} = v_{turb} \times χ^{0.5}$ with $χ = \rho_{cloud}/\rho_{hot}$.

## Momentum Transfer

### Drag Force

Ram pressure drag on clouds:
$$F_{drag} = \frac{1}{2} C_d \rho_{hot} \pi R_{cloud}^2 v_{rel}^2$$

where $C_d \sim 0.5$ is the drag coefficient.

### Momentum Exchange

Through mass transfer:
$$\dot{p}_{transfer} = v_{hot} \dot{M}_{grow} + v_{cloud} \dot{M}_{loss}$$

## Energy Dissipation

Kinetic energy dissipated in mixing:
$$\dot{E}_{diss} = \frac{1}{2} \dot{M}_{mix} (v_{hot} - v_{cloud})^2$$

This energy is radiated away efficiently at $T_{mix} \sim 10^5$ K.

## Implementation Details

### Area Enhancement

Turbulent structure increases surface area:
$$A_{eff} = A_{geom} \times χ^{0.5}$$

### Cooling Time

In the mixing layer:
$$t_{cool} = \frac{3 k_B T_{mix}}{2 n_{mix} \Lambda(T_{mix}, Z_{mix})}$$

### Numerical Considerations

- Regularization when $v_{turb} \to 0$
- Minimum cloud mass threshold
- Cooling time floor to prevent negative values

## Physical Regimes

### Strong Mixing ($f_{turb} > 0.15$)
- Rapid mass exchange
- Efficient cloud growth
- Lower terminal velocities

### Weak Mixing ($f_{turb} < 0.05$)
- Slow mass exchange
- Cloud destruction dominates
- Higher terminal velocities

### Optimal Cooling ($T_{mix} \sim 10^5$ K)
- Maximum cooling efficiency
- Rapid cloud growth
- Strong momentum coupling

## Observational Signatures

1. **Intermediate ions**: O VI, N V from mixing layer
2. **Velocity offset**: Clouds lag behind hot wind
3. **Line widths**: Turbulent broadening
4. **Column densities**: Enhanced by cloud growth

## References

- Fielding et al. (2020) - Fractal mixing layers
- Gronke & Oh (2018) - Cloud growth and entrainment
- Thompson et al. (2016) - Radiative mixing

## See Also

- [Physics Overview](overview.md)
- [Cooling Physics](cooling.md)
- [Cloud Evolution](../api/config.md#cloud-distribution)