# Hot Wind Equations (From Codebase)

## Complete Hot Wind Evolution Equations

These equations are extracted directly from `core_physics.py` and represent the actual implementation.

### State Variables

The hot wind is described by four state variables:
- `v_wind` - Wind velocity [cm/s]
- `rho_wind` - Wind density [g/cm³]
- `Pressure` - Thermal pressure [dyne/cm²]
- `rhoZ_wind` - Metal density [g/cm³]

### Derived Quantities

```python
cs_sq_wind = gamma * Pressure / rho_wind                    # Sound speed squared
Mach_sq_wind = v_wind**2 / cs_sq_wind                      # Mach number squared
Z_wind = rhoZ_wind / rho_wind                              # Metallicity
Phir = v_circ**2 * np.log(r)                              # Gravitational potential
vBsq_wind = 0.5 * v_wind**2 + (gamma/(gamma-1)) * Pressure/rho_wind + Phir  # Bernoulli constant
```

### Source Terms

The hot wind evolution includes source terms from cloud interactions:

**Density Source** (from cloud mass exchange):
$$\frac{d\rho}{dt} = -\sum_i n_{cloud,i} \dot{M}_{cloud,i}$$

where:
- $n_{cloud,i} = N_{dot,i} / (\Omega v_{cloud,i} r^2)$ is the cloud number density
- $\dot{M}_{cloud,i} = \dot{M}_{grow,i} + \dot{M}_{loss,i}$ is the net mass transfer

**Momentum Source** (from drag and mass transfer):
$$\frac{dp}{dt} = -\sum_i n_{cloud,i} \left( \dot{p}_{transfer,i} + \dot{p}_{ram,i} \right)$$

where:
- $\dot{p}_{transfer} = v_{wind} \dot{M}_{grow} + v_{cloud} \dot{M}_{loss}$
- $\dot{p}_{ram} = \frac{1}{2} C_d \rho_{wind} \pi r_{cloud}^2 v_{rel}^2$

**Energy Source** (from cooling and cloud interactions):
$$\frac{de}{dt} = -\sum_i n_{cloud,i} \left( \dot{e}_{transfer,i} + \dot{p}_{ram,i} v_{wind} \right) + \dot{e}_{cool}$$

where:
- $\dot{e}_{transfer} = v_{B,wind}^2 \dot{M}_{grow} + v_{B,cloud}^2 \dot{M}_{loss}$
- $\dot{e}_{cool} = -\left(\frac{\rho_{wind}}{\mu_H m_p}\right)^2 \Lambda(P, \rho)$ (radiative cooling)

**Metallicity Source**:
$$\frac{d(\rho Z)}{dt} = -\sum_i n_{cloud,i} \left( Z_{wind} \dot{M}_{grow,i} + Z_{cloud,i} \dot{M}_{loss,i} \right)$$

### Differential Equations (with Sonic Regularization)

The code implements sonic point regularization to handle the singularity at Mach = 1:

```python
# Regularization near sonic point
sonic_regularization_width = 0.01
if np.abs(Mach_sq_wind - 1.0) < sonic_regularization_width:
    epsilon = Mach_sq_wind - 1.0
    if np.abs(epsilon) < 1e-10:
        epsilon = 1e-10 * np.sign(epsilon) if epsilon != 0 else 1e-10
    denominator = epsilon
else:
    denominator = 1.0 - (1.0/Mach_sq_wind)
```

**Velocity Evolution**:
$$\frac{dv}{dr} = \frac{v/r}{1 - 1/\mathcal{M}^2} \left[ \frac{2}{\mathcal{M}^2} - \left(\frac{v_c}{v}\right)^2 - \mathcal{S}_v \right]$$

where the source term is:
$$\mathcal{S}_v = \frac{1}{\rho v^2/r} \left[ \frac{d\rho}{dt} \frac{\gamma+1}{2} - \gamma \frac{dp/dt}{v} + (\gamma-1) \frac{de/dt}{v^2} - (\gamma-1) \frac{\Phi_r}{v^2} \frac{d\rho}{dt} \right]$$

**Density Evolution**:
$$\frac{d\rho}{dr} = \frac{\rho/r}{1 - 1/\mathcal{M}^2} \left[ -2 + \left(\frac{v_c}{v}\right)^2 + \mathcal{S}_\rho \right]$$

where:
$$\mathcal{S}_\rho = \frac{1}{\rho v/r} \left[ \frac{d\rho}{dt} \frac{\gamma+3}{2} - \gamma \frac{dp/dt}{v} + (\gamma-1) \frac{de/dt}{v^2} - \frac{d\rho/dt}{\mathcal{M}^2} + (\gamma-1) \frac{\Phi_r}{v^2} \frac{d\rho}{dt} \right]$$

**Pressure Evolution**:
$$\frac{dP}{dr} = \frac{P \gamma/r}{1 - 1/\mathcal{M}^2} \left[ -2 + \left(\frac{v_c}{v}\right)^2 + \mathcal{S}_P \right]$$

where:
$$\mathcal{S}_P = \frac{1}{\rho v/r} \left[ \frac{d\rho}{dt} + \frac{d\rho}{dt} \frac{(\gamma-1) \mathcal{M}^2}{2} \left(1 - \frac{2\Phi_r}{v^2}\right) - \frac{dp/dt}{v} + (\gamma-1) \mathcal{M}^2 \frac{de/dt - v \cdot dp/dt}{v^2} \right]$$

**Metal Density Evolution**:
$$\frac{d(\rho Z)}{dr} = \frac{\rho Z/r}{1 - 1/\mathcal{M}^2} \left[ -2 + \left(\frac{v_c}{v}\right)^2 + \mathcal{S}_\rho \right] + \frac{\rho Z/r}{\rho v/r} \left[ \frac{d(\rho Z)/dt}{Z} - \frac{d\rho}{dt} \right]$$

### Hot-Only Wind (Comparison Case)

For the hot-only wind (no clouds), the equations simplify to:

```python
def Hot_Wind_Evo(r, state, params):
    v_wind, rho_wind, Pressure, rhoZ_wind = state
    v_circ = params[0]
    
    cs_sq = gamma * Pressure / rho_wind
    Mach_sq = v_wind**2 / cs_sq
    
    # No source terms for hot-only
    drhodt = 0
    dpdt = 0
    dedt = 0
    drhoZdt = 0
    
    # Apply same sonic regularization
    if np.abs(Mach_sq - 1.0) < 0.01:
        epsilon = Mach_sq - 1.0
        if np.abs(epsilon) < 1e-10:
            epsilon = 1e-10 * np.sign(epsilon)
        factor = (v_wind/r) / epsilon
    else:
        factor = (v_wind/r) / (1 - 1/Mach_sq)
    
    dv_dr = factor * (2/Mach_sq - (v_circ/v_wind)**2)
    drho_dr = (rho_wind/r) / (1 - 1/Mach_sq) * (-2 + (v_circ/v_wind)**2)
    dP_dr = (Pressure * gamma/r) / (1 - 1/Mach_sq) * (-2 + (v_circ/v_wind)**2)
    drhoZ_dr = (rhoZ_wind/r) / (1 - 1/Mach_sq) * (-2 + (v_circ/v_wind)**2)
```

### Injection at Base (r < r₀)

Within the injection radius, additional source terms are applied:

```python
if r < r0:
    # Energy injection
    dedt += Edot_per_Vol
    
    # Mass injection
    drhodt += Mdot_per_Vol
    
    # Momentum injection (optional)
    dpdt += Mdot_per_Vol * v_injection
```

### Key Physical Constants

From the implementation:
- $\gamma = 5/3$ (adiabatic index)
- $\mu = 0.62$ (mean molecular weight)
- $m_p = 1.67262192369 \times 10^{-24}$ g (proton mass)
- $k_B = 1.380649 \times 10^{-16}$ erg/K (Boltzmann constant)

### Numerical Considerations

1. **Sonic Point**: The factor $(1 - 1/\mathcal{M}^2)^{-1}$ diverges at $\mathcal{M} = 1$
   - Solution: Taylor expansion regularization with width $\epsilon = 0.01$

2. **Negative Pressure/Density**: Integration stops if $P \leq 0$ or $\rho \leq 0$

3. **NaN Prevention**: Returns zero derivatives if any NaN detected

4. **Event Detection**: Integration terminates on:
   - Supersonic → subsonic transition
   - Negative velocity
   - Maximum radius reached