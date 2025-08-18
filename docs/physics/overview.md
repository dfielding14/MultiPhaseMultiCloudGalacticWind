# Physics Overview

## Introduction

The multiphase galactic wind model simulates the complex interactions between hot gas and embedded cold clouds in galactic outflows. This implementation follows the framework of **Fielding & Bryan (2024)** "The Structure of Multiphase Galactic Winds".

## Key Physical Components

### 1. Hot Wind Phase

The hot phase represents the smooth, volume-filling component of the wind:

- **Temperature**: $T_h \sim 10^6 - 10^7$ K
- **Density**: $\rho_h \sim 10^{-27} - 10^{-25}$ g/cm³
- **Velocity**: Accelerates from subsonic to supersonic
- **Pressure**: Provides acceleration against gravity

The hot wind evolution follows:

$$\frac{dv_h}{dr} = \frac{v_h/r}{1 - 1/\mathcal{M}^2} \left[ \frac{2}{\mathcal{M}^2} - \left(\frac{v_{circ}}{v_h}\right)^2 - \mathcal{S}_v \right]$$

where $\mathcal{M} = v_h/c_s$ is the Mach number and $\mathcal{S}_v$ accounts for mass/momentum sources.

### 2. Cold Cloud Phase

Cold clouds are embedded within the hot wind:

- **Temperature**: $T_c \sim 10^4$ K (photoionization equilibrium)
- **Density**: $\rho_c \sim 10^{-24} - 10^{-22}$ g/cm³
- **Mass distribution**: Power-law $dN/dM \propto M^{-\alpha}$
- **Velocity**: Lags behind hot wind due to drag

Cloud dynamics include:
- **Drag acceleration**: $a_{drag} = -C_d \rho_h v_{rel}^2 / (2 \rho_c R_c)$
- **Radiative cooling**: Maintains $T_c \approx 10^4$ K
- **Mass exchange**: Through turbulent mixing layers

### 3. Turbulent Radiative Mixing Layer (TRML)

The TRML mediates interactions between phases:

**Turbulent velocity**:
$$v_{turb} = f_{turb} \cdot |v_h - v_c|$$

**Mass exchange rate**:
$$\dot{M}_{mix} = 4\pi R_c^2 \rho_{mix} v_{turb}$$

**Energy dissipation**:
$$\dot{E}_{diss} = \frac{1}{2} \dot{M}_{mix} v_{rel}^2$$

The mixing layer:
- Transfers mass from hot → cold (cooling)
- Transfers mass from cold → hot (shredding)
- Dissipates kinetic energy
- Provides momentum coupling

## Energy and Mass Sources

### Supernova Energy Injection

Energy injection rate at the base:
$$\dot{E}_{SN} = \eta_E \times 10^{51} \text{ erg} \times \text{SFR} / (100 \text{ yr})$$

where $\eta_E$ is the energy loading efficiency.

### Mass Loading

**Hot phase**:
$$\dot{M}_{hot} = \eta_M \times \text{SFR}$$

**Cold phase**:
$$\dot{M}_{cold} = \eta_{M,cold} \times \text{SFR}$$

The cold mass is distributed across cloud species following the power-law distribution.

## Sonic Point Treatment

The wind transitions from subsonic to supersonic at the sonic point where $\mathcal{M} = 1$. This creates a numerical singularity in the ODEs.

### Regularization Strategy

Near the sonic point ($|1 - \mathcal{M}^2| < \epsilon$), we use Taylor expansion:

$$\frac{1}{1 - 1/\mathcal{M}^2} \approx \frac{\mathcal{M}^2}{\mathcal{M}^2 - 1} \approx \frac{1}{\epsilon} + \mathcal{O}(\epsilon)$$

This prevents division by zero while maintaining physical accuracy.

## Cooling Physics

Radiative cooling uses tabulated rates from **Wiersma+09**:

**Cooling time**:
$$t_{cool} = \frac{3 k_B T}{2 n \Lambda(T, Z)}$$

where $\Lambda(T, Z)$ is the metallicity-dependent cooling function.

**Key features**:
- Solar metallicity tables (Z = 0.02)
- Temperature range: $10^4 - 10^9$ K
- Includes metal lines, free-free, Compton cooling
- Photoionization equilibrium below $10^4$ K

## Numerical Methods

### ODE Integration

The system uses `scipy.solve_ivp` with:
- **Method**: LSODA (adaptive switching between stiff/non-stiff)
- **Tolerances**: rtol=$10^{-8}$, atol=$10^{-10}$ (default)
- **Event detection**: For termination conditions

### Event Detection

The integration terminates when:
1. **Supersonic → Subsonic**: Wind decelerates below sound speed
2. **Negative velocity**: Any component reverses
3. **Frozen clouds**: All clouds reach terminal velocity
4. **Maximum radius**: Reaches r_max (default 100 kpc)

## Key Assumptions

1. **Spherical symmetry**: 1D radial evolution
2. **Steady state**: Time-independent at each radius
3. **Pressure equilibrium**: Between cloud and ambient gas
4. **Photoionization**: Maintains $T_c \approx 10^4$ K
5. **Power-law clouds**: $dN/dM \propto M^{-\alpha}$

## Limitations

- No magnetic fields
- No radiation pressure
- Simplified turbulence model
- Single metallicity
- No cloud-cloud collisions
- No thermal conduction

## Next Steps

- [Detailed Equations](equations.md) - Full mathematical framework
- [TRML Physics](trml.md) - Mixing layer details
- [Cooling](cooling.md) - Cooling function implementation
