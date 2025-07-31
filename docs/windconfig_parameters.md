# WindConfig Parameter Documentation

Complete documentation of all configurable parameters in the `WindConfig` class.

## Overview

The `WindConfig` class manages all physics parameters for the multiphase galactic wind model. Parameters can be set when creating a `WindModel` instance.

```python
from multiphasegalacticwind import WindModel, WindConfig

# Method 1: Create a config object
config = WindConfig(f_turb0=0.2, metallicity=2.0)
model = WindModel(SFR=10.0, config=config)

# Method 2: Pass parameters directly
model = WindModel(SFR=10.0, f_turb0=0.2, metallicity=2.0)
```

## Parameter Categories

### Thermodynamic Parameters

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `mu` | 0.62 | dimensionless | 0.5-1.3 | Mean molecular weight. Lower for ionized gas (~0.62), higher for neutral (~1.3) |
| `gamma` | 5/3 | dimensionless | 1.1-5/3 | Adiabatic index. 5/3 for monoatomic gas |

### Metallicity and Environment

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `metallicity` | 10^-0.5 | solar units | 0.01-3.0 | Gas metallicity relative to solar. Affects cooling rates |
| `redshift` | 0.0 | dimensionless | 0-10 | Redshift. Affects cooling function and CMB temperature |

### Turbulence and Mixing

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `f_turb0` | 0.1 | dimensionless | 0.05-0.3 | Turbulent mixing efficiency. Key parameter controlling cloud evolution |
| `drag_coeff` | 0.5 | dimensionless | 0.1-2.0 | Cloud drag coefficient. ~0.5 for spheres, higher for complex shapes |
| `Mdot_coefficient` | 1/3 | dimensionless | 0.1-1.0 | Mass transfer efficiency in mixing layer |
| `geometric_factor` | 1.0 | dimensionless | 1-4 | Cloud geometry factor. 1=sphere, >1 for elongated clouds |

### Cooling Parameters

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `cooling_enabled` | True | boolean | True/False | Enable/disable radiative cooling (set via Cooling_Factor) |
| `Cooling_Factor` | 1.0 | dimensionless | 0-2 | Cooling strength multiplier. 0=off, 1=standard, >1=enhanced |

### Cloud Evolution

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `M_cloud_min` | 0.01 Msun | solar masses | 0.001-10 | Minimum cloud mass. Clouds below this are removed |
| `Omwind` | 4π | steradians | 0.1-4π | Wind solid angle. 4π for spherical, less for biconical |

### Power Law Indices

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `TurbulentVelocityChiPower` | 0.0 | dimensionless | -0.5-0.5 | Controls v_turb ∝ χ^α where χ = ρ_cloud/ρ_hot |
| `CoolingAreaChiPower` | 0.5 | dimensionless | 0-2 | Controls A_cool ∝ χ^β for mixing layer area |
| `ColdTurbulenceChiPower` | -0.5 | dimensionless | -1-0 | Controls v_turb_cold ∝ χ^γ |

### Cloud Injection

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `cold_cloud_injection_radial_extent` | 1.5 r_star | cm | r_star-10r_star | Radius within which clouds are injected |
| `cold_cloud_injection_radial_power` | 6 | dimensionless | 0-10 | Power law for cloud injection rate vs radius |

### Supernova Feedback

| Parameter | Default | Units | Range | Description |
|-----------|---------|-------|-------|-------------|
| `E_SN` | 1e51 | erg | 1e50-1e52 | Energy per supernova |
| `mstar` | 100 | Msun | 50-200 | Stellar mass formed per supernova |
| `epsilon` | 1e-5 | dimensionless | 1e-6-1e-3 | Sonic point Mach = 1 + epsilon |

## Common Use Cases

### High Metallicity Wind
```python
config = WindConfig(metallicity=2.0)
```

### Enhanced Turbulent Mixing
```python
config = WindConfig(f_turb0=0.2, TurbulentVelocityChiPower=0.25)
```

### No Cooling Case
```python
config = WindConfig(Cooling_Factor=0.0)
```

### High Redshift Universe
```python
config = WindConfig(redshift=3.0, metallicity=0.1)
```

### Different Cloud Geometry
```python
config = WindConfig(geometric_factor=2.0, drag_coeff=1.0)
```

### Confined Wind (Biconical)
```python
config = WindConfig(Omwind=2*np.pi)  # Half sphere
```

## Physics Notes

### Key Relationships

1. **Density Contrast**: χ = ρ_cloud/ρ_hot controls many processes
2. **Turbulent Velocity**: v_turb = f_turb0 × v_rel × χ^TurbulentVelocityChiPower
3. **Cooling Area**: A = geometric_factor × χ^CoolingAreaChiPower × A_geometric
4. **Mass Transfer**: Ṁ ∝ Mdot_coefficient × v_turb × A / r_cloud

### Parameter Sensitivities

- **f_turb0**: Most important parameter for cloud evolution
- **metallicity**: Strongly affects cooling rates and wind fate
- **drag_coeff**: Controls cloud deceleration
- **geometric_factor**: Affects cloud surface area for mixing

### Typical Values from Simulations

- f_turb0 ≈ 0.1 (from turbulent box simulations)
- drag_coeff ≈ 0.5-1.0 (depends on cloud shape)
- geometric_factor ≈ 1-2 (clouds become elongated)
- Mdot_coefficient ≈ 0.3-0.6 (from detailed TRML simulations)