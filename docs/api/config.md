# WindConfig API Reference

## Overview

`WindConfig` manages all configuration parameters for wind simulations, providing defaults, validation, and parameter organization.

## Class Definition

```python
class WindConfig:
    """
    Configuration class for wind model parameters.
    
    All parameters can be set via constructor or modified after creation.
    Validation ensures physically reasonable values.
    """
```

## Constructor Parameters

### Cloud Distribution
- `N_cloud_species` (int): Number of cloud mass bins. Default: 10
- `M_cloud_min` (float): Minimum cloud mass [Msun]. Default: 10
- `M_cloud_max` (float): Maximum cloud mass [Msun]. Default: 1e6
- `cloud_alpha` (float): Power-law slope dN/dM ∝ M^(-α). Default: 2.0

### TRML Parameters
- `f_turb0` (float): Turbulent velocity fraction. Default: 0.1
- `drag_coeff` (float): Cloud drag coefficient. Default: 0.475
- `T_cl` (float): Cloud temperature [K]. Default: 1e4
- `geometric_factor` (float): Area enhancement factor. Default: 1.0

### Physical Parameters
- `mu` (float): Mean molecular weight. Default: 0.62
- `gamma` (float): Adiabatic index. Default: 5/3
- `metallicity` (float): Solar units. Default: 1.0
- `Omwind` (float): Wind solid angle [sr]. Default: 4π

### Numerical Parameters
- `sonic_point_tolerance` (float): Regularization width. Default: 0.01
- `v_cloud_min` (float): Minimum cloud velocity [km/s]. Default: 10
- `cloud_removal_threshold` (float): Mass fraction for removal. Default: 0.01

### Chi Power Laws
- `CoolingAreaChiPower` (float): Area scaling with χ. Default: 0.5
- `ColdTurbulenceChiPower` (float): Cold turbulence scaling. Default: 0.5
- `TurbulentVelocityChiPower` (float): Velocity scaling. Default: 0.0

## Methods

### `validate()`

Validate all parameters are physically reasonable.

```python
config = WindConfig(mu=-1)  # Invalid!
config.validate()  # Raises ValueError
```

Checks include:
- Positive physical quantities (mu, gamma, temperatures)
- Reasonable ranges (0 < f_turb0 < 1, gamma > 1)
- Consistent cloud distribution (M_min < M_max, alpha > 0)

### `set_defaults()`

Apply default values to any unset parameters.

```python
config = WindConfig()
config.f_turb0 = None
config.set_defaults()  # Sets f_turb0 = 0.1
```

### `to_dict()`

Export configuration as dictionary.

```python
config = WindConfig(f_turb0=0.2)
params = config.to_dict()
print(params['f_turb0'])  # 0.2
```

### `copy()`

Create deep copy of configuration.

```python
config1 = WindConfig(f_turb0=0.1)
config2 = config1.copy()
config2.f_turb0 = 0.2  # Doesn't affect config1
```

## Usage Examples

### Basic Configuration

```python
from multiphasegalacticwind import WindConfig

# Default configuration
config = WindConfig()

# Custom TRML parameters
config = WindConfig(
    f_turb0=0.2,        # More turbulent
    drag_coeff=0.3,     # Less drag
    T_cl=5000           # Cooler clouds
)
```

### Cloud Distribution Setup

```python
# Fine mass resolution
config = WindConfig(
    N_cloud_species=20,
    M_cloud_min=1,
    M_cloud_max=1e7,
    cloud_alpha=1.8  # Shallower distribution
)

# Get cloud masses
M_clouds = np.logspace(
    np.log10(config.M_cloud_min),
    np.log10(config.M_cloud_max),
    config.N_cloud_species
)
```

### Parameter Study

```python
# Base configuration
base_config = WindConfig()

# Vary turbulence
f_turb_values = [0.05, 0.1, 0.2, 0.4]
configs = []

for f_turb in f_turb_values:
    config = base_config.copy()
    config.f_turb0 = f_turb
    config.validate()
    configs.append(config)
```

### MCMC Fitting Configuration

```python
# Optimized for speed
config = WindConfig(
    N_cloud_species=5,     # Fewer species
    sonic_point_tolerance=0.1,  # Wider regularization
)

# Use with WindModel
model = WindModel(
    config=config,
    SFR=10.0,
    rtol=1e-6,  # Relaxed tolerance
    atol=1e-8
)
```

## Parameter Relationships

### Chi (χ) Parameter
The density ratio χ = ρ_cloud / ρ_hot appears in scaling laws:

- Mixing area: `A_mix = A_geom * χ^CoolingAreaChiPower`
- Turbulent velocity: `v_turb = f_turb0 * v_rel * χ^TurbulentVelocityChiPower`
- Cold turbulence: `v_turb_cold = v_turb * χ^ColdTurbulenceChiPower`

### Cloud Mass Distribution
Power-law distribution with N species:

```python
dN/dM ∝ M^(-alpha)
M_i = M_min * (M_max/M_min)^((i-1)/(N-1))
```

## Validation Rules

The `validate()` method enforces:

1. **Physical constraints**:
   - `mu > 0` (positive molecular weight)
   - `gamma > 1` (physical adiabatic index)
   - `T_cl > 0` (positive temperature)

2. **Fraction constraints**:
   - `0 < f_turb0 < 1` (turbulent fraction)
   - `0 < drag_coeff < 10` (reasonable drag)

3. **Cloud constraints**:
   - `M_cloud_min > 0`
   - `M_cloud_max > M_cloud_min`
   - `cloud_alpha > 0`
   - `N_cloud_species >= 1`

4. **Numerical constraints**:
   - `sonic_point_tolerance > 0`
   - `v_cloud_min >= 0`

## Default Values

| Parameter | Default | Units | Description |
|-----------|---------|-------|-------------|
| N_cloud_species | 10 | - | Number of cloud bins |
| M_cloud_min | 10 | Msun | Minimum cloud mass |
| M_cloud_max | 1e6 | Msun | Maximum cloud mass |
| cloud_alpha | 2.0 | - | Power-law slope |
| f_turb0 | 0.1 | - | Turbulent fraction |
| drag_coeff | 0.475 | - | Drag coefficient |
| T_cl | 1e4 | K | Cloud temperature |
| mu | 0.62 | - | Mean molecular weight |
| gamma | 5/3 | - | Adiabatic index |

## See Also

- [WindModel](wind_model.md) - Main model class
- [Parameter Guide](../guide/parameters.md) - Parameter selection advice
- [Examples](../getting_started/examples.md) - Usage examples