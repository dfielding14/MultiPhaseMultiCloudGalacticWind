# WindConfig Parameter Documentation

Complete documentation for parameters accepted by `WindConfig` in the current codebase.

## Overview

`WindConfig` controls model physics and event thresholds. `WindModel` controls galaxy/wind launch inputs and solver settings.
Parameter names are case-sensitive (for example `Cooling_Factor`, not `cooling_factor`).

```python
from multiphasegalacticwind import WindModel, WindConfig

# Explicit config object
config = WindConfig(f_turb0=0.2, Z_hot_over_Z_solar=1.0)
model = WindModel(SFR=10.0, eta_M=0.2, eta_M_cold=0.1, config=config)

# Or pass config fields directly through WindModel
model = WindModel(SFR=10.0, f_turb0=0.2, Z_hot_over_Z_solar=1.0)
```

## WindConfig Parameters

### Gas and environment

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| `mu` | `0.62` | dimensionless | Mean molecular weight used by thermodynamic closures |
| `Z_hot_over_Z_solar` | `10**-0.5` | solar units | Hot-phase metallicity used in cooling and initialization |
| `metallicity` | alias of `Z_hot_over_Z_solar` | solar units | Backward-compatible alias |
| `redshift` | `0.0` | dimensionless | Cooling-table redshift selector |

### Wind geometry and cloud floor

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| `half_opening_angle` | `pi/2` | radians | Sets `Omwind = 4*pi*(1-cos(half_opening_angle))` |
| `M_cloud_min` | `1e-2 * Msun` | g | Cloud mass floor for active cloud evolution |

### TRML / coupling parameters

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| `f_turb0` | `0.1` | dimensionless | Turbulent mixing efficiency |
| `drag_coeff` | `0.5` | dimensionless | Ram-drag coefficient |
| `Mdot_coefficient` | `1/3` | dimensionless | Mass-transfer prefactor |
| `geometric_factor` | `1.0` | dimensionless | Surface-area boost factor |
| `Cooling_Factor` | `1.0` | dimensionless | Global multiplier on cooling losses |
| `CoolingAreaChiPower` | `0.5` | dimensionless | Exponent in cooling-area scaling with `chi` |
| `ColdTurbulenceChiPower` | `-0.5` | dimensionless | Exponent for cold-phase turbulent velocity scaling |
| `TurbulentVelocityChiPower` | `0.0` | dimensionless | Exponent for turbulent velocity scaling |

### Cloud injection and cloud properties

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| `cold_cloud_injection_radial_power` | `6` | dimensionless | Power-law index inside injection region |
| `cold_cloud_injection_radial_extent_frac` | `1.33` | dimensionless | Injection radius as a fraction of `r_star` |
| `v_cloud_init` | `100.0` | km/s | Initial cloud velocity at launch |
| `v_cloud_min` | `1.0` | km/s | Velocity floor event threshold |
| `cloud_radial_offset` | `0.01` | dimensionless | Integrate hot-only to `(1+offset)*r_star` before cloud launch |
| `Z_cloud_over_Z_solar` | `0.3` | solar units | Initial cloud metallicity |
| `T_cl` | `1e4` | K | Cloud temperature |

### Feedback and sonic controls

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| `E_SN` | `1e51` | erg | Energy per supernova |
| `mstar` | `100.0` | Msun | Stellar mass formed per supernova |
| `sonic_point_offset` | `1e-6` | dimensionless | Initial Mach is set to `1 + sonic_point_offset` |
| `sonic_transition_tolerance` | `0.01` | dimensionless | Event tolerance around Mach 1 transitions |

## Parameters that belong to WindModel (not WindConfig)

The following are **not** `WindConfig` fields and should be passed to `WindModel` directly:
- `SFR`, `eta_M`, `eta_M_cold`, `eta_E`, `v_circ`, `r_star_kpc`, `r_max_kpc`
- `N_cloud_species`, `cloud_mass_range`, `cloud_alpha`
- `rtol`, `atol`, `progress_callback`, `progress_interval`

## Common Config Snippets

### Higher metallicity hot wind

```python
config = WindConfig(Z_hot_over_Z_solar=2.0)
```

### Stronger mixing and drag

```python
config = WindConfig(f_turb0=0.2, drag_coeff=1.0)
```

### Cooling disabled

```python
config = WindConfig(Cooling_Factor=0.0)
```

### Wider cloud injection region

```python
config = WindConfig(
    cold_cloud_injection_radial_extent_frac=2.0,
    cold_cloud_injection_radial_power=4,
)
```

### Offset start from sonic point

```python
config = WindConfig(cloud_radial_offset=0.1)
```

## Validation and defaults

- `WindModel` calls `config.validate()` during initialization.
- Invalid values (for example non-positive `drag_coeff`, negative `redshift`, or non-positive `sonic_point_offset`) raise `ValueError`.
- Global defaults can be overridden for a session using `WindConfig.set_defaults(...)`.

## Notes on units

- Internal ODE equations are in CGS.
- Some config fields are in user-friendly units (`v_cloud_init` in km/s, `mstar` in Msun).
- `M_cloud_min` is stored in grams; use `1e-2 * Msun` style expressions for clarity.
