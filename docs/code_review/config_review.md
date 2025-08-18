# Config Module - Comprehensive Code Review

## Module Overview
**File**: `multiphasegalacticwind/config.py`  
**Lines**: 133  
**Purpose**: Central configuration for all physics parameters used in the multiphase wind model

## Parameter Documentation

### 1. Gas Properties (Lines 36-41)

#### `mu` (Line 36)
- **Default**: 0.62
- **Units**: dimensionless
- **Physics**: Mean molecular weight of gas
- **Usage**: Converting between mass density and number density: n = ρ/(μ·mp)
- **Typical Range**: 0.59-0.62 for solar metallicity gas

#### `Z_hot_over_Z_solar` / `metallicity` (Lines 38-39)
- **Default**: 10^(-0.5) ≈ 0.316
- **Units**: dimensionless (relative to solar)
- **Physics**: Hot gas metallicity for cooling function
- **Usage**: Cooling rate calculation Λ(T, Z)
- **Note**: Dual naming for backward compatibility

#### `redshift` (Line 40)
- **Default**: 0.0
- **Units**: dimensionless
- **Physics**: Cosmological redshift
- **Usage**: Cooling function interpolation for different epochs

### 2. Wind Geometry (Lines 43-44)

#### `half_opening_angle` (Line 43)
- **Default**: π/2 radians (90°)
- **Units**: radians
- **Physics**: Half-angle of bi-conical wind
- **Usage**: Determines solid angle Ω = 4π(1 - cos(θ))

#### `Omwind` (Line 44)
- **Default**: 4π (full sphere for θ=π/2)
- **Units**: steradians
- **Physics**: Solid angle of wind
- **Derived**: Calculated from half_opening_angle
- **Note**: Used in mass flux calculations

### 3. Cloud Destruction (Line 47)

#### `M_cloud_min` (Line 47)
- **Default**: 0.01 M⊙ (≈ 2×10^28 g)
- **Units**: grams
- **Physics**: Minimum cloud mass before destruction
- **Usage**: Event trigger for cloud removal
- **Physical Meaning**: Clouds below this mass are assumed destroyed

### 4. TRML Parameters (Lines 50-57)

#### `CoolingAreaChiPower` (Line 50)
- **Default**: 0.5
- **Units**: dimensionless
- **Physics**: Exponent β_cool in A_boost = g_geom · χ^β_cool
- **Usage**: Scales effective cooling area with density contrast
- **Physical Meaning**: How mixing layer area scales with χ

#### `ColdTurbulenceChiPower` (Line 51)
- **Default**: -0.5
- **Units**: dimensionless
- **Physics**: Exponent β_cold in v_turb,cold = v_turb · χ^β_cold
- **Usage**: Scales cold cloud ablation velocity
- **Physical Meaning**: Reduced turbulence in denser medium

#### `TurbulentVelocityChiPower` (Line 52)
- **Default**: 0.0
- **Units**: dimensionless
- **Physics**: Exponent β_turb in v_turb = f_turb · v_rel · χ^β_turb
- **Usage**: Scales turbulent velocity with density contrast
- **Physical Meaning**: No χ-dependence by default

#### `geometric_factor` (Line 53)
- **Default**: 1.0
- **Units**: dimensionless
- **Physics**: Factor g_geom in cooling area boost
- **Usage**: A_boost = g_geom · χ^β_cool
- **Physical Meaning**: Geometry correction for non-spherical clouds

#### `Mdot_coefficient` (Line 54)
- **Default**: 1/3
- **Units**: dimensionless
- **Physics**: Coefficient C_dot in mass transfer equations
- **Usage**: Scales all TRML mass exchange rates
- **Physical Meaning**: Efficiency of turbulent mixing

#### `Cooling_Factor` (Line 55)
- **Default**: 1.0
- **Units**: dimensionless
- **Physics**: Global cooling rate multiplier
- **Usage**: Scales Λ(T, Z) cooling function
- **Note**: For testing cooling strength effects

#### `drag_coeff` (Line 56)
- **Default**: 0.5
- **Units**: dimensionless
- **Physics**: Drag coefficient C_drag
- **Usage**: Ram pressure F_drag = 0.5·C_drag·ρ·π·r²·v_rel²
- **Typical Range**: 0.3-1.0

#### `f_turb0` (Line 57)
- **Default**: 0.1
- **Units**: dimensionless
- **Physics**: Turbulent velocity fraction
- **Usage**: v_turb = f_turb0 · v_rel
- **Typical Range**: 0.05-0.2
- **Physical Meaning**: Fraction of relative velocity in turbulent motions

### 5. Cloud Injection Parameters (Lines 60-66)

#### `cold_cloud_injection_radial_power` (Line 60)
- **Default**: 6
- **Units**: dimensionless
- **Physics**: Power law exponent p for injection profile
- **Usage**: dN/dr ∝ (r/r_inj)^p for r < r_inj
- **Physical Meaning**: Steep increase in cloud density near galaxy

#### `cold_cloud_injection_radial_extent_frac` (Line 61)
- **Default**: 1.33
- **Units**: dimensionless (fraction of r0)
- **Physics**: Injection radius as fraction of sonic radius
- **Usage**: r_inj = cold_cloud_injection_radial_extent_frac × r0
- **Note**: Recently changed from absolute distance

#### `v_cloud_init` (Line 62)
- **Default**: 100.0 km/s
- **Units**: km/s
- **Physics**: Initial cloud velocity
- **Usage**: Starting velocity for all cloud species
- **Typical Range**: 3-200 km/s

#### `v_cloud_min` (Line 63)
- **Default**: 1.0 km/s
- **Units**: km/s
- **Physics**: Minimum cloud velocity threshold
- **Usage**: Event trigger - terminates if v_cloud < v_cloud_min
- **Physical Meaning**: Clouds essentially stopped

#### `cloud_radial_offset` (Line 64)
- **Default**: 0.01
- **Units**: dimensionless (fraction of r0)
- **Physics**: Starting radius offset from sonic point
- **Usage**: r_start = r0 × (1 + cloud_radial_offset)
- **Physical Meaning**: Allows hot-only evolution before cloud injection

#### `Z_cloud_over_Z_solar` (Line 65)
- **Default**: 0.3
- **Units**: dimensionless (relative to solar)
- **Physics**: Initial cloud metallicity
- **Usage**: Sets Z_cloud for all species initially
- **Physical Meaning**: Sub-solar metallicity typical of star-forming gas

#### `T_cl` (Line 66)
- **Default**: 10^4 K
- **Units**: Kelvin
- **Physics**: Cloud temperature
- **Usage**: Sets density contrast χ = T_hot/T_cloud
- **Physical Meaning**: Cool photoionized gas temperature

### 6. Supernova Feedback (Lines 69-70)

#### `E_SN` (Line 69)
- **Default**: 10^51 erg
- **Units**: erg
- **Physics**: Energy per supernova
- **Usage**: Energy injection Ė = η_E × (E_SN/m_*) × SFR
- **Physical Meaning**: Canonical SN explosion energy

#### `mstar` (Line 70)
- **Default**: 100 M⊙
- **Units**: solar masses
- **Physics**: Stellar mass per supernova
- **Usage**: SN rate = SFR/m_*
- **Physical Meaning**: IMF-averaged mass per SN

### 7. Numerical Parameters (Lines 73-74)

#### `sonic_point_offset` (Line 73)
- **Default**: 10^(-6)
- **Units**: dimensionless
- **Physics**: Offset from exact Mach = 1
- **Usage**: Initial conditions at Mach = 1 + ε
- **Note**: Avoids singularity at sonic point
- **Critical**: Must be small but non-zero

#### `sonic_transition_tolerance` (Line 74)
- **Default**: 0.01
- **Units**: dimensionless
- **Physics**: Tolerance for sonic transition events
- **Usage**: Triggers event when |Mach - 1| < tolerance
- **Note**: Different purpose than sonic_point_offset

## Class Methods Analysis

### `__init__` (Lines 24-75)
- Uses `kwargs.get()` with defaults for all parameters
- Handles backward compatibility (metallicity → Z_hot_over_Z_solar)
- Calculates derived quantity Omwind from half_opening_angle
- **Issue**: No parameter validation

### `to_dict()` (Lines 76-105)
- Returns all parameters as dictionary
- Maintains backward compatibility keys
- Used to pass config to physics functions
- **Good**: Clean interface for parameter passing

### `set_defaults()` (Lines 107-124)
- Class method to change defaults globally
- **Issue**: Implementation incomplete - references non-existent `_default_*` attributes
- **Bug**: Will always raise ValueError since no `_default_*` attributes exist

## Issues Identified

### 1. **Critical Bug in set_defaults()**
```python
# Line 121-122: References non-existent attributes
if hasattr(cls, f'_default_{key}'):
    setattr(cls, f'_default_{key}', value)
```
This will never work as no `_default_*` attributes are defined.

### 2. **No Parameter Validation**
- No range checking (e.g., 0 < f_turb0 < 1)
- No type checking
- No physical consistency checks

### 3. **Unit Inconsistencies**
- M_cloud_min uses Msun constant but stores in grams
- Mix of CGS and convenient units

### 4. **Documentation Gaps**
- Missing units in docstrings
- No typical ranges documented
- Physical meaning not always clear

### 5. **Backward Compatibility Complexity**
- Multiple names for same parameter (metallicity/Z_hot_over_Z_solar)
- Increases maintenance burden

## Recommended Improvements

### Immediate Fixes

1. **Fix set_defaults() method**:
```python
_defaults = {...}  # Dictionary of defaults

@classmethod
def set_defaults(cls, **kwargs):
    for key in kwargs:
        if key in cls._defaults:
            cls._defaults[key] = kwargs[key]
        else:
            raise ValueError(f"Unknown parameter: {key}")
```

2. **Add parameter validation**:
```python
def validate(self):
    if not 0 < self.f_turb0 < 1:
        raise ValueError("f_turb0 must be between 0 and 1")
    if self.mu <= 0:
        raise ValueError("mu must be positive")
    # etc.
```

3. **Add comprehensive docstrings**:
```python
"""
mu : float, default=0.62
    Mean molecular weight of gas (dimensionless).
    Used to convert between mass and number density.
    Typical range: 0.59-0.62 for solar metallicity.
"""
```

### Future Enhancements

1. **Parameter Groups**:
```python
@dataclass
class TRMLParameters:
    f_turb0: float = 0.1
    drag_coeff: float = 0.5
    # etc.
```

2. **Unit Management**:
```python
from astropy import units as u
mu = 0.62 * u.dimensionless
T_cl = 1e4 * u.K
```

3. **Presets**:
```python
@classmethod
def milky_way_preset(cls):
    return cls(f_turb0=0.1, drag_coeff=0.5, ...)

@classmethod
def starburst_preset(cls):
    return cls(f_turb0=0.2, drag_coeff=0.3, ...)
```

## Physical Parameter Relationships

### Key Dependencies
1. **χ = T_hot/T_cl**: Density contrast drives all TRML physics
2. **Ė/Ṁ**: Determines sonic point velocity
3. **f_turb0 × v_rel**: Sets turbulent mixing efficiency
4. **E_SN/m_star**: Energy per unit SFR

### Sensitive Parameters
1. **f_turb0**: Directly controls cloud growth/loss rates
2. **sonic_point_offset**: Must be tiny to avoid errors
3. **drag_coeff**: Controls momentum transfer
4. **T_cl**: Sets density contrast χ

## Testing Requirements

### Unit Tests Needed
1. `test_init_with_defaults`: Verify default values
2. `test_init_with_overrides`: Check parameter overriding
3. `test_to_dict`: Verify dictionary conversion
4. `test_validation`: Check parameter ranges
5. `test_backward_compatibility`: Verify old parameter names work

### Integration Tests
1. Test with wind_model.py
2. Verify parameters flow to core_physics.py
3. Check derived quantities (Omwind)

## Summary

The config module provides centralized parameter management but needs:
1. **Bug fix**: set_defaults() implementation
2. **Validation**: Add parameter range checking
3. **Documentation**: Complete units and ranges
4. **Simplification**: Remove backward compatibility complexity
5. **Organization**: Consider parameter grouping

Total of 24 physics parameters controlling:
- Gas properties (3)
- Geometry (2)
- TRML physics (8)
- Cloud injection (7)
- Supernova feedback (2)
- Numerical controls (2)

The module successfully centralizes configuration but lacks validation and has a critical bug in set_defaults().