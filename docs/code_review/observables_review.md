# Observables Module - Comprehensive Code Review

## Module Overview
**File**: `multiphasegalacticwind/observables.py`  
**Lines**: 401  
**Purpose**: Calculate observable quantities (column densities, velocity distributions) from wind solutions

## Key Constants (Lines 12-13)
```python
mu_cool = 1.4  # Mean atomic mass per proton for ionized gas
```
**Note**: Different from hot gas μ = 0.62, appropriate for cool photoionized gas

## Function Analysis

### 1. `calculate_cloud_density()` (Lines 16-81)
**Purpose**: Calculate cloud number density n_cloud(r)
**Formula**: 
$$n_{cloud} = \frac{\dot{N}_{cloud} \cdot f_{inj}(r)}{\Omega \cdot r^2 \cdot v_{cloud}}$$

**Key Features**:
- Handles single species or sum over all
- Applies injection profile enhancement
- Returns number density in cm⁻³

**Issues**:
- Line 43: Uses `cold_cloud_injection_radial_extent_frac` - long parameter name
- Line 77: Loop could be vectorized for performance

### 2. `calculate_velocity_moments()` (Lines 84-131)
**Purpose**: Calculate statistical moments of velocity distribution
**Calculates**:
- Raw moments: $\mu_n = \int v^n \cdot dN/dv \cdot dv$
- Mean: $\langle v \rangle = \mu_1 / \mu_0$
- Dispersion: $\sigma = \sqrt{\mu_2/\mu_0 - \langle v \rangle^2}$
- Skewness: $\gamma = \mu_3^{central} / \sigma^3$

**Good Design**:
- Generic function works with any distribution
- Handles edge cases (zero normalization)
- Returns structured dictionary

**Issue**: Line 124 - Uses `max(0, var)` to handle numerical errors

### 3. `calculate_column_density_distribution()` (Lines 134-299)
**Purpose**: Core function calculating dN/dv in column density units
**Units**: cm⁻² / (km/s)

**Algorithm**:
1. Select radius range
2. Apply injection profile
3. Calculate cloud mass density: $\rho_{cloud} = \dot{N} M f_{inj} / (\Omega r^2 v)$
4. Convert to H number density: $n_H = \rho / (\mu_{cool} m_p)$
5. Calculate velocity gradient: $dv/dr$
6. Compute: $dN/dv = n_H / (dv/dr)$

**Critical Section** (Lines 216-247):
Handles negative velocity gradients (non-monotonic velocity):
```python
if np.min(grad_v) > 0:
    # Normal case: monotonic velocity
    dN_dv_column = n_H / grad_v
else:
    # Fallback: rebinning method
    # Bin data by velocity and interpolate
```

**Issues**:
- Line 219: Uses `1e-30` to avoid division by zero - arbitrary small number
- Line 234: Arbitrary choice of `n_bins = max(10, len(v_cloud_cms) // 8)`
- Line 256: Comment says all species have same v, but they evolve independently
- Line 291: Fallback for negative gradients too simplistic

### 4. `calculate_column_density_by_species()` (Lines 302-359)
**Purpose**: Calculate dN/dv for each cloud species separately
**Returns**: Dictionary with total and per-species distributions

**Good Design**:
- Reuses `calculate_column_density_distribution()`
- Returns structured output with metadata
- Useful for understanding mass-dependent effects

**Issue**: Inefficient - recalculates total when could sum species

### 5. `calculate_mass_weighted_velocity()` (Lines 362-401)
**Purpose**: Calculate mass-weighted average velocity
**Formula**: 
$$v_{mass} = \frac{\dot{M}_{hot} v_{hot} + \dot{M}_{cold} v_{cold}}{\dot{M}_{hot} + \dot{M}_{cold}}$$

**Issues**:
- Line 390: Wrong variable - uses `v_cl` but should be `v_cl[0]` or average
- Line 395-396: Complex mass flux calculation needs verification
- Missing proper handling of multiple cloud species

## Physics Implementation

### Column Density Calculation
The key physics is the transformation from 3D density to line-of-sight column:

$$\frac{dN}{dv} = \frac{n(r)}{|dv/dr|}$$

This assumes:
1. Spherical symmetry
2. Monotonic velocity field (handled with fallback)
3. Single line of sight through wind

### Injection Profile
Enhanced cloud density near galaxy:
$$f_{inj}(r) = \begin{cases}
(r/r_{inj})^p & r < r_{inj} \\
1 & r \geq r_{inj}
\end{cases}$$

Default: p = 6 (steep enhancement)

## Critical Issues

### 1. **Multi-Species Handling**
Line 256-294: Assumes all species have same velocity at each radius
```python
# Get velocity from first species as reference
v_cloud_cms = solution.sol.y[4 + solution.model.N_cloud_species, mask]
```
But each species evolves independently with different drag!

### 2. **Negative Gradient Handling**
Lines 223-247, 287-292: Fallback methods are inconsistent
- Single species: Sophisticated rebinning
- Multi-species sum: Simple average
Should use consistent approach

### 3. **Unit Conversions**
Multiple unit conversions scattered throughout:
- Line 250: `v_cloud_cms / 1e5` → km/s
- Line 252: `dN_dv_column * 1e5` → per (km/s)
- Line 285: Combined conversion
Error-prone and hard to verify

### 4. **Mass-Weighted Velocity Bug**
Lines 390-396: Implementation appears incorrect
```python
v_cold = np.interp(r_kpc, solution.r, solution.v_cl)  # v_cl undefined!
```
Should access cloud velocities properly

## Documentation Needs

### High Priority
1. Document physical assumptions (spherical symmetry, etc.)
2. Explain column density transformation
3. Document units at each step
4. Clarify multi-species velocity handling

### Medium Priority
1. Explain rebinning algorithm for non-monotonic velocity
2. Document injection profile physics
3. Add references to observational methods

## Recommended Improvements

### Immediate Fixes

1. **Fix multi-species velocity handling**:
```python
# Each species has its own velocity evolution
for i in range(solution.model.N_cloud_species):
    v_cl_i_cms = solution.sol.y[4 + solution.model.N_cloud_species + i, mask]
    # Use v_cl_i_cms for species i calculations
```

2. **Fix mass-weighted velocity**:
```python
def calculate_mass_weighted_velocity(solution, r_eval_kpc=10.0):
    # Properly handle all cloud species
    v_cold_avg = 0
    M_cold_tot = 0
    for i in range(solution.model.N_cloud_species):
        v_cl_i = solution.v_cl[i]  # Get velocity for species i
        M_cl_i = solution.M_clouds[i]
        v_cold_avg += v_cl_i * M_cl_i
        M_cold_tot += M_cl_i
    v_cold_avg /= M_cold_tot
```

3. **Consistent gradient handling**:
```python
def handle_negative_gradient(v, n_H, r):
    """Unified method for non-monotonic velocity."""
    # Use sophisticated rebinning for all cases
    # Not just simple average
```

4. **Unit management**:
```python
class Units:
    KM_TO_CM = 1e5
    
def convert_velocity_units(v_cms):
    return v_cms / Units.KM_TO_CM
```

### Future Enhancements

1. **Vectorized operations**:
```python
# Vectorize species loop
cloud_densities = (Ndot_cloud[:, np.newaxis] * M_clouds * injection_function /
                  (Omwind * r**2 * v_clouds))
dN_dv_total = np.sum(cloud_densities / grad_v, axis=0)
```

2. **Path length corrections**:
```python
def calculate_path_length_correction(r, impact_parameter):
    """Account for different path lengths through wind."""
    # Implement geometric correction
```

3. **Mock observations**:
```python
def create_mock_spectrum(solution, instrument='COS', SNR=10):
    """Generate mock absorption spectrum."""
    # Convolve with instrument response
    # Add noise
```

## Testing Requirements

### Unit Tests
1. `test_cloud_density`: Verify number conservation
2. `test_velocity_moments`: Check moment calculations
3. `test_column_density_units`: Verify unit conversions
4. `test_injection_profile`: Check enhancement function
5. `test_negative_gradient`: Test fallback methods

### Integration Tests
1. Compare with analytic solutions (single cloud)
2. Test conservation (integrate dN/dv)
3. Compare different radius ranges
4. Verify species summation

### Physics Tests
1. Column density should decrease with radius
2. Moments should be positive
3. Total column should equal sum of species

## Performance Analysis

### Bottlenecks
1. Loop over species (not vectorized)
2. Repeated interpolation calls
3. Gradient calculation for each species

### Optimization Opportunities
1. Vectorize species calculations
2. Cache interpolation results
3. Use numpy broadcasting

## Code Quality Metrics

- **Documentation**: ~40% (needs physics explanation)
- **Type Hints**: Minimal (~10%)
- **Error Handling**: Poor (no validation)
- **Test Coverage**: Unknown
- **Complexity**: Medium (longest function ~165 lines)

## Summary

The observables module provides essential functionality for comparing with observations but needs:

1. **Bug fixes**: Multi-species velocity handling, mass-weighted velocity
2. **Consistency**: Unified gradient handling method
3. **Documentation**: Physical assumptions and methods
4. **Performance**: Vectorization of species loops
5. **Testing**: Comprehensive validation suite

The module successfully implements:
- Column density calculations with proper units
- Velocity distribution moments
- Per-species decomposition
- Injection profile effects

Critical bugs in multi-species handling need immediate attention.