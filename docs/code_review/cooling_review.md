# Cooling Module - Comprehensive Code Review

## Module Overview
**File**: `multiphasegalacticwind/cooling.py`  
**Lines**: 327  
**Purpose**: Handles cooling function calculations using Wiersma+09 cooling tables

## Architecture

### Global State Management (Lines 19-36)
```python
# Cooling table data
_Lambda = None          # Main 4D interpolator
_Lambda_tab = None      # Raw table data
_redshifts = None       # Available redshifts
_Zs = None             # Available metallicities
_log_Tbins = None      # Temperature bins
_log_nHbins = None     # Density bins

# Derived interpolators
_Lambda_P_rho = None           # (P,ρ) interpolator
_Lambda_P_rho_params = None    # Parameters for cache check

# Caching
_tcool_cache = {}              # Cache for tcool_P calls
_tcool_cache_size = 0
_MAX_CACHE_SIZE = 10000
```

**Design Pattern**: Lazy loading with global state
- Tables load on first use, not at import
- Cached after loading for reuse
- **Issue**: Global state makes testing difficult

## Function Analysis

### 1. `get_lambda_interpolator()` (Lines 39-53)
**Purpose**: Get main 4D cooling interpolator
**Pattern**: Lazy loading singleton
```python
if _Lambda is None:
    load_cooling_table()
return _Lambda
```
**Good**: Simple, efficient lazy loading

### 2. `load_cooling_table()` (Lines 56-99)
**Purpose**: Load cooling table from disk
**Key Features**:
- Loads from `data/Lambda_tab_redshifts.npz`
- Creates 4D interpolator: Λ(log nH, log T, Z, z)
- Uses `bounds_error=False, fill_value=1e-30`

**Issues**:
- Line 91: Hard-coded fill value `1e-30` - should be configurable
- Line 94-96: Error message could be more helpful
- No validation of loaded data shape/contents

### 3. `get_cooling_interpolator()` (Lines 102-165)
**Purpose**: Create (P,ρ) interpolator for specific parameters
**Algorithm**:
1. Check if cached interpolator matches parameters
2. If not, generate new table:
   - Create P,ρ grids (100×101 points)
   - Calculate T = P·(μ·mp/ρ)
   - Apply bounds to T and ρ
   - Interpolate from main table

**Critical Code** (Lines 144-151):
```python
if rho > 1 * muH * mp:
    rho = 1. * muH * mp     # Max density ~1 cm⁻³
elif rho < 1e-8 * muH * mp:
    rho = 1e-8 * muH * mp    # Min density ~10⁻⁸ cm⁻³
if T > 10**8.98:
    T = 10**8.98             # Max temp ~10⁹ K
elif T < 1e2:
    T = 1e2                  # Min temp 100 K
```

**Issues**:
- Line 153: Bare except clause - should specify exception
- Line 155: Hard-coded fill value `1e-30`
- Performance: Generates 10,100 interpolation calls every time parameters change

### 4. `tcool_P()` (Lines 168-237)
**Purpose**: Calculate cooling time t_cool = 1.5kT/(nΛ)
**Key Features**:
- Smart caching for scalar inputs
- Bounds checking on T and nH
- Formula: `tcool = 1.5 * (muH/mu) * kb * T / (nH_actual * lambda_val)`

**Caching Strategy** (Lines 196-209):
```python
cache_key = (
    round(float(T), 2),      # 0.01 K precision
    round(float(P), 24),     # High precision for P
    round(float(metallicity), 3),
    round(float(redshift), 3),
    round(float(mu), 3)
)
```

**Issues**:
- Line 201: Pressure rounded to 24 decimal places - excessive precision
- Line 210: Bare except for cache key creation
- Line 225: Could be vectorized better

### 5. `Lambda_P()` (Lines 240-271)
**Purpose**: Get cooling function value at (T,P)
**Implementation**:
- Calculates nH from ideal gas law
- Bounds nH to [0, 0.9]
- Calls main interpolator
- Vectorized with `np.vectorize`

**Issue**: Line 271 overwrites function with vectorized version - confusing

### 6. `get_tcool_min_interpolators()` (Lines 274-327)
**Purpose**: Find minimum cooling time and corresponding temperature
**Algorithm**:
1. Create P,Z grids
2. For each (P,Z), find T that minimizes tcool
3. Build interpolators for T_min(P,Z) and tcool_min(P,Z)

**Issues**:
- Line 302: Skips high pressure (P > 10^4.2) without explanation
- Line 299: Hard-coded mu = 0.62
- Line 318: Extrapolation formula `P^-1` - needs physics justification

## Physics Implementation

### Cooling Function
From Wiersma et al. (2009):
- Includes metal-line cooling
- Photoionization equilibrium
- Redshift-dependent UV background

### Key Equations

**Cooling time**:
$$t_{cool} = \frac{3kT/2}{n_H \Lambda(T, n_H, Z, z)}$$

**Ideal gas relation**:
$$n_H = \frac{P}{kT} \cdot \frac{\mu}{\mu_H}$$

**Energy loss rate**:
$$\dot{e}_{cool} = -n_H^2 \Lambda(T, n_H, Z, z)$$

## Performance Analysis

### Bottlenecks
1. **Table Generation** (get_cooling_interpolator):
   - 10,100 interpolation calls
   - Happens every time parameters change
   - No parallelization

2. **Caching**:
   - Only works for scalar inputs
   - Cache key generation try/except overhead
   - Fixed cache size limit

### Optimization Opportunities
1. Vectorize table generation loops
2. Use numpy broadcasting instead of nested loops
3. Cache interpolators by parameter hash
4. Parallelize table generation

## Critical Issues

### 1. **Performance Problem**
The 19,000× slowdown bug was here - every ODE step regenerated tables

### 2. **Error Handling**
Multiple bare except clauses:
- Line 153: Silent failure with fallback value
- Line 210: Cache key generation failure ignored

### 3. **Hard-coded Values**
- Fill values: `1e-30`
- Bounds: T ∈ [100, 10^8.98] K, nH ∈ [10^-8, 1] cm⁻³
- Cache size: 10,000 entries

### 4. **Global State**
Makes testing and parallel execution difficult

## Documentation Needs

### High Priority
1. Document table format and origin
2. Explain bounds on T and nH
3. Document caching strategy
4. Add units to all parameters

### Medium Priority
1. Explain P^-1 extrapolation
2. Document performance characteristics
3. Add references to Wiersma+09

## Recommended Improvements

### Immediate Fixes

1. **Fix bare except clauses**:
```python
try:
    Lambda_P_rho_tab[i,j] = Lambda(...)
except (ValueError, IndexError) as e:
    print(f"Interpolation failed at P={Ps[i]}, rho={rhos[j]}: {e}")
    Lambda_P_rho_tab[i,j] = 1e-30
```

2. **Vectorize table generation**:
```python
# Create meshgrid
P_grid, rho_grid = np.meshgrid(Ps, rhos, indexing='ij')
T_grid = P_grid * (mu * mp / rho_grid)
# Apply bounds vectorized
T_grid = np.clip(T_grid, 1e2, 10**8.98)
# Vectorized interpolation
Lambda_P_rho_tab = Lambda((log_nH_grid, log_T_grid, Z_grid, z_grid))
```

3. **Improve caching**:
```python
@lru_cache(maxsize=10000)
def tcool_P_cached(T_round, P_round, Z, z, mu):
    return tcool_P_uncached(T_round, P_round, Z, z, mu)
```

### Future Enhancements

1. **Class-based design**:
```python
class CoolingFunction:
    def __init__(self, table_path):
        self.load_table(table_path)
        self.interpolators = {}
    
    def get_interpolator(self, mu, Z, z):
        key = (mu, Z, z)
        if key not in self.interpolators:
            self.interpolators[key] = self._build_interpolator(mu, Z, z)
        return self.interpolators[key]
```

2. **Parallel table generation**:
```python
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(calculate_row, enumerate(Ps))
```

3. **Configurable bounds**:
```python
class CoolingConfig:
    T_MIN = 100
    T_MAX = 10**8.98
    NH_MIN = 1e-8
    NH_MAX = 1.0
```

## Testing Requirements

### Unit Tests
1. `test_load_table`: Verify table loading
2. `test_interpolation_bounds`: Check boundary behavior
3. `test_caching`: Verify cache hit/miss
4. `test_vectorization`: Compare scalar vs vector
5. `test_tcool_calculation`: Physics validation

### Integration Tests
1. Compare with published cooling curves
2. Test with wind_model integration
3. Performance benchmarks

### Regression Tests
1. Cooling time at standard conditions
2. Minimum cooling time values
3. High/low metallicity limits

## Code Quality Metrics

- **Documentation**: ~30% (missing detailed physics)
- **Type Hints**: ~50% (good coverage)
- **Error Handling**: Poor (bare excepts)
- **Performance**: Improved after bug fix
- **Test Coverage**: Unknown

## Summary

The cooling module provides essential cooling physics but needs:
1. **Error handling**: Remove bare except clauses
2. **Performance**: Vectorize table generation
3. **Documentation**: Add physics references and units
4. **Design**: Consider class-based approach
5. **Testing**: Add comprehensive test suite

The module successfully implements Wiersma+09 cooling with:
- 4D interpolation in (nH, T, Z, z)
- Efficient caching for repeated calls
- Lazy loading of tables
- Derived (P,ρ) interpolators

Critical bug (19,000× slowdown) has been fixed by caching interpolators.