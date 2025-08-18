# Comprehensive Fix List - Multiphase Galactic Wind Model

## Priority 1: Critical Bugs (Must Fix Immediately)

### 1. config.py - Broken set_defaults() method
**Issue**: References non-existent `_default_*` attributes
**Location**: Lines 107-124
**Fix**: Implement proper default storage mechanism
**Impact**: Method completely non-functional

### 2. analysis_helpers.py - Dimensional error in cloud_radius()
**Issue**: Wrong units in density calculation (line 135)
**Location**: Line 135: `rho_cloud = config.mu * mp * kb * 1e4 / kb`
**Fix**: Use proper pressure-based calculation
**Impact**: Returns wrong cloud radii

### 3. plotting.py - Undefined attributes
**Issue**: References `solution.v_cl` and `solution.Z_cl` that don't exist
**Location**: Lines 252, 273
**Fix**: Properly access cloud velocities from Solution object
**Impact**: Plots fail with AttributeError

### 4. observables.py - Multi-species velocity bug
**Issue**: Assumes all species have same velocity (line 256)
**Location**: Lines 256-294
**Fix**: Use individual species velocities
**Impact**: Wrong velocity distributions

### 5. analysis_helpers.py - Placeholder densities
**Issue**: Uses fake value `1e-10` for cloud density (line 382)
**Location**: Line 382 in calculate_cloud_moments()
**Fix**: Calculate actual cloud densities
**Impact**: Wrong moment calculations

### 6. core_physics.py - Sonic point singularity
**Issue**: Division by (1 - 1/M²) at Mach = 1
**Location**: Lines 222-230 in Wind_Evo()
**Fix**: Add regularization near sonic point
**Impact**: Integration failures near Mach = 1

## Priority 2: Parameter Validation

### 7. wind_model.py - No parameter validation
**Location**: __init__ method
**Fix**: Add comprehensive validation
```python
def _validate_parameters(self):
    if self.SFR <= 0:
        raise ValueError("SFR must be positive")
    if self.eta_E <= 0:
        raise ValueError("eta_E must be positive")
    # etc.
```

### 8. config.py - No range checking
**Location**: __init__ method
**Fix**: Add validate() method

### 9. core_physics.py - No input validation
**Location**: setup_cloud_powerlaw_distribution()
**Fix**: Check alpha > 0, mass range valid

## Priority 3: Error Handling

### 10. cooling.py - Bare except clauses
**Location**: Lines 153, 210
**Fix**: Specify exception types

### 11. plotting.py - Bare except for LaTeX
**Location**: Line 62
**Fix**: Catch specific exception

### 12. analysis_helpers.py - Arbitrary fallbacks
**Location**: Line 214 (1e10*Myr for negative cooling)
**Fix**: Handle properly with warnings

## Priority 4: Hard-coded Values

### 13. analysis_helpers.py - Hard-coded parameters
**Issues**:
- Line 84: `T_cl = 1e4` 
- Line 194: `T_cl = 1e4`
- Line 269: `v_circ = 150.0 * km`
- Line 46, 92: `f_spitzer = 1.0`
**Fix**: Use config parameters

### 14. constants.py - Precision issues
**Issues**:
- `G = 6.673e-8` (low precision)
- `Msun = 2e33` (low precision)
**Fix**: Use CODATA values

### 15. observables.py - Magic numbers
**Issues**:
- Line 219: `1e-30` for zero gradient
- Line 234: `n_bins = max(10, len(v_cloud_cms) // 8)`
**Fix**: Make configurable

## Priority 5: Code Duplication

### 16. plotting.py - Repeated colorbar code
**Location**: Lines 179-216 and 354-391
**Fix**: Extract to function

### 17. observables.py - Repeated masking
**Location**: Multiple locations
**Fix**: Create utility function

## Priority 6: Performance

### 18. cooling.py - Table generation
**Location**: get_cooling_interpolator()
**Fix**: Vectorize loops

### 19. observables.py - Species loops
**Location**: Lines 262-294
**Fix**: Vectorize operations

### 20. wind_model.py - Cooling interpolator
**Location**: Line 355
**Fix**: Cache interpolator

## Priority 7: Missing Implementations

### 21. analysis_helpers.py - Gradient_Components()
**Location**: Lines 284-330
**Fix**: Implement actual calculations

### 22. observables.py - Path length corrections
**Fix**: Add geometric corrections

## Priority 8: Documentation

### 23. Add missing docstrings
**Modules**: All modules need better docstrings

### 24. Add type hints
**Modules**: Especially core_physics.py, plotting.py

### 25. Document units consistently
**Modules**: All modules

## Priority 9: Testing

### 26. Create unit tests
**Focus**: Critical functions in each module

### 27. Add integration tests
**Focus**: Full workflow tests

### 28. Add regression tests
**Focus**: Compare with known solutions

## Implementation Order

### Phase 1: Critical Fixes (Issues 1-6)
1. Fix config.set_defaults()
2. Fix cloud_radius() units
3. Fix plotting attributes
4. Fix observables multi-species
5. Fix placeholder densities
6. Add sonic regularization

### Phase 2: Validation (Issues 7-9)
1. Add WindModel validation
2. Add config validation
3. Add core_physics validation

### Phase 3: Error Handling (Issues 10-12)
1. Fix bare excepts
2. Add proper error messages
3. Remove arbitrary fallbacks

### Phase 4: Clean-up (Issues 13-22)
1. Remove hard-coded values
2. Fix precision issues
3. Extract duplicate code
4. Optimize performance
5. Complete implementations

### Phase 5: Documentation & Testing (Issues 23-28)
1. Add comprehensive docstrings
2. Add type hints
3. Create test suite
4. Setup documentation website

## Success Criteria

- [ ] All critical bugs fixed
- [ ] All functions have parameter validation
- [ ] No bare except clauses
- [ ] No hard-coded physics parameters
- [ ] All functions have docstrings with units
- [ ] Test coverage > 80%
- [ ] Documentation website live

## Estimated Time

- Phase 1: 2 hours
- Phase 2: 1 hour
- Phase 3: 30 minutes
- Phase 4: 2 hours
- Phase 5: 3 hours

**Total: ~8.5 hours**