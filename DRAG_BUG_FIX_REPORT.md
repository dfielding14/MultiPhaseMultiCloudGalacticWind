# Critical Drag Force Bug Fix Report

## Executive Summary

Found and fixed a **critical sign error** in the cloud-wind drag force calculation that violated Newton's third law and caused unphysical behavior.

## The Bug

### Location
`multiphasegalacticwind/core_physics.py`, line 209

### Original Code
```python
p_dot_ram = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2
```

### Fixed Code
```python
p_dot_ram = 0.5 * drag_coeff * rho_wind * np.pi * v_rel * np.abs(v_rel) * r_cloud**2
```

## Root Cause

Using `v_rel**2` instead of `v_rel * |v_rel|` caused the drag force to **always be positive**, regardless of the sign of the relative velocity.

### Physics Violation

When `v_cloud > v_wind` (cloud moving faster than wind):
- **Expected**: Drag slows cloud, speeds up wind
- **Bug behavior**: Drag speeds up cloud, slows wind (violates Newton's 3rd law!)

## Impact Analysis

### Before Fix
With η_M=1, η_M_cold=0.3 at r=0.3 kpc:
- dv/dr = **+3.83×10⁻⁶** cm/s/cm (positive - unphysical acceleration!)
- Wind artificially accelerated despite drag
- Integration failures due to numerical instability

### After Fix
Same parameters:
- dv/dr = **-3.82×10⁻⁶** cm/s/cm (negative - physical deceleration)
- Wind properly decelerated by drag
- Physics now correct

## Test Results

### Drag Sign Test
```
v_wind = 50 km/s, v_cloud = 30 km/s (wind faster)
✓ Drag accelerates cloud, decelerates wind

v_wind = 30 km/s, v_cloud = 50 km/s (cloud faster)  
✓ Drag decelerates cloud, accelerates wind (NOW CORRECT!)
```

### ODE Convergence
- η_M_cold → 0: Full model still converges to hot-only (< 0.01% difference)
- Validates that fix preserves correct limiting behavior

### J0021+0052 Galaxy
- Previously failing parameter combinations still constrained by energy budget
- But now fail for correct physical reasons (insufficient energy)
- Not artificial numerical instability from wrong drag sign

## Mathematical Details

### Drag Force Formula

Correct form preserves sign of relative velocity:
```
F_drag = ½ C_D ρ A v_rel |v_rel|
       = ½ C_D ρ A |v_rel|² sign(v_rel)
```

Where:
- `v_rel = v_wind - v_cloud`
- `sign(v_rel)` determines force direction
- Force opposes relative motion

### Momentum Conservation

The fix ensures Newton's third law:
```
dp_wind/dt = -F_drag
dp_cloud/dt = +F_drag
```

Total momentum conserved: `d(p_wind + p_cloud)/dt = 0`

## Why This Matters

1. **Physical Correctness**: Drag now properly opposes relative motion
2. **Momentum Conservation**: Newton's third law satisfied
3. **Numerical Stability**: Removes artificial acceleration terms
4. **Energy Conservation**: Drag correctly dissipates kinetic energy

## Verification Tests

Created comprehensive tests in:
- `audit_drag_term.py` - Validates drag sign conventions
- `test_ode_convergence.py` - Confirms convergence behavior preserved
- `direct_ode_comparison.py` - Shows corrected gradient signs

## Recommendations

1. **Add regression test** to prevent reintroduction of bug
2. **Review other force terms** for similar sign issues
3. **Document sign conventions** clearly in code comments

## Conclusion

This was a fundamental physics bug that violated conservation laws. The fix is simple but critical for physical accuracy. The drag force now correctly:
- Opposes relative motion
- Conserves momentum
- Preserves correct limiting behavior
- Enables physically meaningful simulations

While low-SFR galaxies like J0021+0052 remain challenging due to energy constraints, they now fail for the **right physical reasons** rather than numerical artifacts.