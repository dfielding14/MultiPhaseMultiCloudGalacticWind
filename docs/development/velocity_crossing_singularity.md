# Handling Velocity Crossing Singularities in Cloud-Wind Interactions

## Problem Statement

The current multiphase wind model terminates integration when the relative velocity between clouds and hot wind approaches zero (`v_rel = v_wind - v_cloud → 0`). This occurs because the drag force contains terms proportional to `1/v_rel`, creating a singularity. However, this is an artificial limitation - physically, clouds can overtake the wind and vice versa, with the drag force simply reversing direction.

### Physical Motivation

In real galactic winds:
- Clouds can be accelerated past the hot wind velocity by momentum transfer
- The hot wind can decelerate below cloud velocities due to energy losses
- Multiple velocity crossings can occur as the system evolves
- The physics remains well-defined through these crossings (drag reverses)

### Mathematical Challenge

The drag force on clouds contains:
```
F_drag ∝ ρ_hot * v_rel * |v_rel| * A_cloud
```

When expressed in the ODE system:
```
dv_cloud/dr = (1/v_cloud) * [...terms involving v_rel in denominator...]
```

As `v_rel → 0`, numerical integrators fail due to the singularity, even though the physical acceleration remains finite.

## Proposed Solutions

### 1. Regularization via Coordinate Transformation

**Approach**: Transform to a coordinate system where the singularity is removed.

```python
# Instead of (r, v_wind, v_cloud), use (r, v_center, v_rel)
v_center = (M_wind * v_wind + M_cloud * v_cloud) / (M_wind + M_cloud)
v_rel = v_wind - v_cloud

# The equations become regular in this coordinate system
dv_center/dr = [smooth function]
dv_rel/dr = [function that goes smoothly through zero]
```

**Advantages**:
- Mathematically rigorous
- Removes singularity entirely
- Natural for momentum-conserving system

**Challenges**:
- Requires reformulating entire ODE system
- Multiple cloud species complicate the transformation

### 2. Analytical Jump Conditions

**Approach**: Derive analytical expressions for state immediately after crossing.

```python
def handle_velocity_crossing(state_before, dr_cross):
    """
    When |v_rel| < threshold, analytically compute state after crossing.
    """
    # Conservation laws during infinitesimal crossing:
    # 1. Total momentum conserved
    # 2. Total energy conserved (minus infinitesimal drag work)
    # 3. Mass fluxes continuous
    
    # Analytical solution for Δv across singularity
    v_wind_after = v_wind_before + Δv_analytical
    v_cloud_after = v_cloud_before - Δv_analytical * (M_wind/M_cloud)
    
    return state_after
```

**Advantages**:
- Preserves conservation laws exactly
- Fast computation
- Physically motivated

**Challenges**:
- Deriving robust analytical expressions
- Handling multiple simultaneous crossings

### 3. Regularized Drag Force

**Approach**: Replace singular drag law with regularized version.

```python
# Standard drag force
F_drag = C_D * ρ * A * v_rel * |v_rel|

# Regularized version
ε = 1.0  # km/s - regularization scale
F_drag_reg = C_D * ρ * A * v_rel * sqrt(v_rel^2 + ε^2)

# As v_rel → 0:
# F_drag_reg → C_D * ρ * A * v_rel * ε  (linear, non-singular)
```

**Advantages**:
- Simple implementation
- Smooth through crossing
- Minimal code changes

**Challenges**:
- Introduces artificial scale ε
- May affect solution accuracy
- Not mathematically exact

### 4. Event Detection with State Jump

**Approach**: Use ODE event detection to identify crossings and apply jump conditions.

```python
def velocity_crossing_event(t, y):
    """Event function for scipy.integrate.solve_ivp"""
    v_rel = y[v_wind_idx] - y[v_cloud_idx]
    return v_rel  # Triggers when this crosses zero

# Set up events for each cloud species
events = []
for i in range(N_cloud_species):
    event = lambda t, y, idx=i: y[1] - y[5 + idx*3 + 1]  # v_wind - v_cloud_i
    event.terminal = False  # Don't stop integration
    event.direction = 0     # Detect both directions
    events.append(event)

# Integrate with event handling
sol = solve_ivp(ode_func, ..., events=events)

# Post-process to apply jump conditions at each crossing
```

**Advantages**:
- Uses existing ODE solver capabilities
- Precise crossing detection
- Can handle multiple crossings

**Challenges**:
- Complex event handling logic
- Post-processing required
- May miss rapid oscillations

### 5. Implicit Time Stepping Near Singularity

**Approach**: Switch to implicit method when |v_rel| small.

```python
def adaptive_integration(state, r_span):
    """
    Use explicit method normally, implicit near singularities.
    """
    while r < r_final:
        v_rel_min = min(abs(v_wind - v_cloud_i) for all i)
        
        if v_rel_min > v_threshold:
            # Standard explicit integration
            state = rk45_step(state, dr)
        else:
            # Implicit method (e.g., backward Euler)
            state = implicit_step(state, dr)
    
    return solution
```

**Advantages**:
- Numerical stability through singularity
- No artificial modifications
- Adaptive approach

**Challenges**:
- Complex implementation
- Computational cost of implicit steps
- Convergence issues

## Recommended Implementation Strategy

### Phase 1: Regularized Drag (Quick Fix)
1. Implement regularized drag force with small ε
2. Validate against current results where v_rel stays large
3. Test sensitivity to regularization parameter

### Phase 2: Event Detection System
1. Add velocity crossing events to ODE solver
2. Implement momentum-conserving jump conditions
3. Validate conservation laws through crossings

### Phase 3: Coordinate Transformation (Optimal)
1. Reformulate ODEs in (v_center, v_rel) coordinates
2. Implement transformed system
3. Benchmark against regularized version

## Implementation Details

### Key Considerations

1. **Multiple Cloud Species**: Each species can cross independently
2. **Simultaneous Crossings**: Handle when multiple clouds cross together
3. **Rapid Oscillations**: Prevent numerical "chattering" near v_rel = 0
4. **Conservation Laws**: Ensure momentum/energy conservation
5. **Backwards Compatibility**: Maintain agreement with current results

### Proposed API

```python
class WindConfig:
    # New parameters
    velocity_crossing_method: str = 'regularized'  # or 'event', 'transform'
    velocity_regularization: float = 1.0  # km/s
    allow_cloud_overtaking: bool = True
    
class WindModel:
    def run(self):
        if self.config.allow_cloud_overtaking:
            solution = self._run_with_crossing_handler()
        else:
            solution = self._run_standard()  # Current behavior
```

### Validation Tests

1. **Conservation Test**: Total momentum before/after crossing
2. **Reversibility Test**: Forward-backward integration
3. **Convergence Test**: Solution as ε → 0 (for regularization)
4. **Physical Test**: Compare with hydrodynamic simulations

## Physical Implications

### Expected Behaviors After Implementation

1. **Cloud Bunching**: Faster clouds catch up to slower ones
2. **Wind Deceleration**: Hot wind can fall below cloud velocities
3. **Oscillatory Dynamics**: Clouds may oscillate around wind velocity
4. **Terminal State Changes**: Different asymptotic configurations

### Impact on Observables

- Column density distributions may show multiple velocity components
- Absorption line profiles could be broader or multi-peaked
- Mass flux profiles may show non-monotonic behavior

## Timeline and Priority

**High Priority** (Enable more parameter space):
- Regularized drag implementation (1-2 days)
- Basic validation tests (1 day)

**Medium Priority** (Improve accuracy):
- Event detection system (3-4 days)
- Jump condition derivation (2 days)
- Comprehensive testing (2 days)

**Low Priority** (Optimal solution):
- Coordinate transformation (1 week)
- Full validation suite (3 days)
- Documentation and examples (2 days)

## References

- Scannapieco & Brüggen (2015) - Cloud-wind interactions in simulations
- Gronke & Oh (2020) - Cloud survival and acceleration
- Numerical Recipes Ch. 17 - Handling ODEs with singularities
- Hairer & Wanner (1996) - Solving Ordinary Differential Equations II

---

**Next Steps**: 
1. Approve approach selection
2. Create feature branch `feature/velocity-crossing`
3. Implement Phase 1 (regularization) as proof of concept
4. Validate and benchmark before proceeding to Phase 2/3