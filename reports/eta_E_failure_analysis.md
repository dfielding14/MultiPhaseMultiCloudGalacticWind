# Comprehensive Analysis: eta_E < 1 Integration Failure

## Executive Summary

**CRITICAL FINDING**: The wind model fails immediately when eta_E < 1, even with infinitesimally small cold cloud content (eta_M_cold = 1e-6). This is not a numerical issue but a fundamental physical inconsistency in how the sonic point initial conditions are calculated.

## The Core Problem

### What Happens

When eta_E < 1:
1. Integration terminates immediately with negative pressure/density
2. Hot-only solution works fine
3. Adding even tiny cold clouds (eta_M_cold = 1e-6) causes immediate failure
4. The failure occurs at the very first integration step

### Root Cause

The sonic point calculation enforces Mach = 1 + ε at radius r*, but with low energy injection (eta_E < 1), this creates physically inconsistent initial conditions:

```
v0 = sqrt(Edot/Mdot) * [1/((γ-1)M) + 1/2]^(-1/2)
```

As eta_E decreases:
- v0 decreases (less energy available)
- To maintain Mach ≈ 1, temperature T0 must also decrease
- This creates extreme conditions at the "sonic" point

## Detailed Analysis

### 1. Initial Condition Scaling with eta_E

| eta_E | v0 [km/s] | T0 [K] | v_infinity [km/s] | v0/v_inf |
|-------|-----------|--------|-------------------|----------|
| 2.0   | 1035.9    | 4.0e7  | 2000.0           | 0.518    |
| 1.0   | 732.5     | 2.0e7  | 1414.2           | 0.518    |
| 0.5   | 518.0     | 1.0e7  | 1000.0           | 0.518    |
| 0.1   | 231.6     | 2.0e6  | 447.2            | 0.518    |

**Key observation**: The ratio v0/v_inf is constant (~0.518) by construction, but the absolute values scale with sqrt(eta_E).

### 2. The Temperature Problem

The sonic point calculation forces:
```
Mach = v0/cs = v0/sqrt(γP0/ρ0) = 1 + ε
```

With fixed Mach and decreasing v0, we must have decreasing sound speed, which means:
- Lower temperature T0
- Lower pressure P0
- Steeper pressure gradients

At eta_E = 0.1, T0 = 2×10^6 K - extremely cold for a galactic wind!

### 3. Why Hot-Only Works But Hot+Cold Fails

The critical difference is in the source terms. Wind_Evo applies source terms when Mach < 1:

```python
# In Wind_Evo:
Edot_SN = Edot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0)
Mdot_SN = Mdot_per_Vol * np.where(Mach_sq_wind<1, 1.0, 0.0)
```

With Mach ≈ 1.0001 at the sonic point:
- We're right at the boundary where source terms activate
- Any numerical noise can trigger the source terms
- The source terms with low eta_E are insufficient to maintain positive pressure

### 4. The Energy Budget Paradox

The fundamental issue is that the sonic point calculation assumes a self-consistent steady-state solution exists, but:

1. **Energy conservation** requires: `v_infinity = sqrt(2*Edot/Mdot)`
2. **Sonic point** requires: `Mach = 1` at some radius r*
3. **Steady flow** requires smooth acceleration from v0 to v_infinity

With low eta_E:
- v_infinity is small (low energy)
- v0 must be even smaller (v0 ≈ 0.518 * v_infinity)
- But maintaining Mach = 1 with small v0 requires unrealistically low T0

## Mathematical Deep Dive

### The Sonic Point Equations

At the sonic point (Mach = 1):
```
dv/dr → ∞ (singularity)
```

The code regularizes this by starting at Mach = 1 + ε, where ε = 1e-5.

The velocity at the sonic point is:
```
v* = sqrt(Edot/Mdot) * [1/((γ-1)(1+ε)) + 1/2]^(-1/2)
   ≈ sqrt(Edot/Mdot) * 0.7303  (for γ=5/3, ε=1e-5)
```

The temperature is then:
```
T* = (v*/Mach)^2 * (mu*mp)/(γ*kb)
   = v*^2 * (mu*mp)/(γ*kb)  (since Mach ≈ 1)
```

For eta_E = 0.1:
```
v* = 231.6 km/s = 2.316e7 cm/s
T* = (2.316e7)^2 * (0.62 * 1.67e-24) / (1.667 * 1.38e-16)
   ≈ 2.4e6 K
```

This is orders of magnitude colder than typical wind temperatures (~10^7 K).

## Why This Matters

### Physical Implications

1. **Unrealistic initial conditions**: T = 2×10^6 K is too cold for a wind driven by supernova feedback
2. **Extreme gradients**: Such low temperatures create steep pressure gradients
3. **Numerical instability**: The equations become stiff near the sonic point

### Code Behavior

The integration fails because:
1. Pressure gradient dP/dr is extremely negative
2. Within one integration step, P becomes negative
3. Negative pressure triggers termination

## Potential Solutions

### 1. Modified Sonic Point Calculation (Recommended)

Instead of forcing Mach = 1, use energy considerations:
```python
def calculate_sonic_point_energy_limited(eta_E, eta_M, SFR):
    if eta_E < 0.5:  # Energy-starved regime
        # Start subsonic and let flow accelerate naturally
        Mach0 = 0.5  # Start subsonic
        # Calculate v0 from energy budget, not Mach number
        v0 = 0.1 * sqrt(2*Edot/Mdot)  # Start slow
        # Set reasonable temperature
        T0 = 1e7  # Reasonable wind temperature
        # Calculate pressure from ideal gas
        P0 = rho0 * kb * T0 / (mu * mp)
```

### 2. Different Initial Radius

For low eta_E, start integration from a different radius:
```python
if eta_E < 1.0:
    # Start from further out where flow is already accelerated
    r_start = r_star * (2.0 / eta_E)  # Scale with energy
```

### 3. Source Term Modification

Modify how source terms are applied near the sonic point:
```python
# Smooth transition instead of sharp cutoff
sonic_region = np.exp(-(Mach - 1.0)**2 / 0.01)
Edot_SN = Edot_per_Vol * sonic_region
```

### 4. Energy-Limited Wind Mode

Implement a separate solver for energy-starved winds:
```python
if eta_E < 0.5:
    # Use pressure-driven wind equations
    # Don't enforce sonic point
    # Let flow naturally accelerate
    return solve_energy_limited_wind(...)
```

## Recommended Fix

The most robust solution is to recognize that **not all parameter combinations produce trans-sonic winds**. 

For eta_E < 1:
1. Check if trans-sonic solution is possible
2. If not, use subsonic initial conditions
3. Let the flow naturally accelerate (or not)
4. Accept that some winds may never become supersonic

### Implementation Sketch

```python
def calculate_initial_conditions(eta_E, eta_M, ...):
    # Calculate energy-limited velocity
    v_energy = np.sqrt(2 * Edot / Mdot)
    
    # Check if trans-sonic solution exists
    v_sonic_min = estimate_minimum_sonic_velocity(...)
    
    if v_energy < v_sonic_min:
        # Energy-starved: start subsonic
        warnings.warn("Low eta_E: using subsonic initial conditions")
        Mach0 = 0.3
        T0 = 1e7  # Reasonable temperature
        v0 = Mach0 * np.sqrt(gamma * kb * T0 / (mu * mp))
    else:
        # Standard trans-sonic solution
        Mach0 = 1.0 + epsilon
        v0 = calculate_sonic_velocity(Edot, Mdot)
        T0 = v0**2 * mu * mp / (gamma * kb * Mach0**2)
    
    return v0, T0, Mach0
```

## Conclusion

The eta_E < 1 failure is not a bug but a fundamental limitation of the current sonic point approach. The code assumes all winds pass through a sonic point, but energy-starved winds (eta_E < 1) cannot sustain the required conditions.

The solution is to:
1. Recognize when parameters don't support trans-sonic flow
2. Use appropriate initial conditions for each regime
3. Allow subsonic or failed wind solutions

This would make the code more physically realistic and numerically robust.