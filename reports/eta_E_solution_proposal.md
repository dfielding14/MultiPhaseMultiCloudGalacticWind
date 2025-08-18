# Proposed Solution for eta_E < 1 Failure

## Quick Summary

The model fails with eta_E < 1 because it tries to force a trans-sonic solution that doesn't physically exist. With low energy injection, the wind cannot reach supersonic speeds at the assumed sonic radius.

## The Fix

### Option 1: Minimum Energy Check (Simplest)

Add a check in `WindModel.__init__`:

```python
def _calculate_sonic_point_conditions(self):
    # Existing code...
    Edot = self.eta_E * (self.config.E_SN / self.config.mstar / Msun) * SFR_cgs
    Mdot = self.eta_M * SFR_cgs
    
    # NEW: Check if we have enough energy for trans-sonic flow
    v_infinity = np.sqrt(2 * Edot / Mdot)
    v_sonic_required = v_infinity * 0.3  # Empirical minimum
    
    if self.eta_E < 0.5:  # Energy-starved regime
        warnings.warn(
            f"Low energy (eta_E={self.eta_E}): Wind may not reach supersonic speeds.\n"
            f"Consider using eta_E >= 0.5 for robust trans-sonic solutions.",
            UserWarning
        )
        
        # Adjust initial conditions for subsonic start
        self.use_subsonic_start = True
        Mach0 = 0.5  # Start subsonic
        T0 = 1e7  # Set reasonable temperature
        v0 = Mach0 * np.sqrt(gamma * kb * T0 / (self.config.mu * mp))
        rho0 = Mdot / (self.config.Omwind * r_star**2 * v0)
        P0 = rho0 * kb * T0 / (self.config.mu * mp)
    else:
        # Standard trans-sonic calculation
        self.use_subsonic_start = False
        # ... existing code ...
```

### Option 2: Adaptive Sonic Radius (Better)

Adjust the sonic radius based on energy availability:

```python
def _calculate_sonic_point_conditions(self):
    # Calculate energy per unit mass
    epsilon = Edot / Mdot  # erg/g
    
    # Sonic radius scales with energy
    # Higher energy -> smaller sonic radius (closer to galaxy)
    # Lower energy -> larger sonic radius (further out)
    if self.eta_E < 1.0:
        # Increase sonic radius for low energy
        r_star_adjusted = self.r_star_kpc * (1.0 / self.eta_E)**0.5
        warnings.warn(
            f"Adjusting sonic radius from {self.r_star_kpc} to {r_star_adjusted:.2f} kpc "
            f"due to low eta_E={self.eta_E}"
        )
        r_star = r_star_adjusted * kpc
    else:
        r_star = self.r_star_kpc * kpc
    
    # Continue with standard calculation...
```

### Option 3: Bypass Sonic Point (Most Robust)

For low energy, skip the sonic point entirely:

```python
def run(self):
    if self.eta_E < 0.5:
        # Start with hydrostatic equilibrium
        return self._run_subsonic_wind()
    else:
        # Standard trans-sonic solution
        return self._run_standard()

def _run_subsonic_wind(self):
    """Run wind that starts subsonic and may never go supersonic."""
    # Start with hydrostatic-like conditions
    T0 = 1e7  # K
    cs0 = np.sqrt(gamma * kb * T0 / (self.config.mu * mp))
    v0 = 0.1 * cs0  # Start very subsonic
    
    # Set density from mass conservation
    rho0 = self.eta_M * self.SFR * Msun/yr / (Omwind * r0**2 * v0)
    P0 = rho0 * kb * T0 / (self.config.mu * mp)
    
    # Integrate without expecting sonic transition
    # ...
```

## Recommended Approach

**For immediate fix**: Use Option 1 - add warning and adjust initial conditions

**For long-term**: Implement Option 3 - properly handle subsonic winds

## Testing the Fix

Test with problematic parameters:
```python
model = WindModel(
    SFR=1.0,
    eta_M=0.5,
    eta_M_cold=1e-6,
    eta_E=0.1,  # This currently fails
    # ...
)

# Should now either:
# 1. Warn and adjust initial conditions
# 2. Run as subsonic wind
# 3. Fail gracefully with informative error
```

## Why This Works

1. **Respects physics**: Doesn't force unphysical trans-sonic solution
2. **Numerically stable**: Avoids extreme gradients at sonic point
3. **Flexible**: Handles both high and low energy regimes
4. **Transparent**: Warns user about regime change

## Implementation Priority

1. **First**: Add energy check and warning
2. **Second**: Implement subsonic initial conditions
3. **Third**: Test thoroughly with various eta_E values
4. **Fourth**: Document the energy regimes in user guide

The key insight is that **not all winds are trans-sonic**. Low-energy winds may remain subsonic, and that's physically valid.