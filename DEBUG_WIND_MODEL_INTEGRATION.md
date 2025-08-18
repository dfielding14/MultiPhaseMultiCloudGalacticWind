# Wind Model Integration Failure Debug Guide

**Problem**: Full (hot+cold) wind model fails immediately while hot-only succeeds for J0021+0052 galaxy (SFR=3.0 M☉/yr)
**Goal**: Identify the exact term(s) causing integration failure when cold component is included

---

## 1. Problem Analysis

### Symptoms
- **Full model**: Fails at r=0.30 kpc (first integration point) with most parameter combinations
- **Hot-only**: Runs successfully to completion
- **Key insight**: Full model should reduce to hot-only when η_M_cold → 0, but doesn't

### Test Case: J0021+0052
- SFR = 3.0 M☉/yr
- v_circ = 0.001 km/s (effectively zero)
- Z = 0.3 solar
- Failing parameters: η_M=0.3, η_M_cold=0.3, η_E=0.5

---

## 2. Implementation Files

### Core ODE Files
- `multiphasegalacticwind/core_physics.py`
  - `Wind_Evo()` - Full hot+cold ODE system (lines ~100-400)
  - `Hot_Wind_Evo()` - Hot-only ODE system (lines ~500-600)

### Key State Variables
```python
# Hot phase state
v_wind    # velocity [km/s]
rho_wind  # density [g/cm³]
P         # pressure [dyne/cm²]
rhoZ_wind # metal density [g/cm³]

# Cold phase state (per species i)
M_cloud_i  # cloud mass [M☉]
v_cloud_i  # cloud velocity [km/s]
Z_cloud_i  # cloud metallicity [solar]
```

---

## 3. Term Decomposition Strategy

### 3.1 Create Term Logger

```python
# In core_physics.py
class ODETerms:
    """Container for decomposed ODE terms"""
    def __init__(self):
        # Velocity equation terms
        self.dv_gravity = 0.0
        self.dv_pressure = 0.0
        self.dv_drag = 0.0
        self.dv_geometry = 0.0
        
        # Pressure equation terms
        self.dP_advection = 0.0
        self.dP_cooling = 0.0
        self.dP_heating = 0.0
        self.dP_mixing = 0.0
        self.dP_work = 0.0
        
        # Density equation terms
        self.drho_continuity = 0.0
        self.drho_mixing = 0.0
        self.drho_injection = 0.0
        
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items()}
```

### 3.2 Instrument ODE Functions

Add term decomposition to both `Wind_Evo` and `Hot_Wind_Evo`:

```python
def Wind_Evo(r, y, params, log_terms=False):
    """Modified to log individual terms"""
    if log_terms:
        terms = ODETerms()
        
    # ... existing unpacking ...
    
    # Velocity equation: dv/dr
    terms.dv_gravity = -g/v_wind
    terms.dv_pressure = -(1/rho_wind) * (1/v_wind) * dP_dr_base
    terms.dv_drag = drag_contribution  # From cloud-wind interaction
    terms.dv_geometry = geometric_factor
    
    # ... similar for pressure and density ...
    
    if log_terms:
        return dydt, terms
    return dydt
```

---

## 4. Epsilon Scaling Tests

### 4.1 Test Matrix

Run models with progressively smaller cold mass loading:

| Test | η_M | η_M_cold | η_E | Expected |
|------|-----|----------|-----|----------|
| Hot-only | 0.3 | 0.0 | 0.5 | Success |
| Full-ε1 | 0.3 | 1e-6 | 0.5 | → Hot-only |
| Full-ε2 | 0.3 | 1e-8 | 0.5 | → Hot-only |
| Full-ε3 | 0.3 | 1e-10 | 0.5 | → Hot-only |
| Full-base | 0.3 | 0.3 | 0.5 | Fails |

### 4.2 Implementation

```python
# epsilon_scaling_test.py
import numpy as np
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.core_physics import Wind_Evo, Hot_Wind_Evo

def run_epsilon_test():
    """Test that full model → hot-only as η_M_cold → 0"""
    
    epsilons = [0.0, 1e-10, 1e-8, 1e-6, 0.3]
    results = []
    
    for eps in epsilons:
        config = WindConfig(
            N_cloud_species=5 if eps > 0 else 0,
            cooling_factor=0.0  # Disable cooling for cleaner test
        )
        
        model = WindModel(
            config=config,
            SFR=3.0,
            v_circ=0.001,
            eta_M=0.3,
            eta_M_cold=eps,
            eta_E=0.5
        )
        
        # Get initial RHS terms
        y0 = model.get_initial_conditions()
        terms = model.get_ode_terms(r=0.3, y=y0)
        
        results.append({
            'epsilon': eps,
            'terms': terms,
            'success': model.can_integrate()
        })
    
    return results
```

---

## 5. Controlled Isolation Switches

### 5.1 Feature Flags

Add temporary flags to isolate suspected components:

```python
class DebugFlags:
    NO_COOLING = False      # Disable radiative cooling
    NO_DRAG = False         # Disable cloud-wind drag
    NO_MIXING = False       # Disable turbulent mixing
    NO_GRAVITY = False      # Disable gravitational terms
    NO_INJECTION = False    # Disable mass/energy injection
```

### 5.2 Modified ODE with Flags

```python
def Wind_Evo_Debug(r, y, params, flags=None):
    """ODE with debug flags to isolate terms"""
    if flags is None:
        flags = DebugFlags()
    
    # ... normal computation ...
    
    if flags.NO_COOLING:
        cooling_term = 0.0
    
    if flags.NO_DRAG:
        drag_term = 0.0
    
    # etc.
```

---

## 6. Diagnostic Implementation

### 6.1 Main Debug Script

```python
#!/usr/bin/env python
"""debug_wind_integration.py - Diagnose integration failure"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig

def diagnose_integration_failure():
    """Main diagnostic routine"""
    
    # 1. Run hot-only baseline
    hot_only = run_hot_only_model()
    
    # 2. Run epsilon scaling tests
    epsilon_results = run_epsilon_scaling()
    
    # 3. Compare RHS terms at r=0.3
    term_comparison = compare_initial_terms(hot_only, epsilon_results)
    
    # 4. Identify divergent terms
    divergent = find_divergent_terms(term_comparison)
    
    # 5. Run isolation tests
    isolation_results = run_isolation_tests(divergent)
    
    # 6. Generate report
    generate_report(all_results)
    
    return divergent_terms

def compare_initial_terms(hot_only, full_models):
    """Compare ODE terms at initial point"""
    
    df = pd.DataFrame()
    
    for model in [hot_only] + full_models:
        terms = model.get_initial_terms()
        df = df.append({
            'model': model.name,
            'epsilon': model.eta_M_cold,
            **terms.to_dict()
        }, ignore_index=True)
    
    # Check convergence to hot-only
    for col in df.columns[2:]:  # Skip model, epsilon
        hot_value = df[df['model'] == 'hot_only'][col].values[0]
        
        for idx, row in df[df['model'] != 'hot_only'].iterrows():
            eps = row['epsilon']
            full_value = row[col]
            
            # Check if scales with epsilon
            if abs(full_value - hot_value) > 1e-10:
                ratio = abs(full_value - hot_value) / (eps + 1e-20)
                print(f"Term {col}: Δ = {full_value - hot_value:.2e}, ratio/ε = {ratio:.2e}")
    
    return df
```

---

## 7. Expected Failure Modes

### 7.1 Common Culprits

1. **Double-counted gravity**: Check if gravity applied twice in full model
2. **Sign errors**: Pressure gradient sign convention
3. **Division by zero**: When ρ_cold → 0 or T → 0
4. **Unit mismatches**: CGS vs code units inconsistency
5. **Missing normalization**: Mass flux normalization differs
6. **Stiff cooling**: Cooling time << dynamical time

### 7.2 Validation Checks

```python
def validate_terms(terms, r, state):
    """Sanity checks on ODE terms"""
    
    # Gravity should be negative (deceleration)
    assert terms.dv_gravity <= 0, f"Gravity term positive: {terms.dv_gravity}"
    
    # Pressure gradient ~ -dP/dr / (ρv)
    expected_dP = -state['P'] / (state['rho'] * state['v'] * r)
    assert abs(terms.dv_pressure - expected_dP) < 0.1 * abs(expected_dP)
    
    # Continuity: d(ρvr²)/dr = sources
    continuity_check = compute_continuity(state, r)
    assert abs(terms.drho_continuity - continuity_check) < 1e-10
    
    # All terms finite
    for key, value in terms.to_dict().items():
        assert np.isfinite(value), f"Non-finite term: {key} = {value}"
```

---

## 8. Report Template

### 8.1 Output Structure

```
DEBUG_RESULTS/
├── term_comparison.csv         # All ODE terms for each model
├── epsilon_convergence.png     # Plot of term convergence
├── isolation_results.json      # Which flags fix the issue
├── divergent_terms.txt         # List of problematic terms
├── fix_validation.py           # Test confirming fix
└── REPORT.md                   # Summary findings
```

### 8.2 Success Criteria

- [ ] Full model runs without crashing
- [ ] As η_M_cold → 0, full model terms → hot-only terms
- [ ] Identified exact term(s) causing divergence
- [ ] Isolation test confirms root cause
- [ ] Fix implemented and validated

---

## 9. Quick Start Commands

```bash
# Run full diagnostic suite
python debug_wind_integration.py

# Run specific epsilon test
python debug_wind_integration.py --epsilon=1e-8

# Run with isolation flags
python debug_wind_integration.py --no-cooling --no-drag

# Generate comparison plots
python debug_wind_integration.py --plot-terms

# Validate fix
pytest tests/test_wind_integration_fix.py
```

---

## 10. Critical Implementation Notes

1. **Ensure identical initial conditions** between hot-only and full models
2. **Use same integration tolerances** (rtol, atol) for fair comparison
3. **Log rejected integration steps** to see where solver struggles
4. **Check array indexing** - cold species indexing may be off by one
5. **Verify parameter passing** - params tuple must be identical
6. **Monitor memory layout** - state vector packing/unpacking errors

---

## Definition of Done

✓ Full model integrates successfully for J0021+0052 test case
✓ Full model → hot-only in limit η_M_cold → 0 (validated numerically)
✓ Root cause identified and documented
✓ Fix implemented with regression test
✓ No degradation in other test cases