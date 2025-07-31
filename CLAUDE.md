# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase implementing multiphase galactic wind models. It simulates steady-state evolution of galactic winds with hot gas and cold embedded clouds, based on Fielding & Bryan's paper "The Structure of Multiphase Galactic Winds".

## CRITICAL: Package Development Status 

### 🚀 Current State: Major Architecture Improvements Complete

The package has undergone significant refactoring to improve code quality, maintainability, and usability:

#### ✅ Completed Refactoring 
1. **Cooling Module Separation** - All cooling functionality moved to dedicated `cooling.py`
   - ~200 lines of code extracted from `core_physics.py`
   - Lazy loading of cooling tables (no import-time execution)
   - Proper parameterization - `mu` flows from config throughout
   - Clean API: `tcool_P()`, `get_cooling_interpolator()`, etc.

2. **Configuration System** - Clean parameter management via `WindConfig`
   - All model parameters in one place
   - No more scattered globals
   - Easy to modify: `WindConfig(f_turb0=0.2, drag_coeff=0.3)`

3. **Constants Separation** - Physical constants in `constants.py`
   - Clear distinction from configurable parameters
   - Standard values: `mp`, `kb`, `Msun`, `kpc`, etc.

4. **Parameter Flow Fixed** - Proper data flow throughout
   - `Wind_Evo` accepts 9-parameter tuple
   - Event functions use factory pattern
   - No undefined globals

5. **Import Safety** - No code execution on import
   - Cooling tables load on first use
   - Clean module initialization

### ⚠️ Known Issues

1. **README Critically Outdated** - Does not reflect new API
   - Old examples won't work
   - No `WindConfig` or `constants` documentation
   - Missing cooling module info

2. **Incomplete Examples** - Some examples still use old patterns
   - Need updating for new API
   - Should show configuration options


The package architecture is now clean and well-organized, BUT we need to verify it actually works!

### Next things to do:
1. **Performance**
   - Need profiling to identify bottlenecks
   - Cooling interpolator regeneration on each call?
     - maybe there is a better faster way to do the cooling interpolation
   - Explore numba JIT optimization
2. **Plotting**
   - the units aren't right in some of the plots so the Mdot/SFR is >20 orders of magnitude off and the same with Mcloud


### Key Principles:
- **No backward compatibility** - clean API only
- **Single source of truth** - parameters only in config
- **Explicit is better** - no hidden globals
- **Testable code** - pure functions where possible

## Key Commands

### Package Testing
```bash
# Test basic functionality
python -c "from multiphasegalacticwind import WindModel; model = WindModel(SFR=10.0)"

# Run examples
python examples/simple_example.py
python examples/basic_example.py
python examples/observational_comparison.py
python examples/column_density_example.py
```

### Legacy Code (for comparison)
```bash
# Original multicloud simulation
python legacy/Multiphase_Wind_Evolution_Multicloud.py

# Interactive analysis
jupyter notebook legacy/multicloud.ipynb
```

### Dependencies
Install with: `pip install numpy scipy matplotlib cmasher h5py jupyter`

## Architecture

### Package Structure (`multiphasegalacticwind/`)
- `wind_model.py`: High-level API wrapper
- `core_physics.py`: Core physics functions (ODEs, event functions)
- `cooling.py`: All cooling-related functions and interpolators
- `config.py`: Configuration management with `WindConfig` class
- `constants.py`: Physical constants (separated from config)
- `observables.py`: Velocity distributions and column densities
- `plotting.py`: Publication-quality plotting functions
- `data/Lambda_tab_redshifts.npz`: Cooling function lookup table

### Legacy Structure (`legacy/`)
- `Multiphase_Wind_Evolution.py`: Original single-cloud implementation
- `Multiphase_Wind_Evolution_Multicloud.py`: Multicloud version
- Notebooks for analysis and figure generation

### Key Functions

#### Core Physics (`core_physics.py`)
- `Wind_Evo()`: Main ODE system for supersonic wind evolution
- `Hot_Wind_Evo()`: Hot phase only (adiabatic baseline)
- `setup_cloud_powerlaw_distribution()`: Generate cloud populations
- Event functions: `wind_negative`, `create_cold_wind_event()`, etc.

#### Cooling Module (`cooling.py`)
- `load_cooling_table()`: Load cooling data (called automatically)
- `get_cooling_interpolator(mu, Z, z)`: Get P-ρ interpolator
- `tcool_P(T, P, Z, z, mu)`: Calculate cooling time
- `get_tcool_min_interpolators()`: Minimum cooling time tables

#### Configuration (`config.py`)
- `WindConfig`: All model parameters
- `get_default_config()`: Default configuration

### Migration Guide: Old Code → New API

```python
# OLD (pre-refactoring):
mu_mol = 0.62
k_B = 1.38e-16
# ... globals everywhere ...

# NEW (current):
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.constants import kb, mu_default

# Custom configuration
config = WindConfig(
    mu=0.62,
    f_turb0=0.1,
    drag_coeff=0.5,
    metallicity=0.3
)
model = WindModel(SFR=10.0, config=config)
```

### Physical Model
The code solves coupled ODEs for:
- Hot phase: Euler equations with cooling/heating
- Cold clouds: Mass exchange via turbulent radiative mixing layers (TRMLs)
- Integration uses `scipy.solve_ivp` with event detection for termination conditions

### Important Parameters
- `eta_M`: Hot phase mass loading factor
- `eta_M_cold`: Cold phase mass loading factor  
- `f_turb`: Turbulent mixing efficiency (≈0.1)
- `T_cl`: Cloud temperature (10^4 K)

## Development Notes

### Code Status
- ✅ Clean imports (no side effects)
- ✅ Proper parameter flow (config → functions)
- ✅ Separated concerns (physics, cooling, config)
- ⚠️ Performance needs optimization
- ⚠️ Documentation needs updating
- ❌ No unit tests yet

### Technical Details
- Units: CGS internally, API uses convenient units (Msun/yr, km/s, kpc)
- Cooling table: Lazy-loaded from `data/Lambda_tab_redshifts.npz`
- Parameters: All in `WindConfig` (see `config.py` for full list)
- Integration: Uses `scipy.solve_ivp` with event detection

### Testing Commands
```bash
# Quick import test
python -c "from multiphasegalacticwind import WindModel; print('OK')"

# Test cooling with different mu
python examples/test_cooling_mu.py

# Full integration test (WARNING: may timeout)
python examples/test_refactored.py
```

## 🚨 CRITICAL: Codebase Status & Next Steps

### 📊 Current Status Summary

The package has undergone a major refactoring and is now architecturally clean and well-organized:

✅ **Completed Improvements:**
- Clean module separation (physics, cooling, config, plotting)
- Proper configuration management via WindConfig
- No global variables or import-time code execution
- Fixed parameter flow throughout codebase
- Lazy loading of cooling tables
- Sonic point calculation from physics principles
- Created comprehensive documentation (migration guide, parameter docs)
- Built Jupyter notebook tutorials

⚠️ **Critical Issues:**
1. **Performance Crisis** - Integration can timeout (>2 minutes)
   - Default rtol=1e-8 may be too strict for stiff ODEs
   - Possible cooling interpolator regeneration on each call
   - URGENT: Need profiling to identify bottleneck

2. **Test Script Cleanup Needed** - 9 test files created during development:
   ```
   test_wind_evo_basic.py
   test_wind_evo_integration.py
   test_wind_evo_detailed.py
   test_refactored.py
   test_integrated.py
   test_cooling_mu.py
   test_cooling_func.py
   test_full_integration.py
   test_global_cleanup.py
   ```
   These should be removed or converted to proper unit tests.

3. **Example Scripts Need Review**:
   - `examples/simple_example.py` - May need updating
   - `examples/observational_comparison.py` - Check for old patterns
   - `examples/column_density_example.py` - Verify new API usage

### 🎯 Priority Tasks for Next Session

#### 1. Performance Profiling (URGENT)
```python
# Profile the integration to find bottlenecks
import cProfile
import pstats

model = WindModel(SFR=10.0)
profiler = cProfile.Profile()
profiler.enable()
solution = model.run()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

Potential optimizations:
- Cache cooling interpolators (currently recreated each call?)
- Relax tolerances: rtol=1e-6, atol=1e-8
- Vectorize operations in Wind_Evo
- Use numba JIT compilation for hot loops

#### 2. Clean Up Test Scripts
```bash
# Remove development test files
rm test_*.py

# OR move to proper test directory
mkdir tests
mv test_*.py tests/
# Then convert to pytest format
```

#### 3. Verify Example Scripts
Test each example with the new API:
```bash
python examples/simple_example.py
python examples/basic_example.py
python examples/observational_comparison.py
python examples/column_density_example.py
```

#### 4. Add Missing Functionality
- Progress reporting for long integrations
- Caching for cooling interpolators
- Type hints throughout codebase
- Proper unit tests using pytest

### 📋 Complete Todo List

**High Priority:**
1. ✅ Test Wind_Evo can compute derivatives from ICs
2. ✅ Update README with new WindConfig API
3. ✅ Document all WindConfig parameters
4. ✅ Create migration guide
5. ✅ Fix sonic point calculation from physics
6. ❌ Profile integration to identify performance bottlenecks
7. ❌ Optimize cooling interpolator caching

**Medium Priority:**
8. ❌ Add docstrings to all cooling.py functions
9. ❌ Update all example scripts for new API
10. ❌ Create unit tests for cooling module
11. ❌ Create unit tests for WindConfig
12. ❌ Add progress reporting for long integrations

**Low Priority:**
13. ❌ Add type hints to all functions
14. ✅ Create Jupyter notebook tutorials

### 🔧 Technical Notes

**Performance Investigation Starting Points:**
1. Check cooling interpolator creation frequency
2. Profile `Wind_Evo` function calls
3. Examine event function overhead
4. Test with looser tolerances
5. Check for repeated expensive calculations

**Key Files to Profile:**
- `core_physics.py`: Wind_Evo function
- `cooling.py`: get_cooling_interpolator, tcool_P
- `wind_model.py`: run method integration setup

**Cleanup Checklist:**
- [ ] Remove test_*.py files
- [ ] Verify examples work with new API
- [ ] Remove any remaining print statements
- [ ] Check for unused imports
- [ ] Ensure all notebooks run without errors