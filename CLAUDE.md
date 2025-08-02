# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase implementing multiphase galactic wind models. It simulates steady-state evolution of galactic winds with hot gas and cold embedded clouds, based on Fielding & Bryan's paper "The Structure of Multiphase Galactic Winds".

## Key Commands

### Testing & Development
```bash
# Quick functionality test
python -c "from multiphasegalacticwind import WindModel; model = WindModel(SFR=10.0)"

# Run examples
python examples/simple_example.py
python examples/basic_example.py
python examples/observational_comparison.py
python examples/column_density_example.py

# Profile performance bottlenecks
python -m cProfile -s cumulative examples/basic_example.py | head -30

# Run tests
pytest tests/
```

### Installation
```bash
pip install -e .
pip install numpy scipy matplotlib cmasher h5py jupyter
```

## Architecture

### Core Package Structure (`multiphasegalacticwind/`)

The package follows clean separation of concerns with no global variables:

- **`wind_model.py`** - High-level API wrapper (`WindModel` class)
- **`config.py`** - Configuration management via `WindConfig` class
- **`constants.py`** - Physical constants only (kb, mp, Msun, etc.)
- **`core_physics.py`** - Core physics ODEs and event functions
- **`cooling.py`** - Cooling functions with lazy-loaded interpolators
- **`observables.py`** - Velocity distributions and column densities
- **`plotting.py`** - Publication-quality plotting functions
- **`analysis_helpers.py`** - Physical analysis utilities
- **`data/Lambda_tab_redshifts.npz`** - Wiersma+09 cooling tables

### Key Design Principles

1. **Configuration Flow** - All parameters flow through `WindConfig`:
   ```python
   config = WindConfig(f_turb0=0.2, drag_coeff=0.3)
   model = WindModel(SFR=10.0, config=config)
   ```

2. **No Import Side Effects** - Cooling tables load on first use, not at import

3. **Parameter Tuple** - `Wind_Evo` receives 10-parameter tuple:
   ```
   (v_circ, Ndot_cloud0, T_cloud, injection_radius, injection_power,
    config_dict, r0, Edot_per_Vol, Mdot_per_Vol, Lambda_P_rho)
   ```

4. **Event Factory Pattern** - All events use factory functions:
   ```python
   event = create_supersonic_event(params)
   ```

### Physics Implementation

**Core Integration Loop:**
1. Solve coupled ODEs for hot wind + N cloud species
2. State vector: `[v_wind, rho_wind, P, rhoZ_wind, M_cloud_i, v_cloud_i, Z_cloud_i]`
3. Cloud mass distribution: dN/dM ∝ M^-α
4. Mass exchange via turbulent radiative mixing layers (TRMLs)

**Key Physics Functions:**
- `Wind_Evo()` - Main ODE system (multicloud)
- `Hot_Wind_Evo()` - Hot phase only baseline
- `setup_cloud_powerlaw_distribution()` - Generate cloud populations
- `tcool_P()` - Cooling time from pressure/temperature

**Event Detection:**
- Supersonic/subsonic transitions
- Wind velocity negative
- Cold wind (T_hot → T_cloud)
- All clouds frozen (M < M_min)
- Cloud velocity/density thresholds

## Current Status & Known Issues

### ✅ Completed Improvements
- Clean module separation (achieved 19,000x cooling speedup)
- Proper configuration management with WindConfig
- Fixed parameter flow throughout codebase
- Lazy loading of cooling tables
- Comprehensive documentation and migration guides
- Performance issues resolved - integration runs efficiently

### 🔧 Active Development Tasks

1. **Observable Functions**
   - Fix velocity indexing bugs in observables.py
   - Correct cloud density calculations
   - Ensure multi-species cloud handling

2. **Examples & Documentation**
   - Create comprehensive example with dN/dv plots
   - Convert to Jupyter notebook tutorial
   - Verify all existing examples work correctly

3. **Plotting Verification**
   - Check mass flux unit conversions
   - Verify cloud species visualization

## Migration Guide

### Old Code → New API
```python
# OLD (pre-refactoring):
mu_mol = 0.62
k_B = 1.38e-16
# ... globals everywhere ...

# NEW (current):
from multiphasegalacticwind import WindModel, WindConfig
config = WindConfig(mu=0.62, f_turb0=0.1)
model = WindModel(SFR=10.0, config=config)
```

### Key Parameter Changes
- `mu` now flows from config throughout (no hardcoded 0.62)
- All parameters in `WindConfig` (see `config.py`)
- Cooling interpolator passed in params[9]

## Critical Configuration Parameters

**Performance-Critical:**
- `f_turb0` (~0.1) - Turbulent mixing efficiency
- `rtol/atol` - Integration tolerances
- `N_cloud_species` - Number of cloud mass bins

**Physics-Critical:**
- `eta_M` - Hot phase mass loading
- `eta_M_cold` - Cold phase mass loading  
- `drag_coeff` - Cloud drag coefficient
- `T_cl` - Cloud temperature (10^4 K)

## Development Guidelines

1. **Always profile before optimizing** - Use cProfile to identify actual bottlenecks
2. **Maintain parameter flow** - Everything through WindConfig, no globals
3. **Test with simple cases first** - Single cloud before multicloud
4. **Check units carefully** - CGS internally, convenient units in API

## Legacy Code Reference

Original implementations in `legacy/` for validation:
- `Multiphase_Wind_Evolution.py` - Single cloud
- `Multiphase_Wind_Evolution_Multicloud.py` - Multiple clouds
- Jupyter notebooks for analysis