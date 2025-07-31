# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase implementing multiphase galactic wind models. It simulates steady-state evolution of galactic winds with hot gas and cold embedded clouds, based on Fielding & Bryan's paper "The Structure of Multiphase Galactic Winds".

## CRITICAL: Package Development Status (as of 2024-07-31)

### Current State
The package structure has been created but has several issues preventing it from running:

#### Problems to Fix:
1. **Variable Naming Inconsistencies**
   - `observables.py` imports `mu_mol` but `core_physics.py` defines `mu`
   - `wind_model.py` uses `k_B` but `core_physics.py` defines `kb`

2. **Missing Event Functions**
   - `wind_model.py` references: `T_eq_Tcl`, `v_zero`, `no_clouds`, `T_eq_Tcl_hot`, `v_zero_hot`
   - These don't exist in `core_physics.py`

3. **Function Signature Mismatch**
   - `Wind_Evo(r, state)` takes 2 arguments but called with 3: `Wind_Evo(r, y, params)`

### Fix Strategy:
1. **Phase 1**: Fix variable names (`mu_mol`→`mu`, `k_B`→`kb`)
2. **Phase 2**: Fix Wind_Evo to accept params or pass them differently
3. **Phase 3**: Add missing event functions from legacy code
4. **Phase 4**: Test against legacy code to ensure physics preserved

### Required Tests:
- Unit tests for constants and imports
- Integration test for basic wind solution
- Regression tests comparing to legacy code
- Physics tests (mass/momentum/energy conservation)

### Key Principles:
- **Don't over-engineer** - minimal changes only
- **Preserve exact physics** - units must stay identical
- **Test incrementally** - one fix at a time
- **Use legacy as ground truth**

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
- `core_physics.py`: Physics functions from original code
- `observables.py`: Velocity distributions and column densities
- `plotting.py`: Publication-quality plotting functions
- `data/Lambda_tab_redshifts.npz`: Cooling function lookup table

### Legacy Structure (`legacy/`)
- `Multiphase_Wind_Evolution.py`: Original single-cloud implementation
- `Multiphase_Wind_Evolution_Multicloud.py`: Multicloud version
- Notebooks for analysis and figure generation

### Key Functions
- `Wind_Evo()`: Main ODE system for supersonic wind evolution
- `Hot_Wind_Evo()`: Hot phase only (adiabatic baseline)
- `cooling_function_*()`: Cooling curves with metallicity/redshift dependence
- `setup_cloud_powerlaw_distribution()`: Generate cloud populations
- `dMcl_dt()`, `dvcl_dt()`, `dZcl_dt()`: Cloud evolution equations

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

- Research code being converted to pip-installable package
- Cooling table now loads from package data directory (not current directory)
- Units: CGS internally, but API uses convenient units (Msun/yr, km/s, kpc)
- No formal test suite yet - validation through physical checks
- Paper and figures in `Paper/` subdirectory (excluded from git)