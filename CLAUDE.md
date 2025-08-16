# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research codebase implementing multiphase galactic wind models with hot gas and cold embedded clouds. Based on Fielding & Bryan's "The Structure of Multiphase Galactic Winds" (ApJ 2024).

## Key Commands

### Quick Testing
```bash
# Basic functionality test
python -c "from multiphasegalacticwind import WindModel; model = WindModel(SFR=10.0); print('Import successful')"

# Run core examples
python examples/simple_example.py
python examples/comprehensive_example.py

# Run Jupyter tutorial
jupyter notebook examples/tutorial_comprehensive.ipynb

# Run tests
pytest tests/ -v

# Profile performance
python -m cProfile -s cumulative examples/simple_example.py | head -30
```

### Installation
```bash
pip install -e .
pip install numpy scipy matplotlib cmasher h5py jupyter pytest
```

## Architecture

### Core Package Structure

```
multiphasegalacticwind/
├── wind_model.py       # High-level API (WindModel class)
├── config.py           # Configuration (WindConfig class)
├── core_physics.py     # ODEs and physics (Wind_Evo, events)
├── cooling.py          # Cooling functions (lazy-loaded)
├── observables.py      # Velocity/column density distributions
├── plotting.py         # Publication plots
├── constants.py        # Physical constants (CGS units)
├── analysis_helpers.py # Analysis utilities
└── data/              # Wiersma+09 cooling tables
```

### Key Design Patterns

1. **Parameter Flow via WindConfig**
   - All physics parameters flow through WindConfig
   - No global variables - clean parameter passing
   - Config dict passed to physics functions

2. **10-Parameter Tuple for Physics**
   ```python
   params = (v_circ, Ndot_cloud0, T_cloud, injection_radius, 
             injection_power, config_dict, r0, Edot_per_Vol, 
             Mdot_per_Vol, Lambda_P_rho)
   ```

3. **Event Detection System**
   - Factory functions create event detectors
   - Events: supersonic/subsonic, negative velocity, cold wind, frozen clouds
   - Events configured based on initial conditions

4. **Lazy Loading**
   - Cooling tables load on first use
   - No import-time side effects
   - Interpolators cached after creation

### Core Physics Implementation

**ODE System (`Wind_Evo` in core_physics.py):**
- State vector: `[v_wind, rho_wind, P, rhoZ_wind, M_cloud_i, v_cloud_i, Z_cloud_i]`
- Solves coupled hot phase + N cloud species
- Cloud distribution: dN/dM ∝ M^(-α)
- Mass exchange through turbulent radiative mixing layers

**Critical Functions:**
- `Wind_Evo()` - Main multicloud ODE system
- `Hot_Wind_Evo()` - Hot-only baseline
- `setup_cloud_powerlaw_distribution()` - Initialize clouds
- `tcool_P()` - Cooling time calculation
- `compute_sonic_point()` - Find sonic radius

## Current Development Status

### Recent Fixes (Completed)
- Fixed observables velocity indexing bugs
- Corrected cloud density calculations  
- Resolved 19,000x cooling performance issue
- Clean module separation achieved

### Active Issues

1. **Observables Module**
   - Verify multi-species cloud handling in `calculate_column_density_by_species()`
   - Check velocity moment calculations
   - Validate column density units [cm^-2/(km/s)]

2. **Examples**
   - `comprehensive_example.py` - Verify dN/dv plots
   - `tutorial_comprehensive.ipynb` - Check all cells run
   - `observational_comparison_notebook.ipynb` - Validate mock observations

3. **Plotting Functions**
   - `plot_column_density_distribution()` - Verify units and scaling
   - Check mass flux conversions in plots
   - Validate cloud species color mapping

## Critical Parameters

### Performance Tuning
```python
# For MCMC/parameter exploration (faster)
model = WindModel(SFR=10.0, rtol=1e-6, atol=1e-8)

# For production runs (accurate)  
model = WindModel(SFR=10.0, rtol=1e-10, atol=1e-12)
```

### Key Physics Parameters
- `f_turb0` (0.1-0.2) - Turbulent mixing efficiency
- `drag_coeff` (0.3-1.0) - Cloud drag coefficient
- `eta_M` / `eta_M_cold` - Mass loading factors
- `N_cloud_species` (5-20) - Cloud mass bins

## Testing Approach

```bash
# Unit tests for individual modules
pytest tests/test_cooling.py -v
pytest tests/test_config.py -v

# Integration test via examples
python examples/simple_example.py
python examples/comprehensive_example.py

# Performance test
python -m cProfile -s cumulative examples/simple_example.py | grep Wind_Evo
```

## Common Issues & Solutions

1. **Long integration times**
   - Reduce `rtol` to 1e-6 for initial runs
   - Decrease `r_max_kpc` to 50
   - Use fewer cloud species (N_cloud_species=5)

2. **Memory issues with cooling tables**
   - Tables lazy-load automatically
   - ~100MB when loaded
   - Cached after first use

3. **Event termination**
   - Check `solution.sol.message` for termination reason
   - Adjust `sonic_point_tolerance` if needed
   - Monitor `v_cloud_min` for low velocity events

## CRITICAL: Development Guidelines

**MUST READ**: See `DEVELOPMENT_GUIDELINES.md` for mandatory development practices.

Key requirements:
- Think critically, push back on poor ideas
- Consider multiple approaches before implementing
- Maximum 3 attempts per issue, then reassess
- Small incremental changes that pass tests
- Simplicity over cleverness