# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase implementing multiphase galactic wind models. It simulates steady-state evolution of galactic winds with hot gas and cold embedded clouds, based on Fielding & Bryan's paper "The Structure of Multiphase Galactic Winds".

## Key Commands

### Running Simulations
```bash
# Single cloud simulation
python Multiphase_Wind_Evolution.py

# Multicloud simulation  
python Multiphase_Wind_Evolution_Multicloud.py

# Interactive analysis
jupyter notebook multicloud.ipynb
jupyter notebook Multiphase_Wind_Fitting_function_for_Classy.ipynb
```

### Dependencies
Install with: `pip install numpy scipy matplotlib cmasher h5py jupyter`

## Architecture

### Core Modules
- `Multiphase_Wind_Evolution.py`: Original single-cloud wind model implementation
- `Multiphase_Wind_Evolution_Multicloud.py`: Extended version supporting power-law cloud mass distributions
- `Lambda_tab_redshifts.npz`: Cooling function lookup table (81×352×15×49 array)

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

- This is research code without formal tests or build system
- Direct Python execution - no setup.py or package structure
- Validation through physical checks and comparison with analytical solutions
- Paper and figures in `Paper/` subdirectory
- Uses cmasher instead of deprecated palettable for colormaps