# MultiPhase MultiCloud Galactic Wind Model

A focused Python implementation for steady-state multiphase galactic wind solutions with multiple cloud-mass species.

This code evolves a hot wind coupled to embedded cold clouds through mass, momentum, and energy exchange terms, with tabulated radiative cooling.

## Scientific Reference
Primary paper:
- [Fielding et al. 2022, ApJ, 924, 82](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract)

## Repository Layout
This repository is intentionally minimal:
- `multiphasegalacticwind/` - production code
- `tests/` - regression/physics tests
- `AGENTS.md` - engineering + physics implementation guide
- `README.md` - project overview and quick usage

## Core Modules
- `multiphasegalacticwind/wind_model.py`
  - User API: `WindModel`, `Solution`
  - Handles setup, ODE integration, event wiring, and output packaging.
- `multiphasegalacticwind/core_physics.py`
  - ODE right-hand sides:
    - `Wind_Evo` (multiphase)
    - `Hot_Wind_Evo` (hot-only baseline)
  - Cloud mass-bin initialization and event factory functions.
- `multiphasegalacticwind/cooling.py`
  - Cooling-table loading and interpolation.
  - Cooling-time utilities (`tcool_P`, `Lambda_P`).
- `multiphasegalacticwind/observables.py`
  - Observable mappings such as `dN/dv` and velocity moments.
- `multiphasegalacticwind/config.py`
  - Configurable physics/numerical controls via `WindConfig`.
- `multiphasegalacticwind/constants.py`
  - Physical constants and unit conversions (CGS-centric internals).

## Physics Implemented (High Level)
The model solves coupled steady-state equations for:
- Hot phase: velocity, density, pressure, metallicity-density.
- Cold phase (per cloud species): cloud mass, velocity, metallicity.

Key ingredients:
- Isothermal gravitational potential (`Phi ~ v_circ^2 ln r`).
- Cloud injection profile inside an injection radius.
- Pressure-equilibrium cloud closure (`rho_cloud ~ P/(k_B T_cloud)`).
- Turbulent mixing-layer coupling and drag-based momentum exchange.
- Tabulated radiative cooling with pressure/temperature dependent cooling times.
- Event-driven termination for unphysical states (negative pressure/density, NaNs, stalled integration, etc.).

For implementation equations and units, see `AGENTS.md`.

## Units
User-facing API (`WindModel`/`Solution`) mainly uses:
- Distance: kpc
- Velocity: km/s
- Mass: Msun
- Fluxes: Msun/yr

ODE internals are CGS.

## Quick Start
Run from repository root:

```python
from multiphasegalacticwind import WindModel

model = WindModel(
    SFR=10.0,
    v_circ=150.0,
    eta_M=0.1,
    eta_M_cold=1.0,
    eta_E=1.0,
    N_cloud_species=10,
)
solution = model.run()

print(solution.v_at_10kpc)
print(solution.mass_loading_at_10kpc)
```

Example post-processing:

```python
v, dN_dv = solution.calculate_column_density_distribution()
moments = solution.calculate_velocity_moments()
```

## Running Tests
From repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## Development Priorities
- Keep implementations concise and explicit.
- Document assumptions and units at API boundaries.
- Prefer physically correct equations over numerically convenient shortcuts.
- Add/adjust tests whenever changing model behavior.

