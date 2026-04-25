# MultiPhase MultiCloud Galactic Wind Model

A focused Python implementation for steady-state multiphase galactic wind solutions with multiple cloud-mass species.

This code evolves a hot wind coupled to embedded cold clouds through mass, momentum, and energy exchange terms, with tabulated radiative cooling.

## Scientific Reference
Primary paper:
- [Fielding et al. 2022, ApJ, 924, 82](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract)

## Repository Layout
Repository structure:
- `multiphasegalacticwind/` - production code
- `tests/` - regression/physics tests
- `docs/` - parameter and migration documentation
- `examples/` - runnable scripts and notebooks
- `AGENTS.md` - engineering + physics implementation guide
- `README.md` - project overview and quick usage

## Core Modules
- `multiphasegalacticwind/wind_model.py`
  - User API: `WindModel`, `Solution`
  - Handles setup, JAX ODE integration, Jacobian access, and output packaging.
- `multiphasegalacticwind/jax_physics.py`
  - JAX-native multiphase/hot-only RHS functions.
  - JAX-jitted RK4 integrators and state-Jacobian helper for physical diagnostics/inference.
- `multiphasegalacticwind/core_physics.py`
  - ODE right-hand sides:
    - `Wind_Evo` (multiphase)
    - `Hot_Wind_Evo` (hot-only baseline)
  - Cloud mass-bin initialization and compatibility/event helper functions.
- `multiphasegalacticwind/cooling.py`
  - Cooling-table loading and interpolation.
  - Cooling-time utilities (`tcool_P`, `Lambda_P`).
- `multiphasegalacticwind/observables.py`
  - Observable mappings such as `dN/dv` and velocity moments.
- `multiphasegalacticwind/inference.py`
  - MAP + Hessian + HMC inference utilities.
  - Current fitted parameters are (`eta_M`, `eta_M_cold`, `eta_E`).
  - Supported observable modes include raw moments, transformed shape moments, and binned `dN/dv`.
  - Corner-plot and moment-fit visualization helpers.
- `multiphasegalacticwind/config.py`
  - Configurable physics/numerical controls via `WindConfig`.
  - Unknown config keywords raise `ValueError`; documented aliases include `metallicity` and `cooling_factor`.
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
- Numeric `Cooling_Factor` scaling in the wind energy equation.
- Cloud thermal enthalpy using `cs_cl^2 / (gamma - 1)` with `cs_cl^2 = gamma k_B T_cloud / (mu m_p)`.
- Runtime validity guards for unphysical states (negative pressure/density, NaNs).

For implementation equations and units, see `AGENTS.md`.

## Documentation Map
- `AGENTS.md` - canonical engineering and physics guide for future code changes.
- `docs/windconfig_parameters.md` - accepted `WindConfig` parameters, aliases, validation rules, and units.
- `docs/inference_validation_roadmap.md` - landing page for the inference validation roadmap.
- `docs/inference_validation_for_physicists.md` - inference validation guide for physicists, especially inference novices.
- `docs/inference_validation_agent_workplan.md` - implementation workplan for agents building prior predictive checks, synthetic recovery, TRML sensitivity studies, and expanded inference.
- `docs/jax_inference_improvements.md` - detailed JAX inference notes and improvement ideas.
- `docs/h100_gpu_migration_playbook.md` - GPU migration notes for larger inference workloads.

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

# Local Jacobian of RHS wrt state at launch radius (for physical intuition)
jac = model.jacobian_rhs(solution.r[0], solution.sol.y[:, 0])
print(jac.shape)
```

Example post-processing:

```python
v, dN_dv = solution.calculate_column_density_distribution()
moments = solution.calculate_velocity_moments()
```

Inference quick-start (fit to observed `M0, M1, M2`):

```python
import numpy as np
from multiphasegalacticwind.inference import MomentInferenceModel, build_covariance

model = MomentInferenceModel(sfr=20.0, r_star_kpc=0.3, v_circ=150.0)
observed = np.array([6.5e19, 2.9e22, 1.5e25])
cov = build_covariance([0.8e19, 0.4e22, 0.2e25])

fit = model.fit_posterior(
    observed_moments=observed,
    covariance_moments=cov,
    initial_theta=(0.2, 0.2, 1.0),
    sampler="nuts",
    num_chains=4,
    hmc_num_warmup=1000,
    hmc_num_samples=1500,
)
print(fit.map.theta_map)
```

Current inference deliberately fits only `eta_M`, `eta_M_cold`, and `eta_E`. The plan for adding deeper turbulent radiative mixing layer or cloud-wind interaction parameters is documented in `docs/inference_validation_for_physicists.md` and `docs/inference_validation_agent_workplan.md`; the next implementation step is a three-parameter prior predictive atlas, not a broad TRML parameter expansion.

## Running Tests
From repository root:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

The CPU backend is the verified local test path on this Apple Silicon machine. Do not use Metal locally unless explicitly debugging JAX backend behavior.

## Running Examples
From repository root:

```bash
python examples/simple_example.py
python examples/column_density_example.py
python examples/comprehensive_example.py
python examples/config_customization_example.py
python examples/event_diagnostics_example.py
python examples/fit_observational_moments.py --help
python examples/inference_case_study.py --quick
```

## Development Priorities
- Keep implementations concise and explicit.
- Document assumptions and units at API boundaries.
- Prefer physically correct equations over numerically convenient shortcuts.
- Add/adjust tests whenever changing model behavior.
- For inference work, validate priors, synthetic recovery, and posterior predictive behavior before adding new fitted physics parameters.
