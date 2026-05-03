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
  - MAP + Hessian + HMC/NUTS inference utilities.
  - Default production fitted parameters are (`eta_M`, `eta_M_cold`, `eta_E`).
  - Supported observable modes include raw moments, transformed shape moments, binned `dN/dv`, and diagnostic log-profile likelihoods.
  - Diagnostic coordinate/physics options include soft-capped `eta_E`, loading-ratio coordinates, support-aware `dN/dv` kernels, and restricted expanded models through `expanded_parameters="a_mix"` and `"a_mix_beta_chi"`.
  - Corner-plot, observable-fit, and diagnostic visualization helpers.
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
- `docs/inference_synthetic_recovery_report.md` - current three-parameter exact synthetic recovery status and high-energy caveats.
- `docs/trml_sensitivity_report.md` - one-at-a-time turbulent radiative mixing layer and cloud-wind sensitivity screen.
- `docs/inference_expanded_parameter_decision.md` - decision note selecting restricted `A_mix` as the first expanded-inference experiment.
- `docs/jax_inference_improvements.md` - detailed JAX inference notes and improvement ideas.
- `docs/h100_gpu_migration_playbook.md` - GPU migration notes for larger inference workloads.
- `paper/paper2_inference_validation/paper2_inference_validation.tex` - tracked Paper 2 manuscript source.

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
    initial_theta=(0.2, 0.2, 0.8),
    sampler="nuts",
    num_chains=4,
    hmc_num_warmup=1000,
    hmc_num_samples=1500,
)
print(fit.map.theta_map)
```

Current production inference deliberately fits only `eta_M`, `eta_M_cold`, and `eta_E`. The three-parameter prior predictive atlas, exact synthetic-recovery diagnostics, and one-at-a-time TRML/cloud-wind sensitivity screen are now in place at the current diagnostic level. The main caveat is the high-specific-energy `strong_wings` recovery case: it is useful as a stress test, but it is not a clean production NUTS pass and should not be used for precise claims about `eta_E` near the nominal energy ceiling.

The next inference-validation step is the restricted expanded model
`expanded_parameters="a_mix"`, where `A_mix` is a single effective multiplier on cloud mass exchange. This is intentionally not a broad fit of all turbulent radiative mixing layer closure knobs. The staged `expanded_parameters="a_mix_beta_chi"` mode is diagnostic only and should wait until the `A_mix` prior-predictive and exact-recovery gate passes. See `docs/inference_expanded_parameter_decision.md` and `docs/inference_validation_agent_workplan.md`.

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
python examples/inference_prior_predictive.py --num-samples 20 --no-usetex
python examples/inference_synthetic_recovery.py --help
python examples/trml_sensitivity_screen.py --help
```

## Development Priorities
- Keep implementations concise and explicit.
- Document assumptions and units at API boundaries.
- Prefer physically correct equations over numerically convenient shortcuts.
- Add/adjust tests whenever changing model behavior.
- For inference work, validate priors, synthetic recovery, and posterior predictive behavior before adding new fitted physics parameters.
