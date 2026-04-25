# Examples

This directory contains runnable examples for the `multiphasegalacticwind` package.

Local JAX examples should use CPU on this Apple Silicon machine unless backend debugging is the task:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu python examples/simple_example.py
```

## Quick Start
- **`simple_example.py`** - Minimal working example
  - Quick test to verify installation
  - Basic model creation and execution
  - Simple plots saved to files

## Research Applications
- **`comprehensive_example.py`** - Production-style research workflow
  - Specific parameters: SFR=20, η_M=0.1, η_M_cold=0.1, 5 cloud masses
  - Full analysis workflow with multiple plots
  - Parameter study varying cold mass loading
  - Saves all plots to `plots/` directory

## Observational Analysis
- **`column_density_example.py`** - Column density calculations
  - Specialized for absorption line observations
  - Shows dN/dv in cm^-2 / (km/s) units
  - Individual cloud species contributions
  - Parameter studies for observational predictions (`plots_column_density/`)

## Configuration and Diagnostics
- **`config_customization_example.py`** - Baseline vs tuned `WindConfig` comparison
  - Demonstrates practical parameter tuning
  - Compares profile-level and observable-level changes (`plots_config/`)

- **`event_diagnostics_example.py`** - Integration/event debugging workflow
  - Runs viable and intentionally non-viable parameter sets
  - Prints event-trigger and termination diagnostics

- **`m82_multiphase_user_script.py`** - Script version of the M82 notebook
  - Mirrors `m82_multiphase_user_notebook.ipynb` workflow
  - Adds CLI controls, timing summary, and optional cProfile mode
  - Supports fast profiling via solver-step knobs

- **`parameter_dependence_sweep.py`** - Key-parameter dependence maps
  - Sweeps `eta_M`, `eta_M_cold`, and `eta_E`
  - Produces 1D trend summaries and 2D slice heatmaps for core observables
  - Saves raw sweep arrays to `parameter_dependence_data.npz`

- **`autodiff_sensitivity_analysis.py`** - Local JAX-autodiff sensitivity analysis
  - Uses Jacobians with respect to (`eta_M`, `eta_M_cold`, `eta_E`) at a fiducial model
  - Produces summary elasticity matrix for `dN/dv` moments (`M0`, `M1`, `M2`), radial sensitivity profiles, and linearization check plots
  - Writes outputs by default to `examples/outputs/autodiff_sensitivity/`
  - Saves Jacobian/profile arrays to `autodiff_sensitivity_data.npz`

- **`inference_prior_predictive.py`** - Three-parameter prior predictive atlas
  - Samples the current (`eta_M`, `eta_M_cold`, `eta_E`) prior and records observable summaries
  - Writes `.npz`, `.csv`, metadata, and paper-style diagnostic figures under `examples/outputs/`
  - Defaults to CPU JAX and the shape-5 observable set

Planned inference-validation scripts are documented in [`docs/inference_validation_agent_workplan.md`](../docs/inference_validation_agent_workplan.md):

- `inference_synthetic_recovery.py` - synthetic input recovery for the current three-parameter inference surface
- `trml_sensitivity_screen.py` - one-at-a-time sensitivity scans for TRML/cloud-wind parameters before expanded inference

- **`fit_observational_moments.py`** - Inference from observed moments
  - Fits (`eta_M`, `eta_M_cold`, `eta_E`) using MAP + Hessian + multi-chain NUTS in log-parameter space
  - Accepts observed (`M0`, `M1`, `M2`) plus full error/correlation specification
  - Saves dense posterior corner plot, moment-fit plot, and posterior sample arrays

- **`m82_publication_inference.py`** - End-to-end publication-style synthetic recovery test
  - M82-like truth setup with heavy MAP + NUTS workflow and wall-time status logging
  - Default likelihood uses transformed observables: `logM0`, mean velocity, dispersion, skewness, kurtosis
  - Optional full-shape mode uses binned `dN/dv` likelihood (`--observable-set dndv_binned`, typically 20-30 bins)

- **`classy_batch_inference.py`** - Batch inference over processed CLASSY galaxies
  - Loads the 43-galaxy processed CLASSY set and builds per-object inference inputs automatically
  - Wires per-object `SFR`, radius-choice (`r_gal_kpc`/`r50_kpc`/`r_star_kpc`), and per-object `v_circ`
  - Supports `m0_m1_m2`, shape-5, and binned `dN/dv` likelihood modes with status logging and per-object summaries

- **`inference_case_study.py`** - SFR/size case-study inference sweep
  - Runs multiple far-ranging galaxy scenarios (dwarf to high-SFR extended systems)
  - Generates per-case corner plots and degeneracy summaries
  - Exports aggregate moment-scaling and correlation heatmap diagnostics

- **`cooling_backend_comparison.py`** - Topaz table-reduction study
  - Runs with the JAX+Topaz backend used by the package
  - Explores Topaz table truncation below 3000 K and uniform downsampling
  - Prints runtime and observable deltas for each variant

## Discretization Study
- **`cloud_species_comparison.py`** - Discretization study for cloud-mass bin count
  - Compares solutions for multiple `N_cloud_species` choices
  - Includes velocity-profile and `dN/dv` comparisons (`plots_species/`)

## Interactive Tutorials

- **`tutorial_comprehensive.ipynb`** - Complete interactive tutorial
  - Step-by-step introduction to the package
  - Detailed explanations of physics
  - Interactive parameter exploration
  - Best starting point for new users

- **`m82_multiphase_user_notebook.ipynb`** - M82-like user-ready setup
  - Top-level editable controls for `eta_M`, `eta_M_cold_tot`, and `eta_E`
  - Fixed M82-like baseline: SFR=20, v_circ=150, 13 cloud bins from 1-1e6 Msun
  - Uses built-in plotting flow to generate standard wind/observable plots

- **`observational_comparison_notebook.ipynb`** - Observatory-focused analysis
  - Mock absorption line profiles
  - M82-like starburst galaxy example
  - Velocity distribution analysis
  - For observers planning observations

## Running the Examples

```bash
# Recommended local JAX backend on this machine
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu

# Python scripts
python examples/simple_example.py
python examples/comprehensive_example.py
python examples/column_density_example.py
python examples/config_customization_example.py
python examples/event_diagnostics_example.py
python examples/cloud_species_comparison.py
python examples/m82_multiphase_user_script.py --skip-plots --profile
python examples/parameter_dependence_sweep.py --quick
python examples/autodiff_sensitivity_analysis.py
python examples/fit_observational_moments.py --help
python examples/m82_publication_inference.py --help
python examples/inference_prior_predictive.py --help
python examples/classy_batch_inference.py --help
python examples/inference_case_study.py --quick
python examples/cooling_backend_comparison.py

# Jupyter notebooks
jupyter notebook examples/tutorial_comprehensive.ipynb
```

Examples save plots to files rather than displaying them interactively to avoid hanging.
