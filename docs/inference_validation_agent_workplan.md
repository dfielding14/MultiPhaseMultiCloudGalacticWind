# Inference Validation Agent Workplan

This document is an execution guide for future coding agents working on inference validation in this repository.

It complements [Inference Guide For Physicists](inference_validation_for_physicists.md). The physicist guide explains why each step matters. This workplan defines what to implement, in what order, what can be parallelized, and how to verify the results.

## Operating Rules

Follow the repository contract in `AGENTS.md`.

Do:

- keep durable scripts under `examples/`,
- keep explanatory documents under `docs/`,
- keep tests under `tests/`,
- use CPU JAX on this Apple Silicon machine unless explicitly debugging backends,
- preserve existing user changes in the worktree,
- add tests for behavior changes.

Do not:

- add generated figures or result tables to the repository root,
- create one-off diagnostic scripts in the project root,
- add broad inference abstractions before baseline recovery exists,
- fit many TRML parameters before sensitivity and recovery studies justify them.
- add prose to the Paper 2 TeX source without following `paper/writing_style_guide.md`.

Local test command:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## Version Control And Paper 2 Cadence

Treat each major task or milestone as a version-control boundary.

Required cadence:

- before starting a major task, make sure `git status --short` is understood,
- after each major task, stage the source, tests, docs, and Paper 2 TeX updates that belong to that task,
- do not stage generated outputs, prior-predictive atlases, sampler chains, temporary notebooks, or rendered paper PDFs unless explicitly requested,
- run the relevant focused tests before staging and the CPU full suite before committing behavior changes,
- commit after each completed major task so later agents can bisect the inference program cleanly.

Paper 2 source:

```text
paper/paper2_inference_validation/paper2_inference_validation.tex
```

Every major inference-validation step should update the Paper 2 TeX document in parallel with the code/report deliverable. Early updates can be commented outlines; later updates should replace comments with manuscript prose once the result is stable. Any prose additions must follow `paper/writing_style_guide.md`.

Paper 2 figures:

- put manuscript figures under `paper/paper2_inference_validation/figures/`,
- include a figure in the TeX document only when the surrounding section calls for it,
- do not include every diagnostic plot by default; keep dense checks in reports or appendices unless they advance the paper's argument,
- every included figure must have a clear caption that states what is plotted, what assumptions/generated sample it uses, and why it matters for the paper,
- figure order should follow the logic of the text: establish model validity, then observable ranges, then recovery, then sensitivity, then real-data posterior predictive checks,
- captions and any prose around figures must follow `paper/writing_style_guide.md`,
- do not commit temporary render outputs, draft figure variants, or generated PDFs unless explicitly requested.

## Current Technical Baseline

Current inference entry point:

```text
multiphasegalacticwind/inference.py
```

Current inference class:

```text
MomentInferenceModel
```

Current fitted parameters:

```text
eta_M
eta_M_cold
eta_E
```

Current observable modes:

```text
m0_m1_m2
logm0_mean_sigma_skew_kurt
dndv_binned
```

Fixed but scientifically important parameters currently enter through `WindConfig` or setup arguments:

```text
f_turb0
drag_coeff
Mdot_coefficient
geometric_factor
CoolingAreaChiPower
ColdTurbulenceChiPower
TurbulentVelocityChiPower
Cooling_Factor
cloud_alpha
cloud_mass_range
v_cloud_init
Z_cloud_over_Z_solar
Z_hot_over_Z_solar
```

## Critical Path

Do these in order.

1. Verify forward-model correctness and CPU-only runtime.
2. Define target observables and covariance assumptions.
3. Implement the current 3-parameter prior predictive harness.
4. Run and summarize the current 3-parameter prior predictive atlas.
5. Implement the current 3-parameter synthetic recovery harness.
6. Run and summarize baseline synthetic recovery.
7. Implement TRML/cloud parameter sensitivity scans.
8. Choose one effective added parameter.
9. Generalize inference parameter handling only as much as needed.
10. Run expanded prior predictive and recovery tests.
11. Fit real data and run posterior predictive checks.

Do not skip steps 3 through 7 before expanding the inferred parameter set.

## Parallel Work Streams

After the correctness baseline, these can run in parallel.

| Workstream | Can Start When | Blocks Critical Path? | Output |
|-----------|----------------|------------------------|--------|
| Prior predictive script | correctness baseline passes | yes | sampled model atlas |
| Synthetic noise/covariance design | target observables chosen | yes for recovery | covariance recipes |
| Plotting/report utilities | observable outputs defined | no | reusable figures |
| TRML sensitivity script | forward model stable | yes before expanded inference | ranked parameter leverage |
| Real data preparation | data format known | no | observed vectors and covariances |
| Performance benchmarks | first harness exists | no | recommended sample sizes |
| Documentation updates | anytime | no | reproducible user guide |
| Paper 2 outline/prose | milestone result exists | no | manuscript sections updated alongside validation results |

## Task 0: Verify Correctness Baseline

Purpose:

Make sure future inference work is not built on known physics or runtime bugs.

Required command:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

Required manual checks:

- `Cooling_Factor=0.5`, `1.0`, and `2.0` produce distinct RHS or trajectory behavior.
- Cloud enthalpy in JAX and core paths uses `cs_cl_sq / (gamma - 1)`.
- `WindConfig(cooling_factor=0.0)` maps to `Cooling_Factor=0.0`.
- Unknown `WindConfig` kwargs raise `ValueError`.
- Invalid `cloud_mass_range` values raise before `log10`.
- `cloud_ksi` uses `chi = T_wind / T_cloud`.
- `Cooling_and_Acceleration` accepts `v_circ_kms`.

Acceptance criteria:

- full test suite passes on CPU,
- no Metal backend is used,
- failures are fixed before any inference expansion work.
- Paper 2 source has a commented baseline-correctness note or outline entry if the correction changes the story.

## Task 1: Implement 3-Parameter Prior Predictive Harness

Suggested file:

```text
examples/inference_prior_predictive.py
```

Purpose:

Sample the current 3-parameter prior, run the model, and save observable summaries.

Must support:

```text
--num-samples
--seed
--observable-set
--sfr
--v-circ
--r-star-kpc
--r-max-kpc
--output
--jax-platform
```

Local default:

```text
--jax-platform cpu
```

Sampled parameters:

```text
eta_M
eta_M_cold
eta_E
```

Recommended first priors:

```text
eta_M       log-uniform or log-normal over about 0.03 to 3
eta_M_cold  log-uniform or log-normal over about 0.001 to 10
eta_E       bounded over about 0.05 to 0.999
```

For every sample, record:

```text
sample_id
eta_M
eta_M_cold
eta_E
valid
first_invalid_r_kpc
observable_set
observable vector
M0
M1
M2
logM0
mean_v
sigma_v
skewness
kurtosis
v_at_10kpc
mass_loading_at_10kpc
L_hot_cgs if available
L_interface_cgs if available
```

Output format:

- use `.npz` for arrays,
- optionally write a compact `.csv` summary,
- do not commit generated outputs.

Minimum plots:

- valid/failure map,
- predicted `dN/dv` envelope if using binned observables,
- histograms of `M0`, mean velocity, and velocity width,
- parameter versus observable scatter plots.

Acceptance criteria:

- script runs for at least 20 samples in CI-scale runtime,
- script can run 200 or more samples locally,
- invalid model outputs are recorded rather than crashing the whole run,
- no generated files are written to the repo root by default,
- Paper 2 prior-predictive section is updated with the planned figure slots and main questions.

Testing:

- add a focused smoke test if script functionality is factored into importable helpers,
- otherwise document a manual smoke command.

## Task 2: Build Prior Predictive Report

Suggested file:

```text
docs/inference_prior_predictive_report.md
```

Purpose:

Summarize what the current 3-parameter model can produce before looking at real data.

Report sections:

- setup and priors,
- sample count and runtime,
- valid/failure fraction,
- observable ranges,
- parameter-observable trends,
- example profiles,
- recommended narrowed priors,
- open questions.

Acceptance criteria:

- report states whether real observed systems are inside the model envelope,
- report identifies invalid regions,
- report recommends whether to proceed to recovery,
- Paper 2 prior-predictive section is updated from outline comments toward result prose.

Parallelizable:

- plotting utilities can be built by a separate worker while the sampling script is being finalized.

## Task 3: Implement Synthetic Recovery Harness

Suggested file:

```text
examples/inference_synthetic_recovery.py
```

Purpose:

Generate fake observations from known inputs and test whether inference recovers those inputs.

Must support:

```text
--truth-case
--observable-set
--noise-fraction
--num-noise-realizations
--num-samples
--num-warmup
--seed
--output
--jax-platform
```

Truth cases:

```text
fiducial
low_eta_m_high_eta_e
high_eta_m_low_eta_e
low_eta_m_cold
high_eta_m_cold
near_failure_boundary
strong_wings
narrow_profile
```

For each realization:

1. create truth observables,
2. draw noise,
3. run MAP fit,
4. run posterior sampler if configured,
5. store posterior samples and diagnostics,
6. compute posterior predictive observables.

Metrics to compute:

```text
truth_in_68pct_interval
truth_in_95pct_interval
posterior_mean_bias
posterior_median_bias
posterior_width
MAP_error
chi2
acceptance_rate
num_divergent
r_hat if available
ESS if available
parameter correlation matrix
```

Acceptance criteria:

- works for all three observable modes,
- can run a small smoke case quickly,
- records sampler failures explicitly,
- outputs enough data to make coverage plots,
- Paper 2 synthetic-recovery section lists the truth cases, covariance choices, and planned recovery metrics.

## Task 4: Build Baseline Recovery Report

Suggested file:

```text
docs/inference_synthetic_recovery_report.md
```

Purpose:

Decide whether the current 3-parameter inference is trustworthy enough to extend.

Report sections:

- truth cases,
- noise model,
- observable modes compared,
- coverage results,
- posterior widths,
- bias results,
- degeneracy/correlation summaries,
- posterior predictive examples,
- recommendation.

Acceptance criteria:

- states whether `eta_M`, `eta_M_cold`, and `eta_E` are recoverable,
- identifies which observable mode should be used for real data,
- identifies any systematic biases,
- explicitly says whether to proceed to TRML inference expansion,
- Paper 2 synthetic-recovery section is updated with the result narrative and figure/table placeholders.

## Task 5: Implement TRML Sensitivity Screen

Suggested file:

```text
examples/trml_sensitivity_screen.py
```

Purpose:

Vary fixed cloud-wind and TRML parameters one at a time and rank their observable leverage.

Candidate parameters:

```text
f_turb0
drag_coeff
Mdot_coefficient
geometric_factor
CoolingAreaChiPower
ColdTurbulenceChiPower
TurbulentVelocityChiPower
Cooling_Factor
cloud_alpha
v_cloud_init
Z_cloud_over_Z_solar
Z_hot_over_Z_solar
```

Must support:

```text
--parameter
--values
--fiducial-case
--observable-set
--output
--jax-platform
```

For each scan point, record:

```text
parameter_name
parameter_value
fiducial eta_M
fiducial eta_M_cold
fiducial eta_E
valid
first_invalid_r_kpc
observable vector
derived summaries
```

Sensitivity metrics:

```text
d log M0 / d log p
d mean_v / d log p
d sigma_v / d log p
profile distance / d log p
valid fraction
cloud survival radius shift
cooling luminosity shift
```

Acceptance criteria:

- rank table separates high-leverage from low-leverage parameters,
- degeneracies are described qualitatively,
- recommends one first effective parameter to add,
- Paper 2 microphysics-sensitivity section is updated with the ranked-parameter story.

Parallelizable:

- can run while synthetic recovery report is being drafted,
- does not require changing inference internals.

## Task 6: Choose Effective Expanded Parameter

Purpose:

Make a documented decision before modifying inference internals.

Recommended first candidate:

```text
A_mix
```

Recommended first mapping:

```text
A_mix multiplies Mdot_coefficient
```

Alternative candidates:

```text
drag_coeff
TurbulentVelocityChiPower
Cooling_Factor
cloud_alpha
```

Decision document should state:

- physical interpretation,
- transform and prior,
- expected degeneracies,
- why this parameter is identifiable enough to try,
- why other parameters are deferred.

Suggested file:

```text
docs/inference_expanded_parameter_decision.md
```

Acceptance criteria:

- only one added parameter is chosen for first implementation,
- prior range is specified,
- synthetic recovery plan is specified,
- Paper 2 microphysics or discussion section states why the chosen parameter is physically interpretable and why alternatives are deferred.

## Task 7: Generalize Inference Parameters Minimally

Purpose:

Support the chosen added parameter without over-engineering.

Current hard-coded assumptions:

```text
PARAM_NAMES length 3
theta transforms length 3
prior arrays length 3
initial theta length 3
Hessian/covariance shape 3
build_state_and_params reads theta[0:3]
```

Implementation options:

Option A:

```text
Add a dedicated 4-parameter mode for A_mix.
```

Pros:

- smaller patch,
- easier to review,
- enough for first expanded recovery.

Cons:

- may require later refactor.

Option B:

```text
Create a generic named-parameter system.
```

Pros:

- flexible for future parameters.

Cons:

- larger abstraction,
- easier to get wrong,
- not justified until at least one added parameter passes recovery.

Recommendation:

Use the smallest design that supports the first added parameter cleanly. Do not build a generic framework until the physics program proves it needs one.

Required tests:

- current 3-parameter inference remains unchanged,
- new parameter transform is valid,
- prior shapes validate,
- prediction changes when new parameter changes,
- synthetic smoke recovery runs.

Paper 2 update:

- add an outline comment or draft paragraph explaining the new parameterization and its prior.

## Task 8: Expanded Prior Predictive And Recovery

Purpose:

Validate the first expanded parameter set before real data fitting.

First expanded set:

```text
eta_M
eta_M_cold
eta_E
A_mix
```

Required checks:

- prior predictive atlas for expanded model,
- synthetic recovery for expanded model,
- compare to baseline 3-parameter recovery,
- posterior predictive checks,
- parameter correlation matrix.

Acceptance criteria:

- `A_mix` posterior is narrower than prior in at least some truth cases,
- recovery is not systematically biased,
- baseline parameters remain recoverable,
- posterior predictive checks improve for physically meaningful reasons,
- sampler diagnostics remain acceptable,
- Paper 2 expanded-inference discussion states whether the added parameter is meaningfully constrained.

Do not add `drag_coeff` or chi exponents until this stage passes.

## Task 9: Real Data Fitting

Purpose:

Fit actual observations only after synthetic validation.

Required inputs:

```text
observed vector
covariance matrix
object metadata
chosen observable mode
chosen parameter set
priors
```

Required outputs:

```text
posterior samples
MAP result
posterior predictive bands
residual plots
derived physical quantity summaries
diagnostic report
```

Posterior predictive diagnostics:

- observed `dN/dv` versus model bands,
- residuals versus velocity,
- moments table,
- cloud survival radius distribution,
- mass loading distribution,
- cooling luminosity distribution,
- parameter corner plot,
- degeneracy summary.

Acceptance criteria:

- report states which parameters are data-constrained,
- report identifies prior-dominated parameters,
- report identifies systematic residuals,
- report avoids overinterpreting weakly identified parameters,
- Paper 2 CLASSY and posterior-predictive sections are updated with the real-data fit narrative.

## Task 10: Population-Level Model

Purpose:

Constrain weakly identifiable subgrid parameters across multiple objects.

Do this only after single-object synthetic and real-data workflows are stable.

Possible hierarchy:

```text
object-specific:
  eta_M
  eta_M_cold
  eta_E

shared or population-level:
  A_mix
  drag_coeff
  chi_power_eff
```

Acceptance criteria:

- single-object workflow works,
- data model for multiple objects is documented,
- computational cost is understood,
- synthetic population recovery passes,
- Paper 2 discussion or future-work section is updated to reflect whether population-level inference is part of this paper or deferred.

## Required Deliverables By Milestone

### Milestone 1: Baseline Prior Predictive Atlas

Files:

```text
examples/inference_prior_predictive.py
docs/inference_prior_predictive_report.md
paper/paper2_inference_validation/paper2_inference_validation.tex
```

Must answer:

- What range of observables does the current model produce?
- Where does it fail?
- Do real observations lie inside the model envelope?

### Milestone 2: Baseline Synthetic Recovery

Files:

```text
examples/inference_synthetic_recovery.py
docs/inference_synthetic_recovery_report.md
paper/paper2_inference_validation/paper2_inference_validation.tex
```

Must answer:

- Are `eta_M`, `eta_M_cold`, and `eta_E` recoverable?
- Which observable mode performs best?
- What are the dominant degeneracies?

### Milestone 3: TRML Sensitivity

Files:

```text
examples/trml_sensitivity_screen.py
docs/trml_sensitivity_report.md
paper/paper2_inference_validation/paper2_inference_validation.tex
```

Must answer:

- Which deeper parameters affect observables?
- Which are degenerate?
- Which one should be added first?

### Milestone 4: First Expanded Inference

Files depend on chosen implementation.

Must answer:

- Is the added effective parameter recoverable?
- Does it improve posterior predictive checks?
- Does it preserve baseline parameter recovery?

### Milestone 5: Real Data Fits

Files:

```text
docs/real_data_inference_report.md
```

or a more specific per-project report.

Must answer:

- What do the data constrain?
- What remains prior-dominated?
- Where does the model fail?

## Verification Checklist For Agents

Before final response on any implementation task:

- run focused tests for changed modules,
- run the CPU-only full suite if behavior changed,
- stage and commit after each major validated task when the user requests version-control cleanup,
- report exact test command,
- report exact files changed,
- mention any tests not run,
- do not hide unrelated dirty worktree files,
- do not commit generated outputs unless explicitly requested.

CPU command:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## Common Failure Modes

Watch for:

- posterior equal to prior,
- sampler divergences,
- very low effective sample size,
- strong parameter correlations near absolute value 1,
- inferred parameter stuck at prior boundary,
- many prior samples producing invalid integrations,
- binned `dN/dv` covariance treated as diagonal without justification,
- improved fit caused by unphysical parameter values,
- generated files accidentally placed in the repository root.

## Immediate Next Task

The next implementation task should be:

```text
Run examples/inference_synthetic_recovery.py across the baseline truth cases and build docs/inference_synthetic_recovery_report.md.
```

Do not add new fitted parameters in that task.

Minimum scope:

- run all baseline truth cases for at least the shape-five observable mode,
- compare raw moments, shape-five, and binned `dN/dv` if runtime allows,
- summarize coverage, bias, posterior widths, MAP errors, and sampler diagnostics,
- write the recovery report under `docs/`,
- keep sampler outputs and generated plots out of version control unless explicitly requested.

Only after that report and the TRML sensitivity screen should agents modify inference internals for deeper TRML parameters.
