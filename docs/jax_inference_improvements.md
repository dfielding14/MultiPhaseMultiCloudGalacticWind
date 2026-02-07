# JAX Inference Improvements Plan

## Purpose

This document translates six targeted improvements into concrete, implementation-ready changes for the JAX branch.

Goals:

- preserve physical consistency and defensibility,
- improve robustness of posterior geometry,
- improve end-to-end inference throughput without sacrificing readability.

Scope:

- `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/inference.py`
- `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/jax_physics.py`
- optional new solver helper module for adaptive inference integration.


## Baseline Snapshot

Current baseline in the JAX path:

- parameterization in posterior objective uses `theta = exp(log_theta)` in `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/inference.py:385`;
- MAP is single-start damped-Newton in `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/inference.py:407`;
- invalid RHS state currently returns zero derivatives in `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/jax_physics.py:377`;
- integrator is fixed-grid RK4 scan in `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/jax_physics.py:381`;
- NUTS currently collects `accept_prob`, `num_steps`, `diverging` only in `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/inference.py:698`;
- predictor recomputes radial injection and masks inside each call near `/Users/dbf75/Work/Research/GalacticWindsMultiphaseAnalytic/multiphasegalacticwind/inference.py:308`.


## Improvement 1: Physically Bounded Parameter Transform (`eta_E < 1`)

### Problem

`exp(log_theta)` guarantees positivity but does not impose the physical upper bound `eta_E < 1`, which allows posterior mass in nonphysical regions and can distort HMC geometry.

### Proposed transform

Use unconstrained latent vector `u = [u_M, u_Mcold, u_E]` and transform:

- `eta_M = softplus(u_M) + eps`
- `eta_M_cold = softplus(u_Mcold) + eps`
- `eta_E = eta_E_max * sigmoid(u_E)`, with `eta_E_max = 0.999`

Recommended constants:

- `eps = 1e-10`
- `eta_E_max = 0.999`

### Why this helps

- enforces physical admissibility by construction;
- smooth derivatives for MAP and NUTS;
- avoids hard clipping discontinuities.

### Implementation notes

1. Replace `log_theta` posterior variable with unconstrained `u_theta`.
2. Add two utility functions:
   - `unconstrained_to_theta(u)`
   - `theta_to_unconstrained(theta)` for initialization.
3. Update priors to the unconstrained space.
4. Keep result reporting in physical `theta` space.

### Acceptance criteria

- no sampled/posterior-reported `eta_E >= 1`;
- equal or better NUTS diagnostics (`divergences`, `Rhat`, `ESS`) versus baseline;
- no regression in moment predictions at prior MAP points.


## Improvement 2: Multi-Start MAP Before NUTS

### Problem

Single-start MAP can land in a poor local mode, reducing sampler efficiency and biasing Hessian-based initialization.

### Proposed approach

Run `N_start = 4..8` starts in unconstrained space, then choose best candidate using:

1. finite objective and gradient;
2. positive-definite Hessian check;
3. minimum NLP among valid minima.

Fallback:

- if no PD Hessian, choose lowest NLP finite candidate and inflate diagonal mass.

### Why this helps

- better warm-start for NUTS;
- lower divergence risk;
- more stable Hessian mass preconditioning.

### Implementation notes

1. Add `_fit_map_multistart_from_nlp(...)`.
2. Generate starts from:
   - transformed user initial guess,
   - mild Gaussian jitter around it,
   - optional prior-mean seed.
3. Reuse existing damped-Newton core per start.

### Acceptance criteria

- MAP objective no worse than current single-start (same seed);
- reduction in average divergences and/or shorter effective warmup;
- deterministic behavior under fixed RNG seed.


## Improvement 3: Failure Handling in RHS and Objective

### Problem

Returning all-zero derivatives on invalid states creates flat trajectories that can produce awkward posterior plateaus.

### Proposed approach

Use explicit validity diagnostics and smooth penalties:

1. RHS returns derivatives plus a validity signal.
2. Integrator tracks first invalid index.
3. Objective applies a smooth barrier penalty tied to invalidity and degree of violation.

Suggested barrier ingredients:

- `softplus(-pressure / p_scale)`,
- `softplus(-rho / rho_scale)`,
- `softplus(-v / v_scale)`,
- finite-state penalty for NaN/Inf.

### Why this helps

- preserves gradient information near boundaries;
- improves NUTS geometry near failure surfaces;
- gives physically interpretable diagnostics.

### Implementation notes

1. Introduce a small diagnostics struct (or tuple) from predictor:
   - `valid_flag`,
   - `first_invalid_r`,
   - `barrier_value`.
2. Use barrier in `nlp(...)` instead of binary large penalty only.
3. Keep hard fail fallback for catastrophic NaN propagation.

### Acceptance criteria

- fewer sampler stalls near invalid regions;
- objective remains finite over broader trial domain;
- no change to valid-state physics outputs beyond tolerance.


## Improvement 4: Adaptive Solver Path for Inference (Prototype)

### Problem

Fixed-grid RK4 is predictable and fast, but may overstep sharp regions or overspend in smooth regions.

### Proposed approach

Add optional adaptive backend for inference experiments:

- Diffrax `Tsit5` + `PIDController` with controlled tolerances;
- keep RK4 path as baseline default for regression reproducibility.

### Why this helps

- adaptive local error control in stiff/sharp segments;
- potential speedup by taking fewer steps in smooth zones.

### Implementation notes

1. Create a small integrator abstraction:
   - `integrate_mode = "rk4"` or `"tsit5"`.
2. Restrict adaptive backend to inference module initially.
3. Expose tolerances in one place and document defaults.
4. Compare against RK4 on:
   - moments (`M0`, `M1`, `M2`),
   - runtime,
   - stability rate.

### Acceptance criteria

- moment deltas below agreed tolerance threshold;
- clear speed win on at least one representative case category;
- no degradation in posterior diagnostics.


## Improvement 5: Expand NUTS Diagnostics and Chain Strategy

### Problem

Current diagnostics are helpful but limited; chain execution strategy can still underutilize available hardware.

### Proposed diagnostics

Add collection/reporting of:

- `energy`,
- `potential_energy`,
- tree-depth metrics (or equivalent depth proxy),
- BFMI-like energy checks,
- per-chain divergence counts and step-size summaries.

### Proposed chain strategy

Primary:

- run multiple independent 1-chain workers in parallel processes;
- combine draws post-run for corner/KDE and summary.

Rationale:

- avoids device contention and can improve wall-clock throughput on CPU clusters.

### Implementation notes

1. Extend `extra_fields` in NumPyro MCMC call.
2. Add compact diagnostics summary utility.
3. Add script-level parallel chain launcher.
4. Keep deterministic seed partitioning:
   - `seed_i = base_seed + i * stride`.

### Acceptance criteria

- denser posterior sample sets in same or lower wall time;
- richer diagnostics included in JSON and console summary;
- reproducible chain-level outputs for fixed seed.


## Improvement 6: Precompute Static Predictor Factors

### Problem

Some radial masks and static factors are recomputed per theta evaluation inside jitted predictor logic.

### Proposed approach

Hoist static terms outside per-theta execution:

- radial observation window mask,
- radius powers and geometry factors not depending on theta,
- any static conversion constants and floors.

### Why this helps

- reduces redundant operations per objective call;
- lowers trace complexity;
- improves readability of theta-dependent core.

### Implementation notes

1. Build static arrays once during model initialization.
2. Pass static arrays into predictor closure.
3. Keep only theta-dependent calculations inside `predict_theta_with_valid`.

### Acceptance criteria

- lower per-call predictor latency after JIT warmup;
- unchanged moments to numerical precision;
- cleaner separation between static and dynamic computations.


## Recommended Rollout Order

1. Bounded transform (`eta_E < 1`) and unconstrained variable refactor.
2. Multi-start MAP and Hessian selection.
3. Expanded NUTS diagnostics + parallel 1-chain execution tooling.
4. Static-factor precomputation cleanup.
5. Failure-barrier objective refinement.
6. Adaptive solver prototype and A/B benchmark against RK4.


## Validation and Benchmark Matrix

For each rollout step, run:

1. Physics checks
- finite and physical state trajectories for canonical scenarios;
- moment-level agreement against baseline (within tolerance bands);
- explicit check that `eta_E < 1` always.

2. Inference checks
- MAP convergence rate and Hessian definiteness;
- NUTS divergences, `Rhat`, ESS, acceptance, tree-depth/energy diagnostics.

3. Runtime checks
- cold-start compile time,
- warm-run per-evaluation objective cost,
- end-to-end wall time per case-study scenario.

4. Output quality checks
- corner density quality (KDE/contours),
- stable degeneracy summaries,
- reproducibility under fixed seeds.


## Suggested Milestone Definition

Declare the migration successful when all are true:

- physics constraints are enforced by parameterization;
- no major regressions in moments across benchmark scenarios;
- NUTS diagnostics improve or remain stable while sample density increases;
- runtime per finished posterior decreases or remains acceptable with richer diagnostics;
- code remains concise, documented, and easy to follow.

