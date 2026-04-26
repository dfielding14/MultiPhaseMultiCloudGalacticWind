# Baseline Synthetic Recovery Report

This report summarizes the first CPU-scale synthetic input-recovery run for the current three-parameter inference model. The fitted parameters remain:

- `eta_M`
- `eta_M_cold`
- `eta_E`

No turbulent radiative mixing-layer or cloud-wind parameters were added.

## Status Update

The first recovery run below exposed a real inference-diagnostic problem and should be treated as superseded for scientific decisions. Follow-up exact-data checks found two objective-level issues:

- the reported MAP used the unconstrained-coordinate posterior, including the softplus/sigmoid transform Jacobian, making MAP recovery coordinate dependent;
- the smooth validity barrier charged ordinary valid trajectories and could dominate the likelihood residuals.

The code now reports MAP estimates from the log-parameter posterior while keeping the transform Jacobian in the HMC/NUTS target, and the validity barrier is dormant for positive finite wind states. The synthetic recovery harness also writes per-realization corner plots and observable-fit plots by default, so future recovery reports can inspect the same posterior geometry that previously had to be checked by hand.

Baseline recovery must be rerun with these fixes before proceeding to TRML inference expansion or using the pilot numbers below in Paper 2.

## Corrected Recovery Roadmap

The next phase is a corrected synthetic-recovery campaign, not TRML sensitivity. The goal is to separate machinery bugs, sampler geometry, true parameter degeneracies, and noise-realization scatter.

1. Run exact-observable recovery first.
   Use `--use-truth-observables` so the synthetic observation is the forward model output at the known truth. This should be the first gate because a noiseless recovery failure points to inference machinery, sampler geometry, priors, or identifiability rather than random noise.

2. Tune NUTS on the exact fiducial case.
   Require `chi2` near zero at the MAP, truth inside the posterior intervals, no serious divergences, `Rhat <= 1.05`, and enough effective samples for the corner plot to be interpretable. The short corrected diagnostic improved the MAP but still showed broad `eta_E` and a few NUTS divergences, so production runs should use dense mass-matrix adaptation, longer warmup, and a high target acceptance.

3. Run exact-observable recovery across the valid truth cases.
   Apply the no-noise test to each truth case that produces valid forward-model observables. This tells us whether each truth case is identifiable under the chosen observables and covariance before adding stochastic scatter.

4. Run the noisy recovery sweep only after exact recovery is acceptable.
   Use multiple noise realizations per valid truth case. Summarize MAP relative error, posterior median bias, posterior widths, 68 and 95 percent truth inclusion, sampler diagnostics, and parameter correlations.

5. Inspect the corner plots and observable-fit plots for every case.
   Treat the corner plots as first-class recovery diagnostics. If `eta_E` remains broad or biased while sampler diagnostics are clean, that is likely a real observable degeneracy. If divergences, low ESS, or unstable contours persist, fix sampler/covariance geometry before interpreting the physics.

6. Update this report and the Paper 2 synthetic-recovery section with corrected results.
   The old pilot figures should remain disabled until replaced by corrected figures. Only after corrected baseline recovery is scientifically acceptable should the project return to TRML/cloud-parameter sensitivity work or add any inferred parameter.

## Corrected Fiducial Exact-Data Gate

A first corrected exact-observable fiducial gate was run on April 25, 2026:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/inference_synthetic_recovery.py \
  --truth-case fiducial \
  --observable-set logm0_mean_sigma_skew_kurt \
  --use-truth-observables \
  --num-noise-realizations 1 \
  --num-samples 256 \
  --num-warmup 512 \
  --seed 20260425 \
  --output examples/outputs/inference_synthetic_recovery/fiducial_shape5_exact_nuts_gate_20260425 \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --sampler nuts \
  --num-chains 2 \
  --nuts-chain-method vectorized \
  --map-max-iter 24 \
  --map-num-starts 4 \
  --hmc-step-size 0.02 \
  --hmc-target-accept 0.95 \
  --jax-platform cpu
```

This run used the reduced CPU diagnostic settings from the pilot so the corrected objective could be compared cleanly to the pre-fix behavior.

Results:

| Quantity | Value |
|---|---:|
| Truth `eta_M`, `eta_M_cold`, `eta_E` | 0.200, 0.200, 0.800 |
| MAP `eta_M`, `eta_M_cold`, `eta_E` | 0.195, 0.195, 0.756 |
| MAP relative error | 2.7 percent, 2.5 percent, 5.5 percent |
| MAP `chi2` | 0.0208 |
| NUTS acceptance rate | 0.948 |
| NUTS divergences | 1 |
| Maximum `Rhat` | 1.000 |
| Minimum bulk ESS | 150 |

The truth lies inside all 68 percent and 95 percent marginal posterior intervals. The posterior medians are `(0.184, 0.186, 0.689)`, and the 68 percent intervals are `(0.136, 0.252)`, `(0.152, 0.218)`, and `(0.526, 0.872)` for `eta_M`, `eta_M_cold`, and `eta_E`, respectively. The strongest posterior correlation is `eta_M_cold`--`eta_E` at about +0.52.

Interpretation:

- The corrected MAP and observable-fit machinery pass the noiseless fiducial gate.
- The posterior still leaves `eta_E` broad and biased low in the median, even though the true value is inside the central intervals.
- The single NUTS divergence means this first gate was not yet a clean production setting. The follow-up tuning below removes this divergence without changing the MAP or observable fit.

## Fiducial NUTS Tuning

Two follow-up exact fiducial runs tested dense mass-matrix adaptation with longer warmup and a deeper NUTS tree limit. The better setting was:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/inference_synthetic_recovery.py \
  --truth-case fiducial \
  --observable-set logm0_mean_sigma_skew_kurt \
  --use-truth-observables \
  --num-noise-realizations 1 \
  --num-samples 256 \
  --num-warmup 1024 \
  --seed 20260427 \
  --output examples/outputs/inference_synthetic_recovery/fiducial_shape5_exact_nuts_dense097_20260425 \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --sampler nuts \
  --num-chains 2 \
  --nuts-chain-method vectorized \
  --nuts-dense-mass \
  --nuts-max-tree-depth 12 \
  --disable-progress-bar \
  --map-max-iter 24 \
  --map-num-starts 4 \
  --hmc-step-size 0.01 \
  --hmc-target-accept 0.97 \
  --jax-platform cpu
```

Tuning comparison:

| Run | Dense mass | Target accept | Warmup | Divergences | Max `Rhat` | Min ESS | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| diagonal baseline | no | 0.95 | 512 | 1 | 1.000 | 150 | 0.948 |
| dense conservative | yes | 0.99 | 1024 | 0 | 1.041 | 62 | 0.989 |
| dense selected | yes | 0.97 | 1024 | 0 | 1.013 | 127 | 0.982 |

The selected dense run preserves the corrected exact-data MAP `(0.195, 0.195, 0.756)` with `chi2 = 0.0208`, includes the truth inside all 68 percent and 95 percent marginal intervals, and removes NUTS divergences while keeping acceptable convergence diagnostics. The 68 percent posterior intervals are `(0.140, 0.258)`, `(0.152, 0.213)`, and `(0.526, 0.854)` for `eta_M`, `eta_M_cold`, and `eta_E`. The posterior still leaves `eta_E` broad, so the broad energy-loading constraint should be interpreted as an identifiability feature to monitor in the all-truth-case exact recovery, not as a remaining sampler failure in the fiducial case.

Recommended exact-recovery settings for the next sweep:

- `--sampler nuts`
- `--nuts-dense-mass`
- `--nuts-max-tree-depth 12`
- `--hmc-target-accept 0.97`
- `--hmc-step-size 0.01`
- `--num-warmup 1024`
- at least `--num-samples 256` for the diagnostic sweep, with more samples for final production figures.

## Exact Recovery Across Truth Cases

The tuned dense-NUTS exact-observable sweep was run on April 25, 2026:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/inference_synthetic_recovery.py \
  --truth-case all \
  --observable-set logm0_mean_sigma_skew_kurt \
  --use-truth-observables \
  --num-noise-realizations 1 \
  --num-samples 256 \
  --num-warmup 1024 \
  --seed 20260428 \
  --output examples/outputs/inference_synthetic_recovery/shape5_exact_all_truth_dense097 \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --sampler nuts \
  --num-chains 2 \
  --nuts-chain-method vectorized \
  --nuts-dense-mass \
  --nuts-max-tree-depth 12 \
  --disable-progress-bar \
  --map-max-iter 24 \
  --map-num-starts 4 \
  --hmc-step-size 0.01 \
  --hmc-target-accept 0.97 \
  --jax-platform cpu
```

The run completed in 2036 seconds and wrote corner plots plus observable-fit plots for each successful valid truth case. Three of eight truth cases were invalid before inference because the forward model did not produce valid truth observables in this reduced CPU setup.

Exact-recovery results:

| Case | Truth `eta_M`, `eta_M_cold`, `eta_E` | MAP `eta_M`, `eta_M_cold`, `eta_E` | `chi2` | Div. | Max `Rhat` | Min ESS | 95 percent inclusion | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `fiducial` | 0.20, 0.20, 0.80 | 0.195, 0.195, 0.756 | 0.0208 | 1 | 1.009 | 80.9 | yes, yes, yes | nearly passes, but one divergence |
| `low_eta_m_high_eta_e` | 0.07, 0.12, 0.96 | 0.0685, 0.112, 0.766 | 0.0512 | 16 | 1.008 | 133 | yes, yes, yes | high-energy geometry problem |
| `high_eta_m_low_eta_e` | 0.90, 0.20, 0.25 | invalid | -- | -- | -- | -- | -- | invalid truth observables |
| `low_eta_m_cold` | 0.22, 0.01, 0.80 | 0.214, 0.0102, 0.786 | 0.0376 | 0 | 1.031 | 56.8 | yes, yes, yes | passes at diagnostic depth |
| `high_eta_m_cold` | 0.35, 1.00, 0.85 | invalid | -- | -- | -- | -- | -- | invalid truth observables |
| `near_failure_boundary` | 0.06, 1.50, 0.40 | invalid | -- | -- | -- | -- | -- | invalid truth observables |
| `strong_wings` | 0.10, 0.35, 0.98 | 0.100, 0.326, 0.848 | 0.153 | 223 | 1.014 | 68.8 | yes, yes, no | fails |
| `narrow_profile` | 0.70, 0.05, 0.45 | 0.761, 0.0516, 0.514 | 0.167 | 0 | 1.032 | 112.5 | yes, yes, yes | passes at diagnostic depth |

Conclusion: exact recovery does not pass globally. The fiducial, low-cold-loading, and narrow-profile cases are scientifically usable as diagnostic successes, although the fiducial case still recorded one divergence in the all-truth run. The high-energy cases are the blocker. `low_eta_m_high_eta_e` keeps the truth inside the 95 percent intervals but has 16 divergences, a failed MAP convergence flag, and a low-energy MAP. `strong_wings` has 223 divergences and excludes the true `eta_E = 0.98` from the 95 percent marginal interval. No noisy recovery sweep should be run until this is fixed or explicitly reclassified as an observable/prior limitation.

The objective decomposition shows why the high-energy cases move to lower `eta_E` even with exact observations:

| Case | Point | `chi2` | Prior `chi2` | Barrier | MAP-objective value |
|---|---|---:|---:|---:|---:|
| `low_eta_m_high_eta_e` | truth | 0.000 | 1.188 | 0.000 | 0.594 |
| `low_eta_m_high_eta_e` | MAP | 0.051 | 0.797 | 0.000 | 0.424 |
| `strong_wings` | truth | 0.000 | 0.964 | 0.000 | 0.482 |
| `strong_wings` | MAP | 0.153 | 0.545 | 0.000 | 0.349 |
| `fiducial` | truth | 0.000 | 0.088 | 0.000 | 0.044 |
| `fiducial` | MAP | 0.0208 | 0.0296 | 0.000 | 0.025 |
| `narrow_profile` | truth | 0.000 | 2.745 | 0.000 | 1.373 |
| `narrow_profile` | MAP | 0.167 | 2.319 | 0.000 | 1.243 |

This is not a recurrence of the validity-barrier bug: the smooth barrier is zero at both truth and MAP for these valid cases. The exact-data posterior can prefer a lower-energy point because the shape-five covariance allows sub-sigma shifts in all observables while the current log prior prefers lower `eta_E`. For `strong_wings`, that weak-identifiability/prior effect is compounded by severe NUTS geometry near the high-`eta_E` boundary.

Immediate roadmap:

1. Do not run the noisy recovery campaign yet.
2. Run targeted high-energy exact diagnostics for `low_eta_m_high_eta_e` and `strong_wings`.
3. Separate prior sensitivity from sampler geometry by comparing the default prior to a wider or high-energy-centered `eta_E` prior, first with MAP/objective checks and then with NUTS only if the objective no longer favors the low-energy mode.
4. Stress-test NUTS on the high-energy cases with higher target acceptance, longer chains, and sequential chains if vectorized chains keep diverging.
5. Test whether the observable set is underconstraining high-energy wings by repeating exact high-energy recovery with binned `dN/dv` or an added wing-sensitive summary before changing the production recovery plan.
6. Only after the high-energy exact cases pass, or after the report explicitly narrows the valid truth grid, proceed to noisy recovery.

## High-Energy Prior And Binned-Profile Diagnostics

The first two roadmap diagnostics were run for `low_eta_m_high_eta_e` and `strong_wings`: a shape-five MAP/objective prior-sensitivity grid and an exact binned-`dN/dv` recovery attempt. The MAP-only prior grid used the same reduced CPU forward model as the exact sweep, `--sampler none`, `--map-max-iter 96`, and `--map-num-starts 16`. The binned runs used 32 velocity bins from 0 to 1600 km/s, exact truth observables, dense NUTS, two sequential chains, `target_accept = 0.99`, 1536 warmup steps, and 512 retained samples per chain.

Shape-five prior sensitivity:

| Case | Prior label | `prior_eta_E` | `sigma_log_eta_E` | MAP `eta_E` | MAP `chi2` | MAP prior `chi2` | MAP objective | Max normalized residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `low_eta_m_high_eta_e` | default | 0.70 | 0.45 | 0.766 | 0.0513 | 0.797 | 0.424 | 0.190 |
| `low_eta_m_high_eta_e` | weak default-centered | 0.70 | 1.20 | 0.929 | 0.0160 | 0.729 | 0.373 | 0.083 |
| `low_eta_m_high_eta_e` | high-energy centered | 0.90 | 0.45 | 0.932 | 0.0163 | 0.678 | 0.347 | 0.083 |
| `low_eta_m_high_eta_e` | high-energy weak | 0.90 | 1.20 | 0.999 | 0.0170 | 0.657 | 0.337 | 0.114 |
| `low_eta_m_high_eta_e` | near-flat diagnostic | 0.90 | 3.00 | 0.999 | 0.0170 | 0.651 | 0.334 | 0.114 |
| `strong_wings` | default | 0.70 | 0.45 | 0.848 | 0.153 | 0.545 | 0.349 | 0.250 |
| `strong_wings` | weak default-centered | 0.70 | 1.20 | 0.960 | 0.0328 | 0.409 | 0.221 | 0.112 |
| `strong_wings` | high-energy centered | 0.90 | 0.45 | 0.951 | 0.0359 | 0.357 | 0.196 | 0.122 |
| `strong_wings` | high-energy weak | 0.90 | 1.20 | 0.986 | 0.0284 | 0.343 | 0.186 | 0.106 |
| `strong_wings` | near-flat diagnostic | 0.90 | 3.00 | 0.995 | 0.0290 | 0.337 | 0.183 | 0.120 |

All valid MAP points have zero validity-barrier contribution. The shape-five high-energy bias is therefore not a remaining barrier bug. It is prior-sensitive and weakly identified: small sub-sigma changes in the five observables can move `eta_E` from the default-prior mode to the truth neighborhood, or even to the `eta_E` ceiling for `low_eta_m_high_eta_e` under weak high-energy priors.

Binned-`dN/dv` exact recovery:

| Case | Prior label | MAP `eta_M`, `eta_M_cold`, `eta_E` | `chi2` | Div. | Max `Rhat` | Min ESS | `eta_E` in 95 percent? | MAP `eta_E` relative error | Runtime/status |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| `low_eta_m_high_eta_e` | default | 0.0707, 0.1175, 0.893 | 0.0862 | 0 | 1.006 | 309 | yes | 6.9 percent | 1051 s, passes |
| `low_eta_m_high_eta_e` | high-energy weak | 0.0700, 0.1200, 0.958 | 0.000485 | 0 | 1.022 | 112 | yes | 0.23 percent | 1095 s, passes |
| `strong_wings` | default | -- | -- | -- | -- | -- | -- | -- | exceeded 2.4 h with 16-start MAP, then exceeded 2.0 h with 4-start MAP; terminated |
| `strong_wings` | high-energy weak | -- | -- | -- | -- | -- | -- | -- | exceeded 2.0 h with 4-start MAP; terminated |

The completed binned `low_eta_m_high_eta_e` runs pass the acceptance criteria. This is important: binned `dN/dv` can recover the low-mass high-energy case cleanly, and the high-energy weak prior nearly recovers the exact truth at the MAP. The diagnostic plots are:

- `examples/outputs/inference_synthetic_recovery/high_energy_dndv_exact_20260425/low_eta_m_high_eta_e_default/diagnostic_plots/low_eta_m_high_eta_e_realization_000_corner.png`
- `examples/outputs/inference_synthetic_recovery/high_energy_dndv_exact_20260425/low_eta_m_high_eta_e_default/diagnostic_plots/low_eta_m_high_eta_e_realization_000_observable_fit.png`
- `examples/outputs/inference_synthetic_recovery/high_energy_dndv_exact_20260425/low_eta_m_high_eta_e_high_energy_weak/diagnostic_plots/low_eta_m_high_eta_e_realization_000_corner.png`
- `examples/outputs/inference_synthetic_recovery/high_energy_dndv_exact_20260425/low_eta_m_high_eta_e_high_energy_weak/diagnostic_plots/low_eta_m_high_eta_e_realization_000_observable_fit.png`

The `strong_wings` binned runs did not produce posterior diagnostics under the planned full NUTS settings. This should be treated as an operational sampler/geometry failure, not as a binned-profile recovery success. The result is still useful: binned `dN/dv` improves the low-mass high-energy case, but it does not yet solve the harder near-boundary `strong_wings` case in a practical CPU workflow.

Updated decision:

- Do not proceed to noisy recovery.
- Do not add TRML/cloud parameters.
- Treat the default shape-five high-energy failure as primarily prior sensitivity plus weak summary identifiability.
- Treat `strong_wings` as an unresolved near-boundary geometry case. The next technical step should be reparameterizing the energy-loading direction, for example sampling a specific-energy proxy such as `eta_E / eta_M` or a launch-speed proxy, before spending more CPU time on full binned NUTS.
- If a short-term production recovery is needed before reparameterization, restrict the validated exact-recovery grid to cases that pass the current exact tests and explicitly exclude `strong_wings`/near-ceiling `eta_E`.

## Setup

The main run used `examples/inference_synthetic_recovery.py` with all eight baseline truth cases and the transformed five-component observable set:

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/inference_synthetic_recovery.py \
  --truth-case all \
  --observable-set logm0_mean_sigma_skew_kurt \
  --num-noise-realizations 1 \
  --num-samples 48 \
  --num-warmup 48 \
  --seed 20260425 \
  --output examples/outputs/inference_synthetic_recovery/shape5_all_truth_cases \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --sampler hmc \
  --map-max-iter 8 \
  --map-num-starts 2 \
  --hmc-leapfrog-steps 8 \
  --hmc-target-accept 0.70 \
  --jax-platform cpu
```

This is a CI-scale recovery pilot, not the final production recovery atlas. The model was shortened to `r_max=6 kpc` and 4 cloud species to keep the full truth-case sweep practical on CPU. Each truth case has one noise realization and short HMC chains, so the coverage fractions below should be read as diagnostics of the pipeline and degeneracy structure rather than calibrated frequentist coverage estimates.

Two fiducial comparison pilots were also run:

- `m0_m1_m2`: `examples/outputs/inference_synthetic_recovery/moments3_fiducial_pilot`
- `dndv_binned`: `examples/outputs/inference_synthetic_recovery/dndv_fiducial_pilot`

Generated outputs are intentionally under `examples/outputs/`, which is ignored by version control.

## Truth Cases

The eight truth cases were:

| Case | `eta_M` | `eta_M_cold` | `eta_E` | Status |
|---|---:|---:|---:|---|
| `fiducial` | 0.20 | 0.20 | 0.80 | valid, sampled |
| `low_eta_m_high_eta_e` | 0.07 | 0.12 | 0.96 | valid, sampled |
| `high_eta_m_low_eta_e` | 0.90 | 0.20 | 0.25 | invalid |
| `low_eta_m_cold` | 0.22 | 0.01 | 0.80 | valid, sampled |
| `high_eta_m_cold` | 0.35 | 1.00 | 0.85 | invalid |
| `near_failure_boundary` | 0.06 | 1.50 | 0.40 | invalid |
| `strong_wings` | 0.10 | 0.35 | 0.98 | valid, sampled |
| `narrow_profile` | 0.70 | 0.05 | 0.45 | valid, sampled |

The invalid cases failed at the truth-forward-model stage and were recorded as invalid rows rather than crashing the campaign. In this reduced setup, validity is already informative: cold-loaded or low-energy combinations can leave the physically useful model envelope even before adding noise or inference.

## Noise And Covariance

For the main shape-five run, the observable vector was:

```text
logM0, mean_v, sigma_v, skewness, kurtosis
```

The covariance used a mixed model:

- `logM0`: absolute uncertainty `log1p(noise_fraction)`,
- `mean_v` and `sigma_v`: fractional uncertainties,
- `skewness` and `kurtosis`: absolute floors of 0.20 and 0.40,
- modest fixed correlations among column/velocity/shape terms.

The raw-moment pilot used correlated fractional errors for `M0`, `M1`, and `M2`. The binned `dN/dv` pilot used fractional bin errors with an amplitude floor and AR(1) bin-to-bin correlation.

## Recovery Results

Five of eight shape-five truth cases produced valid truth observables and completed HMC sampling. Among those successful cases:

- 68 percent interval inclusion by parameter: `eta_M` 40 percent, `eta_M_cold` 20 percent, `eta_E` 20 percent.
- 95 percent interval inclusion by parameter: `eta_M` 80 percent, `eta_M_cold` 60 percent, `eta_E` 40 percent.
- Median relative MAP error: `eta_M` 29 percent, `eta_M_cold` 14 percent, `eta_E` 20 percent.
- Median relative posterior width: `eta_M` 37 percent, `eta_M_cold` 14 percent, `eta_E` 16 percent.
- HMC acceptance rates for successful runs ranged from 0.54 to 0.78.
- No HMC divergences were recorded.

The raw recovery rows were:

| Case | MAP `eta_M` | MAP `eta_M_cold` | MAP `eta_E` | `chi2` | 95 percent inclusion |
|---|---:|---:|---:|---:|---|
| `fiducial` | 0.126 | 0.202 | 0.634 | 4.53 | no, yes, yes |
| `low_eta_m_high_eta_e` | 0.086 | 0.107 | 0.787 | 8.82 | yes, yes, no |
| `low_eta_m_cold` | 0.172 | 0.0127 | 0.711 | 6.43 | yes, no, no |
| `strong_wings` | 0.129 | 0.300 | 0.781 | 3.82 | yes, no, no |
| `narrow_profile` | 0.924 | 0.0421 | 0.604 | 5.22 | yes, yes, yes |

The strongest repeated failure mode is that `eta_E` is biased low in several successful truth cases, especially the high-energy truth cases. This could be a real degeneracy, a consequence of the short chains and priors, or a symptom of the reduced radial/cloud resolution used for this pilot.

## Observable Mode Comparison

The fiducial pilot comparison gave:

| Observable mode | MAP `eta_M` | MAP `eta_M_cold` | MAP `eta_E` | `chi2` | Notes |
|---|---:|---:|---:|---:|---|
| `m0_m1_m2` | 0.571 | 0.0713 | 0.602 | 1.12 | Low `chi2`, but large parameter compensation and strong correlations. |
| `logm0_mean_sigma_skew_kurt` | 0.126 | 0.202 | 0.634 | 4.53 | Best cold-loading recovery for the fiducial draw, but low `eta_M` and `eta_E`. |
| `dndv_binned` | 0.391 | 0.101 | 0.588 | 3.34 | No fiducial parameter landed inside the 95 percent intervals in the short pilot. |

These comparisons are too small to choose a final real-data observable mode. They do show that raw moments can fit the noisy observables while allowing substantial parameter tradeoffs, and that binned `dN/dv` needs a more careful covariance and sampler test before it can be trusted.

## Degeneracies

For the successful shape-five runs, the median absolute parameter correlations were:

| Pair | Median correlation | Median absolute correlation |
|---|---:|---:|
| `eta_M` vs `eta_M_cold` | -0.427 | 0.427 |
| `eta_M` vs `eta_E` | +0.274 | 0.274 |
| `eta_M_cold` vs `eta_E` | +0.335 | 0.335 |

The `narrow_profile` case showed the largest single correlation, with `eta_M` and `eta_E` correlated at about 0.93. This is the clearest sign that some truth cases constrain only combinations of hot mass loading and energy loading, not each parameter independently.

## Recommendation

The baseline three-parameter inference path is operational, but exact recovery is not yet scientifically acceptable across the intended truth grid. The high-energy diagnostics show that `eta_E` failures are dominated by prior sensitivity, weak shape-summary identifiability, and unresolved near-boundary sampler geometry for `strong_wings`. Binned `dN/dv` helps the `low_eta_m_high_eta_e` case, but it is not yet a practical solution for `strong_wings`. Do not proceed to noisy recovery or TRML/cloud-parameter sensitivity until the energy-loading direction is reparameterized and the high-energy exact cases are recovered cleanly, or until the validated truth grid is explicitly narrowed with a physical justification.
