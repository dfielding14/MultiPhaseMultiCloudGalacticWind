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

The baseline three-parameter inference path is operational, but the pilot above was affected by the MAP-objective and validity-barrier bugs described in the status update. The immediate next scientific step should be a corrected baseline recovery run with:

- more noise realizations per truth case,
- longer HMC or NUTS chains,
- the intended radial range and cloud-species resolution,
- and a narrowed set of truth cases that are valid under the production forward model.

Do not proceed to the TRML/cloud-parameter sensitivity screen as the active next task until the corrected synthetic recovery run has been inspected. It is not yet reasonable to add an inferred TRML parameter such as `A_mix`; the baseline recovery needs stronger evidence that `eta_M`, `eta_M_cold`, and `eta_E` are reliably recoverable first.
