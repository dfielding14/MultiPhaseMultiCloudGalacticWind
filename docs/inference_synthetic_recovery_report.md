# Baseline Synthetic Recovery Report

This report summarizes the first CPU-scale synthetic input-recovery run for the current three-parameter inference model. The fitted parameters remain:

- `eta_M`
- `eta_M_cold`
- `eta_E`

No turbulent radiative mixing-layer or cloud-wind parameters were added.

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

The baseline three-parameter inference path is operational, but this pilot does not yet justify expanding the inferred parameter set. The immediate next scientific step should be a production-scale baseline recovery run with:

- more noise realizations per truth case,
- longer HMC or NUTS chains,
- the intended radial range and cloud-species resolution,
- and a narrowed set of truth cases that are valid under the production forward model.

It is reasonable to proceed to the TRML/cloud-parameter sensitivity screen as a forward-model study, because that does not modify the inference internals. It is not yet reasonable to add an inferred TRML parameter such as `A_mix`; the baseline recovery needs stronger evidence that `eta_M`, `eta_M_cold`, and `eta_E` are reliably recoverable first.

