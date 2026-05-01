# Baseline Synthetic Recovery Report

This report summarizes the first CPU-scale synthetic input-recovery run for the current three-parameter inference model. The fitted parameters remain:

- `eta_M`
- `eta_M_cold`
- `eta_E`

No turbulent radiative mixing-layer or cloud-wind parameters were added.

## Status Update

As of April 28, 2026, the synthetic-recovery campaign has reached a useful stopping point. The initial recovery run below is still superseded for scientific decisions, but the follow-up work has now separated three distinct issues:

- real machinery bugs, which were fixed;
- stalled-wind cliffs, which were converted into finite low-probability model evaluations;
- residual high-energy posterior geometry, which remains difficult but is now physically understood.

The code now reports MAP estimates from the log-parameter posterior while keeping the transform Jacobian in the HMC/NUTS target. The validity barrier is dormant for positive finite wind states. The default inference policy treats stalled winds as finite diagnostics rather than hidden hard cliffs. The synthetic-recovery harness writes corner plots, observable-fit plots, NUTS fields, posterior components, and divergent-sample diagnostics.

The remaining `strong_wings` problem is not a known code bug. It is a thin high-specific-energy ridge coupled to support-edge log-profile bins. In the best observed-window diagnostic, the truth is recovered at the MAP and lies inside the 95 percent intervals, but the short NUTS gate still has a few divergences and low-to-marginal ESS. Longer warmup, higher target acceptance, MAP whitening, support-aware kernels, and low-signal censoring each helped some part of the problem but did not produce a robust formal pass.

Project decision: stop trying to force `strong_wings` through a clean short-NUTS gate before every other milestone. Do not use the current results for noisy high-energy recovery claims or precise production claims about `eta_E` near the energy ceiling. Do use them as the best current baseline diagnostic, with the explicit caveat that high-energy `eta_E` is weakly identified and geometry-sensitive. The next overall project task is the one-at-a-time TRML/cloud microphysics sensitivity screen, not another round of `strong_wings` sampler tuning.

## Current Recovery Roadmap

The recovery status now supports a narrowed and caveated path forward.

1. Keep the exact synthetic-recovery results as a baseline diagnostic, not as a global production pass.
   Fiducial, low-cold-loading, narrow-profile, and binned `low_eta_m_high_eta_e` tests give useful recovery checks. `strong_wings` remains a stress case for high-specific-energy geometry.

2. Do not run the noisy synthetic recovery campaign yet.
   Noise would mix random-realization scatter with an already-known high-energy geometry limitation. Noisy recovery should wait until the observational likelihood and final parameterization are chosen.

3. If `strong_wings` is revisited, start from the best current diagnostic setup.
   Use observed-window `log_dndv_binned`, `energy_coordinate = loading_ratios`, `eta_e_parameterization = softcap`, `failure_policy = stalled_wind`, MAP whitening, and surgical low-signal censoring. A more physical next fix would be a launch-speed or terminal-speed coordinate, not another generic NUTS tuning sweep.

4. Proceed to TRML/cloud sensitivity as a forward-model sensitivity screen.
   This does not add fitted parameters and does not require claiming that high-energy `eta_E` recovery is solved. It asks which fixed microphysics parameters have observable leverage and which are degenerate, so the later expanded-inference step can be chosen deliberately.

5. Carry the caveat into Paper 2.
   The paper should state that baseline inference is operational and informative, but high-energy energy loading remains the limiting recovery direction. The next validation stage is sensitivity ranking, not production high-energy posterior claims.

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

## `eta_E` Soft-Cap Diagnostic

An optional diagnostic inference mode, `eta_e_parameterization = softcap`, now replaces the hard bounded transform for `eta_E` with a positive softplus transform and adds a smooth penalty above the nominal `eta_E = 1` energy budget. The default production mode remains bounded below 1. The synthetic-recovery harness records `posterior_prob_eta_E_gt_1` in CSV and NPZ outputs.

The soft-cap penalty used for this diagnostic is:

```text
excess = transition * softplus((eta_E - center) / transition)
penalty = 0.5 * (excess / sigma)^2
```

with `center = 1.0`, `sigma = 0.10`, and `transition = 0.01`.

Full planned CPU/NUTS runs for exact `strong_wings` shape-five recovery used two sequential dense-NUTS chains, 1536 warmup steps, 512 samples, `target_accept = 0.99`, and the reduced CPU forward-model settings. Both the default-prior and high-energy-weak-prior runs exceeded two hours without writing result files and were terminated. The corresponding binned `dN/dv` full runs were not restarted because the earlier bounded binned runs already showed multi-hour non-completion for `strong_wings`.

Short sanity runs were then used only to verify the new soft-cap mode, output fields, and qualitative posterior behavior:

| Observable | Prior label | MAP `eta_E` | `chi2` | Div. | Max `Rhat` | Min ESS | `P(eta_E > 1)` | `eta_E` in 95 percent? | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| shape-five | default | 0.848 | 0.153 | 59 | 1.073 | 19.6 | 0.055 | yes | 178 s |
| shape-five | high-energy weak | 0.982 | 0.0285 | 71 | 1.002 | 30.1 | 0.125 | yes | 172 s |
| binned `dN/dv` | default | 0.972 | 0.0192 | 21 | 1.130 | 12.8 | 0.266 | yes | 161 s |
| binned `dN/dv` | high-energy weak | 0.978 | 0.00213 | 12 | 1.000 | 31.9 | 0.266 | yes | 167 s |

These short runs are not acceptance-quality posterior samples: all four have divergences and low ESS, and the one-chain diagnostics are not sufficient for final coverage claims. They do show that the soft-cap mode can place the MAP near the true `eta_E = 0.98` for `strong_wings`, especially for binned `dN/dv`, and that the posterior has non-negligible mass above the nominal energy budget (`P(eta_E > 1) ~= 0.06-0.27` in the short runs). Therefore the hard ceiling was hiding part of the near-boundary geometry, but simply replacing it with a soft cap does not make the current CPU NUTS workflow production-ready.

Updated soft-cap interpretation:

- `eta_E_softcap` should remain diagnostic, not the default production model.
- If later GPU or reparameterized runs confirm substantial `P(eta_E > 1)`, then `eta_E` must be described as an effective energy-loading parameter rather than a literal SN-budget efficiency.
- The next technical step is still a better energy-direction parameterization, such as specific energy `eta_E / eta_M` or a launch-speed proxy, before repeating full binned recovery.

## Specific-Energy Ratio Diagnostic

An additional opt-in diagnostic coordinate, `energy_coordinate = eta_e_over_eta_m`, now samples the positive ratio

```text
q_E = eta_E / eta_M
```

and derives the physical energy loading as `eta_E = eta_M * q_E`. Priors and public outputs remain defined on the physical parameters `(eta_M, eta_M_cold, eta_E)`. This mode is allowed only with `eta_e_parameterization = softcap`; the default production inference path remains the bounded direct-`eta_E` coordinate.

Exact `strong_wings` recovery was rerun with reduced CPU settings, truth observables, softcap, and the ratio coordinate. The truth is `eta_M = 0.10`, `eta_M_cold = 0.35`, `eta_E = 0.98`, and `eta_E / eta_M = 9.8`.

MAP-only results:

| Observable | Prior label | MAP `eta_M` | MAP `eta_M_cold` | MAP `eta_E` | MAP `eta_E/eta_M` | `chi2` | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| shape-five | default | 0.100 | 0.326 | 0.848 | 8.45 | 0.153 | 504 s |
| shape-five | high-energy weak | 0.110 | 0.346 | 0.982 | 8.96 | 0.0285 | 501 s |
| binned `dN/dv` | default | 0.101 | 0.347 | 0.972 | 9.63 | 0.0192 | 533 s |
| binned `dN/dv` | high-energy weak | 0.100 | 0.349 | 0.978 | 9.74 | 0.00213 | 540 s |

Short one-chain NUTS results used 256 warmup steps, 128 posterior samples, dense mass adaptation, `target_accept = 0.90`, and `max_tree_depth = 8`:

| Observable | Prior label | MAP `eta_E` | MAP `eta_E/eta_M` | Div. | Max `Rhat` | Min ESS | `P(eta_E > 1)` | Truth in 95 percent for `eta_E` and ratio? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| shape-five | default | 0.848 | 8.45 | 60 | 1.088 | 18.6 | 0.070 | yes / yes |
| shape-five | high-energy weak | 0.982 | 8.96 | 46 | 1.018 | 39.4 | 0.109 | yes / yes |
| binned `dN/dv` | default | 0.972 | 9.63 | 36 | 1.022 | 36.6 | 0.188 | yes / yes |
| binned `dN/dv` | high-energy weak | 0.978 | 9.74 | 47 | 1.061 | 20.2 | 0.211 | yes / yes |

The ratio coordinate improves the optimizer behavior, especially for binned `dN/dv`: the exact binned MAP lands within 1 percent of the true `eta_E` under both priors and within 2 percent of the true ratio. However, the short NUTS chains remain sampler-broken. All four short runs have many divergences and low ESS, so the full two-chain NUTS stage was not launched under the pre-set gate. The current conclusion is therefore:

- Coordinate choice matters: the specific-energy coordinate plus binned `dN/dv` can find the high-energy solution at MAP.
- Coordinate choice alone does not solve the posterior geometry: NUTS still sees difficult curvature or boundary structure near the high-energy ridge.
- Do not proceed to noisy recovery or production high-energy claims.
- Next technical work should reparameterize closer to a launch-speed or specific-energy observable, reduce the boundary geometry further, or use a more specialized sampler/metric diagnostic before another expensive recovery sweep.

Generated diagnostic outputs are under `examples/outputs/inference_synthetic_recovery/specific_energy_ratio_strong_wings/`. Useful plots include:

- `short_dndv_default/diagnostic_plots/strong_wings_realization_000_corner.png`
- `short_dndv_default/diagnostic_plots/strong_wings_realization_000_energy_coordinate_corner.png`
- `short_dndv_default/diagnostic_plots/strong_wings_realization_000_observable_fit.png`
- `short_dndv_high_energy_weak/diagnostic_plots/strong_wings_realization_000_corner.png`
- `short_dndv_high_energy_weak/diagnostic_plots/strong_wings_realization_000_energy_coordinate_corner.png`
- `short_dndv_high_energy_weak/diagnostic_plots/strong_wings_realization_000_observable_fit.png`

## Short-NUTS Geometry Repair

The next diagnostic made NUTS failures inspectable and added `nuts_coordinate = map_whitened`, which samples a local MAP-whitened coordinate `z` and maps it back to the native unconstrained coordinate before each posterior evaluation. Public outputs remain physical. The harness now writes per-sample NUTS diagnostics (`diverging`, `accept_prob`, `num_steps`, `energy`, `potential_energy`, native unconstrained coordinates, and active NUTS coordinates), divergent-sample overlay corner plots, and posterior objective components at truth, MAP, and samples.

Exact `strong_wings` binned `dN/dv` was rerun with 512 warmup steps, 256 posterior samples, dense mass adaptation, `target_accept = 0.99`, and `max_tree_depth = 12`.

| Prior label | NUTS coordinate | MAP `eta_E` | MAP `eta_E/eta_M` | Div. | Tree-depth hits | Max `Rhat` | Min ESS | `P(eta_E > 1)` | Truth in 95 percent for `eta_E` and ratio? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| default | native | 0.972 | 9.63 | 53 | 0 | 1.007 | 42.8 | 0.273 | yes / yes |
| high-energy weak | native | 0.978 | 9.74 | 44 | 0 | 1.008 | 105 | 0.207 | yes / yes |
| default | MAP-whitened | 0.972 | 9.63 | 34 | 0 | 1.027 | 118 | 0.195 | yes / yes |
| high-energy weak | MAP-whitened | 0.978 | 9.74 | 58 | 0 | 0.997 | 56.9 | 0.277 | yes / yes |

Posterior component checks show that the exact truth and MAP are both valid and have zero validity barrier. The binned likelihood is excellent: truth has `chi2 ~= 0`, while the MAP has `chi2 = 0.0192` under the default prior and `chi2 = 0.00213` under the high-energy weak prior. The soft-cap penalty is tiny at both truth and MAP (`< 1e-4`). Therefore this is not a validity-barrier failure, not a tree-depth saturation failure, and not a failure to find the high-energy optimum.

The short gate still fails because divergences remain common in every run. MAP whitening improves ESS and reduces divergences for the default prior, but it does not make the high-energy weak run acceptable and does not remove the geometry problem. Divergent samples are not explained solely by exceeding `eta_E = 1`: in the high-energy weak native run, divergent and non-divergent samples have nearly the same `eta_E > 1` fraction, while in the MAP-whitened runs divergences are somewhat enriched in the above-budget tail. The current evidence points to a curved high-energy/specific-energy ridge with residual soft-cap interaction, not just poor local covariance scaling.

Decision:

- Do not launch full NUTS.
- Do not proceed to noisy recovery.
- Do not treat MAP-whitened NUTS as a sufficient fix.
- Next technical step should be a stronger geometry change: either a launch-speed/specific-energy physical proxy that aligns directly with wind speed, or a gentler effective-energy prior/soft-cap diagnostic that removes the sharp near-budget curvature before another NUTS sweep.

Generated outputs are under `examples/outputs/inference_synthetic_recovery/short_nuts_geometry_fix_strong_wings/`. The most useful plots are:

- `native_high_energy_weak/diagnostic_plots/strong_wings_realization_000_divergent_overlay_corner.png`
- `native_high_energy_weak/diagnostic_plots/strong_wings_realization_000_energy_coordinate_divergent_overlay_corner.png`
- `map_whitened_high_energy_weak/diagnostic_plots/strong_wings_realization_000_divergent_overlay_corner.png`
- `map_whitened_high_energy_weak/diagnostic_plots/strong_wings_realization_000_energy_coordinate_divergent_overlay_corner.png`

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

The baseline three-parameter inference path is operational, but it is only partially validated. Exact recovery is scientifically useful for the fiducial, low-cold-loading, narrow-profile, and binned `low_eta_m_high_eta_e` cases. It is not a clean production pass across the full truth grid because `strong_wings` remains sampler-sensitive near the high-specific-energy ridge.

The practical recommendation is to narrow the claim rather than keep tuning indefinitely. Do not run noisy recovery or make production claims for high-energy `eta_E` near the ceiling from the current `strong_wings` diagnostics. Do proceed to TRML/cloud microphysics sensitivity as a forward-model ranking exercise, because that step does not add fitted parameters or require the high-energy posterior to be production-ready. Any later expanded inference should carry this caveat and should revisit the high-energy geometry with a launch-speed or terminal-speed coordinate if precise high-energy recovery becomes central to the paper.

## Stalled-Wind Policy Update

The inference model now treats stalled or finite nonphysical hot-wind trajectories as finite model evaluations by default via `failure_policy = stalled_wind`. A finite failed solution is no longer assigned the legacy hard invalid cliff solely because the hot phase crosses a failure condition before the observable radius. Instead, the predictor attenuates observables to the finite/alive prefix and reports stalled-trajectory diagnostics, including `trajectory_status_code`, `soft_reach_radius_kpc`, `min_hot_velocity_kms`, `stall_penalty`, and `numerical_failure_penalty`.

This change is intended to convert the strongest `strong_wings` sampler failure from a numerical cliff into an inferential statement: parameter values that choke before the observable radius are allowed, but receive low posterior probability. The legacy behavior remains available as `failure_policy = hard_invalid` for comparisons. The next validation gate is to rerun the exact `strong_wings` binned `dN/dv`, soft-cap, ratio-coordinate, MAP-whitened short NUTS test and reconstruct the worst transition again. Passing requires no catastrophic Hamiltonian spike from stalled winds, zero or inspectably rare divergences, and unchanged recovery for valid high-energy solutions.

Initial validation of the exact saved divergent leaf now reproduces a finite stalled-wind evaluation at `r = 0.755 kpc`: `trajectory_status_code = 1`, finite observables/components, `stall_penalty > 0`, and no hard-invalid or numerical-failure penalty under the default policy. The legacy `hard_invalid` policy still assigns the `1e6` hard penalty to the same point.

The exact `strong_wings` binned `dN/dv`, soft-cap, ratio-coordinate, MAP-whitened short NUTS gate was rerun in `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_short_nuts/`. The MAP remains close to truth (`eta_E = 0.9776` for truth `0.98`; `eta_E/eta_M = 9.741` for truth `9.8`), and truth is inside the 95 percent intervals for all three physical parameters and for `eta_E/eta_M`. However, the short gate still fails sampler acceptance with `57/256` divergent samples and minimum bulk ESS `84.5`. All posterior samples, including divergent samples, have `trajectory_status_code = 0`, `hard_invalid_penalty = 0`, `stall_penalty = 0`, and `numerical_failure_penalty = 0`, so the remaining blocker should be treated as posterior geometry rather than a hidden stalled-wind cliff.

Follow-up divergent-sample diagnostics are in `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_short_nuts/divergent_sample_diagnostics/`. Divergent samples are valid wind solutions and differ only mildly from nondivergent samples in physical coordinates: median `eta_M` is lower (`0.0985` vs `0.1011`), median `eta_E/eta_M` is higher (`9.93` vs `9.66`), and median launch speed is higher (`1580` vs `1558 km/s`). The largest separation is sampler-side accept probability, not failed-wind diagnostics or observable residuals. This supports the interpretation that divergences live along the curved high-specific-energy ridge rather than at a model-failure boundary.

More aggressive linear-profile NUTS tuning (`target_accept = 0.995`, `step_size = 0.01`, `warmup = 1024`, `max_tree_depth = 14`) was run in `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_short_nuts_ta0995_step001/`. It reduced divergences only from `57/256` to `45/256`, lowered minimum bulk ESS from `84.5` to `55.2`, and increased runtime to about `31 min`. This is not a viable fix by itself.

A diagnostic log-profile likelihood was then added as `observable_set = log_dndv_binned`, using the same binned `dN/dv` forward model but fitting `log(dN/dv)` with a correlated Gaussian covariance. The short MAP-whitened log-profile gate in `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_log_dndv_short_nuts/` reduced divergences to `7/256` and gave a near-exact MAP (`eta_E = 0.9797`, `eta_E/eta_M = 9.791`) with truth inside all 95 percent intervals, but minimum bulk ESS remained low at `18.1`. The aggressive log-profile run in `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_log_dndv_ta0995_step001/` further reduced divergences to `4/256`, with no stalled/numerical/hard-invalid penalties and truth still inside all 95 percent intervals, but minimum bulk ESS remained low at `23.2`. This points to log-profile likelihood as the right direction, but the short gate is still not clean enough to proceed to full/noisy recovery.

The four divergent samples from the aggressive log-profile run were then reconstructed as full wind trajectories and compared to nearest non-divergent samples. The physical tracks are nearly indistinguishable: hot velocity, Mach number, hot temperature, cloud velocities, and cloud survival fractions overlap to sub-km/s or sub-percent levels. The largest visible difference is not a stalled wind or a nonphysical state, but the highest-velocity log-profile residual flipping by roughly one to two sigma between nearly identical tracks. Useful figures are:

- `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_log_dndv_ta0995_step001/divergent_physical_profiles/physical_profiles_div_207_vs_nondiv_58.png`
- `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_log_dndv_ta0995_step001/divergent_physical_profiles/divergent_cloud_survival_and_residuals.png`

Because the divergent tracks are valid and physically smooth, an additional diagnostic coordinate was added as `energy_coordinate = loading_ratios`. It samples `(eta_M, eta_M_cold / eta_M, eta_E / eta_M)` while preserving priors and public outputs on physical `(eta_M, eta_M_cold, eta_E)`. This is motivated by the divergent samples being enriched at lower `eta_M`, higher cold/hot loading, and higher specific energy.

The exact `strong_wings` log-profile, soft-cap, MAP-whitened, loading-ratio diagnostic was run in `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_log_dndv_loading_ratios_ta0995_step001/`. Three exact repeats all recovered the truth and had no stalled/numerical penalties:

| realization | divergences | max-tree hits | max Rhat | min bulk ESS | `P(eta_E > 1)` | truth in 95 pct for physical params | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 4/256 | 0 | 1.039 | 17.9 | 0.004 | yes | fails ESS/divergence gate |
| 1 | 1/256 | 0 | 1.022 | 28.9 | 0.016 | yes | fails ESS gate |
| 2 | 2/256 | 0 | 1.027 | 68.7 | 0.004 | yes | short-gate borderline pass |

This is a partial improvement, not a production pass. The coordinate makes the physical ridge clearer and one repeat meets the short one-chain divergence/ESS threshold, but the result is not robust across repeats. Divergent samples in loading-ratio space remain concentrated toward lower `eta_M`, higher `eta_M_cold/eta_M`, higher `eta_E/eta_M`, and slightly stronger soft-cap contribution. The next fix should therefore target the ridge geometry itself, not stalled-wind handling. Candidate next steps are a prior/coordinate on launch-speed or cloud terminal-speed proxy, or a likelihood/covariance refinement that reduces the leverage of the final high-velocity bins without discarding them.

Follow-up inspection showed why the highest-velocity log-profile bins were unstable. In the aggressive log-profile run, sample 70 and nearby non-divergent sample 42 have visually indistinguishable full radial tracks, but the fastest cloud terminal velocity differs by only `0.257 km/s`. Because the binned observable used a Gaussian velocity kernel with `sigma = 25 km/s`, bins at `1525-1575 km/s` were measuring the far Gaussian tail of material whose fastest cloud speed was only about `1231 km/s`. The predicted log-profile change from the terminal-velocity shift alone matches the actual top-bin change to within the log-error scale. The diagnostic figure is:

- `examples/outputs/inference_synthetic_recovery/stalled_wind_policy_strong_wings_log_dndv_ta0995_step001/divergent_physical_profiles/top_bin_flip_pair70_vs_42.png`

Two diagnostic remedies were then tested. First, the inference model gained opt-in support-aware kernels for binned `dN/dv`: `dndv_kernel = truncated_gaussian` and `dndv_kernel = compact_cosine`, with default `gaussian` preserved. The wide `0-1600 km/s` exact `strong_wings` run using `truncated_gaussian` recovered the truth but did not improve sampler geometry: it had `13/256` divergences, max `Rhat = 1.002`, and min bulk ESS `51.0`. The hard cutoff also produced 11 floored log-profile bins, so this is not a good production likelihood by itself. The output is:

- `examples/outputs/inference_synthetic_recovery/strong_wings_log_dndv_loading_ratios_truncated_kernel_wide_short_nuts/`

Second, an observed-window diagnostic used the default Gaussian kernel but restricted the log-profile to `100-600 km/s`, closer to realistic observed velocity coverage. This removed most of the artificial far-tail leverage and still recovered the truth, but it broadened the energy posterior and did not fully pass the one-chain short gate: `5/256` divergences, max `Rhat = 1.051`, min bulk ESS `21.9`, and `P(eta_E > 1) = 0.398`. The MAP remained close to truth, `eta_E = 0.9775`, and truth stayed inside the 95 percent intervals for all physical parameters and `eta_E/eta_M`. The output is:

- `examples/outputs/inference_synthetic_recovery/strong_wings_log_dndv_loading_ratios_observed_window_100_600_short_nuts/`

Interpretation: the very-high-velocity Gaussian tail was a real artificial stress test, but removing it exposes the observational limitation: a `100-600 km/s` profile is much less decisive about high-energy loading. The next production-facing direction should be an explicitly observational likelihood: fixed observed velocity window, realistic masks/upper limits for unobserved velocities, and a tail treatment that does not assign high precision to kernel leakage. The support-aware hard cutoff is useful diagnostically, but the first production candidate should be the observed-window likelihood with a defensible covariance/model-error term, not the wide hard-truncated profile.

The `100-600 km/s` observed-window divergences were then inspected sample-by-sample in:

- `examples/outputs/inference_synthetic_recovery/strong_wings_log_dndv_loading_ratios_observed_window_100_600_short_nuts/observed_window_divergence_inspection/`

The five divergent retained samples were all valid wind solutions with `trajectory_status_code = 0`, no stall penalty, no numerical-failure penalty, and no hard-invalid penalty. Their radial wind/cloud tracks were smooth. The remaining instability was instead concentrated near the low-velocity support edge: the `112` and `138 km/s` bins are floored, and the `162 km/s` bin is a tiny Gaussian leakage tail below the real cloud material in the radial window. Treating those bins as precise two-sided log-profile detections with `sigma ~= 0.095` therefore creates artificial curvature, analogous to the previous high-velocity tail problem.

An opt-in low-signal censored likelihood was added for `observable_set = log_dndv_binned`: `log_dndv_low_signal_policy = censored_upper`. It leaves the default Gaussian likelihood unchanged, but bins below `max(log dN/dv) - log_dndv_censor_delta_log` become smooth one-sided upper limits. The likelihood contribution is split into `gaussian_chi2` and `censored_chi2` while keeping total `chi2` for continuity.

Two exact `strong_wings`, `100-600 km/s`, loading-ratio, MAP-whitened short-NUTS diagnostics were run:

| likelihood | censored bins | divergences | max Rhat | min ESS | `P(eta_E > 1)` | truth in 95% | output |
|---|---:|---:|---:|---:|---:|---|---|
| Gaussian baseline | 0/20 | 5/256 | 1.051 | 21.9 | 0.398 | yes | `strong_wings_log_dndv_loading_ratios_observed_window_100_600_short_nuts/` |
| censored, `delta_log = 20` | 7/20 | 6/256 | 1.100 | 27.4 | 0.242 | yes | `strong_wings_log_dndv_loading_ratios_observed_window_100_600_censored_short_nuts/` |
| censored, `delta_log = 100` | 3/20 | 4/256 | 1.024 | 62.8 | 0.219 | yes | `strong_wings_log_dndv_loading_ratios_observed_window_100_600_censored_delta100_short_nuts/` |
| censored, `delta_log = 100`, 2048 warmup | 3/20 | 9/256 | 1.060 | 13.2 | 0.344 | yes | `strong_wings_log_dndv_loading_ratios_observed_window_100_600_censored_delta100_warmup2048_short_nuts/` |

The broad censoring pass was too aggressive: censoring through `262 km/s` removed real ramp/body information and worsened the ridge. The surgical `delta_log = 100` pass, censoring only `112`, `138`, and `162 km/s`, is the best short gate so far for this observed-window family. It still is not a production pass because four divergences remain, but it improves Rhat/ESS substantially and keeps the MAP essentially exact (`eta_E = 0.9775`). A longer 2048-warmup rerun did not cure the problem; it produced more divergences clustered at samples `32-45` and lower ESS. The remaining divergent retained samples are again valid winds, concentrated near the same low-`eta_M`, high `eta_M_cold/eta_M` ridge, with the largest residual jumps still often tied to the `162 km/s` low-support edge. This supports a likelihood/geometry interpretation, not a physical failure or insufficient-warmup interpretation.

## Final Disposition And Next Step

This is the stopping point for the current synthetic-recovery push. We have learned enough to avoid both overclaiming and over-tuning.

What is solid:

- the corrected objective and MAP machinery recover ordinary exact truth cases;
- the stalled-wind policy removes the old catastrophic invalid-cliff failure mode;
- binned/log-profile information can recover high-energy MAP solutions when the likelihood is well matched to the observable window;
- `strong_wings` divergent samples are valid wind solutions, not hidden numerical failures;
- the main remaining pathology is a narrow high-specific-energy ridge whose log-profile residuals are sensitive to support-edge bins.

What is not solid:

- `strong_wings` is not a formal production NUTS pass;
- current synthetic recovery should not be used to claim precise high-energy `eta_E` constraints near the nominal energy ceiling;
- the noisy recovery campaign should remain deferred until the final observational likelihood and parameterization are chosen.

Decision:

- stop spending project time on incremental `strong_wings` short-NUTS tuning for now;
- retain the observed-window, loading-ratio, stalled-wind, low-signal-censored setup as the best available diagnostic baseline;
- move to the next overall project step: one-at-a-time TRML/cloud microphysics sensitivity screening;
- treat that next step as forward-model leverage ranking, not as expanded inference.

The next durable deliverable should be `examples/trml_sensitivity_screen.py` plus `docs/trml_sensitivity_report.md`. The report should rank which fixed microphysics parameters move the observables, identify degeneracies with the three loading parameters, and recommend at most one effective parameter for a later expanded-recovery campaign.
