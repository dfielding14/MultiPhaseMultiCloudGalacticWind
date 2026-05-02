# TRML And Cloud-Wind Sensitivity Report

This report tracks one-at-a-time sensitivity scans for fixed cloud-wind and turbulent radiative mixing-layer parameters. It is a forward-model diagnostic, not expanded inference. The fitted parameter set is still:

- `eta_M`
- `eta_M_cold`
- `eta_E`

The goal is to rank which fixed microphysics parameters move the observables enough to justify a later expanded-recovery test, and which changes are nearly degenerate with the three loading parameters.

## Status Update

As of April 30, 2026, the Step 7 staged sensitivity screen is complete enough to inform the next inference-validation decision. The full matrices were rerun after adding the restricted effective parameters `A_mix` and `beta_chi_mix` to the forward model, so the screen now includes both the original microphysics knobs and the proposed effective expansion directions.

Main result:

- Many cloud-wind microphysics parameters have strong observable leverage.
- The strongest levers are not clean broad-expansion parameters, because they either create stalled/invalid winds in several truth cases or have leading responses that are mostly aligned with the existing loading-parameter subspace.
- The covariance-whitened projection upgrade shows an important nuance: the high-leverage profile displacements are mostly loading-like, but not fully degenerate. After the best local loading refit, the strongest mixing directions still leave large orthogonal residuals under the adopted covariance.
- No parameter passes the current stability plus identifiability gate for immediate expanded inference.
- If an expanded model is later forced for scientific reasons, the least-bad first candidate is the restricted effective mixing amplitude `A_mix`, not a broad fit of all TRML closure parameters. In the screen, `A_mix` exactly mirrors the old `Mdot_coefficient` proxy because both multiply the same mass-exchange terms when `beta_chi_mix = 0`.

Recommendation:

```text
Do not fit a broad microphysics model.
The only plausible next expansion is a deliberately restricted A_mix experiment with its own validation gate.
```

That experiment should specify a deliberately narrow `A_mix` prior that avoids the stalled high-amplitude regime seen here. The `beta_chi_mix` direction has large leverage, but it is still a staged diagnostic because it is risky and mostly loading-like in this one-at-a-time screen.

## Implementation Status

The sensitivity harness is:

```text
examples/trml_sensitivity_screen.py
```

It now supports both legacy single-selection flags and multi-selection flags:

```text
--parameter / --parameters
--fiducial-case / --truth-cases
```

For each scan point, the script records validity, stalled-wind diagnostics, observables, raw velocity moments, shape summaries, observable-distance metrics, a legacy local cosine similarity against finite-difference loading directions, and a covariance-whitened projection onto the full three-loading subspace. The projection reports:

- `loading_subspace_fraction`: the fraction of the whitened squared microphysics displacement explained by the best linear combination of `eta_M`, `eta_M_cold`, and `eta_E` changes.
- `loading_orthogonal_chi`: the residual `sqrt(Delta chi2_perp)` left after projecting out that loading subspace.

A high subspace fraction means the leading response is loading-like. A large orthogonal residual means the microphysics effect is not fully absorbed by refitting the loadings.

The parameter registry now also includes:

```text
A_mix = 0.5, 1.0, 2.0, 3.0
beta_chi_mix = -0.5, 0.0, 0.5
```

These are still screened as fixed forward-model settings, not inferred in Step 7.

The aggregation and manuscript-figure helper is:

```text
examples/make_paper2_trml_sensitivity_figures.py
```

It writes cross-case summary tables under:

```text
examples/outputs/trml_sensitivity_screen/paper2_trml_sensitivity_aggregate_20260428/
```

and manuscript figures under:

```text
paper/paper2_inference_validation/figures/
```

Generated CSV/NPZ/output directories remain diagnostic artifacts and should not be committed by default.

## Runs

All runs used the reduced CPU settings from the exact synthetic-recovery diagnostics:

```text
r_max_kpc = 6.0
step_kpc = 0.08
n_cloud_species = 4
cloud_mass_range = (10, 1e4) Msun
failure_policy = stalled_wind
```

The truth cases were:

```text
fiducial
low_eta_m_cold
low_eta_m_high_eta_e
narrow_profile
strong_wings
```

### Stage A: Shape-Five Multi-Case Screen

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/trml_sensitivity_screen.py \
  --parameters all \
  --truth-cases fiducial,low_eta_m_cold,low_eta_m_high_eta_e,narrow_profile,strong_wings \
  --observable-set logm0_mean_sigma_skew_kurt \
  --output examples/outputs/trml_sensitivity_screen/multicase_shape5_reduced_20260428 \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --failure-policy stalled_wind \
  --no-plots
```

Runtime was about 22 seconds.

### Stage B: Observed-Window Log-Profile Screen

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/trml_sensitivity_screen.py \
  --parameters A_mix,Mdot_coefficient,beta_chi_mix,geometric_factor,f_turb0,cloud_alpha,CoolingAreaChiPower,TurbulentVelocityChiPower,drag_coeff,Cooling_Factor \
  --truth-cases fiducial,low_eta_m_cold,low_eta_m_high_eta_e,narrow_profile,strong_wings \
  --observable-set log_dndv_binned \
  --dndv-num-bins 20 \
  --dndv-vmin-kms 100 \
  --dndv-vmax-kms 600 \
  --dndv-bin-corr 0.60 \
  --dndv-sigma-floor-frac 0.03 \
  --log-dndv-low-signal-policy censored_upper \
  --log-dndv-censor-delta-log 100 \
  --output examples/outputs/trml_sensitivity_screen/multicase_log_dndv_observed_window_20260428 \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --failure-policy stalled_wind \
  --no-plots
```

Runtime was about 17 seconds. This screen uses the same observed velocity window and low-signal censoring logic as the best current `strong_wings` synthetic-recovery diagnostic.

### Stage C: Linear-Profile Robustness Screen

```bash
JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONDONTWRITEBYTECODE=1 \
python examples/trml_sensitivity_screen.py \
  --parameters A_mix,Mdot_coefficient,geometric_factor,f_turb0,cloud_alpha \
  --truth-cases fiducial,low_eta_m_cold,low_eta_m_high_eta_e,narrow_profile,strong_wings \
  --observable-set dndv_binned \
  --dndv-num-bins 20 \
  --dndv-vmin-kms 100 \
  --dndv-vmax-kms 600 \
  --dndv-bin-corr 0.60 \
  --dndv-sigma-floor-frac 0.03 \
  --output examples/outputs/trml_sensitivity_screen/multicase_dndv_observed_window_shortlist_20260428 \
  --r-max-kpc 6.0 \
  --step-kpc 0.08 \
  --n-cloud-species 4 \
  --cloud-mass-min 10.0 \
  --cloud-mass-max 1.0e4 \
  --failure-policy stalled_wind \
  --no-plots
```

Runtime was about 10 seconds. The raw linear-profile observable distance can be enormous when low-signal bins move by large fractional amounts, so the cross-mode comparison uses profile log-distance rather than the raw linear observable-distance score.

## Cross-Case Ranking

The aggregate summary is:

```text
examples/outputs/trml_sensitivity_screen/paper2_trml_sensitivity_aggregate_20260428/trml_sensitivity_cross_case_summary.csv
```

The per-case table used to check whether the result is driven only by
`strong_wings` is:

```text
examples/outputs/trml_sensitivity_screen/paper2_trml_sensitivity_aggregate_20260428/trml_sensitivity_per_case_summary.csv
```

Compact ranking:

| Rank | Parameter | Classification | Shape-five max shift | Log-profile max distance | Min valid fraction | Max loading fraction | Max orthogonal residual |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `TurbulentVelocityChiPower` | high leverage but risky | 1.116 | 104.0 | 0.667 | 0.955 | 2398 |
| 2 | `CoolingAreaChiPower` | high leverage but risky | 1.557 | 101.8 | 0.600 | 0.955 | 714 |
| 3 | `beta_chi_mix` | high leverage but risky | 0.633 | 91.7 | 0.867 | 0.979 | 1669 |
| 4 | `geometric_factor` | high leverage but risky | 0.562 | 84.9 | 0.900 | 0.998 | 1633 |
| 5 | `A_mix` | high leverage but risky | 0.338 | 74.9 | 0.900 | 0.998 | 1698 |
| 6 | `Mdot_coefficient` | high leverage but risky | 0.338 | 74.9 | 0.900 | 0.998 | 1698 |
| 7 | `f_turb0` | high leverage but risky | 0.234 | 72.2 | 0.850 | 0.997 | 1475 |
| 8 | `cloud_alpha` | high leverage but risky | 0.335 | 13.3 | 0.933 | 0.995 | 36.1 |
| 9 | `drag_coeff` | stable high leverage; loading-degenerate | 0.153 | 9.47 | 1.000 | 0.996 | 37.8 |
| 10 | `ColdTurbulenceChiPower` | high leverage but risky | 0.224 | -- | 0.667 | 0.996 | 3.48 |
| 11 | `Z_hot_over_Z_solar` | low leverage | 0.142 | -- | 1.000 | 0.999 | 1.05 |
| 12 | `Z_cloud_over_Z_solar` | low leverage | 0.139 | -- | 1.000 | 0.998 | 1.44 |
| 13 | `Cooling_Factor` | low leverage | 0.0087 | 0.33 | 1.000 | 0.998 | 0.53 |
| 14 | `v_cloud_init` | low leverage | 0.045 | -- | 0.750 | 0.949 | 0.75 |

The highest-leverage parameters are the chi-dependent turbulence/area exponents, the effective chi-tilt `beta_chi_mix`, and the effective mixing-amplitude knobs (`A_mix`, `Mdot_coefficient`, `geometric_factor`, and `f_turb0`). However, the exponent and chi-tilt directions create stalled or invalid solutions. The amplitude knobs are physically cleaner but still produce repeated stalled high-energy solutions at the upper scan values, especially for `strong_wings` and `low_eta_m_high_eta_e`.

## Invalid And Stalled Scan Points

The full invalid/stalled table is:

```text
examples/outputs/trml_sensitivity_screen/paper2_trml_sensitivity_aggregate_20260428/trml_sensitivity_invalid_scan_points.csv
```

The recurring failure patterns are:

- `CoolingAreaChiPower = 1.0` stalls several truth cases, including `fiducial`, `narrow_profile`, and `strong_wings`.
- `CoolingAreaChiPower = 0.0` stalls the high-energy cases.
- `TurbulentVelocityChiPower = -0.5` stalls the high-energy cases, while `TurbulentVelocityChiPower = 0.5` creates failures in lower-cold or fiducial-like cases.
- `beta_chi_mix = -0.5` stalls the high-energy cases in the profile screen, while `beta_chi_mix = 0.5` stalls `strong_wings` in the compact screen.
- `A_mix = 2` stalls `strong_wings`, and `A_mix = 3` stalls `low_eta_m_high_eta_e`.
- `Mdot_coefficient = 2/3` stalls `strong_wings`, and `Mdot_coefficient = 1` stalls `low_eta_m_high_eta_e`.
- `geometric_factor = 4` stalls `narrow_profile` and `strong_wings`.
- `f_turb0 = 0.2` stalls `strong_wings`; `f_turb0 = 0.4` stalls both `low_eta_m_high_eta_e` and `strong_wings`.
- `cloud_alpha = 2.5` stalls `strong_wings`.
- `v_cloud_init = 0` stalls several cases, and `v_cloud_init = 50 km/s` stalls `strong_wings`.

These are physically useful failures. They say that stronger mixing, stronger area enhancement, or altered turbulence scaling can choke the wind before the observable radius in high-specific-energy or narrow-profile regimes. This is not the old hard sampler cliff: the stalled-wind policy makes these finite, reported forward-model outcomes.

## Degeneracy Interpretation

The strongest result is not simply that microphysics matters. It is that most microphysics directions have leading responses that are mostly aligned with the subspace already spanned by the three loading parameters, but the profile-level response is often too large to be completely absorbed by loadings.

The covariance-whitened projection gives two useful diagnostics:

- `A_mix`, `Mdot_coefficient`, `geometric_factor`, `f_turb0`, `cloud_alpha`, and `drag_coeff` have maximum loading-subspace fractions of about `0.995-0.998`. Their leading response is therefore mostly loading-like.
- `beta_chi_mix` is less perfectly aligned than the pure amplitude knobs, but it is still strongly loading-like, with a maximum loading-subspace fraction of about `0.979`.
- The high-leverage profile perturbations still leave large orthogonal residuals after the best loading refit. For `A_mix`, `Mdot_coefficient`, `geometric_factor`, and `f_turb0`, the maximum residuals are of order `1e3` in `sqrt(Delta chi2_perp)` under the adopted profile covariance.
- `Cooling_Factor` is validity-stable but nearly inert in this reduced screen, and its orthogonal residual stays below one.

This means a one-object posterior that adds a broad microphysics parameter would still trade strongly against `eta_M`, `eta_M_cold`, and `eta_E`, but the trade is not exact. A restricted effective mixing-amplitude test is therefore more justified than it looked under the old max-cosine metric, provided it gets its own prior predictive and synthetic-recovery gate.

## Cloud Mass Distribution Caveat

The screen includes `cloud_alpha`, and the result is informative:

- shape-five max shift: `0.335`
- observed-window log-profile max distance: `13.3`
- minimum valid fraction: `0.933`
- maximum loading-subspace fraction: `0.995`
- maximum orthogonal residual: `36.1`

This means the cloud mass-spectrum slope has real observable leverage, but its leading effect is mostly loading-like. Physically, this is expected. A steeper cloud spectrum weights small clouds more heavily; small clouds mix, decelerate, and disrupt more readily. A shallower spectrum weights massive clouds more heavily; massive clouds survive farther and can carry cold column to different velocities. These effects can mimic changes in `eta_M_cold` and in the effective mixing amplitude `A_mix`.

The cloud mass limits are an additional hidden systematic. Changing `cloud_mass_min` or `cloud_mass_max` changes the physical cloud population, but with only four cloud species it also changes the numerical quadrature of that population. That means a mass-limit test is not just "another microphysics knob"; it also checks whether the discretized cloud population is stable enough for inference.

Decision for this step:

```text
Do not infer cloud_alpha, cloud_mass_min, or cloud_mass_max yet.
Treat the fiducial cloud mass distribution as a fixed model assumption for the next A_mix gate.
Require a later robustness check over reasonable cloud mass ranges and slopes before production expanded-inference claims.
```

Suggested robustness grid after the first `A_mix` recovery gate:

```text
cloud_alpha = 1.5, 2.0, 2.5
cloud_mass_min = 1, 10, 100 Msun
cloud_mass_max = 1e3, 1e4, 1e5 Msun
```

This should be a forward-model or exact-recovery robustness check around the restricted `A_mix` experiment, not a blocker before the first `A_mix` validation run.

## Decision

Step 7 is complete enough to make the immediate expanded-inference decision:

```text
Do not add a broad fitted microphysics model yet.
```

No scanned parameter simultaneously satisfies:

- high cross-case leverage,
- stable validity across the scan range,
- a small loading-subspace fraction or a controlled orthogonal residual,
- and a clean profile-level interpretation.

The best future candidate remains a restricted effective mixing amplitude, most naturally:

```text
A_mix -> effective TRML/cloud mass-exchange amplitude
```

The implementation decision is now recorded in `docs/inference_expanded_parameter_decision.md`. The restricted experiment uses:

```text
A_chi = A_mix * (chi / 100)^beta_chi_mix
```

with `beta_chi_mix=0` in the first `A_mix`-only gate. The rerun supports this staging: `A_mix` reproduces the intended mass-exchange-amplitude response, while `beta_chi_mix` is scientifically interesting but higher leverage, riskier, and still mostly loading-like. The `beta_chi_mix` extension should only be run after the four-parameter `A_mix` model passes exact recovery.

`drag_coeff` is validity-stable and has profile leverage, but it has lower leverage than the effective mixing-amplitude directions and remains lower priority than `A_mix`. The chi exponents are powerful but too risky as first expanded-inference parameters because they change the validity structure of the wind solutions.

## Paper 2 Artifacts

The manuscript figures are:

- `paper/paper2_inference_validation/figures/trml_sensitivity_cross_case_ranking.png`
- `paper/paper2_inference_validation/figures/trml_sensitivity_per_case_heatmap.png`
- `paper/paper2_inference_validation/figures/trml_sensitivity_leverage_degeneracy.png`

Paper 2 Section 6 now presents this as a negative-but-useful validation result: the model is sensitive to TRML/cloud microphysics, but the current data vector does not yet justify fitting a broad microphysics parameter. The added per-case heatmap is important because it shows that the result is not driven solely by the `strong_wings` stress case. The upgraded degeneracy figure is important because it separates "mostly loading-like" from "fully degenerate."
