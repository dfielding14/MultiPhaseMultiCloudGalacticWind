# Expanded Inference Parameter Decision

## Decision

The first expanded-inference experiment is a restricted effective mixing amplitude:

```text
expanded_parameters = "a_mix"
theta = (eta_M, eta_M_cold, eta_E, A_mix)
```

`A_mix` multiplies the TRML/cloud mass-exchange terms and defaults to `1`. It is not a broad fit of all TRML closure parameters.

The staged second diagnostic is:

```text
expanded_parameters = "a_mix_beta_chi"
theta = (eta_M, eta_M_cold, eta_E, A_mix, beta_chi_mix)
```

with

```text
A_chi = A_mix * (chi / mixing_chi_pivot)^beta_chi_mix
mixing_chi_pivot = 100
```

Both cloud growth and cloud-loss mass exchange are multiplied by `A_chi`. Defaults `A_mix=1` and `beta_chi_mix=0` recover the baseline model.

## Priors

Recommended initial priors for validation:

```text
A_mix ~ lognormal(log 1.0, sigma_log=0.35)
beta_chi_mix ~ normal(0.0, 0.25), bounded to +/- 0.75
```

The `A_mix` prior is deliberately narrow because high-amplitude mixing directions stalled high-energy sensitivity cases. The beta prior is diagnostic and should remain centered on no extra chi tilt until exact recovery shows it is identifiable.

## Rationale

The covariance-whitened TRML sensitivity screen showed that `A_mix`, `Mdot_coefficient`, `geometric_factor`, and `f_turb0` have large profile leverage, but most of their response is aligned with existing loading directions. `A_mix` exactly mirrors `Mdot_coefficient` in the forward screen when `beta_chi_mix=0`, as intended, because both multiply the same cloud mass-exchange terms. These directions still leave nonzero orthogonal residuals after the best loading refit, so a restricted `A_mix` experiment is defensible.

The chi exponents have very high leverage but are validity-risky when fit directly. The direct `beta_chi_mix` screen is also high leverage and scientifically interesting, but still risky and mostly loading-like. A single effective `beta_chi_mix` remains the staged way to test the density-contrast question without selecting one microscopic exponent or fitting a broad closure model.

## Validation Gate

Run `A_mix` first:

- prior predictive atlas for the four-parameter model,
- exact synthetic recovery for ordinary truth cases,
- observed-window profile checks using the current support-aware/censored treatment,
- comparison against baseline three-parameter recovery.

Only run `A_mix + beta_chi_mix` if `A_mix` is acceptable. The beta diagnostic passes only if the posterior is not prior-dominated, the truth is recovered in exact tests, sampler diagnostics remain usable, and the result is not driven by `strong_wings` alone.

Do not use either expanded mode for production Paper 2 claims until synthetic recovery passes.
