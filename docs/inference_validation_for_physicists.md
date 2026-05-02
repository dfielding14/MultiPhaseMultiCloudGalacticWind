# Inference Guide For Physicists

This document explains the inference validation program for the multiphase galactic wind model in physics terms. It assumes comfort with the wind model, cloud-wind interaction physics, and the limits of subgrid closures, but it does not assume much background in Bayesian inference.

The short version is:

1. First map what the model can produce before looking at data.
2. Then test whether the inference can recover known fake inputs.
3. Then ask which deeper cloud-wind parameters actually leave observable signatures.
4. Only then expand the fitted parameter set.

The goal is not to make the sampler return more parameters. The goal is to learn which physical degrees of freedom the data can actually constrain.

## Current Situation

The current inference code is centered on `MomentInferenceModel` in `multiphasegalacticwind/inference.py`.

At present it fits only:

```text
eta_M        hot phase mass loading
eta_M_cold   cold phase mass loading
eta_E        hot phase energy loading
```

The turbulent radiative mixing layer and cloud-wind interaction parameters are fixed by `WindConfig`. They affect the forward model, but they are not currently sampled or optimized as inference parameters.

Important fixed or setup-level quantities include:

```text
SFR
v_circ
r_star_kpc
r_max_kpc
cloud_mass_range
cloud_alpha
n_cloud_species
WindConfig fields
```

The current observable modes are:

```text
m0_m1_m2
logm0_mean_sigma_skew_kurt
dndv_binned
```

`m0_m1_m2` is compact, but loses profile-shape information. `logm0_mean_sigma_skew_kurt` keeps more shape information. `dndv_binned` is the most information-rich current option, but it requires a realistic covariance model.

## What Inference Is Doing

The forward model is:

```text
physical parameters -> wind ODE solution -> cloud column/velocity distribution -> observable vector
```

For example:

```text
eta_M, eta_M_cold, eta_E -> dN/dv -> M0, mean velocity, velocity width
```

Inference runs this map backward in a probabilistic sense. It asks which physical parameters could plausibly have produced the observed data.

This is not exactly inversion. Many different parameter combinations can produce similar observables, especially when the observables are only low-order summaries. The posterior is the set of plausible parameter values after accounting for prior physical expectations and data uncertainties.

## Core Vocabulary

### Forward Model

The forward model takes physical parameters and predicts observables.

In this repository the forward model includes:

- launch condition calculation,
- JAX ODE integration,
- cloud mass, velocity, and metallicity evolution,
- conversion to column-density velocity structure,
- observable summaries such as moments or binned `dN/dv`.

The forward model must be physically correct before inference results can be trusted.

### Prior

A prior is the physically plausible parameter distribution before looking at the data.

Examples:

```text
eta_M should be positive and roughly order 0.01 to a few, not 1e6.
eta_E should be positive and is currently capped below 1 in the inference code.
f_turb0 must be between 0 and 1 under WindConfig validation.
```

A prior should encode physics judgment. It should not be so broad that most samples produce failed or irrelevant winds. If the prior mostly generates invalid solutions, the scientific question is not yet well posed.

### Likelihood

The likelihood says how close model predictions must be to the data to count as plausible.

In the current code, this is effectively:

```text
observed vector = predicted vector + Gaussian noise
```

with a user-supplied covariance matrix.

The covariance is not a minor detail. It encodes measurement errors, correlations, systematic uncertainty, and how much mismatch we tolerate. Binned `dN/dv` points are probably correlated. Treating them as independent can make the inference look more constraining than it really is.

### Posterior

The posterior is the remaining plausible parameter distribution after comparing model predictions to data.

Informally:

```text
posterior = prior x likelihood
```

The posterior is not just the best-fit point. The posterior contains uncertainty, parameter degeneracies, and information about which parameters are or are not constrained.

### Prior Predictive Check

A prior predictive check asks:

```text
If I draw parameters from the prior and run the model, what range of observables do I get?
```

This is the first major sanity check.

It answers:

- Does the prior generate physical winds?
- Does it generate plausible velocities and columns?
- Does it produce the observed systems at all?
- Does most of the prior fail integration?
- Which observables are naturally broad or narrow?
- Which parameters visibly control which outputs?

If the prior predictive distribution is absurd, posterior inference will not fix the problem.

### Posterior Predictive Check

A posterior predictive check asks:

```text
If I draw parameters from the fitted posterior and rerun the model, do the predictions look like the data?
```

This is more important than only reporting best-fit parameters.

It answers:

- Does the fit reproduce the full shape of the data?
- Are there systematic residuals?
- Did the model fit total column but miss the velocity wings?
- Did it match low-order moments while missing profile structure?
- Are derived physical quantities reasonable?

### Input Recovery

Input recovery asks:

```text
If I generate fake data from known physical parameters, then fit that fake data, do I recover the known inputs?
```

This is the main validation step before trusting real-data inference.

The workflow is:

1. Choose true parameters.
2. Run the forward model.
3. Add realistic noise.
4. Fit the synthetic data.
5. Check whether the true parameters are contained in the recovered posterior.

Good recovery does not mean the best-fit point always equals the exact input. With noise, that is not expected. Good recovery means the true values fall inside credible regions at the expected rate and the posterior is not strongly biased.

### Identifiability

A parameter is identifiable if changing it leaves a distinct signature in the observables that cannot be easily mimicked by changing other parameters.

This is the central issue for deeper cloud-wind physics. The model contains many subgrid knobs, but the data may only constrain a few combinations of those knobs.

## Why The Deeper Parameters Are Hard

The turbulent radiative mixing layer and cloud-wind interaction model is intentionally phenomenological. Several parameters control related physical effects.

Candidate deeper parameters include:

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

These parameters do not all leave cleanly separable observational signatures.

Likely degeneracies:

```text
f_turb0 vs Mdot_coefficient vs geometric_factor
```

All can change the effective mass-exchange strength.

```text
eta_M_cold vs cloud_alpha vs cloud_mass_range
```

All affect how much cold material contributes to absorption and which cloud species dominate.

```text
eta_M vs eta_E
```

Both affect launch velocity, hot temperature, cooling, and cloud acceleration.

```text
Cooling_Factor vs eta_M vs eta_E vs Z_hot_over_Z_solar
```

All affect thermal evolution and cooling losses.

```text
drag_coeff vs f_turb0
```

Both can alter cloud velocity evolution, although through different physical channels.

```text
TurbulentVelocityChiPower vs CoolingAreaChiPower vs ColdTurbulenceChiPower
```

These are separate closures in the equations, but observations may only constrain one effective chi dependence.

This is why the right first expansion is probably not to fit every TRML knob. It is more likely to fit one effective mixing-amplitude parameter, then test whether a second parameter such as `drag_coeff` is identifiable.

## Recommended Scientific Roadmap

### Stage 0: Establish A Correct Forward Model

The first priority is correctness.

The following issues must be fixed before serious inference work:

- `Cooling_Factor` must scale cooling numerically in the JAX RHS.
- Cloud enthalpy must use `cs_cl_sq / (gamma - 1)`.
- Unknown config keys must raise errors.
- Invalid cloud mass ranges must fail before `log10`.
- Apple Silicon local tests must use CPU rather than Metal.
- `cloud_ksi` must use `chi = rho_cloud / rho_wind = T_wind / T_cloud`.
- `Cooling_and_Acceleration` must accept the correct circular velocity.

Without this stage, synthetic recovery can validate bugs rather than physics.

### Stage 1: Baseline Prior Predictive Atlas

Start with the current three parameters:

```text
eta_M
eta_M_cold
eta_E
```

Draw many samples from physically plausible priors, run the model, and record:

```text
input parameters
valid or invalid integration
first invalid radius if failed
dN/dv profile
M0, M1, M2
logM0, mean velocity, sigma velocity, skewness, kurtosis
mass loading at 10 kpc
hot and cloud velocities
cloud survival information
cooling loss diagnostics
```

Useful plots:

- prior samples in parameter space,
- valid and invalid regions,
- random predicted `dN/dv` profiles,
- envelopes of predicted `dN/dv`,
- observable histograms,
- parameter-observable scatter plots,
- cooling luminosity versus parameters,
- cloud survival radius versus parameters.

Questions to answer:

- Does the model produce the observed range of velocities?
- Does it produce plausible columns?
- Does it fail in large parts of prior space?
- Which parameters control which observables?
- Do real observations sit inside the prior predictive envelope?

If the answer is no, tune the prior or model setup before fitting data.

### Stage 2: Baseline Input Recovery

Use the current three-parameter model to generate fake data and recover it.

Choose several truth cases:

```text
fiducial
low eta_M, high eta_E
high eta_M, lower eta_E
low eta_M_cold
high eta_M_cold
near a model-failure boundary
strong velocity wings
narrow velocity profile
```

For each truth case:

1. Run the model.
2. Compute observables.
3. Add realistic noise.
4. Fit with MAP and posterior sampling.
5. Check whether the true values are recovered.

Repeat for:

```text
m0_m1_m2
logm0_mean_sigma_skew_kurt
dndv_binned
```

Expected result:

- moment-only inference may recover broad combinations,
- shape summaries should improve constraints,
- binned `dN/dv` should be best if the covariance is realistic.

Do not add deeper TRML parameters until this baseline recovery is acceptable.

### Stage 3: TRML Sensitivity Screen

Before fitting deeper physics, vary fixed TRML/cloud parameters one at a time.

For each candidate parameter:

- choose a plausible range,
- hold the three baseline inference parameters fixed,
- run a sweep,
- compare observable changes to expected measurement errors.

Rank each parameter as:

```text
high leverage and plausibly identifiable
high leverage but degenerate
low leverage
mostly causes model failures
not worth fitting
```

This stage tells us what the data could plausibly constrain.

Important cloud-population caveat:

```text
cloud_alpha
cloud_mass_min
cloud_mass_max
```

These affect the injected cloud population, not only the local cloud-wind interaction. A steeper `cloud_alpha` or a lower `cloud_mass_min` emphasizes smaller clouds that mix and disrupt more easily. A shallower `cloud_alpha` or a higher `cloud_mass_max` emphasizes massive clouds that survive farther. Those changes can look like changes in `eta_M_cold` or `A_mix`.

So the next step should not fit the cloud mass distribution directly. Keep it fixed for the first `A_mix` validation gate, then run a robustness check over reasonable slopes and mass limits before making production claims.

### Stage 4: Define Effective Deeper Parameters

Avoid fitting every ad hoc subgrid knob directly.

Recommended first deeper parameter:

```text
A_mix
```

Interpretation:

`A_mix` is an effective turbulent mass-exchange amplitude. It asks whether the data prefer stronger or weaker cloud-wind mass exchange than the fiducial closure.

The old sensitivity screen used `Mdot_coefficient` as the closest proxy. The direct rerun now includes `A_mix` itself and confirms that it mirrors `Mdot_coefficient` when `beta_chi_mix=0`, as expected.

Related but deferred mappings:

```text
A_mix multiplies f_turb0
A_mix multiplies geometric_factor
```

The implementation decision is:

```text
A_chi = A_mix * (chi / 100)^beta_chi_mix
```

with `beta_chi_mix = 0` in the first `A_mix`-only experiment. This changes the mass-exchange amplitude without changing the turbulent velocity diagnostic itself.

Second candidate:

```text
drag_coeff
```

This controls momentum exchange and may leave a distinct imprint on velocity-profile shape.

Only after `A_mix` passes exact recovery, consider the effective chi-tilt diagnostic:

```text
beta_chi_mix
```

Do not start by fitting all three chi exponents separately.

### Stage 5: Expand Inference One Parameter At A Time

First test:

```text
eta_M
eta_M_cold
eta_E
A_mix
```

Then separately test:

```text
eta_M
eta_M_cold
eta_E
A_mix
beta_chi_mix
```

Only then consider:

```text
eta_M
eta_M_cold
eta_E
A_mix
drag_coeff
```

For every expanded parameter set, repeat:

- prior predictive checks,
- input recovery,
- posterior predictive checks,
- degeneracy analysis.

Add a parameter only if:

- it has high sensitivity,
- it is recoverable in synthetic data,
- its posterior is narrower than its prior,
- it improves posterior predictive checks for a physically interpretable reason.

### Stage 6: Fit Real Data

Only after synthetic recovery is acceptable, fit real objects.

For each object:

- draw posterior samples,
- generate posterior predictive `dN/dv` bands,
- plot residuals,
- inspect derived physical quantities,
- check whether the posterior is constrained by data or dominated by the prior.

Useful checks:

```text
observed dN/dv with model bands
residuals versus velocity
moment comparison table
posterior corner plot
cloud survival radius distribution
mass loading distribution
cooling luminosity distribution
```

### Stage 7: Population Inference

Some TRML parameters may be weakly constrained for one galaxy but meaningful across many galaxies.

A plausible hierarchy is:

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

This should come after single-object recovery is understood.

## Practical Decision Rules

### Move Forward If

- prior predictive outputs cover physically relevant observations,
- invalid model regions are understood,
- synthetic recovery works for the current parameter set,
- added parameters are measurably constrained by synthetic data,
- posterior predictive checks improve in physically meaningful ways.

### Stop And Reconsider If

- most prior samples fail,
- posterior equals prior,
- recovery is biased,
- fit quality improves only by pushing a parameter to an extreme prior edge,
- added parameters are completely degenerate,
- binned profile residuals show coherent model failure.

## Recommended First Experiments

### Experiment 1: Current 3-Parameter Prior Predictive

Use:

```text
observable_set = dndv_binned
n_samples = 200 to 1000 for first pass
JAX CPU backend on this machine
```

Output:

- predicted `dN/dv` envelopes,
- moment distributions,
- parameter-observable plots,
- failure map.

### Experiment 2: Current 3-Parameter Input Recovery

Use:

```text
8 truth cases
10 to 50 noise realizations each
compare m0_m1_m2, shape5, and dndv_binned
```

Output:

- truth versus recovered parameter plots,
- posterior coverage statistics,
- degeneracy summaries,
- posterior predictive checks.

### Experiment 3: TRML One-At-A-Time Screen

Use:

```text
3 fiducial wind parameter points
10 to 20 values per deeper physics parameter
same observables as Experiment 1
```

Output:

- sensitivity rankings,
- failure regions,
- recommendation for first effective deeper parameter.

## Final Physics Guidance

The inference program should be conservative. The cloud-wind interaction model is ad hoc enough that fitting all closure knobs at once would give a false sense of precision.

The defensible path is:

```text
map the model
validate recovery
measure sensitivity
fit the smallest meaningful extension
check posterior predictions
```

The first expanded model should probably be:

```text
eta_M, eta_M_cold, eta_E, A_mix
```

with `A_mix` acting as an effective turbulent mass-exchange amplitude. After that, test whether `drag_coeff` is separately identifiable from velocity-profile shape.
