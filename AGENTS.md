# AGENTS.md

## Purpose
This file is the canonical engineering and physics guide for this repository.
It defines:
- what this codebase does,
- how the model is organized,
- the governing equations implemented in code,
- and how to extend it while staying concise, documented, and physically accurate.

If you change model behavior, update this file.

---

## Canonical Scientific Reference
Primary paper to align with:
- [Fielding et al. 2022, ApJ, 924, 82](https://ui.adsabs.harvard.edu/abs/2022ApJ...924...82F/abstract)

Model-level assumptions and scaling arguments should remain consistent with that work unless explicitly documented as a deliberate deviation.

---

## Repository Contract
Keep this repository lean and purpose-built for model development.

Expected top-level project content:
- `multiphasegalacticwind/` (production package)
- `tests/` (regression and physics tests)
- `docs/` (developer/user documentation)
- `examples/` (usage scripts and optional notebooks)
- `README.md` (project overview)
- `AGENTS.md` (this document)

Do **not** add:
- generated figures or fitting outputs,
- ad-hoc one-off diagnostics in the project root,
- temporary reports,
- cache directories.

Notebook guidance:
- Notebooks are allowed only under `examples/` when they provide durable instructional value.
- Keep exploratory scratch notebooks out of version control.

---

## High-Level Architecture

### Package modules
- `multiphasegalacticwind/constants.py`
  - Physical constants and unit conversions (CGS-centered).
- `multiphasegalacticwind/config.py`
  - `WindConfig`: runtime physics/numerical parameters.
- `multiphasegalacticwind/core_physics.py`
  - ODE right-hand sides for multiphase and hot-only winds.
  - Cloud mass distribution setup.
  - Event factory functions for robust integration termination.
- `multiphasegalacticwind/cooling.py`
  - Cooling table loading, interpolation, and cooling-time utilities.
- `multiphasegalacticwind/wind_model.py`
  - User-facing API (`WindModel`) and solution container (`Solution`).
  - Unit conversion boundary between user units and CGS internals.
- `multiphasegalacticwind/observables.py`
  - Post-processing into observational quantities (notably `dN/dv`).
- `multiphasegalacticwind/analysis_helpers.py`
  - Diagnostics and derived-quantity helpers.
- `multiphasegalacticwind/plotting.py`, `multiphasegalacticwind/plotting_helpers.py`
  - Publication-style plotting wrappers.

### Data files
- `multiphasegalacticwind/data/Lambda_tab_redshifts.npz`
  - Cooling lookup table consumed by `cooling.py`.

---

## Core Data Model and Units

## User-facing units (`WindModel` inputs/outputs)
- Distances: kpc
- Velocities: km/s
- Masses: Msun
- SFR / flux-like outputs: Msun/yr

## Internal ODE units
- CGS everywhere in core equations:
  - length: cm
  - velocity: cm/s
  - mass: g
  - pressure: dyne/cm^2
  - density: g/cm^3

## State vector in `Wind_Evo`
For `N` cloud species:
- Hot phase (4):
  - `v_wind`, `rho_wind`, `Pressure`, `rhoZ_wind`
- Cold phase (3N):
  - `M_cloud[0:N]`
  - `v_cloud[0:N]`
  - `Z_cloud[0:N]`

Total length: `4 + 3N`.

---

## Governing Equations Implemented
These are the practical equations as encoded in the current code.

## 1) Launch / sonic-point initialization (`wind_model.py`)
Hot-phase injection rates:
- `Mdot_hot = eta_M * SFR * Msun/yr`
- `Edot_hot = eta_E * (E_SN / (mstar * Msun)) * SFR * Msun/yr`

Given `Mach0 = 1 + sonic_point_offset`:
- `v_star = sqrt(Edot_hot / Mdot_hot) * (1/((gamma-1)*Mach0) + 1/2)^(-1/2)`
- `rho_star = Mdot_hot / (Omwind * r_star^2 * v_star)`
- `P_star = rho_star * v_star^2 / (Mach0^2 * gamma)`

This defines the initial hot-wind state used by integration.

## 2) Cloud mass distribution (`setup_cloud_powerlaw_distribution`)
Power law:
- `dN/dM ∝ M^{-alpha_cloud}`
- Equivalent in log-space: `dN/dlogM ∝ M^{1-alpha_cloud}`

The total cold mass loading `eta_M_cold_tot` is partitioned across bins and converted to:
- species mass fluxes `Mdot_cold0[i]`
- number injection rates `Ndot_cloud0[i] = Mdot_cold0[i] / M_cloud0[i]`

## 3) Thermodynamics and potentials (`Wind_Evo`)
- `c_s^2 = gamma * P / rho`
- `Mach^2 = v^2 / c_s^2`
- `Phi(r) = v_circ^2 * ln(r)` (isothermal potential)
- Bernoulli-like terms are built from kinetic + enthalpy + potential pieces.

## 4) Cloud injection profile
For each cloud species:
- `Ndot_cloud(r) = Ndot_cloud0 * (r/injection_radius)^{injection_power}` for `r < injection_radius`
- otherwise `Ndot_cloud(r) = Ndot_cloud0`

Cloud number density from flux conservation:
- `n_cloud = Ndot_cloud / (Omwind * r^2 * v_cloud)`

## 5) Pressure equilibrium and mixing-layer closure
- `rho_cloud = P * (mu * m_p) / (k_B * T_cloud)`
- `chi = rho_cloud / rho_wind`
- `v_rel = v_wind - v_cloud`
- `v_turb = f_turb0 * |v_rel| * chi^{TurbulentVelocityChiPower}`
- `T_wind = (P/k_B) * (mu*m_p/rho_wind)`
- `T_mix = sqrt(T_wind * T_cloud)`
- `Z_mix = sqrt(Z_wind * Z_cloud)`

## 6) Cooling-time coupling
Cooling time in the mixed layer (note pressure argument unit):
- `t_cool_layer = tcool_P(T_mix, P/k_B, Z_mix/Z_solar, z=0, mu)`

Then:
- `ksi = r_cloud / (max(v_turb, eps) * t_cool_layer)`

## 7) Cloud mass exchange terms
For active clouds (`M_cloud > M_cloud_min`):
- `Mdot_grow ∝ M_cloud * v_turb * AreaBoost / (r_cloud * chi) * ksi^{1/2 or 1/4}`
- `Mdot_loss ∝ - M_cloud * v_turb_cold / r_cloud`
- `Mdot_cloud = Mdot_grow + Mdot_loss`

with:
- `AreaBoost = geometric_factor * chi^{CoolingAreaChiPower}`
- `v_turb_cold = v_turb * chi^{ColdTurbulenceChiPower}`

## 8) Momentum and energy exchange
Ram drag term (sign-preserving):
- `p_dot_ram = 0.5 * drag_coeff * rho_wind * pi * r_cloud^2 * v_rel * |v_rel|`

Volumetric source terms include SN source inside `r0`, cloud exchange, and cooling:
- `drhodt = Mdot_SN - Σ(n_cloud * Mdot_cloud)`
- `dpdt = - Σ(n_cloud * (p_dot_transfer + p_dot_ram))`
- `dedt = Edot_SN - Σ(n_cloud * (e_dot_transfer + p_dot_ram * v_wind)) + e_dot_cool`

Cooling term in wind evolution:
- `e_dot_cool = -(rho_wind/(muH*m_p))^2 * Lambda_P_rho((Pressure, rho_wind))`

## 9) Wind gradients and sonic regularization
Gradient denominator:
- nominal `D = 1 - 1/Mach^2`
- near sonic point, code regularizes to avoid singular blow-up.

The ODE returns derivatives for all hot + cloud state variables, with guards for unphysical states (negative pressure/density/velocity or NaNs).

## 10) Hot-only control system (`Hot_Wind_Evo`)
A reduced 3-variable ODE is used as a consistency baseline and for offset initialization.
It now includes gravity/source consistency with the full model.

## 11) Cooling module equations (`cooling.py`)
`tcool_P` expects pressure as `P/k_B` in `K cm^-3`.

From implementation:
- `nH_actual = (P/T) * (mu/muH)`
- `t_cool = 1.5 * (muH/mu) * k_B * T / (nH_actual * Lambda)`

Important: `tcool_P` can be negative in net-heating regimes; callers handle that explicitly.

## 12) Observable mapping (`observables.py`)
Cloud mass density along flow:
- `rho_cl ∝ Ndot_cloud * M_cloud / (Omwind * r^2 * v_cloud)`

Hydrogen number density:
- `n_H = rho_cl / (mu_cool * m_p)`

Core mapping to column-density velocity profile:
- `dN/dv ≈ n_H / (dv/dr)`
- mapping is done per cloud species and then interpolated onto a common velocity grid before summation.
- only active-cloud segments (`M_cloud >= M_cloud_min`) contribute to the transform.
- with fallback rebinned treatment for non-monotonic velocity gradients.

Velocity moments are computed from integrals over `dN/dv`.

---

## Numerical Strategy and Termination
Integration uses `scipy.integrate.solve_ivp` with event-based termination:
- sonic transitions,
- wind velocity sign flips,
- cold-wind transition,
- cloud density floor,
- cloud velocity floor,
- NaN state,
- negative pressure/density,
- stuck-step detection.

Any new stiff term must include:
- a physically meaningful limit,
- a numerical guard,
- and at least one event or explicit failure condition if it can go singular.

---

## Testing Map
Current tests live in `tests/`:
- `tests/test_ode_convergence.py`
  - full-vs-hot ODE consistency as `eta_M_cold -> 0`
  - viability behavior checks
- `tests/test_drag_physics.py`
  - drag-force sign and finite derivative behavior
- `tests/test_cooling.py`
  - cooling interpolation contracts and edge cases
- `tests/test_observables.py`
  - consistency of total `dN/dv` vs sum of species contributions
  - finiteness/ordering contracts for species-level distributions
- `tests/test_config.py`, `tests/test_cooling_simple.py`
  - legacy/simple regression-style tests

Run tests:
- `PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider`

---

## Implementation Guide

## Core principles
1. Be concise.
   - Prefer small, focused functions and minimal branching.
   - Avoid speculative abstractions.
2. Be well documented.
   - Every public function gets a docstring with units and assumptions.
   - Any nontrivial formula gets a short comment and variable-unit clarity.
3. Be physically accurate.
   - Preserve dimensional consistency first.
   - Preserve sign conventions second.
   - Add numerical safeguards only after preserving the correct physics.

## Required checklist before merging physics changes
- Units checked for every new term.
- Sign convention checked against momentum/energy directionality.
- Limiting-case behavior checked (e.g., no clouds, no cooling, weak gravity).
- Regression test added or updated.
- `pytest` passes.
- AGENTS updated if model behavior changed.

## Adding a new physical parameter
1. Add parameter to `WindConfig.__init__` with clear units and default.
2. Add to `to_dict()`.
3. Add range checks in `validate()`.
4. Thread through `WindModel` and `core_physics` parameter tuple only where needed.
5. Add tests covering default, override, and failure ranges.

## Adding/modifying ODE terms
1. Write the intended equation in comments first.
2. Implement with explicit CGS units.
3. Verify sign for both `v_rel > 0` and `v_rel < 0` if term is directional.
4. Add near-singular safeguards (but keep correct limiting trend).
5. Add/adjust event triggers if term can produce unphysical states.

## Adding observables
1. Define the physical mapping equation (e.g., `dN/dv` transform).
2. State assumptions (symmetry, monotonicity, path treatment).
3. Implement fallback for known pathological cases (e.g., negative gradients).
4. Validate against simple synthetic states.

---

## Common Pitfalls to Avoid
- Passing pressure in the wrong units to cooling functions.
  - `tcool_P` uses `P/k_B` (`K cm^-3`).
- Confusing `Lambda_P` and `get_cooling_interpolator` input spaces.
  - `get_cooling_interpolator` returns interpolator on `(Pressure [dyne/cm^2], rho [g/cm^3])`.
- Introducing hidden unit conversions in one-line expressions.
- Adding magic constants without context.
- Swallowing NaNs silently without a corresponding event or diagnostic.
- Using backup/legacy modules as canonical code paths.

---

## Style Rules for This Repo
- ASCII source unless existing file requires otherwise.
- Keep imports minimal and used.
- Keep comments technical and sparse.
- Avoid duplicate code paths.
- Prefer explicit variable names over compressed algebra in critical physics routines.

---

## Scope Discipline
This repository is not a notebook/report/artifact store.
If an output is generated during analysis, keep it out of versioned tree unless it is:
- core package code,
- tests,
- or this AGENTS document.
