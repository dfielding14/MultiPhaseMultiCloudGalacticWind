#!/usr/bin/env python3
"""One-at-a-time sensitivity screen for fixed cloud-wind microphysics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference_synthetic_recovery import TRUTH_CASES, build_synthetic_covariance, configure_jax_platform


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "trml_sensitivity_screen")
PARAM_NAMES = ("eta_M", "eta_M_cold", "eta_E")
OBSERVABLE_SETS = ("m0_m1_m2", "logm0_mean_sigma_skew_kurt", "dndv_binned", "log_dndv_binned")


@dataclass(frozen=True)
class ParameterSpec:
    """A fixed microphysics parameter to vary in the sensitivity screen."""

    name: str
    target: str
    default: float
    values: tuple[float, ...]
    description: str


PARAMETER_SPECS: dict[str, ParameterSpec] = {
    "f_turb0": ParameterSpec(
        "f_turb0",
        "config",
        0.1,
        (0.05, 0.1, 0.2, 0.4),
        "Turbulent velocity normalization in the mixing layer.",
    ),
    "drag_coeff": ParameterSpec(
        "drag_coeff",
        "config",
        0.5,
        (0.25, 0.5, 1.0, 2.0),
        "Ram-drag coefficient coupling clouds to the hot wind.",
    ),
    "Mdot_coefficient": ParameterSpec(
        "Mdot_coefficient",
        "config",
        1.0 / 3.0,
        (1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
        "Mass-growth and destruction prefactor.",
    ),
    "geometric_factor": ParameterSpec(
        "geometric_factor",
        "config",
        1.0,
        (0.5, 1.0, 2.0, 4.0),
        "Surface-area boost factor for mixing-layer exchange.",
    ),
    "CoolingAreaChiPower": ParameterSpec(
        "CoolingAreaChiPower",
        "config",
        0.5,
        (0.0, 0.5, 1.0),
        "Exponent for the cooling-area dependence on density contrast.",
    ),
    "ColdTurbulenceChiPower": ParameterSpec(
        "ColdTurbulenceChiPower",
        "config",
        -0.5,
        (-1.0, -0.5, 0.0),
        "Exponent for cold-side turbulent velocity scaling.",
    ),
    "TurbulentVelocityChiPower": ParameterSpec(
        "TurbulentVelocityChiPower",
        "config",
        0.0,
        (-0.5, 0.0, 0.5),
        "Exponent for turbulent velocity scaling with density contrast.",
    ),
    "Cooling_Factor": ParameterSpec(
        "Cooling_Factor",
        "config",
        1.0,
        (0.0, 0.5, 1.0, 2.0),
        "Global multiplier on hot-phase cooling losses.",
    ),
    "cloud_alpha": ParameterSpec(
        "cloud_alpha",
        "model",
        2.0,
        (1.5, 2.0, 2.5),
        "Initial cloud mass-spectrum slope.",
    ),
    "v_cloud_init": ParameterSpec(
        "v_cloud_init",
        "config",
        100.0,
        (0.0, 50.0, 100.0, 200.0),
        "Initial cloud velocity at launch [km/s].",
    ),
    "Z_cloud_over_Z_solar": ParameterSpec(
        "Z_cloud_over_Z_solar",
        "config",
        0.3,
        (0.1, 0.3, 1.0),
        "Initial cloud metallicity in solar units.",
    ),
    "Z_hot_over_Z_solar": ParameterSpec(
        "Z_hot_over_Z_solar",
        "config",
        10.0**-0.5,
        (0.1, 10.0**-0.5, 1.0),
        "Hot-phase metallicity in solar units.",
    ),
}


@dataclass
class Evaluation:
    """One model evaluation for one truth case and one microphysics value."""

    parameter_name: str
    parameter_value: float
    truth_case: str
    theta: np.ndarray
    valid: bool
    trajectory_status_code: float
    first_invalid_r_kpc: float
    soft_reach_radius_kpc: float
    min_hot_velocity_kms: float
    stall_penalty: float
    observables: np.ndarray
    raw_moments: np.ndarray
    shape: dict[str, float]
    baseline_observables: np.ndarray
    baseline_raw_moments: np.ndarray
    observable_distance: float
    profile_distance: float
    loading_degeneracy_cosine: float
    loading_subspace_fraction: float
    loading_orthogonal_chi: float
    loading_orthogonal_rms: float
    loading_whitened_distance: float
    error: str = ""


def parse_values(value_text: str | None, default_values: Sequence[float]) -> tuple[float, ...]:
    """Parse a comma-separated value list or return defaults."""
    if value_text is None or value_text.strip() == "":
        return tuple(float(v) for v in default_values)
    values = tuple(float(part.strip()) for part in value_text.split(",") if part.strip())
    if not values:
        raise ValueError("--values must contain at least one numeric value")
    return values


def parse_name_list(value_text: str, choices: Sequence[str], label: str) -> list[str]:
    """Parse a comma-separated name list, supporting the special value all."""
    names = [part.strip() for part in value_text.split(",") if part.strip()]
    if not names:
        raise ValueError(f"{label} must contain at least one name")
    if "all" in names:
        if len(names) != 1:
            raise ValueError(f"{label} cannot mix 'all' with explicit names")
        return list(choices)
    unknown = [name for name in names if name not in choices]
    if unknown:
        known = ", ".join(choices)
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}. Known values: {known}")
    return names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter", choices=["all", *PARAMETER_SPECS.keys()], default="all")
    parser.add_argument(
        "--parameters",
        default=None,
        help="Comma-separated parameter list, or all. Overrides legacy --parameter when provided.",
    )
    parser.add_argument(
        "--values",
        default=None,
        help="Comma-separated values. Only valid when scanning a single --parameter.",
    )
    parser.add_argument("--fiducial-case", choices=["all", *TRUTH_CASES.keys()], default="fiducial")
    parser.add_argument(
        "--truth-cases",
        default=None,
        help="Comma-separated truth-case list, or all. Overrides legacy --fiducial-case when provided.",
    )
    parser.add_argument("--observable-set", choices=OBSERVABLE_SETS, default="logm0_mean_sigma_skew_kurt")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jax-platform", choices=["cpu"], default="cpu")

    parser.add_argument("--sfr", type=float, default=20.0)
    parser.add_argument("--v-circ", type=float, default=150.0)
    parser.add_argument("--r-star-kpc", type=float, default=0.3)
    parser.add_argument("--r-max-kpc", type=float, default=6.0)
    parser.add_argument("--step-kpc", type=float, default=0.08)
    parser.add_argument("--first-step-kpc", type=float, default=1.0e-12)
    parser.add_argument("--n-cloud-species", type=int, default=4)
    parser.add_argument("--cloud-mass-min", type=float, default=10.0)
    parser.add_argument("--cloud-mass-max", type=float, default=1.0e4)
    parser.add_argument("--integrator-mode", choices=["rk2", "rk3", "rk4", "tsit5"], default="rk4")
    parser.add_argument("--integrator-rtol", type=float, default=1.0e-5)
    parser.add_argument("--integrator-atol", type=float, default=1.0e-8)
    parser.add_argument("--integrator-max-steps", type=int, default=131072)

    parser.add_argument("--dndv-num-bins", type=int, default=25)
    parser.add_argument("--dndv-vmin-kms", type=float, default=0.0)
    parser.add_argument("--dndv-vmax-kms", type=float, default=1200.0)
    parser.add_argument("--dndv-kernel-sigma-kms", type=float, default=None)
    parser.add_argument(
        "--dndv-kernel",
        choices=["gaussian", "truncated_gaussian", "compact_cosine"],
        default="gaussian",
    )
    parser.add_argument("--dndv-kernel-truncate-sigma", type=float, default=3.0)
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=0.10,
        help="Fractional observable scale used to build covariance-whitened degeneracy metrics.",
    )
    parser.add_argument("--shape-skew-sigma", type=float, default=0.20)
    parser.add_argument("--shape-kurt-sigma", type=float, default=0.40)
    parser.add_argument("--dndv-sigma-floor-frac", type=float, default=0.03)
    parser.add_argument("--dndv-bin-corr", type=float, default=0.60)
    parser.add_argument(
        "--log-dndv-low-signal-policy",
        choices=["gaussian", "censored_upper"],
        default="gaussian",
    )
    parser.add_argument("--log-dndv-censor-delta-log", type=float, default=20.0)
    parser.add_argument("--log-dndv-censor-transition", type=float, default=0.25)
    parser.add_argument("--log-dndv-censor-sigma", type=float, default=0.50)
    parser.add_argument("--log-dndv-censor-upper-margin", type=float, default=1.0)

    parser.add_argument("--failure-policy", choices=["stalled_wind", "hard_invalid"], default="stalled_wind")
    parser.add_argument("--stall-velocity-floor-kms", type=float, default=0.0)
    parser.add_argument("--stall-velocity-transition-kms", type=float, default=50.0)
    parser.add_argument("--stall-radius-sigma-kpc", type=float, default=0.25)
    parser.add_argument("--stall-radius-transition-kpc", type=float, default=0.05)
    parser.add_argument(
        "--loading-perturbation-log",
        type=float,
        default=np.log(1.25),
        help="Log-space step for estimating degeneracy with eta_M, eta_M_cold, and eta_E.",
    )
    parser.add_argument("--skip-loading-degeneracy", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args(argv)
    try:
        selected_parameter_names(args)
        selected_truth_case_names(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.values is not None and len(selected_parameter_names(args)) != 1:
        parser.error("--values can only be used when scanning exactly one parameter")
    if args.r_max_kpc <= args.r_star_kpc:
        parser.error("--r-max-kpc must exceed --r-star-kpc")
    if args.step_kpc <= 0.0:
        parser.error("--step-kpc must be positive")
    if args.n_cloud_species < 1:
        parser.error("--n-cloud-species must be >= 1")
    if args.cloud_mass_min <= 0.0 or args.cloud_mass_max <= args.cloud_mass_min:
        parser.error("--cloud-mass-min/max must be positive and increasing")
    if args.loading_perturbation_log <= 0.0:
        parser.error("--loading-perturbation-log must be positive")
    if args.noise_fraction <= 0.0:
        parser.error("--noise-fraction must be positive")
    if args.shape_skew_sigma <= 0.0 or args.shape_kurt_sigma <= 0.0:
        parser.error("--shape-skew-sigma and --shape-kurt-sigma must be positive")
    if args.dndv_sigma_floor_frac <= 0.0:
        parser.error("--dndv-sigma-floor-frac must be positive")
    if not -0.95 < args.dndv_bin_corr < 0.95:
        parser.error("--dndv-bin-corr must be between -0.95 and 0.95")
    return args


def selected_truth_case_names(args: argparse.Namespace) -> list[str]:
    """Return selected truth-case names from new or legacy CLI arguments."""
    choices = list(TRUTH_CASES.keys())
    if args.truth_cases is not None:
        return parse_name_list(args.truth_cases, choices, "--truth-cases")
    if args.fiducial_case == "all":
        return choices
    return [args.fiducial_case]


def selected_parameter_names(args: argparse.Namespace) -> list[str]:
    """Return selected parameter names from new or legacy CLI arguments."""
    choices = list(PARAMETER_SPECS.keys())
    if args.parameters is not None:
        return parse_name_list(args.parameters, choices, "--parameters")
    if args.parameter == "all":
        return choices
    return [args.parameter]


def selected_truth_cases(args: argparse.Namespace):
    """Return truth cases requested by CLI."""
    return [TRUTH_CASES[name] for name in selected_truth_case_names(args)]


def selected_parameter_values(args: argparse.Namespace) -> dict[str, tuple[float, ...]]:
    """Return parameter values requested by CLI."""
    names = selected_parameter_names(args)
    values: dict[str, tuple[float, ...]] = {}
    for name in names:
        spec = PARAMETER_SPECS[name]
        values[name] = parse_values(args.values, spec.values) if len(names) == 1 else spec.values
    return values


def build_model(args: argparse.Namespace, parameter_name: str | None = None, parameter_value: float | None = None):
    """Construct an inference forward model with one fixed parameter overridden."""
    from multiphasegalacticwind.config import WindConfig
    from multiphasegalacticwind.inference import MomentInferenceModel

    cloud_alpha = PARAMETER_SPECS["cloud_alpha"].default
    config_overrides: dict[str, float] = {}
    if parameter_name is not None and parameter_value is not None:
        spec = PARAMETER_SPECS[parameter_name]
        if spec.target == "model":
            if parameter_name != "cloud_alpha":
                raise ValueError(f"Unsupported model-target parameter: {parameter_name}")
            cloud_alpha = float(parameter_value)
        elif spec.target == "config":
            config_overrides[parameter_name] = float(parameter_value)
        else:
            raise ValueError(f"Unsupported parameter target: {spec.target}")

    return MomentInferenceModel(
        sfr=args.sfr,
        r_star_kpc=args.r_star_kpc,
        v_circ=args.v_circ,
        r_max_kpc=args.r_max_kpc,
        step_kpc=args.step_kpc,
        first_step_kpc=args.first_step_kpc,
        n_cloud_species=args.n_cloud_species,
        cloud_mass_range=(args.cloud_mass_min, args.cloud_mass_max),
        cloud_alpha=cloud_alpha,
        integrator_mode=args.integrator_mode,
        integrator_rtol=args.integrator_rtol,
        integrator_atol=args.integrator_atol,
        integrator_max_steps=args.integrator_max_steps,
        observable_set=args.observable_set,
        dndv_num_bins=args.dndv_num_bins,
        dndv_vmin_kms=args.dndv_vmin_kms,
        dndv_vmax_kms=args.dndv_vmax_kms,
        dndv_kernel_sigma_kms=args.dndv_kernel_sigma_kms,
        dndv_kernel=args.dndv_kernel,
        dndv_kernel_truncate_sigma=args.dndv_kernel_truncate_sigma,
        log_dndv_low_signal_policy=args.log_dndv_low_signal_policy,
        log_dndv_censor_delta_log=args.log_dndv_censor_delta_log,
        log_dndv_censor_transition=args.log_dndv_censor_transition,
        log_dndv_censor_sigma=args.log_dndv_censor_sigma,
        log_dndv_censor_upper_margin=args.log_dndv_censor_upper_margin,
        failure_policy=args.failure_policy,
        stall_velocity_floor_kms=args.stall_velocity_floor_kms,
        stall_velocity_transition_kms=args.stall_velocity_transition_kms,
        stall_radius_sigma_kpc=args.stall_radius_sigma_kpc,
        stall_radius_transition_kpc=args.stall_radius_transition_kpc,
        config=WindConfig(**config_overrides),
    )


def shape_from_raw_moments(raw_moments: Sequence[float]) -> dict[str, float]:
    """Convert raw velocity moments into shape summaries."""
    raw = np.asarray(raw_moments, dtype=float)
    if raw.shape[0] < 5 or not np.all(np.isfinite(raw[:5])) or raw[0] <= 0.0:
        return {"logM0": np.nan, "mean_v": np.nan, "sigma_v": np.nan, "skewness": np.nan, "kurtosis": np.nan}

    m0 = raw[0]
    mean_v = raw[1] / m0
    second = raw[2] / m0
    var_v = second - mean_v * mean_v
    if var_v <= 0.0 or not np.isfinite(var_v):
        return {"logM0": float(np.log(m0)), "mean_v": float(mean_v), "sigma_v": np.nan, "skewness": np.nan, "kurtosis": np.nan}
    sigma_v = float(np.sqrt(var_v))
    third = raw[3] / m0
    fourth = raw[4] / m0
    mu3 = third - 3.0 * mean_v * second + 2.0 * mean_v**3
    mu4 = fourth - 4.0 * mean_v * third + 6.0 * mean_v * mean_v * second - 3.0 * mean_v**4
    return {
        "logM0": float(np.log(m0)),
        "mean_v": float(mean_v),
        "sigma_v": sigma_v,
        "skewness": float(mu3 / max(sigma_v**3, 1.0e-24)),
        "kurtosis": float(mu4 / max(sigma_v**4, 1.0e-24)),
    }


def empty_shape() -> dict[str, float]:
    """Return NaN shape summaries for invalid scan points."""
    return {"logM0": np.nan, "mean_v": np.nan, "sigma_v": np.nan, "skewness": np.nan, "kurtosis": np.nan}


def evaluate_model(model: Any, theta: Sequence[float]) -> dict[str, Any]:
    """Evaluate observables and trajectory diagnostics for one theta."""
    import jax.numpy as jnp

    from multiphasegalacticwind.constants import kpc

    theta_arr = np.asarray(theta, dtype=float)
    prediction = model._predict_theta_with_valid_fn(jnp.asarray(theta_arr, dtype=jnp.float64))
    observables = np.asarray(prediction[0], dtype=float)
    raw = np.asarray(prediction[1], dtype=float)
    valid = bool(float(np.asarray(prediction[2])) > 0.5)
    finite = bool(np.all(np.isfinite(observables)) and np.all(np.isfinite(raw)))
    return {
        "observables": observables,
        "raw_moments": raw,
        "valid": valid and finite,
        "first_invalid_r_kpc": float(np.asarray(prediction[4], dtype=float) / kpc),
        "trajectory_status_code": float(np.asarray(prediction[5], dtype=float)),
        "soft_reach_radius_kpc": float(np.asarray(prediction[6], dtype=float) / kpc),
        "min_hot_velocity_kms": float(np.asarray(prediction[7], dtype=float)),
        "stall_penalty": float(np.asarray(prediction[8], dtype=float)),
    }


def observable_distance(observables: np.ndarray, baseline: np.ndarray) -> float:
    """Return a dimensionless RMS observable shift."""
    y = np.asarray(observables, dtype=float)
    y0 = np.asarray(baseline, dtype=float)
    mask = np.isfinite(y) & np.isfinite(y0)
    if not np.any(mask):
        return np.nan
    scale = np.maximum(np.abs(y0[mask]), 1.0)
    return float(np.sqrt(np.mean(((y[mask] - y0[mask]) / scale) ** 2)))


def profile_distance(model: Any, observables: np.ndarray, baseline: np.ndarray) -> float:
    """Return an RMS log-profile distance for binned observables."""
    y = np.asarray(observables, dtype=float)
    y0 = np.asarray(baseline, dtype=float)
    mask = np.isfinite(y) & np.isfinite(y0)
    if not np.any(mask):
        return np.nan
    if model.observable_set == "log_dndv_binned":
        return float(np.sqrt(np.mean((y[mask] - y0[mask]) ** 2)))
    if model.observable_set == "dndv_binned":
        positive = mask & (y > 0.0) & (y0 > 0.0)
        if not np.any(positive):
            return np.nan
        return float(np.sqrt(np.mean((np.log(y[positive]) - np.log(y0[positive])) ** 2)))
    return np.nan


def covariance_precision_for_case(args: argparse.Namespace, model: Any, baseline_observables: np.ndarray) -> np.ndarray:
    """Return the active likelihood precision used for whitened degeneracy metrics."""
    _, covariance = build_synthetic_covariance(
        model,
        baseline_observables,
        noise_fraction=args.noise_fraction,
        shape_skew_sigma=args.shape_skew_sigma,
        shape_kurt_sigma=args.shape_kurt_sigma,
        dndv_sigma_floor_frac=args.dndv_sigma_floor_frac,
        dndv_bin_corr=args.dndv_bin_corr,
    )
    setup = model._observable_likelihood_setup(baseline_observables, covariance)
    return np.asarray(setup["gaussian_cov_inv"], dtype=float)


def loading_basis_for_case(args: argparse.Namespace, theta: np.ndarray, baseline_observables: np.ndarray) -> np.ndarray:
    """Estimate observable derivatives for the three loading parameters."""
    model = build_model(args)
    step = float(args.loading_perturbation_log)
    basis = []
    for idx in range(3):
        plus = np.asarray(theta, dtype=float).copy()
        minus = np.asarray(theta, dtype=float).copy()
        plus[idx] *= np.exp(step)
        minus[idx] /= np.exp(step)
        if idx == 2:
            plus[idx] = min(plus[idx], 0.995)
            minus[idx] = max(minus[idx], 0.02)
        plus_eval = evaluate_model(model, plus)
        minus_eval = evaluate_model(model, minus)
        if not plus_eval["valid"] or not minus_eval["valid"]:
            basis.append(np.full_like(baseline_observables, np.nan, dtype=float))
        else:
            basis.append((plus_eval["observables"] - minus_eval["observables"]) / (2.0 * step))
    return np.vstack(basis).astype(float)


def precision_whitener(precision: np.ndarray) -> np.ndarray:
    """Return W such that ||W x||^2 approximates x^T precision x."""
    p = np.asarray(precision, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.size == 0:
        return np.empty((0, 0), dtype=float)
    sym = 0.5 * (p + p.T)
    try:
        eigval, eigvec = np.linalg.eigh(sym)
    except np.linalg.LinAlgError:
        return np.empty((0, p.shape[0]), dtype=float)
    scale = max(float(np.nanmax(np.abs(eigval))) if eigval.size else 0.0, 1.0)
    keep = np.isfinite(eigval) & (eigval > scale * 1.0e-12)
    if not np.any(keep):
        return np.empty((0, p.shape[0]), dtype=float)
    return np.sqrt(np.clip(eigval[keep], 0.0, None))[:, None] * eigvec[:, keep].T


def loading_subspace_metrics(delta: np.ndarray, basis: np.ndarray, precision: np.ndarray) -> dict[str, float]:
    """Project a microphysics displacement onto the whitened loading subspace."""
    d = np.asarray(delta, dtype=float)
    b = np.asarray(basis, dtype=float)
    p = np.asarray(precision, dtype=float)
    if b.ndim != 2 or p.shape != (d.size, d.size):
        return {
            "fraction": np.nan,
            "orthogonal_chi": np.nan,
            "orthogonal_rms": np.nan,
            "whitened_distance": np.nan,
        }

    basis_good = np.asarray([row for row in b if np.all(np.isfinite(row)) and np.linalg.norm(row) > 0.0], dtype=float)
    if basis_good.size == 0:
        return {
            "fraction": np.nan,
            "orthogonal_chi": np.nan,
            "orthogonal_rms": np.nan,
            "whitened_distance": np.nan,
        }

    mask = np.isfinite(d) & np.all(np.isfinite(basis_good), axis=0)
    if not np.any(mask):
        return {
            "fraction": np.nan,
            "orthogonal_chi": np.nan,
            "orthogonal_rms": np.nan,
            "whitened_distance": np.nan,
        }

    p_sub = p[np.ix_(mask, mask)]
    whitener = precision_whitener(p_sub)
    if whitener.size == 0:
        return {
            "fraction": np.nan,
            "orthogonal_chi": np.nan,
            "orthogonal_rms": np.nan,
            "whitened_distance": np.nan,
        }

    d_w = whitener @ d[mask]
    b_w = whitener @ basis_good[:, mask].T
    total_sq = float(np.dot(d_w, d_w))
    if not np.isfinite(total_sq) or total_sq <= 0.0:
        return {
            "fraction": 0.0,
            "orthogonal_chi": 0.0,
            "orthogonal_rms": 0.0,
            "whitened_distance": 0.0,
        }

    try:
        coeff, *_ = np.linalg.lstsq(b_w, d_w, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "fraction": np.nan,
            "orthogonal_chi": np.nan,
            "orthogonal_rms": np.nan,
            "whitened_distance": np.sqrt(total_sq),
        }
    projection = b_w @ coeff
    residual = d_w - projection
    projection_sq = float(np.dot(projection, projection))
    residual_sq = max(float(np.dot(residual, residual)), 0.0)
    rank = int(np.linalg.matrix_rank(b_w))
    active_dim = int(d_w.size)
    return {
        "fraction": float(np.clip(projection_sq / total_sq, 0.0, 1.0)),
        "orthogonal_chi": float(np.sqrt(residual_sq)),
        "orthogonal_rms": float(np.sqrt(residual_sq / max(active_dim - rank, 1))),
        "whitened_distance": float(np.sqrt(total_sq)),
    }


def max_loading_cosine(delta: np.ndarray, basis: np.ndarray) -> float:
    """Return max absolute cosine similarity with the loading-parameter basis."""
    d = np.asarray(delta, dtype=float)
    b = np.asarray(basis, dtype=float)
    values: list[float] = []
    for row in b:
        mask = np.isfinite(d) & np.isfinite(row)
        if not np.any(mask):
            continue
        d_norm = float(np.linalg.norm(d[mask]))
        row_norm = float(np.linalg.norm(row[mask]))
        if d_norm <= 0.0 or row_norm <= 0.0:
            continue
        values.append(abs(float(np.dot(d[mask], row[mask]) / (d_norm * row_norm))))
    if not values:
        return np.nan
    return float(np.max(values))


def evaluate_screen(args: argparse.Namespace) -> tuple[list[Evaluation], dict[str, Any]]:
    """Run the sensitivity screen and return row-level evaluations."""
    truth_cases = selected_truth_cases(args)
    scans = selected_parameter_values(args)
    baseline_model = build_model(args)

    baselines: dict[str, dict[str, Any]] = {}
    loading_bases: dict[str, np.ndarray] = {}
    loading_precisions: dict[str, np.ndarray] = {}
    for case in truth_cases:
        theta = np.asarray(case.theta, dtype=float)
        baseline = evaluate_model(baseline_model, theta)
        baselines[case.name] = baseline
        if args.skip_loading_degeneracy or not baseline["valid"]:
            loading_bases[case.name] = np.empty((0, baseline_model.observable_dim), dtype=float)
            loading_precisions[case.name] = np.zeros((baseline_model.observable_dim, baseline_model.observable_dim), dtype=float)
        else:
            loading_bases[case.name] = loading_basis_for_case(args, theta, baseline["observables"])
            loading_precisions[case.name] = covariance_precision_for_case(args, baseline_model, baseline["observables"])

    rows: list[Evaluation] = []
    for parameter_name, values in scans.items():
        for value in values:
            try:
                model = build_model(args, parameter_name, value)
                build_error = ""
            except Exception as exc:  # noqa: BLE001 - keep scan failures in the output table.
                model = None
                build_error = str(exc)

            for case in truth_cases:
                theta = np.asarray(case.theta, dtype=float)
                baseline = baselines[case.name]
                if model is None:
                    empty_obs = np.full_like(baseline["observables"], np.nan, dtype=float)
                    empty_raw = np.full_like(baseline["raw_moments"], np.nan, dtype=float)
                    rows.append(
                        Evaluation(
                            parameter_name=parameter_name,
                            parameter_value=float(value),
                            truth_case=case.name,
                            theta=theta,
                            valid=False,
                            trajectory_status_code=np.nan,
                            first_invalid_r_kpc=np.nan,
                            soft_reach_radius_kpc=np.nan,
                            min_hot_velocity_kms=np.nan,
                            stall_penalty=np.nan,
                            observables=empty_obs,
                            raw_moments=empty_raw,
                            shape=shape_from_raw_moments(empty_raw),
                            baseline_observables=baseline["observables"],
                            baseline_raw_moments=baseline["raw_moments"],
                            observable_distance=np.nan,
                            profile_distance=np.nan,
                            loading_degeneracy_cosine=np.nan,
                            loading_subspace_fraction=np.nan,
                            loading_orthogonal_chi=np.nan,
                            loading_orthogonal_rms=np.nan,
                            loading_whitened_distance=np.nan,
                            error=build_error,
                        )
                    )
                    continue

                try:
                    evaluated = evaluate_model(model, theta)
                    obs_delta = evaluated["observables"] - baseline["observables"]
                    usable_distance = bool(evaluated["valid"] and baseline["valid"])
                    subspace = (
                        loading_subspace_metrics(
                            obs_delta,
                            loading_bases[case.name],
                            loading_precisions[case.name],
                        )
                        if usable_distance
                        else {
                            "fraction": np.nan,
                            "orthogonal_chi": np.nan,
                            "orthogonal_rms": np.nan,
                            "whitened_distance": np.nan,
                        }
                    )
                    rows.append(
                        Evaluation(
                            parameter_name=parameter_name,
                            parameter_value=float(value),
                            truth_case=case.name,
                            theta=theta,
                            valid=bool(evaluated["valid"]),
                            trajectory_status_code=float(evaluated["trajectory_status_code"]),
                            first_invalid_r_kpc=float(evaluated["first_invalid_r_kpc"]),
                            soft_reach_radius_kpc=float(evaluated["soft_reach_radius_kpc"]),
                            min_hot_velocity_kms=float(evaluated["min_hot_velocity_kms"]),
                            stall_penalty=float(evaluated["stall_penalty"]),
                            observables=evaluated["observables"],
                            raw_moments=evaluated["raw_moments"],
                            shape=shape_from_raw_moments(evaluated["raw_moments"])
                            if evaluated["valid"]
                            else empty_shape(),
                            baseline_observables=baseline["observables"],
                            baseline_raw_moments=baseline["raw_moments"],
                            observable_distance=observable_distance(evaluated["observables"], baseline["observables"])
                            if usable_distance
                            else np.nan,
                            profile_distance=profile_distance(model, evaluated["observables"], baseline["observables"])
                            if usable_distance
                            else np.nan,
                            loading_degeneracy_cosine=max_loading_cosine(obs_delta, loading_bases[case.name])
                            if usable_distance
                            else np.nan,
                            loading_subspace_fraction=subspace["fraction"],
                            loading_orthogonal_chi=subspace["orthogonal_chi"],
                            loading_orthogonal_rms=subspace["orthogonal_rms"],
                            loading_whitened_distance=subspace["whitened_distance"],
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - one failed scan point should not abort the screen.
                    empty_obs = np.full_like(baseline["observables"], np.nan, dtype=float)
                    empty_raw = np.full_like(baseline["raw_moments"], np.nan, dtype=float)
                    rows.append(
                        Evaluation(
                            parameter_name=parameter_name,
                            parameter_value=float(value),
                            truth_case=case.name,
                            theta=theta,
                            valid=False,
                            trajectory_status_code=np.nan,
                            first_invalid_r_kpc=np.nan,
                            soft_reach_radius_kpc=np.nan,
                            min_hot_velocity_kms=np.nan,
                            stall_penalty=np.nan,
                            observables=empty_obs,
                            raw_moments=empty_raw,
                            shape=shape_from_raw_moments(empty_raw),
                            baseline_observables=baseline["observables"],
                            baseline_raw_moments=baseline["raw_moments"],
                            observable_distance=np.nan,
                            profile_distance=np.nan,
                            loading_degeneracy_cosine=np.nan,
                            loading_subspace_fraction=np.nan,
                            loading_orthogonal_chi=np.nan,
                            loading_orthogonal_rms=np.nan,
                            loading_whitened_distance=np.nan,
                            error=str(exc),
                        )
                    )

            print(f"completed {parameter_name}={value:g}", flush=True)

    metadata = {
        "truth_cases": [case.name for case in truth_cases],
        "observable_names": list(baseline_model.observable_names),
        "baseline_valid": {name: bool(value["valid"]) for name, value in baselines.items()},
        "degeneracy_metric": {
            "description": "covariance-whitened projection of microphysics displacements onto the eta_M, eta_M_cold, eta_E loading subspace",
            "noise_fraction": float(args.noise_fraction),
            "shape_skew_sigma": float(args.shape_skew_sigma),
            "shape_kurt_sigma": float(args.shape_kurt_sigma),
            "dndv_sigma_floor_frac": float(args.dndv_sigma_floor_frac),
            "dndv_bin_corr": float(args.dndv_bin_corr),
        },
    }
    return rows, metadata


def observable_column_name(name: str) -> str:
    """Convert an observable label into a CSV-safe column name."""
    safe = name.replace("/", "_per_").replace("@", "_at_").replace(" ", "_")
    safe = safe.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    safe = safe.replace(",", "_").replace("=", "_").replace("-", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"obs_{safe}"


def write_rows_csv(output_dir: str, observable_names: Sequence[str], rows: Sequence[Evaluation]) -> str:
    """Write one row per scan point and truth case."""
    obs_columns = [observable_column_name(name) for name in observable_names]
    fieldnames = [
        "parameter_name",
        "parameter_value",
        "truth_case",
        "eta_M",
        "eta_M_cold",
        "eta_E",
        "valid",
        "trajectory_status_code",
        "first_invalid_r_kpc",
        "soft_reach_radius_kpc",
        "min_hot_velocity_kms",
        "stall_penalty",
        "observable_distance",
        "profile_distance",
        "loading_degeneracy_cosine",
        "loading_subspace_fraction",
        "loading_orthogonal_chi",
        "loading_orthogonal_rms",
        "loading_whitened_distance",
        "logM0",
        "mean_v",
        "sigma_v",
        "skewness",
        "kurtosis",
        "delta_logM0",
        "delta_mean_v",
        "delta_sigma_v",
        "delta_skewness",
        "delta_kurtosis",
        *obs_columns,
        "error",
    ]
    path = os.path.join(output_dir, "trml_sensitivity_rows.csv")
    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            baseline_shape = shape_from_raw_moments(row.baseline_raw_moments)
            out: dict[str, Any] = {
                "parameter_name": row.parameter_name,
                "parameter_value": row.parameter_value,
                "truth_case": row.truth_case,
                "eta_M": row.theta[0],
                "eta_M_cold": row.theta[1],
                "eta_E": row.theta[2],
                "valid": row.valid,
                "trajectory_status_code": row.trajectory_status_code,
                "first_invalid_r_kpc": row.first_invalid_r_kpc,
                "soft_reach_radius_kpc": row.soft_reach_radius_kpc,
                "min_hot_velocity_kms": row.min_hot_velocity_kms,
                "stall_penalty": row.stall_penalty,
                "observable_distance": row.observable_distance,
                "profile_distance": row.profile_distance,
                "loading_degeneracy_cosine": row.loading_degeneracy_cosine,
                "loading_subspace_fraction": row.loading_subspace_fraction,
                "loading_orthogonal_chi": row.loading_orthogonal_chi,
                "loading_orthogonal_rms": row.loading_orthogonal_rms,
                "loading_whitened_distance": row.loading_whitened_distance,
                "logM0": row.shape["logM0"],
                "mean_v": row.shape["mean_v"],
                "sigma_v": row.shape["sigma_v"],
                "skewness": row.shape["skewness"],
                "kurtosis": row.shape["kurtosis"],
                "delta_logM0": row.shape["logM0"] - baseline_shape["logM0"],
                "delta_mean_v": row.shape["mean_v"] - baseline_shape["mean_v"],
                "delta_sigma_v": row.shape["sigma_v"] - baseline_shape["sigma_v"],
                "delta_skewness": row.shape["skewness"] - baseline_shape["skewness"],
                "delta_kurtosis": row.shape["kurtosis"] - baseline_shape["kurtosis"],
                "error": row.error,
            }
            for name, value in zip(obs_columns, row.observables):
                out[name] = value
            writer.writerow(out)
    return path


def summarize_rows(rows: Sequence[Evaluation]) -> list[dict[str, Any]]:
    """Aggregate row-level sensitivity metrics by parameter."""
    def nanmax_abs(values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return np.nan
        return float(np.max(np.abs(finite)))

    summaries: list[dict[str, Any]] = []
    for parameter_name in sorted({row.parameter_name for row in rows}):
        group = [row for row in rows if row.parameter_name == parameter_name]
        obs_dist = np.asarray([row.observable_distance for row in group], dtype=float)
        profile_dist = np.asarray([row.profile_distance for row in group], dtype=float)
        cosines = np.asarray([row.loading_degeneracy_cosine for row in group], dtype=float)
        subspace_fraction = np.asarray([row.loading_subspace_fraction for row in group], dtype=float)
        orthogonal_chi = np.asarray([row.loading_orthogonal_chi for row in group], dtype=float)
        orthogonal_rms = np.asarray([row.loading_orthogonal_rms for row in group], dtype=float)
        whitened_distance = np.asarray([row.loading_whitened_distance for row in group], dtype=float)
        valid = np.asarray([row.valid for row in group], dtype=bool)
        delta_shape = {
            name: np.asarray(
                [row.shape[name] - shape_from_raw_moments(row.baseline_raw_moments)[name] for row in group],
                dtype=float,
            )
            for name in ("logM0", "mean_v", "sigma_v", "skewness", "kurtosis")
        }

        max_observable_distance = float(np.nanmax(obs_dist)) if np.any(np.isfinite(obs_dist)) else np.nan
        median_observable_distance = float(np.nanmedian(obs_dist)) if np.any(np.isfinite(obs_dist)) else np.nan
        max_profile_distance = float(np.nanmax(profile_dist)) if np.any(np.isfinite(profile_dist)) else np.nan
        max_loading_cosine = float(np.nanmax(cosines)) if np.any(np.isfinite(cosines)) else np.nan
        max_loading_subspace_fraction = (
            float(np.nanmax(subspace_fraction)) if np.any(np.isfinite(subspace_fraction)) else np.nan
        )
        median_loading_subspace_fraction = (
            float(np.nanmedian(subspace_fraction)) if np.any(np.isfinite(subspace_fraction)) else np.nan
        )
        max_loading_orthogonal_chi = (
            float(np.nanmax(orthogonal_chi)) if np.any(np.isfinite(orthogonal_chi)) else np.nan
        )
        median_loading_orthogonal_chi = (
            float(np.nanmedian(orthogonal_chi)) if np.any(np.isfinite(orthogonal_chi)) else np.nan
        )
        max_loading_orthogonal_rms = (
            float(np.nanmax(orthogonal_rms)) if np.any(np.isfinite(orthogonal_rms)) else np.nan
        )
        max_loading_whitened_distance = (
            float(np.nanmax(whitened_distance)) if np.any(np.isfinite(whitened_distance)) else np.nan
        )
        summaries.append(
            {
                "parameter_name": parameter_name,
                "num_rows": len(group),
                "num_values": len({row.parameter_value for row in group}),
                "valid_fraction": float(np.mean(valid)) if valid.size else np.nan,
                "num_invalid": int(np.sum(~valid)),
                "max_observable_distance": max_observable_distance,
                "median_observable_distance": median_observable_distance,
                "max_profile_distance": max_profile_distance,
                "max_loading_degeneracy_cosine": max_loading_cosine,
                "max_loading_subspace_fraction": max_loading_subspace_fraction,
                "median_loading_subspace_fraction": median_loading_subspace_fraction,
                "max_loading_orthogonal_chi": max_loading_orthogonal_chi,
                "median_loading_orthogonal_chi": median_loading_orthogonal_chi,
                "max_loading_orthogonal_rms": max_loading_orthogonal_rms,
                "max_loading_whitened_distance": max_loading_whitened_distance,
                "max_abs_delta_logM0": nanmax_abs(delta_shape["logM0"]),
                "max_abs_delta_mean_v": nanmax_abs(delta_shape["mean_v"]),
                "max_abs_delta_sigma_v": nanmax_abs(delta_shape["sigma_v"]),
                "max_abs_delta_skewness": nanmax_abs(delta_shape["skewness"]),
                "max_abs_delta_kurtosis": nanmax_abs(delta_shape["kurtosis"]),
                "leverage_rank_score": max_observable_distance,
            }
        )
    summaries.sort(key=lambda item: (-np.nan_to_num(item["leverage_rank_score"], nan=-np.inf), item["parameter_name"]))
    return summaries


def write_summary_csv(output_dir: str, summaries: Sequence[dict[str, Any]]) -> str:
    """Write the parameter-ranked sensitivity summary."""
    fieldnames = [
        "rank",
        "parameter_name",
        "num_rows",
        "num_values",
        "valid_fraction",
        "num_invalid",
        "max_observable_distance",
        "median_observable_distance",
        "max_profile_distance",
        "max_loading_degeneracy_cosine",
        "max_loading_subspace_fraction",
        "median_loading_subspace_fraction",
        "max_loading_orthogonal_chi",
        "median_loading_orthogonal_chi",
        "max_loading_orthogonal_rms",
        "max_loading_whitened_distance",
        "max_abs_delta_logM0",
        "max_abs_delta_mean_v",
        "max_abs_delta_sigma_v",
        "max_abs_delta_skewness",
        "max_abs_delta_kurtosis",
        "leverage_rank_score",
    ]
    path = os.path.join(output_dir, "trml_sensitivity_summary.csv")
    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, summary in enumerate(summaries, start=1):
            writer.writerow({"rank": rank, **summary})
    return path


def write_npz(output_dir: str, observable_names: Sequence[str], rows: Sequence[Evaluation]) -> str:
    """Write machine-readable scan arrays."""
    path = os.path.join(output_dir, "trml_sensitivity_samples.npz")
    np.savez(
        path,
        parameter_name=np.asarray([row.parameter_name for row in rows], dtype=str),
        parameter_value=np.asarray([row.parameter_value for row in rows], dtype=float),
        truth_case=np.asarray([row.truth_case for row in rows], dtype=str),
        theta=np.vstack([row.theta for row in rows]).astype(float),
        valid=np.asarray([row.valid for row in rows], dtype=bool),
        trajectory_status_code=np.asarray([row.trajectory_status_code for row in rows], dtype=float),
        first_invalid_r_kpc=np.asarray([row.first_invalid_r_kpc for row in rows], dtype=float),
        soft_reach_radius_kpc=np.asarray([row.soft_reach_radius_kpc for row in rows], dtype=float),
        min_hot_velocity_kms=np.asarray([row.min_hot_velocity_kms for row in rows], dtype=float),
        stall_penalty=np.asarray([row.stall_penalty for row in rows], dtype=float),
        observables=np.vstack([row.observables for row in rows]).astype(float),
        baseline_observables=np.vstack([row.baseline_observables for row in rows]).astype(float),
        raw_moments=np.vstack([row.raw_moments for row in rows]).astype(float),
        baseline_raw_moments=np.vstack([row.baseline_raw_moments for row in rows]).astype(float),
        observable_distance=np.asarray([row.observable_distance for row in rows], dtype=float),
        profile_distance=np.asarray([row.profile_distance for row in rows], dtype=float),
        loading_degeneracy_cosine=np.asarray([row.loading_degeneracy_cosine for row in rows], dtype=float),
        loading_subspace_fraction=np.asarray([row.loading_subspace_fraction for row in rows], dtype=float),
        loading_orthogonal_chi=np.asarray([row.loading_orthogonal_chi for row in rows], dtype=float),
        loading_orthogonal_rms=np.asarray([row.loading_orthogonal_rms for row in rows], dtype=float),
        loading_whitened_distance=np.asarray([row.loading_whitened_distance for row in rows], dtype=float),
        observable_names=np.asarray(observable_names, dtype=str),
        theta_names=np.asarray(PARAM_NAMES, dtype=str),
    )
    return path


def make_ranking_plot(output_dir: str, summaries: Sequence[dict[str, Any]]) -> str:
    """Plot parameter ranking by max observable distance."""
    path = os.path.join(output_dir, "trml_sensitivity_ranking.png")
    top = list(summaries[: min(12, len(summaries))])
    names = [item["parameter_name"] for item in top][::-1]
    scores = [item["leverage_rank_score"] for item in top][::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.35 * len(names) + 1.2)), constrained_layout=True)
    ax.barh(names, scores, color="tab:blue", alpha=0.8)
    ax.set_xlabel("max dimensionless observable shift")
    ax.set_ylabel("fixed parameter")
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle("TRML/cloud one-at-a-time sensitivity ranking")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_metadata(
    output_dir: str,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    output_files: dict[str, str],
    runtime_seconds: float,
) -> str:
    """Write run metadata."""
    payload = {
        "created_by": "examples/trml_sensitivity_screen.py",
        "runtime_seconds": float(runtime_seconds),
        "arguments": vars(args),
        "parameter_specs": {
            name: {
                "target": spec.target,
                "default": spec.default,
                "values": list(spec.values),
                "description": spec.description,
            }
            for name, spec in PARAMETER_SPECS.items()
        },
        "screen_metadata": metadata,
        "output_files": output_files,
        "notes": [
            "This is a forward-model sensitivity screen, not expanded inference.",
            "loading_degeneracy_cosine is a local observable-space cosine against finite-difference eta_M, eta_M_cold, and eta_E directions.",
            "loading_subspace_fraction is the covariance-whitened fraction of the microphysics displacement explained by the full three-loading subspace.",
            "loading_orthogonal_chi is the covariance-whitened residual norm left after projecting onto that loading subspace.",
            "Generated outputs are diagnostic artifacts and should not be committed by default.",
        ],
    }
    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_jax_platform(args.jax_platform)
    os.makedirs(args.output, exist_ok=True)

    start = time.perf_counter()
    rows, metadata = evaluate_screen(args)
    summaries = summarize_rows(rows)

    output_files: dict[str, str] = {}
    output_files["rows_csv"] = write_rows_csv(args.output, metadata["observable_names"], rows)
    output_files["summary_csv"] = write_summary_csv(args.output, summaries)
    output_files["npz"] = write_npz(args.output, metadata["observable_names"], rows)
    if not args.no_plots:
        output_files["ranking_plot"] = make_ranking_plot(args.output, summaries)
    output_files["metadata"] = write_metadata(
        args.output,
        args,
        metadata,
        output_files,
        runtime_seconds=time.perf_counter() - start,
    )

    print(f"Saved TRML sensitivity screen to {args.output}")
    if summaries:
        best = summaries[0]
        print(
            "Top leverage parameter: "
            f"{best['parameter_name']} (score={best['leverage_rank_score']:.4g}, "
            f"valid_fraction={best['valid_fraction']:.3f})"
        )


if __name__ == "__main__":
    main()
