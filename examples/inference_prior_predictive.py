#!/usr/bin/env python3
"""Prior predictive atlas for baseline and restricted expanded inference models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind.constants import kpc


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "inference_prior_predictive")
DEFAULT_CLASSY_PROFILES_PATH = os.path.join(REPO_ROOT, "multiphasegalacticwind", "data", "classy_profiles.npz")
PARAM_NAMES = ("eta_M", "eta_M_cold", "eta_E")
EXPANDED_PARAMETER_CHOICES = ("none", "a_mix", "a_mix_beta_chi")
ETA_E_MAX = 0.999
DEFAULT_PRIOR_BOUNDS = {
    "eta_M": (0.03, 3.0),
    "eta_M_cold": (0.001, 10.0),
    "eta_E": (0.05, ETA_E_MAX),
    "A_mix": (0.5, 2.0),
    "beta_chi_mix": (-0.5, 0.5),
}
DNDV_PLOT_YMIN = 1.0e12


def configure_matplotlib_for_paper(use_tex: bool = True) -> None:
    """Apply paper-style Matplotlib settings for generated atlas figures."""
    if use_tex and shutil.which("latex") is None:
        raise RuntimeError("LaTeX rendering requested, but no 'latex' executable was found on PATH.")

    matplotlib.rcParams.update(
        {
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "lines.dash_capstyle": "round",
            "text.usetex": bool(use_tex),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    if use_tex:
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


@dataclass
class SampleResult:
    """One prior predictive sample outcome."""

    sample_id: int
    theta: np.ndarray
    valid: bool
    first_invalid_r_kpc: float
    observables: np.ndarray
    raw_moments: np.ndarray
    v_at_10kpc: float = np.nan
    mass_loading_at_10kpc: float = np.nan
    L_hot_cgs: float = np.nan
    L_interface_cgs: float = np.nan
    error: str = ""


def configure_jax_platform(platform: str) -> str:
    """Configure JAX backend environment variables before JAX-heavy imports."""
    requested = platform.strip().lower()
    if requested != "cpu":
        raise ValueError("Only --jax-platform cpu is supported on this machine for this harness.")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    return requested


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the prior predictive atlas."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--num-samples", type=int, default=20, help="Number of prior samples to evaluate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--observable-set",
        choices=["m0_m1_m2", "logm0_mean_sigma_skew_kurt", "dndv_binned"],
        default="logm0_mean_sigma_skew_kurt",
        help="Observable vector to compute.",
    )
    parser.add_argument("--sfr", type=float, default=20.0, help="Star formation rate [Msun/yr]")
    parser.add_argument("--v-circ", type=float, default=150.0, help="Circular velocity [km/s]")
    parser.add_argument("--r-star-kpc", type=float, default=0.3, help="Launch/sonic radius [kpc]")
    parser.add_argument("--r-max-kpc", type=float, default=30.0, help="Outer integration radius [kpc]")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--jax-platform", default="cpu", choices=["cpu"], help="JAX backend platform")

    parser.add_argument("--n-cloud-species", type=int, default=13)
    parser.add_argument("--cloud-mass-min", type=float, default=1.0)
    parser.add_argument("--cloud-mass-max", type=float, default=1.0e6)
    parser.add_argument("--cloud-alpha", type=float, default=2.0)
    parser.add_argument("--step-kpc", type=float, default=0.02)
    parser.add_argument("--first-step-kpc", type=float, default=1.0e-12)
    parser.add_argument("--integrator-mode", choices=["rk2", "rk3", "rk4", "tsit5"], default="rk4")
    parser.add_argument("--integrator-rtol", type=float, default=1e-5)
    parser.add_argument("--integrator-atol", type=float, default=1e-8)
    parser.add_argument("--integrator-max-steps", type=int, default=131072)
    parser.add_argument("--expanded-parameters", choices=EXPANDED_PARAMETER_CHOICES, default="none")
    parser.add_argument("--beta-chi-max-abs", type=float, default=0.75)
    parser.add_argument("--mixing-chi-pivot", type=float, default=100.0)

    parser.add_argument("--dndv-num-bins", type=int, default=25)
    parser.add_argument("--dndv-vmin-kms", type=float, default=0.0)
    parser.add_argument("--dndv-vmax-kms", type=float, default=1200.0)
    parser.add_argument(
        "--dndv-kernel-sigma-kms",
        type=float,
        default=None,
        help="Gaussian kernel width for dN/dv projection [km/s]. Default: half-bin width.",
    )
    parser.add_argument(
        "--dndv-kernel",
        choices=["gaussian", "truncated_gaussian", "compact_cosine"],
        default="gaussian",
        help="Velocity kernel for binned dN/dv projection.",
    )
    parser.add_argument(
        "--dndv-kernel-truncate-sigma",
        type=float,
        default=3.0,
        help="Finite support |dv| cutoff in sigma units for --dndv-kernel truncated_gaussian.",
    )
    parser.add_argument(
        "--no-usetex",
        action="store_true",
        help="Disable Matplotlib LaTeX text rendering. Default uses LaTeX.",
    )
    parser.add_argument(
        "--overlay-classy",
        action="store_true",
        help="Overlay processed CLASSY dN/dv profiles on the dndv_binned envelope plot.",
    )
    parser.add_argument(
        "--classy-profiles-path",
        default=DEFAULT_CLASSY_PROFILES_PATH,
        help="Processed CLASSY profile NPZ used when --overlay-classy is set.",
    )

    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.dndv_kernel_truncate_sigma <= 0.0:
        parser.error("--dndv-kernel-truncate-sigma must be positive")
    if args.beta_chi_max_abs <= 0.0:
        parser.error("--beta-chi-max-abs must be positive")
    if args.mixing_chi_pivot <= 0.0:
        parser.error("--mixing-chi-pivot must be positive")
    return args


def parameter_names_for_mode(expanded_parameters: str) -> tuple[str, ...]:
    """Return active parameter names for a prior-predictive mode."""
    if expanded_parameters == "a_mix":
        return (*PARAM_NAMES, "A_mix")
    if expanded_parameters == "a_mix_beta_chi":
        return (*PARAM_NAMES, "A_mix", "beta_chi_mix")
    return PARAM_NAMES


def sample_prior(
    rng: np.random.Generator,
    num_samples: int,
    expanded_parameters: str = "none",
    beta_chi_max_abs: float = 0.75,
) -> np.ndarray:
    """Draw samples for the active prior-predictive parameter set."""
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    parameter_names = parameter_names_for_mode(expanded_parameters)
    samples = np.empty((int(num_samples), len(parameter_names)), dtype=float)
    for i, name in enumerate(parameter_names):
        lo, hi = DEFAULT_PRIOR_BOUNDS[name]
        if name == "beta_chi_mix":
            lo = max(float(lo), -float(beta_chi_max_abs))
            hi = min(float(hi), float(beta_chi_max_abs))
            samples[:, i] = rng.uniform(lo, hi, size=int(num_samples))
        else:
            log_lo = np.log10(lo)
            log_hi = np.log10(hi)
            samples[:, i] = 10.0 ** rng.uniform(log_lo, log_hi, size=int(num_samples))

    samples[:, 2] = np.minimum(samples[:, 2], np.nextafter(ETA_E_MAX, 0.0))
    return samples


def build_model(args: argparse.Namespace):
    """Construct the active inference forward model."""
    from multiphasegalacticwind.inference import MomentInferenceModel

    return MomentInferenceModel(
        sfr=args.sfr,
        r_star_kpc=args.r_star_kpc,
        v_circ=args.v_circ,
        r_max_kpc=args.r_max_kpc,
        step_kpc=args.step_kpc,
        first_step_kpc=args.first_step_kpc,
        n_cloud_species=args.n_cloud_species,
        cloud_mass_range=(args.cloud_mass_min, args.cloud_mass_max),
        cloud_alpha=args.cloud_alpha,
        integrator_mode=args.integrator_mode,
        integrator_rtol=args.integrator_rtol,
        integrator_atol=args.integrator_atol,
        integrator_max_steps=args.integrator_max_steps,
        expanded_parameters=args.expanded_parameters,
        beta_chi_max_abs=args.beta_chi_max_abs,
        mixing_chi_pivot=args.mixing_chi_pivot,
        observable_set=args.observable_set,
        dndv_num_bins=args.dndv_num_bins,
        dndv_vmin_kms=args.dndv_vmin_kms,
        dndv_vmax_kms=args.dndv_vmax_kms,
        dndv_kernel_sigma_kms=args.dndv_kernel_sigma_kms,
        dndv_kernel=args.dndv_kernel,
        dndv_kernel_truncate_sigma=args.dndv_kernel_truncate_sigma,
    )


def _empty_result(model: Any, theta: np.ndarray, sample_id: int, error: str) -> SampleResult:
    observable_dim = int(getattr(model, "observable_dim", 0))
    return SampleResult(
        sample_id=int(sample_id),
        theta=np.asarray(theta, dtype=float),
        valid=False,
        first_invalid_r_kpc=np.nan,
        observables=np.full((observable_dim,), np.nan, dtype=float),
        raw_moments=np.full((5,), np.nan, dtype=float),
        error=error,
    )


def evaluate_sample(model: Any, theta: Sequence[float], sample_id: int) -> SampleResult:
    """Evaluate one parameter vector and return validity plus observables."""
    theta_arr = np.asarray(theta, dtype=float)
    theta_dim = len(model.parameter_names()) if hasattr(model, "parameter_names") else len(PARAM_NAMES)
    if theta_arr.shape != (theta_dim,):
        return _empty_result(model, theta_arr, sample_id, f"theta must have shape ({theta_dim},), got {theta_arr.shape}")

    try:
        import jax.numpy as jnp

        predictor = getattr(model, "_predict_theta_with_valid_fn", None)
        if predictor is None:
            observables = np.asarray(model.predict_observables(theta_arr), dtype=float)
            raw_moments = np.asarray(model.predict_raw_moments(theta_arr), dtype=float)
            valid = bool(np.all(np.isfinite(observables)) and np.all(np.isfinite(raw_moments)) and raw_moments[0] > 0.0)
            first_invalid_r_kpc = np.nan
        else:
            prediction = predictor(jnp.asarray(theta_arr, dtype=jnp.float64))
            observables_jax, raw_jax, valid_jax, _barrier, first_invalid_r = prediction[:5]
            observables = np.asarray(observables_jax, dtype=float)
            raw_moments = np.asarray(raw_jax, dtype=float)
            valid = bool(float(np.asarray(valid_jax)) > 0.5)
            first_invalid_r_kpc = float(np.asarray(first_invalid_r, dtype=float) / kpc)

        valid = valid and bool(np.all(np.isfinite(observables)) and np.all(np.isfinite(raw_moments)))
        if not valid:
            observables = np.full((int(getattr(model, "observable_dim", observables.size)),), np.nan, dtype=float)
            raw_moments = np.full((5,), np.nan, dtype=float)

        return SampleResult(
            sample_id=int(sample_id),
            theta=theta_arr,
            valid=valid,
            first_invalid_r_kpc=first_invalid_r_kpc,
            observables=observables,
            raw_moments=raw_moments,
        )
    except Exception as exc:  # noqa: BLE001 - per-sample failures must not abort the atlas.
        return _empty_result(model, theta_arr, sample_id, str(exc))


def _shape_from_raw_moments(raw_moments: Sequence[float]) -> dict[str, float]:
    raw = np.asarray(raw_moments, dtype=float)
    if raw.shape[0] < 5 or not np.all(np.isfinite(raw[:5])) or raw[0] <= 0.0:
        return {
            "logM0": np.nan,
            "mean_v": np.nan,
            "sigma_v": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
        }

    m0 = raw[0]
    mean_v = raw[1] / m0
    second = raw[2] / m0
    var_v = second - mean_v * mean_v
    if var_v <= 0.0 or not np.isfinite(var_v):
        sigma_v = np.nan
        skewness = np.nan
        kurtosis = np.nan
    else:
        sigma_v = float(np.sqrt(var_v))
        third = raw[3] / m0
        fourth = raw[4] / m0
        mu3 = third - 3.0 * mean_v * second + 2.0 * mean_v**3
        mu4 = fourth - 4.0 * mean_v * third + 6.0 * mean_v * mean_v * second - 3.0 * mean_v**4
        skewness = float(mu3 / max(sigma_v**3, 1e-24))
        kurtosis = float(mu4 / max(sigma_v**4, 1e-24))

    return {
        "logM0": float(np.log(m0)),
        "mean_v": float(mean_v),
        "sigma_v": float(sigma_v),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def _observable_column_name(name: str) -> str:
    safe = name.replace("/", "_per_").replace("@", "at_").replace(" ", "_")
    safe = safe.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    safe = safe.replace(",", "_").replace("=", "_").replace("-", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"obs_{safe}"


def _arrays_from_results(results: Sequence[SampleResult]) -> dict[str, np.ndarray]:
    return {
        "theta_samples": np.vstack([result.theta for result in results]).astype(float),
        "valid": np.asarray([result.valid for result in results], dtype=bool),
        "first_invalid_r_kpc": np.asarray([result.first_invalid_r_kpc for result in results], dtype=float),
        "observables": np.vstack([result.observables for result in results]).astype(float),
        "raw_moments": np.vstack([result.raw_moments for result in results]).astype(float),
        "v_at_10kpc": np.asarray([result.v_at_10kpc for result in results], dtype=float),
        "mass_loading_at_10kpc": np.asarray([result.mass_loading_at_10kpc for result in results], dtype=float),
        "L_hot_cgs": np.asarray([result.L_hot_cgs for result in results], dtype=float),
        "L_interface_cgs": np.asarray([result.L_interface_cgs for result in results], dtype=float),
        "errors": np.asarray([result.error for result in results], dtype=str),
    }


def write_npz(output_dir: str, model: Any, results: Sequence[SampleResult]) -> str:
    """Write machine-readable sample arrays."""
    arrays = _arrays_from_results(results)
    path = os.path.join(output_dir, "prior_predictive_samples.npz")
    parameter_names = model.parameter_names() if hasattr(model, "parameter_names") else PARAM_NAMES
    np.savez(
        path,
        **arrays,
        theta_names=np.asarray(parameter_names, dtype=str),
        observable_names=np.asarray(model.observable_names, dtype=str),
        observable_set=np.asarray(model.observable_set, dtype=str),
    )
    return path


def write_csv_summary(output_dir: str, model: Any, results: Sequence[SampleResult]) -> str:
    """Write a compact one-row-per-sample CSV summary."""
    parameter_names = model.parameter_names() if hasattr(model, "parameter_names") else PARAM_NAMES
    observable_columns = [_observable_column_name(name) for name in model.observable_names]
    fieldnames = [
        "sample_id",
        *parameter_names,
        "valid",
        "first_invalid_r_kpc",
        "observable_set",
        "M0",
        "M1",
        "M2",
        "logM0",
        "mean_v",
        "sigma_v",
        "skewness",
        "kurtosis",
        *observable_columns,
        "v_at_10kpc",
        "mass_loading_at_10kpc",
        "L_hot_cgs",
        "L_interface_cgs",
        "error",
    ]

    path = os.path.join(output_dir, "prior_predictive_summary.csv")
    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            shape = _shape_from_raw_moments(result.raw_moments)
            raw = np.asarray(result.raw_moments, dtype=float)
            row: dict[str, Any] = {
                "sample_id": result.sample_id,
                "valid": result.valid,
                "first_invalid_r_kpc": result.first_invalid_r_kpc,
                "observable_set": model.observable_set,
                "M0": raw[0] if raw.size > 0 else np.nan,
                "M1": raw[1] if raw.size > 1 else np.nan,
                "M2": raw[2] if raw.size > 2 else np.nan,
                **shape,
                "v_at_10kpc": result.v_at_10kpc,
                "mass_loading_at_10kpc": result.mass_loading_at_10kpc,
                "L_hot_cgs": result.L_hot_cgs,
                "L_interface_cgs": result.L_interface_cgs,
                "error": result.error,
            }
            for name, value in zip(parameter_names, result.theta):
                row[name] = value
            for name, value in zip(observable_columns, result.observables):
                row[name] = value
            writer.writerow(row)
    return path


def _plot_empty(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _finite_valid_mask(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return valid & np.all(np.isfinite(arr), axis=1)


def load_classy_profiles(path: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load processed CLASSY dN/dv profiles as positive outflow speeds."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CLASSY profile file not found: {path}")

    data = np.load(path)
    required = {"object_ids", "profile_offsets", "velocity_kms", "dndv_cm2_per_kms"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"CLASSY profile file is missing required arrays: {sorted(missing)}")

    object_ids = np.asarray(data["object_ids"], dtype=str)
    offsets = np.asarray(data["profile_offsets"], dtype=int)
    velocity = np.asarray(data["velocity_kms"], dtype=float)
    dndv = np.asarray(data["dndv_cm2_per_kms"], dtype=float)
    if offsets.shape != (object_ids.size + 1,):
        raise ValueError("profile_offsets must have length len(object_ids) + 1")
    if velocity.shape != dndv.shape:
        raise ValueError("velocity_kms and dndv_cm2_per_kms must have matching shapes")

    profiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for object_id, start, stop in zip(object_ids, offsets[:-1], offsets[1:]):
        v = np.abs(velocity[start:stop])
        y = dndv[start:stop]
        mask = np.isfinite(v) & np.isfinite(y) & (y > 0.0)
        if not np.any(mask):
            continue
        order = np.argsort(v[mask])
        profiles.append((str(object_id), v[mask][order], y[mask][order]))
    return profiles


def _plot_validity(output_dir: str, theta: np.ndarray, valid: np.ndarray) -> str:
    path = os.path.join(output_dir, "prior_predictive_validity.png")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    pairs = [(0, 1), (0, 2), (1, 2)]
    labels = [r"$\eta_M$", r"$\eta_{M,\mathrm{cold}}$", r"$\eta_E$"]
    colors = np.where(valid, "tab:green", "tab:red")
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(theta[:, i], theta[:, j], c=colors, s=30, alpha=0.8, edgecolor="black", linewidth=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(alpha=0.25)
    fig.suptitle("Prior predictive validity: green=valid, red=invalid")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_observable_histograms(output_dir: str, raw_moments: np.ndarray, valid: np.ndarray) -> str:
    path = os.path.join(output_dir, "prior_predictive_observable_histograms.png")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    shape_rows = [_shape_from_raw_moments(row) for row in raw_moments]
    metrics = {
        "M0": raw_moments[:, 0],
        "mean_v": np.asarray([row["mean_v"] for row in shape_rows], dtype=float),
        "sigma_v": np.asarray([row["sigma_v"] for row in shape_rows], dtype=float),
    }
    labels = {"M0": "M0", "mean_v": "mean velocity [km/s]", "sigma_v": "velocity width [km/s]"}
    for ax, (name, values) in zip(axes, metrics.items()):
        finite = values[valid & np.isfinite(values)]
        if finite.size == 0:
            _plot_empty(ax, f"No valid {name}")
            continue
        if finite.size == 1 or float(np.nanmin(finite)) == float(np.nanmax(finite)):
            ax.axvline(float(finite[0]), color="tab:blue", lw=2.0)
            ax.set_ylim(0.0, 1.0)
        else:
            ax.hist(finite, bins=min(20, max(5, finite.size)), color="tab:blue", alpha=0.75)
        ax.set_xlabel(labels[name])
        ax.set_ylabel("count")
        if name == "M0" and np.all(finite > 0.0):
            ax.set_xscale("log")
        ax.grid(alpha=0.25)
    fig.suptitle("Prior predictive observable summaries")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_parameter_scatter(output_dir: str, theta: np.ndarray, raw_moments: np.ndarray, valid: np.ndarray) -> str:
    path = os.path.join(output_dir, "prior_predictive_parameter_scatter.png")
    shape_rows = [_shape_from_raw_moments(row) for row in raw_moments]
    y_values = [
        ("M0", raw_moments[:, 0]),
        ("mean_v", np.asarray([row["mean_v"] for row in shape_rows], dtype=float)),
        ("sigma_v", np.asarray([row["sigma_v"] for row in shape_rows], dtype=float)),
    ]
    x_labels = [r"$\eta_M$", r"$\eta_{M,\mathrm{cold}}$", r"$\eta_E$"]

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.0), constrained_layout=True)
    for row, (ylabel, y) in enumerate(y_values):
        for col, xlabel in enumerate(x_labels):
            ax = axes[row, col]
            mask = valid & np.isfinite(y) & np.isfinite(theta[:, col])
            if not np.any(mask):
                _plot_empty(ax, "No valid samples")
                continue
            ax.scatter(theta[mask, col], y[mask], s=24, alpha=0.75, color="tab:purple")
            ax.set_xscale("log")
            if ylabel == "M0" and np.all(y[mask] > 0.0):
                ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    fig.suptitle("Parameter versus observable prior predictive trends")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_classy_profile_overlay(
    ax: plt.Axes,
    profiles: Sequence[tuple[str, np.ndarray, np.ndarray]] | None,
) -> float:
    if not profiles:
        return np.nan

    y_max = np.nan
    for i, (_object_id, velocity_kms, dndv) in enumerate(profiles):
        label = "CLASSY profiles" if i == 0 else None
        ax.plot(
            velocity_kms,
            dndv,
            color="0.2",
            lw=0.45,
            alpha=0.28,
            zorder=1,
            label=label,
        )
        finite = dndv[np.isfinite(dndv) & (dndv > 0.0)]
        if finite.size:
            y_max = np.nanmax([y_max, float(np.nanmax(finite))])
    return float(y_max)


def _plot_dndv_envelope(
    output_dir: str,
    model: Any,
    observables: np.ndarray,
    valid: np.ndarray,
    classy_profiles: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> str | None:
    if model.observable_set != "dndv_binned":
        return None

    path = os.path.join(output_dir, "prior_predictive_dndv_envelope.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    classy_ymax = _plot_classy_profile_overlay(ax, classy_profiles)
    mask = _finite_valid_mask(observables, valid)
    if not np.any(mask):
        _plot_empty(ax, "No valid dN/dv samples")
    else:
        v = model.get_dndv_velocity_bins()
        y = observables[mask]
        p05, p16, p50, p84, p95 = np.percentile(y, [5, 16, 50, 84, 95], axis=0)
        ax.fill_between(v, p05, p95, color="tab:blue", alpha=0.18, label="5-95 percent", zorder=2)
        ax.fill_between(v, p16, p84, color="tab:blue", alpha=0.32, label="16-84 percent", zorder=3)
        ax.plot(v, p50, color="tab:blue", lw=2.0, label="median", zorder=4)
        ax.set_yscale("log")
        y_upper = np.nanmax([float(np.nanmax(p95)), classy_ymax, DNDV_PLOT_YMIN * 10.0])
        ax.set_ylim(DNDV_PLOT_YMIN, 1.2 * y_upper)
        ax.set_xlabel("velocity [km/s]")
        ax.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km/s)$^{-1}$]")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig.suptitle("Prior predictive dN/dv envelope")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def make_plots(
    output_dir: str,
    model: Any,
    results: Sequence[SampleResult],
    classy_profiles: Sequence[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, str]:
    """Create minimum prior predictive diagnostic plots."""
    arrays = _arrays_from_results(results)
    paths = {
        "validity": _plot_validity(output_dir, arrays["theta_samples"], arrays["valid"]),
        "observable_histograms": _plot_observable_histograms(output_dir, arrays["raw_moments"], arrays["valid"]),
        "parameter_scatter": _plot_parameter_scatter(output_dir, arrays["theta_samples"], arrays["raw_moments"], arrays["valid"]),
    }
    dndv_path = _plot_dndv_envelope(output_dir, model, arrays["observables"], arrays["valid"], classy_profiles)
    if dndv_path is not None:
        paths["dndv_envelope"] = dndv_path
    return paths


def write_metadata(
    output_dir: str,
    args: argparse.Namespace,
    model: Any,
    results: Sequence[SampleResult],
    output_files: dict[str, str],
    runtime_seconds: float,
) -> str:
    """Write run metadata and notes about intentionally unavailable diagnostics."""
    valid = np.asarray([result.valid for result in results], dtype=bool)
    metadata = {
        "created_by": "examples/inference_prior_predictive.py",
        "runtime_seconds": float(runtime_seconds),
        "arguments": vars(args),
        "theta_names": list(model.parameter_names()) if hasattr(model, "parameter_names") else list(PARAM_NAMES),
        "prior_family": "log-uniform",
        "prior_bounds": DEFAULT_PRIOR_BOUNDS,
        "expanded_parameters": getattr(model, "expanded_parameters", "none"),
        "mixing_chi_pivot": getattr(model, "mixing_chi_pivot", None),
        "beta_chi_max_abs": getattr(model, "beta_chi_max_abs", None),
        "observable_set": model.observable_set,
        "observable_names": list(model.observable_names),
        "num_samples": int(len(results)),
        "num_valid": int(np.sum(valid)),
        "valid_fraction": float(np.mean(valid)) if valid.size else np.nan,
        "output_files": output_files,
        "notes": [
            "v_at_10kpc, mass_loading_at_10kpc, L_hot_cgs, and L_interface_cgs are NaN here.",
            "Those diagnostics require a later pass through WindModel or analysis_helpers, not MomentInferenceModel.",
            "Invalid model samples are kept in the tables but their observable and raw-moment arrays are NaN.",
        ],
    }
    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w", encoding="ascii") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_jax_platform(args.jax_platform)
    configure_matplotlib_for_paper(use_tex=not args.no_usetex)

    start = time.perf_counter()
    os.makedirs(args.output, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    model = build_model(args)
    theta_samples = sample_prior(
        rng,
        args.num_samples,
        expanded_parameters=args.expanded_parameters,
        beta_chi_max_abs=args.beta_chi_max_abs,
    )
    classy_profiles = load_classy_profiles(args.classy_profiles_path) if args.overlay_classy else None

    results: list[SampleResult] = []
    for i, theta in enumerate(theta_samples):
        result = evaluate_sample(model, theta, i)
        results.append(result)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == args.num_samples:
            status = "valid" if result.valid else "invalid"
            print(
                f"[{i + 1:4d}/{args.num_samples}] "
                f"eta_M={theta[0]:.4g}, eta_M_cold={theta[1]:.4g}, eta_E={theta[2]:.4g}: {status}",
                flush=True,
            )

    output_files: dict[str, str] = {}
    output_files["npz"] = write_npz(args.output, model, results)
    output_files["csv"] = write_csv_summary(args.output, model, results)
    output_files.update(make_plots(args.output, model, results, classy_profiles))
    output_files["metadata"] = write_metadata(
        args.output,
        args,
        model,
        results,
        output_files,
        runtime_seconds=time.perf_counter() - start,
    )

    valid_count = sum(result.valid for result in results)
    print(f"Saved prior predictive atlas to {args.output}")
    print(f"Valid samples: {valid_count}/{len(results)}")


if __name__ == "__main__":
    main()
