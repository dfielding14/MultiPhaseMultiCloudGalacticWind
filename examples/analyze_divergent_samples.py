"""Summarize divergent-vs-nondivergent samples from synthetic recovery NPZ output."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import Msun, gamma, kb, mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", help="Path to synthetic_recovery_results.npz")
    parser.add_argument("--output-dir", default=None, help="Directory for summary files and figures")
    parser.add_argument("--realization-index", type=int, default=0)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_metadata(npz_path: Path) -> dict[str, Any]:
    meta_path = npz_path.with_name("run_metadata.json")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="ascii") as fh:
        return json.load(fh)


def _launch_quantities(theta: np.ndarray, config: WindConfig) -> tuple[np.ndarray, np.ndarray]:
    eta_m = np.maximum(theta[:, 0], 1e-30)
    eta_e = np.maximum(theta[:, 2], 1e-30)
    mach0 = 1.0 + float(config.sonic_point_offset)
    specific_energy = (eta_e / eta_m) * (config.E_SN / (config.mstar * Msun))
    v_star = np.sqrt(specific_energy) * (1.0 / ((gamma - 1.0) * mach0) + 0.5) ** (-0.5)
    cs_sq = v_star * v_star / (mach0 * mach0)
    t_star = config.mu * mp * cs_sq / (gamma * kb)
    return v_star / 1e5, t_star


def _quantile_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    group = values[mask]
    group = group[np.isfinite(group)]
    if group.size == 0:
        return {"q16": np.nan, "median": np.nan, "q84": np.nan, "mean": np.nan}
    q16, q50, q84 = np.quantile(group, [0.16, 0.50, 0.84])
    return {"q16": float(q16), "median": float(q50), "q84": float(q84), "mean": float(np.mean(group))}


def _comparison_summary(values: np.ndarray, divergent: np.ndarray) -> dict[str, Any]:
    finite_values = np.asarray(values, dtype=float)
    nondiv = ~divergent
    div_summary = _quantile_summary(finite_values, divergent)
    nondiv_summary = _quantile_summary(finite_values, nondiv)
    finite = finite_values[np.isfinite(finite_values)]
    if finite.size == 0:
        scale = np.nan
    else:
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-30)
    median_difference = div_summary["median"] - nondiv_summary["median"]
    return {
        "divergent": div_summary,
        "nondivergent": nondiv_summary,
        "median_difference": median_difference,
        "median_difference_over_iqr": median_difference / scale if np.isfinite(scale) else np.nan,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hist_overlay(ax, values: np.ndarray, divergent: np.ndarray, label: str) -> None:
    nondiv_values = values[~divergent]
    div_values = values[divergent]
    bins = np.histogram_bin_edges(values[np.isfinite(values)], bins=24)
    ax.hist(nondiv_values, bins=bins, density=True, alpha=0.35, color="tab:blue", label="nondivergent")
    ax.hist(div_values, bins=bins, density=True, alpha=0.55, color="tab:red", label="divergent")
    ax.set_xlabel(label)
    ax.set_ylabel("density")


def _plot_parameter_comparison(path: Path, features: dict[str, np.ndarray], divergent: np.ndarray) -> None:
    names = [
        ("eta_M", r"$\eta_M$"),
        ("eta_M_cold_over_eta_M", r"$\eta_{M,cold}/\eta_M$"),
        ("eta_E", r"$\eta_E$"),
        ("eta_E_over_eta_M", r"$\eta_E/\eta_M$"),
        ("v_star_kms", r"$v_\star$ [km/s]"),
        ("total_objective", "total objective"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, names):
        _hist_overlay(ax, features[key], divergent, label)
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Divergent vs nondivergent posterior samples")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_phase_scatter(path: Path, features: dict[str, np.ndarray], divergent: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    pairs = [
        ("eta_M", "eta_E_over_eta_M", r"$\eta_M$", r"$\eta_E/\eta_M$"),
        ("eta_M_cold_over_eta_M", "eta_E_over_eta_M", r"$\eta_{M,cold}/\eta_M$", r"$\eta_E/\eta_M$"),
    ]
    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(features[xkey][~divergent], features[ykey][~divergent], s=16, alpha=0.45, color="tab:blue")
        ax.scatter(features[xkey][divergent], features[ykey][divergent], s=22, alpha=0.80, color="tab:red")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle("Divergent samples in energy coordinates")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_observable_residuals(
    path: Path,
    velocity_bins: np.ndarray,
    residuals: np.ndarray,
    divergent: np.ndarray,
) -> dict[str, Any]:
    nondiv = ~divergent
    summary: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    if residuals.shape[0] != divergent.size or not np.any(np.isfinite(residuals)):
        ax.text(0.5, 0.5, "posterior predictive samples are not sample-aligned", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return {"skipped": "posterior predictive samples are not sample-aligned"}

    for mask, color, label in [(nondiv, "tab:blue", "nondivergent"), (divergent, "tab:red", "divergent")]:
        group = residuals[mask]
        if group.size == 0:
            continue
        q16, q50, q84 = np.nanquantile(group, [0.16, 0.50, 0.84], axis=0)
        ax.plot(velocity_bins, q50, color=color, label=label)
        ax.fill_between(velocity_bins, q16, q84, color=color, alpha=0.20)
        summary[label] = {
            "rms_residual_median": float(np.nanmedian(np.sqrt(np.nanmean(group * group, axis=1)))),
            "max_abs_residual_median": float(np.nanmedian(np.nanmax(np.abs(group), axis=1))),
        }
    if np.any(divergent) and np.any(nondiv):
        div_med = np.nanmedian(residuals[divergent], axis=0)
        nondiv_med = np.nanmedian(residuals[nondiv], axis=0)
        diff = div_med - nondiv_med
        strongest = int(np.nanargmax(np.abs(diff)))
        summary["largest_median_residual_difference"] = {
            "velocity_kms": float(velocity_bins[strongest]),
            "difference_sigma": float(diff[strongest]),
            "divergent_median": float(div_med[strongest]),
            "nondivergent_median": float(nondiv_med[strongest]),
        }
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("velocity bin [km/s]")
    ax.set_ylabel("normalized residual")
    ax.legend(loc="best")
    ax.set_title("Posterior predictive residuals")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return summary


def _plot_sampler_sequence(path: Path, features: dict[str, np.ndarray], divergent: np.ndarray) -> None:
    idx = np.arange(divergent.size)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    keys = [
        ("accept_prob", "accept prob"),
        ("num_steps", "NUTS steps"),
        ("energy", "energy"),
        ("total_objective", "total objective"),
    ]
    for ax, (key, label) in zip(axes, keys):
        ax.plot(idx, features[key], color="0.35", lw=1)
        ax.scatter(idx[divergent], features[key][divergent], color="tab:red", s=18, zorder=3)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("sample index")
    fig.suptitle("Sampler diagnostics with divergent samples highlighted")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    npz_path = Path(args.input_npz)
    output_dir = Path(args.output_dir) if args.output_dir else npz_path.with_name("divergent_sample_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    realization_index = int(args.realization_index)
    metadata = _load_metadata(npz_path)
    config = WindConfig()

    theta = np.asarray(data["samples_theta"][realization_index], dtype=float)
    divergent = np.asarray(data["sample_diverging"][realization_index], dtype=bool)
    posterior_predictive = np.asarray(data["posterior_predictive"][realization_index], dtype=float)
    observed = np.asarray(data["observed"][realization_index], dtype=float)
    sigma = np.asarray(data["sigma"][realization_index], dtype=float)
    velocity_bins = np.asarray([float(str(name).split("@")[1].split("km/s")[0]) for name in data["observable_names"]])
    component_names = [str(name) for name in data["posterior_component_names"]]
    components = np.asarray(data["posterior_components_samples"][realization_index], dtype=float)

    v_star_kms, t_star = _launch_quantities(theta, config)
    features: dict[str, np.ndarray] = {
        "eta_M": theta[:, 0],
        "eta_M_cold": theta[:, 1],
        "eta_M_cold_over_eta_M": theta[:, 1] / np.maximum(theta[:, 0], 1e-30),
        "eta_E": theta[:, 2],
        "eta_E_over_eta_M": theta[:, 2] / np.maximum(theta[:, 0], 1e-30),
        "v_star_kms": v_star_kms,
        "T_star_K": t_star,
        "accept_prob": np.asarray(data["sample_accept_prob"][realization_index], dtype=float),
        "num_steps": np.asarray(data["sample_num_steps"][realization_index], dtype=float),
        "energy": np.asarray(data["sample_energy"][realization_index], dtype=float),
        "potential_energy": np.asarray(data["sample_potential_energy"][realization_index], dtype=float),
    }
    for idx, name in enumerate(component_names):
        features[name] = components[:, idx]

    if posterior_predictive.shape[0] == theta.shape[0]:
        residuals = (posterior_predictive - observed[None, :]) / np.maximum(sigma[None, :], 1e-300)
        features["rms_residual"] = np.sqrt(np.nanmean(residuals * residuals, axis=1))
        features["high_velocity_rms_residual"] = np.sqrt(
            np.nanmean(residuals[:, velocity_bins >= 900.0] ** 2, axis=1)
        )
        features["max_abs_residual"] = np.nanmax(np.abs(residuals), axis=1)
    else:
        residuals = np.full((theta.shape[0], observed.shape[0]), np.nan, dtype=float)
        features["rms_residual"] = np.full((theta.shape[0],), np.nan, dtype=float)
        features["high_velocity_rms_residual"] = np.full((theta.shape[0],), np.nan, dtype=float)
        features["max_abs_residual"] = np.full((theta.shape[0],), np.nan, dtype=float)

    rows: list[dict[str, Any]] = []
    for i in range(theta.shape[0]):
        row = {
            "sample_index": i,
            "divergent": bool(divergent[i]),
        }
        for key in [
            "eta_M",
            "eta_M_cold",
            "eta_M_cold_over_eta_M",
            "eta_E",
            "eta_E_over_eta_M",
            "v_star_kms",
            "T_star_K",
            "accept_prob",
            "num_steps",
            "energy",
            "potential_energy",
            "total_objective",
            "chi2",
            "prior_chi2",
            "eta_e_softcap_penalty",
            "validity_barrier",
            "trajectory_status_code",
            "soft_reach_radius_kpc",
            "first_invalid_r_kpc",
            "min_hot_velocity_kms",
            "rms_residual",
            "high_velocity_rms_residual",
            "max_abs_residual",
        ]:
            row[key] = float(features[key][i])
        rows.append(row)
    _write_csv(output_dir / "divergent_sample_metrics.csv", rows)

    comparison_keys = [
        "eta_M",
        "eta_M_cold",
        "eta_M_cold_over_eta_M",
        "eta_E",
        "eta_E_over_eta_M",
        "v_star_kms",
        "T_star_K",
        "chi2",
        "prior_chi2",
        "eta_e_softcap_penalty",
        "validity_barrier",
        "total_objective",
        "accept_prob",
        "num_steps",
        "energy",
        "rms_residual",
        "high_velocity_rms_residual",
        "max_abs_residual",
        "min_hot_velocity_kms",
    ]
    comparisons = {key: _comparison_summary(features[key], divergent) for key in comparison_keys}
    largest_effects = sorted(
        (
            {
                "name": key,
                "median_difference_over_iqr": abs(value["median_difference_over_iqr"]),
                "signed_median_difference_over_iqr": value["median_difference_over_iqr"],
            }
            for key, value in comparisons.items()
            if np.isfinite(value["median_difference_over_iqr"])
        ),
        key=lambda item: item["median_difference_over_iqr"],
        reverse=True,
    )[:8]

    plot_paths = {
        "parameter_comparison": str(output_dir / "divergent_vs_nondivergent_parameters.png"),
        "phase_scatter": str(output_dir / "divergent_energy_coordinate_scatter.png"),
        "observable_residuals": str(output_dir / "divergent_vs_nondivergent_residuals.png"),
        "sampler_sequence": str(output_dir / "divergent_sampler_sequence.png"),
    }
    _plot_parameter_comparison(Path(plot_paths["parameter_comparison"]), features, divergent)
    _plot_phase_scatter(Path(plot_paths["phase_scatter"]), features, divergent)
    residual_summary = _plot_observable_residuals(Path(plot_paths["observable_residuals"]), velocity_bins, residuals, divergent)
    _plot_sampler_sequence(Path(plot_paths["sampler_sequence"]), features, divergent)

    summary = {
        "input_npz": str(npz_path),
        "realization_index": realization_index,
        "metadata_failure_policy": metadata.get("failure_policy"),
        "metadata_nuts_coordinate": metadata.get("nuts_coordinate"),
        "num_samples": int(theta.shape[0]),
        "num_divergent": int(np.sum(divergent)),
        "divergent_fraction": float(np.mean(divergent)),
        "comparisons": comparisons,
        "largest_effects": largest_effects,
        "residual_summary": residual_summary,
        "plot_paths": plot_paths,
        "csv": str(output_dir / "divergent_sample_metrics.csv"),
    }
    with (output_dir / "divergent_sample_summary.json").open("w", encoding="ascii") as fh:
        json.dump(_jsonable(summary), fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps(_jsonable({k: summary[k] for k in ["num_samples", "num_divergent", "divergent_fraction", "largest_effects", "plot_paths"]}), indent=2))


if __name__ == "__main__":
    main()
