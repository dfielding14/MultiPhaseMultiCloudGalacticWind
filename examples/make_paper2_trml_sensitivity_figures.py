#!/usr/bin/env python3
"""Aggregate TRML sensitivity screens and build Paper 2 figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "examples" / "outputs" / "trml_sensitivity_screen"
FIGURE_DIR = REPO_ROOT / "paper" / "paper2_inference_validation" / "figures"
AGGREGATE_DIR = OUTPUT_ROOT / "paper2_trml_sensitivity_aggregate_20260428"

STAGES = {
    "shape5": OUTPUT_ROOT / "multicase_shape5_reduced_20260428",
    "log_dndv_window": OUTPUT_ROOT / "multicase_log_dndv_observed_window_20260428",
    "dndv_window": OUTPUT_ROOT / "multicase_dndv_observed_window_shortlist_20260428",
}
STAGE_LABELS = {
    "shape5": "shape-five",
    "log_dndv_window": "log profile",
    "dndv_window": "linear profile",
}
PAPER_PARAM_ORDER = [
    "A_mix",
    "Mdot_coefficient",
    "beta_chi_mix",
    "geometric_factor",
    "f_turb0",
    "cloud_alpha",
    "CoolingAreaChiPower",
    "TurbulentVelocityChiPower",
    "drag_coeff",
    "ColdTurbulenceChiPower",
    "Z_hot_over_Z_solar",
    "Z_cloud_over_Z_solar",
    "v_cloud_init",
    "Cooling_Factor",
]
TRUTH_CASE_ORDER = [
    "fiducial",
    "low_eta_m_cold",
    "low_eta_m_high_eta_e",
    "narrow_profile",
    "strong_wings",
]
PARAM_LABELS = {
    "A_mix": r"$A_{\rm mix}$",
    "Mdot_coefficient": r"$A_{\dot M}$",
    "beta_chi_mix": r"$\beta_{\chi,\rm mix}$",
    "geometric_factor": r"$A_{\rm area}$",
    "f_turb0": r"$f_{\rm turb}$",
    "cloud_alpha": r"$\alpha_{\rm cl}$",
    "CoolingAreaChiPower": r"$\chi_{\rm area}$",
    "TurbulentVelocityChiPower": r"$\chi_{\rm turb}$",
    "drag_coeff": r"$C_D$",
    "ColdTurbulenceChiPower": r"$\chi_{\rm cold}$",
    "Z_hot_over_Z_solar": r"$Z_{\rm hot}$",
    "Z_cloud_over_Z_solar": r"$Z_{\rm cold}$",
    "v_cloud_init": r"$v_{\rm cl,0}$",
    "Cooling_Factor": r"$f_{\rm cool}$",
}
TRUTH_CASE_LABELS = {
    "fiducial": "fid.",
    "low_eta_m_cold": r"low $\eta_{M,c}$",
    "low_eta_m_high_eta_e": r"high $\eta_E$",
    "narrow_profile": "narrow",
    "strong_wings": "wings",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    """Parse a float, preserving missing values as NaN."""
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return np.nan


def as_bool(row: dict[str, str], key: str) -> bool:
    """Parse boolean-like CSV fields."""
    return str(row.get(key, "")).strip().lower() == "true"


def finite_values(values: Iterable[float]) -> np.ndarray:
    """Return finite float values."""
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def nanmax_abs(rows: list[dict[str, str]], key: str) -> float:
    """Maximum absolute finite value for a row key."""
    values = finite_values(as_float(row, key) for row in rows)
    if values.size == 0:
        return np.nan
    return float(np.max(np.abs(values)))


def nanmedian(values: Iterable[float]) -> float:
    """Median finite value."""
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan
    return float(np.median(finite))


def nanmax(values: Iterable[float]) -> float:
    """Maximum finite value."""
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan
    return float(np.max(finite))


def load_stage_rows() -> dict[str, list[dict[str, str]]]:
    """Load row-level CSVs for every planned stage."""
    stage_rows: dict[str, list[dict[str, str]]] = {}
    for stage, directory in STAGES.items():
        stage_rows[stage] = read_csv(directory / "trml_sensitivity_rows.csv")
    return stage_rows


def classify(row: dict[str, float]) -> str:
    """Assign a compact interpretation label."""
    shape = row["shape5_max_observable_distance"]
    log_profile = row["log_dndv_window_max_profile_distance"]
    dndv_profile = row["dndv_window_max_profile_distance"]
    valid_fraction = row["min_valid_fraction"]
    status_failures = row["total_stalled_or_numerical"]
    subspace_fraction = row.get("max_loading_subspace_fraction", np.nan)
    leverage = np.nanmax([shape, log_profile / 5.0 if np.isfinite(log_profile) else np.nan, dndv_profile / 5.0 if np.isfinite(dndv_profile) else np.nan])

    high = bool(np.isfinite(leverage) and leverage >= 0.15)
    degenerate = bool(np.isfinite(subspace_fraction) and subspace_fraction >= 0.95)
    if not high:
        return "low leverage"
    if valid_fraction < 0.85 or status_failures > 1:
        return "high leverage but risky"
    if degenerate:
        return "stable high leverage; loading-degenerate"
    return "stable high leverage"


def aggregate(stage_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, float | str]]:
    """Aggregate all stages into one row per parameter."""
    parameters = sorted({row["parameter_name"] for rows in stage_rows.values() for row in rows})
    summaries: list[dict[str, float | str]] = []
    for parameter in parameters:
        out: dict[str, float | str] = {"parameter_name": parameter}
        valid_fractions = []
        status_failures = 0
        max_cosines = []
        median_cosines = []
        max_subspace_fractions = []
        median_subspace_fractions = []
        max_orthogonal_chi = []
        median_orthogonal_chi = []
        max_orthogonal_rms = []
        max_whitened_distances = []
        max_shape_deltas = defaultdict(float)
        for stage, rows in stage_rows.items():
            group = [row for row in rows if row["parameter_name"] == parameter]
            if not group:
                out[f"{stage}_num_rows"] = 0
                out[f"{stage}_valid_fraction"] = np.nan
                out[f"{stage}_max_observable_distance"] = np.nan
                out[f"{stage}_median_observable_distance"] = np.nan
                out[f"{stage}_max_profile_distance"] = np.nan
                out[f"{stage}_median_profile_distance"] = np.nan
                out[f"{stage}_max_loading_degeneracy_cosine"] = np.nan
                out[f"{stage}_max_loading_subspace_fraction"] = np.nan
                out[f"{stage}_median_loading_subspace_fraction"] = np.nan
                out[f"{stage}_max_loading_orthogonal_chi"] = np.nan
                out[f"{stage}_median_loading_orthogonal_chi"] = np.nan
                out[f"{stage}_max_loading_orthogonal_rms"] = np.nan
                out[f"{stage}_max_loading_whitened_distance"] = np.nan
                continue

            valid = np.asarray([as_bool(row, "valid") for row in group], dtype=bool)
            status = np.asarray([as_float(row, "trajectory_status_code") for row in group], dtype=float)
            stage_valid_fraction = float(np.mean(valid)) if valid.size else np.nan
            valid_fractions.append(stage_valid_fraction)
            status_failures += int(np.sum((status == 1.0) | (status == 2.0)))
            max_cosines.append(nanmax(as_float(row, "loading_degeneracy_cosine") for row in group))
            median_cosines.append(nanmedian(as_float(row, "loading_degeneracy_cosine") for row in group))
            max_subspace_fractions.append(nanmax(as_float(row, "loading_subspace_fraction") for row in group))
            median_subspace_fractions.append(nanmedian(as_float(row, "loading_subspace_fraction") for row in group))
            max_orthogonal_chi.append(nanmax(as_float(row, "loading_orthogonal_chi") for row in group))
            median_orthogonal_chi.append(nanmedian(as_float(row, "loading_orthogonal_chi") for row in group))
            max_orthogonal_rms.append(nanmax(as_float(row, "loading_orthogonal_rms") for row in group))
            max_whitened_distances.append(nanmax(as_float(row, "loading_whitened_distance") for row in group))

            out[f"{stage}_num_rows"] = len(group)
            out[f"{stage}_valid_fraction"] = stage_valid_fraction
            out[f"{stage}_max_observable_distance"] = nanmax(as_float(row, "observable_distance") for row in group)
            out[f"{stage}_median_observable_distance"] = nanmedian(as_float(row, "observable_distance") for row in group)
            out[f"{stage}_max_profile_distance"] = nanmax(as_float(row, "profile_distance") for row in group)
            out[f"{stage}_median_profile_distance"] = nanmedian(as_float(row, "profile_distance") for row in group)
            out[f"{stage}_max_loading_degeneracy_cosine"] = max_cosines[-1]
            out[f"{stage}_max_loading_subspace_fraction"] = max_subspace_fractions[-1]
            out[f"{stage}_median_loading_subspace_fraction"] = median_subspace_fractions[-1]
            out[f"{stage}_max_loading_orthogonal_chi"] = max_orthogonal_chi[-1]
            out[f"{stage}_median_loading_orthogonal_chi"] = median_orthogonal_chi[-1]
            out[f"{stage}_max_loading_orthogonal_rms"] = max_orthogonal_rms[-1]
            out[f"{stage}_max_loading_whitened_distance"] = max_whitened_distances[-1]
            for key in ("delta_logM0", "delta_mean_v", "delta_sigma_v", "delta_skewness", "delta_kurtosis"):
                max_shape_deltas[key] = max(max_shape_deltas[key], nanmax_abs(group, key))

        out["min_valid_fraction"] = float(np.nanmin(valid_fractions)) if valid_fractions else np.nan
        out["total_stalled_or_numerical"] = int(status_failures)
        out["max_loading_degeneracy_cosine"] = nanmax(max_cosines)
        out["median_loading_degeneracy_cosine"] = nanmedian(median_cosines)
        out["max_loading_subspace_fraction"] = nanmax(max_subspace_fractions)
        out["median_loading_subspace_fraction"] = nanmedian(median_subspace_fractions)
        out["max_loading_orthogonal_chi"] = nanmax(max_orthogonal_chi)
        out["median_loading_orthogonal_chi"] = nanmedian(median_orthogonal_chi)
        out["max_loading_orthogonal_rms"] = nanmax(max_orthogonal_rms)
        out["max_loading_whitened_distance"] = nanmax(max_whitened_distances)
        for key, value in max_shape_deltas.items():
            out[f"max_abs_{key}"] = float(value)
        shape = float(out.get("shape5_max_observable_distance", np.nan))
        log_profile = float(out.get("log_dndv_window_max_profile_distance", np.nan))
        linear_profile = float(out.get("dndv_window_max_profile_distance", np.nan))
        out["ranking_score"] = float(
            np.nanmax(
                [
                    shape,
                    log_profile / 5.0 if np.isfinite(log_profile) else np.nan,
                    linear_profile / 5.0 if np.isfinite(linear_profile) else np.nan,
                ]
            )
        )
        out["classification"] = classify(out)  # type: ignore[arg-type]
        summaries.append(out)

    summaries.sort(key=lambda row: (-float(row["ranking_score"]), str(row["parameter_name"])))
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def write_table(path: Path, rows: list[dict[str, float | str]]) -> Path:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "rank",
        "parameter_name",
        "classification",
        "ranking_score",
        "min_valid_fraction",
        "total_stalled_or_numerical",
        "max_loading_subspace_fraction",
        "max_loading_orthogonal_chi",
        "max_loading_orthogonal_rms",
        "max_loading_whitened_distance",
        "max_loading_degeneracy_cosine",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def invalid_rows(stage_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """Collect invalid, stalled, or numerical scan points."""
    rows: list[dict[str, str]] = []
    for stage, stage_data in stage_rows.items():
        for row in stage_data:
            status = as_float(row, "trajectory_status_code")
            if not as_bool(row, "valid") or status in (1.0, 2.0) or row.get("error"):
                rows.append(
                    {
                        "stage": stage,
                        "parameter_name": row["parameter_name"],
                        "parameter_value": row["parameter_value"],
                        "truth_case": row["truth_case"],
                        "valid": row["valid"],
                        "trajectory_status_code": row["trajectory_status_code"],
                        "first_invalid_r_kpc": row["first_invalid_r_kpc"],
                        "soft_reach_radius_kpc": row["soft_reach_radius_kpc"],
                        "stall_penalty": row["stall_penalty"],
                        "error": row.get("error", ""),
                    }
                )
    return rows


def per_case_summary(stage_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, float | str]]:
    """Aggregate response and validity by stage, parameter, and truth case."""
    rows: list[dict[str, float | str]] = []
    for stage, stage_data in stage_rows.items():
        keys = sorted({(row["parameter_name"], row["truth_case"]) for row in stage_data})
        for parameter, truth_case in keys:
            group = [
                row
                for row in stage_data
                if row["parameter_name"] == parameter and row["truth_case"] == truth_case
            ]
            valid = np.asarray([as_bool(row, "valid") for row in group], dtype=bool)
            status = np.asarray([as_float(row, "trajectory_status_code") for row in group], dtype=float)
            rows.append(
                {
                    "stage": stage,
                    "parameter_name": parameter,
                    "truth_case": truth_case,
                    "num_rows": len(group),
                    "valid_fraction": float(np.mean(valid)) if valid.size else np.nan,
                    "stalled_or_numerical": int(np.sum((status == 1.0) | (status == 2.0))),
                    "max_observable_distance": nanmax(
                        as_float(row, "observable_distance") for row in group
                    ),
                    "median_observable_distance": nanmedian(
                        as_float(row, "observable_distance") for row in group
                    ),
                    "max_profile_distance": nanmax(
                        as_float(row, "profile_distance") for row in group
                    ),
                    "max_loading_degeneracy_cosine": nanmax(
                        as_float(row, "loading_degeneracy_cosine") for row in group
                    ),
                    "max_loading_subspace_fraction": nanmax(
                        as_float(row, "loading_subspace_fraction") for row in group
                    ),
                    "max_loading_orthogonal_chi": nanmax(
                        as_float(row, "loading_orthogonal_chi") for row in group
                    ),
                    "max_loading_orthogonal_rms": nanmax(
                        as_float(row, "loading_orthogonal_rms") for row in group
                    ),
                }
            )
    return rows


def configure_matplotlib() -> None:
    """Set compact paper-like plotting defaults."""
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a manuscript figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def ordered_rows(summary_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Return rows in paper-preferred order."""
    by_name = {str(row["parameter_name"]): row for row in summary_rows}
    names = [name for name in PAPER_PARAM_ORDER if name in by_name]
    names += sorted(name for name in by_name if name not in names)
    return [by_name[name] for name in names]


def make_cross_case_ranking(summary_rows: list[dict[str, float | str]]) -> Path:
    """Plot compact shape/profile leverage ranking."""
    rows = ordered_rows(summary_rows)
    labels = [PARAM_LABELS.get(str(row["parameter_name"]), str(row["parameter_name"])) for row in rows]
    y = np.arange(len(rows), dtype=float)
    shape = np.asarray([float(row.get("shape5_max_observable_distance", np.nan)) for row in rows], dtype=float)
    log_profile = np.asarray([float(row.get("log_dndv_window_max_profile_distance", np.nan)) for row in rows], dtype=float)
    linear_profile = np.asarray([float(row.get("dndv_window_max_profile_distance", np.nan)) for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), sharey=True, constrained_layout=True)
    panels = [
        (axes[0], shape, "shape-five\nmax shift", "tab:blue"),
        (axes[1], log_profile, r"$\log dN/dv$ window" + "\nmax distance", "tab:green"),
        (axes[2], linear_profile, r"$dN/dv$ window" + "\nmax log distance", "tab:orange"),
    ]
    for ax, values, title, color in panels:
        plot_values = np.where(np.isfinite(values), values, 0.0)
        ax.barh(y, plot_values, color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel("sensitivity")
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmax(finite) / max(np.nanmin(finite[finite > 0]) if np.any(finite > 0) else 1.0, 1.0e-12) > 100:
            ax.set_xscale("log")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    return save_figure(fig, "trml_sensitivity_cross_case_ranking.png")


def aggregate_matrix(
    rows: list[dict[str, str]],
    parameters: list[str],
    truth_cases: list[str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return max metric and stalled/invalid counts for heatmap panels."""
    values = np.full((len(parameters), len(truth_cases)), np.nan)
    failures = np.zeros((len(parameters), len(truth_cases)), dtype=int)
    for i, parameter in enumerate(parameters):
        for j, truth_case in enumerate(truth_cases):
            group = [
                row
                for row in rows
                if row["parameter_name"] == parameter and row["truth_case"] == truth_case
            ]
            if not group:
                continue
            values[i, j] = nanmax(as_float(row, metric) for row in group)
            failures[i, j] = int(
                np.sum(
                    [
                        (not as_bool(row, "valid"))
                        or as_float(row, "trajectory_status_code") in (1.0, 2.0)
                        or bool(row.get("error"))
                        for row in group
                    ]
                )
            )
    return values, failures


def annotate_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    failures: np.ndarray,
    *,
    value_format: str,
) -> None:
    """Add compact numeric labels and stall markers to a heatmap."""
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color="white" if np.log10(1.0 + value) > 0.45 else "black",
                )
            if failures[i, j] > 0:
                ax.scatter(
                    [j + 0.31],
                    [i - 0.31],
                    marker="x",
                    s=22,
                    color="crimson",
                    linewidths=1.2,
                    clip_on=False,
                )


def make_per_case_heatmap(stage_rows: dict[str, list[dict[str, str]]]) -> Path:
    """Plot per-case leverage and validity for shape and profile screens."""
    truth_cases = TRUTH_CASE_ORDER
    shape_parameters = [name for name in PAPER_PARAM_ORDER if any(row["parameter_name"] == name for row in stage_rows["shape5"])]
    profile_parameters = [
        name
        for name in PAPER_PARAM_ORDER
        if any(row["parameter_name"] == name for row in stage_rows["log_dndv_window"])
    ]
    shape_values, shape_failures = aggregate_matrix(
        stage_rows["shape5"], shape_parameters, truth_cases, "observable_distance"
    )
    profile_values, profile_failures = aggregate_matrix(
        stage_rows["log_dndv_window"], profile_parameters, truth_cases, "profile_distance"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3), constrained_layout=True)
    panels = [
        (
            axes[0],
            shape_values,
            shape_failures,
            shape_parameters,
            "shape-five max shift",
            "{:.2f}",
        ),
        (
            axes[1],
            profile_values,
            profile_failures,
            profile_parameters,
            r"observed-window $\log dN/dv$ max distance",
            "{:.0f}",
        ),
    ]
    for ax, values, failures, parameters, title, value_format in panels:
        image_values = np.log10(1.0 + values)
        im = ax.imshow(image_values, aspect="auto", cmap="viridis", vmin=0.0)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(truth_cases)))
        ax.set_xticklabels([TRUTH_CASE_LABELS.get(name, name) for name in truth_cases], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(parameters)))
        ax.set_yticklabels([PARAM_LABELS.get(name, name) for name in parameters])
        ax.set_xticks(np.arange(-0.5, len(truth_cases), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(parameters), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        annotate_heatmap(ax, values, failures, value_format=value_format)
        cbar = fig.colorbar(im, ax=ax, shrink=0.78)
        cbar.set_label(r"$\log_{10}(1+\mathrm{distance})$")
    axes[1].scatter([], [], marker="x", color="crimson", label="stalled/invalid scan point")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), fontsize=8)
    return save_figure(fig, "trml_sensitivity_per_case_heatmap.png")


def make_leverage_degeneracy(summary_rows: list[dict[str, float | str]]) -> Path:
    """Plot leverage versus covariance-whitened loading-subspace degeneracy."""
    rows = summary_rows
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True, constrained_layout=True)
    label_points: list[tuple[str, float, float]] = []
    for row in rows:
        name = str(row["parameter_name"])
        frac = float(row.get("max_loading_subspace_fraction", np.nan))
        residual = float(row.get("max_loading_orthogonal_chi", np.nan))
        y = float(row.get("ranking_score", np.nan))
        valid = float(row.get("min_valid_fraction", np.nan))
        if not np.isfinite(frac) or not np.isfinite(y):
            continue
        color = "tab:red" if valid < 0.85 else "tab:blue"
        marker = "s" if "risky" in str(row.get("classification", "")) else "o"
        axes[0].scatter(frac, y, s=65, color=color, marker=marker, alpha=0.86)
        if np.isfinite(residual):
            axes[1].scatter(residual, y, s=65, color=color, marker=marker, alpha=0.86)
        if y >= 0.12 or name in {"Cooling_Factor", "v_cloud_init"}:
            label_points.append((PARAM_LABELS.get(name, name), residual, y))
    axes[0].axvline(0.95, color="0.35", ls="--", lw=1.1)
    axes[0].set_xlim(0.0, 1.03)
    axes[0].set_xlabel("max fraction explained by loadings")
    axes[0].set_ylabel("combined sensitivity score")
    axes[0].set_title("Projection onto loading subspace")
    axes[1].axvline(3.0, color="0.35", ls="--", lw=1.1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"max orthogonal residual $\sqrt{\Delta\chi^2_\perp}$")
    axes[1].set_title("Residual after loading refit")
    for ax in axes:
        ax.set_yscale("log")
    axes[0].scatter([], [], color="tab:blue", marker="o", label="validity-stable")
    axes[0].scatter([], [], color="tab:red", marker="s", label="risky/invalid points")
    axes[0].legend(loc="lower left")

    label_points.sort(key=lambda item: item[2], reverse=True)
    adjusted: list[tuple[str, float, float, float]] = []
    min_delta = 0.075
    previous_log_y = np.inf
    for label, x, y in label_points:
        log_y = float(np.log10(y))
        if previous_log_y - log_y < min_delta:
            log_y = previous_log_y - min_delta
        adjusted.append((label, x, y, 10.0**log_y))
        previous_log_y = log_y
    for label, x, y, y_label in adjusted:
        if not np.isfinite(x) or x <= 0.0:
            continue
        axes[1].plot([x, x * 1.15], [y, y_label], color="0.55", lw=0.6, alpha=0.7)
        axes[1].text(x * 1.18, y_label, label, fontsize=8, va="center", ha="left")
    return save_figure(fig, "trml_sensitivity_leverage_degeneracy.png")


def write_recommendation(summary_rows: list[dict[str, float | str]]) -> Path:
    """Write a machine-readable recommendation summary."""
    by_name = {str(row["parameter_name"]): row for row in summary_rows}
    shortlist = [name for name in ("A_mix", "Mdot_coefficient", "geometric_factor", "f_turb0") if name in by_name]
    candidate = by_name.get("A_mix") or by_name.get("Mdot_coefficient")
    recommend_restricted_amix = False
    if candidate is not None:
        recommend_restricted_amix = (
            float(candidate.get("min_valid_fraction", 0.0)) >= 0.85
            and float(candidate.get("ranking_score", 0.0)) >= 0.15
            and float(candidate.get("max_loading_orthogonal_chi", 0.0)) >= 3.0
        )
    payload = {
        "recommendation": (
            "restricted_A_mix_decision_required"
            if recommend_restricted_amix
            else "no_go_for_expanded_inference"
        ),
        "rationale": (
            "A_mix/Mdot_coefficient has cross-case leverage and a large covariance-whitened residual after loading refit, but upper scan values stall high-energy cases; a narrow A_mix experiment needs its own validation gate."
            if recommend_restricted_amix
            else "No candidate met the stability and interpretability checks."
        ),
        "shortlist": shortlist,
        "top_parameters": [str(row["parameter_name"]) for row in summary_rows[:5]],
    }
    path = AGGREGATE_DIR / "trml_sensitivity_recommendation.json"
    with path.open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> None:
    """Build all aggregate tables and manuscript figures."""
    configure_matplotlib()
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    stage_rows = load_stage_rows()
    summaries = aggregate(stage_rows)
    invalid = invalid_rows(stage_rows)
    outputs = [
        write_table(AGGREGATE_DIR / "trml_sensitivity_cross_case_summary.csv", summaries),
        write_table(AGGREGATE_DIR / "trml_sensitivity_invalid_scan_points.csv", invalid),
        write_table(
            AGGREGATE_DIR / "trml_sensitivity_per_case_summary.csv",
            per_case_summary(stage_rows),
        ),
        write_recommendation(summaries),
        make_cross_case_ranking(summaries),
        make_per_case_heatmap(stage_rows),
        make_leverage_degeneracy(summaries),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
