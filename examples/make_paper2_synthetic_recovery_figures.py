#!/usr/bin/env python3
"""Build Paper 2 synthetic-recovery figures from existing diagnostic outputs."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "examples" / "outputs" / "inference_synthetic_recovery"
FIGURE_DIR = REPO_ROOT / "paper" / "paper2_inference_validation" / "figures"

SHAPE5_SUMMARY = OUTPUT_ROOT / "shape5_exact_all_truth_dense097" / "synthetic_recovery_summary.csv"
PRIOR_SENSITIVITY = (
    OUTPUT_ROOT
    / "high_energy_prior_sensitivity_20260425"
    / "prior_sensitivity_objective_decomposition.csv"
)
LOWM_DNDV_DEFAULT = (
    OUTPUT_ROOT
    / "high_energy_dndv_exact_20260425"
    / "low_eta_m_high_eta_e_default"
    / "synthetic_recovery_summary.csv"
)
LOWM_DNDV_WEAK = (
    OUTPUT_ROOT
    / "high_energy_dndv_exact_20260425"
    / "low_eta_m_high_eta_e_high_energy_weak"
    / "synthetic_recovery_summary.csv"
)
STRONG_WINGS_OBS = (
    OUTPUT_ROOT
    / "strong_wings_log_dndv_loading_ratios_observed_window_100_600_censored_delta100_short_nuts"
    / "synthetic_recovery_summary.csv"
)
STRONG_WINGS_CORNER = (
    OUTPUT_ROOT
    / "strong_wings_log_dndv_loading_ratios_observed_window_100_600_censored_delta100_short_nuts"
    / "diagnostic_plots"
    / "strong_wings_realization_000_energy_coordinate_corner.png"
)
STRONG_WINGS_FIT = (
    OUTPUT_ROOT
    / "strong_wings_log_dndv_loading_ratios_observed_window_100_600_censored_delta100_short_nuts"
    / "diagnostic_plots"
    / "strong_wings_realization_000_observable_fit.png"
)

TRUTH_CASES = {
    "fiducial": (0.20, 0.20, 0.80),
    "low_eta_m_high_eta_e": (0.07, 0.12, 0.96),
    "high_eta_m_low_eta_e": (0.90, 0.20, 0.25),
    "low_eta_m_cold": (0.22, 0.01, 0.80),
    "high_eta_m_cold": (0.35, 1.00, 0.85),
    "near_failure_boundary": (0.06, 1.50, 0.40),
    "strong_wings": (0.10, 0.35, 0.98),
    "narrow_profile": (0.70, 0.05, 0.45),
}
CASE_LABELS = {
    "fiducial": "fiducial",
    "low_eta_m_high_eta_e": r"low $\eta_M$, high $\eta_E$",
    "high_eta_m_low_eta_e": r"high $\eta_M$, low $\eta_E$",
    "low_eta_m_cold": r"low $\eta_{M,\rm cold}$",
    "high_eta_m_cold": r"high $\eta_{M,\rm cold}$",
    "near_failure_boundary": "near failure",
    "strong_wings": "strong wings",
    "narrow_profile": "narrow",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    """Parse a float from a CSV row, preserving NaN."""
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return np.nan


def as_bool(row: dict[str, str], key: str) -> bool:
    """Parse boolean-like CSV fields."""
    return str(row.get(key, "")).strip().lower() == "true"


def configure_matplotlib() -> None:
    """Use compact, manuscript-like defaults."""
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
            "figure.dpi": 120,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure in the Paper 2 figure directory."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def make_exact_shape5_summary() -> Path:
    """Create the exact shape-five truth-case summary figure."""
    rows = read_csv(SHAPE5_SUMMARY)
    by_case = {row["truth_case"]: row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    ax0, ax1 = axes

    for name, theta in TRUTH_CASES.items():
        row = by_case.get(name, {})
        valid = as_bool(row, "truth_valid")
        marker = "o" if valid else "x"
        color = "tab:green" if valid else "tab:red"
        size = 82 if valid else 90
        ax0.scatter(theta[0], theta[1], marker=marker, s=size, color=color, linewidth=2.4, alpha=0.85)
        xoff = 1.04 if theta[0] < 0.4 else 0.92
        yoff = 1.08 if theta[1] < 0.5 else 0.85
        ax0.text(theta[0] * xoff, theta[1] * yoff, CASE_LABELS[name], fontsize=8)

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"true $\eta_M$")
    ax0.set_ylabel(r"true $\eta_{M,\rm cold}$")
    ax0.set_title("Exact truth cases")
    ax0.scatter([], [], marker="o", s=70, color="tab:green", label="finite truth")
    ax0.scatter([], [], marker="x", s=70, color="tab:red", label="invalid truth")
    ax0.legend(loc="lower left")

    valid_cases = [
        "fiducial",
        "low_eta_m_high_eta_e",
        "low_eta_m_cold",
        "strong_wings",
        "narrow_profile",
    ]
    labels = ["fid", "lowM\nhighE", "lowMcold", "wings", "narrow"]
    x = np.arange(len(valid_cases), dtype=float)
    width = 0.23
    params = [
        ("eta_M", r"$\eta_M$", "tab:blue", -width),
        ("eta_M_cold", r"$\eta_{M,\rm cold}$", "tab:orange", 0.0),
        ("eta_E", r"$\eta_E$", "tab:green", width),
    ]
    for param, label, color, offset in params:
        ratios = []
        for case in valid_cases:
            row = by_case[case]
            truth = as_float(row, f"true_{param}")
            mapped = as_float(row, f"map_{param}")
            ratios.append(mapped / truth if truth > 0 else np.nan)
        ax1.bar(x + offset, ratios, width=width, color=color, alpha=0.88, label=label)

    ax1.axhline(1.0, color="black", lw=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_yscale("log")
    ax1.set_ylim(0.35, 2.2)
    ax1.set_ylabel("MAP / truth")
    ax1.set_title("Shape-five MAP recovery")
    ax1.legend(ncols=3, loc="upper center")

    return save_figure(fig, "synthetic_recovery_exact_shape5_summary.png")


def prior_variant_order() -> list[tuple[str, str]]:
    """Return prior-sensitivity labels in plotting order."""
    return [
        ("default", "default"),
        ("weak_default_centered", "weak\ndefault"),
        ("high_energy_centered", "highE\ncentered"),
        ("high_energy_weak", "highE\nweak"),
        ("near_flat_diagnostic", "near-flat"),
    ]


def make_high_energy_diagnostics() -> Path:
    """Create the high-energy prior/profile diagnostic figure."""
    prior_rows = [
        row
        for row in read_csv(PRIOR_SENSITIVITY)
        if row.get("point") == "map" and row.get("truth_case") in {"low_eta_m_high_eta_e", "strong_wings"}
    ]
    row_map = {(row["truth_case"], row["prior_label"]): row for row in prior_rows}

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)
    ax0, ax1 = axes

    variants = prior_variant_order()
    x = np.arange(len(variants), dtype=float)
    for case, color, marker in [
        ("low_eta_m_high_eta_e", "tab:blue", "o"),
        ("strong_wings", "tab:purple", "s"),
    ]:
        y = [as_float(row_map[(case, label)], "eta_E") for label, _ in variants]
        ax0.plot(x, y, marker=marker, color=color, lw=2.0, label=CASE_LABELS[case])
        ax0.axhline(TRUTH_CASES[case][2], color=color, ls="--", lw=1.1, alpha=0.65)
    ax0.set_xticks(x)
    ax0.set_xticklabels([short for _, short in variants])
    ax0.set_ylabel(r"shape-five MAP $\eta_E$")
    ax0.set_title("Prior sensitivity")
    ax0.set_ylim(0.70, 1.03)
    ax0.legend(loc="lower right")

    profile_rows = [
        (
            "lowM highE\nbinned default",
            read_csv(LOWM_DNDV_DEFAULT)[0],
            TRUTH_CASES["low_eta_m_high_eta_e"][2],
            "tab:blue",
        ),
        (
            "lowM highE\nbinned highE weak",
            read_csv(LOWM_DNDV_WEAK)[0],
            TRUTH_CASES["low_eta_m_high_eta_e"][2],
            "tab:cyan",
        ),
        (
            "strong wings\nobs. window",
            read_csv(STRONG_WINGS_OBS)[0],
            TRUTH_CASES["strong_wings"][2],
            "tab:purple",
        ),
    ]
    labels = [item[0] for item in profile_rows]
    ratios = [as_float(row, "map_eta_E") / truth for _, row, truth, _ in profile_rows]
    colors = [item[3] for item in profile_rows]
    divs = [as_float(row, "num_divergent") for _, row, _, _ in profile_rows]
    esses = [as_float(row, "min_ess_bulk") for _, row, _, _ in profile_rows]
    rhats = [as_float(row, "max_r_hat") for _, row, _, _ in profile_rows]
    p_gt1 = [as_float(row, "posterior_prob_eta_E_gt_1") for _, row, _, _ in profile_rows]

    ax1.bar(np.arange(len(labels)), ratios, color=colors, alpha=0.86)
    ax1.axhline(1.0, color="black", lw=1.0)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r"profile MAP $\eta_E$ / truth")
    ax1.set_title("Profile-level recovery")
    ax1.set_ylim(0.88, 1.03)
    for i, (ratio, div, ess, rhat, prob) in enumerate(zip(ratios, divs, esses, rhats, p_gt1)):
        extra = "" if not np.isfinite(prob) else f"\n$P(>1)={prob:.2f}$"
        ax1.text(
            i,
            ratio + 0.006,
            f"div={int(div) if np.isfinite(div) else 0}\nESS={ess:.0f}\n$\\hat R={rhat:.3f}$" + extra,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    return save_figure(fig, "synthetic_recovery_high_energy_diagnostics.png")


def crop_image(img: np.ndarray, threshold: float = 0.985) -> np.ndarray:
    """Trim nearly white border from an RGB/RGBA image."""
    rgb = img[..., :3]
    mask = np.any(rgb < threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    pad = 16
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, img.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def make_strong_wings_corner_fit() -> Path:
    """Create a two-panel strong-wings diagnostic from existing PNG outputs."""
    if not STRONG_WINGS_CORNER.exists() or not STRONG_WINGS_FIT.exists():
        missing = [str(path) for path in (STRONG_WINGS_CORNER, STRONG_WINGS_FIT) if not path.exists()]
        raise FileNotFoundError(", ".join(missing))

    corner = crop_image(mpimg.imread(STRONG_WINGS_CORNER))
    fit = crop_image(mpimg.imread(STRONG_WINGS_FIT))

    fig = plt.figure(figsize=(12.2, 5.4), constrained_layout=True)
    gridspec = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax0 = fig.add_subplot(gridspec[0, 0])
    ax1 = fig.add_subplot(gridspec[0, 1])

    ax0.imshow(corner)
    ax0.set_axis_off()
    ax0.set_title("Loading-ratio posterior")

    ax1.imshow(fit)
    ax1.set_axis_off()
    ax1.set_title(r"Observed-window $\log dN/dv$ fit")

    return save_figure(fig, "synthetic_recovery_strong_wings_corner_fit.png")


def main() -> None:
    configure_matplotlib()
    outputs = [
        make_exact_shape5_summary(),
        make_high_energy_diagnostics(),
        make_strong_wings_corner_fit(),
    ]
    for output in outputs:
        print(output.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
