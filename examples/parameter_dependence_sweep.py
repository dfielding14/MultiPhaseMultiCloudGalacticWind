#!/usr/bin/env python
"""
Parameter-dependence sweep for key wind controls.

This script visualizes how core observables vary with:
- eta_M (hot mass loading)
- eta_M_cold (cold mass loading)
- eta_E (energy loading)

Outputs are saved as PDF figures under the selected output directory.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind import WindModel


METRICS = {
    "v10": ("$v(10\\,\\mathrm{kpc})$", "km/s"),
    "eta10": ("$\\eta(10\\,\\mathrm{kpc})$", "dimensionless"),
    "v_mean": ("$\\langle v \\rangle$", "km/s"),
    "v_disp": ("$\\sigma_v$", "km/s"),
    "r_end": ("$r_\\mathrm{end}$", "kpc"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep eta_M, eta_M_cold, eta_E and plot observable dependence.")

    parser.add_argument("--output-dir", default="plots_parameter_dependence", help="Directory for output plots")
    parser.add_argument("--quick", action="store_true", help="Use a smaller sweep for fast iteration")

    parser.add_argument("--n-eta-m", type=int, default=7)
    parser.add_argument("--n-eta-m-cold", type=int, default=7)
    parser.add_argument("--n-eta-e", type=int, default=7)

    parser.add_argument("--eta-m-min", type=float, default=0.05)
    parser.add_argument("--eta-m-max", type=float, default=0.30)
    parser.add_argument("--eta-m-cold-min", type=float, default=0.02)
    parser.add_argument("--eta-m-cold-max", type=float, default=0.40)
    parser.add_argument("--eta-e-min", type=float, default=0.5)
    parser.add_argument("--eta-e-max", type=float, default=2.0)

    parser.add_argument("--sfr", type=float, default=20.0)
    parser.add_argument("--v-circ", type=float, default=150.0)
    parser.add_argument("--n-cloud-species", type=int, default=13)
    parser.add_argument("--cloud-mass-min", type=float, default=1.0)
    parser.add_argument("--cloud-mass-max", type=float, default=1.0e6)
    parser.add_argument("--r-max-kpc", type=float, default=30.0)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)

    return parser.parse_args()


def make_grid(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.quick:
        n_eta_m = min(args.n_eta_m, 4)
        n_eta_m_cold = min(args.n_eta_m_cold, 4)
        n_eta_e = min(args.n_eta_e, 4)
    else:
        n_eta_m = args.n_eta_m
        n_eta_m_cold = args.n_eta_m_cold
        n_eta_e = args.n_eta_e

    eta_m_values = np.linspace(args.eta_m_min, args.eta_m_max, n_eta_m)
    eta_m_cold_values = np.linspace(args.eta_m_cold_min, args.eta_m_cold_max, n_eta_m_cold)
    eta_e_values = np.linspace(args.eta_e_min, args.eta_e_max, n_eta_e)
    return eta_m_values, eta_m_cold_values, eta_e_values


def run_case(args: argparse.Namespace, eta_m: float, eta_m_cold: float, eta_e: float) -> Dict[str, float]:
    model = WindModel(
        SFR=args.sfr,
        v_circ=args.v_circ,
        eta_M=eta_m,
        eta_M_cold=eta_m_cold,
        eta_E=eta_e,
        N_cloud_species=args.n_cloud_species,
        cloud_mass_range=(args.cloud_mass_min, args.cloud_mass_max),
        r_max_kpc=args.r_max_kpc,
        rtol=args.rtol,
        atol=args.atol,
        cooling_backend="topaz",
    )

    solution = model.run()

    r_end = float(solution.r[-1])
    v10 = float(solution.v_at_10kpc)
    eta10 = float(solution.mass_loading_at_10kpc)

    if r_end > 0.6:
        r_upper = min(args.r_max_kpc, r_end)
        moments = solution.calculate_velocity_moments(r_min_kpc=0.5, r_max_kpc=r_upper)
        v_mean = float(moments.get("mean", np.nan))
        v_disp = float(moments.get("dispersion", np.nan))
    else:
        v_mean = np.nan
        v_disp = np.nan

    return {
        "v10": v10,
        "eta10": eta10,
        "v_mean": v_mean,
        "v_disp": v_disp,
        "r_end": r_end,
    }


def summarize_over_other_axes(values: np.ndarray, axis_to_keep: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    moved = np.moveaxis(values, axis_to_keep, 0)
    flat = moved.reshape(moved.shape[0], -1)

    median = np.full(moved.shape[0], np.nan)
    p16 = np.full(moved.shape[0], np.nan)
    p84 = np.full(moved.shape[0], np.nan)

    for i in range(moved.shape[0]):
        row = flat[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        median[i] = float(np.median(finite))
        p16[i] = float(np.percentile(finite, 16.0))
        p84[i] = float(np.percentile(finite, 84.0))

    return median, p16, p84


def plot_1d_dependence(
    output_dir: str,
    eta_m_values: np.ndarray,
    eta_m_cold_values: np.ndarray,
    eta_e_values: np.ndarray,
    metric_cube: np.ndarray,
    metric_key: str,
) -> None:
    label, unit = METRICS[metric_key]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    axis_defs = [
        (eta_m_values, 0, "$\\eta_M$"),
        (eta_m_cold_values, 1, "$\\eta_{M,\\mathrm{cold}}$"),
        (eta_e_values, 2, "$\\eta_E$"),
    ]

    for ax, (x, keep_axis, xlabel) in zip(axes, axis_defs):
        median, p16, p84 = summarize_over_other_axes(metric_cube, keep_axis)
        ax.plot(x, median, lw=2.0, color="tab:blue")
        ax.fill_between(x, p16, p84, color="tab:blue", alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{label} [{unit}]")
        ax.grid(alpha=0.25)

    fig.suptitle(f"{label} dependence (median and 16-84 percentile band)")

    path = os.path.join(output_dir, f"dependence_1d_{metric_key}.pdf")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_2d_slices(
    output_dir: str,
    eta_m_values: np.ndarray,
    eta_m_cold_values: np.ndarray,
    eta_e_values: np.ndarray,
    metric_cube: np.ndarray,
    metric_key: str,
) -> None:
    label, unit = METRICS[metric_key]

    idxs = [0, len(eta_e_values) // 2, len(eta_e_values) - 1]
    unique_idxs = []
    for idx in idxs:
        if idx not in unique_idxs:
            unique_idxs.append(idx)

    ncol = len(unique_idxs)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.8), constrained_layout=True)
    if ncol == 1:
        axes = [axes]

    finite_all = metric_cube[np.isfinite(metric_cube)]
    if finite_all.size == 0:
        return
    vmin = float(np.nanpercentile(finite_all, 5.0))
    vmax = float(np.nanpercentile(finite_all, 95.0))
    if vmax <= vmin:
        vmax = vmin + 1e-12

    for ax, idx in zip(axes, unique_idxs):
        arr = metric_cube[:, :, idx]
        im = ax.imshow(
            arr.T,
            origin="lower",
            aspect="auto",
            extent=[eta_m_values.min(), eta_m_values.max(), eta_m_cold_values.min(), eta_m_cold_values.max()],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_xlabel("$\\eta_M$")
        ax.set_ylabel("$\\eta_{M,\\mathrm{cold}}$")
        ax.set_title(f"$\\eta_E={eta_e_values[idx]:.2f}$")

    cbar = fig.colorbar(im, ax=axes, shrink=0.95)
    cbar.set_label(f"{label} [{unit}]")
    fig.suptitle(f"{label} across ($\\eta_M$, $\\eta_{{M,\\mathrm{{cold}}}}$) slices")

    path = os.path.join(output_dir, f"dependence_2d_slices_{metric_key}.pdf")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eta_m_values, eta_m_cold_values, eta_e_values = make_grid(args)

    shape = (len(eta_m_values), len(eta_m_cold_values), len(eta_e_values))
    metrics = {k: np.full(shape, np.nan, dtype=float) for k in METRICS}

    total = shape[0] * shape[1] * shape[2]
    done = 0

    for i, eta_m in enumerate(eta_m_values):
        for j, eta_m_cold in enumerate(eta_m_cold_values):
            for k, eta_e in enumerate(eta_e_values):
                out = run_case(args, float(eta_m), float(eta_m_cold), float(eta_e))
                for name in METRICS:
                    metrics[name][i, j, k] = out[name]

                done += 1
                if done == 1 or done % 10 == 0 or done == total:
                    print(
                        f"[{done:4d}/{total}] eta_M={eta_m:.3f}, "
                        f"eta_M_cold={eta_m_cold:.3f}, eta_E={eta_e:.3f}, "
                        f"r_end={out['r_end']:.2f} kpc"
                    )

    for metric_key, cube in metrics.items():
        plot_1d_dependence(
            args.output_dir,
            eta_m_values,
            eta_m_cold_values,
            eta_e_values,
            cube,
            metric_key,
        )
        plot_2d_slices(
            args.output_dir,
            eta_m_values,
            eta_m_cold_values,
            eta_e_values,
            cube,
            metric_key,
        )

    np.savez(
        os.path.join(args.output_dir, "parameter_dependence_data.npz"),
        eta_m_values=eta_m_values,
        eta_m_cold_values=eta_m_cold_values,
        eta_e_values=eta_e_values,
        **metrics,
    )

    print(f"Saved sweep outputs to {args.output_dir}/")


if __name__ == "__main__":
    main()
