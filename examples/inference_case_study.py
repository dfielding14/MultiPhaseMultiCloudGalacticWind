#!/usr/bin/env python
"""Case-study inference runs across far-ranging SFR and galaxy size choices."""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind.inference import (
    MomentInferenceModel,
    build_covariance,
    plot_corner,
    plot_moment_fit,
    summarize_parameter_degeneracies,
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "inference_case_study")


CASES = [
    {
        "name": "dwarf_low_sfr",
        "sfr": 0.3,
        "r_star_kpc": 0.08,
        "v_circ": 80.0,
        "theta_true": np.array([0.45, 0.10, 0.90], dtype=float),
        "frac_err": 0.15,
    },
    {
        "name": "disk_main_sequence",
        "sfr": 3.0,
        "r_star_kpc": 0.30,
        "v_circ": 140.0,
        "theta_true": np.array([0.25, 0.30, 1.00], dtype=float),
        "frac_err": 0.12,
    },
    {
        "name": "m82_like_starburst",
        "sfr": 20.0,
        "r_star_kpc": 0.30,
        "v_circ": 150.0,
        "theta_true": np.array([0.10, 0.20, 1.00], dtype=float),
        "frac_err": 0.10,
    },
    {
        "name": "high_sfr_extended",
        "sfr": 50.0,
        "r_star_kpc": 0.70,
        "v_circ": 200.0,
        "theta_true": np.array([0.15, 0.25, 1.00], dtype=float),
        "frac_err": 0.08,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-case inference study for SFR/size coverage")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-cloud-species", type=int, default=11)
    parser.add_argument("--r-max-kpc", type=float, default=30.0)
    parser.add_argument("--step-kpc", type=float, default=0.03)
    parser.add_argument("--first-step-kpc", type=float, default=1e-12)

    parser.add_argument("--hmc-warmup", type=int, default=140)
    parser.add_argument("--hmc-samples", type=int, default=280)
    parser.add_argument("--hmc-step-size", type=float, default=0.02)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=12)
    parser.add_argument("--hmc-target-accept", type=float, default=0.8)
    parser.add_argument("--sampler", choices=["nuts", "hmc"], default="nuts")
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--map-max-iter", type=int, default=25)

    parser.add_argument("--quick", action="store_true", help="Use fewer HMC steps for a faster dry run")
    return parser.parse_args()


def plot_case_overview(rows: list[dict[str, float]], output_path: str) -> None:
    sfr = np.asarray([row["sfr"] for row in rows], dtype=float)
    r_star = np.asarray([row["r_star_kpc"] for row in rows], dtype=float)

    m0 = np.asarray([row["true_M0"] for row in rows], dtype=float)
    m1 = np.asarray([row["true_M1"] for row in rows], dtype=float)
    m2 = np.asarray([row["true_M2"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    vals = [m0, m1, m2]
    labels = ["M0", "M1", "M2"]

    for ax, y, lab in zip(axes, vals, labels):
        sc = ax.scatter(sfr, y, c=r_star, cmap="viridis", s=95, edgecolor="black", linewidth=0.4)
        for i, row in enumerate(rows):
            ax.annotate(row["case"], (sfr[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("SFR [Msun/yr]")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=axes, location="right")
    cbar.set_label("r_star [kpc]")
    fig.suptitle("Case-study true moment scale across SFR/size regimes")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_degeneracy_heatmap(rows: list[dict[str, float]], output_path: str) -> None:
    pair_labels = ["eta_M-eta_M_cold", "eta_M-eta_E", "eta_M_cold-eta_E"]
    matrix = np.asarray(
        [
            [row["corr_etaM_etaMcold"], row["corr_etaM_etaE"], row["corr_etaMcold_etaE"]]
            for row in rows
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.8, 3.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["case"] for row in rows])
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(pair_labels, rotation=20, ha="right")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", color="black")

    ax.set_title("Posterior parameter correlations (HMC)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("correlation")

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.quick:
        args.map_max_iter = min(args.map_max_iter, 10)
        args.hmc_warmup = min(args.hmc_warmup, 40)
        args.hmc_samples = min(args.hmc_samples, 80)
        args.n_cloud_species = min(args.n_cloud_species, 8)
        args.num_chains = min(args.num_chains, 2)

    rng = np.random.default_rng(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    rows: list[dict[str, float]] = []

    for i, case in enumerate(CASES):
        case_name = case["name"]
        case_dir = os.path.join(args.output_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        model = MomentInferenceModel(
            sfr=case["sfr"],
            r_star_kpc=case["r_star_kpc"],
            v_circ=case["v_circ"],
            r_max_kpc=args.r_max_kpc,
            step_kpc=args.step_kpc,
            first_step_kpc=args.first_step_kpc,
            n_cloud_species=args.n_cloud_species,
        )

        theta_true = np.asarray(case["theta_true"], dtype=float)
        true_moments = model.predict_moments(theta_true)

        sigma = np.maximum(case["frac_err"] * true_moments, 1e-30)
        corr = np.asarray(
            [
                [1.0, 0.55, 0.35],
                [0.55, 1.0, 0.65],
                [0.35, 0.65, 1.0],
            ],
            dtype=float,
        )
        covariance = build_covariance(sigma, corr)

        observed = rng.multivariate_normal(true_moments, covariance)
        observed = np.maximum(observed, 1e-24)

        fit = model.fit_posterior(
            observed_moments=observed,
            covariance_moments=covariance,
            initial_theta=(0.2, 0.2, 1.0),
            map_max_iter=args.map_max_iter,
            hmc_num_warmup=args.hmc_warmup,
            hmc_num_samples=args.hmc_samples,
            hmc_step_size=args.hmc_step_size,
            hmc_leapfrog_steps=args.hmc_leapfrog_steps,
            hmc_target_accept=args.hmc_target_accept,
            sampler=args.sampler,
            num_chains=args.num_chains,
            seed=args.seed + 101 * i,
        )

        posterior_moment_samples = model.predict_moments_for_log_samples(fit.hmc.samples_log, max_samples=320)

        corner_path = os.path.join(case_dir, "posterior_corner.png")
        fit_path = os.path.join(case_dir, "moment_fit.png")

        plot_corner(
            fit.hmc.samples_theta,
            labels=MomentInferenceModel.PARAM_NAMES,
            output_path=corner_path,
            truths=theta_true,
            map_theta=fit.map.theta_map,
        )
        plot_moment_fit(
            observed_moments=observed,
            covariance_moments=covariance,
            map_moments=fit.map.predicted_moments,
            posterior_moment_samples=posterior_moment_samples,
            output_path=fit_path,
        )

        print(f"\nCASE: {case_name}")
        print(f"  SFR={case['sfr']:.3g} Msun/yr, r_star={case['r_star_kpc']:.3g} kpc, v_circ={case['v_circ']:.1f} km/s")
        print("  True moments [M0, M1, M2]:", [f"{x:.4e}" for x in true_moments])
        print("  Observed moments          :", [f"{x:.4e}" for x in observed])
        print("  MAP theta [eta_M, eta_M_cold, eta_E]:", [f"{x:.5f}" for x in fit.map.theta_map])
        print(
            f"  MAP chi2={fit.map.chi2:.3f}, "
            f"{fit.hmc.sampler} acceptance={fit.hmc.acceptance_rate:.3f}, "
            f"divergences={fit.hmc.num_divergent}"
        )

        deg_lines = summarize_parameter_degeneracies(fit.hmc.correlation_theta, MomentInferenceModel.PARAM_NAMES)
        for line in deg_lines:
            print(f"    {line}")

        rows.append(
            {
                "case": case_name,
                "sfr": float(case["sfr"]),
                "r_star_kpc": float(case["r_star_kpc"]),
                "v_circ": float(case["v_circ"]),
                "true_M0": float(true_moments[0]),
                "true_M1": float(true_moments[1]),
                "true_M2": float(true_moments[2]),
                "obs_M0": float(observed[0]),
                "obs_M1": float(observed[1]),
                "obs_M2": float(observed[2]),
                "map_eta_M": float(fit.map.theta_map[0]),
                "map_eta_M_cold": float(fit.map.theta_map[1]),
                "map_eta_E": float(fit.map.theta_map[2]),
                "chi2": float(fit.map.chi2),
                "hmc_acceptance": float(fit.hmc.acceptance_rate),
                "hmc_divergent": float(fit.hmc.num_divergent),
                "max_rhat": float(np.max(fit.hmc.r_hat)) if fit.hmc.r_hat is not None else np.nan,
                "corr_etaM_etaMcold": float(fit.hmc.correlation_theta[0, 1]),
                "corr_etaM_etaE": float(fit.hmc.correlation_theta[0, 2]),
                "corr_etaMcold_etaE": float(fit.hmc.correlation_theta[1, 2]),
            }
        )

    summary_csv = os.path.join(args.output_dir, "case_summary.csv")
    fieldnames = list(rows[0].keys())
    with open(summary_csv, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overview_plot = os.path.join(args.output_dir, "case_moment_overview.png")
    degeneracy_plot = os.path.join(args.output_dir, "case_degeneracy_heatmap.png")
    plot_case_overview(rows, overview_plot)
    plot_degeneracy_heatmap(rows, degeneracy_plot)

    print("\n=== CASE SUMMARY TABLE ===")
    print("case, SFR, r_star_kpc, true_M0, true_M1, true_M2, map_eta_M, map_eta_M_cold, map_eta_E")
    for row in rows:
        print(
            f"{row['case']}, {row['sfr']:.3g}, {row['r_star_kpc']:.3g}, "
            f"{row['true_M0']:.4e}, {row['true_M1']:.4e}, {row['true_M2']:.4e}, "
            f"{row['map_eta_M']:.4f}, {row['map_eta_M_cold']:.4f}, {row['map_eta_E']:.4f}"
        )

    print(f"\nSaved case-study outputs to {args.output_dir}")
    print(f"  - {summary_csv}")
    print(f"  - {overview_plot}")
    print(f"  - {degeneracy_plot}")


if __name__ == "__main__":
    main()
