#!/usr/bin/env python3
"""
M82-like multiphase wind example script (notebook-equivalent workflow).

This script mirrors `m82_multiphase_user_notebook.ipynb` and adds:
- CLI overrides for key parameters
- step timing output
- optional cProfile capture for performance analysis
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind import (
    WindModel,
    setup_plotting_style,
    plot_wind_solution,
    plot_profiles,
    plot_column_density_distribution,
)
from multiphasegalacticwind.constants import Msun


@contextmanager
def timed(label: str, timings: dict[str, float]) -> None:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[label] = time.perf_counter() - start


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sfr", type=float, default=20.0, help="Star formation rate [Msun/yr]")
    p.add_argument("--v-circ", type=float, default=150.0, help="Circular velocity [km/s]")
    p.add_argument("--eta-m-hot", type=float, default=0.1, help="Hot mass-loading factor")
    p.add_argument("--eta-m-cold", type=float, default=0.2, help="Total cold mass-loading factor")
    p.add_argument("--eta-e-hot", type=float, default=1.0, help="Hot energy-loading factor")
    p.add_argument("--m-cloud-min", type=float, default=1.0, help="Minimum cloud mass [Msun]")
    p.add_argument("--m-cloud-max", type=float, default=1.0e6, help="Maximum cloud mass [Msun]")
    p.add_argument("--alpha-cloud", type=float, default=2.0, help="Cloud mass-spectrum slope")
    p.add_argument("--n-cloud-bins", type=int, default=13, help="Number of cloud mass bins")
    p.add_argument("--r-star-kpc", type=float, default=0.3, help="Sonic-point radius [kpc]")
    p.add_argument("--r-max-kpc", type=float, default=30.0, help="Max integration radius [kpc]")
    p.add_argument(
        "--cloud-radial-offset",
        type=float,
        default=0.01,
        help="Fractional cloud launch offset from sonic point (config.cloud_radial_offset)",
    )
    p.add_argument(
        "--solver-max-step-kpc",
        type=float,
        default=0.3,
        help="Max RK solver step [kpc]",
    )
    p.add_argument(
        "--solver-first-step-kpc",
        type=float,
        default=1e-12,
        help="Initial RK solver step [kpc]; set <=0 to disable explicit first_step",
    )
    p.add_argument(
        "--r-min-obs-kpc",
        type=float,
        default=None,
        help="Minimum radius for observables [kpc] (default: r_star_kpc)",
    )
    p.add_argument("--r-max-obs-kpc", type=float, default=30.0, help="Max radius for observables [kpc]")
    p.add_argument("--rtol", type=float, default=1e-6, help="ODE relative tolerance")
    p.add_argument("--atol", type=float, default=1e-8, help="ODE absolute tolerance")
    p.add_argument("--save-plots", action="store_true", help="Save PDF plots")
    p.add_argument("--plots-dir", default="plots_m82_script", help="Directory for output plots")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    p.add_argument("--skip-plots", action="store_true", help="Skip all plot generation")
    p.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile and print top cumulative functions",
    )
    p.add_argument(
        "--profile-lines",
        type=int,
        default=20,
        help="Number of lines to print from cProfile summary",
    )
    return p


def run_workflow(args: argparse.Namespace) -> dict[str, float]:
    timings: dict[str, float] = {}
    r_min_obs_kpc = args.r_min_obs_kpc if args.r_min_obs_kpc is not None else args.r_star_kpc

    with timed("setup_plotting", timings):
        setup_plotting_style()

    print("Current run setup:")
    print(f"  SFR = {args.sfr:.1f} Msun/yr")
    print(f"  v_circ = {args.v_circ:.1f} km/s")
    print(f"  eta_M_hot = {args.eta_m_hot:.3f}")
    print(f"  eta_M_cold_tot = {args.eta_m_cold:.3f}")
    print(f"  eta_E_hot = {args.eta_e_hot:.3f}")
    print(f"  alpha_cloud = {args.alpha_cloud:.1f} -> dN/dlogM ~ M^{1.0 - args.alpha_cloud:.1f}")
    print(f"  cloud mass range = [{args.m_cloud_min:.1f}, {args.m_cloud_max:.1e}] Msun")
    print(f"  cloud bins = {args.n_cloud_bins}")

    with timed("construct_model", timings):
        model = WindModel(
            SFR=args.sfr,
            v_circ=args.v_circ,
            eta_M=args.eta_m_hot,
            eta_M_cold_tot=args.eta_m_cold,
            eta_E=args.eta_e_hot,
            r_star_kpc=args.r_star_kpc,
            cloud_mass_range=(args.m_cloud_min, args.m_cloud_max),
            cloud_alpha=args.alpha_cloud,
            N_cloud_species=args.n_cloud_bins,
            r_max_kpc=args.r_max_kpc,
            rtol=args.rtol,
            atol=args.atol,
            cloud_radial_offset=args.cloud_radial_offset,
            solver_max_step_kpc=args.solver_max_step_kpc,
            solver_first_step_kpc=(
                args.solver_first_step_kpc if args.solver_first_step_kpc > 0 else None
            ),
        )

    print("Model constructed. Initial cloud masses [Msun]:")
    print(np.array2string(model.M_cloud0 / Msun, precision=3))

    with timed("integrate_solution", timings):
        solution = model.run()

    print("Integration summary:")
    print(f"  status = {solution.sol.status}")
    print(f"  message = {solution.sol.message}")
    print(f"  max radius reached = {solution.r[-1]:.2f} kpc")

    if solution.r[-1] >= 10.0:
        print(f"  v(10 kpc) = {solution.v_at_10kpc:.1f} km/s")
        print(f"  Mdot/SFR(10 kpc) = {solution.mass_loading_at_10kpc:.3f}")

    obs_r_max = min(args.r_max_obs_kpc, float(solution.r[-1]))
    if obs_r_max <= r_min_obs_kpc:
        raise RuntimeError("Insufficient radial span for column-density diagnostics.")

    with timed("velocity_moments", timings):
        moments = solution.calculate_velocity_moments(
            r_min_kpc=r_min_obs_kpc,
            r_max_kpc=obs_r_max,
        )

    if "mean" in moments:
        print(f"  <v> = {moments['mean']:.1f} km/s")
        print(f"  sigma_v = {moments['dispersion']:.1f} km/s")

    with timed("column_density_total", timings):
        solution.calculate_column_density_distribution(
            r_min_kpc=r_min_obs_kpc,
            r_max_kpc=obs_r_max,
        )

    with timed("column_density_by_species", timings):
        solution.calculate_column_density_by_species(
            r_min_kpc=r_min_obs_kpc,
            r_max_kpc=obs_r_max,
        )

    if not args.skip_plots:
        with timed("plot_wind_solution", timings):
            fig_wind, _ = plot_wind_solution(
                solution,
                show_hot_only=True,
                show_clouds=True,
                figsize=(5.0, 8.0),
            )
            fig_wind.suptitle(
                rf"${{\rm SFR}}={{{args.sfr}}} \, M_\odot / {{\rm yr}} \quad  \eta_M={{{args.eta_m_hot}}}, \eta_{{M,cold}}={{{args.eta_m_cold}}}, \eta_E={{{args.eta_e_hot}}}$",
            )
            if args.save_plots:
                os.makedirs(args.plots_dir, exist_ok=True)
                fig_wind.savefig(
                    os.path.join(args.plots_dir, "m82_wind_solution.pdf"),
                    dpi=300,
                    bbox_inches="tight",
                )

        with timed("plot_profiles", timings):
            fig_profiles, _ = plot_profiles(
                solution,
                quantities=["velocity", "density", "temperature", "metallicity"],
                figsize=(6.0, 9.0),
            )
            fig_profiles.suptitle("Profile summary", y=1.01)
            if args.save_plots:
                fig_profiles.savefig(
                    os.path.join(args.plots_dir, "m82_profile_summary.pdf"),
                    dpi=300,
                    bbox_inches="tight",
                )

        with timed("plot_column_density_total", timings):
            fig_col_total, _ = plot_column_density_distribution(
                solution,
                r_min_kpc=r_min_obs_kpc,
                r_max_kpc=obs_r_max,
                show_moments=True,
                xlim=(0, 1200),
                figsize=(6.5, 4.5),
            )
            if args.save_plots:
                fig_col_total.savefig(
                    os.path.join(args.plots_dir, "m82_column_density_total.pdf"),
                    dpi=300,
                    bbox_inches="tight",
                )

        with timed("plot_column_density_by_species", timings):
            fig_col_species, _ = plot_column_density_distribution(
                solution,
                r_min_kpc=r_min_obs_kpc,
                r_max_kpc=obs_r_max,
                show_species=True,
                species_alpha=0.6,
                log_scale=True,
                xlim=(0, 1200),
                figsize=(6.5, 4.5),
            )
            if args.save_plots:
                fig_col_species.savefig(
                    os.path.join(args.plots_dir, "m82_column_density_by_species.pdf"),
                    dpi=300,
                    bbox_inches="tight",
                )
                print(f"Saved plots to {args.plots_dir}/")

        if not args.no_show:
            plt.show()
        else:
            plt.close("all")

    total = sum(timings.values())
    print("\nTiming summary:")
    for key, val in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
        frac = (100.0 * val / total) if total > 0 else 0.0
        print(f"  {key:28s} {val:8.4f} s  ({frac:5.1f}%)")
    print(f"  {'TOTAL':28s} {total:8.4f} s")

    return timings


def main() -> None:
    args = build_parser().parse_args()
    if not args.profile:
        run_workflow(args)
        return

    profiler = cProfile.Profile()
    profiler.enable()
    run_workflow(args)
    profiler.disable()

    print("\nTop cProfile entries (cumulative time):")
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.profile_lines)


if __name__ == "__main__":
    main()
