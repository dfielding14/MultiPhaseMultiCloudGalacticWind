#!/usr/bin/env python
"""
Comprehensive multiphase galactic-wind example with plotting and parameter study.

This example demonstrates:
1. Model setup with explicit physical parameters
2. Integration and key diagnostics
3. Publication-style plotting
4. A small eta_M_cold sweep
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from multiphasegalacticwind import (
    WindModel,
    setup_plotting_style,
    plot_wind_solution,
    plot_column_density_distribution,
)
from multiphasegalacticwind.constants import Msun


def main() -> None:
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    setup_plotting_style()

    print("Setting up wind model with specified parameters...")
    model = WindModel(
        v_circ=150.0,
        redshift=0.0,
        SFR=20.0,
        eta_M=0.1,
        eta_M_cold=0.1,
        eta_E=1.0,
        r_star_kpc=0.3,
        cloud_mass_range=(10, 1e5),
        cloud_alpha=2.0,
        N_cloud_species=5,
        T_cl=1e4,
        r_max_kpc=100.0,
        rtol=1e-6,
        atol=1e-8,
        progress_callback="print",
        progress_interval=10.0,
    )

    print("\nModel parameters:")
    print(f"  SFR = {model.SFR} Msun/yr")
    print(f"  eta_M = {model.eta_M}")
    print(f"  eta_M_cold = {model.eta_M_cold}")
    print(f"  eta_E = {model.eta_E}")
    print(f"  r_sonic = {model.r_star_kpc} kpc = {model.r_star_kpc * 1000:.1f} pc")
    print(f"  N_cloud_species = {model.N_cloud_species}")
    print(f"  Cloud masses: {(model.M_cloud0 / Msun)} Msun")

    print("\nRunning wind integration...")
    solution = model.run()

    print("\nIntegration complete!")
    print(f"Maximum radius reached: {solution.r[-1]:.1f} kpc")
    print(f"Final status: {solution.sol.status} - {solution.sol.message}")

    if solution.sol.status < 0:
        raise RuntimeError("Primary model failed; aborting comprehensive example")

    if solution.r[-1] >= 10:
        idx_10kpc = int(np.argmin(np.abs(solution.r - 10.0)))
        print("\nResults at 10 kpc:")
        print(f"  Wind velocity: {solution.v_at_10kpc:.1f} km/s")
        print(f"  Mass loading: {solution.mass_loading_at_10kpc:.3f}")
        print(f"  Wind density: {solution.n[idx_10kpc]:.2e} cm^-3")
        print(f"  Wind temperature: {solution.T[idx_10kpc]:.2e} K")
        print(f"  Total cloud mass: {solution.M_cloud_tot[idx_10kpc]:.2e} Msun")

    print("\nCreating wind solution plot...")
    fig1, _ = plot_wind_solution(solution, show_hot_only=True, show_clouds=True)
    fig1.suptitle(
        f"Wind Solution: SFR={model.SFR}, $\\eta_M$={model.eta_M}, "
        f"$\\eta_{{M,cold}}$={model.eta_M_cold}"
    )
    plot_path = os.path.join(plots_dir, "wind_solution_comprehensive.pdf")
    fig1.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {plot_path}")

    print("\nCalculating column density distribution...")
    v_cloud, dN_dv = solution.calculate_column_density_distribution(
        r_min_kpc=0.5,
        r_max_kpc=50.0,
    )

    moments = solution.calculate_velocity_moments(r_min_kpc=0.5, r_max_kpc=50.0)
    print("\nVelocity distribution statistics:")
    if "mean" in moments:
        print(f"  Mean velocity: {moments['mean']:.1f} km/s")
        print(f"  Velocity dispersion: {moments['dispersion']:.1f} km/s")
    else:
        print("  Could not calculate velocity moments")

    fig2, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(v_cloud, dN_dv, "k-", lw=1.5)
    ax.set_xlabel(r"$v$ [km s$^{-1}$]")
    ax.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]")
    ax.set_xlim(0, 900)
    ax.set_ylim(0, None)

    if "mean" in moments:
        ax.axvline(moments["mean"], color="red", ls="--", alpha=0.7, label=f"Mean={moments['mean']:.0f} km/s")
        if moments.get("dispersion", 0) > 0:
            ax.axvline(moments["mean"] - moments["dispersion"], color="blue", ls=":", alpha=0.5)
            ax.axvline(moments["mean"] + moments["dispersion"], color="blue", ls=":", alpha=0.5)
            ax.plot([], [], color="blue", ls=":", label=f"$\\sigma$={moments['dispersion']:.0f} km/s")

    ax.legend(frameon=False)
    ax.set_title("Velocity Distribution (r = 0.5-50 kpc)")
    fig2.tight_layout()
    plot_path = os.path.join(plots_dir, "velocity_distribution_comprehensive.pdf")
    fig2.savefig(plot_path, dpi=300)
    plt.close(fig2)
    print(f"Saved: {plot_path}")

    fig3, ax = plot_column_density_distribution(
        solution,
        r_min_kpc=0.5,
        r_max_kpc=50.0,
        show_species=True,
        species_alpha=0.6,
        figsize=(6, 4.5),
        xlim=(0, 900),
        log_scale=True,
    )
    ax.set_title("Column Density Distribution by Cloud Mass")
    plot_path = os.path.join(plots_dir, "column_density_comprehensive.pdf")
    fig3.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {plot_path}")

    print("\nPerforming parameter study: varying eta_M_cold...")
    eta_M_cold_values = [0.01, 0.05, 0.1, 0.15, 0.2]

    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for eta_M_cold in eta_M_cold_values:
        print(f"  Running with eta_M_cold = {eta_M_cold}...")
        model_var = WindModel(
            SFR=20.0,
            eta_M=0.1,
            eta_M_cold=eta_M_cold,
            eta_E=1.0,
            r_star_kpc=0.3,
            cloud_mass_range=(10, 1e5),
            N_cloud_species=5,
            progress_callback=None,
        )
        sol_var = model_var.run()

        if sol_var.sol.status < 0:
            print(f"    Skipping failed run: {sol_var.sol.message}")
            continue

        ax1.loglog(sol_var.r, sol_var.v, label=fr"$\eta_{{M,cold}}={eta_M_cold}$")

        v_cl, dN_dv_var = sol_var.calculate_column_density_distribution(r_min_kpc=0.5, r_max_kpc=50.0)
        peak = np.max(dN_dv_var)
        if peak > 0:
            ax2.plot(v_cl, dN_dv_var / peak, label=fr"$\eta_{{M,cold}}={eta_M_cold}$")

    ax1.set_xlabel(r"$r$ [kpc]")
    ax1.set_ylabel(r"$v$ [km/s]")
    ax1.set_xlim(0.3, 100)
    ax1.set_ylim(50, 1000)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("Wind Velocity Profiles")

    ax2.set_xlabel(r"$v$ [km/s]")
    ax2.set_ylabel(r"Normalized $dN/dv$")
    ax2.set_xlim(0, 900)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title("Velocity Distributions")

    fig4.tight_layout()
    plot_path = os.path.join(plots_dir, "parameter_study_eta_M_cold.pdf")
    fig4.savefig(plot_path, dpi=300)
    plt.close(fig4)
    print(f"Saved: {plot_path}")

    print("\nAll plots generated successfully!")
    print(f"Generated files in {plots_dir}/:")
    print("  - wind_solution_comprehensive.pdf")
    print("  - velocity_distribution_comprehensive.pdf")
    print("  - column_density_comprehensive.pdf")
    print("  - parameter_study_eta_M_cold.pdf")


if __name__ == "__main__":
    main()
