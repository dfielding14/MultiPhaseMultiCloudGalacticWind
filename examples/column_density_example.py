#!/usr/bin/env python
"""
Example showing robust column-density calculations and plotting.

This example demonstrates:
- Computing dN/dv for a stable baseline model
- Comparing linear vs log representations safely
- Parameter sweep over cold mass loading
- Species-level contribution diagnostics
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from multiphasegalacticwind import WindModel, plot_column_density_distribution
from multiphasegalacticwind.constants import Msun


def run_model(eta_M_cold: float) -> WindModel:
    model = WindModel(
        SFR=20.0,
        eta_M=0.2,
        eta_M_cold=eta_M_cold,
        eta_E=1.0,
        v_circ=150.0,
        N_cloud_species=10,
        r_max_kpc=40.0,
        rtol=1e-6,
        atol=1e-8,
    )
    return model


def main() -> None:
    output_dir = "plots_column_density"
    os.makedirs(output_dir, exist_ok=True)

    print("Creating baseline wind model...")
    model = run_model(eta_M_cold=0.2)
    print("Running baseline simulation...")
    solution = model.run()

    if solution.sol.status < 0:
        raise RuntimeError(f"Baseline integration failed: {solution.sol.message}")

    print("\nPlotting column density distribution...")
    fig1, ax1 = plot_column_density_distribution(
        solution,
        r_min_kpc=0.5,
        r_max_kpc=30.0,
        show_species=True,
        log_scale=True,
        xlim=(0, 900),
        figsize=(6, 4.5),
    )
    ax1.set_title("Column Density Distribution")
    fig1.savefig(
        os.path.join(output_dir, "column_density_distribution.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig1)

    v_cloud, dN_dv_column = solution.calculate_column_density_distribution(
        r_min_kpc=0.5,
        r_max_kpc=30.0,
    )

    print("\nColumn density distribution properties:")
    print(f"  Velocity range: {v_cloud.min():.1f} - {v_cloud.max():.1f} km/s")
    print(f"  Max dN/dv: {dN_dv_column.max():.2e} cm^-2 / (km/s)")
    print(f"  Integrated column density: {np.trapezoid(dN_dv_column, v_cloud):.2e} cm^-2")

    # Compare linear vs log scale with safe handling for non-positive values.
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax2.plot(v_cloud, dN_dv_column, "k-", lw=1.5)
    ax2.set_xlabel(r"$v$ [km s$^{-1}$]")
    ax2.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]")
    ax2.set_title("Linear Scale")
    ax2.set_xlim(0, 900)

    positive = dN_dv_column > 0
    if np.any(positive):
        ax3.plot(v_cloud[positive], dN_dv_column[positive], "k-", lw=1.5)
        ax3.set_yscale("log")
    else:
        ax3.text(0.5, 0.5, "No positive dN/dv", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel(r"$v$ [km s$^{-1}$]")
    ax3.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]")
    ax3.set_title("Log Scale")
    ax3.set_xlim(0, 900)

    fig2.savefig(
        os.path.join(output_dir, "column_density_linear_vs_log.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig2)

    print("\nParameter study: effect of eta_M_cold on dN/dv")
    eta_M_cold_values = [0.05, 0.1, 0.2, 0.3]

    fig3, ax4 = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    for eta_cold in eta_M_cold_values:
        model_var = run_model(eta_M_cold=eta_cold)
        sol_var = model_var.run()

        if sol_var.sol.status < 0:
            print(f"  Skipping eta_M_cold={eta_cold:.2f} (failed: {sol_var.sol.message})")
            continue

        v_var, dN_dv_var = sol_var.calculate_column_density_distribution(
            r_min_kpc=0.5,
            r_max_kpc=30.0,
        )
        peak = np.max(dN_dv_var)
        if peak > 0:
            ax4.plot(v_var, dN_dv_var / peak, label=fr"$\eta_{{M,cold}}={eta_cold:.2f}$")

    ax4.set_xlabel(r"$v$ [km s$^{-1}$]")
    ax4.set_ylabel(r"Normalized $dN/dv$")
    ax4.set_xlim(0, 900)
    ax4.set_ylim(0, 1.05)
    ax4.legend(frameon=False)
    ax4.set_title("Column Density vs Cold Mass Loading")

    fig3.savefig(
        os.path.join(output_dir, "column_density_parameter_study.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig3)

    print("\nAnalyzing individual cloud species...")
    fig4, ax5 = plt.subplots(figsize=(6.5, 5), constrained_layout=True)

    v_total, dN_dv_total = solution.calculate_column_density_distribution(
        r_min_kpc=0.5,
        r_max_kpc=30.0,
    )
    ax5.plot(v_total, dN_dv_total, "k-", lw=2, label="Total")

    species_indices = [0, model.N_cloud_species // 2, model.N_cloud_species - 1]
    for i in species_indices:
        v_i, dN_dv_i = solution.calculate_column_density_distribution(
            cloud_index=i,
            r_min_kpc=0.5,
            r_max_kpc=30.0,
        )
        M_cloud0_msun = model.M_cloud0[i] / Msun
        ax5.plot(
            v_i,
            dN_dv_i,
            "--",
            alpha=0.8,
            label=fr"$M_{{cl,0}}={M_cloud0_msun:.1e}\,M_\odot$",
        )

    ax5.set_xlabel(r"$v$ [km s$^{-1}$]")
    ax5.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]")
    ax5.set_xlim(0, 900)
    ax5.legend(frameon=False)
    ax5.set_title("Column Density by Cloud Mass")

    fig4.savefig(
        os.path.join(output_dir, "column_density_by_mass_manual.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig4)

    print("\nUsing automatic species decomposition...")
    v_species, dN_dv_species = solution.calculate_column_density_by_species(
        r_min_kpc=0.5,
        r_max_kpc=30.0,
    )

    print(f"  Number of cloud species: {len(dN_dv_species['species'])}")
    print(
        f"  Cloud masses range from {dN_dv_species['M_cloud0'].min():.1e} to "
        f"{dN_dv_species['M_cloud0'].max():.1e} Msun"
    )

    fig5, ax6 = plot_column_density_distribution(
        solution,
        show_species=True,
        r_min_kpc=0.5,
        r_max_kpc=30.0,
        xlim=(0, 900),
        log_scale=True,
    )
    ax6.set_title("Column Density by Cloud Mass (Automatic)")
    fig5.savefig(
        os.path.join(output_dir, "column_density_by_mass_auto.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig5)

    print("\nDominant cloud species by velocity:")
    v_bins = [100, 250, 400, 550, 700]  # km/s

    for v_target in v_bins:
        idx = int(np.argmin(np.abs(v_species - v_target)))
        v_actual = float(v_species[idx])

        contributions = np.array([dN_dv[idx] for dN_dv in dN_dv_species["species"]])
        total_here = float(dN_dv_species["total"][idx])

        dominant_idx = int(np.argmax(contributions))
        M_dominant = float(dN_dv_species["M_cloud0"][dominant_idx])

        if total_here > 0:
            fraction = contributions[dominant_idx] / total_here
            frac_msg = f"{fraction * 100:.0f}% of total"
        else:
            frac_msg = "no signal"

        print(
            f"  At v={v_actual:.0f} km/s: "
            f"M_cl,0={M_dominant:.1e} Msun dominates ({frac_msg})"
        )

    print(f"\nExample complete. Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
