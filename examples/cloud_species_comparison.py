#!/usr/bin/env python
"""
Cloud species discretization comparison.

This script compares solutions with different `N_cloud_species` values and
produces summary plots showing convergence of key observables.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.observables import calculate_velocity_moments


def build_model(n_species: int) -> WindModel:
    """Create a model for a given cloud species count."""
    return WindModel(
        v_circ=150.0,
        redshift=0.0,
        SFR=20.0,
        eta_M=0.2,
        eta_M_cold=0.2,
        eta_E=1.0,
        r_star_kpc=0.3,
        cloud_mass_range=(10.0, 1e6),
        cloud_alpha=2.0,
        N_cloud_species=n_species,
        T_cl=1e4,
        v_cloud_init=30.0,
        r_max_kpc=20.0,
        rtol=1e-6,
        atol=1e-8,
    )


def run_case(n_species: int):
    """Run one discretization case and return model/solution/metrics."""
    model = build_model(n_species)
    solution = model.run()

    v_col, dN_dv = solution.calculate_column_density_distribution(
        r_min_kpc=0.5,
        r_max_kpc=20.0,
    )

    moments = calculate_velocity_moments(v_col, dN_dv)
    mean_v = moments.get("mean", np.nan)
    disp_v = moments.get("dispersion", np.nan)

    return model, solution, v_col, dN_dv, mean_v, disp_v


def main() -> None:
    output_dir = "plots_species"
    os.makedirs(output_dir, exist_ok=True)

    n_species_list = [1, 3, 6, 11, 21]

    print("Running discretization comparison...")
    results = {}
    for n_species in n_species_list:
        print(f"\n{'=' * 48}")
        print(f"Case: N_cloud_species = {n_species}")

        model, solution, v_col, dN_dv, mean_v, disp_v = run_case(n_species)
        results[n_species] = {
            "model": model,
            "solution": solution,
            "v_col": v_col,
            "dN_dv": dN_dv,
            "mean_v": mean_v,
            "disp_v": disp_v,
        }

        print(f"  status: {solution.sol.status}")
        print(f"  message: {solution.sol.message}")
        print(f"  r_end: {solution.r[-1]:.2f} kpc")
        print(f"  v_end: {solution.v[-1]:.1f} km/s")
        print(f"  eta(10 kpc): {solution.mass_loading_at_10kpc:.3f}")
        if np.isfinite(mean_v):
            print(f"  <v>: {mean_v:.1f} km/s, sigma_v: {disp_v:.1f} km/s")

    # Plot 1: wind velocity profiles for different discretizations.
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for n_species in n_species_list:
        sol = results[n_species]["solution"]
        ax1.loglog(sol.r, sol.v, lw=1.5, label=f"N={n_species}")

    ax1.set_xlabel(r"$r$ [kpc]")
    ax1.set_ylabel(r"$v_{\rm wind}$ [km/s]")
    ax1.set_xlim(0.3, 20)
    ax1.set_ylim(50, 3000)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("Wind Velocity vs Cloud-Mass Discretization")
    fig1.savefig(os.path.join(output_dir, "species_velocity_profiles.pdf"), dpi=300)
    plt.close(fig1)

    # Plot 2: dN/dv profiles.
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    for n_species in n_species_list:
        v_col = results[n_species]["v_col"]
        dN_dv = results[n_species]["dN_dv"]
        positive = dN_dv > 0
        if np.any(positive):
            ax2.plot(v_col[positive], dN_dv[positive], lw=1.5, label=f"N={n_species}")

    ax2.set_xlabel(r"$v$ [km/s]")
    ax2.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km/s)$^{-1}$]")
    ax2.set_xlim(0, 1000)
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("Column-Density Distribution vs Discretization")
    fig2.savefig(os.path.join(output_dir, "species_column_density_profiles.pdf"), dpi=300)
    plt.close(fig2)

    # Plot 3: summary metrics vs N.
    mass_loading = [results[n]["solution"].mass_loading_at_10kpc for n in n_species_list]
    mean_velocity = [results[n]["mean_v"] for n in n_species_list]
    disp_velocity = [results[n]["disp_v"] for n in n_species_list]

    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    ax3.plot(n_species_list, mass_loading, "o-", lw=1.8)
    ax3.set_xlabel("N_cloud_species")
    ax3.set_ylabel(r"$\eta(10\,\mathrm{kpc})$")
    ax3.set_title("Mass Loading Convergence")
    ax3.grid(alpha=0.3)

    ax4.errorbar(n_species_list, mean_velocity, yerr=disp_velocity, fmt="o-", lw=1.8, capsize=4)
    ax4.set_xlabel("N_cloud_species")
    ax4.set_ylabel(r"$\langle v \rangle$ [km/s]")
    ax4.set_title("Velocity Moments Convergence")
    ax4.grid(alpha=0.3)

    fig3.savefig(os.path.join(output_dir, "species_summary_metrics.pdf"), dpi=300)
    plt.close(fig3)

    print(f"\nSaved plots to {output_dir}/")
    print("  - species_velocity_profiles.pdf")
    print("  - species_column_density_profiles.pdf")
    print("  - species_summary_metrics.pdf")


if __name__ == "__main__":
    main()
