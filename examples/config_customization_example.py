#!/usr/bin/env python
"""
Example: compare baseline vs custom WindConfig choices.

This demonstrates practical configuration tuning and its impact on
velocity, temperature, and column-density observables.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from multiphasegalacticwind import WindModel, WindConfig


def run_case(label: str, config: WindConfig):
    model = WindModel(
        SFR=12.0,
        eta_M=0.2,
        eta_M_cold=0.15,
        eta_E=1.0,
        v_circ=160.0,
        r_star_kpc=0.35,
        cloud_mass_range=(10, 1e5),
        cloud_alpha=2.0,
        N_cloud_species=8,
        r_max_kpc=30.0,
        rtol=1e-6,
        atol=1e-8,
        config=config,
    )
    solution = model.run()
    return label, model, solution


def main() -> None:
    output_dir = "plots_config"
    os.makedirs(output_dir, exist_ok=True)

    baseline_config = WindConfig()

    tuned_config = WindConfig(
        Z_hot_over_Z_solar=1.5,
        f_turb0=0.18,
        drag_coeff=0.8,
        Cooling_Factor=0.6,
        cold_cloud_injection_radial_extent_frac=2.0,
        cold_cloud_injection_radial_power=4,
    )

    cases = [
        run_case("Baseline", baseline_config),
        run_case("Tuned", tuned_config),
    ]

    for label, _, sol in cases:
        status = "ok" if sol.sol.status >= 0 else "failed"
        print(f"{label}: {status}, r_end={sol.r[-1]:.2f} kpc, v_end={sol.v[-1]:.1f} km/s")

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True, constrained_layout=True)

    for label, _, sol in cases:
        ax1.loglog(sol.r, sol.v, lw=1.7, label=label)
        ax2.loglog(sol.r, sol.T, lw=1.7, label=label)
        ax3.loglog(sol.r, sol.Mdot / sol.model.SFR, lw=1.7, label=label)

    ax1.set_ylabel(r"$v$ [km/s]")
    ax1.legend(frameon=False)
    ax1.set_title("Profile Comparison: Baseline vs Tuned Config")

    ax2.set_ylabel(r"$T$ [K]")
    ax3.set_ylabel(r"$\dot{M}/\mathrm{SFR}$")
    ax3.set_xlabel(r"$r$ [kpc]")

    fig1.savefig(os.path.join(output_dir, "config_profile_comparison.pdf"), dpi=300)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)

    for label, _, sol in cases:
        v, dN_dv = sol.calculate_column_density_distribution(r_min_kpc=0.5, r_max_kpc=30.0)
        positive = dN_dv > 0
        if np.any(positive):
            ax.plot(v[positive], dN_dv[positive], lw=1.7, label=label)

        moments = sol.calculate_velocity_moments(r_min_kpc=0.5, r_max_kpc=30.0)
        if "mean" in moments:
            print(
                f"{label}: <v>={moments['mean']:.1f} km/s, "
                f"sigma_v={moments['dispersion']:.1f} km/s, "
                f"eta(10kpc)={sol.mass_loading_at_10kpc:.3f}"
            )

    ax.set_xlabel(r"$v$ [km/s]")
    ax.set_ylabel(r"$dN/dv$ [cm$^{-2}$ (km/s)$^{-1}$]")
    ax.set_xlim(0, 1000)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.set_title("Column-Density Comparison")

    fig2.savefig(os.path.join(output_dir, "config_column_density_comparison.pdf"), dpi=300)
    plt.close(fig2)

    print(f"Saved plots to {output_dir}/")


if __name__ == "__main__":
    main()
