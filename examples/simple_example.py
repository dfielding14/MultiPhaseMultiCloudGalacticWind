#!/usr/bin/env python
"""
Simple working example of the multiphase galactic wind model.

This example shows basic model setup, integration, and plotting.
"""

import os
import matplotlib.pyplot as plt

from multiphasegalacticwind import (
    WindModel,
    plot_wind_solution,
    plot_column_density_distribution,
)


def main() -> None:
    output_dir = "plots_simple"
    os.makedirs(output_dir, exist_ok=True)

    print("Creating wind model...")
    model = WindModel(
        SFR=10.0,       # Msun/yr
        eta_M=0.2,      # hot mass loading
        eta_M_cold=0.1, # cold mass loading
        r_max_kpc=30.0,
        rtol=1e-6,
    )

    print("Running wind integration...")
    solution = model.run()
    if solution.sol.status < 0:
        raise RuntimeError(f"Integration failed: {solution.sol.message}")

    print("Integration complete!")
    print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
    print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")

    moments = solution.calculate_velocity_moments(r_min_kpc=0.5, r_max_kpc=30.0)
    if "mean" in moments:
        print("\nVelocity distribution:")
        print(f"  Mean: {moments['mean']:.1f} km/s")
        print(f"  Dispersion: {moments['dispersion']:.1f} km/s")

    v_cloud, dN_dv = solution.calculate_column_density_distribution(
        r_min_kpc=0.5,
        r_max_kpc=30.0,
    )
    print("\nColumn density distribution:")
    print(f"  Velocity span: {v_cloud.min():.1f} to {v_cloud.max():.1f} km/s")
    print(f"  Max dN/dv: {dN_dv.max():.2e} cm^-2 / (km/s)")

    fig1, _ = plot_wind_solution(solution)
    fig1.savefig(os.path.join(output_dir, "wind_profiles.pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, _ = plot_column_density_distribution(
        solution,
        show_moments=True,
        r_min_kpc=0.5,
        r_max_kpc=30.0,
    )
    fig2.savefig(
        os.path.join(output_dir, "column_density_distribution.pdf"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig2)

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
