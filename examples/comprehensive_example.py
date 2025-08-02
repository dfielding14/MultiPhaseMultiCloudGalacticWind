#!/usr/bin/env python
"""
Comprehensive example of multiphase galactic wind model with observational comparisons.

This example demonstrates:
1. Setting up a wind model with specific physical parameters
2. Running the integration
3. Creating publication-quality plots
4. Calculating velocity distributions (dN/dv) for observational comparison

Physical parameters:
- SFR = 20 Msun/yr
- eta_E = 1.0 (energy loading)
- eta_M = 0.1 (hot phase mass loading)
- eta_M_cold = 0.1 (cold phase mass loading)
- 5 cloud masses from 10 to 1e5 Msun
- Sonic radius = 300 pc
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import (WindModel, setup_plotting_style, 
                                   plot_wind_solution, plot_column_density_distribution)


def main():
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}/")
    
    # Set up plotting style for publication-quality figures
    setup_plotting_style()
    
    # Create wind model with specified parameters
    print("Setting up wind model with specified parameters...")
    model = WindModel(
        # Galaxy properties
        v_circ=150.0,           # km/s, circular velocity (Milky Way-like)
        redshift=0.0,           
        
        # Wind launch properties - as specified
        SFR=20.0,               # Msun/yr
        eta_M=0.1,              # hot phase mass loading
        eta_M_cold=0.1,         # cold phase mass loading  
        eta_E=1.0,              # energy loading
        
        # Sonic point - 300 pc as specified
        r_star_kpc=0.3,         # kpc (= 300 pc)
        
        # Cloud properties - 5 masses from 10 to 1e5 Msun as specified
        cloud_mass_range=(10, 1e5),    # Msun
        cloud_alpha=2.0,               # power law slope
        N_cloud_species=5,             # 5 cloud masses as specified
        T_cl=1e4,                      # K, cloud temperature
        
        # Integration settings
        r_max_kpc=100.0,        # kpc, maximum radius
        rtol=1e-6,              # relaxed tolerance for efficiency
        atol=1e-8,
        
        # Progress reporting
        progress_callback='print',
        progress_interval=10.0,
    )
    
    # Print model summary
    print(f"\nModel parameters:")
    print(f"  SFR = {model.SFR} Msun/yr")
    print(f"  eta_M = {model.eta_M}")
    print(f"  eta_M_cold = {model.eta_M_cold}")
    print(f"  eta_E = {model.eta_E}")
    print(f"  r_sonic = {model.r_star_kpc} kpc = {model.r_star_kpc * 1000} pc")
    print(f"  N_cloud_species = {model.N_cloud_species}")
    print(f"  Cloud masses: {model.M_cloud0 / 2e33} Msun")  # Convert from g to Msun
    
    # Run the model
    print("\nRunning wind integration...")
    solution = model.run()
    
    # Print key results
    print(f"\nIntegration complete!")
    print(f"Maximum radius reached: {solution.r[-1]:.1f} kpc")
    print(f"Final status: {solution.sol.status} - {solution.sol.message}")
    
    # Extract results at 10 kpc if available
    if solution.r[-1] >= 10:
        print(f"\nResults at 10 kpc:")
        print(f"  Wind velocity: {solution.v_at_10kpc:.1f} km/s")
        print(f"  Mass loading: {solution.mass_loading_at_10kpc:.3f}")
        
        # Find index closest to 10 kpc
        idx_10kpc = np.argmin(np.abs(solution.r - 10.0))
        print(f"  Wind density: {solution.n[idx_10kpc]:.2e} cm^-3")
        print(f"  Wind temperature: {solution.T[idx_10kpc]:.2e} K")
        print(f"  Total cloud mass: {solution.M_cloud_tot[idx_10kpc]:.2e} Msun")
    
    # Create standard multi-panel plot
    print("\nCreating wind solution plot...")
    fig1, axes = plot_wind_solution(solution, show_hot_only=True, show_clouds=True)
    fig1.suptitle(f'Wind Solution: SFR={model.SFR}, $\\eta_M$={model.eta_M}, $\\eta_{{M,cold}}$={model.eta_M_cold}')
    plot_path = os.path.join(plots_dir, 'wind_solution_comprehensive.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {plot_path}")
    
    # Calculate velocity distribution
    print("\nCalculating velocity distribution...")
    v_cloud, dN_dv = solution.calculate_velocity_distribution(
        r_min_kpc=0.5,
        r_max_kpc=50.0,
        velocity_units='km/s'
    )
    
    # Calculate moments
    moments = solution.calculate_velocity_moments(r_min_kpc=0.5, r_max_kpc=50.0)
    print(f"\nVelocity distribution statistics:")
    if 'mean' in moments:
        print(f"  Mean velocity: {moments['mean']:.1f} km/s")
        print(f"  Velocity dispersion: {moments['dispersion']:.1f} km/s")
    else:
        print("  Warning: Could not calculate velocity moments")
        print(f"  Velocity range: {v_cloud.min():.1f} - {v_cloud.max():.1f} km/s")
        print(f"  dN/dv range: {dN_dv.min():.2e} - {dN_dv.max():.2e}")
    
    # Plot velocity distribution
    fig2, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(v_cloud, dN_dv, 'k-', lw=1.5)
    ax.set_xlabel(r'$v$ [km s$^{-1}$]')
    ax.set_ylabel(r'$dN/dv$ [(km s$^{-1}$)$^{-1}$]')
    ax.set_xlim(0, 800)
    ax.set_ylim(0, None)
    
    # Add vertical lines for mean and dispersion if available
    if 'mean' in moments:
        ax.axvline(moments['mean'], color='red', ls='--', alpha=0.7, label=f"Mean = {moments['mean']:.0f} km/s")
        if moments.get('dispersion', 0) > 0:
            ax.axvline(moments['mean'] - moments['dispersion'], color='blue', ls=':', alpha=0.5)
            ax.axvline(moments['mean'] + moments['dispersion'], color='blue', ls=':', alpha=0.5, 
                      label=f"$\\sigma$ = {moments['dispersion']:.0f} km/s")
    
    ax.legend(frameon=False)
    ax.set_title(f'Velocity Distribution (r = 0.5-50 kpc)')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'velocity_distribution_comprehensive.pdf')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig2)
    print(f"Saved: {plot_path}")
    
    # Calculate and plot column density distribution (dN/dv in observational units)
    print("\nCalculating column density distribution...")
    fig3, ax = plot_column_density_distribution(
        solution, 
        r_min_kpc=0.5,
        r_max_kpc=50.0,
        show_species=True,  # Show individual cloud contributions
        species_alpha=0.6,
        figsize=(6, 4.5),
        xlim=(0, 800),
        log_scale=True
    )
    ax.set_title('Column Density Distribution by Cloud Mass')
    plot_path = os.path.join(plots_dir, 'column_density_comprehensive.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {plot_path}")
    
    # Parameter study: vary eta_M_cold
    print("\n\nPerforming parameter study: varying eta_M_cold...")
    eta_M_cold_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Run models with different eta_M_cold
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
            progress_callback=None  # Suppress progress for parameter study
        )
        sol_var = model_var.run()
        
        # Plot velocity profile
        ax1.loglog(sol_var.r, sol_var.v, label=f'$\\eta_{{M,cold}} = {eta_M_cold}$')
        
        # Calculate and plot velocity distribution
        v_cl, dN_dv = sol_var.calculate_velocity_distribution(r_min_kpc=0.5, r_max_kpc=50.0)
        ax2.plot(v_cl, dN_dv/np.max(dN_dv), label=f'$\\eta_{{M,cold}} = {eta_M_cold}$')
    
    # Format velocity profile plot
    ax1.set_xlabel(r'$r$ [kpc]')
    ax1.set_ylabel(r'$v$ [km/s]')
    ax1.set_xlim(0.3, 100)
    ax1.set_ylim(50, 1000)
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title('Wind Velocity Profiles')
    
    # Format velocity distribution plot
    ax2.set_xlabel(r'$v$ [km/s]')
    ax2.set_ylabel(r'$dN/dv$ (normalized)')
    ax2.set_xlim(0, 800)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title('Velocity Distributions')
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'parameter_study_eta_M_cold.pdf')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig4)
    print(f"\nSaved: {plot_path}")
    
    print("\n✅ All plots generated successfully!")
    print(f"\nGenerated files in {plots_dir}/:")
    print("  - wind_solution_comprehensive.pdf")
    print("  - velocity_distribution_comprehensive.pdf") 
    print("  - column_density_comprehensive.pdf")
    print("  - parameter_study_eta_M_cold.pdf")
    

if __name__ == "__main__":
    main()