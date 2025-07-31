"""
Example showing how to calculate velocity distributions and moments
for comparison with observations.

This demonstrates:
1. Calculating dN/dv velocity distributions
2. Computing velocity moments (mean, dispersion)
3. Exploring parameter dependencies
4. Creating plots suitable for observational comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import (WindModel, setup_plotting_style, 
                                   plot_velocity_distribution,
                                   calculate_mass_weighted_velocity)


def main():
    # Set up plotting style
    setup_plotting_style()
    
    # Create a fiducial model
    print("Running fiducial wind model...")
    model = WindModel(
        SFR=20.0,               # Msun/yr
        eta_M=0.1,              # hot phase mass loading
        eta_M_cold=1.0,         # cold phase mass loading
        v_circ=150.0,           # km/s
        N_cloud_species=10,     # number of cloud mass bins
    )
    solution = model.run()
    
    # Calculate and plot velocity distribution
    print("\nCalculating velocity distribution...")
    fig, ax = plot_velocity_distribution(solution, 
                                       r_min_kpc=0.1,
                                       r_max_kpc=10.0,
                                       injection_radius_kpc=0.3,
                                       injection_power=6.0)
    plt.savefig('velocity_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Get velocity moments
    moments = solution.calculate_velocity_moments()
    print(f"\nVelocity distribution moments:")
    print(f"  Mean velocity: {moments['mean']:.1f} km/s")
    print(f"  Velocity dispersion: {moments['dispersion']:.1f} km/s")
    if 'skewness' in moments:
        print(f"  Skewness: {moments['skewness']:.2f}")
    
    # Calculate mass-weighted velocity for comparison
    r_eval = np.array([1, 5, 10])  # kpc
    v_mass = calculate_mass_weighted_velocity(solution, r_eval)
    print(f"\nMass-weighted velocities:")
    for r, v in zip(r_eval, v_mass):
        print(f"  At {r} kpc: {v:.1f} km/s")
    
    # Parameter study: Effect of mass loading on velocity distribution
    print("\n\nParameter study: Effect of eta_M_cold on velocity moments")
    eta_M_cold_values = [0.5, 1.0, 2.0, 5.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    mean_velocities = []
    dispersions = []
    
    for eta_cold in eta_M_cold_values:
        # Run model
        model_var = WindModel(SFR=20.0, eta_M=0.1, eta_M_cold=eta_cold)
        sol_var = model_var.run()
        
        # Calculate velocity distribution
        v_cloud, dN_dv = sol_var.calculate_velocity_distribution()
        
        # Get moments
        moments_var = sol_var.calculate_velocity_moments()
        mean_velocities.append(moments_var.get('mean', np.nan))
        dispersions.append(moments_var.get('dispersion', np.nan))
        
        # Plot distribution
        ax1.plot(v_cloud, dN_dv, label=f'$\\eta_{{M,cold}} = {eta_cold}$')
    
    ax1.set_xlabel(r'$v$ [km/s]')
    ax1.set_ylabel(r'$dN/dv$ [(km/s)$^{-1}$]')
    ax1.legend(frameon=False)
    ax1.set_xlim(0, 800)
    
    # Plot moments vs eta_M_cold
    ax2.plot(eta_M_cold_values, mean_velocities, 'ko-', label=r'$\langle v \rangle$')
    ax2.plot(eta_M_cold_values, dispersions, 'ks--', label=r'$\sigma_v$')
    ax2.set_xlabel(r'$\eta_{M,cold}$')
    ax2.set_ylabel(r'Velocity [km/s]')
    ax2.legend(frameon=False)
    ax2.set_xscale('log')
    
    plt.savefig('parameter_study_velocity_dist.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example: Single cloud species analysis
    print("\n\nAnalyzing individual cloud species:")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.flatten()
    
    # Look at 4 different cloud masses
    cloud_indices = [0, 3, 6, 9]  # Different mass bins
    
    for i, idx in enumerate(cloud_indices):
        ax = axes[i]
        
        # Get cloud mass for this species
        M_cloud_init = solution.model.M_cloud0[idx]
        
        # Calculate distribution for this species only
        v_cloud, dN_dv = solution.calculate_velocity_distribution(cloud_index=idx)
        
        # Plot
        ax.plot(v_cloud, dN_dv, 'k-', lw=1.5)
        ax.set_xlabel(r'$v$ [km/s]')
        ax.set_ylabel(r'$dN/dv$ [(km/s)$^{-1}$]')
        ax.set_title(f'$M_{{cl,0}} = 10^{{{np.log10(M_cloud_init):.1f}}} M_\\odot$')
        ax.set_xlim(0, 800)
    
    plt.savefig('velocity_dist_by_mass.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExample complete! Check the generated PDF files.")


if __name__ == "__main__":
    main()