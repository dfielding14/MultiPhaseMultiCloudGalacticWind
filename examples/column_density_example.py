#!/usr/bin/env python
"""
Example showing how to calculate and plot column density distributions.

Column density distributions (dN/dv) are essential for comparing model
predictions with absorption line observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, plot_column_density_distribution


def main():
    # Create a wind model
    print("Creating wind model...")
    model = WindModel(
        SFR=20.0,               # Msun/yr
        eta_M=0.1,              # hot phase mass loading
        eta_M_cold=1.0,         # cold phase mass loading
        v_circ=150.0,           # km/s
        N_cloud_species=10,
    )
    
    # Run the simulation
    print("Running simulation...")
    solution = model.run()
    
    # Plot column density distribution
    print("\nPlotting column density distribution...")
    fig1, ax1 = plot_column_density_distribution(solution)
    ax1.set_title('Column Density Distribution')
    plt.savefig('column_density_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate column density distribution manually
    v_cloud, dN_dv_column = solution.calculate_column_density_distribution()
    
    print(f"\nColumn density distribution properties:")
    print(f"  Velocity range: {v_cloud.min():.1f} - {v_cloud.max():.1f} km/s")
    print(f"  Max dN/dv: {dN_dv_column.max():.2e} cm^-2 / (km/s)")
    print(f"  Integrated column density: {np.trapz(dN_dv_column, v_cloud):.2e} cm^-2")
    
    # Compare linear vs log scale
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    # Linear scale
    ax2.plot(v_cloud, dN_dv_column, 'k-', lw=1.5)
    ax2.set_xlabel(r'$v$ [km s$^{-1}$]')
    ax2.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]')
    ax2.set_title('Linear Scale')
    ax2.set_xlim(0, 800)
    
    # Log scale
    ax3.plot(v_cloud, dN_dv_column, 'k-', lw=1.5)
    ax3.set_xlabel(r'$v$ [km s$^{-1}$]')
    ax3.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]')
    ax3.set_yscale('log')
    ax3.set_title('Log Scale')
    ax3.set_xlim(0, 800)
    ax3.set_ylim(1e10, 1e16)
    
    plt.savefig('column_density_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Parameter study: Effect of mass loading
    print("\n\nParameter study: Effect of eta_M_cold on column density")
    eta_M_cold_values = [0.5, 1.0, 2.0, 5.0]
    
    fig3, ax4 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    
    for eta_cold in eta_M_cold_values:
        model_var = WindModel(SFR=20.0, eta_M=0.1, eta_M_cold=eta_cold)
        sol_var = model_var.run()
        
        # Use plot function with custom label
        v_cloud, dN_dv = sol_var.calculate_column_density_distribution()
        ax4.plot(v_cloud, dN_dv, label=f'$\\eta_{{M,cold}} = {eta_cold}$')
    
    ax4.set_xlabel(r'$v$ [km s$^{-1}$]')
    ax4.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]')
    ax4.set_xlim(0, 800)
    ax4.legend(frameon=False)
    ax4.set_title('Column Density vs Mass Loading')
    
    plt.savefig('column_density_parameter_study.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show individual cloud species
    print("\n\nAnalyzing individual cloud species...")
    fig4, ax5 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    
    # Plot total
    v_total, dN_dv_total = solution.calculate_column_density_distribution()
    ax5.plot(v_total, dN_dv_total, 'k-', lw=2, label='Total')
    
    # Plot some individual species
    for i in [0, 4, 9]:  # Low, medium, high mass clouds
        v_i, dN_dv_i = solution.calculate_column_density_distribution(cloud_index=i)
        M_cloud = model.M_cloud0[i]
        ax5.plot(v_i, dN_dv_i, '--', alpha=0.7, 
                label=f'$M_{{cl}} = 10^{{{np.log10(M_cloud):.1f}}} M_\\odot$')
    
    ax5.set_xlabel(r'$v$ [km s$^{-1}$]')
    ax5.set_ylabel(r'$dN/dv$ [cm$^{-2}$ (km s$^{-1}$)$^{-1}$]')
    ax5.set_xlim(0, 800)
    ax5.legend(frameon=False)
    ax5.set_title('Column Density by Cloud Mass')
    
    plt.savefig('column_density_by_mass.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExample complete! Check the generated PDF files.")


if __name__ == "__main__":
    main()