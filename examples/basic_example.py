"""
Basic example of using the multiphase galactic wind model.

This example shows how to:
1. Initialize a wind model
2. Run the simulation
3. Access the results
4. Create plots
"""

import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, setup_plotting_style, plot_wind_solution


def main():
    # Set up plotting style
    setup_plotting_style()
    
    # Create a wind model with default Milky Way-like parameters
    model = WindModel(
        # Galaxy properties
        v_circ=150.0,           # km/s, circular velocity
        redshift=0.0,           # redshift
        
        # Wind properties
        SFR=20.0,               # Msun/yr, star formation rate
        eta_M=0.1,              # hot phase mass loading
        eta_M_cold=1.0,         # cold phase mass loading
        eta_E=1.0,              # energy loading
        
        # Initial conditions
        r_star_kpc=0.3,         # kpc, sonic radius
        n_star=0.1,             # cm^-3, hot phase density
        v_star=200.0,           # km/s, initial velocity
        T_star=5e6,             # K, hot phase temperature
        Z_star=1.0,             # solar metallicity
        
        # Cloud properties
        cloud_mass_range=(1, 1e5),    # Msun
        cloud_alpha=2.0,              # power law slope
        N_cloud_species=10,           # number of mass bins
        T_cl=1e4,                     # K, cloud temperature
    )
    
    # Run the model
    print("Running wind model...")
    solution = model.run()
    
    # Print some key results
    print(f"\nResults:")
    print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
    print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")
    print(f"Maximum radius reached: {solution.r[-1]:.1f} kpc")
    
    # Create the standard multi-panel plot
    fig, axes = plot_wind_solution(solution)
    plt.savefig('wind_solution.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example of accessing raw data
    # Find properties at r = 1 kpc
    r_target = 1.0  # kpc
    idx = np.argmin(np.abs(solution.r - r_target))
    
    print(f"\nProperties at r = {r_target} kpc:")
    print(f"  Velocity: {solution.v[idx]:.1f} km/s")
    print(f"  Density: {solution.n[idx]:.2e} cm^-3")
    print(f"  Temperature: {solution.T[idx]:.2e} K")
    print(f"  Total cloud mass: {solution.M_cloud_tot[idx]:.2e} Msun")
    
    # Example of parameter study
    print("\n\nParameter study: varying eta_M")
    eta_M_values = [0.05, 0.1, 0.2, 0.5]
    
    plt.figure(figsize=(5, 4))
    for eta_M in eta_M_values:
        model_variant = WindModel(SFR=20.0, eta_M=eta_M, eta_M_cold=1.0)
        sol_variant = model_variant.run()
        plt.loglog(sol_variant.r, sol_variant.v, label=f'$\\eta_M = {eta_M}$')
    
    plt.xlabel(r'$r$ [kpc]')
    plt.ylabel(r'$v$ [km/s]')
    plt.legend(frameon=False)
    plt.xlim(0.3, 30)
    plt.ylim(50, 1000)
    plt.tight_layout()
    plt.savefig('parameter_study.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()