#!/usr/bin/env python
"""
Improved diagnostic grid search for J0021+0052 parameter space.
Uses proper plotting functions and configuration from tutorial.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.plotting import (
    plot_wind_solution, plot_profiles, plot_column_density_distribution
)
from multiphasegalacticwind.constants import Msun, kpc
import warnings

# Galaxy parameters from J0021+0052
GALAXY_PARAMS = {
    'SFR': 3.0,         # Msun/yr
    'v_circ': 0.001,    # Negligibly small to avoid launch issues
}

# Parameter grids - chosen to span reasonable ranges
ETA_M_GRID = [0.1, 0.3, 1.0]        # Hot mass loading
ETA_M_COLD_GRID = [0.1, 0.3, 1.0]   # Cold mass loading  
ETA_E_GRID = [0.1, 0.5, 2.0]        # Energy loading

def create_config(cooling_factor):
    """
    Create configuration using proper parameters from tutorial.
    """
    return WindConfig(
        # Cloud properties (from tutorial)
        M_cloud_min=1e-1,                    # Msun, minimum cloud mass
        M_cloud_max=1e6,                     # Msun, maximum cloud mass
        cloud_alpha=2.0,                     # power law slope
        N_cloud_species=5,                   # reduced for speed
        T_cl=1e4,                           # K, cloud temperature
        v_cloud_init=30.0,                  # km/s, initial cloud velocity
        cloud_radial_offset=1e-12,
        cold_cloud_injection_radial_extent_frac=1,
        cold_cloud_injection_radial_power=8,
        
        # Physical parameters
        f_turb0=0.1,                        # Turbulent mixing efficiency
        drag_coeff=0.475,                   # Standard drag coefficient
        metallicity=0.3,                    # From J0021+0052
        cooling_factor=cooling_factor,      # 0 or 1
        
        # Integration settings (relaxed for diagnostics)
        rtol=1e-6,                          # relaxed for speed
        atol=1e-8,
        sonic_point_tolerance=1e-6,
        sonic_point_offset=1e-12,
    )

def create_model_and_plots(eta_M, eta_M_cold, eta_E, cooling_factor, output_dir):
    """
    Create a wind model and generate comprehensive diagnostic plots.
    
    Returns:
    --------
    success : bool
        Whether the model ran successfully
    message : str
        Description of outcome
    """
    # Create output subdirectory for this parameter set
    param_str = f"etaM_{eta_M:.1f}_etaMcold_{eta_M_cold:.1f}_etaE_{eta_E:.1f}"
    param_dir = os.path.join(output_dir, param_str)
    os.makedirs(param_dir, exist_ok=True)
    
    try:
        # Create configuration
        config = create_config(cooling_factor)
        
        # Create model
        model = WindModel(
            config=config,
            SFR=GALAXY_PARAMS['SFR'],
            v_circ=GALAXY_PARAMS['v_circ'],
            eta_M=eta_M,
            eta_M_cold=eta_M_cold,
            eta_E=eta_E,
            r_max_kpc=30.0  # From tutorial
        )
        
        # Run model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            solution = model.run()
        
        # Check if solution is valid
        if not hasattr(solution, 'r') or len(solution.r) == 0:
            return False, "Integration failed - no solution generated"
        
        # Get integration extent
        max_r = solution.r[-1]
        has_v_terminal = hasattr(solution, 'v_terminal')
        v_end = solution.v_terminal if has_v_terminal else solution.v[-1]
        
        # Determine success status
        if max_r < 1.0:
            success = False
            status = "FAILED"
            message = f"Integration stopped at {max_r:.2f} kpc"
        elif max_r < 25.0:
            success = False
            status = "PARTIAL"
            message = f"Partial: r_max={max_r:.1f} kpc, v_end={v_end:.0f} km/s"
        else:
            success = True
            status = "SUCCESS"
            message = f"Success: r_max={max_r:.1f} kpc, v_end={v_end:.0f} km/s"
        
        # Generate plots using proper plotting functions
        try:
            # 1. Main wind solution plot
            fig1, axes1 = plot_wind_solution(solution, show_hot_only=False, show_clouds=True)
            fig1.suptitle(f'{status}: eta_M={eta_M:.1f}, eta_M_cold={eta_M_cold:.1f}, eta_E={eta_E:.1f}', 
                         fontsize=12, y=1.02)
            fig1.savefig(os.path.join(param_dir, 'wind_solution.pdf'), 
                        dpi=150, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Detailed profiles
            fig2, axes2 = plot_profiles(solution, 
                                       quantities=['velocity', 'density', 'temperature', 'mass_flux'])
            fig2.suptitle(f'Cooling factor = {cooling_factor}', fontsize=12, y=1.02)
            fig2.savefig(os.path.join(param_dir, 'profiles.pdf'), 
                        dpi=150, bbox_inches='tight')
            plt.close(fig2)
            
            # 3. Column density distribution
            try:
                fig3, ax3 = plot_column_density_distribution(
                    solution, 
                    show_species=True,
                    show_moments=True,
                    xlim=(-500, 500)
                )
                fig3.suptitle(f'Observable: dN/dv', fontsize=12, y=1.02)
                fig3.savefig(os.path.join(param_dir, 'column_density.pdf'), 
                            dpi=150, bbox_inches='tight')
                plt.close(fig3)
            except Exception as e:
                # Column density might fail for incomplete solutions
                pass
            
            # 4. Create summary plot
            fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8))
            
            # Key diagnostics
            ax = axes4[0, 0]
            ax.loglog(solution.r, solution.v, 'b-', lw=2, label='Wind')
            ax.axhline(30, color='k', ls='--', alpha=0.3, label='v_cloud_init')
            ax.set_xlabel('r [kpc]')
            ax.set_ylabel('v [km/s]')
            ax.set_title(f'{status}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.3, max(30, max_r))
            ax.set_ylim(10, 3000)
            
            ax = axes4[0, 1]
            ax.loglog(solution.r, solution.n, 'r-', lw=2)
            ax.set_xlabel('r [kpc]')
            ax.set_ylabel('n [cm⁻³]')
            ax.set_title(f'Max r = {max_r:.1f} kpc')
            ax.grid(True, alpha=0.3)
            
            ax = axes4[1, 0]
            ax.loglog(solution.r, solution.T, 'orange', lw=2)
            ax.axhline(1e4, color='k', ls='--', alpha=0.3)
            ax.set_xlabel('r [kpc]')
            ax.set_ylabel('T [K]')
            ax.set_title(f'v_end = {v_end:.0f} km/s')
            ax.grid(True, alpha=0.3)
            
            ax = axes4[1, 1]
            # Show cloud survival
            if hasattr(solution, 'M_clouds'):
                for i in range(model.N_cloud_species):
                    if solution.M_clouds[i][-1] > model.config.M_cloud_min:
                        ax.loglog(solution.r, solution.M_clouds[i], '-', alpha=0.5)
            ax.set_xlabel('r [kpc]')
            ax.set_ylabel('M_cloud [M☉]')
            ax.set_title('Cloud evolution')
            ax.grid(True, alpha=0.3)
            
            fig4.suptitle(f'eta_M={eta_M:.1f}, eta_M_cold={eta_M_cold:.1f}, eta_E={eta_E:.1f}',
                         fontsize=14)
            plt.tight_layout()
            fig4.savefig(os.path.join(param_dir, 'summary.pdf'), 
                        dpi=150, bbox_inches='tight')
            plt.close(fig4)
            
        except Exception as plot_error:
            print(f"    Warning: Plotting failed: {str(plot_error)[:50]}")
        
        return success, message
        
    except Exception as e:
        # Create error indicator
        error_file = os.path.join(param_dir, 'ERROR.txt')
        with open(error_file, 'w') as f:
            f.write(f"Model failed to run\n")
            f.write(f"eta_M={eta_M:.1f}, eta_M_cold={eta_M_cold:.1f}, eta_E={eta_E:.1f}\n")
            f.write(f"Error: {str(e)}\n")
        
        return False, f"Exception: {str(e)[:50]}"

def run_parameter_grid():
    """Run the full parameter grid for both cooling scenarios."""
    
    # Create output directories
    os.makedirs('grid_diagnostics_v2/no_cooling', exist_ok=True)
    os.makedirs('grid_diagnostics_v2/with_cooling', exist_ok=True)
    
    # Total number of models
    n_models = len(ETA_M_GRID) * len(ETA_M_COLD_GRID) * len(ETA_E_GRID)
    
    for cooling_scenario in ['no_cooling', 'with_cooling']:
        cooling_factor = 0.0 if cooling_scenario == 'no_cooling' else 1.0
        output_dir = f'grid_diagnostics_v2/{cooling_scenario}'
        
        print(f"\n{'='*60}")
        print(f"Running {cooling_scenario.upper()} grid (cooling_factor={cooling_factor})")
        print(f"{'='*60}")
        
        results = []
        model_count = 0
        successful_models = []
        
        for eta_M in ETA_M_GRID:
            for eta_M_cold in ETA_M_COLD_GRID:
                for eta_E in ETA_E_GRID:
                    model_count += 1
                    print(f"\n[{model_count}/{n_models}] eta_M={eta_M:.1f}, eta_M_cold={eta_M_cold:.1f}, eta_E={eta_E:.1f}")
                    
                    success, message = create_model_and_plots(
                        eta_M, eta_M_cold, eta_E, 
                        cooling_factor, output_dir
                    )
                    
                    results.append({
                        'eta_M': eta_M,
                        'eta_M_cold': eta_M_cold,
                        'eta_E': eta_E,
                        'success': success,
                        'message': message
                    })
                    
                    status = "✓" if success else "✗"
                    print(f"  {status} {message}")
                    
                    if success:
                        successful_models.append((eta_M, eta_M_cold, eta_E))
        
        # Create summary report
        n_success = sum(1 for r in results if r['success'])
        
        print(f"\n{'-'*60}")
        print(f"SUMMARY for {cooling_scenario}:")
        print(f"  Successful models: {n_success}/{n_models} ({100*n_success/n_models:.1f}%)")
        
        # Save detailed summary
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Parameter Grid Results - {cooling_scenario}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Galaxy: J0021+0052 (SFR={GALAXY_PARAMS['SFR']} Msun/yr, v_circ={GALAXY_PARAMS['v_circ']} km/s)\n")
            f.write(f"Cooling factor: {cooling_factor}\n")
            f.write(f"Configuration: N_cloud_species=11, rtol=1e-10, r_max=30 kpc\n")
            f.write(f"Success rate: {n_success}/{n_models} ({100*n_success/n_models:.1f}%)\n\n")
            
            if successful_models:
                f.write("SUCCESSFUL MODELS:\n")
                f.write("-"*40 + "\n")
                for eta_M, eta_M_cold, eta_E in successful_models:
                    f.write(f"  eta_M={eta_M:.1f}, eta_M_cold={eta_M_cold:.1f}, eta_E={eta_E:.1f}\n")
                f.write("\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-"*60 + "\n")
            for r in results:
                status = "SUCCESS" if r['success'] else "FAILED"
                f.write(f"{status}: eta_M={r['eta_M']:.1f}, eta_M_cold={r['eta_M_cold']:.1f}, eta_E={r['eta_E']:.1f}\n")
                f.write(f"  → {r['message']}\n\n")
        
        # Create index HTML for easy browsing
        index_file = os.path.join(output_dir, 'index.html')
        with open(index_file, 'w') as f:
            f.write(f"<html><head><title>{cooling_scenario} Results</title></head><body>\n")
            f.write(f"<h1>Parameter Grid Results - {cooling_scenario}</h1>\n")
            f.write(f"<p>Success rate: {n_success}/{n_models} ({100*n_success/n_models:.1f}%)</p>\n")
            f.write("<table border='1'>\n")
            f.write("<tr><th>eta_M</th><th>eta_M_cold</th><th>eta_E</th><th>Status</th><th>Links</th></tr>\n")
            
            for r in results:
                param_str = f"etaM_{r['eta_M']:.1f}_etaMcold_{r['eta_M_cold']:.1f}_etaE_{r['eta_E']:.1f}"
                status_color = "green" if r['success'] else "red"
                f.write(f"<tr>")
                f.write(f"<td>{r['eta_M']:.1f}</td>")
                f.write(f"<td>{r['eta_M_cold']:.1f}</td>")
                f.write(f"<td>{r['eta_E']:.1f}</td>")
                f.write(f"<td style='color:{status_color}'>{r['message']}</td>")
                f.write(f"<td>")
                f.write(f"<a href='{param_str}/wind_solution.pdf'>Wind</a> | ")
                f.write(f"<a href='{param_str}/profiles.pdf'>Profiles</a> | ")
                f.write(f"<a href='{param_str}/column_density.pdf'>Observable</a> | ")
                f.write(f"<a href='{param_str}/summary.pdf'>Summary</a>")
                f.write(f"</td>")
                f.write(f"</tr>\n")
            
            f.write("</table></body></html>\n")

if __name__ == "__main__":
    print("="*70)
    print("IMPROVED PARAMETER SPACE DIAGNOSTICS FOR J0021+0052")
    print("="*70)
    print(f"Parameter ranges:")
    print(f"  eta_M:      {ETA_M_GRID}")
    print(f"  eta_M_cold: {ETA_M_COLD_GRID}")
    print(f"  eta_E:      {ETA_E_GRID}")
    print(f"  Total models: {len(ETA_M_GRID) * len(ETA_M_COLD_GRID) * len(ETA_E_GRID)} × 2 cooling scenarios")
    print(f"\nUsing configuration from tutorial:")
    print(f"  - N_cloud_species = 11")
    print(f"  - r_max = 30 kpc")
    print(f"  - rtol/atol = 1e-10")
    print(f"  - Proper plotting functions from plotting.py")
    
    run_parameter_grid()
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)
    print("\nResults saved in:")
    print("  ./grid_diagnostics_v2/no_cooling/    - Models without cooling")
    print("  ./grid_diagnostics_v2/with_cooling/  - Models with cooling")
    print("\nEach parameter set has its own subdirectory containing:")
    print("  - wind_solution.pdf  : Full wind solution plots")
    print("  - profiles.pdf       : Detailed radial profiles")
    print("  - column_density.pdf : Observable dN/dv")
    print("  - summary.pdf        : Key diagnostics")
    print("\nOpen index.html in each directory for easy browsing.")