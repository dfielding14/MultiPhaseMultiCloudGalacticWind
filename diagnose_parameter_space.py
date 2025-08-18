#!/usr/bin/env python
"""
Diagnostic grid search to understand parameter space viability for J0021+0052.
Creates wind profile plots for a 3x3x3 grid of (eta_M, eta_M_cold, eta_E).
Runs twice: once without cooling, once with cooling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.constants import kpc
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

def create_model_and_plot(eta_M, eta_M_cold, eta_E, cooling_factor, output_dir):
    """
    Create a wind model and generate diagnostic plots.
    
    Returns:
    --------
    success : bool
        Whether the model ran successfully
    message : str
        Description of outcome
    """
    # Create output filename
    filename = f"etaM_{eta_M:.1f}_etaMcold_{eta_M_cold:.1f}_etaE_{eta_E:.1f}.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # Create configuration with specified cooling
        config = WindConfig(
            N_cloud_species=5,        # Fewer for speed
            M_cloud_min=10.0,         
            M_cloud_max=1e6,          
            cloud_alpha=2.0,          
            f_turb0=0.1,              
            drag_coeff=0.475,         
            T_cl=1e4,                 
            metallicity=0.3,          # From J0021+0052
            rtol=1e-6,                
            atol=1e-8,
            sonic_point_tolerance=0.01,
            cooling_factor=cooling_factor  # KEY PARAMETER
        )
        
        # Create model
        model = WindModel(
            config=config,
            SFR=GALAXY_PARAMS['SFR'],
            v_circ=GALAXY_PARAMS['v_circ'],
            eta_M=eta_M,
            eta_M_cold=eta_M_cold,
            eta_E=eta_E,
            r_max_kpc=50  # Shorter for diagnostic purposes
        )
        
        # Run model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
            solution = model.run()
        
        # Check if solution is valid
        has_v_terminal = hasattr(solution, 'v_terminal')
        if not hasattr(solution, 'r') or len(solution.r) == 0:
            return False, "Integration failed - no solution generated"
        
        # Get max radius and terminal velocity
        max_r = solution.r[-1]
        v_term = solution.v_terminal if has_v_terminal else solution.v[-1]
        
        # Mark as partial success if integration stopped early
        if max_r < 1.0:  # Less than 1 kpc
            integration_status = "FAILED"
            success = False
            message = f"Integration stopped at {max_r:.2f} kpc"
        elif not has_v_terminal:
            integration_status = "PARTIAL"
            success = False
            message = f"Partial: r_max={max_r:.1f} kpc, v_end={v_term:.0f} km/s"
        elif v_term <= 0:
            integration_status = "FAILED"
            success = False
            message = f"Negative terminal velocity: {v_term:.1f} km/s"
        elif np.any(np.isnan(solution.v)):
            integration_status = "FAILED"
            success = False
            message = "NaN values in velocity profile"
        else:
            integration_status = "SUCCESS"
            success = True
            message = f"Success: v_term={v_term:.0f} km/s, r_max={max_r:.1f} kpc"
        
        # Create diagnostic plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Panel 1: Velocity
        ax = axes[0, 0]
        ax.loglog(solution.r, solution.v, 'b-', lw=2, label='Wind')
        if hasattr(solution, 'v_cl'):
            for i in range(model.N_cloud_species):
                if solution.M_clouds[i][-1] > model.config.M_cloud_min:
                    ax.loglog(solution.r, solution.v_cl[i], '--', alpha=0.5)
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('v [km/s]')
        ax.set_title(f'η_M={eta_M:.1f}, η_M,cold={eta_M_cold:.1f}, η_E={eta_E:.1f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(10, 2000)
        
        # Panel 2: Density
        ax = axes[0, 1]
        ax.loglog(solution.r, solution.n, 'r-', lw=2)
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('n [cm⁻³]')
        ax.set_title(f'Cooling factor = {cooling_factor}')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Temperature
        ax = axes[1, 0]
        ax.loglog(solution.r, solution.T, 'orange', lw=2)
        ax.axhline(1e4, color='k', ls='--', alpha=0.5, label='T_cloud')
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('T [K]')
        ax.set_title(f'{integration_status}: v_end = {v_term:.0f} km/s')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Panel 4: Mass flux
        ax = axes[1, 1]
        ax.loglog(solution.r, solution.Mdot/GALAXY_PARAMS['SFR'], 'g-', lw=2)
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('Ṁ/SFR')
        ax.set_title(f'Max r = {max_r:.1f} kpc')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.01, 100)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return success, message
        
    except Exception as e:
        # Create error plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"MODEL FAILED\n\nη_M = {eta_M:.1f}\nη_M_cold = {eta_M_cold:.1f}\nη_E = {eta_E:.1f}\n\nError: {str(e)[:100]}",
                ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return False, f"Exception: {str(e)[:50]}"

def run_parameter_grid():
    """Run the full parameter grid for both cooling scenarios."""
    
    # Create output directories
    os.makedirs('grid_diagnostics/no_cooling', exist_ok=True)
    os.makedirs('grid_diagnostics/with_cooling', exist_ok=True)
    
    # Total number of models
    n_models = len(ETA_M_GRID) * len(ETA_M_COLD_GRID) * len(ETA_E_GRID)
    
    for cooling_scenario in ['no_cooling', 'with_cooling']:
        cooling_factor = 0.0 if cooling_scenario == 'no_cooling' else 1.0
        output_dir = f'grid_diagnostics/{cooling_scenario}'
        
        print(f"\n{'='*60}")
        print(f"Running {cooling_scenario.upper()} grid (cooling_factor={cooling_factor})")
        print(f"{'='*60}")
        
        results = []
        model_count = 0
        
        for eta_M in ETA_M_GRID:
            for eta_M_cold in ETA_M_COLD_GRID:
                for eta_E in ETA_E_GRID:
                    model_count += 1
                    print(f"\n[{model_count}/{n_models}] η_M={eta_M:.1f}, η_M_cold={eta_M_cold:.1f}, η_E={eta_E:.1f}")
                    
                    success, message = create_model_and_plot(
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
        
        # Summary statistics
        n_success = sum(1 for r in results if r['success'])
        print(f"\n{'-'*60}")
        print(f"SUMMARY for {cooling_scenario}:")
        print(f"  Successful models: {n_success}/{n_models} ({100*n_success/n_models:.1f}%)")
        
        if n_success > 0:
            print(f"\n  Successful parameter combinations:")
            for r in results:
                if r['success']:
                    print(f"    η_M={r['eta_M']:.1f}, η_M_cold={r['eta_M_cold']:.1f}, η_E={r['eta_E']:.1f}")
        
        # Save results summary
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Parameter Grid Results - {cooling_scenario}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Galaxy: J0021+0052 (SFR={GALAXY_PARAMS['SFR']} Msun/yr, v_circ={GALAXY_PARAMS['v_circ']} km/s)\n")
            f.write(f"Cooling factor: {cooling_factor}\n")
            f.write(f"Success rate: {n_success}/{n_models} ({100*n_success/n_models:.1f}%)\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-"*60 + "\n")
            for r in results:
                status = "SUCCESS" if r['success'] else "FAILED"
                f.write(f"{status}: η_M={r['eta_M']:.1f}, η_M_cold={r['eta_M_cold']:.1f}, η_E={r['eta_E']:.1f}\n")
                f.write(f"  → {r['message']}\n\n")

if __name__ == "__main__":
    print("Starting parameter space diagnostics for J0021+0052")
    print(f"Parameter ranges:")
    print(f"  η_M:      {ETA_M_GRID}")
    print(f"  η_M_cold: {ETA_M_COLD_GRID}")
    print(f"  η_E:      {ETA_E_GRID}")
    print(f"  Total models: {len(ETA_M_GRID) * len(ETA_M_COLD_GRID) * len(ETA_E_GRID)} × 2 cooling scenarios")
    
    run_parameter_grid()
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)
    print("\nResults saved in:")
    print("  ./grid_diagnostics/no_cooling/    - Models without cooling")
    print("  ./grid_diagnostics/with_cooling/  - Models with cooling")
    print("\nEach directory contains:")
    print("  - Individual plots for each parameter combination")
    print("  - summary.txt with detailed results")