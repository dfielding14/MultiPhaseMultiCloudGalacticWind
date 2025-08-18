#!/usr/bin/env python
"""
Simple direct comparison of Wind_Evo vs Hot_Wind_Evo ODEs.
Tests if full model reduces to hot-only when eta_M_cold → 0.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.core_physics import Wind_Evo, Hot_Wind_Evo
from multiphasegalacticwind.constants import kpc, Msun

def test_ode_convergence():
    """Test that Wind_Evo → Hot_Wind_Evo as eta_M_cold → 0"""
    
    print("="*60)
    print("ODE CONVERGENCE TEST")
    print("Testing: Wind_Evo → Hot_Wind_Evo as η_M_cold → 0")
    print("="*60)
    
    # Base parameters for J0021+0052
    base_params = {
        'SFR': 3.0,
        'v_circ': 0.001,
        'eta_M': 0.3,
        'eta_E': 0.5,
    }
    
    # Create hot-only model first
    config_hot = WindConfig(
        N_cloud_species=0,
        cooling_factor=0.0,  # Disable cooling for cleaner comparison
        rtol=1e-10,
        atol=1e-12
    )
    
    model_hot = WindModel(
        config=config_hot,
        eta_M_cold=0.0,
        **base_params
    )
    
    # Get initial conditions and parameters for hot-only
    r0 = model_hot.r_star_kpc * kpc  # Starting radius in CGS
    
    # Create parameter tuple for hot-only
    from multiphasegalacticwind.core_physics import setup_sonic_point_params
    
    # Get parameters for hot-only model
    params_hot = setup_sonic_point_params(
        v_circ=model_hot.v_circ * 1e5,  # Convert to cm/s
        N_cloud_species=0,
        T_cloud=model_hot.config.T_cl,
        injection_radius=model_hot.injection_radius,
        injection_power=model_hot.injection_power,
        config_dict=model_hot.config.to_dict(),
        r0=r0,
        Edot_per_Vol=model_hot.Edot_per_Vol,
        Mdot_per_Vol=model_hot.Mdot_per_Vol,
        Lambda_P_rho=None
    )
    
    # Hot-only initial state: [v_wind, rho_wind, P, rhoZ_wind]
    y0_hot = np.array([
        30.0 * 1e5,  # v_wind in cm/s
        model_hot.rho_star,  # rho in g/cm^3
        model_hot.P_star,  # P in dyne/cm^2
        model_hot.rho_star * model_hot.config.metallicity  # rhoZ
    ])
    
    # Evaluate hot-only ODE
    dydt_hot = Hot_Wind_Evo(r0, y0_hot, params_hot)
    
    print(f"\nHot-only ODE at r = {r0/kpc:.3f} kpc:")
    print(f"  dv/dr = {dydt_hot[0]:.2e} cm/s/cm")
    print(f"  drho/dr = {dydt_hot[1]:.2e} g/cm³/cm")
    print(f"  dP/dr = {dydt_hot[2]:.2e} dyne/cm²/cm")
    
    # Now test full model with decreasing eta_M_cold
    epsilons = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.3]
    
    results = []
    
    for eps in epsilons:
        print(f"\n{'='*40}")
        print(f"Testing η_M_cold = {eps:.1e}")
        
        # Create full model with small cold component
        config_full = WindConfig(
            N_cloud_species=5,
            M_cloud_min=0.1,
            M_cloud_max=1e6,
            cooling_factor=0.0,
            rtol=1e-10,
            atol=1e-12
        )
        
        model_full = WindModel(
            config=config_full,
            eta_M_cold=eps,
            **base_params
        )
        
        # Get parameters for full model
        params_full = setup_sonic_point_params(
            v_circ=model_full.v_circ * 1e5,
            N_cloud_species=model_full.N_cloud_species,
            T_cloud=model_full.config.T_cl,
            injection_radius=model_full.injection_radius,
            injection_power=model_full.injection_power,
            config_dict=model_full.config.to_dict(),
            r0=r0,
            Edot_per_Vol=model_full.Edot_per_Vol,
            Mdot_per_Vol=model_full.Mdot_per_Vol,
            Lambda_P_rho=None
        )
        
        # Add cloud distribution parameters
        params_full = params_full + (model_full.Ndot_cloud0,)
        
        # Full model initial state
        y0_full = [
            30.0 * 1e5,  # v_wind
            model_full.rho_star,  # rho_wind
            model_full.P_star,  # P
            model_full.rho_star * model_full.config.metallicity  # rhoZ_wind
        ]
        
        # Add cloud species
        for i in range(model_full.N_cloud_species):
            y0_full.append(model_full.M_cloud0[i])  # M_cloud in grams
            y0_full.append(model_full.config.v_cloud_init * 1e5)  # v_cloud in cm/s
            y0_full.append(model_full.config.metallicity)  # Z_cloud
        
        y0_full = np.array(y0_full)
        
        # Evaluate full ODE
        try:
            dydt_full = Wind_Evo(r0, y0_full, params_full)
            
            # Extract hot phase derivatives (first 4 components)
            dydt_full_hot = dydt_full[:4]
            
            # Compare to hot-only
            diff_v = abs(dydt_full_hot[0] - dydt_hot[0])
            diff_rho = abs(dydt_full_hot[1] - dydt_hot[1])
            diff_P = abs(dydt_full_hot[2] - dydt_hot[2])
            
            # Relative differences
            rel_v = diff_v / (abs(dydt_hot[0]) + 1e-20)
            rel_rho = diff_rho / (abs(dydt_hot[1]) + 1e-20)
            rel_P = diff_P / (abs(dydt_hot[2]) + 1e-20)
            
            print(f"  Full ODE hot phase:")
            print(f"    dv/dr = {dydt_full_hot[0]:.2e} cm/s/cm")
            print(f"    drho/dr = {dydt_full_hot[1]:.2e} g/cm³/cm")
            print(f"    dP/dr = {dydt_full_hot[2]:.2e} dyne/cm²/cm")
            
            print(f"  Differences from hot-only:")
            print(f"    Δ(dv/dr) = {diff_v:.2e} ({rel_v:.1%})")
            print(f"    Δ(drho/dr) = {diff_rho:.2e} ({rel_rho:.1%})")
            print(f"    Δ(dP/dr) = {diff_P:.2e} ({rel_P:.1%})")
            
            # Check convergence
            converged = (rel_v < 0.01) and (rel_rho < 0.01) and (rel_P < 0.01)
            print(f"  Converged to hot-only? {'YES' if converged else 'NO'}")
            
            results.append({
                'eta_M_cold': eps,
                'diff_v': diff_v,
                'diff_rho': diff_rho,
                'diff_P': diff_P,
                'rel_v': rel_v,
                'rel_rho': rel_rho,
                'rel_P': rel_P,
                'converged': converged,
                'dydt_v_full': dydt_full_hot[0],
                'dydt_rho_full': dydt_full_hot[1],
                'dydt_P_full': dydt_full_hot[2],
            })
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            results.append({
                'eta_M_cold': eps,
                'error': str(e)[:200]
            })
    
    # Save and plot results
    df = pd.DataFrame(results)
    df.to_csv('ode_convergence_test.csv', index=False)
    
    # Plot convergence
    if 'diff_v' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        valid = df.dropna(subset=['diff_v'])
        
        # Velocity convergence
        ax = axes[0]
        ax.loglog(valid['eta_M_cold'], valid['diff_v'], 'o-', label='|Δ(dv/dr)|')
        ax.axhline(abs(dydt_hot[0]) * 0.01, color='r', linestyle='--', 
                   alpha=0.5, label='1% of hot-only')
        ax.set_xlabel('η_M_cold')
        ax.set_ylabel('|Full - Hot| [cm/s/cm]')
        ax.set_title('Velocity Derivative Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Density convergence
        ax = axes[1]
        ax.loglog(valid['eta_M_cold'], valid['diff_rho'], 'o-', label='|Δ(drho/dr)|')
        ax.axhline(abs(dydt_hot[1]) * 0.01, color='r', linestyle='--',
                   alpha=0.5, label='1% of hot-only')
        ax.set_xlabel('η_M_cold')
        ax.set_ylabel('|Full - Hot| [g/cm³/cm]')
        ax.set_title('Density Derivative Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Pressure convergence
        ax = axes[2]
        ax.loglog(valid['eta_M_cold'], valid['diff_P'], 'o-', label='|Δ(dP/dr)|')
        ax.axhline(abs(dydt_hot[2]) * 0.01, color='r', linestyle='--',
                   alpha=0.5, label='1% of hot-only')
        ax.set_xlabel('η_M_cold')
        ax.set_ylabel('|Full - Hot| [dyne/cm²/cm]')
        ax.set_title('Pressure Derivative Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('ODE Convergence: Full → Hot-only as η_M_cold → 0')
        plt.tight_layout()
        plt.savefig('ode_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n" + "="*60)
        print("Convergence plot saved to ode_convergence.png")
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not results:
        print("No results obtained - all tests failed")
    elif 'converged' in df.columns:
        n_converged = df['converged'].sum()
        n_total = len(df[df['converged'].notna()])
        print(f"Converged: {n_converged}/{n_total} tests")
        
        if n_converged < n_total:
            print("\nNon-converged cases:")
            for _, row in df[~df['converged']].iterrows():
                if 'eta_M_cold' in row:
                    print(f"  η_M_cold = {row['eta_M_cold']:.1e}:")
                    if 'rel_v' in row:
                        print(f"    Relative errors: v={row['rel_v']:.1%}, "
                              f"ρ={row['rel_rho']:.1%}, P={row['rel_P']:.1%}")
    
    return df

if __name__ == "__main__":
    df = test_ode_convergence()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("Results saved to:")
    print("  - ode_convergence_test.csv")
    print("  - ode_convergence.png (if successful)")
    
    # Key finding
    if 'converged' in df.columns and df['converged'].any():
        print("\n✓ Full model DOES converge to hot-only as η_M_cold → 0")
    else:
        print("\n✗ Full model DOES NOT converge to hot-only")
        print("  This indicates a bug in the ODE implementation")