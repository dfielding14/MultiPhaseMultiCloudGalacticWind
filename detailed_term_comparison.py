#!/usr/bin/env python
"""
Detailed term-by-term comparison between hot-only and full model.
Identifies which gradient terms cause wind launch failure.
"""

import numpy as np
import pandas as pd
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.core_physics import Wind_Evo, Hot_Wind_Evo
from multiphasegalacticwind.constants import kpc, Msun, kb, mp, yr, G

def extract_detailed_terms(model, is_hot_only=False):
    """Extract all ODE terms at the starting radius"""
    
    r0 = model.r_star_kpc * kpc  # Starting radius in cm
    
    # Calculate source terms
    Edot = model.eta_E * 3e42 * model.SFR  # erg/s
    Mdot = model.eta_M * (model.SFR * Msun/yr)  # g/s
    source_volume = 4./3. * np.pi * r0**3
    Edot_per_Vol = Edot / source_volume
    Mdot_per_Vol = Mdot / source_volume
    
    # Get initial conditions
    if is_hot_only:
        # Hot-only: [v_wind, rho_wind, P, rhoZ_wind]
        v_wind = 30.0 * 1e5  # cm/s
        rho_wind = model.rho_star
        P = model.P_star
        rhoZ_wind = rho_wind * model.config.metallicity
        y0 = np.array([v_wind, rho_wind, P, rhoZ_wind])
        
        # Hot-only parameters
        v_circ_cgs = model.v_circ * 1e5
        params = (v_circ_cgs, True, r0, Edot_per_Vol, Mdot_per_Vol)
        
        # Evaluate ODE
        dydt = Hot_Wind_Evo(r0, y0, params)
        
    else:
        # Full model with clouds
        v_wind = 30.0 * 1e5
        rho_wind = model.rho_star
        P = model.P_star
        rhoZ_wind = rho_wind * model.config.metallicity
        
        y0 = [v_wind, rho_wind, P, rhoZ_wind]
        
        # Add clouds if present
        if model.N_cloud_species > 0:
            for i in range(model.N_cloud_species):
                y0.append(model.M_cloud0[i])  # M_cloud in grams
                y0.append(model.config.v_cloud_init * 1e5)  # v_cloud in cm/s
                y0.append(model.config.metallicity)  # Z_cloud
        
        y0 = np.array(y0)
        
        # Full model parameters
        from multiphasegalacticwind.core_physics import setup_sonic_point_params
        
        v_circ_cgs = model.v_circ * 1e5
        injection_radius = model.config.cold_cloud_injection_radial_extent_frac * r0
        injection_power = model.config.cold_cloud_injection_radial_power
        
        params = setup_sonic_point_params(
            v_circ=v_circ_cgs,
            N_cloud_species=model.N_cloud_species,
            T_cloud=model.config.T_cl,
            injection_radius=injection_radius,
            injection_power=injection_power,
            config_dict=model.config.to_dict(),
            r0=r0,
            Edot_per_Vol=Edot_per_Vol,
            Mdot_per_Vol=Mdot_per_Vol,
            Lambda_P_rho=None
        )
        
        if model.N_cloud_species > 0:
            params = params + (model.Ndot_cloud0,)
        
        # Evaluate ODE
        dydt = Wind_Evo(r0, y0, params)
    
    # Decompose terms
    terms = {}
    
    # Basic state
    terms['r_kpc'] = r0 / kpc
    terms['v_wind_km/s'] = v_wind / 1e5
    terms['rho_wind'] = rho_wind
    terms['P'] = P
    terms['T'] = P / (rho_wind / (0.61 * mp)) / kb
    
    # Derivatives
    terms['dv/dr'] = dydt[0]
    terms['drho/dr'] = dydt[1]
    terms['dP/dr'] = dydt[2]
    terms['drhoZ/dr'] = dydt[3] if len(dydt) > 3 else 0.0
    
    # Decompose velocity equation: dv/dr = -g/v - (1/ρv)(dP/dr) + drag + injection
    g = (model.v_circ * 1e5)**2 / r0  # Gravitational acceleration
    
    terms['dv/dr_gravity'] = -g / v_wind
    terms['dv/dr_pressure'] = -(1/rho_wind) * (1/v_wind) * dydt[2]  # Using actual dP/dr
    
    # The difference is drag + injection + other terms
    terms['dv/dr_other'] = dydt[0] - terms['dv/dr_gravity'] - terms['dv/dr_pressure']
    
    # Decompose pressure equation
    gamma = 5/3
    terms['dP/dr_adiabatic'] = -gamma * P * v_wind / r0  # P∇·v term
    terms['dP/dr_other'] = dydt[2] - terms['dP/dr_adiabatic']
    
    # Decompose density equation
    terms['drho/dr_continuity'] = -rho_wind * (2*v_wind/r0 + dydt[0]/v_wind)
    terms['drho/dr_other'] = dydt[1] - terms['drho/dr_continuity']
    
    return terms

def compare_models():
    """Compare hot-only vs full model with tiny cold component"""
    
    print("="*70)
    print("DETAILED TERM-BY-TERM COMPARISON")
    print("="*70)
    
    # Test parameters
    params = {
        'SFR': 3.0,
        'v_circ': 0.001,
        'eta_M': 1.0,
        'eta_E': 1.0,
    }
    
    print(f"\nBase parameters:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    
    results = []
    
    # Test 1: Hot-only model
    print(f"\n{'='*50}")
    print("MODEL 1: Hot-only (η_M_cold = 0)")
    
    config_hot = WindConfig(
        N_cloud_species=0,
        cooling_factor=0.0,
        metallicity=0.3,
        rtol=1e-6,
        atol=1e-8
    )
    
    model_hot = WindModel(
        config=config_hot,
        eta_M_cold=0.0,
        **params
    )
    
    terms_hot = extract_detailed_terms(model_hot, is_hot_only=True)
    
    print(f"\nInitial conditions:")
    print(f"  r = {terms_hot['r_kpc']:.3f} kpc")
    print(f"  v = {terms_hot['v_wind_km/s']:.1f} km/s")
    print(f"  T = {terms_hot['T']:.2e} K")
    
    print(f"\nGradient terms:")
    print(f"  dv/dr total     = {terms_hot['dv/dr']:.3e} cm/s/cm")
    print(f"    gravity       = {terms_hot['dv/dr_gravity']:.3e}")
    print(f"    pressure      = {terms_hot['dv/dr_pressure']:.3e}")
    print(f"    other         = {terms_hot['dv/dr_other']:.3e}")
    
    print(f"  dP/dr total     = {terms_hot['dP/dr']:.3e} dyne/cm²/cm")
    print(f"    adiabatic     = {terms_hot['dP/dr_adiabatic']:.3e}")
    print(f"    other         = {terms_hot['dP/dr_other']:.3e}")
    
    print(f"  drho/dr total   = {terms_hot['drho/dr']:.3e} g/cm³/cm")
    print(f"    continuity    = {terms_hot['drho/dr_continuity']:.3e}")
    print(f"    other         = {terms_hot['drho/dr_other']:.3e}")
    
    results.append({'model': 'hot_only', **terms_hot})
    
    # Test 2: Full model with tiny cold
    print(f"\n{'='*50}")
    print("MODEL 2: Full with η_M_cold = 1e-12")
    
    config_tiny = WindConfig(
        N_cloud_species=5,
        M_cloud_min=0.1,
        M_cloud_max=1e6,
        cooling_factor=0.0,
        metallicity=0.3,
        rtol=1e-6,
        atol=1e-8
    )
    
    model_tiny = WindModel(
        config=config_tiny,
        eta_M_cold=1e-12,  # Vanishingly small
        **params
    )
    
    terms_tiny = extract_detailed_terms(model_tiny, is_hot_only=False)
    
    print(f"\nInitial conditions:")
    print(f"  r = {terms_tiny['r_kpc']:.3f} kpc")
    print(f"  v = {terms_tiny['v_wind_km/s']:.1f} km/s")
    print(f"  T = {terms_tiny['T']:.2e} K")
    
    print(f"\nGradient terms:")
    print(f"  dv/dr total     = {terms_tiny['dv/dr']:.3e} cm/s/cm")
    print(f"    gravity       = {terms_tiny['dv/dr_gravity']:.3e}")
    print(f"    pressure      = {terms_tiny['dv/dr_pressure']:.3e}")
    print(f"    other         = {terms_tiny['dv/dr_other']:.3e}")
    
    print(f"  dP/dr total     = {terms_tiny['dP/dr']:.3e} dyne/cm²/cm")
    print(f"    adiabatic     = {terms_tiny['dP/dr_adiabatic']:.3e}")
    print(f"    other         = {terms_tiny['dP/dr_other']:.3e}")
    
    print(f"  drho/dr total   = {terms_tiny['drho/dr']:.3e} g/cm³/cm")
    print(f"    continuity    = {terms_tiny['drho/dr_continuity']:.3e}")
    print(f"    other         = {terms_tiny['drho/dr_other']:.3e}")
    
    results.append({'model': 'tiny_cold', **terms_tiny})
    
    # Test 3: Full model with significant cold
    print(f"\n{'='*50}")
    print("MODEL 3: Full with η_M_cold = 0.3")
    
    config_full = WindConfig(
        N_cloud_species=5,
        M_cloud_min=0.1,
        M_cloud_max=1e6,
        cooling_factor=0.0,
        metallicity=0.3,
        rtol=1e-6,
        atol=1e-8
    )
    
    model_full = WindModel(
        config=config_full,
        eta_M_cold=0.3,  # Significant cold component
        **params
    )
    
    terms_full = extract_detailed_terms(model_full, is_hot_only=False)
    
    print(f"\nInitial conditions:")
    print(f"  r = {terms_full['r_kpc']:.3f} kpc")
    print(f"  v = {terms_full['v_wind_km/s']:.1f} km/s")
    print(f"  T = {terms_full['T']:.2e} K")
    
    print(f"\nGradient terms:")
    print(f"  dv/dr total     = {terms_full['dv/dr']:.3e} cm/s/cm")
    print(f"    gravity       = {terms_full['dv/dr_gravity']:.3e}")
    print(f"    pressure      = {terms_full['dv/dr_pressure']:.3e}")
    print(f"    other         = {terms_full['dv/dr_other']:.3e}")
    
    print(f"  dP/dr total     = {terms_full['dP/dr']:.3e} dyne/cm²/cm")
    print(f"    adiabatic     = {terms_full['dP/dr_adiabatic']:.3e}")
    print(f"    other         = {terms_full['dP/dr_other']:.3e}")
    
    print(f"  drho/dr total   = {terms_full['drho/dr']:.3e} g/cm³/cm")
    print(f"    continuity    = {terms_full['drho/dr_continuity']:.3e}")
    print(f"    other         = {terms_full['drho/dr_other']:.3e}")
    
    results.append({'model': 'full_cold', **terms_full})
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    df = pd.DataFrame(results)
    
    # Compare tiny cold to hot-only
    print("\n1. Tiny cold (1e-12) vs Hot-only:")
    for col in ['dv/dr', 'dP/dr', 'drho/dr']:
        hot_val = df[df['model'] == 'hot_only'][col].values[0]
        tiny_val = df[df['model'] == 'tiny_cold'][col].values[0]
        diff = abs(tiny_val - hot_val)
        rel_diff = diff / (abs(hot_val) + 1e-20)
        print(f"  {col:12s}: Δ = {diff:.2e} ({rel_diff:.1%})")
    
    # Compare full cold to hot-only
    print("\n2. Full cold (0.3) vs Hot-only:")
    for col in ['dv/dr', 'dP/dr', 'drho/dr']:
        hot_val = df[df['model'] == 'hot_only'][col].values[0]
        full_val = df[df['model'] == 'full_cold'][col].values[0]
        diff = abs(full_val - hot_val)
        rel_diff = diff / (abs(hot_val) + 1e-20)
        print(f"  {col:12s}: Δ = {diff:.2e} ({rel_diff:.1%})")
    
    # Identify problematic terms
    print("\n3. Which terms change most with cold component?")
    
    for term in ['dv/dr_gravity', 'dv/dr_pressure', 'dv/dr_other',
                 'dP/dr_adiabatic', 'dP/dr_other',
                 'drho/dr_continuity', 'drho/dr_other']:
        if term in df.columns:
            hot_val = df[df['model'] == 'hot_only'][term].values[0]
            full_val = df[df['model'] == 'full_cold'][term].values[0]
            diff = full_val - hot_val
            
            if abs(diff) > 1e-20:
                print(f"  {term:20s}: {hot_val:.2e} → {full_val:.2e} (Δ = {diff:.2e})")
    
    # Save results
    df.to_csv('term_comparison.csv', index=False)
    print(f"\nResults saved to term_comparison.csv")
    
    # Test actual integration
    print(f"\n{'='*70}")
    print("INTEGRATION TEST")
    print(f"{'='*70}")
    
    for model_name, model in [('Hot-only', model_hot), ('Tiny cold', model_tiny), ('Full cold', model_full)]:
        try:
            solution = model.run()
            print(f"{model_name:12s}: SUCCESS - reached {solution.r[-1]:.1f} kpc")
        except Exception as e:
            print(f"{model_name:12s}: FAILED - {str(e)[:50]}")
    
    return df

if __name__ == "__main__":
    df = compare_models()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    print("\nThe comparison reveals which gradient terms are affected by the cold component.")
    print("Look for terms where 'other' components are large - these include drag and mixing.")