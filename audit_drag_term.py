#!/usr/bin/env python
"""
Careful audit of the drag term implementation in Wind_Evo.
Check sign conventions and momentum conservation.
"""

import numpy as np
from multiphasegalacticwind.core_physics import Wind_Evo
from multiphasegalacticwind.constants import kpc, Msun, kb, mp

def test_drag_sign_convention():
    """Test that drag force has correct sign for different relative velocities"""
    
    print("="*70)
    print("DRAG TERM AUDIT")
    print("="*70)
    
    # Setup test parameters
    r = 0.3 * kpc
    v_circ = 0.001 * 1e5  # Nearly zero
    
    # Common state
    rho_wind = 1e-24  # g/cm³
    P = 1e-10  # dyne/cm²
    metallicity = 0.3
    rhoZ_wind = rho_wind * metallicity
    
    # Test cases: wind faster, equal, cloud faster
    test_cases = [
        {'v_wind': 50*1e5, 'v_cloud': 30*1e5, 'label': 'Wind faster (v_rel > 0)'},
        {'v_wind': 30*1e5, 'v_cloud': 30*1e5, 'label': 'Equal velocities (v_rel = 0)'},
        {'v_wind': 30*1e5, 'v_cloud': 50*1e5, 'label': 'Cloud faster (v_rel < 0)'},
    ]
    
    for test in test_cases:
        print(f"\n{'='*50}")
        print(f"TEST: {test['label']}")
        
        v_wind = test['v_wind']
        v_cloud = test['v_cloud']
        v_rel = v_wind - v_cloud
        
        print(f"  v_wind = {v_wind/1e5:.1f} km/s")
        print(f"  v_cloud = {v_cloud/1e5:.1f} km/s")
        print(f"  v_rel = {v_rel/1e5:.1f} km/s")
        
        # Create state with one cloud
        M_cloud = 1e30  # grams (small cloud)
        Z_cloud = metallicity
        
        y = np.array([
            v_wind, rho_wind, P, rhoZ_wind,
            M_cloud, v_cloud, Z_cloud
        ])
        
        # Parameters
        config_dict = {
            'N_cloud_species': 1,
            'metallicity': metallicity,
            'cooling_factor': 0.0,
            'Cooling_Factor': 0.0,
            'f_turb0': 0.0,  # No turbulence for clean test
            'drag_coeff': 1.0,  # Unity for simplicity
            'M_cloud_min': 0.1 * Msun,
            'CoolingAreaChiPower': 0.5,
            'ColdTurbulenceChiPower': -0.5,
            'TurbulentVelocityChiPower': 0.0,
            'geometric_factor': 1.0,
            'Mdot_coefficient': 0.0,  # No mass transfer
            'Omwind': 4*np.pi,
            'mu': 0.61,
        }
        
        T_cloud = 1e4
        r0 = 0.3 * kpc
        injection_radius = r0
        injection_power = 1.0
        Ndot_cloud0 = np.array([1e-10])  # Very small
        
        params = (
            v_circ, Ndot_cloud0, T_cloud, injection_radius,
            injection_power, config_dict, r0, 0.0, 0.0, None
        )
        
        # Get derivatives
        dydt = Wind_Evo(r, y, params)
        
        dv_wind_dr = dydt[0]
        dv_cloud_dr = dydt[5]
        
        print(f"\nResults:")
        print(f"  dv_wind/dr = {dv_wind_dr:.3e} cm/s/cm")
        print(f"  dv_cloud/dr = {dv_cloud_dr:.3e} cm/s/cm")
        
        # Physical expectation
        print(f"\nPhysical expectation:")
        if v_rel > 0:
            print("  Wind faster → drag should:")
            print("    - Slow down wind (dv_wind/dr < 0 or less positive)")
            print("    - Speed up cloud (dv_cloud/dr > 0)")
            
            # Check signs
            if dv_cloud_dr > 0:
                print("  ✓ Cloud acceleration correct")
            else:
                print("  ✗ ERROR: Cloud should accelerate but dv_cloud/dr < 0")
                
        elif v_rel < 0:
            print("  Cloud faster → drag should:")
            print("    - Speed up wind (dv_wind/dr > 0 or less negative)")
            print("    - Slow down cloud (dv_cloud/dr < 0)")
            
            # Check signs
            if dv_cloud_dr < 0:
                print("  ✓ Cloud deceleration correct")
            else:
                print("  ✗ ERROR: Cloud should decelerate but dv_cloud/dr > 0")
                
        else:
            print("  Equal velocities → no drag force")
            print("    - Both accelerations should be from gravity only")
        
        # Calculate drag force directly
        print(f"\nDrag force analysis:")
        
        # From the code (line 209):
        drag_coeff = config_dict['drag_coeff']
        rho_cloud = P * (config_dict['mu']*mp) / (kb*T_cloud)
        r_cloud = (M_cloud / (4*np.pi/3. * rho_cloud))**(1/3.)
        
        # Current implementation
        p_dot_ram_current = 0.5 * drag_coeff * rho_wind * np.pi * v_rel**2 * r_cloud**2
        
        # Correct implementation should be
        p_dot_ram_correct = 0.5 * drag_coeff * rho_wind * np.pi * abs(v_rel) * v_rel * r_cloud**2
        
        print(f"  Current implementation: F_drag = {p_dot_ram_current:.3e}")
        print(f"  Correct implementation: F_drag = {p_dot_ram_correct:.3e}")
        
        if v_rel != 0:
            print(f"  Ratio: {p_dot_ram_correct/p_dot_ram_current:.3f}")
            if p_dot_ram_correct/p_dot_ram_current < 0:
                print("  ✗ SIGN ERROR: Forces have opposite signs!")
    
    # Test momentum conservation
    print(f"\n{'='*70}")
    print("MOMENTUM CONSERVATION TEST")
    print(f"{'='*70}")
    
    # For the wind faster case
    v_wind = 50*1e5
    v_cloud = 30*1e5
    
    y = np.array([
        v_wind, rho_wind, P, rhoZ_wind,
        M_cloud, v_cloud, Z_cloud
    ])
    
    dydt = Wind_Evo(r, y, params)
    
    # Momentum change rates
    # Wind momentum: p_wind = rho_wind * Volume * v_wind
    # But we need the actual mass, which involves number density
    
    print("\nThe current implementation uses v_rel**2, which is ALWAYS positive.")
    print("This means:")
    print("  1. Clouds ALWAYS feel positive force (accelerate)")
    print("  2. Wind ALWAYS feels negative force (decelerate)")
    print("\nThis violates Newton's third law when v_cloud > v_wind!")
    
    return

def analyze_drag_implementation():
    """Analyze the mathematical form of the drag term"""
    
    print("\n" + "="*70)
    print("DRAG IMPLEMENTATION ANALYSIS")
    print("="*70)
    
    print("\nCurrent implementation (line 209):")
    print("  p_dot_ram = 0.5 * drag_coeff * rho_wind * pi * v_rel**2 * r_cloud**2")
    print("  where v_rel = v_wind - v_cloud")
    
    print("\nIssue: v_rel**2 is ALWAYS positive regardless of sign of v_rel")
    
    print("\nCorrect implementation should be:")
    print("  p_dot_ram = 0.5 * drag_coeff * rho_wind * pi * |v_rel| * v_rel * r_cloud**2")
    print("  or equivalently:")
    print("  p_dot_ram = 0.5 * drag_coeff * rho_wind * pi * v_rel * |v_rel| * r_cloud**2")
    
    print("\nThis preserves the sign of v_rel:")
    print("  - If v_rel > 0 (wind faster): p_dot_ram > 0")
    print("  - If v_rel < 0 (cloud faster): p_dot_ram < 0")
    
    print("\nEffect on accelerations:")
    print("  Cloud: dv_cloud/dr ∝ +p_dot_ram  (line 253)")
    print("  Wind:  through dpdt ∝ -p_dot_ram  (line 211)")
    
    print("\nWith correct signs:")
    print("  - v_rel > 0: p_dot_ram > 0 → cloud speeds up, wind slows down ✓")
    print("  - v_rel < 0: p_dot_ram < 0 → cloud slows down, wind speeds up ✓")
    
    print("\nWith current bug (v_rel**2):")
    print("  - v_rel > 0: p_dot_ram > 0 → cloud speeds up, wind slows down ✓")
    print("  - v_rel < 0: p_dot_ram > 0 → cloud speeds up, wind slows down ✗")
    
    print("\nThis explains the huge positive dv/dr we saw!")
    print("When clouds are slower than wind at injection, drag artificially")
    print("accelerates them forward, creating unphysical positive velocity gradient.")

if __name__ == "__main__":
    test_drag_sign_convention()
    analyze_drag_implementation()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nThe drag term has a CRITICAL BUG:")
    print("  Line 209 uses v_rel**2 instead of |v_rel|*v_rel")
    print("\nThis causes:")
    print("  1. Incorrect sign when v_cloud > v_wind")
    print("  2. Violation of Newton's third law")
    print("  3. Unphysical acceleration of clouds")
    print("  4. Numerical instability and integration failure")
    print("\nFIX: Change line 209 to:")
    print("  p_dot_ram = 0.5 * drag_coeff * rho_wind * np.pi * v_rel * np.abs(v_rel) * r_cloud**2")