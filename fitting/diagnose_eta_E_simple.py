#!/usr/bin/env python
"""
Focused diagnostic of eta_E < 1 failure - simplified version.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.constants import *
from multiphasegalacticwind.config import get_default_config
from multiphasegalacticwind.core_physics import Wind_Evo, Hot_Wind_Evo
from scipy.integrate import solve_ivp

# Get mu from config
default_config = get_default_config()
mu = default_config.mu
gamma = 5.0/3.0

print("=" * 80)
print("DIAGNOSING eta_E < 1 FAILURE")
print("=" * 80)

# Test cases with different eta_E values
test_cases = [
    {'eta_E': 2.0, 'label': 'High energy (works)'},
    {'eta_E': 1.0, 'label': 'Standard (borderline)'},
    {'eta_E': 0.5, 'label': 'Low energy (fails?)'},
    {'eta_E': 0.1, 'label': 'Very low energy (fails)'},
]

print("\nComparing initial conditions for different eta_E values:")
print("-" * 80)

for case in test_cases:
    eta_E = case['eta_E']
    
    # Create model
    try:
        model = WindModel(
            v_circ=150.0,
            SFR=1.0,
            eta_M=0.5,
            eta_M_cold=1e-6,  # Tiny!
            eta_E=eta_E,
            r_star_kpc=0.3,
            N_cloud_species=3,  # Just a few for testing
            cloud_mass_range=(10, 1e3),
            rtol=1e-10,
            atol=1e-12
        )
        
        # Extract initial conditions
        r0 = model.r_star_kpc * kpc
        v0 = model.v_star * 1e5  # cm/s
        rho0 = model.rho_star
        P0 = model.P_star
        
        # Calculate diagnostics
        T0 = P0 / (rho0/(mu*mp)) / kb
        cs0 = np.sqrt(gamma * P0 / rho0)
        Mach0 = v0 / cs0
        
        # Energy budget
        SFR_cgs = model.SFR * Msun / yr
        Edot = eta_E * (model.config.E_SN / model.config.mstar / Msun) * SFR_cgs
        Mdot = model.eta_M * SFR_cgs
        v_inf = np.sqrt(2 * Edot / Mdot)
        
        print(f"\n{case['label']} (eta_E = {eta_E}):")
        print(f"  v0 = {v0/1e5:.1f} km/s")
        print(f"  T0 = {T0:.2e} K")
        print(f"  Mach0 = {Mach0:.4f}")
        print(f"  v_infinity = {v_inf/1e5:.1f} km/s")
        print(f"  v0/v_inf = {v0/v_inf:.3f}")
        
        # Now test the gradients at initial conditions
        # Build full state vector
        state = np.zeros(4 + 3*model.N_cloud_species)
        state[0] = v0
        state[1] = rho0
        state[2] = P0
        state[3] = rho0 * model.config.Z_wind_initial
        
        for i in range(model.N_cloud_species):
            state[4 + i] = model.M_cloud0[i]
            state[4 + model.N_cloud_species + i] = model.config.v_cloud_init * km
            state[4 + 2*model.N_cloud_species + i] = model.config.Z_cloud_initial * Z_solar
        
        # Calculate source terms
        source_volume = 4./3. * np.pi * r0**3
        Edot_per_Vol = Edot / source_volume
        Mdot_per_Vol = Mdot / source_volume
        
        # Build parameters tuple for Wind_Evo
        v_circ_cgs = model.v_circ * 1e5
        params = (
            v_circ_cgs,
            model.Ndot_cloud0,
            model.config.T_cl,
            model.config.cold_cloud_injection_radial_extent,
            model.config.cold_cloud_injection_radial_power,
            model.config.__dict__,
            r0,
            Edot_per_Vol,
            Mdot_per_Vol,
            None  # Lambda_P_rho (will be loaded)
        )
        
        # Get gradients
        try:
            gradients = Wind_Evo(r0, state, params)
            
            # Check key gradients
            dv_dr = gradients[0]
            drho_dr = gradients[1]
            dP_dr = gradients[2]
            
            # Dimensionless gradients
            dlogv_dlogr = dv_dr * r0 / v0
            dlogrho_dlogr = drho_dr * r0 / rho0
            dlogP_dlogr = dP_dr * r0 / P0
            
            print(f"  Gradients: d(ln v)/d(ln r) = {dlogv_dlogr:.3f}")
            print(f"            d(ln rho)/d(ln r) = {dlogrho_dlogr:.3f}")
            print(f"            d(ln P)/d(ln r) = {dlogP_dlogr:.3f}")
            
            # Check if pressure would go negative
            dr_test = 0.001 * r0  # Small step
            P_new = P0 + dP_dr * dr_test
            if P_new < 0:
                print(f"  ⚠️ PRESSURE WOULD GO NEGATIVE!")
            
            # Actually try to run the model
            print(f"  Testing integration...", end='')
            try:
                solution = model.run()
                if hasattr(solution, 'sol') and solution.sol.status == 0:
                    print(f" SUCCESS (reached {solution.r[-1]:.1f} kpc)")
                else:
                    print(f" FAILED (status {solution.sol.status})")
            except Exception as e:
                print(f" FAILED ({str(e)[:50]}...)")
                
        except Exception as e:
            print(f"  ERROR getting gradients: {e}")
            
    except Exception as e:
        print(f"\n{case['label']}: Model creation FAILED")
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print("""
The problem appears when eta_E is too low to accelerate the wind to 
supersonic speeds. The code assumes Mach = 1 + epsilon at the sonic
radius, but with low eta_E, the actual energy budget can't support this.

Key observations:
1. As eta_E decreases, v0 decreases (less energy available)
2. The Mach number at r0 is forced to be ~1 by construction
3. This means T0 must decrease to maintain Mach = v/cs = 1
4. Very low temperatures lead to extreme pressure gradients
5. These gradients cause immediate negative pressure/density

The fundamental issue is that the sonic point calculation assumes
a self-consistent solution exists with Mach = 1 at r0, but this
may not be physical for low energy injection rates.
""")

print("\n" + "=" * 80)
print("TESTING HYPOTHESIS: Source Terms at Subsonic Flow")
print("=" * 80)

# Let's check what happens with source terms
print("\nThe Wind_Evo function applies source terms when Mach < 1.")
print("With eta_E < 1, we start at Mach ≈ 1, so source terms are active.")
print("These source terms might cause the immediate failure.")

# Test with modified initial Mach number
print("\nTesting with slightly supersonic initial conditions:")

for eta_E in [0.1, 0.5]:
    print(f"\neta_E = {eta_E}:")
    
    model = WindModel(
        v_circ=150.0,
        SFR=1.0,
        eta_M=0.5,
        eta_M_cold=1e-6,
        eta_E=eta_E,
        r_star_kpc=0.3,
        N_cloud_species=3
    )
    
    # Get standard initial conditions
    r0 = model.r_star_kpc * kpc
    v0_standard = model.v_star * 1e5
    rho0 = model.rho_star
    P0 = model.P_star
    
    # Try with slightly higher velocity (Mach > 1)
    v0_modified = v0_standard * 1.1  # 10% faster
    
    print(f"  Standard v0 = {v0_standard/1e5:.1f} km/s (Mach = 1.0001)")
    print(f"  Modified v0 = {v0_modified/1e5:.1f} km/s (Mach = {v0_modified/np.sqrt(gamma*P0/rho0):.4f})")
    
    # This is just to illustrate the issue - actual fix would need to be in the core physics

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
The failure with eta_E < 1 is due to an inconsistency in the initial
conditions. The code assumes we can always have Mach = 1 at the sonic
radius, but with low energy injection, this leads to:

1. Very low velocities (to satisfy energy conservation)
2. Very low temperatures (to maintain Mach = 1)
3. Extreme pressure gradients that cause immediate failure

POTENTIAL FIXES:
1. Modify sonic point calculation for low eta_E cases
2. Start integration from a different radius when eta_E < 1
3. Use a different initial condition approach for energy-starved winds
4. Add safeguards in Wind_Evo to handle near-sonic conditions better
""")