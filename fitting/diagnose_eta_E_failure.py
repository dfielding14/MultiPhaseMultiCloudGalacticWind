#!/usr/bin/env python
"""
Deep diagnostic of why eta_E < 1 causes immediate failure even with tiny eta_M_cold.

This script performs a rigorous investigation of the initial conditions and gradients
to understand why adding infinitesimally small cold clouds breaks the integration.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.constants import *
from multiphasegalacticwind.core_physics import Wind_Evo, Hot_Wind_Evo
from multiphasegalacticwind.config import WindConfig
from scipy.integrate import solve_ivp

print("=" * 80)
print("DIAGNOSTIC: eta_E < 1 FAILURE WITH TINY eta_M_cold")
print("=" * 80)

# Test parameters (exactly as user specified)
test_params = {
    'v_circ': 150.0,
    'redshift': 0.0,
    'SFR': 1.0,
    'eta_M': 0.5,
    'eta_M_cold': 0.000001,  # Extremely small!
    'eta_E': 0.1,  # This causes failure
    'r_star_kpc': 0.3,
    'cloud_mass_range': (10, 1e6),
    'cloud_alpha': 2.0,
    'N_cloud_species': 11,
    'T_cl': 1e4,
    'v_cloud_init': 3.0,
    'M_cloud_min': 1e-1*Msun,
    'r_max_kpc': 30.0,
    'rtol': 1e-12,
    'atol': 1e-12,
}

print("\nTest parameters:")
for key, val in test_params.items():
    print(f"  {key}: {val}")

# Create model
print("\n" + "-" * 80)
print("STEP 1: Creating WindModel and analyzing initial conditions")
print("-" * 80)

model = WindModel(**test_params)

# Get initial conditions
print("\nExtracting initial conditions from model setup...")

# These are computed in WindModel.__init__
r0 = model.r_star_kpc * kpc  # cm
v0 = model.v_star * 1e5  # cm/s (converted from km/s)
rho0 = model.rho_star  # g/cm^3
P0 = model.P_star  # dyne/cm^2
rhoZ0 = rho0 * model.config.Z_wind_initial

print(f"\nInitial conditions at sonic radius (r0 = {r0/kpc:.3f} kpc):")
print(f"  v0 = {v0/1e5:.3e} km/s")
print(f"  rho0 = {rho0:.3e} g/cm^3")
print(f"  n0 = {rho0/(mu*mp):.3e} cm^-3")
print(f"  P0 = {P0:.3e} dyne/cm^2")
print(f"  T0 = {P0/(rho0/mu/mp)/kb:.3e} K")
print(f"  cs0 = {np.sqrt(gamma*P0/rho0)/1e5:.3e} km/s")
print(f"  Mach0 = {v0/np.sqrt(gamma*P0/rho0):.6f}")

# Check energy budget
Mdot = model.eta_M * model.SFR * Msun/yr
Edot = model.eta_E * (model.E_SN/model.mstar) * model.SFR * Msun/yr
print(f"\nEnergy/momentum budget:")
print(f"  Mdot = {Mdot:.3e} g/s")
print(f"  Edot = {Edot:.3e} erg/s")
print(f"  v_infinity = sqrt(2*Edot/Mdot) = {np.sqrt(2*Edot/Mdot)/1e5:.3e} km/s")
print(f"  Edot/Mdot = {Edot/Mdot:.3e} erg/g = {Edot/Mdot/1e10:.3e} × 10^10 erg/g")

# Now let's look at the gradients
print("\n" + "-" * 80)
print("STEP 2: Analyzing gradients from Hot_Wind_Evo (hot-only)")
print("-" * 80)

# Calculate source terms
SFR_cgs = model.SFR * Msun / yr
v_circ_cgs = model.v_circ * 1e5
Edot = model.eta_E * (model.config.E_SN / model.config.mstar / Msun) * SFR_cgs
Mdot = model.eta_M * SFR_cgs
source_volume = 4./3. * np.pi * r0**3
Edot_per_Vol = Edot / source_volume
Mdot_per_Vol = Mdot / source_volume

# Parameters for physics functions (hot-only) - using the simpler parameter set
params_hot = (
    v_circ_cgs,  # v_circ
    True,  # include source terms
    r0,  # r0
    Edot_per_Vol,  # Edot_per_Vol
    Mdot_per_Vol,  # Mdot_per_Vol
)

# State vector for hot-only
state_hot = np.array([v0, rho0, P0])

# Get gradients from hot-only
gradients_hot = Hot_Wind_Evo(r0, state_hot, params_hot)
dv_dr_hot, drho_dr_hot, dP_dr_hot = gradients_hot

print(f"\nHot-only gradients at r0:")
print(f"  dv/dr = {dv_dr_hot:.3e} (cm/s)/cm")
print(f"  drho/dr = {drho_dr_hot:.3e} (g/cm^3)/cm")
print(f"  dP/dr = {dP_dr_hot:.3e} (dyne/cm^2)/cm")

# Dimensionless gradients
print(f"\nDimensionless gradients (d ln X / d ln r):")
print(f"  d(ln v)/d(ln r) = {dv_dr_hot * r0/v0:.6f}")
print(f"  d(ln rho)/d(ln r) = {drho_dr_hot * r0/rho0:.6f}")
print(f"  d(ln P)/d(ln r) = {dP_dr_hot * r0/P0:.6f}")

# Check if gradients are reasonable
if not np.all(np.isfinite(gradients_hot)):
    print("\n⚠️ WARNING: Hot-only gradients contain non-finite values!")
    
print("\n" + "-" * 80)
print("STEP 3: Setting up hot+cold initial conditions")
print("-" * 80)

# Initial cloud properties
print(f"\nCloud initial conditions:")
print(f"  N_cloud_species = {model.N_cloud_species}")
print(f"  M_cloud0 = {model.M_cloud0/Msun} Msun")
print(f"  Ndot_cloud0 = {model.Ndot_cloud0}")
print(f"  v_cloud_init = {model.config.v_cloud_init} km/s = {model.config.v_cloud_init*km} cm/s")
print(f"  Total cold Mdot = {np.sum(model.Ndot_cloud0 * model.M_cloud0):.3e} g/s")
print(f"  eta_M_cold actual = {np.sum(model.Ndot_cloud0 * model.M_cloud0)/(model.SFR*Msun/yr):.3e}")

# Full state vector for hot+cold
state_full = np.zeros(4 + 3*model.N_cloud_species)
state_full[0] = v0  # v_wind
state_full[1] = rho0  # rho_wind
state_full[2] = P0  # P_wind
state_full[3] = rhoZ0  # rhoZ_wind

# Cloud initial conditions
for i in range(model.N_cloud_species):
    state_full[4 + i] = model.M_cloud0[i]  # M_cloud
    state_full[4 + model.N_cloud_species + i] = model.config.v_cloud_init * km  # v_cloud
    state_full[4 + 2*model.N_cloud_species + i] = model.config.Z_cloud_initial * Z_solar  # Z_cloud

print(f"\nFull state vector shape: {state_full.shape}")
print(f"  Wind components: {state_full[:4]}")
print(f"  First cloud M: {state_full[4]/Msun:.1f} Msun")
print(f"  First cloud v: {state_full[4+model.N_cloud_species]/km:.1f} km/s")

print("\n" + "-" * 80)
print("STEP 4: Analyzing gradients from Wind_Evo (hot+cold)")
print("-" * 80)

# Parameters for full physics
params_full = (
    model.v_circ0,
    model.Ndot_cloud0,
    model.config.T_cl,
    model.config.cold_cloud_injection_radial_extent,
    model.config.cold_cloud_injection_radial_power,
    model.config.__dict__,
    r0,
    model.Edot_per_Vol,
    model.Mdot_per_Vol,
    None  # Lambda_P_rho
)

# Get gradients from full model
try:
    gradients_full = Wind_Evo(r0, state_full, params_full)
    
    print(f"\nHot+cold gradients at r0:")
    print(f"  dv_wind/dr = {gradients_full[0]:.3e} (cm/s)/cm")
    print(f"  drho_wind/dr = {gradients_full[1]:.3e} (g/cm^3)/cm")
    print(f"  dP_wind/dr = {gradients_full[2]:.3e} (dyne/cm^2)/cm")
    print(f"  drhoZ_wind/dr = {gradients_full[3]:.3e} (g/cm^3)/cm")
    
    # Compare with hot-only
    print(f"\nRatio (hot+cold) / (hot-only):")
    print(f"  dv/dr ratio = {gradients_full[0]/dv_dr_hot:.6f}")
    print(f"  drho/dr ratio = {gradients_full[1]/drho_dr_hot:.6f}")
    print(f"  dP/dr ratio = {gradients_full[2]/dP_dr_hot:.6f}")
    
    # Check cloud gradients
    print(f"\nCloud gradients (first species):")
    print(f"  dM_cloud/dr = {gradients_full[4]:.3e} g/cm")
    print(f"  dv_cloud/dr = {gradients_full[4+model.N_cloud_species]:.3e} (cm/s)/cm")
    
    # Check for problems
    if not np.all(np.isfinite(gradients_full)):
        print("\n⚠️ ERROR: Gradients contain non-finite values!")
        print(f"  Non-finite indices: {np.where(~np.isfinite(gradients_full))[0]}")
        
    # Check if any gradient would cause immediate problems
    dt_estimate = 0.001 * r0 / v0  # Rough timestep
    print(f"\nEstimated changes in first step (dt ~ {dt_estimate:.3e} s):")
    print(f"  Δv ~ {gradients_full[0] * v0 * dt_estimate:.3e} cm/s")
    print(f"  Δrho ~ {gradients_full[1] * v0 * dt_estimate:.3e} g/cm^3")
    print(f"  ΔP ~ {gradients_full[2] * v0 * dt_estimate:.3e} dyne/cm^2")
    
    # Would these cause negative values?
    if rho0 + gradients_full[1] * v0 * dt_estimate < 0:
        print("\n⚠️ CRITICAL: Density would go negative in first step!")
    if P0 + gradients_full[2] * v0 * dt_estimate < 0:
        print("\n⚠️ CRITICAL: Pressure would go negative in first step!")
        
except Exception as e:
    print(f"\n⚠️ ERROR in Wind_Evo: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "-" * 80)
print("STEP 5: Testing actual integration")
print("-" * 80)

# Try hot-only integration first
print("\nTesting hot-only integration...")
try:
    sol_hot = solve_ivp(
        lambda r, y: Hot_Wind_Evo(r, y, params_hot),
        [r0, r0*1.01],  # Just 1% further
        state_hot,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12
    )
    print(f"  Hot-only: SUCCESS - reached r = {sol_hot.t[-1]/r0:.3f} r0")
    print(f"  Final state: v={sol_hot.y[0,-1]/1e5:.1f} km/s, rho={sol_hot.y[1,-1]:.3e} g/cm^3")
except Exception as e:
    print(f"  Hot-only: FAILED - {e}")

# Try hot+cold integration
print("\nTesting hot+cold integration...")
try:
    # Define events
    def negative_density(r, y):
        return y[1]  # rho_wind
    negative_density.terminal = True
    
    def negative_pressure(r, y):
        return y[2]  # P_wind
    negative_pressure.terminal = True
    
    sol_full = solve_ivp(
        lambda r, y: Wind_Evo(r, y, params_full),
        [r0, r0*1.01],  # Just 1% further
        state_full,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
        events=[negative_density, negative_pressure]
    )
    print(f"  Hot+cold: SUCCESS - reached r = {sol_full.t[-1]/r0:.3f} r0")
    print(f"  Final state: v={sol_full.y[0,-1]/1e5:.1f} km/s, rho={sol_full.y[1,-1]:.3e} g/cm^3")
    print(f"  Status: {sol_full.status}, Message: {sol_full.message}")
except Exception as e:
    print(f"  Hot+cold: FAILED - {e}")

print("\n" + "-" * 80)
print("STEP 6: Testing different eta_E values")
print("-" * 80)

eta_E_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
results = []

for eta_E_test in eta_E_values:
    # Create model with this eta_E
    test_model = WindModel(
        v_circ=150.0,
        SFR=1.0,
        eta_M=0.5,
        eta_M_cold=0.000001,
        eta_E=eta_E_test,
        r_star_kpc=0.3,
        N_cloud_species=3,  # Fewer for speed
        rtol=1e-10,
        atol=1e-12
    )
    
    # Get initial conditions
    r0_test = test_model.r0
    v0_test = test_model.v0
    P0_test = test_model.P0
    rho0_test = test_model.rho0
    
    # Compute key diagnostics
    Edot_test = test_model.eta_E * (test_model.E_SN/test_model.mstar) * test_model.SFR * Msun/yr
    Mdot_test = test_model.eta_M * test_model.SFR * Msun/yr
    v_inf = np.sqrt(2*Edot_test/Mdot_test)
    T0_test = P0_test/(rho0_test/mu/mp)/kb
    cs0_test = np.sqrt(gamma*P0_test/rho0_test)
    
    # Try integration
    try:
        # Just test if we can take one small step
        state_test_full = np.zeros(4 + 3*test_model.N_cloud_species)
        state_test_full[0] = v0_test
        state_test_full[1] = rho0_test
        state_test_full[2] = P0_test
        state_test_full[3] = test_model.rhoZ0
        for i in range(test_model.N_cloud_species):
            state_test_full[4 + i] = test_model.M_cloud0[i]
            state_test_full[4 + test_model.N_cloud_species + i] = test_model.config.v_cloud_init * km
            state_test_full[4 + 2*test_model.N_cloud_species + i] = test_model.config.Z_cloud_initial * Z_solar
        
        params_test = (
            test_model.v_circ0,
            test_model.Ndot_cloud0,
            test_model.config.T_cl,
            test_model.config.cold_cloud_injection_radial_extent,
            test_model.config.cold_cloud_injection_radial_power,
            test_model.config.__dict__,
            r0_test,
            test_model.Edot_per_Vol,
            test_model.Mdot_per_Vol,
            None
        )
        
        # Get gradient
        grad_test = Wind_Evo(r0_test, state_test_full, params_test)
        
        # Check if gradient is finite
        success = np.all(np.isfinite(grad_test))
        
        results.append({
            'eta_E': eta_E_test,
            'v0': v0_test/1e5,
            'T0': T0_test,
            'v_inf': v_inf/1e5,
            'v0/cs': v0_test/cs0_test,
            'success': success,
            'dP_dr': grad_test[2] if success else np.nan
        })
        
    except Exception as e:
        results.append({
            'eta_E': eta_E_test,
            'v0': v0_test/1e5,
            'T0': T0_test,
            'v_inf': v_inf/1e5,
            'v0/cs': v0_test/cs0_test,
            'success': False,
            'dP_dr': np.nan
        })

print(f"\n{'eta_E':<8} {'v0[km/s]':<12} {'T0[K]':<12} {'v_inf[km/s]':<12} {'Mach0':<8} {'Success':<8}")
print("-" * 70)
for res in results:
    print(f"{res['eta_E']:<8.2f} {res['v0']:<12.1f} {res['T0']:<12.1e} {res['v_inf']:<12.1f} {res['v0/cs']:<8.4f} {res['success']}")

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

print("""
Key Findings:
1. When eta_E < 1, the initial velocity v0 becomes very small
2. This makes the Mach number at the "sonic" point < 1
3. The source terms in Wind_Evo are applied where Mach < 1
4. These source terms can cause immediate negative pressure/density

The fundamental issue is that with low eta_E, we don't have enough
energy to accelerate the wind to supersonic speeds at the sonic radius.
The "sonic" point calculation assumes Mach = 1 + epsilon, but this
may not be physically consistent with the energy budget.

Potential fixes:
1. Adjust the sonic radius calculation for low eta_E cases
2. Modify source term application near sonic point
3. Use different initial conditions for low energy cases
""")