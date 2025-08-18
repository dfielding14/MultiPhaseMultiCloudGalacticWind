#!/usr/bin/env python
"""
Diagnose why small cloud masses show jagged dN/dv behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.constants import kpc, Msun, mp, km
from multiphasegalacticwind.observables import mu_cool

# Create model matching tutorial_comprehensive.ipynb
model = WindModel(
    v_circ=150.0,
    redshift=0.0,
    SFR=20.0,
    eta_M=0.1,
    eta_M_cold=0.3,
    eta_E=1.0,
    r_star_kpc=0.3,
    cloud_mass_range=(10, 1e6),
    cloud_alpha=2.0,
    N_cloud_species=11,
    T_cl=1e4,
    v_cloud_init=20.0,
    M_cloud_min=1e-1*Msun,
    r_max_kpc=30.0,
    rtol=1e-6,
    atol=1e-8,
)

print("Running model...")
solution = model.run()

print(f"\nCloud masses: {model.M_cloud0/Msun} Msun")
print(f"Ndot_cloud0: {model.Ndot_cloud0}")

# Analyze each cloud species
r = solution.sol.t  # cm
r_kpc = r / kpc
mask = (r_kpc >= 0.05) & (r_kpc <= 5.0)
r_use = r[mask]
r_kpc_use = r_kpc[mask]

# Get injection function
injection_radius_kpc = 0.3
injection_power = 6.0
injection_function = np.where(r_kpc_use < injection_radius_kpc,
                             (r_kpc_use / injection_radius_kpc)**injection_power,
                             1.0)

Omwind = model.config.Omwind

print("\n" + "="*70)
print("ANALYZING EACH CLOUD SPECIES")
print("="*70)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(min(model.N_cloud_species, 12)):
    ax = axes[i]
    
    # Get cloud data
    v_cloud_cms = solution.sol.y[4 + model.N_cloud_species + i, mask]  # cm/s
    M_cloud = solution.sol.y[4 + i, mask]  # grams
    Ndot_cloud = model.Ndot_cloud0[i]
    
    print(f"\nSpecies {i} (M0 = {model.M_cloud0[i]/Msun:.1f} Msun):")
    print(f"  M_cloud range: {M_cloud.min()/Msun:.3e} - {M_cloud.max()/Msun:.3e} Msun")
    print(f"  Survival: {M_cloud.max()/model.M_cloud0[i]:.1%}")
    
    # Skip if destroyed
    if np.max(M_cloud) < model.config.M_cloud_min:
        print(f"  DESTROYED (M_max < M_cloud_min)")
        ax.text(0.5, 0.5, f'Species {i}\nDESTROYED', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'M = {model.M_cloud0[i]/Msun:.1f} Msun')
        continue
    
    # Calculate cloud density
    cloud_density = (Ndot_cloud * M_cloud * injection_function / 
                    (Omwind * r_use**2 * v_cloud_cms))
    n_H = cloud_density / (mu_cool * mp)
    
    # Calculate velocity gradient
    grad_v = np.gradient(v_cloud_cms, r_use)
    
    # Check gradient behavior
    n_negative = np.sum(grad_v <= 0)
    print(f"  Velocity gradient: {n_negative}/{len(grad_v)} negative points")
    
    if n_negative > 0:
        print(f"    Min grad_v: {grad_v.min():.3e}")
        print(f"    Max grad_v: {grad_v.max():.3e}")
        neg_idx = np.where(grad_v <= 0)[0]
        if len(neg_idx) > 0:
            print(f"    First negative at r = {r_kpc_use[neg_idx[0]]:.2f} kpc")
    
    # Calculate dN/dv two ways
    v_cloud_kms = v_cloud_cms / 1e5
    
    # Method 1: Direct gradient (will fail if grad_v < 0)
    if np.min(grad_v) > 0:
        grad_v_safe = np.where(grad_v == 0, 1e-30, grad_v)
        dN_dv_direct = n_H / grad_v_safe * 1e5
        ax.loglog(v_cloud_kms, dN_dv_direct, 'b-', lw=2, alpha=0.7, label='Direct')
        method = "Direct gradient"
    else:
        method = "Binning (negative grad_v)"
        # Method 2: Binning approach
        dr = np.gradient(r_use)
        dN = n_H * dr
        
        # Create bins
        sort_idx = np.argsort(v_cloud_cms)
        v_sorted = v_cloud_cms[sort_idx]
        dN_sorted = dN[sort_idx]
        
        n_bins = max(10, len(v_cloud_cms) // 8)
        v_bins = np.linspace(v_sorted.min(), v_sorted.max(), n_bins)
        v_centers = 0.5 * (v_bins[:-1] + v_bins[1:])
        
        dN_binned, _ = np.histogram(v_sorted, bins=v_bins, weights=dN_sorted)
        dv_bins = np.diff(v_bins)
        dN_dv_binned = dN_binned / dv_bins * 1e5  # Convert to per km/s
        
        ax.loglog(v_centers/1e5, dN_dv_binned, 'r-', lw=2, alpha=0.7, label='Binned')
        
        # Also show what direct would look like (with issues)
        grad_v_clipped = np.where(grad_v <= 0, 1e-30, np.abs(grad_v))
        dN_dv_clipped = n_H / grad_v_clipped * 1e5
        ax.loglog(v_cloud_kms, dN_dv_clipped, 'k:', lw=1, alpha=0.5, label='Direct (clipped)')
    
    ax.set_xlabel('v [km/s]')
    ax.set_ylabel('dN/dv')
    ax.set_title(f'M = {model.M_cloud0[i]/Msun:.1f} Msun ({method})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('dN/dv for Each Cloud Species - Diagnosing Jagged Behavior', fontsize=14)
plt.tight_layout()
plt.savefig('cloud_species_diagnosis.pdf', dpi=150, bbox_inches='tight')
print("\nSaved plot to cloud_species_diagnosis.pdf")

# Now let's look specifically at velocity profiles
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Velocity profiles
ax = axes2[0, 0]
for i in range(min(5, model.N_cloud_species)):
    v_cl = solution.sol.y[4 + model.N_cloud_species + i, :] / 1e5  # km/s
    ax.semilogx(solution.r, v_cl, label=f'M = {model.M_cloud0[i]/Msun:.0f} Msun')
ax.set_xlabel('r [kpc]')
ax.set_ylabel('v_cloud [km/s]')
ax.set_title('Cloud Velocities')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Velocity gradients
ax = axes2[0, 1]
for i in range(min(5, model.N_cloud_species)):
    v_cl = solution.sol.y[4 + model.N_cloud_species + i, mask]
    grad_v = np.gradient(v_cl, r_use)
    ax.plot(r_kpc_use, grad_v, label=f'M = {model.M_cloud0[i]/Msun:.0f} Msun')
ax.axhline(0, color='k', ls='--', alpha=0.5)
ax.set_xlabel('r [kpc]')
ax.set_ylabel('dv/dr [(cm/s)/cm]')
ax.set_title('Velocity Gradients')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Cloud masses
ax = axes2[1, 0]
for i in range(min(5, model.N_cloud_species)):
    M_cl = solution.sol.y[4 + i, :] / Msun
    ax.loglog(solution.r, M_cl, label=f'M0 = {model.M_cloud0[i]/Msun:.0f} Msun')
ax.axhline(model.config.M_cloud_min/Msun, color='k', ls='--', alpha=0.5, label='M_min')
ax.set_xlabel('r [kpc]')
ax.set_ylabel('M_cloud [Msun]')
ax.set_title('Cloud Mass Evolution')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Summary statistics
ax = axes2[1, 1]
ax.axis('off')

summary = "DIAGNOSIS SUMMARY\n" + "="*30 + "\n\n"
summary += "Jagged behavior occurs when:\n"
summary += "1. Velocity gradient becomes negative\n"
summary += "   (cloud deceleration)\n\n"
summary += "2. Fallback binning method kicks in\n"
summary += "   (histogram instead of gradient)\n\n"
summary += "3. Small clouds destroyed quickly\n"
summary += "   (M_cloud approaches M_cloud_min)\n\n"

summary += "Species with issues:\n"
for i in range(min(8, model.N_cloud_species)):
    v_cl = solution.sol.y[4 + model.N_cloud_species + i, mask]
    grad_v = np.gradient(v_cl, r_use)
    if np.min(grad_v) <= 0:
        summary += f"  Species {i}: negative grad_v\n"

ax.text(0.1, 0.5, summary, transform=ax.transAxes, 
        fontsize=11, verticalalignment='center', fontfamily='monospace')

plt.suptitle('Velocity and Mass Evolution Diagnostics', fontsize=14)
plt.tight_layout()
plt.savefig('velocity_diagnostics.pdf', dpi=150, bbox_inches='tight')
print("Saved plot to velocity_diagnostics.pdf")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nThe jagged behavior for small clouds is caused by:")
print("1. Negative velocity gradients (cloud deceleration)")
print("2. Switching to binning method instead of direct gradient")
print("3. Binning creates discrete jumps rather than smooth curves")
print("\nThis is expected physics - small clouds slow down and get destroyed")