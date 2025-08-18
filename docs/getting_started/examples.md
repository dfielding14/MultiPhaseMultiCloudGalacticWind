# Examples

## Simple Example

A basic example demonstrating core functionality:

```python
from multiphasegalacticwind import WindModel
from multiphasegalacticwind.plotting import plot_wind_solution
import matplotlib.pyplot as plt

# Create a Milky Way-like model
model = WindModel(
    SFR=1.0,           # Solar mass per year
    v_circ=200.0,      # km/s circular velocity
    eta_M=0.1,         # Hot mass loading
    eta_M_cold=1.0,    # Cold mass loading
    eta_E=1.0          # Energy loading
)

# Run the simulation
solution = model.run()

# Display key results
print(f"Terminal velocity: {solution.v_terminal:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")

# Create visualization
fig, axes = plot_wind_solution(solution)
plt.show()
```

## Comprehensive Example

Advanced usage with custom configuration and analysis:

```python
from multiphasegalacticwind import WindConfig, WindModel
from multiphasegalacticwind.plotting import (
    plot_wind_solution, 
    plot_cloud_evolution,
    plot_column_density_distribution
)
import matplotlib.pyplot as plt
import numpy as np

# Custom configuration for starburst galaxy
config = WindConfig(
    f_turb0=0.2,           # Enhanced turbulence
    drag_coeff=0.3,        # Lower drag
    T_cl=5e3,              # Cooler clouds
    N_cloud_species=10,    # More cloud species
    M_cloud_max=1e7,       # Larger maximum cloud mass
    alpha=1.8              # Steeper mass distribution
)

# Starburst galaxy model
model = WindModel(
    config=config,
    SFR=100.0,            # High star formation rate
    v_circ=150.0,         # Smaller galaxy
    eta_M=0.3,            # Higher mass loading
    eta_M_cold=3.0,       # Much more cold gas
    eta_E=3.0,            # Strong energy injection
    r_max_kpc=200         # Extend to larger radius
)

# Run simulation with progress monitoring
def progress(r, state):
    if r % 10 < 0.1:  # Every ~10 kpc
        v = state[0]
        print(f"  r = {r:6.1f} kpc, v = {v:6.1f} km/s")

print("Running simulation...")
solution = model.run(progress_callback=progress)

# Analyze results
print("\n=== Results ===")
print(f"Terminal velocity: {solution.v_terminal:.1f} km/s")
print(f"Maximum radius reached: {solution.r[-1]:.1f} kpc")
print(f"Final Mach number: {solution.Mach[-1]:.2f}")
print(f"Mass flux at 10 kpc: {solution.mass_flux[solution.r <= 10][-1]:.1f} Msun/yr")

# Calculate observables
v_cloud, dN_dv = solution.calculate_column_density_distribution()
moments = solution.calculate_velocity_moments()

print(f"\nVelocity distribution:")
print(f"  Mean: {moments['mean']:.1f} km/s")
print(f"  Dispersion: {moments['dispersion']:.1f} km/s")
print(f"  Skewness: {moments['skewness']:.2f}")

# Create comprehensive plots
fig = plt.figure(figsize=(16, 12))

# Main solution
fig1, axes1 = plot_wind_solution(solution)
fig1.suptitle('Wind Solution - Starburst Galaxy')

# Cloud evolution
fig2, axes2 = plot_cloud_evolution(solution)
fig2.suptitle('Cloud Evolution')

# Column density distribution
fig3, ax3 = plt.subplots(figsize=(8, 6))
plot_column_density_distribution(solution, ax=ax3)
ax3.set_title('Column Density Distribution')

plt.show()
```

## Parameter Study Example

Exploring parameter space:

```python
import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel

# Parameter grid
sfr_values = np.logspace(-1, 2, 15)  # 0.1 to 100 Msun/yr
eta_E_values = [0.1, 0.3, 1.0, 3.0]  # Energy loading values

# Storage for results
results = {eta: {'sfr': [], 'v_terminal': [], 'mass_loading': []} 
           for eta in eta_E_values}

# Run parameter study
for eta_E in eta_E_values:
    print(f"\nRunning eta_E = {eta_E}")
    for sfr in sfr_values:
        try:
            model = WindModel(
                SFR=sfr,
                eta_E=eta_E,
                rtol=1e-6,  # Relaxed tolerance for speed
                atol=1e-8
            )
            solution = model.run()
            
            results[eta_E]['sfr'].append(sfr)
            results[eta_E]['v_terminal'].append(solution.v_terminal)
            results[eta_E]['mass_loading'].append(solution.mass_loading_at_10kpc)
            print(f"  SFR={sfr:.1f}: v_term={solution.v_terminal:.0f} km/s")
        except Exception as e:
            print(f"  SFR={sfr:.1f}: Failed - {e}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Terminal velocity vs SFR
for eta_E, data in results.items():
    if data['sfr']:
        ax1.loglog(data['sfr'], data['v_terminal'], 
                   'o-', label=f'η_E = {eta_E}')
ax1.set_xlabel('SFR [Msun/yr]')
ax1.set_ylabel('Terminal Velocity [km/s]')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Mass loading vs SFR
for eta_E, data in results.items():
    if data['sfr']:
        ax2.loglog(data['sfr'], data['mass_loading'], 
                   's-', label=f'η_E = {eta_E}')
ax2.set_xlabel('SFR [Msun/yr]')
ax2.set_ylabel('Mass Loading at 10 kpc')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Parameter Study: Energy Loading Effects')
plt.tight_layout()
plt.show()
```

## Observational Comparison Example

Comparing model with observations:

```python
from multiphasegalacticwind import WindModel
import numpy as np
import matplotlib.pyplot as plt

# Model for M82-like starburst
model = WindModel(
    SFR=10.0,          # M82 SFR
    v_circ=100.0,      # M82 rotation
    eta_M=0.3,
    eta_M_cold=2.0,
    eta_E=1.0
)

solution = model.run()

# Calculate observables
v_cloud, dN_dv = solution.calculate_column_density_distribution()

# Mock observational data (example)
v_obs = np.array([100, 200, 300, 400, 500])
dN_dv_obs = np.array([1e14, 5e13, 2e13, 8e12, 3e12])
dN_dv_err = dN_dv_obs * 0.3  # 30% errors

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 7))

# Model prediction
ax.semilogy(v_cloud, dN_dv, 'b-', linewidth=2, 
            label='Model', alpha=0.7)

# "Observations"
ax.errorbar(v_obs, dN_dv_obs, yerr=dN_dv_err,
            fmt='ko', capsize=5, label='Observations')

ax.set_xlabel('Velocity [km/s]')
ax.set_ylabel('dN/dv [cm⁻² / (km/s)]')
ax.set_xlim(0, 600)
ax.set_ylim(1e11, 1e15)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('M82-like Starburst: Model vs Observations')

plt.show()
```

## Hot vs Cold Comparison

Comparing hot-only vs full multiphase model:

```python
from multiphasegalacticwind import WindModel
from multiphasegalacticwind.plotting import plot_comparison
import matplotlib.pyplot as plt

# Create model
model = WindModel(SFR=10.0, v_circ=200.0)

# Run both versions
print("Running hot-only model...")
hot_solution = model.run_hot_only()

print("Running full multiphase model...")
full_solution = model.run()

# Compare results
print(f"\nHot-only terminal velocity: {hot_solution.v_terminal:.1f} km/s")
print(f"Full model terminal velocity: {full_solution.v_terminal:.1f} km/s")
print(f"Velocity reduction: {(1 - full_solution.v_terminal/hot_solution.v_terminal)*100:.1f}%")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Velocity comparison
axes[0,0].plot(hot_solution.r, hot_solution.v, 'r-', 
               label='Hot only', linewidth=2)
axes[0,0].plot(full_solution.r, full_solution.v, 'b-', 
               label='Multiphase', linewidth=2)
axes[0,0].set_xlabel('r [kpc]')
axes[0,0].set_ylabel('v [km/s]')
axes[0,0].legend()
axes[0,0].set_title('Velocity Evolution')

# Mass flux comparison
axes[0,1].loglog(hot_solution.r, hot_solution.mass_flux, 'r-', 
                 label='Hot only', linewidth=2)
axes[0,1].loglog(full_solution.r, full_solution.mass_flux, 'b-', 
                 label='Multiphase', linewidth=2)
axes[0,1].set_xlabel('r [kpc]')
axes[0,1].set_ylabel('Mass Flux [Msun/yr]')
axes[0,1].legend()
axes[0,1].set_title('Mass Flux')

# Temperature comparison
axes[1,0].loglog(hot_solution.r, hot_solution.T, 'r-', 
                 label='Hot only', linewidth=2)
axes[1,0].loglog(full_solution.r, full_solution.T, 'b-', 
                 label='Multiphase', linewidth=2)
axes[1,0].set_xlabel('r [kpc]')
axes[1,0].set_ylabel('T [K]')
axes[1,0].legend()
axes[1,0].set_title('Temperature')

# Mach number comparison
axes[1,1].plot(hot_solution.r, hot_solution.Mach, 'r-', 
               label='Hot only', linewidth=2)
axes[1,1].plot(full_solution.r, full_solution.Mach, 'b-', 
               label='Multiphase', linewidth=2)
axes[1,1].axhline(1, color='k', linestyle='--', alpha=0.5)
axes[1,1].set_xlabel('r [kpc]')
axes[1,1].set_ylabel('Mach Number')
axes[1,1].legend()
axes[1,1].set_title('Mach Number')

plt.suptitle('Hot-Only vs Multiphase Comparison')
plt.tight_layout()
plt.show()
```

## Next Steps

### Essential Reading
- **[MCMC Fitting Guide](../fitting/overview.md)** - Learn to fit models to real data
- **[Parameter Guide](../guide/parameters.md)** - Understand parameter selection
- **[API Reference](../api/wind_model.md)** - Detailed class documentation

### Tutorials
- **[Basic Wind Model](../tutorials/basic_model.md)** - Step-by-step first model
- **[Fitting Example](../tutorials/fitting_example.md)** - Fit CLASSY data
- **[Parameter Study](../tutorials/parameter_study.md)** - Explore parameter space

### Interactive Notebooks
The `examples/` directory contains Jupyter notebooks:
- `simple_example.py` - Basic usage
- `comprehensive_example.py` - Advanced features  
- `tutorial_comprehensive.ipynb` - Interactive tutorial
- `observational_comparison_notebook.ipynb` - Compare with observations

Run locally with:
```bash
cd examples/
jupyter notebook tutorial_comprehensive.ipynb
```