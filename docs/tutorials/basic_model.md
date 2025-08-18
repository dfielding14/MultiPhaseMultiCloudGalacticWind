# Tutorial: Basic Wind Model

## Introduction

This tutorial walks through creating your first wind model step-by-step. We'll start simple and gradually add complexity.

## Step 1: Import and Setup

```python
# Import the package
from multiphasegalacticwind import WindModel, WindConfig
import numpy as np
import matplotlib.pyplot as plt

# Set up clean plotting style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

## Step 2: Your First Model

Let's create a simple Milky Way-like galaxy:

```python
# Create the model
model = WindModel(
    SFR=1.0,       # Star formation rate: 1 Msun/yr (Milky Way-like)
    v_circ=200.0   # Circular velocity: 200 km/s
)

# Check the parameters
params = model.get_parameters()
print(f"Sonic radius: {params['r0']:.2f} kpc")
print(f"Energy injection: {params['Edot']:.2e} erg/s")
```

## Step 3: Run the Simulation

```python
# Run the model
print("Running simulation...")
solution = model.run()

# Check the results
print(f"\nResults:")
print(f"Terminal velocity: {solution.v_terminal:.1f} km/s")
print(f"Maximum radius reached: {solution.r[-1]:.1f} kpc")
print(f"Final Mach number: {solution.Mach[-1]:.2f}")
```

## Step 4: Visualize the Solution

```python
# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Velocity profile
ax = axes[0, 0]
ax.plot(solution.r, solution.v, 'b-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.set_title('Wind Velocity')
ax.grid(True, alpha=0.3)

# Density profile
ax = axes[0, 1]
ax.loglog(solution.r, solution.rho, 'r-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Density [g/cm³]')
ax.set_title('Wind Density')
ax.grid(True, alpha=0.3)

# Temperature profile
ax = axes[1, 0]
ax.loglog(solution.r, solution.T, 'g-', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Temperature [K]')
ax.set_title('Wind Temperature')
ax.grid(True, alpha=0.3)

# Mach number
ax = axes[1, 1]
ax.plot(solution.r, solution.Mach, 'm-', linewidth=2)
ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='Sonic')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Mach Number')
ax.set_title('Mach Number')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Step 5: Add Loading Parameters

Now let's add mass and energy loading:

```python
# Model with loading parameters
model_loaded = WindModel(
    SFR=1.0,
    v_circ=200.0,
    eta_M=0.1,        # 10% of SFR goes into hot wind
    eta_M_cold=1.0,   # 100% of SFR in cold clouds
    eta_E=1.0         # Energy loading factor
)

solution_loaded = model_loaded.run()

# Compare with original
plt.figure(figsize=(10, 6))
plt.plot(solution.r, solution.v, 'b-', label='Default', linewidth=2)
plt.plot(solution_loaded.r, solution_loaded.v, 'r--', 
         label='With loading', linewidth=2)
plt.xlabel('Radius [kpc]')
plt.ylabel('Velocity [km/s]')
plt.title('Effect of Loading Parameters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 6: Add Cold Clouds

Let's include the multiphase aspect:

```python
# Configure cloud distribution
config = WindConfig(
    N_cloud_species=10,    # Number of cloud mass bins
    M_cloud_min=10,        # Minimum cloud mass [Msun]
    M_cloud_max=1e6,       # Maximum cloud mass [Msun]
    cloud_alpha=2.0,       # Power-law slope
    f_turb0=0.1           # Turbulent mixing efficiency
)

# Create model with clouds
model_clouds = WindModel(
    config=config,
    SFR=1.0,
    v_circ=200.0,
    eta_M=0.1,
    eta_M_cold=1.0,
    eta_E=1.0
)

solution_clouds = model_clouds.run()

# Plot cloud velocities
plt.figure(figsize=(10, 6))
plt.plot(solution_clouds.r, solution_clouds.v, 'b-', 
         label='Hot wind', linewidth=2)

# Plot each cloud species
for i in range(config.N_cloud_species):
    plt.plot(solution_clouds.r, solution_clouds.v_cloud[i, :], 
             alpha=0.5, linewidth=1)

plt.xlabel('Radius [kpc]')
plt.ylabel('Velocity [km/s]')
plt.title('Hot Wind and Cloud Velocities')
plt.legend(['Hot wind'] + [f'Cloud {i+1}' for i in range(3)])
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 7: Calculate Observables

```python
# Calculate column density distribution
v_cloud, dN_dv = solution_clouds.calculate_column_density_distribution()

# Plot dN/dv
plt.figure(figsize=(10, 6))
plt.semilogy(v_cloud, dN_dv, 'b-', linewidth=2)
plt.xlabel('Velocity [km/s]')
plt.ylabel('dN/dv [cm⁻² / (km/s)]')
plt.title('Column Density Distribution')
plt.grid(True, alpha=0.3)
plt.xlim(0, 500)
plt.show()

# Calculate velocity moments
moments = solution_clouds.calculate_velocity_moments()
print(f"Mean velocity: {moments['mean']:.1f} km/s")
print(f"Velocity dispersion: {moments['dispersion']:.1f} km/s")
print(f"Skewness: {moments['skewness']:.2f}")
```

## Step 8: Compare Hot-Only vs Multiphase

```python
# Run hot-only version
solution_hot = model_clouds.run_hot_only()

# Compare
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Velocity comparison
ax = axes[0]
ax.plot(solution_hot.r, solution_hot.v, 'r-', 
        label='Hot only', linewidth=2)
ax.plot(solution_clouds.r, solution_clouds.v, 'b--', 
        label='Multiphase', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.set_title('Velocity Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Mass flux comparison
ax = axes[1]
ax.loglog(solution_hot.r, solution_hot.mass_flux, 'r-', 
          label='Hot only', linewidth=2)
ax.loglog(solution_clouds.r, solution_clouds.mass_flux, 'b--', 
          label='Multiphase', linewidth=2)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Mass Flux [Msun/yr]')
ax.set_title('Mass Flux Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print comparison
print(f"Terminal velocities:")
print(f"  Hot only: {solution_hot.v_terminal:.1f} km/s")
print(f"  Multiphase: {solution_clouds.v_terminal:.1f} km/s")
print(f"  Reduction: {(1 - solution_clouds.v_terminal/solution_hot.v_terminal)*100:.1f}%")
```

## Key Takeaways

1. **Basic models** are easy to create with just SFR and v_circ
2. **Loading parameters** (η_M, η_M_cold, η_E) control wind properties
3. **Cold clouds** reduce terminal velocity through drag
4. **Observables** can be calculated for comparison with data
5. **Hot-only comparison** shows the effect of multiphase physics

## Next Steps

- Try different galaxy parameters (dwarf, starburst)
- Explore parameter space systematically
- Compare with observational data
- Learn about [MCMC fitting](fitting_example.md)

## Exercises

1. **Dwarf Galaxy**: Create a model with SFR=0.1, v_circ=50 km/s
2. **Starburst**: Try SFR=100, v_circ=200 km/s
3. **Parameter Study**: Vary η_E from 0.1 to 3.0
4. **Cloud Distribution**: Change cloud_alpha and see the effect
5. **Convergence Test**: Compare rtol=1e-6 vs rtol=1e-10

## Common Issues

- **Wind fails to launch**: Increase η_E or decrease η_M_cold
- **Integration stops early**: Check for negative velocities
- **Slow execution**: Reduce N_cloud_species or relax tolerances

## See Also

- [Parameter Guide](../guide/parameters.md)
- [MCMC Fitting Tutorial](fitting_example.md)
- [API Reference](../api/wind_model.md)