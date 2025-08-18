# Tutorial: Parameter Studies

## Introduction

This tutorial shows how to systematically explore parameter space to understand how different parameters affect wind properties.

## Basic Parameter Scan

### Single Parameter Variation

```python
import numpy as np
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel

# Vary energy loading
eta_E_values = np.logspace(-1, 0.5, 10)  # 0.1 to ~3
results = []

for eta_E in eta_E_values:
    model = WindModel(
        SFR=10.0,
        v_circ=200.0,
        eta_M=0.1,
        eta_M_cold=1.0,
        eta_E=eta_E,
        rtol=1e-6  # Relaxed for speed
    )
    
    try:
        solution = model.run()
        results.append({
            'eta_E': eta_E,
            'v_terminal': solution.v_terminal,
            'v_at_10kpc': solution.v_at_10kpc,
            'mass_loading': solution.mass_loading_at_10kpc
        })
        print(f"η_E = {eta_E:.2f}: v_term = {solution.v_terminal:.0f} km/s")
    except Exception as e:
        print(f"η_E = {eta_E:.2f}: Failed - {e}")
        results.append(None)
```

### Visualize Results

```python
# Extract successful runs
valid_results = [r for r in results if r is not None]

# Plot terminal velocity vs eta_E
plt.figure(figsize=(10, 6))
eta_E_vals = [r['eta_E'] for r in valid_results]
v_term_vals = [r['v_terminal'] for r in valid_results]

plt.semilogx(eta_E_vals, v_term_vals, 'bo-', linewidth=2, markersize=8)
plt.xlabel(r'$\eta_E$ (Energy Loading)', fontsize=12)
plt.ylabel('Terminal Velocity [km/s]', fontsize=12)
plt.title('Wind Velocity vs Energy Loading', fontsize=14)
plt.grid(True, alpha=0.3)

# Add scaling relation
eta_E_theory = np.logspace(-1, 0.5, 50)
v_theory = 3 * 200 * np.sqrt(eta_E_theory)  # Theoretical scaling
plt.plot(eta_E_theory, v_theory, 'r--', alpha=0.5, 
         label=r'$v \propto \sqrt{\eta_E}$')
plt.legend()
plt.show()
```

## 2D Parameter Grid

### Mass and Energy Loading

```python
from itertools import product

# Define parameter grid
eta_M_values = np.logspace(-2, 0, 5)     # 0.01 to 1
eta_E_values = np.logspace(-1, 0.5, 5)   # 0.1 to ~3

# Run grid
results_grid = {}
for eta_M, eta_E in product(eta_M_values, eta_E_values):
    print(f"Running η_M={eta_M:.3f}, η_E={eta_E:.2f}")
    
    model = WindModel(
        SFR=10.0,
        v_circ=200.0,
        eta_M=eta_M,
        eta_M_cold=1.0,
        eta_E=eta_E,
        rtol=1e-6
    )
    
    try:
        solution = model.run()
        results_grid[(eta_M, eta_E)] = {
            'v_terminal': solution.v_terminal,
            'mass_flux': solution.mass_flux[-1],
            'momentum_flux': solution.momentum_flux[-1]
        }
    except:
        results_grid[(eta_M, eta_E)] = None
```

### Create Heatmap

```python
# Prepare data for heatmap
v_terminal_grid = np.zeros((len(eta_M_values), len(eta_E_values)))

for i, eta_M in enumerate(eta_M_values):
    for j, eta_E in enumerate(eta_E_values):
        result = results_grid.get((eta_M, eta_E))
        if result:
            v_terminal_grid[i, j] = result['v_terminal']
        else:
            v_terminal_grid[i, j] = np.nan

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(v_terminal_grid, origin='lower', aspect='auto', 
           cmap='viridis', interpolation='nearest')
plt.colorbar(label='Terminal Velocity [km/s]')

# Add contours
X, Y = np.meshgrid(range(len(eta_E_values)), range(len(eta_M_values)))
CS = plt.contour(X, Y, v_terminal_grid, colors='white', alpha=0.5)
plt.clabel(CS, inline=True, fontsize=10)

# Labels
plt.xticks(range(len(eta_E_values)), 
           [f'{x:.2f}' for x in eta_E_values])
plt.yticks(range(len(eta_M_values)), 
           [f'{x:.3f}' for x in eta_M_values])
plt.xlabel(r'$\eta_E$', fontsize=12)
plt.ylabel(r'$\eta_M$', fontsize=12)
plt.title('Terminal Velocity Parameter Space', fontsize=14)
plt.show()
```

## Latin Hypercube Sampling

### Efficient Parameter Exploration

```python
from scipy.stats import qmc

# Define bounds (in log space)
param_bounds = [
    [-2, 0],    # log10(eta_M): 0.01 to 1
    [-1, 1],    # log10(eta_M_cold): 0.1 to 10
    [-1, 0.5]   # log10(eta_E): 0.1 to ~3
]

# Generate Latin Hypercube samples
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(n=50)
scaled_sample = qmc.scale(sample, 
                         [b[0] for b in param_bounds],
                         [b[1] for b in param_bounds])

# Convert from log space
parameters = 10**scaled_sample

# Run models
lhs_results = []
for i, (eta_M, eta_M_cold, eta_E) in enumerate(parameters):
    print(f"Sample {i+1}/50: η_M={eta_M:.3f}, η_M_cold={eta_M_cold:.2f}, η_E={eta_E:.2f}")
    
    model = WindModel(
        SFR=10.0,
        v_circ=200.0,
        eta_M=eta_M,
        eta_M_cold=eta_M_cold,
        eta_E=eta_E,
        rtol=1e-6
    )
    
    try:
        solution = model.run()
        lhs_results.append({
            'eta_M': eta_M,
            'eta_M_cold': eta_M_cold,
            'eta_E': eta_E,
            'v_terminal': solution.v_terminal,
            'mass_loading': solution.mass_loading_at_10kpc
        })
    except:
        lhs_results.append(None)
```

### Analyze Correlations

```python
# Extract successful runs
valid_lhs = [r for r in lhs_results if r is not None]

# Create correlation matrix plot
import pandas as pd

df = pd.DataFrame(valid_lhs)
correlation_matrix = df.corr()

# Plot correlation matrix manually
plt.figure(figsize=(10, 8))
im = plt.imshow(correlation_matrix, cmap='coolwarm', 
                aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)

# Add labels
params = list(df.columns)
plt.xticks(range(len(params)), params, rotation=45, ha='right')
plt.yticks(range(len(params)), params)

# Add correlation values as text
for i in range(len(params)):
    for j in range(len(params)):
        text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black')

plt.title('Parameter Correlations')
plt.tight_layout()
plt.show()

# Scatter plot matrix
pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(12, 12), 
                          diagonal='kde')
plt.suptitle('Parameter Relationships', y=0.995)
plt.show()
```

## Galaxy Scaling Relations

### SFR and Circular Velocity

```python
# Galaxy sample
galaxies = [
    {'name': 'Dwarf', 'SFR': 0.01, 'v_circ': 30},
    {'name': 'LMC', 'SFR': 0.3, 'v_circ': 70},
    {'name': 'Milky Way', 'SFR': 1.0, 'v_circ': 200},
    {'name': 'Starburst', 'SFR': 10.0, 'v_circ': 150},
    {'name': 'ULIRG', 'SFR': 100.0, 'v_circ': 250}
]

# Fixed loading parameters
eta_M = 0.1
eta_M_cold = 1.0
eta_E = 1.0

# Run models
galaxy_results = []
for gal in galaxies:
    model = WindModel(
        SFR=gal['SFR'],
        v_circ=gal['v_circ'],
        eta_M=eta_M,
        eta_M_cold=eta_M_cold,
        eta_E=eta_E
    )
    
    solution = model.run()
    galaxy_results.append({
        'name': gal['name'],
        'SFR': gal['SFR'],
        'v_circ': gal['v_circ'],
        'v_terminal': solution.v_terminal,
        'mass_flux': solution.mass_flux[-1]
    })

# Plot scaling
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Terminal velocity vs v_circ
ax = axes[0]
v_circ_vals = [r['v_circ'] for r in galaxy_results]
v_term_vals = [r['v_terminal'] for r in galaxy_results]

ax.loglog(v_circ_vals, v_term_vals, 'bo', markersize=10)
for r in galaxy_results:
    ax.annotate(r['name'], (r['v_circ'], r['v_terminal']),
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel(r'$v_{circ}$ [km/s]', fontsize=12)
ax.set_ylabel(r'$v_{terminal}$ [km/s]', fontsize=12)
ax.set_title('Wind Velocity Scaling', fontsize=14)
ax.grid(True, alpha=0.3)

# Mass loading vs SFR
ax = axes[1]
sfr_vals = [r['SFR'] for r in galaxy_results]
loading_vals = [r['mass_flux']/r['SFR'] for r in galaxy_results]

ax.loglog(sfr_vals, loading_vals, 'ro', markersize=10)
for r in galaxy_results:
    ax.annotate(r['name'], (r['SFR'], r['mass_flux']/r['SFR']),
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('SFR [Msun/yr]', fontsize=12)
ax.set_ylabel('Mass Loading Factor', fontsize=12)
ax.set_title('Mass Loading vs SFR', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Time Evolution Study

### Track Cloud Evolution

```python
# Model with detailed cloud tracking
from multiphasegalacticwind import WindConfig

config = WindConfig(N_cloud_species=5)
model = WindModel(
    config=config,
    SFR=10.0,
    v_circ=200.0,
    eta_M=0.1,
    eta_M_cold=1.0,
    eta_E=1.0
)

solution = model.run()

# Plot cloud mass evolution
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Cloud masses
ax = axes[0]
for i in range(config.N_cloud_species):
    ax.loglog(solution.r, solution.M_cloud[i, :], 
              label=f'Species {i+1}')
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Cloud Mass [Msun]')
ax.set_title('Cloud Mass Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# Cloud velocities
ax = axes[1]
ax.plot(solution.r, solution.v, 'k-', linewidth=2, label='Hot wind')
for i in range(config.N_cloud_species):
    ax.plot(solution.r, solution.v_cloud[i, :], '--', alpha=0.7)
ax.set_xlabel('Radius [kpc]')
ax.set_ylabel('Velocity [km/s]')
ax.set_title('Velocity Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Optimization Study

### Find Optimal Parameters

```python
from scipy.optimize import differential_evolution

# Define objective function
def objective(params):
    """Minimize difference from target velocity."""
    eta_M, eta_M_cold, eta_E = params
    target_velocity = 300  # km/s
    
    try:
        model = WindModel(
            SFR=10.0,
            v_circ=200.0,
            eta_M=eta_M,
            eta_M_cold=eta_M_cold,
            eta_E=eta_E,
            rtol=1e-6
        )
        solution = model.run()
        
        # Return squared difference from target
        return (solution.v_terminal - target_velocity)**2
    except:
        return 1e10  # Penalty for failed models

# Bounds
bounds = [(0.01, 1), (0.1, 10), (0.1, 3)]

# Run optimization
result = differential_evolution(objective, bounds, seed=42, 
                               maxiter=50, popsize=10)

print(f"Optimal parameters for v_terminal = 300 km/s:")
print(f"η_M = {result.x[0]:.3f}")
print(f"η_M_cold = {result.x[1]:.2f}")
print(f"η_E = {result.x[2]:.2f}")
print(f"Achieved velocity = {np.sqrt(result.fun):.1f} km/s")
```

## Key Insights

1. **Energy loading** has strongest effect on terminal velocity
2. **Mass loading** affects density but not velocity much
3. **Cold mass loading** reduces velocity through drag
4. **Parameter degeneracies** exist - multiple combinations give similar results
5. **Galaxy mass** (v_circ) sets overall scale

## Exercises

1. **3D parameter space**: Add f_turb0 as third dimension
2. **Observable constraints**: Find parameters matching observed N(v)
3. **Computational study**: How does N_cloud_species affect results?
4. **Extreme parameters**: Explore failure modes
5. **Bayesian optimization**: Use GPs for efficient exploration

## See Also

- [MCMC Fitting](fitting_example.md) - Bayesian parameter constraints
- [Parameter Guide](../guide/parameters.md) - Parameter descriptions
- [Visualization](plotting.md) - Advanced plotting techniques