# Model Emulator and Grid Interpolation Plan

## Motivation

MCMC fitting of multiphase wind models is computationally expensive because:
1. Each model evaluation requires solving stiff ODEs (5-60 seconds per model)
2. MCMC requires ~10^5-10^6 model evaluations for convergence
3. Parameter space has strong degeneracies between η_E, η_M, and η_M_cold
4. Poor initial guesses lead to inefficient sampling and long burn-in periods

## Proposed Solution: Grid-Based Emulator

### Phase 1: Parameter Grid Generation

Create a dense grid of pre-computed models:

```python
# Parameter ranges (log space)
log_eta_M_grid = np.linspace(-1.5, 0.5, 20)      # 20 points
log_eta_M_cold_grid = np.linspace(-1.0, 1.0, 20) # 20 points  
log_eta_E_grid = np.linspace(-1.5, 0.7, 20)      # 20 points
# Total: 8000 models

# Additional parameters to vary
v_circ_grid = [50, 100, 200, 300]  # 4 values
SFR_grid = [0.1, 1.0, 10.0, 100.0] # 4 values
# Total: 128,000 models if fully crossed
```

### Phase 2: Model Computation

```python
def compute_model_grid():
    """
    Pre-compute models on parameter grid.
    Store key observables:
    - Terminal velocity
    - Column density distribution (compressed)
    - Mass loading at 10 kpc
    - Velocity moments
    """
    grid_results = {}
    
    for params in parameter_combinations:
        model = WindModel(**params)
        solution = model.run()
        
        # Store compressed results
        grid_results[params] = {
            'v_terminal': solution.v_terminal,
            'dN_dv_spline': compress_column_density(solution),
            'mass_loading': solution.mass_loading_at_10kpc,
            'v_mean': solution.v_mean,
            'v_sigma': solution.v_dispersion
        }
    
    return grid_results
```

### Phase 3: Interpolator/Emulator Construction

#### Option A: Multi-dimensional Interpolation
```python
from scipy.interpolate import RegularGridInterpolator

# Create interpolator for each observable
v_terminal_interpolator = RegularGridInterpolator(
    (log_eta_M_grid, log_eta_M_cold_grid, log_eta_E_grid),
    v_terminal_grid,
    method='linear',  # or 'cubic' for smoother
    bounds_error=False,
    fill_value=None
)
```

#### Option B: Gaussian Process Emulator
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# Train GP on grid points
kernel = RBF(length_scale=0.3) * Matern(nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(parameter_samples, observable_values)

# Fast prediction with uncertainty
mean, std = gp.predict(new_params, return_std=True)
```

#### Option C: Neural Network Emulator
```python
import torch.nn as nn

class WindEmulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 128),  # 5 params in
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # 50 observables out
        )
    
    def forward(self, params):
        return self.network(params)
```

### Phase 4: MCMC Acceleration

#### 1. Better Initial Conditions
```python
def find_best_initial_guess(observed_data, emulator):
    """
    Use emulator to quickly find good starting point.
    """
    # Coarse grid search using emulator (fast!)
    chi2_grid = np.zeros((10, 10, 10))
    
    for i, eta_M in enumerate(coarse_grid_eta_M):
        for j, eta_M_cold in enumerate(coarse_grid_eta_M_cold):
            for k, eta_E in enumerate(coarse_grid_eta_E):
                predicted = emulator.predict([eta_M, eta_M_cold, eta_E])
                chi2_grid[i,j,k] = compute_chi2(predicted, observed_data)
    
    # Find minimum
    best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    return coarse_grid[best_idx]
```

#### 2. Proposal Function Guidance
```python
def guided_proposal(current_params, emulator, step_size=0.1):
    """
    Use emulator gradient to guide MCMC proposals.
    """
    # Estimate local gradient using emulator
    gradient = emulator.estimate_gradient(current_params)
    
    # Bias proposal in favorable direction
    proposal = current_params + step_size * (
        np.random.randn(3) + 0.3 * gradient / np.linalg.norm(gradient)
    )
    return proposal
```

#### 3. Likelihood Approximation for Burn-in
```python
def hybrid_mcmc(n_burn_emulated=1000, n_burn_exact=500, n_production=5000):
    """
    Use emulator for initial burn-in, then switch to exact model.
    """
    # Phase 1: Fast burn-in with emulator
    for i in range(n_burn_emulated):
        proposal = propose_step()
        emulated_likelihood = emulator.predict_likelihood(proposal)
        # Accept/reject based on emulated likelihood
    
    # Phase 2: Refined burn-in with exact model
    for i in range(n_burn_exact):
        proposal = propose_step()
        exact_likelihood = run_exact_model(proposal)
        # Accept/reject based on exact likelihood
    
    # Phase 3: Production with exact model
    # ...
```

## Demonstrating Parameter Degeneracies

### Visualization Tools

```python
def plot_degeneracy_surfaces(emulator):
    """
    Create 2D slices showing parameter degeneracies.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Fix v_terminal = 300 km/s
    target_v = 300
    
    # eta_M vs eta_E slice (fix eta_M_cold)
    ax = axes[0, 0]
    eta_M_vals = np.logspace(-1.5, 0.5, 100)
    eta_E_vals = np.logspace(-1.5, 0.7, 100)
    
    for eta_M in eta_M_vals:
        for eta_E in eta_E_vals:
            v_pred = emulator.predict([eta_M, 0.5, eta_E])['v_terminal']
            if abs(v_pred - target_v) < 10:  # Within 10 km/s
                ax.plot(eta_M, eta_E, 'r.', alpha=0.5)
    
    ax.set_xlabel('η_M')
    ax.set_ylabel('η_E')
    ax.set_title('Degeneracy: v_terminal = 300 km/s')
    # ... repeat for other parameter pairs
```

### Degeneracy Quantification

```python
def compute_degeneracy_metric(emulator, observable='v_terminal'):
    """
    Quantify parameter degeneracy using Fisher matrix.
    """
    # Compute Fisher information matrix
    fisher = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            # Numerical derivative
            fisher[i, j] = compute_fisher_element(
                emulator, param_i=i, param_j=j, observable=observable
            )
    
    # Eigenvalues indicate degeneracy directions
    eigenvals, eigenvecs = np.linalg.eig(fisher)
    
    # Small eigenvalue = strong degeneracy
    degeneracy_strength = min(eigenvals) / max(eigenvals)
    
    return degeneracy_strength, eigenvecs
```

## Implementation Timeline

### Step 1: Grid Generation Script (Week 1)
```python
# generate_grid.py
if __name__ == "__main__":
    # Define parameter grid
    params = create_parameter_grid()
    
    # Parallel computation
    with Pool(processes=20) as pool:
        results = pool.map(compute_model, params)
    
    # Save to HDF5
    save_grid_results('wind_model_grid.h5', results)
```

### Step 2: Emulator Training (Week 2)
- Train multiple emulator types
- Cross-validate performance
- Select best approach

### Step 3: MCMC Integration (Week 3)
- Modify existing MCMC code
- Add emulator-guided initialization
- Implement hybrid sampling

### Step 4: Validation (Week 4)
- Compare emulated vs exact MCMC
- Quantify speedup
- Verify posterior accuracy

## Expected Benefits

1. **Speed**: 100-1000x faster initial parameter exploration
2. **Convergence**: Better starting points reduce burn-in by ~50%
3. **Understanding**: Visualize full parameter degeneracies
4. **Reliability**: Identify problematic parameter regions before MCMC
5. **Reusability**: Emulator can be shared with community

## Storage Requirements

- Full grid (128k models): ~10 GB compressed
- Reduced grid (8k models): ~1 GB compressed
- Trained emulator: ~100 MB
- Can be hosted on Zenodo for community use

## Code Example: Quick Demo

```python
# Quick demonstration with coarse grid
from multiphasegalacticwind import WindModel
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle

# Small test grid
eta_M_vals = np.logspace(-1, 0, 5)
eta_E_vals = np.logspace(-0.5, 0.5, 5)
v_terminal_grid = np.zeros((5, 5))

# Compute grid (simplified, 2D)
for i, eta_M in enumerate(eta_M_vals):
    for j, eta_E in enumerate(eta_E_vals):
        model = WindModel(SFR=10, v_circ=200, 
                         eta_M=eta_M, eta_E=eta_E)
        solution = model.run()
        v_terminal_grid[i, j] = solution.v_terminal

# Create interpolator
interp = RegularGridInterpolator(
    (np.log10(eta_M_vals), np.log10(eta_E_vals)),
    v_terminal_grid
)

# Fast prediction at any point
test_point = [np.log10(0.3), np.log10(1.5)]
v_predicted = interp(test_point)
print(f"Predicted v_terminal: {v_predicted:.1f} km/s")

# Show degeneracy contours
import matplotlib.pyplot as plt
plt.contour(eta_M_vals, eta_E_vals, v_terminal_grid.T, 
           levels=[200, 300, 400, 500])
plt.xlabel('η_M')
plt.ylabel('η_E')
plt.title('Terminal Velocity Degeneracy')
plt.show()
```

## References

- Heitmann et al. (2009) - "The Coyote Universe" (cosmological emulators)
- Jennings et al. (2016) - "bacco: Bayesian emulator for galaxy clustering"
- McClintock et al. (2019) - "Aemulus Project: Emulating halo mass functions"
- Euclid Collaboration (2019) - "Euclid preparation: Emulators for cosmology"

## Next Steps

1. Implement grid generation script
2. Test different emulator architectures
3. Validate against exact MCMC
4. Document degeneracy patterns
5. Publish emulator for community use