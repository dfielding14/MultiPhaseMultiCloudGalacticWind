# MCMC/EMCEE Fitting Documentation Plan

## Overview
This plan outlines the comprehensive documentation for the MCMC fitting functionality using emcee to fit multiphase wind models to observational data from the CLASSY survey.

## Documentation Structure

### 1. Theory & Background (docs/fitting/theory.md)
- **Bayesian Inference for Wind Models**
  - Prior distributions for η_M, η_M_cold, η_E
  - Likelihood function formulation
  - Posterior sampling strategy
- **Observable Quantities**
  - Column density distributions dN/dv
  - Velocity centroids and widths
  - Total column densities
- **Parameter Degeneracies**
  - η_M vs η_E trade-offs
  - Mass loading correlations
  - Resolution strategies

### 2. API Documentation (docs/api/fitting.md)
- **CLASSYFitter Class**
  - Constructor parameters
  - Methods: evaluate_model, log_likelihood, log_prior
  - Configuration options
- **MCMCSampler Class**
  - Initialization with data
  - run_mcmc method
  - Convergence diagnostics
- **Utility Functions**
  - Data preprocessing
  - Result visualization
  - Chain analysis

### 3. User Guide (docs/guide/mcmc_fitting.md)
- **Quick Start**
  - Basic fitting example
  - Required data format
  - Interpreting results
- **Advanced Usage**
  - Custom priors
  - Multi-ion fitting
  - Parallel processing
- **Best Practices**
  - Chain initialization
  - Burn-in determination
  - Convergence checks

### 4. Examples (docs/examples/fitting/)
- **Single Galaxy Fit** (single_galaxy.py)
  - J0021+0052 complete example
  - Corner plots
  - Model vs data comparison
- **Batch Processing** (batch_fitting.py)
  - Multiple galaxies
  - Result aggregation
  - Statistical analysis
- **Custom Likelihood** (custom_likelihood.py)
  - Implementing new observables
  - Weighted likelihoods
  - Joint constraints

### 5. Technical Details (docs/fitting/technical.md)
- **Implementation Notes**
  - emcee configuration
  - Parallelization strategy
  - Memory management
- **Performance Optimization**
  - Model caching
  - Vectorization
  - GPU acceleration (future)
- **Known Issues & Limitations**
  - Convergence challenges
  - Parameter boundaries
  - Numerical stability

## Key Components to Document

### Data Format
```python
galaxy_data = {
    'name': 'J0021+0052',
    'sfr': 3.0,  # Msun/yr
    'v_circ': 69.0,  # km/s
    'r50': 1.13,  # kpc
    'ions': {
        'SiII': {
            'v': np.array([...]),  # km/s
            'N': np.array([...]),  # cm^-2/(km/s)
            'N_err': np.array([...])
        },
        'CII': {...}
    }
}
```

### Likelihood Function
```python
def log_likelihood(theta, data):
    eta_M, eta_M_cold, eta_E = theta
    model_v, model_N = run_model(eta_M, eta_M_cold, eta_E)
    chi2 = np.sum(((data['N'] - model_N) / data['N_err'])**2)
    return -0.5 * chi2
```

### Prior Distributions
```python
def log_prior(theta):
    eta_M, eta_M_cold, eta_E = theta
    if 0.001 < eta_M < 10 and 0.01 < eta_M_cold < 100 and 0.01 < eta_E < 10:
        # Log-uniform priors
        return -np.log(eta_M) - np.log(eta_M_cold) - np.log(eta_E)
    return -np.inf
```

## Implementation Priority

### Phase 1: Core Documentation (Immediate)
1. Extract equations from core_physics.py
2. Create missing documentation pages (parameters.md, config.md, etc.)
3. Fix math rendering issues

### Phase 2: MCMC Documentation (High Priority)
1. Theory and mathematical framework
2. API reference for CLASSYFitter
3. Basic fitting example

### Phase 3: Advanced Topics (Medium Priority)
1. Multi-ion fitting strategies
2. Convergence diagnostics
3. Performance optimization

### Phase 4: Case Studies (Lower Priority)
1. Full CLASSY sample analysis
2. Comparison with other models
3. Statistical insights

## Testing Requirements
- Validate all code examples
- Test parallel execution
- Benchmark performance
- Verify convergence metrics

## Resources Needed
- Access to CLASSY observation files
- Example chains from production runs
- Performance benchmarks
- Literature references

## Success Metrics
- All fitting functions documented
- Working examples for each use case
- Performance benchmarks included
- Troubleshooting guide complete
- Mathematical framework clearly explained