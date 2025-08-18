# Comprehensive Report: New EMCEE Fitting Strategy for CLASSY Data

## Executive Summary
This report outlines a modern implementation strategy for fitting multiphase galactic wind models to CLASSY observations using the refactored `multiphasegalacticwind` package with emcee MCMC sampling.

## 1. Fitting Strategy Overview

### 1.1 Parameters to Fit
- **eta_M**: Hot phase mass loading factor (1e-4 - 1e2)
- **eta_M_cold**: Cold phase mass loading factor (1e-4 - 1e2)
- **eta_E**: Energy loading factor (1e-4 - 1)

### 1.2 Fixed Parameters
- **Cloud mass range**: M_min = 10^1 Msun, M_max = 10^6 Msun
- **Power-law slope**: α = 2.0
- **Cloud temperature**: T_cloud = 10^4 K
- **Number of cloud species**: 11 (logarithmically spaced)
- **All other cloud properties** from WindConfig defaults

### 1.3 Observable Targets
From CLASSY dN/dv profiles:
1. **Velocity centroid**: Mean outflow velocity (log scale)
2. **Velocity width**: HWHM or FWHM (log scale)
3. **Column density**: Total NH (log scale)

## 2. Implementation Architecture

### 2.1 Core Module Structure
```python
# fitting/classy_emcee_fit.py

import numpy as np
import emcee
from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.observables import calculate_column_density_distribution

class CLASSYFitter:
    def __init__(self, galaxy_data, n_cloud_species=10):
        self.galaxy_data = galaxy_data
        self.n_cloud_species = n_cloud_species
        self.setup_fixed_config()

    def setup_fixed_config(self):
        """Setup fixed configuration with cloud distribution"""
        M_min, M_max = 1e1, 1e6  # Msun
        alpha = 2.0
        self.M_cloud = np.logspace(np.log10(M_min), np.log10(M_max), self.n_cloud_species)
        self.base_config = WindConfig(
            T_cl=1e4,
            drag_coeff=0.5,
            f_turb0=0.1,
            mu=0.62
        )
```

### 2.2 Model Evaluation Pipeline
```python
def evaluate_model(self, eta_M, eta_M_cold, eta_E, sfr, v_circ, r50):
    """Run wind model and compute observables"""

    # Create model with parameters
    config = WindConfig(
        eta_M=eta_M,
        eta_M_cold=eta_M_cold,
        eta_E=eta_E,
        N_cloud_species=self.n_cloud_species,
        M_cloud_min=1e1,
        M_cloud_max=1e6,
        cloud_alpha=2.0,
        **self.base_config.__dict__
    )

    try:
        model = WindModel(
            SFR=sfr,
            v_circ=v_circ,
            r_50=r50,
            config=config
        )

        # Run integration
        solution = model.run()

        # Check for failed solutions
        if solution.status == -1:
            return None, None, None

        # Compute observables
        return self.compute_observables(model, solution)

    except Exception as e:
        return None, None, None
```

### 2.3 Observable Computation
```python
def compute_observables(self, model, solution):
    """Extract velocity centroid, width, and column density"""

    # Get dN/dv distribution
    v_bins = np.linspace(-800, 0, 41)  # 40 km/s bins
    dN_dv = calculate_column_density_distribution(
        model, solution, v_bins
    )

    # Compute moments
    v_mean = self.compute_velocity_centroid(v_bins, dN_dv)
    v_width = self.compute_velocity_width(v_bins, dN_dv, v_mean)
    NH_total = self.compute_total_column_density(dN_dv, v_bins)

    # Convert to log scale for comparison
    log_v = np.log10(abs(v_mean) * 1e5)  # cm/s
    log_width = np.log10(v_width * 1e5)   # cm/s
    log_NH = np.log10(NH_total)           # cm^-2

    return log_v, log_width, log_NH
```

## 3. EMCEE Implementation

### 3.1 Prior Function
```python
def log_prior(theta):
    """Uniform priors on fitting parameters"""
    eta_M, eta_M_cold, eta_E = theta

    # Physical bounds
    if not (0.01 <= eta_M <= 10):
        return -np.inf
    if not (0.01 <= eta_M_cold <= 50):
        return -np.inf
    if not (0.01 <= eta_E <= 10):
        return -np.inf

    # Additional constraint: eta_E typically < eta_M
    if eta_E > 2 * eta_M:
        return -np.inf

    return 0.0
```

### 3.2 Likelihood Function
```python
def log_likelihood(theta, observed, errors, galaxy_props):
    """Gaussian likelihood comparing model to observations"""

    # Unpack parameters
    eta_M, eta_M_cold, eta_E = theta
    v_obs, width_obs, NH_obs = observed
    v_err, width_err, NH_err = errors
    sfr, v_circ, r50 = galaxy_props

    # Evaluate model
    v_model, width_model, NH_model = evaluate_model(
        eta_M, eta_M_cold, eta_E, sfr, v_circ, r50
    )

    # Handle failed models
    if v_model is None:
        return -np.inf

    # Compute chi-squared
    chi2 = 0
    chi2 += ((v_model - v_obs) / v_err)**2
    chi2 += ((width_model - width_obs) / width_err)**2
    chi2 += ((NH_model - NH_obs) / NH_err)**2

    return -0.5 * chi2
```

### 3.3 MCMC Sampling
```python
def run_mcmc(galaxy_id, n_walkers=32, n_steps=5000):
    """Run MCMC for a single galaxy"""

    # Load galaxy data
    obs_data = load_galaxy_observations(galaxy_id)
    galaxy_props = load_galaxy_properties(galaxy_id)

    # Initialize walkers
    ndim = 3  # eta_M, eta_M_cold, eta_E
    initial = np.array([0.3, 1.0, 0.1])  # Initial guess
    pos = initial + 1e-2 * np.random.randn(n_walkers, ndim)

    # Setup sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability,
        args=(obs_data, galaxy_props)
    )

    # Run MCMC
    sampler.run_mcmc(pos, n_steps, progress=True)

    return sampler
```

## 4. Optimization Strategies

### 4.1 Performance Enhancements
1. **Parallel Evaluation**: Use multiprocessing pool for walker evaluations
2. **Caching**: Cache cooling function interpolators
3. **Adaptive Integration**: Use coarse tolerances for initial exploration
4. **Failed Model Handling**: Return -inf likelihood immediately

### 4.2 Numerical Stability
```python
class RobustWindModel(WindModel):
    """Enhanced model with better error handling"""

    def run(self, max_retries=3):
        """Run with automatic retry on stiff problems"""
        for attempt in range(max_retries):
            try:
                # Try with current tolerances
                solution = super().run()
                if solution.status != -1:
                    return solution

                # Adjust tolerances for retry
                self.config.rtol *= 10
                self.config.atol *= 10

            except ValueError as e:
                if "strictly increasing" in str(e):
                    # Return failed solution
                    return FailedSolution(self.r0, self.y0)
                raise

        return FailedSolution(self.r0, self.y0)
```

### 4.3 Convergence Diagnostics
```python
def check_convergence(sampler, threshold=50):
    """Check if chains have converged"""

    # Get autocorrelation time
    try:
        tau = sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * threshold < sampler.iteration)
        return converged, tau
    except:
        return False, None
```

## 5. Data Processing Pipeline

### 5.1 Galaxy Selection
```python
def select_valid_galaxies():
    """Select galaxies with complete observations"""

    valid_galaxies = []
    for galaxy in CLASSY_GALAXIES:
        if galaxy.has_valid_velocity_profile():
            if galaxy.has_galaxy_properties():
                valid_galaxies.append(galaxy)

    return valid_galaxies  # Expected: 42 out of 50
```

### 5.2 Batch Processing
```python
def fit_all_galaxies(galaxy_list, n_cores=4):
    """Fit all galaxies in parallel"""

    from multiprocessing import Pool

    with Pool(n_cores) as pool:
        results = pool.map(
            run_mcmc_single_galaxy,
            galaxy_list
        )

    return results
```

## 6. Output and Visualization

### 6.1 Results Storage
```python
def save_results(galaxy_id, sampler):
    """Save MCMC chains and derived parameters"""

    # Get chains
    chain = sampler.get_chain(discard=1000, flat=True)

    # Compute percentiles
    percentiles = np.percentile(chain, [16, 50, 84], axis=0)

    results = {
        'galaxy_id': galaxy_id,
        'chain': chain,
        'eta_M': percentiles[:, 0],
        'eta_M_cold': percentiles[:, 1],
        'eta_E': percentiles[:, 2],
        'acceptance_fraction': sampler.acceptance_fraction.mean()
    }

    np.save(f'results/{galaxy_id}_mcmc.npy', results)
```

### 6.2 Diagnostic Plots
```python
def create_diagnostic_plots(galaxy_id, sampler):
    """Generate corner plots and trace plots"""

    import corner
    import matplotlib.pyplot as plt

    # Corner plot
    chain = sampler.get_chain(discard=1000, flat=True)
    fig = corner.corner(
        chain,
        labels=[r'$\eta_M$', r'$\eta_{M,cold}$', r'$\eta_E$'],
        truths=None,
        quantiles=[0.16, 0.5, 0.84]
    )
    fig.savefig(f'plots/{galaxy_id}_corner.pdf')

    # Trace plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):
        axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
    fig.savefig(f'plots/{galaxy_id}_trace.pdf')
```

## 7. Expected Challenges and Solutions

### 7.1 Numerical Stiffness
**Challenge**: Some parameter combinations cause integration failures
**Solution**:
- Implement graceful failure handling
- Use adaptive tolerance adjustment
- Return -inf likelihood for failed models

### 7.2 Computational Cost
**Challenge**: Each model evaluation takes ~0.1-1 second
**Solution**:
- Parallelize walker evaluations
- Use coarse tolerances for burn-in
- Implement intelligent starting positions

### 7.3 Parameter Degeneracies
**Challenge**: eta_M and eta_M_cold may be degenerate
**Solution**:
- Use informative priors based on physical expectations
- Consider fitting eta_M_total = eta_M + eta_M_cold
- Analyze posterior correlations

## 8. Implementation Timeline

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create CLASSYFitter class
- [ ] Implement model evaluation pipeline
- [ ] Add observable computation functions
- [ ] Test on single galaxy

### Phase 2: MCMC Implementation (Week 2)
- [ ] Implement prior and likelihood functions
- [ ] Setup emcee sampler
- [ ] Add convergence diagnostics
- [ ] Test on subset of galaxies

### Phase 3: Production Runs (Week 3-4)
- [ ] Run fits for all 42 valid galaxies
- [ ] Generate diagnostic plots
- [ ] Analyze parameter distributions
- [ ] Compare with previous results

### Phase 4: Analysis and Refinement (Week 5)
- [ ] Identify problematic galaxies
- [ ] Refine priors if needed
- [ ] Systematic uncertainty analysis
- [ ] Prepare publication-ready plots

## 9. Validation Strategy

### 9.1 Sanity Checks
1. **Parameter Recovery**: Test on synthetic data with known parameters
2. **Consistency**: Compare with previous fitting results where available
3. **Physical Bounds**: Ensure all solutions are physically reasonable

### 9.2 Cross-Validation
```python
def cross_validate(galaxy_subset):
    """Leave-one-out cross validation"""

    for test_galaxy in galaxy_subset:
        # Train on others
        training_set = [g for g in galaxy_subset if g != test_galaxy]
        prior_params = compute_prior_from_training(training_set)

        # Test on held-out
        test_results = fit_with_prior(test_galaxy, prior_params)

        # Compare with standard fit
        standard_results = fit_with_uniform_prior(test_galaxy)
```

## 10. Key Advantages of New Approach

1. **Modern Codebase**: Uses refactored, well-tested wind model
2. **Robust Error Handling**: Graceful failure for stiff parameters
3. **Efficient Implementation**: Optimized cooling functions, parallelization
4. **Clean Architecture**: Separation of concerns, configuration management
5. **Reproducibility**: Fixed random seeds, versioned dependencies
6. **Extensibility**: Easy to add new observables or modify fitting strategy

## 11. Success Metrics

- **Convergence Rate**: >90% of galaxies should converge
- **Parameter Constraints**: Typical uncertainties <50%
- **Computational Time**: <1 hour per galaxy on 4 cores
- **Physical Consistency**: All solutions satisfy energy/momentum conservation

## 12. Next Steps

1. **Immediate**: Implement core CLASSYFitter class
2. **Short-term**: Test on 3-5 representative galaxies
3. **Medium-term**: Full production runs on all galaxies
4. **Long-term**: Extend to other surveys (e.g., COS-Halos)

This comprehensive strategy provides a clear path forward for fitting the CLASSY data with the new multiphase wind model codebase, addressing the limitations of the previous approach while leveraging modern computational techniques.