# MCMC Fitting Guide

## Overview

This guide explains how to fit multiphase wind models to observational data using MCMC (Markov Chain Monte Carlo) methods with the `emcee` package. The primary application is fitting CLASSY galaxy observations to constrain wind parameters (η_M, η_M_cold, η_E).

## Theory

### Bayesian Framework

We use Bayesian inference to constrain model parameters:

$$P(\theta | D) \propto P(D | \theta) \cdot P(\theta)$$

where:
- $\theta$ = model parameters (η_M, η_M_cold, η_E)
- $D$ = observational data (column densities, velocities)
- $P(D|\theta)$ = likelihood
- $P(\theta)$ = prior
- $P(\theta|D)$ = posterior

### Likelihood Function

For column density distributions:

$$\ln \mathcal{L} = -\frac{1}{2} \sum_i \left( \frac{N_i^{obs} - N_i^{model}(\theta)}{\sigma_i} \right)^2$$

For velocity centroids and widths:

$$\ln \mathcal{L} = -\frac{1}{2} \left[ \left(\frac{v_{obs} - v_{model}}{\sigma_v}\right)^2 + \left(\frac{w_{obs} - w_{model}}{\sigma_w}\right)^2 \right]$$

### Prior Distributions

We use log-uniform priors for scale parameters:

$$P(\eta_M) \propto \frac{1}{\eta_M}, \quad 0.001 < \eta_M < 10$$
$$P(\eta_{M,cold}) \propto \frac{1}{\eta_{M,cold}}, \quad 0.01 < \eta_{M,cold} < 100$$
$$P(\eta_E) \propto \frac{1}{\eta_E}, \quad 0.01 < \eta_E < 10$$

## Basic Implementation

### Step 1: Prepare Observational Data

```python
import numpy as np

# Example: J0021+0052 from CLASSY
galaxy_data = {
    'name': 'J0021+0052',
    'sfr': 3.0,           # Msun/yr
    'v_circ': 69.0,       # km/s
    'r50': 1.13,          # kpc
    'ions': {
        'SiII': {
            'v': np.array([-200, -100, 0, 100, 200]),  # km/s
            'N': np.array([1e13, 5e13, 1e14, 5e13, 1e13]),  # cm^-2/(km/s)
            'N_err': np.array([1e12, 5e12, 1e13, 5e12, 1e12])
        },
        'CII': {
            'v': np.array([-150, -50, 50, 150]),
            'N': np.array([2e13, 8e13, 8e13, 2e13]),
            'N_err': np.array([2e12, 8e12, 8e12, 2e12])
        }
    }
}
```

### Step 2: Define Model Function

```python
from multiphasegalacticwind import WindModel, WindConfig

def run_model(theta, galaxy_data):
    """
    Run wind model with given parameters.
    
    Parameters
    ----------
    theta : array
        [eta_M, eta_M_cold, eta_E]
    galaxy_data : dict
        Galaxy properties and observations
        
    Returns
    -------
    dict
        Model predictions for each ion
    """
    eta_M, eta_M_cold, eta_E = theta
    
    # Configure model
    config = WindConfig(
        N_cloud_species=10,
        f_turb0=0.1,
        T_cl=1e4
    )
    
    try:
        # Run model
        model = WindModel(
            config=config,
            SFR=galaxy_data['sfr'],
            v_circ=galaxy_data['v_circ'],
            eta_M=eta_M,
            eta_M_cold=eta_M_cold,
            eta_E=eta_E,
            rtol=1e-6,  # Relaxed for MCMC
            atol=1e-8
        )
        
        solution = model.run()
        
        # Calculate observables
        v_cloud, dN_dv = solution.calculate_column_density_distribution()
        
        # Extract predictions for each ion
        # (Simplified - real implementation maps ions to cloud species)
        predictions = {}
        for ion in galaxy_data['ions']:
            predictions[ion] = {
                'v': v_cloud,
                'N': dN_dv  # Would be ion-specific in reality
            }
            
        return predictions
        
    except Exception:
        return None
```

### Step 3: Define Likelihood

```python
def log_likelihood(theta, galaxy_data):
    """
    Calculate log-likelihood for parameters.
    """
    # Run model
    predictions = run_model(theta, galaxy_data)
    
    if predictions is None:
        return -np.inf
    
    # Calculate chi-squared for each ion
    log_L = 0.0
    
    for ion, obs in galaxy_data['ions'].items():
        pred = predictions[ion]
        
        # Interpolate model to observation velocities
        N_model = np.interp(obs['v'], pred['v'], pred['N'])
        
        # Chi-squared
        chi2 = np.sum(((obs['N'] - N_model) / obs['N_err'])**2)
        log_L -= 0.5 * chi2
    
    return log_L
```

### Step 4: Define Prior

```python
def log_prior(theta):
    """
    Log-prior for parameters.
    """
    eta_M, eta_M_cold, eta_E = theta
    
    # Bounds
    if not (0.001 < eta_M < 10):
        return -np.inf
    if not (0.01 < eta_M_cold < 100):
        return -np.inf
    if not (0.01 < eta_E < 10):
        return -np.inf
    
    # Log-uniform priors
    return -np.log(eta_M) - np.log(eta_M_cold) - np.log(eta_E)
```

### Step 5: Define Posterior

```python
def log_posterior(theta, galaxy_data):
    """
    Log-posterior = log-prior + log-likelihood
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, galaxy_data)
```

### Step 6: Run MCMC

```python
import emcee
from multiprocessing import Pool

# Setup
ndim = 3  # Number of parameters
nwalkers = 32  # Number of walkers
nsteps = 5000  # Number of steps

# Initialize walkers
# Start near reasonable values with small scatter
initial = np.array([0.1, 1.0, 1.0])  # eta_M, eta_M_cold, eta_E
pos = initial + 0.1 * np.random.randn(nwalkers, ndim)

# Run MCMC with parallel processing
with Pool() as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(galaxy_data,),
        pool=pool
    )
    
    # Run burn-in
    print("Running burn-in...")
    pos, _, _ = sampler.run_mcmc(pos, 500, progress=True)
    sampler.reset()
    
    # Run production
    print("Running production...")
    sampler.run_mcmc(pos, nsteps, progress=True)

# Save chain
chain = sampler.get_chain()
log_prob = sampler.get_log_prob()
```

## Advanced Topics

### Parallel Tempering

For multimodal posteriors:

```python
from emcee import PTSampler

# Temperature ladder
ntemps = 8
betas = np.logspace(0, -3, ntemps)

# Initialize
sampler = PTSampler(
    ntemps, nwalkers, ndim,
    log_likelihood, log_prior,
    loglargs=(galaxy_data,),
    betas=betas
)

# Run
sampler.run_mcmc(pos, nsteps)
```

### Multi-Ion Fitting

Simultaneously fit multiple ions with different cloud associations:

```python
def multi_ion_likelihood(theta, galaxy_data):
    """
    Fit multiple ions with ion-specific cloud mappings.
    """
    eta_M, eta_M_cold, eta_E = theta[:3]
    
    # Additional parameters for ion fractions
    f_SiII = theta[3]  # Fraction of clouds with Si II
    f_CII = theta[4]   # Fraction with C II
    
    # Run model once
    solution = run_wind_model(eta_M, eta_M_cold, eta_E)
    
    # Calculate ion-specific column densities
    v_SiII, N_SiII = calculate_ion_column_density(solution, f_SiII)
    v_CII, N_CII = calculate_ion_column_density(solution, f_CII)
    
    # Combined likelihood
    log_L = 0.0
    log_L += ion_likelihood(v_SiII, N_SiII, galaxy_data['ions']['SiII'])
    log_L += ion_likelihood(v_CII, N_CII, galaxy_data['ions']['CII'])
    
    return log_L
```

### Convergence Diagnostics

```python
import corner
from emcee import autocorr

# Autocorrelation time
tau = sampler.get_autocorr_time(quiet=True)
print(f"Autocorrelation time: {tau}")

# Effective sample size
n_eff = nsteps * nwalkers / np.mean(tau)
print(f"Effective samples: {n_eff:.0f}")

# Gelman-Rubin statistic
def gelman_rubin(chain):
    """
    Calculate Gelman-Rubin statistic for convergence.
    Chain shape: (nsteps, nwalkers, ndim)
    """
    m, n, _ = chain.shape
    
    # Split chain
    chain1 = chain[:m//2]
    chain2 = chain[m//2:]
    
    # Within-chain variance
    W = np.mean([np.var(chain1, axis=0), np.var(chain2, axis=0)])
    
    # Between-chain variance
    mean1 = np.mean(chain1, axis=0)
    mean2 = np.mean(chain2, axis=0)
    B = m/2 * np.var([mean1, mean2], axis=0)
    
    # Potential scale reduction
    var_est = (1 - 1/m) * W + B/m
    R_hat = np.sqrt(var_est / W)
    
    return R_hat

R_hat = gelman_rubin(chain)
print(f"Gelman-Rubin R_hat: {R_hat}")
# Should be < 1.1 for convergence
```

## Visualization

### Corner Plot

```python
# Flatten chain
samples = sampler.get_chain(discard=1000, flat=True)

# Make corner plot
fig = corner.corner(
    samples,
    labels=[r"$\eta_M$", r"$\eta_{M,cold}$", r"$\eta_E$"],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)
```

### Trace Plots

```python
fig, axes = plt.subplots(3, figsize=(10, 7))

labels = [r"$\eta_M$", r"$\eta_{M,cold}$", r"$\eta_E$"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(chain[:, :, i], alpha=0.3)
    ax.set_ylabel(labels[i])
    ax.set_xlabel("Step")
    
plt.tight_layout()
```

### Model vs Data

```python
# Get best-fit parameters
samples = sampler.get_chain(discard=1000, flat=True)
theta_best = np.percentile(samples, 50, axis=0)

# Run model with best-fit
predictions = run_model(theta_best, galaxy_data)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, ion in enumerate(['SiII', 'CII']):
    ax = axes[i]
    
    # Observations
    obs = galaxy_data['ions'][ion]
    ax.errorbar(obs['v'], obs['N'], obs['N_err'],
                fmt='ko', label='Observed')
    
    # Model
    pred = predictions[ion]
    ax.plot(pred['v'], pred['N'], 'r-', label='Model')
    
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('dN/dv [cm$^{-2}$/(km/s)]')
    ax.set_title(ion)
    ax.legend()
    ax.set_yscale('log')
```

## Performance Optimization

### Model Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_model(eta_M, eta_M_cold, eta_E, sfr, v_circ):
    """
    Cache model results for repeated parameters.
    """
    # Round parameters to reduce cache misses
    eta_M = round(eta_M, 3)
    eta_M_cold = round(eta_M_cold, 3)
    eta_E = round(eta_E, 3)
    
    return run_model([eta_M, eta_M_cold, eta_E], 
                    {'sfr': sfr, 'v_circ': v_circ})
```

### Vectorized Likelihood

```python
def vectorized_likelihood(theta_array, galaxy_data):
    """
    Evaluate multiple parameter sets at once.
    """
    log_L = np.zeros(len(theta_array))
    
    for i, theta in enumerate(theta_array):
        log_L[i] = log_likelihood(theta, galaxy_data)
    
    return log_L
```

## Common Issues

### Poor Mixing
- **Symptom**: Chains stuck in local modes
- **Solutions**:
  - Increase walker count
  - Use parallel tempering
  - Improve initial positions

### Slow Convergence
- **Symptom**: Long autocorrelation times
- **Solutions**:
  - Reparameterize (use log-parameters)
  - Tighten priors
  - Use affine-invariant moves

### Model Failures
- **Symptom**: Many -inf likelihoods
- **Solutions**:
  - Expand prior bounds carefully
  - Add error handling in model
  - Use try/except blocks

## Best Practices

1. **Always check convergence**: Use multiple diagnostics
2. **Visualize everything**: Traces, corners, model fits
3. **Start simple**: Test with fixed parameters first
4. **Use parallel processing**: Significant speedup
5. **Save chains frequently**: Checkpoint long runs
6. **Document priors**: Justify choices physically

## See Also

- [CLASSYFitter API](../api/fitting.md) - Detailed API
- [Fitting Examples](../examples/fitting/) - Complete examples
- [emcee documentation](https://emcee.readthedocs.io/) - MCMC details