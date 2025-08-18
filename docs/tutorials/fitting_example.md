# Tutorial: Fitting CLASSY Data with MCMC

## Introduction

This tutorial demonstrates how to fit wind model parameters to real galaxy observations from the CLASSY survey using MCMC.

## The Data

We'll fit J0021+0052, a star-forming galaxy with:
- SFR = 3.0 Msun/yr
- v_circ = 69 km/s  
- Multiple UV absorption lines (Si II, C II)

## Step 1: Load Observational Data

```python
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiphasegalacticwind import WindModel, WindConfig

# J0021+0052 observations (simplified)
galaxy_data = {
    'name': 'J0021+0052',
    'sfr': 3.0,        # Msun/yr
    'v_circ': 69.0,    # km/s
    'r50': 1.13,       # kpc (half-light radius)
}

# Si II absorption line data
v_obs = np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200])  # km/s
N_obs = np.array([1e12, 3e12, 1e13, 5e13, 1e14, 5e13, 1e13, 3e12, 1e12])  # cm^-2/(km/s)
N_err = N_obs * 0.2  # 20% errors
```

## Step 2: Define the Model Function

```python
def run_wind_model(theta, galaxy_data):
    """
    Run wind model with given parameters.
    
    Parameters
    ----------
    theta : array
        [log10(eta_M), log10(eta_M_cold), log10(eta_E)]
    """
    # Unpack parameters (in log space for better sampling)
    log_eta_M, log_eta_M_cold, log_eta_E = theta
    eta_M = 10**log_eta_M
    eta_M_cold = 10**log_eta_M_cold
    eta_E = 10**log_eta_E
    
    # Create configuration
    config = WindConfig(
        N_cloud_species=5,  # Fewer for speed
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
        v_model, dN_dv = solution.calculate_column_density_distribution()
        
        return v_model, dN_dv
        
    except Exception as e:
        print(f"Model failed: {e}")
        return None, None
```

## Step 3: Define Likelihood Function

```python
def log_likelihood(theta, galaxy_data, v_obs, N_obs, N_err):
    """
    Calculate log-likelihood for parameters.
    """
    # Run model
    v_model, N_model = run_wind_model(theta, galaxy_data)
    
    if v_model is None:
        return -np.inf
    
    # Interpolate model to observation points
    N_interp = np.interp(v_obs, v_model, N_model)
    
    # Calculate chi-squared
    chi2 = np.sum(((N_obs - N_interp) / N_err)**2)
    
    return -0.5 * chi2
```

## Step 4: Define Prior

```python
def log_prior(theta):
    """
    Log-prior for parameters (uniform in log space).
    """
    log_eta_M, log_eta_M_cold, log_eta_E = theta
    
    # Reasonable bounds in log space
    if -3 < log_eta_M < 1:           # 0.001 to 10
        if -2 < log_eta_M_cold < 2:  # 0.01 to 100
            if -2 < log_eta_E < 1:    # 0.01 to 10
                return 0.0  # Uniform in log space
    
    return -np.inf
```

## Step 5: Define Posterior

```python
def log_posterior(theta, galaxy_data, v_obs, N_obs, N_err):
    """
    Log-posterior = log-prior + log-likelihood
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + log_likelihood(theta, galaxy_data, v_obs, N_obs, N_err)
```

## Step 6: Initialize and Run MCMC

```python
# Set up MCMC
ndim = 3  # Number of parameters
nwalkers = 32  # Number of walkers

# Initialize walkers near reasonable values
initial_guess = np.array([-1.0, 0.0, 0.0])  # log10 values
pos = initial_guess + 0.1 * np.random.randn(nwalkers, ndim)

# Create sampler
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_posterior,
    args=(galaxy_data, v_obs, N_obs, N_err)
)

# Run burn-in
print("Running burn-in...")
pos, prob, state = sampler.run_mcmc(pos, 200, progress=True)
sampler.reset()

# Run production
print("Running production chain...")
sampler.run_mcmc(pos, 1000, progress=True)

print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
```

## Step 7: Analyze Results

```python
# Get chain
chain = sampler.get_chain()
log_prob = sampler.get_log_prob()

# Plot chains
fig, axes = plt.subplots(3, figsize=(10, 7))
labels = [r"$\log \eta_M$", r"$\log \eta_{M,cold}$", r"$\log \eta_E$"]

for i in range(ndim):
    ax = axes[i]
    ax.plot(chain[:, :, i], alpha=0.3)
    ax.set_ylabel(labels[i])
    ax.set_xlabel("Step")
    
plt.tight_layout()
plt.show()

# Get samples (discard burn-in)
samples = sampler.get_chain(discard=200, flat=True)

# Print results
for i in range(ndim):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    print(f"{labels[i]} = {mcmc[1]:.3f} +{q[1]:.3f} -{q[0]:.3f}")
```

## Step 8: Make Corner Plot

```python
# Corner plot
fig = corner.corner(
    samples,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)

plt.show()

# Get best-fit values (median)
theta_best = np.percentile(samples, 50, axis=0)
eta_M_best = 10**theta_best[0]
eta_M_cold_best = 10**theta_best[1]
eta_E_best = 10**theta_best[2]

print(f"\nBest-fit parameters:")
print(f"η_M = {eta_M_best:.3f}")
print(f"η_M_cold = {eta_M_cold_best:.2f}")
print(f"η_E = {eta_E_best:.2f}")
```

## Step 9: Plot Best-Fit Model

```python
# Run best-fit model
v_best, N_best = run_wind_model(theta_best, galaxy_data)

# Plot comparison
plt.figure(figsize=(10, 6))

# Observations
plt.errorbar(v_obs, N_obs, yerr=N_err, fmt='ko', 
             capsize=5, label='J0021+0052 Data')

# Best-fit model
plt.semilogy(v_best, N_best, 'r-', linewidth=2, 
             label='Best-fit Model', alpha=0.8)

# Sample from posterior
for i in np.random.randint(len(samples), size=50):
    theta_sample = samples[i]
    v_sample, N_sample = run_wind_model(theta_sample, galaxy_data)
    if v_sample is not None:
        plt.semilogy(v_sample, N_sample, 'b-', alpha=0.05)

plt.xlabel('Velocity [km/s]')
plt.ylabel('dN/dv [cm⁻² / (km/s)]')
plt.title('MCMC Fit to J0021+0052')
plt.legend()
plt.xlim(-300, 300)
plt.ylim(1e11, 1e15)
plt.grid(True, alpha=0.3)
plt.show()
```

## Step 10: Convergence Diagnostics

```python
# Autocorrelation time
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"Autocorrelation time: {tau}")
    print(f"Mean tau: {np.mean(tau):.1f}")
    
    # Effective samples
    n_eff = len(samples) / np.mean(tau)
    print(f"Effective samples: {n_eff:.0f}")
except:
    print("Chain too short for autocorrelation analysis")

# Gelman-Rubin statistic (if you ran multiple chains)
def gelman_rubin(chain):
    m, n, d = chain.shape
    chain_means = np.mean(chain, axis=0)
    B = n * np.var(chain_means, axis=0)
    W = np.mean(np.var(chain, axis=0), axis=0)
    V = (n-1)/n * W + B/n
    R_hat = np.sqrt(V/W)
    return R_hat

# R_hat = gelman_rubin(chain)
# print(f"Gelman-Rubin R_hat: {R_hat}")
```

## Advanced: Parallel Processing

```python
from multiprocessing import Pool

# Run with parallel processing (much faster!)
with Pool() as pool:
    sampler_parallel = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(galaxy_data, v_obs, N_obs, N_err),
        pool=pool
    )
    
    # Run MCMC
    print("Running parallel MCMC...")
    sampler_parallel.run_mcmc(pos, 2000, progress=True)
```

## Key Insights

1. **Parameter Degeneracies**: η_M and η_E are often correlated
2. **Convergence**: Check chains and autocorrelation
3. **Model Failures**: Some parameter combinations fail - handle gracefully
4. **Uncertainties**: MCMC provides full posterior, not just best-fit

## Exercises

1. **Different Galaxy**: Try fitting a different CLASSY galaxy
2. **Multi-Ion**: Fit Si II and C II simultaneously
3. **Priors**: Try informative priors based on simulations
4. **Model Comparison**: Compare with hot-only model
5. **Systematic Errors**: Include systematic uncertainties

## Troubleshooting

- **Poor Mixing**: Increase nwalkers or adjust initial positions
- **Slow Convergence**: Tighten priors or reparameterize
- **Many Failures**: Check parameter bounds are physical

## Next Steps

- [Advanced MCMC Topics](../fitting/advanced.md)
- [Parameter Studies](parameter_study.md)
- [Full CLASSY Analysis](../fitting/classy_example.md)

## References

- Foreman-Mackey et al. (2013) - emcee paper
- Xu et al. (2022) - CLASSY observations
- Fielding & Bryan (2024) - Wind model