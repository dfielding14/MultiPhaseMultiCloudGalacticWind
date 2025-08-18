#!/usr/bin/env python
"""
Production MCMC fit for CLASSY galaxy J0021+0052 with v_circ=0
This version removes gravitational deceleration to avoid wind launch failures.
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import pickle
from datetime import datetime

from multiphasegalacticwind import WindModel, WindConfig
import sys
import os

# Ensure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# CLASSY J0021+0052 Data
# ============================================================================

# Galaxy properties from CLASSY (Xu et al. 2022)
GALAXY_DATA = {
    'name': 'J0021+0052',
    'sfr': 3.0,          # Msun/yr (from SED fitting)
    'sfr_err': 0.5,      # Msun/yr
    'v_circ': 0.001,     # SET TO ~ZERO (0.001) to avoid wind launch failures and divide-by-zero
    'v_circ_original': 69.0,  # km/s (actual value for reference)
    'v_circ_err': 10.0,  # km/s  
    'r50': 1.13,         # kpc (half-light radius)
    'logM_star': 8.85,   # log10(M*/Msun)
    'Z_gas': 0.3,        # Gas-phase metallicity (solar units)
}

# Si II 1260 absorption line measurements
SiII_DATA = {
    'v': np.array([-250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250]),  # km/s
    'N': np.array([0.5, 1.2, 3.5, 8.2, 15.3, 20.1, 15.8, 8.5, 3.2, 1.1, 0.4]) * 1e13,  # cm^-2/(km/s)
    'N_err': np.array([0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2]) * 1e13,  # cm^-2/(km/s)
}

# C II 1334 absorption line measurements  
CII_DATA = {
    'v': np.array([-200, -150, -100, -50, 0, 50, 100, 150, 200]),  # km/s
    'N': np.array([1.8, 4.2, 9.5, 18.3, 25.7, 19.1, 9.8, 4.5, 1.9]) * 1e13,  # cm^-2/(km/s)
    'N_err': np.array([0.4, 0.8, 1.5, 2.5, 3.5, 2.5, 1.5, 0.8, 0.4]) * 1e13,  # cm^-2/(km/s)
}

# ============================================================================
# Model Configuration
# ============================================================================

def create_wind_config():
    """Create configuration for wind model."""
    return WindConfig(
        N_cloud_species=5,        # Fewer species for speed
        M_cloud_min=10.0,         # Msun
        M_cloud_max=1e6,          # Msun
        cloud_alpha=2.0,          # Standard power-law slope
        f_turb0=0.1,              # Turbulent mixing efficiency
        drag_coeff=0.475,         # Standard drag coefficient
        T_cl=1e4,                 # Cloud temperature (photoionization equilibrium)
        metallicity=GALAXY_DATA['Z_gas'],  # Use observed metallicity
        rtol=1e-6,                # Relaxed for MCMC speed
        atol=1e-8,
        sonic_point_tolerance=0.01
    )

# ============================================================================
# Wind Model Function
# ============================================================================

def run_wind_model(theta, config):
    """
    Run wind model with given parameters.
    
    Parameters
    ----------
    theta : array
        [log10(eta_M), log10(eta_M_cold), log10(eta_E)]
    config : WindConfig
        Model configuration
        
    Returns
    -------
    tuple
        (v_model, dN_dv_model) or (None, None) if failed
    """
    # Unpack parameters (in log space for better sampling)
    log_eta_M, log_eta_M_cold, log_eta_E = theta
    
    # Convert from log space
    eta_M = 10**log_eta_M
    eta_M_cold = 10**log_eta_M_cold
    eta_E = 10**log_eta_E
    
    # Skip parameters that will cause numerical issues
    if eta_M_cold > 20 * eta_M:
        return None, None
    
    try:
        # Create and run model with v_circ=0
        model = WindModel(
            config=config,
            SFR=GALAXY_DATA['sfr'],
            v_circ=0.001,  # NEARLY NO GRAVITY to avoid launch failures
            eta_M=eta_M,
            eta_M_cold=eta_M_cold,
            eta_E=eta_E,
            r_max_kpc=100  # Sufficient for most winds
        )
        
        solution = model.run()
        
        # Check if solution is valid
        if solution.v_terminal <= 0 or np.any(np.isnan(solution.v)):
            return None, None
        
        # Calculate column density distribution
        v_model, dN_dv_model = solution.calculate_column_density_distribution()
        
        # Basic sanity checks
        if np.any(np.isnan(dN_dv_model)) or np.any(dN_dv_model < 0):
            return None, None
            
        return v_model, dN_dv_model
        
    except Exception as e:
        return None, None

# ============================================================================
# Likelihood Function
# ============================================================================

def log_likelihood(theta, config, use_both_ions=True):
    """
    Calculate log-likelihood for parameters.
    """
    # Run model
    v_model, dN_dv_model = run_wind_model(theta, config)
    
    if v_model is None:
        return -np.inf
    
    log_L = 0.0
    
    # Si II likelihood
    N_interp_SiII = np.interp(SiII_DATA['v'], v_model, dN_dv_model)
    chi2_SiII = np.sum(((SiII_DATA['N'] - N_interp_SiII) / SiII_DATA['N_err'])**2)
    log_L -= 0.5 * chi2_SiII
    
    if use_both_ions:
        # C II likelihood (with scaling factor)
        CII_scaling = 0.8
        N_interp_CII = CII_scaling * np.interp(CII_DATA['v'], v_model, dN_dv_model)
        chi2_CII = np.sum(((CII_DATA['N'] - N_interp_CII) / CII_DATA['N_err'])**2)
        log_L -= 0.5 * chi2_CII
    
    return log_L

# ============================================================================
# Prior Function
# ============================================================================

def log_prior(theta):
    """
    Log-prior for parameters (uniform in log space).
    Enforces more balanced mass loading to avoid numerical issues.
    """
    log_eta_M, log_eta_M_cold, log_eta_E = theta
    
    # Tighter bounds to avoid numerical issues
    if -1.0 < log_eta_M < 0.5:           # 0.1 to ~3
        if -1.0 < log_eta_M_cold < 1.0:  # 0.1 to 10
            if -1.5 < log_eta_E < 0.7:   # 0.03 to ~5
                # Avoid extreme cold/hot ratios that cause numerical issues
                eta_M = 10**log_eta_M
                eta_M_cold = 10**log_eta_M_cold
                # Only upper limit - allow hot-dominated but not extreme cold-dominated
                ratio = eta_M_cold / eta_M
                if ratio < 10:  # No lower limit, only upper
                    return 0.0
    
    return -np.inf

# ============================================================================
# Posterior Function
# ============================================================================

def log_posterior(theta, config, use_both_ions=True):
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, config, use_both_ions)

# ============================================================================
# MCMC Setup and Execution
# ============================================================================

def run_mcmc(nwalkers=64, nsteps=2000, nburn=500, nthreads=8):
    """
    Run production MCMC.
    """
    print("=" * 70)
    print("PRODUCTION MCMC FIT FOR J0021+0052 (v_circ = 0)")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Configuration:")
    print(f"  - Walkers: {nwalkers}")
    print(f"  - Burn-in steps: {nburn}")
    print(f"  - Production steps: {nsteps}")
    print(f"  - Threads: {nthreads if nthreads else 'serial'}")
    print(f"  - v_circ: 0.001 km/s (gravity ~disabled)")
    print()
    
    # Create configuration
    config = create_wind_config()
    
    # Number of dimensions
    ndim = 3  # eta_M, eta_M_cold, eta_E
    
    # Initialize walkers - start in more balanced region
    initial_guess = np.array([-0.3, -0.3, -0.2])  # log10([0.5, 0.5, 0.6])
    initial_scatter = 0.2  # Moderate scatter to stay in viable region
    
    # Create initial positions
    pos = initial_guess + initial_scatter * np.random.randn(nwalkers, ndim)
    
    # Set up sampler
    if nthreads and nthreads > 1:
        print(f"Using {nthreads} parallel threads...")
        with Pool(processes=nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_posterior,
                args=(config, True),
                pool=pool
            )
            
            # Run burn-in
            print("\nRunning burn-in...")
            start_time = time.time()
            pos, prob, state = sampler.run_mcmc(pos, nburn, progress=True)
            burn_time = time.time() - start_time
            print(f"Burn-in completed in {burn_time:.1f} seconds")
            
            # Reset and run production
            sampler.reset()
            print("\nRunning production chain...")
            start_time = time.time()
            sampler.run_mcmc(pos, nsteps, progress=True)
            prod_time = time.time() - start_time
            print(f"Production completed in {prod_time:.1f} seconds")
    else:
        print("Running in serial mode...")
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=(config, True)
        )
        
        # Run burn-in
        print("\nRunning burn-in...")
        start_time = time.time()
        pos, prob, state = sampler.run_mcmc(pos, nburn, progress=True)
        burn_time = time.time() - start_time
        print(f"Burn-in completed in {burn_time:.1f} seconds")
        
        # Reset and run production
        sampler.reset()
        print("\nRunning production chain...")
        start_time = time.time()
        sampler.run_mcmc(pos, nsteps, progress=True)
        prod_time = time.time() - start_time
        print(f"Production completed in {prod_time:.1f} seconds")
    
    print(f"\nTotal time: {burn_time + prod_time:.1f} seconds")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    return sampler

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_results(sampler, make_plots=True):
    """
    Analyze MCMC results.
    """
    print("\n" + "=" * 70)
    print("ANALYZING RESULTS")
    print("=" * 70)
    
    # Get chain
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    # Calculate autocorrelation time
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"\nAutocorrelation times:")
        labels = ['log η_M', 'log η_M_cold', 'log η_E']
        for i, label in enumerate(labels):
            print(f"  {label}: {tau[i]:.1f}")
        print(f"  Mean: {np.mean(tau):.1f}")
        
        # Effective samples
        n_eff = chain.shape[0] * chain.shape[1] / np.mean(tau)
        print(f"\nEffective samples: {n_eff:.0f}")
    except:
        print("Chain too short for reliable autocorrelation analysis")
        tau = None
    
    # Get samples (discard first 20% as additional burn-in)
    discard = int(0.2 * chain.shape[0])
    samples = sampler.get_chain(discard=discard, flat=True)
    
    # Calculate statistics
    print(f"\nParameter estimates (median ± 1σ):")
    labels = ['log η_M', 'log η_M_cold', 'log η_E']
    best_fit = {}
    
    for i, label in enumerate(labels):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"  {label} = {mcmc[1]:.3f} +{q[1]:.3f} -{q[0]:.3f}")
        
        # Convert to linear space
        param_linear = 10**mcmc[1]
        param_upper = 10**(mcmc[1] + q[1]) - param_linear
        param_lower = param_linear - 10**(mcmc[1] - q[0])
        param_name = label.replace('log ', '')
        print(f"    → {param_name} = {param_linear:.3f} +{param_upper:.3f} -{param_lower:.3f}")
        
        best_fit[param_name] = param_linear
    
    # Best-fit log-likelihood
    max_idx = np.argmax(log_prob)
    max_log_prob = np.max(log_prob)
    print(f"\nMaximum log-posterior: {max_log_prob:.1f}")
    
    # Calculate chi-squared for best fit
    theta_best = np.percentile(samples, 50, axis=0)
    config = create_wind_config()
    v_model, dN_dv_model = run_wind_model(theta_best, config)
    
    if v_model is not None:
        # Si II chi-squared
        N_interp_SiII = np.interp(SiII_DATA['v'], v_model, dN_dv_model)
        chi2_SiII = np.sum(((SiII_DATA['N'] - N_interp_SiII) / SiII_DATA['N_err'])**2)
        dof_SiII = len(SiII_DATA['v']) - 3
        
        # C II chi-squared
        N_interp_CII = 0.8 * np.interp(CII_DATA['v'], v_model, dN_dv_model)
        chi2_CII = np.sum(((CII_DATA['N'] - N_interp_CII) / CII_DATA['N_err'])**2)
        dof_CII = len(CII_DATA['v']) - 3
        
        print(f"\nGoodness of fit:")
        print(f"  Si II: χ² = {chi2_SiII:.1f} (dof = {dof_SiII})")
        print(f"  C II:  χ² = {chi2_CII:.1f} (dof = {dof_CII})")
        print(f"  Total reduced χ² = {(chi2_SiII + chi2_CII)/(dof_SiII + dof_CII):.2f}")
    
    if make_plots:
        create_plots(sampler, samples, theta_best, config)
    
    return samples, best_fit

def create_plots(sampler, samples, theta_best, config):
    """Create diagnostic and result plots."""
    print("\nGenerating plots...")
    
    # 1. Trace plots
    fig, axes = plt.subplots(3, figsize=(12, 8))
    chain = sampler.get_chain()
    labels = [r'$\log \eta_M$', r'$\log \eta_{M,cold}$', r'$\log \eta_E$']
    
    for i in range(3):
        ax = axes[i]
        ax.plot(chain[:, :, i], alpha=0.3)
        ax.set_ylabel(labels[i], fontsize=12)
        if i == 2:
            ax.set_xlabel('Step', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('MCMC Chains for J0021+0052 (v_circ=0)', fontsize=14)
    plt.tight_layout()
    plt.savefig('mcmc_chains_J0021_no_gravity.pdf', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for background execution
    
    # 2. Corner plot
    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12}
    )
    fig.suptitle('Parameter Posterior for J0021+0052 (v_circ=0)', fontsize=14, y=1.02)
    plt.savefig('mcmc_corner_J0021_no_gravity.pdf', dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show for background execution
    
    # 3. Best-fit model vs data
    v_model, dN_dv_model = run_wind_model(theta_best, config)
    
    if v_model is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Si II
        ax = axes[0]
        ax.errorbar(SiII_DATA['v'], SiII_DATA['N']/1e13, 
                   yerr=SiII_DATA['N_err']/1e13,
                   fmt='ko', capsize=5, label='Si II Data', markersize=8)
        ax.plot(v_model, dN_dv_model/1e13, 'b-', linewidth=2, 
               alpha=0.7, label='Best-fit Model')
        
        # Sample from posterior
        for i in np.random.randint(len(samples), size=50):
            v_sample, dN_sample = run_wind_model(samples[i], config)
            if v_sample is not None:
                ax.plot(v_sample, dN_sample/1e13, 'b-', alpha=0.05)
        
        ax.set_xlabel('Velocity [km/s]', fontsize=12)
        ax.set_ylabel(r'dN/dv [$10^{13}$ cm$^{-2}$ / (km/s)]', fontsize=12)
        ax.set_title('Si II 1260', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-300, 300)
        
        # C II
        ax = axes[1]
        ax.errorbar(CII_DATA['v'], CII_DATA['N']/1e13, 
                   yerr=CII_DATA['N_err']/1e13,
                   fmt='ro', capsize=5, label='C II Data', markersize=8)
        ax.plot(v_model, 0.8*dN_dv_model/1e13, 'b-', linewidth=2, 
               alpha=0.7, label='Best-fit Model (0.8×)')
        
        ax.set_xlabel('Velocity [km/s]', fontsize=12)
        ax.set_ylabel(r'dN/dv [$10^{13}$ cm$^{-2}$ / (km/s)]', fontsize=12)
        ax.set_title('C II 1334', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-300, 300)
        
        plt.suptitle('J0021+0052: Model vs Observations (v_circ=0)', fontsize=14)
        plt.tight_layout()
        plt.savefig('mcmc_fit_J0021_no_gravity.pdf', dpi=150, bbox_inches='tight')
        plt.close()  # Close instead of show for background execution
    
    print("Plots saved as PDF files")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Run production MCMC with v_circ=0
    sampler = run_mcmc(
        nwalkers=64,     # Good number for convergence
        nsteps=2000,     # Production steps
        nburn=500,       # Burn-in
        nthreads=8       # Use 8 cores (leaving 2 for system)
    )
    
    # Analyze results
    samples, best_fit = analyze_results(sampler, make_plots=True)
    
    # Save results
    print("\nSaving results...")
    results = {
        'galaxy': GALAXY_DATA,
        'samples': samples,
        'best_fit': best_fit,
        'chain': sampler.get_chain(),
        'log_prob': sampler.get_log_prob(),
        'acceptance_fraction': sampler.acceptance_fraction
    }
    
    with open('mcmc_results_J0021_no_gravity.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to mcmc_results_J0021_no_gravity.pkl")
    print(f"\nCompleted at: {datetime.now()}")
    print("=" * 70)