#!/usr/bin/env python
"""
Production MCMC run for galaxy J0021+0052.

This script runs a full production-quality MCMC fit with:
- 64 walkers
- 10,000 steps  
- 10 parallel cores
- Full degeneracy analysis
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classy_data_loader import load_classy_observations, get_galaxy_for_fitting
from classy_emcee_fit import (
    CLASSYFitter, run_mcmc_single_galaxy, 
    save_results, create_diagnostic_plots, check_convergence
)
from analyze_degeneracies import (
    analyze_degeneracies, create_enhanced_corner_plot,
    create_degeneracy_heatmap, create_pairwise_evolution_plot
)
import emcee
from multiprocessing import Pool


def run_production_fit():
    """Run production MCMC for J0021+0052."""
    
    # Target galaxy
    galaxy_name = 'J0021+0052'
    
    # Production parameters
    n_walkers = 64
    n_steps = 10000
    n_threads = 10
    
    print("="*70)
    print("PRODUCTION MCMC RUN FOR J0021+0052")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Galaxy: {galaxy_name}")
    print(f"  Walkers: {n_walkers}")
    print(f"  Steps: {n_steps}")
    print(f"  Threads: {n_threads}")
    print(f"  Expected runtime: ~{n_steps * 0.5 / n_threads / 60:.1f} minutes")
    print("="*70)
    
    # Load galaxy data
    print("\nLoading galaxy data...")
    galaxies = load_classy_observations()
    
    if galaxy_name not in galaxies:
        print(f"Error: Galaxy {galaxy_name} not found in data!")
        return
    
    galaxy_data = get_galaxy_for_fitting(galaxy_name, galaxies)
    
    # Print galaxy properties
    print(f"\nGalaxy properties:")
    print(f"  SFR: {galaxy_data['sfr']:.1f} Msun/yr")
    print(f"  v_circ: {galaxy_data['v_circ']:.1f} km/s")
    print(f"  r50: {galaxy_data['r50']:.2f} kpc")
    print(f"\nObservables:")
    print(f"  log(v): {galaxy_data['log_v']:.2f} ± {galaxy_data['log_v_err']:.3f}")
    print(f"  log(width): {galaxy_data['log_width']:.2f} ± {galaxy_data['log_width_err']:.3f}")
    print(f"  log(NH): {galaxy_data['log_NH']:.2f} ± {galaxy_data['log_NH_err']:.3f}")
    
    # Set up MCMC
    print("\n" + "-"*70)
    print("SETTING UP MCMC")
    print("-"*70)
    
    # Extract observables and errors
    observed = (galaxy_data['log_v'], galaxy_data['log_width'], galaxy_data['log_NH'])
    errors = (galaxy_data['log_v_err'], galaxy_data['log_width_err'], galaxy_data['log_NH_err'])
    galaxy_props = (galaxy_data['sfr'], galaxy_data['v_circ'], galaxy_data['r50'])
    
    # Initialize fitter
    fitter = CLASSYFitter(galaxy_data, n_cloud_species=10)
    
    # Set up initial positions - use a tighter ball around reasonable values
    ndim = 3  # eta_M, eta_M_cold, eta_E
    
    # Initial guess based on typical values
    initial = np.array([0.3, 1.0, 0.05])  
    
    # Create initial walker positions with small scatter
    np.random.seed(42)  # For reproducibility
    pos = initial + 0.1 * initial[np.newaxis, :] * np.random.randn(n_walkers, ndim)
    
    # Ensure all initial positions are within priors
    from classy_emcee_fit import log_prior
    for i in range(n_walkers):
        attempts = 0
        while not np.isfinite(log_prior(pos[i])) and attempts < 100:
            pos[i] = initial + 0.1 * initial * np.random.randn(ndim)
            attempts += 1
        if attempts >= 100:
            print(f"Warning: Could not find valid initial position for walker {i}")
    
    print(f"\nInitial parameter ranges:")
    print(f"  eta_M: {pos[:, 0].min():.3f} - {pos[:, 0].max():.3f}")
    print(f"  eta_M_cold: {pos[:, 1].min():.3f} - {pos[:, 1].max():.3f}")
    print(f"  eta_E: {pos[:, 2].min():.3f} - {pos[:, 2].max():.3f}")
    
    # Run MCMC
    print("\n" + "-"*70)
    print("RUNNING MCMC")
    print("-"*70)
    print("Progress will be shown below...")
    
    start_time = time.time()
    
    # Import log_probability function
    from classy_emcee_fit import log_probability
    
    # Set up sampler with multiprocessing
    with Pool(n_threads) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_probability,
            args=(observed, errors, galaxy_props, fitter),
            pool=pool
        )
        
        # Run MCMC with progress bar
        print("")
        for i, result in enumerate(sampler.sample(pos, iterations=n_steps, progress=True)):
            # Check convergence every 1000 steps
            if (i + 1) % 1000 == 0:
                print(f"\nStep {i + 1}/{n_steps}")
                print(f"  Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
                
                # Try to compute autocorrelation time
                try:
                    tau = sampler.get_autocorr_time(tol=0)
                    print(f"  Autocorrelation time: {tau}")
                    converged = np.all(tau * 50 < sampler.iteration)
                    if converged:
                        print("  ✓ Chains appear to have converged!")
                except:
                    print("  Autocorrelation time not yet reliable")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ MCMC completed in {elapsed_time/60:.1f} minutes")
    print(f"  Final mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Save results
    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70)
    
    output_dir = 'production_results'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    os.makedirs(f'{output_dir}/chains', exist_ok=True)
    
    # Save full chain
    np.save(f'{output_dir}/chains/{galaxy_name}_full_chain.npy', sampler.get_chain())
    print(f"  ✓ Full chain saved")
    
    # Save results with burn-in removed
    results = save_results(galaxy_name, sampler, output_dir=f'{output_dir}/chains')
    print(f"  ✓ Results saved")
    
    # Print best-fit parameters
    print(f"\nBest-fit parameters (median ± 1σ):")
    print(f"  eta_M:      {results['eta_M']['median']:.3e} "
          f"({results['eta_M']['lower']:.3e}, {results['eta_M']['upper']:.3e})")
    print(f"  eta_M_cold: {results['eta_M_cold']['median']:.3e} "
          f"({results['eta_M_cold']['lower']:.3e}, {results['eta_M_cold']['upper']:.3e})")
    print(f"  eta_E:      {results['eta_E']['median']:.3e} "
          f"({results['eta_E']['lower']:.3e}, {results['eta_E']['upper']:.3e})")
    
    # Create diagnostic plots
    print("\n" + "-"*70)
    print("CREATING DIAGNOSTIC PLOTS")
    print("-"*70)
    
    create_diagnostic_plots(
        galaxy_name, sampler, 
        output_dir=f'{output_dir}/plots',
        analyze_degeneracy=True
    )
    print("  ✓ Standard diagnostic plots created")
    
    # Get flattened chain for degeneracy analysis
    n_burn = min(2000, sampler.iteration // 4)
    chain = sampler.get_chain(discard=n_burn, flat=True)
    
    # Enhanced degeneracy analysis
    print("\n" + "-"*70)
    print("DEGENERACY ANALYSIS")
    print("-"*70)
    
    deg_results = analyze_degeneracies(
        chain,
        param_names=['eta_M', 'eta_M_cold', 'eta_E'],
        output_file=f'{output_dir}/{galaxy_name}_degeneracy_analysis.txt'
    )
    
    # Create evolution plot
    try:
        create_pairwise_evolution_plot(
            sampler, galaxy_name,
            output_dir=f'{output_dir}/plots',
            n_thin=100
        )
        print("  ✓ Parameter evolution plot created")
    except Exception as e:
        print(f"  ⚠ Could not create evolution plot: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("PRODUCTION RUN COMPLETE")
    print("="*70)
    print(f"\nGalaxy: {galaxy_name}")
    print(f"Runtime: {elapsed_time/60:.1f} minutes")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"Effective samples: ~{n_walkers * n_steps * np.mean(sampler.acceptance_fraction):.0f}")
    
    # Check convergence
    converged, tau = check_convergence(sampler, threshold=50)
    if converged:
        print(f"\n✓ CHAINS CONVERGED")
        print(f"  Autocorrelation times: {tau}")
    else:
        print(f"\n⚠ Chains may not have fully converged")
        if tau is not None:
            print(f"  Autocorrelation times: {tau}")
            print(f"  Recommendation: Run for {int(50 * np.max(tau))} steps")
    
    # Degeneracy summary
    corr_matrix = deg_results['correlation_matrix']
    print(f"\nKey correlations:")
    print(f"  eta_M - eta_M_cold: {corr_matrix[0,1]:+.3f}")
    print(f"  eta_M - eta_E:      {corr_matrix[0,2]:+.3f}")
    print(f"  eta_M_cold - eta_E: {corr_matrix[1,2]:+.3f}")
    print(f"\nCondition number: {deg_results['condition_number']:.1f}")
    
    if deg_results['condition_number'] > 100:
        print("  ⚠ Strong degeneracies detected")
    elif deg_results['condition_number'] > 30:
        print("  ⚠ Moderate degeneracies present")
    else:
        print("  ✓ Parameters well-constrained")
    
    print(f"\nAll results saved to: {output_dir}/")
    print("\nProduction run complete!")
    
    return sampler, results, deg_results


if __name__ == "__main__":
    sampler, results, deg_results = run_production_fit()