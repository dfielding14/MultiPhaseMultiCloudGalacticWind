"""
Production script for fitting all CLASSY galaxies with the multiphase wind model.

This script runs MCMC fits for all valid galaxies in the CLASSY sample,
saving results and diagnostic plots for each galaxy.
"""

import numpy as np
import os
import sys
import time
from multiprocessing import Pool
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classy_data_loader import load_classy_observations, get_galaxy_for_fitting
from classy_emcee_fit import (
    run_mcmc_single_galaxy, save_results, create_diagnostic_plots
)


def fit_single_galaxy_wrapper(args):
    """
    Wrapper function for parallel processing.
    
    Parameters
    ----------
    args : tuple
        (galaxy_name, galaxy_data, n_walkers, n_steps, output_dir)
        
    Returns
    -------
    dict
        Results dictionary with galaxy name and fitting results
    """
    galaxy_name, galaxy_data, n_walkers, n_steps, output_dir = args
    
    print(f"\n{'='*60}")
    print(f"Starting fit for galaxy: {galaxy_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run MCMC
        sampler = run_mcmc_single_galaxy(
            galaxy_name,
            galaxy_data,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_threads=1,  # Single thread per galaxy when running in parallel
            progress=True
        )
        
        # Save results
        results = save_results(
            galaxy_name, sampler, 
            output_dir=os.path.join(output_dir, 'results')
        )
        
        # Create diagnostic plots
        create_diagnostic_plots(
            galaxy_name, sampler,
            output_dir=os.path.join(output_dir, 'plots')
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Galaxy {galaxy_name} completed in {elapsed_time:.1f} seconds")
        print(f"  Best-fit: eta_M={results['eta_M']['median']:.3e}, "
              f"eta_M_cold={results['eta_M_cold']['median']:.3e}, "
              f"eta_E={results['eta_E']['median']:.3e}")
        
        return {
            'galaxy': galaxy_name,
            'success': True,
            'results': results,
            'time': elapsed_time
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Galaxy {galaxy_name} failed after {elapsed_time:.1f} seconds")
        print(f"  Error: {str(e)}")
        
        return {
            'galaxy': galaxy_name,
            'success': False,
            'error': str(e),
            'time': elapsed_time
        }


def run_batch_fits(galaxy_list=None, n_walkers=32, n_steps=5000, 
                   n_cores=4, output_dir='classy_fits', 
                   test_mode=False):
    """
    Run MCMC fits for multiple galaxies.
    
    Parameters
    ----------
    galaxy_list : list, optional
        List of galaxy names to fit. If None, fit all valid galaxies.
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_cores : int
        Number of parallel processes
    output_dir : str
        Output directory for results
    test_mode : bool
        If True, only fit first 3 galaxies with fewer steps
        
    Returns
    -------
    list
        List of result dictionaries
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    
    # Load galaxy data
    print("Loading CLASSY galaxy data...")
    galaxies = load_classy_observations()
    
    # Select galaxies to fit
    if galaxy_list is None:
        galaxy_list = [name for name, data in galaxies.items() if data['valid']]
    
    if test_mode:
        galaxy_list = galaxy_list[:3]
        n_steps = 500
        print(f"\nTEST MODE: Fitting first 3 galaxies with {n_steps} steps")
    
    print(f"\nFitting {len(galaxy_list)} galaxies")
    print(f"MCMC parameters: {n_walkers} walkers, {n_steps} steps")
    print(f"Using {n_cores} parallel processes")
    
    # Prepare arguments for parallel processing
    fit_args = []
    for galaxy_name in galaxy_list:
        galaxy_data = get_galaxy_for_fitting(galaxy_name, galaxies)
        fit_args.append((galaxy_name, galaxy_data, n_walkers, n_steps, output_dir))
    
    # Run fits
    start_time = time.time()
    
    if n_cores > 1:
        print(f"\nRunning fits in parallel on {n_cores} cores...")
        with Pool(n_cores) as pool:
            results = pool.map(fit_single_galaxy_wrapper, fit_args)
    else:
        print("\nRunning fits sequentially...")
        results = [fit_single_galaxy_wrapper(args) for args in fit_args]
    
    total_time = time.time() - start_time
    
    # Summary statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("\n" + "="*60)
    print("BATCH FITTING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful fits: {len(successful)}/{len(results)}")
    
    if failed:
        print(f"\nFailed galaxies ({len(failed)}):")
        for r in failed:
            print(f"  - {r['galaxy']}: {r['error']}")
    
    # Save summary
    summary_file = os.path.join(output_dir, 'fitting_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"CLASSY Fitting Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total galaxies: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"MCMC parameters: {n_walkers} walkers, {n_steps} steps\n")
        f.write(f"\nResults:\n")
        
        for r in successful:
            res = r['results']
            f.write(f"\n{r['galaxy']}:\n")
            f.write(f"  eta_M = {res['eta_M']['median']:.3e} "
                   f"({res['eta_M']['lower']:.3e}, {res['eta_M']['upper']:.3e})\n")
            f.write(f"  eta_M_cold = {res['eta_M_cold']['median']:.3e} "
                   f"({res['eta_M_cold']['lower']:.3e}, {res['eta_M_cold']['upper']:.3e})\n")
            f.write(f"  eta_E = {res['eta_E']['median']:.3e} "
                   f"({res['eta_E']['lower']:.3e}, {res['eta_E']['upper']:.3e})\n")
            f.write(f"  Acceptance fraction: {res['acceptance_fraction']:.3f}\n")
            f.write(f"  Time: {r['time']:.1f} seconds\n")
    
    print(f"\nSummary saved to: {summary_file}")
    
    return results


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(
        description='Run MCMC fits for CLASSY galaxies'
    )
    parser.add_argument(
        '--galaxies', nargs='+', default=None,
        help='Specific galaxy names to fit (default: all valid galaxies)'
    )
    parser.add_argument(
        '--walkers', type=int, default=32,
        help='Number of MCMC walkers (default: 32)'
    )
    parser.add_argument(
        '--steps', type=int, default=5000,
        help='Number of MCMC steps (default: 5000)'
    )
    parser.add_argument(
        '--cores', type=int, default=4,
        help='Number of parallel processes (default: 4)'
    )
    parser.add_argument(
        '--output', default='classy_fits',
        help='Output directory (default: classy_fits)'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: fit only 3 galaxies with 500 steps'
    )
    
    args = parser.parse_args()
    
    # Run batch fits
    results = run_batch_fits(
        galaxy_list=args.galaxies,
        n_walkers=args.walkers,
        n_steps=args.steps,
        n_cores=args.cores,
        output_dir=args.output,
        test_mode=args.test
    )
    
    return 0 if all(r['success'] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())