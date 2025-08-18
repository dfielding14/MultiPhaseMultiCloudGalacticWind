"""
EMCEE fitting implementation for CLASSY galaxy observations using the 
multiphasegalacticwind package.

This module provides tools to fit multiphase wind model parameters
(eta_M, eta_M_cold, eta_E) to observed velocity distributions and 
column densities from the CLASSY survey.
"""

import numpy as np
import emcee
from multiprocessing import Pool
import corner
import matplotlib.pyplot as plt
from scipy import integrate

from multiphasegalacticwind import WindModel, WindConfig
from multiphasegalacticwind.observables import calculate_column_density_distribution


class CLASSYFitter:
    """
    Main class for fitting CLASSY observations with multiphase wind models.
    
    Parameters
    ----------
    galaxy_data : dict
        Dictionary containing galaxy observations and properties
    n_cloud_species : int
        Number of cloud mass bins (default: 10)
    """
    
    def __init__(self, galaxy_data, n_cloud_species=10):
        self.galaxy_data = galaxy_data
        self.n_cloud_species = n_cloud_species
        self.setup_fixed_config()
        
    def setup_fixed_config(self):
        """Setup fixed configuration parameters for all models."""
        # Fixed cloud distribution parameters
        self.M_cloud_min = 1e1  # Msun
        self.M_cloud_max = 1e6  # Msun
        self.cloud_alpha = 2.0
        
        # Generate cloud mass array
        self.M_cloud = np.logspace(
            np.log10(self.M_cloud_min), 
            np.log10(self.M_cloud_max), 
            self.n_cloud_species
        )
        
        # Base configuration (will be updated with fitting parameters)
        self.base_config = {
            'T_cl': 1e4,           # Cloud temperature [K]
            'drag_coeff': 0.5,     # Drag coefficient
            'f_turb0': 0.1,        # Turbulent mixing efficiency
            'mu': 0.62,            # Mean molecular weight
            'rtol': 1e-6,          # Integration tolerance
            'atol': 1e-8,          # Integration tolerance
        }
        
    def evaluate_model(self, eta_M, eta_M_cold, eta_E, sfr, v_circ, r50):
        """
        Run wind model with given parameters and compute observables.
        
        Parameters
        ----------
        eta_M : float
            Hot phase mass loading factor
        eta_M_cold : float
            Cold phase mass loading factor
        eta_E : float
            Energy loading factor
        sfr : float
            Star formation rate [Msun/yr]
        v_circ : float
            Circular velocity [km/s]
        r50 : float
            Half-light radius [kpc]
            
        Returns
        -------
        tuple
            (log_v, log_width, log_NH) or (None, None, None) if failed
        """
        # Create configuration with fitting parameters
        config = WindConfig(
            eta_M=eta_M,
            eta_M_cold=eta_M_cold,
            eta_E=eta_E,
            N_cloud_species=self.n_cloud_species,
            M_cloud_min=self.M_cloud_min,
            M_cloud_max=self.M_cloud_max,
            cloud_alpha=self.cloud_alpha,
            **self.base_config
        )
        
        try:
            # Initialize model - pass eta parameters directly too
            model = WindModel(
                SFR=sfr,
                v_circ=v_circ,
                r_50=r50,
                eta_M=eta_M,
                eta_M_cold=eta_M_cold,
                eta_E=eta_E,
                config=config
            )
            
            # Run integration
            solution = model.run()
            
            # Check if solution is None
            if solution is None:
                print(f"  Solution is None")
                return None, None, None
            
            # Check for failed solutions (Solution object has sol attribute)
            if hasattr(solution, 'sol') and hasattr(solution.sol, 'status') and solution.sol.status == -1:
                print(f"  Solution failed with status -1")
                return None, None, None
                
            # Compute observables
            return self.compute_observables(model, solution)
            
        except Exception as e:
            print(f"Model evaluation failed: {e}")
            return None, None, None
            
    def compute_observables(self, model, solution):
        """
        Extract velocity centroid, width, and column density from model.
        
        Following Xinfeng_data approach for consistency.
        
        Parameters
        ----------
        model : WindModel
            The wind model instance
        solution : OdeResult
            Integration solution
            
        Returns
        -------
        tuple
            (log_v, log_width, log_NH) in CGS units
        """
        # Calculate dN/dv distribution using corrected method
        # Use similar range as Xinfeng_data (up to 5 kpc)
        v_centers, dN_dv = calculate_column_density_distribution(
            solution,
            r_min_kpc=0.05,
            r_max_kpc=5.0  # Xinfeng uses 5 kpc (line 456: ir10 = 5*kpc)
        )
        
        # Remove any NaN or invalid values
        valid = np.isfinite(dN_dv) & (dN_dv > 0)
        if not np.any(valid):
            return None, None, None
            
        v_centers = v_centers[valid]
        dN_dv = dN_dv[valid]
        
        # Check if we have reasonable velocity range
        if len(v_centers) < 3 or v_centers.max() < 10:  # km/s
            return None, None, None
        
        # Compute moments following Xinfeng_data (lines 510-512)
        # First moment: velocity centroid
        first_moment = np.trapz(dN_dv * v_centers, v_centers)
        zeroth_moment = np.trapz(dN_dv, v_centers)
        
        if zeroth_moment <= 0:
            return None, None, None
            
        v_mean = first_moment / zeroth_moment
        
        # Find peak for HWHM calculation (Xinfeng_data line 517)
        peak_idx = np.argmax(dN_dv)
        v_peak = v_centers[peak_idx]
        
        # Compute HWHM following Xinfeng_data approach
        v_width = self.compute_hwhm_xinfeng(v_centers, dN_dv, v_peak)
        
        # Compute total column density by integration
        # This is the critical fix - use trapz not sum!
        NH_total = zeroth_moment  # This is already the integral ∫ dN/dv dv
        
        # Convert to log scale (CGS units)
        log_v = np.log10(np.abs(v_mean) * 1e5)      # km/s -> cm/s
        log_width = np.log10(v_width * 1e5)         # km/s -> cm/s
        log_NH = np.log10(NH_total)                 # cm^-2
        
        return log_v, log_width, log_NH
        
    def compute_hwhm_xinfeng(self, v, dN_dv, v_peak):
        """
        Compute HWHM following Xinfeng_data approach.
        
        Parameters
        ----------
        v : array
            Velocity array [km/s]
        dN_dv : array
            Column density distribution
        v_peak : float
            Peak velocity [km/s]
            
        Returns
        -------
        float
            HWHM in km/s
        """
        # Find peak and minimum values (Xinfeng lines 515-516)
        max_dN_dv = np.max(dN_dv)
        min_dN_dv = np.min(dN_dv)
        
        # Half maximum between min and max (Xinfeng line 525)
        half_max = min_dN_dv + (max_dN_dv - min_dN_dv) / 2.0
        
        # Find where dN/dv crosses half maximum
        peak_idx = np.argmax(dN_dv)
        
        # Find left side of HWHM
        left_side = dN_dv[:peak_idx]
        if len(left_side) > 0:
            left_idx = np.argmin(np.abs(left_side - half_max))
        else:
            left_idx = 0
            
        # HWHM is difference between peak and left edge
        v_width = v[peak_idx] - v[left_idx]
        
        # Fallback to RMS if width is too small
        if v_width < 1.0:  # km/s
            v_width = np.sqrt(np.trapz((v - v_peak)**2 * dN_dv, v) / np.trapz(dN_dv, v))
            
        return v_width
    
    def compute_hwhm(self, v, dN_dv, v_center):
        """
        Compute Half Width at Half Maximum of the distribution.
        
        Parameters
        ----------
        v : array
            Velocity array [km/s]
        dN_dv : array
            Column density distribution
        v_center : float
            Central velocity [km/s]
            
        Returns
        -------
        float
            HWHM in km/s
        """
        # Find peak value
        peak_value = np.max(dN_dv)
        half_max = peak_value / 2.0
        
        # Find velocities where dN/dv crosses half maximum
        from scipy.interpolate import interp1d
        
        # Create interpolation function
        f_interp = interp1d(v, dN_dv, kind='linear', fill_value=0, bounds_error=False)
        
        # Find HWHM
        v_fine = np.linspace(v.min(), v.max(), 1000)
        dN_dv_fine = f_interp(v_fine)
        
        # Find where it crosses half maximum
        above_half = dN_dv_fine >= half_max
        if np.any(above_half):
            v_half = v_fine[above_half]
            hwhm = max(abs(v_half.max() - v_center), abs(v_center - v_half.min()))
        else:
            # Fallback to RMS width
            hwhm = np.sqrt(np.sum((v - v_center)**2 * dN_dv) / np.sum(dN_dv))
            
        return hwhm


def log_prior(theta):
    """
    Uniform prior on fitting parameters.
    
    Parameters
    ----------
    theta : array
        [eta_M, eta_M_cold, eta_E]
        
    Returns
    -------
    float
        Log prior probability
    """
    eta_M, eta_M_cold, eta_E = theta
    
    # Physical bounds (updated ranges)
    if not (1e-4 <= eta_M <= 1e2):
        return -np.inf
    if not (1e-4 <= eta_M_cold <= 1e2):
        return -np.inf
    if not (1e-4 <= eta_E <= 1):
        return -np.inf
        
    return 0.0


def log_likelihood(theta, observed, errors, galaxy_props, fitter):
    """
    Gaussian likelihood comparing model to observations.
    
    Parameters
    ----------
    theta : array
        [eta_M, eta_M_cold, eta_E]
    observed : tuple
        (v_obs, width_obs, NH_obs) in log scale
    errors : tuple
        (v_err, width_err, NH_err) in log scale
    galaxy_props : tuple
        (sfr, v_circ, r50)
    fitter : CLASSYFitter
        Fitter instance with configuration
        
    Returns
    -------
    float
        Log likelihood
    """
    # Unpack parameters
    eta_M, eta_M_cold, eta_E = theta
    v_obs, width_obs, NH_obs = observed
    v_err, width_err, NH_err = errors
    sfr, v_circ, r50 = galaxy_props
    
    # Evaluate model
    model_results = fitter.evaluate_model(
        eta_M, eta_M_cold, eta_E, sfr, v_circ, r50
    )
    
    # Handle failed models
    if model_results[0] is None:
        return -np.inf
        
    v_model, width_model, NH_model = model_results
    
    # Compute chi-squared
    chi2 = 0
    chi2 += ((v_model - v_obs) / v_err)**2
    chi2 += ((width_model - width_obs) / width_err)**2
    chi2 += ((NH_model - NH_obs) / NH_err)**2
    
    return -0.5 * chi2


def log_probability(theta, observed, errors, galaxy_props, fitter):
    """
    Log posterior probability.
    
    Parameters
    ----------
    theta : array
        [eta_M, eta_M_cold, eta_E]
    observed : tuple
        Observed values
    errors : tuple
        Observation errors
    galaxy_props : tuple
        Galaxy properties
    fitter : CLASSYFitter
        Fitter instance
        
    Returns
    -------
    float
        Log posterior probability
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, observed, errors, galaxy_props, fitter)


def run_mcmc_single_galaxy(galaxy_id, galaxy_data, n_walkers=32, n_steps=5000, 
                          n_threads=1, progress=True):
    """
    Run MCMC for a single galaxy.
    
    Parameters
    ----------
    galaxy_id : str
        Galaxy identifier
    galaxy_data : dict
        Dictionary with observations and properties
    n_walkers : int
        Number of MCMC walkers
    n_steps : int
        Number of MCMC steps
    n_threads : int
        Number of parallel threads
    progress : bool
        Show progress bar
        
    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler with chains
    """
    # Extract data for this galaxy
    observed = (galaxy_data['log_v'], galaxy_data['log_width'], galaxy_data['log_NH'])
    errors = (galaxy_data['log_v_err'], galaxy_data['log_width_err'], galaxy_data['log_NH_err'])
    galaxy_props = (galaxy_data['sfr'], galaxy_data['v_circ'], galaxy_data['r50'])
    
    # Initialize fitter
    fitter = CLASSYFitter(galaxy_data, n_cloud_species=10)
    
    # Setup initial positions
    ndim = 3  # eta_M, eta_M_cold, eta_E
    initial = np.array([0.3, 1.0, 0.1])  # Initial guess
    pos = initial + 1e-2 * np.random.randn(n_walkers, ndim)
    
    # Ensure initial positions are within priors
    for i in range(n_walkers):
        while not np.isfinite(log_prior(pos[i])):
            pos[i] = initial + 1e-2 * np.random.randn(ndim)
    
    # Setup sampler
    if n_threads > 1:
        with Pool(n_threads) as pool:
            sampler = emcee.EnsembleSampler(
                n_walkers, ndim, log_probability,
                args=(observed, errors, galaxy_props, fitter),
                pool=pool
            )
            sampler.run_mcmc(pos, n_steps, progress=progress)
    else:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_probability,
            args=(observed, errors, galaxy_props, fitter)
        )
        sampler.run_mcmc(pos, n_steps, progress=progress)
    
    print(f"Galaxy {galaxy_id} completed. Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    return sampler


def save_results(galaxy_id, sampler, output_dir='results'):
    """
    Save MCMC results for a galaxy.
    
    Parameters
    ----------
    galaxy_id : str
        Galaxy identifier
    sampler : emcee.EnsembleSampler
        The sampler with chains
    output_dir : str
        Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get chains (discard burn-in)
    n_burn = min(1000, sampler.iteration // 4)
    chain = sampler.get_chain(discard=n_burn, flat=True)
    
    # Compute percentiles
    percentiles = np.percentile(chain, [16, 50, 84], axis=0)
    
    # Create results dictionary
    results = {
        'galaxy_id': galaxy_id,
        'chain': chain,
        'eta_M': {
            'median': percentiles[1, 0],
            'lower': percentiles[0, 0],
            'upper': percentiles[2, 0]
        },
        'eta_M_cold': {
            'median': percentiles[1, 1],
            'lower': percentiles[0, 1],
            'upper': percentiles[2, 1]
        },
        'eta_E': {
            'median': percentiles[1, 2],
            'lower': percentiles[0, 2],
            'upper': percentiles[2, 2]
        },
        'acceptance_fraction': sampler.acceptance_fraction.mean(),
        'n_steps': sampler.iteration
    }
    
    # Save as numpy file
    np.save(f'{output_dir}/{galaxy_id}_mcmc_results.npy', results, allow_pickle=True)
    
    return results


def create_diagnostic_plots(galaxy_id, sampler, output_dir='plots', analyze_degeneracy=True):
    """
    Create diagnostic plots for MCMC results.
    
    Parameters
    ----------
    galaxy_id : str
        Galaxy identifier
    sampler : emcee.EnsembleSampler
        The sampler with chains
    output_dir : str
        Output directory for plots
    analyze_degeneracy : bool
        Whether to perform degeneracy analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get chains
    n_burn = min(1000, sampler.iteration // 4)
    chain = sampler.get_chain(discard=n_burn, flat=True)
    
    # Standard corner plot
    fig = corner.corner(
        chain,
        labels=[r'$\eta_M$', r'$\eta_{M,cold}$', r'$\eta_E$'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        levels=(0.68, 0.95),  # 1σ, 2σ contours
    )
    fig.suptitle(f'Galaxy {galaxy_id}', fontsize=14)
    fig.savefig(f'{output_dir}/{galaxy_id}_corner.pdf', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # Enhanced degeneracy analysis if requested
    if analyze_degeneracy:
        try:
            from analyze_degeneracies import (
                create_enhanced_corner_plot, 
                create_degeneracy_heatmap,
                analyze_degeneracies
            )
            
            # Create enhanced corner plot
            create_enhanced_corner_plot(chain, galaxy_id, output_dir=output_dir)
            
            # Create correlation heatmap
            create_degeneracy_heatmap(chain, galaxy_id, output_dir=output_dir)
            
            # Perform degeneracy analysis
            deg_results = analyze_degeneracies(
                chain, 
                param_names=['eta_M', 'eta_M_cold', 'eta_E'],
                output_file=f'{output_dir}/{galaxy_id}_degeneracy_analysis.txt'
            )
            
        except ImportError:
            print("Note: Enhanced degeneracy analysis not available. Install analyze_degeneracies module.")
    
    # Trace plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    labels = [r'$\eta_M$', r'$\eta_{M,cold}$', r'$\eta_E$']
    
    for i in range(3):
        axes[i].plot(sampler.get_chain()[:, :, i], alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel('Step' if i == 2 else '')
        
    fig.suptitle(f'Galaxy {galaxy_id} - Chain Evolution', fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/{galaxy_id}_chains.pdf', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Diagnostic plots saved for galaxy {galaxy_id}")


def check_convergence(sampler, threshold=50):
    """
    Check if MCMC chains have converged.
    
    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The sampler to check
    threshold : int
        Threshold for autocorrelation time
        
    Returns
    -------
    converged : bool
        Whether chains have converged
    tau : array or None
        Autocorrelation times
    """
    try:
        tau = sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * threshold < sampler.iteration)
        return converged, tau
    except:
        return False, None


if __name__ == "__main__":
    # Example usage
    print("CLASSY EMCEE Fitting Module")
    print("Run 'python fitting/run_classy_fits.py' to fit all galaxies")