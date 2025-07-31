"""
Cooling functions for multiphase galactic wind models.

This module handles all cooling-related calculations including:
- Loading/generating cooling tables
- Interpolating cooling rates  
- Calculating cooling times
"""

import numpy as np
import os
import time
from scipy import interpolate
import h5py
import glob
from .constants import *

# Global cooling table data (initialized on first use)
_Lambda = None
_Lambda_tab = None
_redshifts = None
_Zs = None
_log_Tbins = None
_log_nHbins = None

# Global cooling interpolators
_Lambda_P_rho = None
_Lambda_P_rho_params = None
_T_tcool_min_P = None
_tcool_min_P = None


def get_lambda_interpolator():
    """
    Get the main 4D cooling function interpolator.
    
    Loads the cooling table on first call.
    
    Returns
    -------
    Lambda : RegularGridInterpolator
        4D interpolator for cooling function Lambda(log_nH, log_T, Z, z)
    """
    global _Lambda
    if _Lambda is None:
        load_cooling_table()
    return _Lambda


def load_cooling_table(verbose=False):
    """
    Load the cooling table from package data directory.
    
    Parameters
    ----------
    verbose : bool
        Print loading progress
        
    Returns
    -------
    Lambda : RegularGridInterpolator
        4D cooling function interpolator
    """
    global _Lambda, _Lambda_tab, _redshifts, _Zs, _log_Tbins, _log_nHbins
    
    if _Lambda is not None:
        return _Lambda
    
    # Location of cooling table in package data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    cooling_table_path = os.path.join(data_dir, 'Lambda_tab_redshifts.npz')
    
    # Check if the cooling table exists in the package
    if os.path.exists(cooling_table_path):
        if verbose:
            print(f"Loading cooling table from {cooling_table_path}")
        data = np.load(cooling_table_path)
        _Lambda_tab = data['Lambda_tab']
        _redshifts = data['redshifts']
        _Zs = data['Zs']
        _log_Tbins = data['log_Tbins']
        _log_nHbins = data['log_nHbins']
        _Lambda = interpolate.RegularGridInterpolator(
            (_log_nHbins, _log_Tbins, _Zs, _redshifts), 
            _Lambda_tab, bounds_error=False, fill_value=1e-30
        )
    else:
        raise FileNotFoundError(
            f"Cooling table not found at {cooling_table_path}. "
            "Please ensure the cooling table is generated and placed in the package data directory."
        )
    
    return _Lambda


def get_cooling_interpolator(mu, metallicity, redshift, verbose=False):
    """
    Get or create the cooling table interpolator for given parameters.
    
    This function creates a cooling table interpolator on first call or when
    parameters change, and returns the cached version on subsequent calls.
    
    Parameters
    ----------
    mu : float
        Mean molecular weight
    metallicity : float
        Metallicity relative to solar
    redshift : float
        Redshift for cooling function
    verbose : bool
        Print progress during table generation
        
    Returns
    -------
    Lambda_P_rho : RegularGridInterpolator
        Interpolator for cooling function Lambda(P, rho)
    """
    global _Lambda_P_rho, _Lambda_P_rho_params
    
    # Check if we need to regenerate the table
    params = (mu, metallicity, redshift)
    if _Lambda_P_rho is None or _Lambda_P_rho_params != params:
        if verbose:
            print(f"Generating cooling interpolator for mu={mu}, Z={metallicity}, z={redshift}")
        
        # Get the main cooling interpolator
        Lambda = get_lambda_interpolator()
            
        Ps = np.logspace(-8, 10, 100)
        rhos = np.logspace(-10, 5, 101) * mu * mp
        Lambda_P_rho_tab = np.zeros((len(Ps), len(rhos)))
        
        for i in range(len(Ps)):
            for j in range(len(rhos)):
                rho = rhos[j]
                T = Ps[i] * (mu * mp / rho)
                if rho > 1 * muH * mp:
                    rho = 1. * muH * mp
                elif rho < 1e-8 * muH * mp:
                    rho = 1e-8 * muH * mp
                if T > 10**8.98:
                    T = 10**8.98
                elif T < 1e2:
                    T = 1e2
                try:
                    Lambda_P_rho_tab[i,j] = Lambda((np.log10(rho/(muH*mp)), np.log10(T), metallicity, redshift))
                except:
                    Lambda_P_rho_tab[i,j] = 1e-30
            if verbose and i % 10 == 0:
                print(f"  Progress: {i}/{len(Ps)}")
                
        _Lambda_P_rho = interpolate.RegularGridInterpolator(
            (Ps * kb, rhos), Lambda_P_rho_tab, 
            bounds_error=False, fill_value=0.
        )
        _Lambda_P_rho_params = params
        
    return _Lambda_P_rho


def tcool_P(T, P, metallicity, redshift, mu):
    """
    Calculate cooling time as a function of temperature and pressure.
    
    Parameters
    ----------
    T : float or array
        Temperature [K]
    P : float or array
        Pressure [dyne/cm^2]
    metallicity : float
        Metallicity relative to solar
    redshift : float
        Redshift
    mu : float
        Mean molecular weight
        
    Returns
    -------
    tcool : float or array
        Cooling time [s]
    """
    # Get the main cooling interpolator
    Lambda = get_lambda_interpolator()
    
    T = np.where(T > 10**8.98, 10**8.98, T)
    T = np.where(T < 10**2, 10**2, T)
    nH_actual = P/T * (mu/muH)
    nH = np.where(nH_actual > 1, 1, nH_actual)
    nH = np.where(nH < 10**-8, 10**-8, nH)
    
    # Use Lambda interpolator with redshift
    lambda_val = Lambda((np.log10(nH), np.log10(T), metallicity, redshift))
    return 1.5 * (muH/mu) * kb * T / (nH_actual * lambda_val)


def Lambda_P(T, P, metallicity, redshift, mu):
    """
    Calculate cooling function Lambda as a function of temperature and pressure.
    
    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [dyne/cm^2]
    metallicity : float
        Metallicity relative to solar
    redshift : float
        Redshift
    mu : float
        Mean molecular weight
        
    Returns
    -------
    lambda_val : float
        Cooling function value
    """
    # Get the main cooling interpolator
    Lambda = get_lambda_interpolator()
    
    nH = P/T * (mu/muH)
    if nH > 0.9:
        nH = 0.9
    return Lambda((np.log10(nH), np.log10(T), metallicity, redshift))

# Vectorize Lambda_P for array inputs
Lambda_P = np.vectorize(Lambda_P)


def get_tcool_min_interpolators():
    """
    Get or create the minimum cooling time interpolators.
    
    Returns
    -------
    T_tcool_min_P : RegularGridInterpolator
        Temperature at minimum cooling time as function of (P, Z)
    tcool_min_P : RegularGridInterpolator
        Minimum cooling time as function of (P, Z)
    """
    global _T_tcool_min_P, _tcool_min_P
    
    if _T_tcool_min_P is not None:
        return _T_tcool_min_P, _tcool_min_P
    
    # Create pressure and metallicity grids
    Ps = np.logspace(-8, 10, 100)
    Zs = 10.**np.array([0., -0.5, -1., -1.5, -2., -3.])
    T = np.logspace(3.5, 6.5, 1000)
    
    T_tcool_min_array = np.zeros((len(Ps), len(Zs)))
    tcool_min_array = np.zeros((len(Ps), len(Zs)))
    
    # Use default mu for minimum cooling time calculation
    mu_default = 0.62
    
    for i, P in enumerate(Ps):
        if P > 10**4.2:
            continue
        for j, Z in enumerate(Zs):
            tcools = tcool_P(T, P, Z, 0.0, mu_default)
            positive_tcools = tcools[tcools > 0]
            if len(positive_tcools) > 0:
                min_idx = np.where(tcools == np.min(positive_tcools))[0][0]
                T_tcool_min_array[i,j] = T[min_idx]
                tcool_min_array[i,j] = np.min(positive_tcools)
    
    # Fill high pressure values
    high_P_idx = np.where(Ps > 10**4.2)[0]
    if len(high_P_idx) > 0:
        last_valid_idx = high_P_idx[0] - 1
        for i in high_P_idx:
            T_tcool_min_array[i] = T_tcool_min_array[last_valid_idx]
            tcool_min_array[i] = tcool_min_array[last_valid_idx] * (Ps[i] / Ps[last_valid_idx])**-1
    
    _T_tcool_min_P = interpolate.RegularGridInterpolator(
        (Ps, Zs), T_tcool_min_array, bounds_error=False, fill_value=None
    )
    _tcool_min_P = interpolate.RegularGridInterpolator(
        (Ps, Zs), tcool_min_array, bounds_error=False, fill_value=None
    )
    
    return _T_tcool_min_P, _tcool_min_P