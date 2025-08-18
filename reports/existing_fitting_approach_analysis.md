# Comprehensive Report: Existing CLASSY Fitting Approach Analysis

## Overview
The Xinfeng_Data directory contains a previous implementation for fitting multiphase galactic wind models to CLASSY galaxy observations. This approach uses MCMC (emcee) to fit model parameters to observed column density distributions.

## 1. Data Sources

### 1.1 Observational Data
- **Out10.0_NH_Profile_Info.txt**: Contains velocity profiles from 50 CLASSY galaxies
  - Mean cloud velocity (km/s) - typically -20 to -700 km/s
  - HWHM (Half Width at Half Maximum) - typically 20-180 km/s
  - Total log NH (column density) - typically 19.6-21.3 log(cm^-2)
  - Integration ranges for NH calculations
  - Some galaxies marked as -100 (no valid measurements)

### 1.2 Galaxy Properties (Out5.3_AncillaryParams.txt)
- **50 CLASSY galaxies** with:
  - Redshift (z) - ranging from 0.0017 to 0.182
  - Half-light radius (r50) - 0.08 to 8.86 arcsec / 0.02 to 7.42 kpc
  - Circular velocity (v_circ) - 10.6 to 159.5 km/s with errors
  - Metallicity proxies
  - Star formation rates (from Out6.5_HLSP.xlsx)

### 1.3 Outflow Rates (Out5.1_HLSP.xlsx)
- Mass outflow rate (Mdot) in Msun/yr
- Momentum flux (Pdot) in log(dynes)
- Energy flux (Edot) in log(erg/s)
- Outflow velocity and FWHM measurements
- Column densities

## 2. Previous Fitting Implementation

### 2.1 Core Architecture (Multiphase_Wind_Fitting_function_for_Classy.py)

**Key Function: `calculate_moments`**
```python
def calculate_moments(eta_M, eta_M_cold_tot, eta_E, sfr_gal, v_circ_gal, 
                      r50, M_cloud0, M_cloud1, cloud_alpha, Method)
```

This function:
1. Interfaces with the wind model using input parameters
2. Computes velocity distribution (dN/dv)
3. Returns three moments or three observables depending on Method:
   - Method 001-003: Returns mass/momentum/energy flux moments
   - Method 001_FitN: Returns log(v), log(FWHM), log(NH)

**Integration Strategy:**
- Uses solve_ivp with adaptive stepping
- Implements multiple event detectors (supersonic, negative velocity, cold wind)
- Cloud mass distribution: power-law with fixed slope α

### 2.2 MCMC Implementation (Call_MWFF_CLASSY.py)

**Prior Setup:**
```python
def log_prior(theta):
    eta_M, eta_M_cold_tot, log_M_cloud0 = theta
    if 0.0 < eta_M < 10 and 0.0 < eta_M_cold_tot < 200 and 0.01 < log_M_cloud0 < 10:
        return 0.0
    return -np.inf
```

**Likelihood Function:**
```python
def log_likelihood(theta, x, y, yerr, galaxy_properties, Method):
    # Calls calculate_moments
    # Computes chi-squared between model and observations
    # Returns log-likelihood
```

**Fitting Parameters:**
- eta_M: Hot phase mass loading (0-10)
- eta_M_cold_tot: Total cold phase mass loading (0-200)
- log_M_cloud0: Log minimum cloud mass (0.01-10)
- Fixed: Maximum cloud mass (10^6 Msun), slope α=2

### 2.3 Observables Comparison

The code supports two measurement methods:
1. **Method 1**: Uses CLASSY III paper measurements (Vout, FWHMout, total NH)
2. **Method 2**: Uses dN/dV profile measurements (abs(Vout), HWHM, same NH)

Both convert observables to log scale for fitting:
- N1 = log10(Vout * 10^5) [cm/s]
- N2 = log10(FWHMout * 10^5) [cm/s]
- N3 = log(NH) [cm^-2]

## 3. Key Physics Implementation

### 3.1 Cloud Distribution
- Power-law mass distribution: dN/dM ∝ M^(-α)
- Typically 10-20 cloud species logarithmically spaced
- Mass range: 10^1 to 10^6 Msun (previously fitted)

### 3.2 Wind Evolution
- Coupled ODEs for hot wind + N cloud species
- State vector: [v_wind, rho_wind, P, rhoZ_wind, M_cloud_i, v_cloud_i, Z_cloud_i]
- Mass exchange via turbulent radiative mixing layers (TRMLs)

### 3.3 Termination Events
- Supersonic/subsonic transitions
- Wind velocity becomes negative
- Temperature drops to cloud temperature
- All clouds freeze out (M < M_min)

## 4. Strengths of Previous Approach

1. **Comprehensive Data**: Full CLASSY sample with detailed measurements
2. **Flexible Observable Fitting**: Can fit to either moments or direct observables
3. **MCMC Robustness**: Uses emcee for proper parameter exploration
4. **Physical Model**: Based on Fielding & Bryan physics

## 5. Limitations and Issues

### 5.1 Code Organization
- Hardcoded paths and parameters scattered throughout
- No clear configuration management
- Missing error handling for failed integrations

### 5.2 Performance
- Uses older scipy.integrate.ode instead of solve_ivp
- No parallelization of model evaluations
- Cooling function likely not optimized

### 5.3 Parameter Space
- Cloud mass range was fitted (adds complexity)
- Very wide prior ranges (eta_M_cold up to 200)
- No handling of numerical stiffness at extreme parameters

### 5.4 Missing Features
- No visualization of posterior distributions
- Limited diagnostic outputs
- No systematic parameter study capabilities

## 6. Data Quality Assessment

### Good Quality Galaxies (42/50):
- Have valid velocity, HWHM, and NH measurements
- Span wide range of properties (SFR, v_circ, metallicity)

### Missing Data (8/50):
- J0337-0502, J0405-3648, J0934+5514, J0944+3424
- J1044+0353, J1113+2930, J1323-0132, J1612+0817
- Marked with -100 values in velocity profiles

## 7. Key Insights for New Implementation

1. **Fixed Parameters Should Include**:
   - Cloud mass range: 10^1 to 10^6 Msun
   - Power-law slope: α = 2
   - All cloud properties except mass loading

2. **Critical Observables**:
   - Focus on velocity centroid and width from dN/dv
   - Column density as key constraint
   - Integration ranges important for NH calculations

3. **Parameter Ranges from Data**:
   - SFR: 0.1 to 100 Msun/yr
   - v_circ: 10 to 160 km/s  
   - r50: 0.02 to 7.4 kpc

4. **Numerical Considerations**:
   - Need robust handling of stiff parameter combinations
   - Event detection critical for physical solutions
   - Log-space comparisons reduce numerical issues