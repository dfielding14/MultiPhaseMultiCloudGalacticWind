# Degeneracy Analysis Quick Reference

## Quick Start

### 1. After fitting a single galaxy:
```bash
python test_single_galaxy_fit.py
python analyze_test_degeneracies.py
```

### 2. After batch fitting:
```python
from analyze_degeneracies import analyze_all_fits
analyze_all_fits('classy_fits/results')
```

## Understanding the Output

### Correlation Matrix
- **|ρ| < 0.3**: No correlation - parameters independent ✓
- **0.3 < |ρ| < 0.5**: Weak correlation - minor degeneracy
- **0.5 < |ρ| < 0.8**: Moderate correlation - significant degeneracy ⚠️
- **|ρ| > 0.8**: Strong correlation - severe degeneracy ⚠️⚠️

### Condition Number
- **< 30**: Well-conditioned, good constraints ✓
- **30-100**: Moderate conditioning, some degeneracies ⚠️
- **> 100**: Ill-conditioned, strong degeneracies ⚠️⚠️

## Interpreting Physical Correlations

### Expected Correlations (OK):
- **eta_M ↔ eta_E positive**: More mass needs more energy ✓
- **eta_M_total well-constrained**: Total mass loading from data

### Problematic Correlations (Need attention):
- **eta_M ↔ eta_M_cold strong negative**: Can't separate hot/cold
- **All three strongly correlated**: Insufficient data constraints

## Solutions for Strong Degeneracies

### Option 1: Reparameterize
```python
# Instead of (eta_M, eta_M_cold, eta_E)
# Use (eta_M_total, f_cold, eta_E)
eta_M_total = eta_M + eta_M_cold
f_cold = eta_M_cold / eta_M_total
```

### Option 2: Fix Ratios
```python
# Fix from theory/simulations
eta_E = 0.1 * eta_M  # Energy ~10% of mass loading
```

### Option 3: Add Priors
```python
# Informative priors from other studies
def log_prior(theta):
    eta_M, eta_M_cold, eta_E = theta
    # Gaussian prior on eta_E/eta_M ratio
    ratio_prior = -0.5 * ((eta_E/eta_M - 0.1) / 0.05)**2
    return ratio_prior
```

## Files Generated

Per galaxy:
- `{galaxy}_enhanced_corner.pdf` - Corner plot with correlations
- `{galaxy}_correlation_heatmap.pdf` - Visual correlation matrix  
- `{galaxy}_degeneracy_analysis.txt` - Numerical results
- `{galaxy}_parameter_evolution.pdf` - MCMC chain evolution

Summary:
- `degeneracy_summary.csv` - All galaxies correlation statistics
- `degeneracy_summary.pdf` - Population-level analysis

## Command Line Usage

### Test mode (3 galaxies, quick):
```bash
python run_classy_fits.py --test --cores 2
python -c "from analyze_degeneracies import analyze_all_fits; analyze_all_fits('classy_fits/results')"
```

### Production (all galaxies):
```bash
python run_classy_fits.py --walkers 64 --steps 10000 --cores 8
python -c "from analyze_degeneracies import analyze_all_fits; analyze_all_fits('classy_fits/results')"
```

### Single galaxy analysis:
```bash
python -c "
import numpy as np
from analyze_degeneracies import create_enhanced_corner_plot
results = np.load('results/J0021+0052_mcmc_results.npy', allow_pickle=True).item()
create_enhanced_corner_plot(results['chain'], 'J0021+0052')
"
```

## Warning Signs in Output

🚨 **Immediate attention needed**:
- "Strong degeneracy detected" 
- Condition number > 100
- Acceptance fraction < 1%

⚠️ **Monitor closely**:
- "Moderate degeneracy"
- Condition number 30-100  
- Acceptance fraction < 10%

✅ **Good to proceed**:
- "Parameters well-constrained"
- Condition number < 30
- Acceptance fraction 20-50%