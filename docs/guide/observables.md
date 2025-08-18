# Observables Guide

## Overview

The model predicts observable quantities that can be directly compared with spectroscopic data from galaxies.

## Column Density Distribution

### What is dN/dv?

The column density per unit velocity:
$$\frac{dN}{dv} = \int n(r, v) \, dr$$

where integration is along the line of sight for gas at velocity $v$.

### Calculation

```python
# Calculate for all clouds
v_cloud, dN_dv = solution.calculate_column_density_distribution()

# By species
v_cloud, dN_dv_species = solution.calculate_column_density_by_species()
```

### Physical Interpretation

- **Peak location**: Typical outflow velocity
- **Width**: Velocity dispersion
- **Asymmetry**: Acceleration/deceleration
- **Total area**: Total column density

## Velocity Moments

### Mean Velocity

$$\langle v \rangle = \frac{\int v \, dN/dv \, dv}{\int dN/dv \, dv}$$

### Velocity Dispersion

$$\sigma_v = \sqrt{\langle v^2 \rangle - \langle v \rangle^2}$$

### Calculation

```python
moments = solution.calculate_velocity_moments()
print(f"Mean: {moments['mean']} km/s")
print(f"Dispersion: {moments['dispersion']} km/s")
print(f"Skewness: {moments['skewness']}")
```

## Ion Mapping

Different ions trace different gas phases:

| Ion | Temperature [K] | Phase | Typical v [km/s] |
|-----|----------------|-------|------------------|
| Si II | ~10⁴ | Cold clouds | 100-300 |
| C II | ~10⁴ | Cold clouds | 100-300 |
| Si IV | ~10⁵ | Mixing layer | 200-400 |
| O VI | 10⁵-10⁶ | Hot/mixing | 300-600 |

## Line Profiles

### Absorption Lines

For down-the-barrel observations:
$$I(v) = I_0 \exp(-\tau(v))$$

where optical depth:
$$\tau(v) = \frac{\pi e^2}{m_e c} f_{osc} \lambda_0 N(v)$$

### Emission Lines

For transverse observations:
$$L(v) \propto n^2 \Lambda(T) \times \text{Volume}(v)$$

## Mock Observations

### Creating Mock Spectra

```python
from multiphasegalacticwind.observables import create_mock_spectrum

# Generate mock COS spectrum
spectrum = create_mock_spectrum(
    solution,
    ions=['SiII', 'CII', 'OVI'],
    noise_level=0.1,
    resolution=20  # km/s
)
```

### Comparing with Data

```python
# Load observed spectrum
obs_data = load_observation('J0021+0052')

# Calculate chi-squared
chi2 = np.sum((obs_data['flux'] - model_flux)**2 / obs_data['error']**2)
```

## Best Practices

### Resolution Effects

Account for instrumental resolution:
```python
# Convolve with LSF
from scipy.ndimage import gaussian_filter1d
sigma_instrument = 20  # km/s
dN_dv_convolved = gaussian_filter1d(dN_dv, sigma_instrument)
```

### Projection Effects

Consider viewing angle:
- **Down-the-barrel**: Maximum blueshifted absorption
- **Transverse**: Emission lines, P-Cygni profiles
- **Intermediate**: Complex profiles

### Multi-Ion Analysis

Combine multiple ions for better constraints:
```python
ions = ['SiII', 'CII', 'SiIV', 'OVI']
for ion in ions:
    v, N = calculate_ion_column_density(solution, ion)
    plot_comparison(v, N, observed_data[ion])
```

## Common Issues

### Low Signal-to-Noise
- Bin velocity channels
- Stack multiple observations
- Use priors from other ions

### Blending
- Deblend components carefully
- Use multiple transitions
- Check for contamination

### Systematic Effects
- Account for stellar continuum
- Consider ISM absorption
- Check for geocoronal emission

## See Also

- [API: Observables](../api/observables.md)
- [Fitting Guide](../fitting/mcmc_guide.md)
- [Examples](../getting_started/examples.md)