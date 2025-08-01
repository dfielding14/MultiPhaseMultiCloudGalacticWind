# MultiPhase MultiCloud Galactic Wind Model

A fast Python package for simulating multiphase galactic winds with embedded clouds, optimized for MCMC fitting and observational comparisons.

Based on: **"The Structure of Multiphase Galactic Winds"** by Drummond B. Fielding & Greg L. Bryan (*Astrophysical Journal*, 2024)

## Features

- **Clean API**: Simple `WindModel` class with configurable physics via `WindConfig`
- **Fast Integration**: Optimized for MCMC parameter exploration (19,000x faster cooling)
- **Observables**: Built-in velocity and column density distributions
- **Publication-Ready Plots**: Matplotlib-based plotting with customizable styles
- **Flexible Configuration**: All physics parameters easily adjustable
- **No Global Variables**: Clean module design with proper parameter flow
- **Flexible Initial Conditions**: Support for cloud injection with radial offset from sonic point

## Installation

```bash
# Clone the repository
git clone https://github.com/dfielding14/MultiPhaseMultiCloudGalacticWind.git
cd MultiPhaseMultiCloudGalacticWind

# Install the package
pip install -e .

# Install dependencies
pip install numpy scipy matplotlib cmasher h5py
```

## Quick Start

```python
from multiphasegalacticwind import WindModel, WindConfig

# Create and run a basic wind model
model = WindModel(SFR=20.0)  # Star formation rate in Msun/yr
solution = model.run()

# Get key results
print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")

# Plot the solution
from multiphasegalacticwind import plot_wind_solution
fig, axes = plot_wind_solution(solution)
```

## Configuration System

The package uses a flexible configuration system via `WindConfig` to manage all model parameters:

```python
from multiphasegalacticwind import WindModel, WindConfig

# Method 1: Create custom configuration
config = WindConfig(
    f_turb0=0.2,           # Turbulent mixing efficiency
    drag_coeff=0.3,        # Cloud drag coefficient  
    metallicity=0.5,       # Wind metallicity (solar units)
    mu=0.62,               # Mean molecular weight
)
model = WindModel(SFR=10.0, config=config)

# Method 2: Pass parameters directly as kwargs (forwarded to WindConfig)
model = WindModel(
    SFR=10.0,
    f_turb0=0.2,
    drag_coeff=0.3,
    metallicity=0.5
)

# Method 3: Mix both approaches
config = WindConfig(f_turb0=0.2)
model = WindModel(SFR=10.0, config=config, drag_coeff=0.3)

# View default configuration
from multiphasegalacticwind import get_default_config
default_config = get_default_config()
print(default_config.to_dict())
```

## Key Parameters

### Galaxy and Wind Properties
```python
model = WindModel(
    # Galaxy properties
    v_circ=150.0,              # Circular velocity [km/s]
    redshift=0.0,              # Redshift
    
    # Wind launch properties
    SFR=20.0,                  # Star formation rate [Msun/yr]
    eta_M=0.1,                 # Hot phase mass loading
    eta_M_cold=1.0,            # Cold phase mass loading
    eta_E=1.0,                 # Energy loading
    
    # Sonic point properties
    r_star_kpc=0.3,            # Sonic radius [kpc]
    Z_star=1.0,                # Initial metallicity (solar)
    
    # Cloud properties
    cloud_mass_range=(1, 1e5), # Cloud mass range [Msun]
    cloud_alpha=2.0,           # Power law slope
    N_cloud_species=10,        # Number of cloud mass bins
    T_cl=1e4,                  # Cloud temperature [K]
    
    # Solver settings
    r_max_kpc=100.0,           # Maximum radius [kpc]
    rtol=1e-8,                 # Relative tolerance
    atol=1e-10,                # Absolute tolerance
)
```

### Configurable Physics Parameters

All physics parameters can be customized via `WindConfig`:

```python
# Common parameters to modify
config = WindConfig(
    # Thermodynamics
    mu=0.62,                   # Mean molecular weight
    gamma=5/3,                 # Adiabatic index
    
    # Mixing and turbulence
    f_turb0=0.1,               # Turbulent mixing efficiency (key parameter!)
    drag_coeff=0.5,            # Cloud drag coefficient
    Mdot_coefficient=1/3,      # Mass transfer coefficient
    
    # Cooling
    metallicity=1.0,           # Metallicity (solar units)
    Cooling_Factor=1.0,        # Cooling strength (0=off, 1=standard)
    
    # Cloud evolution
    M_cloud_min=0.01,          # Minimum cloud mass [Msun]
    geometric_factor=1.0,      # Cloud geometry factor
    
    # Power law indices
    TurbulentVelocityChiPower=0.0,  # v_turb ∝ χ^α
    CoolingAreaChiPower=0.5,        # A_cool ∝ χ^β
    ColdTurbulenceChiPower=-0.5,    # v_turb_cold ∝ χ^γ
)
```

See `docs/windconfig_parameters.md` for complete parameter documentation.

## Calculating Observables

### Velocity Distributions

Calculate velocity distributions for comparison with absorption line observations:

```python
# Basic velocity distribution (number per velocity)
v_cloud, dN_dv = solution.calculate_velocity_distribution()

# Column density distribution in standard observational units
v_cloud, dN_dv_column = solution.calculate_column_density_distribution()
# Returns dN/dv in cm^-2 / (km/s) - ready for comparison with observations

# Get contributions from each cloud species separately
v_cloud, dN_dv_dict = solution.calculate_column_density_by_species()
# Returns dict with:
#   'total': Total column density distribution
#   'species': List of distributions for each cloud mass
#   'M_cloud0': Initial cloud masses

# Calculate velocity moments for characterizing distributions
moments = solution.calculate_velocity_moments()
print(f"Mean velocity: {moments['mean']:.1f} km/s")
print(f"Velocity dispersion: {moments['dispersion']:.1f} km/s")
print(f"Skewness: {moments['skewness']:.2f}")
print(f"Kurtosis: {moments['kurtosis']:.2f}")
```

### Plotting

Create publication-quality plots:

```python
from multiphasegalacticwind import (
    plot_wind_solution,
    plot_velocity_distribution,
    plot_column_density_distribution
)

# Plot wind evolution
fig1, axes = plot_wind_solution(solution)

# Plot velocity distribution
fig2, ax = plot_velocity_distribution(solution, show_moments=True)

# Plot column density distribution
fig3, ax = plot_column_density_distribution(solution, log_scale=True)
```

## Examples

See the `examples/` directory for complete examples:
- `simple_example.py` - Minimal working example
- `basic_example.py` - Parameter studies and plotting  
- `observational_comparison.py` - Velocity distributions
- `column_density_example.py` - Column density calculations
- `tutorial_wind_models.ipynb` - Comprehensive Jupyter notebook tutorial
- `observational_comparison_notebook.ipynb` - Mock observations notebook

## Physical Model

This code simulates the steady-state evolution of multiphase galactic winds with:
- A hot, volume-filling phase that drives the wind
- Cold embedded clouds that exchange mass, momentum, and energy
- Turbulent radiative mixing layers (TRMLs) mediating the interaction
- Radiative cooling and heating via tabulated cooling functions
- Gravitational deceleration in an isothermal potential

The model solves coupled ODEs for:
- **Hot phase**: Euler equations with cooling/heating and cloud interactions
- **Cold clouds**: Mass exchange, drag, and radiative acceleration

### Key Physical Processes

1. **Turbulent Mixing**: Mass transfer rate ∝ f_turb × v_rel × A_cloud / r_cloud
2. **Radiative Cooling**: Using Wiersma+09 cooling tables (redshift-dependent)
3. **Cloud Drag**: F_drag = C_d × ρ_hot × v_rel² × A_cloud
4. **Sonic Point**: Calculated from energy and mass conservation (not user-specified)

### Key Assumptions
- Steady-state solutions (∂/∂t = 0)
- Spherical symmetry within solid angle Ω_wind
- Pressure equilibrium between phases
- Clouds maintain T_cl = 10^4 K via rapid cooling
- Power-law cloud mass distribution: dN/dM ∝ M^(-α)

## Performance Notes

For MCMC applications or parameter studies, consider adjusting tolerances:
```python
# Faster integration (suitable for parameter exploration)
model = WindModel(
    SFR=10.0,
    rtol=1e-6,      # Default is 1e-8
    atol=1e-8,      # Default is 1e-10
    r_max_kpc=50    # Default is 100 kpc
)

# For production runs with strict accuracy
model = WindModel(
    SFR=10.0,
    rtol=1e-10,
    atol=1e-12
)
```

**Note**: The default tolerances (rtol=1e-8) may cause long integration times for some parameter combinations. We recommend starting with rtol=1e-6 for initial explorations.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fielding2024multiphase,
  title={The Structure of Multiphase Galactic Winds},
  author={Fielding, Drummond B. and Bryan, Greg L.},
  journal={Astrophysical Journal},
  year={2024}
}
```

## Architecture Notes

### Module Structure
- `wind_model.py` - High-level API and `WindModel` class
- `config.py` - Configuration via `WindConfig` class
- `constants.py` - Physical constants (kb, mp, Msun, etc.)
- `core_physics.py` - ODE system and physics functions
- `cooling.py` - Cooling functions and interpolators
- `observables.py` - Velocity and column density distributions
- `plotting.py` - Publication-quality plotting functions

### Key Design Principles
- No global variables - all parameters flow through config
- Lazy loading - cooling tables load on first use
- Clean separation - physics, configuration, and plotting are separate
- Type safety - parameters validated in WindConfig

## Migration from Legacy Code

If you're migrating from the old script-based approach, see `docs/migration_guide.md` for detailed instructions. Key changes:
- Parameters now managed by `WindConfig` (no globals)
- Sonic point calculated from physics (not user-specified)
- Cooling functions moved to separate module
- Clean API via `WindModel` class

## License

This code is released under the MIT License. See LICENSE file for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: Drummond Fielding (dfielding@flatironinstitute.org)