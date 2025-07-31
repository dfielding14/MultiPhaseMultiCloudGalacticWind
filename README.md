# MultiPhase MultiCloud Galactic Wind Model

A fast Python package for simulating multiphase galactic winds with embedded clouds, optimized for MCMC fitting.

Based on: **"The Structure of Multiphase Galactic Winds"** by Drummond B. Fielding & Greg L. Bryan (*Astrophysical Journal*, 2024)

## Overview

This code simulates the steady-state evolution of multiphase galactic winds consisting of a hot, volume-filling phase and cold, embedded clouds. The model accounts for bidirectional mass, momentum, and energy exchange between phases mediated by turbulent radiative mixing layers (TRMLs), providing a physically realistic treatment of galactic wind structure and evolution.

## Physical Model

### Hot Phase Evolution

The hot, volume-filling wind component evolves according to:

**Mass Conservation:**
```
∂/∂r(r² ρ v) = ρ̇
```

**Momentum Conservation:**
```
∂/∂r(r² ρ v²) + ∂P/∂r = -ρ v_c²/r + ṗ
```

**Energy Conservation:**
```
∂/∂r[r² ρ v (½v² + γ/(γ-1) P/ρ - ½v_esc²)] = ε̇ - L
```

where:
- `ρ`, `v`, `P` are hot phase density, velocity, and pressure
- `ρ̇`, `ṗ`, `ε̇` are source/sink terms from cloud-wind interaction
- `L = n²Λ - nΓ` represents radiative cooling/heating
- `v_c` is the circular velocity, `v_esc` is the escape velocity

### Cloud Evolution

Cold clouds evolve through mass exchange with the hot phase:

**Mass Evolution:**
```
Ṁ_cl = 3 f_turb f_cool (M_cl v_rel)/(χ^(1/2) r_cl) (ξ^α - 1)
```

where:
- `ξ = r_cl/(v_turb τ_cool)` controls growth (ξ > 1) vs destruction (ξ < 1)
- `α = 1/4` for rapid cooling (ξ ≥ 1), `α = 1/2` for slow cooling (ξ < 1)
- `χ = ρ_cl/ρ` is the density contrast
- `f_turb ≈ 0.1` is the turbulent mixing efficiency
- `f_cool = 1/3` accounts for cloud elongation

**Velocity Evolution:**
```
v̇_cl = (v-v_cl) Ṁ_cl,grow/M_cl + (3C_drag/8)(v-v_cl)²/(χ r_cl) - v_c²/r
```

**Metallicity Evolution:**
```
Ż_cl = (Z-Z_cl) Ṁ_cl,grow/M_cl
```

### Key Physical Parameters

- **Hot phase mass loading**: `η_M = Ṁ_wind/Ṁ_star`
- **Cold phase mass loading**: `η_M,cold = Ṁ_cold/Ṁ_star`
- **Energy loading**: `η_E = Ė_wind/Ė_star`
- **Cloud temperature**: `T_cl = 10^4 K` (photoionization equilibrium)
- **Sonic radius**: `r_star = 300 pc` (typical)

## Code Structure

### Main Components

1. **`Multiphase_Wind_Evolution.py`** - Original single-cloud simulation module containing:
   - Physical constants and unit definitions
   - Cooling function interpolations (`cooling_function_*`)
   - Wind evolution ODEs (`Wind_Evo`, `Hot_Wind_Evo`)
   - Cloud dynamics (`dMcl_dt`, `dvcl_dt`, `dZcl_dt`)
   - Field length calculations for thermal conduction
   - Integration routines with termination conditions

2. **`Multiphase_Wind_Evolution_Multicloud.py`** - Extended multicloud version supporting:
   - Multiple cloud species with power-law mass distributions (dN/dM ∝ M^-α)
   - Cloud population setup function (`setup_cloud_powerlaw_distribution`)
   - Statistical analysis tools (`calculate_cloud_moments`, `get_cloud_mass_spectrum`)
   - Full backward compatibility with single-cloud mode

3. **Analysis Notebooks**:
   - **`Multiphase_Wind_Fitting_function_for_Classy.ipynb`** - Generates fitting functions for galaxy evolution models and computes velocity moments
   - **`multicloud.ipynb`** - Interactive multicloud simulations and analysis

4. **Data Files**:
   - **`Lambda_tab_redshifts.npz`** - Pre-computed cooling function lookup table (81×352×15×49 array) with metallicity and redshift dependence

### Key Functions

- `Wind_Evo(s, y, params)` - Main ODE system for supersonic wind region
- `Hot_Wind_Evo(s, y, params)` - ODE system for hot phase only
- `cooling_function_Wiersma(*)` - Cooling curves with metallicity/redshift dependence
- `dMcl_dt()` - Cloud mass evolution rate
- `field_length_*()` - Various thermal conduction prescriptions
- `calculate_moments()` - Velocity moment calculations (notebook)

## Usage

### Single Cloud Example

```python
import numpy as np
from Multiphase_Wind_Evolution import *

# Set initial conditions at sonic radius
n_star = 0.1  # cm^-3, hot phase density
v_star = 200  # km/s, initial velocity
T_star = 5e6  # K, hot phase temperature
Z_star = 1.0  # solar metallicity
eta_M = 0.1   # hot phase mass loading
eta_M_cold = 1.0  # cold phase mass loading

# Cloud properties
r_cl_0 = 10  # pc, initial cloud radius
n_cl = 100   # cm^-3, cloud density

# Run simulation (see notebooks for complete examples)
```

### Multicloud Example

```python
from Multiphase_Wind_Evolution_Multicloud import *

# Set up power-law cloud distribution
log_M_cloud_min = 0   # 10^0 = 1 Msun
log_M_cloud_max = 5   # 10^5 Msun  
N_cloud_species = 10  # Number of mass bins
alpha_cloud = 2.0     # Power-law exponent

# Generate cloud population
M_cloud0, eta_M_cold, Mdot_cold0, Ndot_cloud0 = setup_cloud_powerlaw_distribution(
    log_M_cloud_min, log_M_cloud_max, N_cloud_species, 
    alpha_cloud=alpha_cloud, eta_M_cold_tot=1.0, SFR=20*Msun/yr
)

# See multicloud.ipynb for complete implementation
```

### Parameter Ranges

Typical parameter ranges for galactic winds:
- Hot phase density: `n ~ 10^-3 - 10^-1 cm^-3`
- Hot phase temperature: `T ~ 10^6 - 10^7 K`
- Wind velocity: `v ~ 100 - 1000 km/s`
- Cloud radius: `r_cl ~ 0.1 - 100 pc`
- Cloud density: `n_cl ~ 10 - 10^4 cm^-3`
- Mass loading: `η_M ~ 0.01 - 10`

## Installation

### Dependencies

```bash
pip install numpy scipy matplotlib cmasher h5py jupyter
```

Required packages:
- `numpy` - Numerical computations
- `scipy` - ODE integration and interpolation
- `matplotlib` - Plotting and visualization
- `cmasher` - Scientific colormaps (replaces deprecated palettable)
- `h5py` - For HDF5 data I/O
- `jupyter` - For running analysis notebooks

## Physics Notes

### Key Assumptions

1. **Steady-state** - Time-independent solutions
2. **Spherical symmetry** - Within solid angle Ω_wind
3. **Pressure equilibrium** - Clouds adjust instantaneously
4. **Thermal equilibrium** - Clouds maintain T_cl = 10^4 K
5. **Turbulent mixing** - f_turb = 0.1 (constant)
6. **No thermal conduction** between phases (can be enabled)
7. **No magnetic fields** (future extension)

### Termination Conditions

Simulations terminate when:
- Hot phase cools to cloud temperature (T → T_cl)
- Wind stalls (v → 0)
- Clouds completely evaporate (M_cl → 0)
- Integration reaches large radius (r ~ 100 kpc)

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

## Installation

```bash
# Clone the repository
git clone https://github.com/dfielding14/MultiPhaseMultiCloudGalacticWind.git
cd MultiPhaseMultiCloudGalacticWind

# Install the package
pip install -e .

# For plotting features (optional)
pip install -e ".[plotting]"
```

## Quick Start

```python
from multiphasegalacticwind import WindModel, plot_wind_solution

# Create a wind model
model = WindModel(
    SFR=20.0,           # Star formation rate [Msun/yr]
    eta_M=0.1,          # Hot phase mass loading
    eta_M_cold=1.0,     # Cold phase mass loading
    v_circ=150.0        # Circular velocity [km/s]
)

# Run the simulation
solution = model.run()

# Access results
print(f"Wind velocity at 10 kpc: {solution.v_at_10kpc:.1f} km/s")
print(f"Mass loading at 10 kpc: {solution.mass_loading_at_10kpc:.2f}")

# Create publication-quality plots
fig, axes = plot_wind_solution(solution)
```

See `examples/basic_example.py` for a complete example.

## License

This code is released under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please contact:
- Drummond Fielding (dfielding@flatironinstitute.org)
- Greg Bryan (gbryan@astro.columbia.edu)