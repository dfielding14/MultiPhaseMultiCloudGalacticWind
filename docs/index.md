# Multiphase Galactic Wind Model

## What is This?

A Python package for simulating **multiphase galactic winds** - powerful outflows from galaxies driven by stellar feedback. The model tracks hot gas and embedded cold clouds, their interactions through turbulent mixing layers, and predicts observable quantities that can be compared with real data.

## Quick Navigation

### 🚀 **Getting Started**
New to the package? Start here!
- [**Installation**](getting_started/installation.md) - Set up the package
- [**Quick Start**](getting_started/quickstart.md) - Your first wind model

### 📈 **MCMC Fitting** 
Fit models to real galaxy data
- [**Fitting Overview**](fitting/overview.md) - Why and how to fit data
- [**MCMC Guide**](fitting/mcmc_guide.md) - Complete fitting tutorial

### ⚙️ **Run Models**
Configure and run wind simulations
- [**Parameter Guide**](guide/parameters.md) - Understanding all parameters
- [**Examples**](getting_started/examples.md) - Working examples

### ⚛️ **Physics**
Understand the underlying physics
- [**Physics Overview**](physics/overview.md) - Theory behind the model
- [**Equations**](physics/hot_wind_equations.md) - Mathematical formulation

## Key Features

### 🎯 Core Capabilities

- **Multiphase Winds**: Hot gas + embedded cold clouds
- **TRML Physics**: Turbulent Radiative Mixing Layers
- **Observable Predictions**: Column densities, velocity distributions
- **MCMC Fitting**: Constrain parameters with Bayesian inference
- **Fast Integration**: Optimized ODE solver with event detection

### 📊 What You Can Do

1. **Model Galaxy Winds**
   - Simulate outflows from dwarf to starburst galaxies
   - Track hot and cold gas phases
   - Predict terminal velocities and mass loading

2. **Fit Observational Data**  
   - Use MCMC to constrain η_M, η_M_cold, η_E
   - Fit UV absorption lines from CLASSY
   - Quantify parameter uncertainties

3. **Explore Parameter Space**
   - Study how feedback efficiency affects winds
   - Investigate cloud survival and acceleration
   - Compare with observations

## For Different Users

### 🔬 Observers
**"I have galaxy spectra and want to constrain wind properties"**

1. Start with [MCMC Fitting Overview](fitting/overview.md)
2. Follow the [Fitting Guide](fitting/mcmc_guide.md)
3. See [CLASSY examples](fitting/classy_example.md)

### 🌌 Theorists
**"I want to understand the wind physics"**

1. Read [Physics Overview](physics/overview.md)
2. Study [Hot Wind Equations](physics/hot_wind_equations.md)
3. Explore [TRML Physics](physics/trml.md)

### 💻 Modelers
**"I want to run wind simulations"**

1. Follow [Quick Start](getting_started/quickstart.md)
2. Learn [Parameter Selection](guide/parameters.md)
3. Try [Examples](getting_started/examples.md)

### 🔧 Developers
**"I want to contribute or modify the code"**

1. Read [Contributing Guide](development/contributing.md)
2. Review [Code Structure](development/code_review.md)
3. See [Testing Guide](development/testing.md)

## Quick Example

```python
from multiphasegalacticwind import WindModel
import matplotlib.pyplot as plt

# Create a wind model for a starburst galaxy
model = WindModel(
    SFR=100.0,         # Star formation rate [Msun/yr]
    v_circ=200.0,      # Circular velocity [km/s]
    eta_M=0.1,         # Hot mass loading
    eta_M_cold=3.0,    # Cold mass loading
    eta_E=2.0          # Energy loading
)

# Run the simulation
solution = model.run()

# Check results
print(f"Terminal velocity: {solution.v_terminal:.0f} km/s")
print(f"Mass flux at 10 kpc: {solution.mass_flux[solution.r <= 10][-1]:.1f} Msun/yr")

# Visualize
from multiphasegalacticwind.plotting import plot_wind_solution
fig, axes = plot_wind_solution(solution)
plt.show()
```

## Recent Updates ✨

### Version 1.1.0 - Major Improvements
- **Added MCMC fitting** with emcee for parameter constraints
- **Fixed 14 critical bugs** including sonic point crashes
- **Created comprehensive documentation** (you're reading it!)
- **Improved performance** by 19,000× in cooling calculations
- **Added parameter validation** throughout

[View Full Changelog](about/changelog.md)

## The Science

Based on **Fielding & Bryan (2024)** ["The Structure of Multiphase Galactic Winds"](about/references.md#fielding-bryan-2024), this model implements:

- Conservation equations for mass, momentum, and energy
- Cloud-hot gas interactions via drag and mixing
- Radiative cooling from Wiersma+09 tables
- Power-law cloud mass distributions
- Observable predictions for UV/optical lines

## Getting Help

- 📖 **Documentation**: You're here!
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/GalacticWindsMultiphaseAnalytic/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/GalacticWindsMultiphaseAnalytic/discussions)
- 📧 **Contact**: your.email@example.com

## Citation

If you use this code, please cite:

```bibtex
@article{fielding2024,
  title={The Structure of Multiphase Galactic Winds},
  author={Fielding, Drummond B. and Bryan, Greg L.},
  journal={The Astrophysical Journal},
  year={2024}
}
```

---

**Ready to start?** → [Installation Guide](getting_started/installation.md) | **Want to fit data?** → [MCMC Fitting](fitting/overview.md)