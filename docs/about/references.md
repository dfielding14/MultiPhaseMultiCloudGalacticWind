# References

## Primary References

### Fielding & Bryan (2024)
**The Structure of Multiphase Galactic Winds**  
*The Astrophysical Journal*  
- Comprehensive framework for multiphase wind modeling
- TRML physics implementation
- Cloud-hot gas interaction formalism

## Cooling Physics

### Wiersma, Schaye, & Smith (2009)
**The effect of photoionization on the cooling rates of enriched, astrophysical plasmas**  
*Monthly Notices of the Royal Astronomical Society, 393, 99*  
- Cooling tables used in this code
- Metallicity-dependent cooling functions
- Temperature range: 10⁴ - 10⁹ K

## Observational Context

### CLASSY Survey
**The COS Legacy Archive Spectroscopy Survey**  
Berg et al. (2022), *ApJS, 261, 31*  
- UV spectroscopy of star-forming galaxies
- Ion column densities and kinematics
- Primary data source for model fitting

### Xu et al. (2022)
**CLASSY III: The Properties of Starburst-Driven Warm Ionized Outflows**  
*The Astrophysical Journal, 933, 222*  
- Outflow velocities and mass loading factors
- Multi-ion observations
- Comparison data for models

## Theoretical Background

### Chevalier & Clegg (1985)
**Wind from a starburst galaxy nucleus**  
*Nature, 317, 44*  
- Foundational hot wind model
- Adiabatic wind solutions
- Energy-driven outflow theory

### Thompson et al. (2016)
**Radiative Stellar Feedback in Galaxy Formation: Methods and Physics**  
*MNRAS, 455, 334*  
- Cloud crushing and survival
- Radiative cooling in multiphase media
- Turbulent mixing layers

### Schneider & Robertson (2017)
**Hydrodynamical coupling of mass and momentum in multiphase galactic winds**  
*The Astrophysical Journal, 834, 144*  
- Mass and momentum transfer
- Cloud acceleration mechanisms
- Multiphase wind dynamics

## Numerical Methods

### Virtanen et al. (2020)
**SciPy 1.0: fundamental algorithms for scientific computing in Python**  
*Nature Methods, 17, 261*  
- ODE integration (solve_ivp)
- Interpolation methods
- Scientific computing tools

### Foreman-Mackey et al. (2013)
**emcee: The MCMC Hammer**  
*Publications of the Astronomical Society of the Pacific, 125, 306*  
- Affine-invariant ensemble sampler
- Parallel tempering
- Bayesian inference implementation

## Cloud Physics

### Klein, McKee, & Colella (1994)
**On the hydrodynamic interaction of shock waves with interstellar clouds**  
*The Astrophysical Journal, 420, 213*  
- Cloud crushing timescales
- Drag coefficients
- Cloud destruction mechanisms

### Scannapieco & Brüggen (2015)
**The launching of cold clouds by galaxy outflows**  
*The Astrophysical Journal, 805, 158*  
- Cloud entrainment
- Mass loading from cold gas
- Cloud survival criteria

## Turbulent Mixing

### Fielding et al. (2020)
**Multiphase Gas and the Fractal Nature of Radiative Turbulent Mixing Layers**  
*The Astrophysical Journal Letters, 894, L24*  
- Fractal structure of mixing layers
- Turbulent velocity scaling
- Area enhancement factors

### Gronke & Oh (2018)
**The growth and entrainment of cold gas in a hot wind**  
*MNRAS, 480, L111*  
- Cloud growth via cooling
- Entrainment efficiency
- Critical cloud sizes

## Galaxy Properties

### Kennicutt & Evans (2012)
**Star Formation in the Milky Way and Nearby Galaxies**  
*Annual Review of Astronomy and Astrophysics, 50, 531*  
- Star formation rate indicators
- Galaxy scaling relations
- Feedback efficiencies

### Veilleux et al. (2005)
**Galactic Winds**  
*Annual Review of Astronomy and Astrophysics, 43, 769*  
- Observational signatures
- Wind driving mechanisms
- Multi-wavelength observations

## Software Citations

### Harris et al. (2020)
**Array programming with NumPy**  
*Nature, 585, 357*

### Hunter (2007)
**Matplotlib: A 2D graphics environment**  
*Computing in Science & Engineering, 9, 90*

### Astropy Collaboration (2018)
**The Astropy Project**  
*The Astronomical Journal, 156, 123*

## Data Sources

### Cooling Tables
- **Source**: Wiersma+09 tables
- **Format**: HDF5
- **Location**: `data/z_0.020.hdf5`

### CLASSY Observations
- **Source**: HST/COS spectra
- **Archive**: MAST
- **Processed**: Xu+22 measurements

## Acknowledgments

This work builds upon decades of theoretical and observational research in galaxy evolution and feedback processes. We acknowledge all contributors to the open-source scientific Python ecosystem.