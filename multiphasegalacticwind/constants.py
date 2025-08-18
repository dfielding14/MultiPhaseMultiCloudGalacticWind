"""
Physical constants for the multiphase galactic wind model.

This module contains only true physical constants and unit conversions.
Model parameters should be set via WindConfig in config.py.
"""

# Physical constants (CGS units)
gamma = 5/3.                    # Adiabatic index
kb = 1.3806488e-16             # Boltzmann constant [erg/K]
mp = 1.67373522381e-24         # Proton mass [g]
G = 6.67430e-8                 # Gravitational constant [cm^3/g/s^2] (CODATA 2018)

# Unit conversions
km = 1e5                       # km to cm
s = 1                          # second
yr = 3.1536e7                  # year [s]
Myr = 3.1536e13               # Megayear [s]
Gyr = 3.1536e16               # Gigayear [s]
pc = 3.086e18                  # parsec [cm]
kpc = 1.0e3 * pc              # kiloparsec [cm]
Mpc = 1.0e6 * pc              # Megaparsec [cm]
Msun = 1.98892e33              # Solar mass [g] (IAU 2015)
keV = 1.60218e-9              # keV to erg

# Cosmology (Planck 2015)
H0 = 67.74*km/s/Mpc           # Hubble constant
Om = 0.3075                    # Matter density parameter
OL = 1 - Om                    # Dark energy density parameter
fb = 0.158                     # Baryon fraction

# Standard abundances
Z_solar = 0.02                 # Solar metallicity mass fraction
muH = 1/0.75                   # Mean molecular weight per hydrogen