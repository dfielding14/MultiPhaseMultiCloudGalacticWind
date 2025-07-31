"""
MultiPhase MultiCloud Galactic Wind Model

A Python package for simulating multiphase galactic winds with embedded clouds.
Based on Fielding & Bryan (2024).
"""

from .wind_model import WindModel
from .config import WindConfig, get_default_config
from .constants import *
from .plotting import (setup_plotting_style, plot_wind_solution, 
                      plot_velocity_distribution, plot_column_density_distribution)
from .observables import (calculate_velocity_distribution, 
                         calculate_velocity_moments,
                         calculate_mass_weighted_velocity,
                         calculate_column_density_distribution,
                         calculate_column_density_by_species)

__version__ = "0.1.0"
__all__ = ["WindModel", "WindConfig", "get_default_config", "setup_plotting_style", "plot_wind_solution",
           "plot_velocity_distribution", "plot_column_density_distribution",
           "calculate_velocity_distribution", "calculate_velocity_moments", 
           "calculate_mass_weighted_velocity", "calculate_column_density_distribution",
           "calculate_column_density_by_species"]