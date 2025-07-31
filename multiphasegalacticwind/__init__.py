"""
MultiPhase MultiCloud Galactic Wind Model

A Python package for simulating multiphase galactic winds with embedded clouds.
Based on Fielding & Bryan (2024).
"""

from .wind_model import WindModel
from .plotting import setup_plotting_style, plot_wind_solution

__version__ = "0.1.0"
__all__ = ["WindModel", "setup_plotting_style", "plot_wind_solution"]