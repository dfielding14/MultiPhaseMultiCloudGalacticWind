"""
MultiPhase MultiCloud Galactic Wind Model

A Python package for simulating multiphase galactic winds with embedded clouds.
Based on Fielding et al. (2022, ApJ 924, 82).
"""

from .wind_model import WindModel
from .config import WindConfig, get_default_config
from .constants import *
from .plotting import (setup_plotting_style, plot_wind_solution, plot_profiles,
                      plot_column_density_distribution)
from .observables import (calculate_velocity_moments,
                         calculate_mass_weighted_velocity,
                         calculate_column_density_distribution,
                         calculate_column_density_by_species)
from .analysis_helpers import (Field_Length, Field_Length_mix, cloud_radius, cloud_ksi,
                             Cooling_and_Acceleration, Gradient_Components,
                             calculate_cloud_moments, get_cloud_mass_spectrum)
from .inference import (
    MomentInferenceModel,
    MAPFitResult,
    HMCResult,
    PosteriorFitResult,
    build_covariance,
    covariance_to_correlation,
    summarize_parameter_degeneracies,
    plot_corner,
    plot_moment_fit,
)

__version__ = "0.1.0"
__all__ = ["WindModel", "WindConfig", "get_default_config", "setup_plotting_style", "plot_wind_solution",
           "plot_profiles", "plot_column_density_distribution",
           "calculate_velocity_moments", 
           "calculate_mass_weighted_velocity", "calculate_column_density_distribution",
           "calculate_column_density_by_species",
           "MomentInferenceModel", "MAPFitResult", "HMCResult", "PosteriorFitResult",
           "build_covariance", "covariance_to_correlation", "summarize_parameter_degeneracies",
           "plot_corner", "plot_moment_fit"]
