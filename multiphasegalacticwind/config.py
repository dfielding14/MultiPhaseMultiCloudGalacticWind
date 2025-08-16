"""
Configuration module for multiphase galactic wind model parameters.

This module defines the default parameters used throughout the physics calculations.
Users can modify these parameters by:
1. Creating a WindConfig instance and passing it to WindModel
2. Using WindConfig.set_defaults() to change defaults globally
3. Passing individual parameters to WindModel
"""

import numpy as np
from typing import Dict, Any, Optional
from .constants import Msun, pc


class WindConfig:
    """
    Configuration class for wind model parameters.
    
    All parameters have sensible defaults based on the Fielding & Bryan model,
    but can be easily customized for different physical scenarios.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize configuration with default values.
        
        Parameters can be overridden by passing them as keyword arguments.
        
        Examples
        --------
        >>> config = WindConfig(f_turb0=0.2, drag_coeff=0.3)
        >>> model = WindModel(config=config, SFR=10.0)
        """
        # Gas properties
        self.mu = kwargs.get('mu', 0.62)  # Mean molecular weight
        # Metallicity parameters (relative to solar)
        self.Z_hot_over_Z_solar = kwargs.get('Z_hot_over_Z_solar', kwargs.get('metallicity', 10**-0.5))  # Hot gas metallicity
        self.metallicity = self.Z_hot_over_Z_solar  # Keep for backward compatibility
        self.redshift = kwargs.get('redshift', 0.0)  # Redshift for cooling function
        
        # Wind geometry
        self.half_opening_angle = kwargs.get('half_opening_angle', np.pi/2)
        self.Omwind = 4*np.pi*(1.0 - np.cos(self.half_opening_angle))
        
        # Cloud destruction threshold
        self.M_cloud_min = kwargs.get('M_cloud_min', 1e-2*Msun)
        
        # TRML (Turbulent Radiative Mixing Layer) parameters
        self.CoolingAreaChiPower = kwargs.get('CoolingAreaChiPower', 0.5)
        self.ColdTurbulenceChiPower = kwargs.get('ColdTurbulenceChiPower', -0.5)
        self.TurbulentVelocityChiPower = kwargs.get('TurbulentVelocityChiPower', 0.0)
        self.geometric_factor = kwargs.get('geometric_factor', 1.0)
        self.Mdot_coefficient = kwargs.get('Mdot_coefficient', 1.0/3.0)
        self.Cooling_Factor = kwargs.get('Cooling_Factor', 1.0)
        self.drag_coeff = kwargs.get('drag_coeff', 0.5)
        self.f_turb0 = kwargs.get('f_turb0', 0.1)
        
        # Cold cloud injection parameters
        self.cold_cloud_injection_radial_power = kwargs.get('cold_cloud_injection_radial_power', 6)
        self.cold_cloud_injection_radial_extent = kwargs.get('cold_cloud_injection_radial_extent', 1.33 * 300 * pc)
        self.v_cloud_init = kwargs.get('v_cloud_init', 100.0)  # km/s, initial cloud velocity
        self.v_cloud_min = kwargs.get('v_cloud_min', 1.0)  # km/s, minimum cloud velocity before termination
        self.cloud_radial_offset = kwargs.get('cloud_radial_offset', 0.0)  # fractional offset from sonic radius
        self.Z_cloud_over_Z_solar = kwargs.get('Z_cloud_over_Z_solar', 0.3)  # Cloud metallicity relative to solar
        self.T_cl = kwargs.get('T_cl', 1e4)  # K, cloud temperature
        
        # Supernova feedback parameters
        self.E_SN = kwargs.get('E_SN', 1e51)  # erg, energy per supernova
        self.mstar = kwargs.get('mstar', 100.0)  # Msun, stellar mass per supernova
        
        # Event detection parameters
        self.sonic_point_tolerance = kwargs.get('sonic_point_tolerance', 0.1)  # Tolerance for detecting sonic transitions
        
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            'mu': self.mu,
            'metallicity': self.metallicity,  # Keep for backward compatibility
            'Z_hot_over_Z_solar': self.Z_hot_over_Z_solar,
            'redshift': self.redshift,
            'half_opening_angle': self.half_opening_angle,
            'Omwind': self.Omwind,
            'M_cloud_min': self.M_cloud_min,
            'CoolingAreaChiPower': self.CoolingAreaChiPower,
            'ColdTurbulenceChiPower': self.ColdTurbulenceChiPower,
            'TurbulentVelocityChiPower': self.TurbulentVelocityChiPower,
            'geometric_factor': self.geometric_factor,
            'Mdot_coefficient': self.Mdot_coefficient,
            'Cooling_Factor': self.Cooling_Factor,
            'drag_coeff': self.drag_coeff,
            'f_turb0': self.f_turb0,
            'cold_cloud_injection_radial_power': self.cold_cloud_injection_radial_power,
            'cold_cloud_injection_radial_extent': self.cold_cloud_injection_radial_extent,
            'v_cloud_init': self.v_cloud_init,
            'v_cloud_min': self.v_cloud_min,
            'cloud_radial_offset': self.cloud_radial_offset,
            'Z_cloud_over_Z_solar': self.Z_cloud_over_Z_solar,
            'T_cl': self.T_cl,
            'E_SN': self.E_SN,
            'mstar': self.mstar,
            'sonic_point_tolerance': self.sonic_point_tolerance
        }
    
    @classmethod
    def set_defaults(cls, **kwargs: Any) -> None:
        """
        Set default values that will be used for all new WindConfig instances.
        
        This is useful for changing defaults globally in a script or notebook.
        
        Examples
        --------
        >>> WindConfig.set_defaults(f_turb0=0.2, drag_coeff=0.3)
        >>> # Now all new configs will use these defaults
        >>> config = WindConfig()  # Will have f_turb0=0.2, drag_coeff=0.3
        """
        for key, value in kwargs.items():
            if hasattr(cls, f'_default_{key}'):
                setattr(cls, f'_default_{key}', value)
            else:
                raise ValueError(f"Unknown parameter: {key}")


# Create a default configuration instance
_default_config = WindConfig()


def get_default_config() -> WindConfig:
    """Get a copy of the default configuration."""
    return WindConfig()