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
        # Get custom defaults if they exist
        custom_defaults = getattr(self.__class__, '_custom_defaults', {})
        
        # Helper function to get value with custom defaults
        def get_param(key, default):
            return kwargs.get(key, custom_defaults.get(key, default))
        
        # Gas properties
        self.mu = get_param('mu', 0.62)  # Mean molecular weight
        # Metallicity parameters (relative to solar)
        self.Z_hot_over_Z_solar = kwargs.get('Z_hot_over_Z_solar', 
                                            kwargs.get('metallicity', 
                                                      custom_defaults.get('Z_hot_over_Z_solar', 
                                                                        custom_defaults.get('metallicity', 10**-0.5))))
        self.metallicity = self.Z_hot_over_Z_solar  # Keep for backward compatibility
        self.redshift = get_param('redshift', 0.0)  # Redshift for cooling function

        # Wind geometry
        self.half_opening_angle = get_param('half_opening_angle', np.pi/2)
        # Biconical solid angle saturates at full sphere (4π) for wide cones.
        self.Omwind = min(4*np.pi*(1.0 - np.cos(self.half_opening_angle)), 4*np.pi)

        # Cloud destruction threshold
        self.M_cloud_min = get_param('M_cloud_min', 1e-2*Msun)

        # TRML (Turbulent Radiative Mixing Layer) parameters
        self.CoolingAreaChiPower = get_param('CoolingAreaChiPower', 0.5)
        self.ColdTurbulenceChiPower = get_param('ColdTurbulenceChiPower', -0.5)
        self.TurbulentVelocityChiPower = get_param('TurbulentVelocityChiPower', 0.0)
        self.geometric_factor = get_param('geometric_factor', 1.0)
        self.Mdot_coefficient = get_param('Mdot_coefficient', 1.0/3.0)
        self.Cooling_Factor = get_param('Cooling_Factor', 1.0)
        self.drag_coeff = get_param('drag_coeff', 0.5)
        self.f_turb0 = get_param('f_turb0', 0.1)

        # Cold cloud injection parameters
        self.cold_cloud_injection_radial_power = get_param('cold_cloud_injection_radial_power', 6)
        self.cold_cloud_injection_radial_extent_frac = get_param('cold_cloud_injection_radial_extent_frac', 1.33)  # fraction of r0
        self.v_cloud_init = get_param('v_cloud_init', 100.0)  # km/s, initial cloud velocity
        self.v_cloud_min = get_param('v_cloud_min', 1.0)  # km/s, minimum cloud velocity before termination
        self.cloud_radial_offset = get_param('cloud_radial_offset', 0.01)  # fractional offset from sonic radius
        self.Z_cloud_over_Z_solar = get_param('Z_cloud_over_Z_solar', 0.3)  # Cloud metallicity relative to solar
        self.T_cl = get_param('T_cl', 1e4)  # K, cloud temperature

        # Supernova feedback parameters
        self.E_SN = get_param('E_SN', 1e51)  # erg, energy per supernova
        self.mstar = get_param('mstar', 100.0)  # Msun, stellar mass per supernova

        # Event detection parameters
        self.sonic_point_offset = get_param('sonic_point_offset', 1e-6)  # Small offset from Mach=1 for initial conditions
        self.sonic_transition_tolerance = get_param('sonic_transition_tolerance', 0.01)  # Tolerance for detecting sonic transitions

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
            'cold_cloud_injection_radial_extent_frac': self.cold_cloud_injection_radial_extent_frac,
            'v_cloud_init': self.v_cloud_init,
            'v_cloud_min': self.v_cloud_min,
            'cloud_radial_offset': self.cloud_radial_offset,
            'Z_cloud_over_Z_solar': self.Z_cloud_over_Z_solar,
            'T_cl': self.T_cl,
            'E_SN': self.E_SN,
            'mstar': self.mstar,
            'sonic_point_offset': self.sonic_point_offset,
            'sonic_transition_tolerance': self.sonic_transition_tolerance
        }

    def validate(self) -> None:
        """
        Validate configuration parameters are physically reasonable.
        
        Raises
        ------
        ValueError
            If any parameter is outside valid range
        """
        # Gas properties
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.Z_hot_over_Z_solar < 0:
            raise ValueError(f"Z_hot_over_Z_solar must be non-negative, got {self.Z_hot_over_Z_solar}")
        if self.redshift < 0:
            raise ValueError(f"redshift must be non-negative, got {self.redshift}")
            
        # Wind geometry
        if not 0 < self.half_opening_angle <= np.pi:
            raise ValueError(f"half_opening_angle must be in (0, π], got {self.half_opening_angle}")
            
        # Cloud properties
        if self.M_cloud_min <= 0:
            raise ValueError(f"M_cloud_min must be positive, got {self.M_cloud_min}")
        if self.T_cl <= 0:
            raise ValueError(f"T_cl must be positive, got {self.T_cl}")
            
        # TRML parameters
        if not 0 < self.f_turb0 < 1:
            raise ValueError(f"f_turb0 must be in (0, 1), got {self.f_turb0}")
        if self.drag_coeff <= 0:
            raise ValueError(f"drag_coeff must be positive, got {self.drag_coeff}")
        if self.Mdot_coefficient <= 0:
            raise ValueError(f"Mdot_coefficient must be positive, got {self.Mdot_coefficient}")
            
        # Cloud injection
        if self.v_cloud_init < 0:
            raise ValueError(f"v_cloud_init must be non-negative, got {self.v_cloud_init}")
        if self.v_cloud_min < 0:
            raise ValueError(f"v_cloud_min must be non-negative, got {self.v_cloud_min}")
        if self.cloud_radial_offset < 0:
            raise ValueError(f"cloud_radial_offset must be non-negative, got {self.cloud_radial_offset}")
        if self.cold_cloud_injection_radial_extent_frac <= 0:
            raise ValueError(f"cold_cloud_injection_radial_extent_frac must be positive, got {self.cold_cloud_injection_radial_extent_frac}")
            
        # Supernova parameters
        if self.E_SN <= 0:
            raise ValueError(f"E_SN must be positive, got {self.E_SN}")
        if self.mstar <= 0:
            raise ValueError(f"mstar must be positive, got {self.mstar}")
            
        # Numerical parameters
        if self.sonic_point_offset <= 0:
            raise ValueError(f"sonic_point_offset must be positive, got {self.sonic_point_offset}")
        if self.sonic_transition_tolerance <= 0:
            raise ValueError(f"sonic_transition_tolerance must be positive, got {self.sonic_transition_tolerance}")

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
        # Store custom defaults as class attribute
        if not hasattr(cls, '_custom_defaults'):
            cls._custom_defaults = {}
        
        # Validate that keys are valid parameter names
        valid_params = {
            'mu', 'Z_hot_over_Z_solar', 'metallicity', 'redshift',
            'half_opening_angle', 'M_cloud_min', 'CoolingAreaChiPower',
            'ColdTurbulenceChiPower', 'TurbulentVelocityChiPower',
            'geometric_factor', 'Mdot_coefficient', 'Cooling_Factor',
            'drag_coeff', 'f_turb0', 'cold_cloud_injection_radial_power',
            'cold_cloud_injection_radial_extent_frac', 'v_cloud_init',
            'v_cloud_min', 'cloud_radial_offset', 'Z_cloud_over_Z_solar',
            'T_cl', 'E_SN', 'mstar', 'sonic_point_offset',
            'sonic_transition_tolerance'
        }
        
        for key, value in kwargs.items():
            if key not in valid_params:
                raise ValueError(f"Unknown parameter: {key}")
            cls._custom_defaults[key] = value


# Create a default configuration instance
_default_config = WindConfig()


def get_default_config() -> WindConfig:
    """Get a copy of the default configuration."""
    return WindConfig()
