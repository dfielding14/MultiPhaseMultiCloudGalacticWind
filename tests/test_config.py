#!/usr/bin/env python3
"""
Unit tests for the WindConfig class.

Tests configuration initialization, default values, and parameter handling.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiphasegalacticwind.config import WindConfig, get_default_config
from multiphasegalacticwind.constants import pc, Msun


def test_default_initialization():
    """Test that WindConfig initializes with correct defaults."""
    print("Testing default initialization...", end="")
    try:
        config = WindConfig()
        
        # Check key default values
        assert config.mu == 0.62, f"Wrong mu: {config.mu}"
        assert config.Z_hot_over_Z_solar == 10**-0.5, f"Wrong Z_hot: {config.Z_hot_over_Z_solar}"
        assert config.redshift == 0.0, f"Wrong redshift: {config.redshift}"
        assert config.half_opening_angle == np.pi/2, f"Wrong opening angle: {config.half_opening_angle}"
        assert config.v_cloud_init == 100.0, f"Wrong v_cloud_init: {config.v_cloud_init}"
        assert config.v_cloud_min == 1.0, f"Wrong v_cloud_min: {config.v_cloud_min}"
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_custom_initialization():
    """Test WindConfig with custom parameters."""
    print("Testing custom initialization...", end="")
    try:
        config = WindConfig(
            mu=0.59,
            Z_hot_over_Z_solar=0.1,
            redshift=2.0,
            v_cloud_init=200.0,
            v_cloud_min=10.0,
            f_turb0=0.2
        )
        
        assert config.mu == 0.59, f"Wrong mu: {config.mu}"
        assert config.Z_hot_over_Z_solar == 0.1, f"Wrong Z_hot: {config.Z_hot_over_Z_solar}"
        assert config.redshift == 2.0, f"Wrong redshift: {config.redshift}"
        assert config.v_cloud_init == 200.0, f"Wrong v_cloud_init: {config.v_cloud_init}"
        assert config.v_cloud_min == 10.0, f"Wrong v_cloud_min: {config.v_cloud_min}"
        assert config.f_turb0 == 0.2, f"Wrong f_turb0: {config.f_turb0}"
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_backward_compatibility():
    """Test that old parameter name 'metallicity' still works."""
    print("Testing backward compatibility...", end="")
    try:
        # Test with old 'metallicity' parameter
        config = WindConfig(metallicity=0.5)
        assert config.Z_hot_over_Z_solar == 0.5, f"Wrong Z_hot: {config.Z_hot_over_Z_solar}"
        assert config.metallicity == 0.5, f"Wrong metallicity: {config.metallicity}"
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_to_dict():
    """Test that to_dict returns all parameters."""
    print("Testing to_dict method...", end="")
    try:
        config = WindConfig()
        config_dict = config.to_dict()
        
        # Check that key parameters are in the dict
        assert 'mu' in config_dict, "mu not in dict"
        assert 'Z_hot_over_Z_solar' in config_dict, "Z_hot_over_Z_solar not in dict"
        assert 'metallicity' in config_dict, "metallicity not in dict (backward compat)"
        assert 'v_cloud_init' in config_dict, "v_cloud_init not in dict"
        assert 'v_cloud_min' in config_dict, "v_cloud_min not in dict"
        assert 'Z_cloud_over_Z_solar' in config_dict, "Z_cloud_over_Z_solar not in dict"
        
        # Check values match
        assert config_dict['mu'] == config.mu
        assert config_dict['Z_hot_over_Z_solar'] == config.Z_hot_over_Z_solar
        assert config_dict['v_cloud_min'] == config.v_cloud_min
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_get_default_config():
    """Test the get_default_config function."""
    print("Testing get_default_config...", end="")
    try:
        config1 = get_default_config()
        config2 = get_default_config()
        
        # Should return new instances
        assert config1 is not config2, "Same instance returned"
        
        # But with same default values
        assert config1.mu == config2.mu
        assert config1.Z_hot_over_Z_solar == config2.Z_hot_over_Z_solar
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_computed_properties():
    """Test computed properties like Omwind."""
    print("Testing computed properties...", end="")
    try:
        # Default: half_opening_angle = pi/2 gives hemisphere
        # Omwind = 4*pi*(1 - cos(pi/2)) = 4*pi*(1 - 0) = 4*pi = 2*(2*pi) = 2 hemispheres
        config1 = WindConfig(half_opening_angle=np.pi/2)
        expected1 = 4*np.pi*(1.0 - np.cos(np.pi/2))  # Should be 4*pi
        assert np.isclose(config1.Omwind, expected1), f"Wrong Omwind for hemisphere: {config1.Omwind}, expected {expected1}"
        
        # Narrow cone: half_opening_angle = pi/4 
        config2 = WindConfig(half_opening_angle=np.pi/4)
        expected2 = 4*np.pi*(1.0 - np.cos(np.pi/4))
        assert np.isclose(config2.Omwind, expected2), f"Wrong Omwind for narrow cone: {config2.Omwind}, expected {expected2}"
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_physical_constants():
    """Test that physical constants have reasonable values."""
    print("Testing physical constants...", end="")
    try:
        config = WindConfig()
        
        # Check E_SN
        assert config.E_SN == 1e51, f"Wrong E_SN: {config.E_SN}"
        
        # Check mstar
        assert config.mstar == 100.0, f"Wrong mstar: {config.mstar}"
        
        # Check M_cloud_min
        assert config.M_cloud_min == 1e-2 * Msun, f"Wrong M_cloud_min: {config.M_cloud_min}"
        
        # Check injection extent
        expected_extent = 1.33 * 300 * pc
        assert np.isclose(config.cold_cloud_injection_radial_extent, expected_extent), \
            f"Wrong injection extent: {config.cold_cloud_injection_radial_extent}"
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_parameter_ranges():
    """Test parameters with extreme values."""
    print("Testing parameter ranges...", end="")
    try:
        # Very low metallicity
        config1 = WindConfig(Z_hot_over_Z_solar=1e-3)
        assert config1.Z_hot_over_Z_solar == 1e-3
        
        # High redshift
        config2 = WindConfig(redshift=10.0)
        assert config2.redshift == 10.0
        
        # Large opening angle
        config3 = WindConfig(half_opening_angle=np.pi*0.99)
        assert config3.half_opening_angle == np.pi*0.99
        
        # Small epsilon
        config4 = WindConfig(epsilon=1e-10)
        assert config4.epsilon == 1e-10
        
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running WindConfig tests")
    print("=" * 60)
    
    tests = [
        test_default_initialization,
        test_custom_initialization,
        test_backward_compatibility,
        test_to_dict,
        test_get_default_config,
        test_computed_properties,
        test_physical_constants,
        test_parameter_ranges,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f" FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f" ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests completed: {len(tests) - failed}/{len(tests)} passed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)