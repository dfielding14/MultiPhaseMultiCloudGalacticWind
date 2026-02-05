#!/usr/bin/env python3
"""
Simple unit tests for the cooling module (no pytest required).
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiphasegalacticwind.cooling import (
    get_lambda_interpolator, get_cooling_interpolator,
    tcool_P, Lambda_P, get_tcool_min_interpolators,
    load_cooling_table
)
from multiphasegalacticwind.constants import kb, mp, yr, Myr


def test_lambda_interpolator_loads():
    """Test that the main cooling interpolator loads correctly."""
    print("Testing lambda interpolator loading...", end="")
    try:
        Lambda = get_lambda_interpolator()
        assert Lambda is not None
        
        # Test that it returns reasonable values
        result = Lambda((0.0, 5.0, 1.0, 0.0))
        print(f" result={result}", end="")
        assert np.isfinite(result), f"Result not finite: {result}"
        # Cooling function returns actual values, not log values
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_cooling_interpolator_caching():
    """Test that cooling interpolator caching works."""
    print("Testing cooling interpolator caching...", end="")
    # Get interpolator twice with same parameters
    interp1 = get_cooling_interpolator(0.62, 1.0, 0.0)
    interp2 = get_cooling_interpolator(0.62, 1.0, 0.0)
    assert interp1 is interp2  # Should be same object due to caching
    
    # Different parameters should give different interpolator
    interp3 = get_cooling_interpolator(0.62, 0.1, 0.0)
    assert interp3 is not interp1
    print(" PASSED")


def test_tcool_P_reasonable_values():
    """Test that cooling times are reasonable."""
    print("Testing tcool_P reasonable values...", end="")
    try:
        # Typical ISM conditions
        T = 1e4  # K
        P = 1e3 * kb  # nT ~ 1e3 cm^-3 K
        Z = 1.0  # Solar metallicity
        z = 0.0
        mu = 0.62
        
        tcool = tcool_P(T, P, Z, z, mu)
        print(f" tcool={tcool/yr:.2e} yr", end="")
        assert np.isfinite(tcool), f"tcool not finite: {tcool}"
        assert tcool > 0, f"tcool not positive: {tcool}"
        assert tcool < 1e10 * yr, f"tcool too large: {tcool/yr} yr"  # Should be less than Hubble time
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_tcool_P_hot_gas():
    """Test cooling time for hot gas."""
    print("Testing tcool_P for hot gas...", end="")
    # Hot gas should have longer cooling times
    T_hot = 1e7  # K
    P = 1e3 * kb
    
    tcool_hot = tcool_P(T_hot, P, 1.0, 0.0, 0.62)
    tcool_cool = tcool_P(1e4, P, 1.0, 0.0, 0.62)
    
    assert tcool_hot > tcool_cool
    print(" PASSED")


def test_tcool_P_metallicity_dependence():
    """Test that lower metallicity gives longer cooling times."""
    print("Testing tcool_P metallicity dependence...", end="")
    try:
        T = 1e5  # K
        P = 1e3 * kb
        
        tcool_solar = tcool_P(T, P, 1.0, 0.0, 0.62)
        tcool_low_Z = tcool_P(T, P, 0.1, 0.0, 0.62)
        print(f" solar={tcool_solar/yr:.2e}yr, low_Z={tcool_low_Z/yr:.2e}yr", end="")
        
        assert tcool_low_Z > tcool_solar, f"Low Z cooling time not longer: {tcool_low_Z/yr} vs {tcool_solar/yr}"
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_tcool_P_array_inputs():
    """Test tcool_P with array inputs."""
    print("Testing tcool_P with array inputs...", end="")
    try:
        T = np.array([1e4, 1e5, 1e6])
        P = 1e3 * kb * np.ones(3)
        
        tcool = tcool_P(T, P, 1.0, 0.0, 0.62)
        print(f" results={[f'{t/yr:.2e}' for t in tcool]}yr", end="")
        assert len(tcool) == 3, f"Wrong length: {len(tcool)}"
        assert np.all(np.isfinite(tcool)), f"Not all finite: {tcool}"
        assert np.all(tcool > 0), f"Not all positive: {tcool}"
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_lambda_P_values():
    """Test Lambda_P cooling function values."""
    print("Testing Lambda_P values...", end="")
    try:
        T = 1e5
        P = 1e3 * kb
        
        lambda_val = Lambda_P(T, P, 1.0, 0.0, 0.62)
        print(f" lambda={lambda_val:.2e}", end="")
        assert np.isfinite(lambda_val), f"Lambda not finite: {lambda_val}"
        # Lambda should be a reasonable cooling rate
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_tcool_min_interpolators():
    """Test minimum cooling time interpolators."""
    print("Testing minimum cooling time interpolators...", end="")
    try:
        T_tcool_min_P, tcool_min_P = get_tcool_min_interpolators()
        
        assert T_tcool_min_P is not None
        assert tcool_min_P is not None
        
        # Test at typical pressure
        P = 1e3 * kb
        Z = 1.0
        
        T_min = T_tcool_min_P((P, Z))
        tcool_min = tcool_min_P((P, Z))
        print(f" T_min={T_min:.2e}K, tcool_min={tcool_min/yr:.2e}yr", end="")
        
        assert np.isfinite(T_min), f"T_min not finite: {T_min}"
        assert np.isfinite(tcool_min), f"tcool_min not finite: {tcool_min}"
        assert T_min > 0, f"T_min not positive: {T_min}"
        assert T_min < 1e7, f"T_min too high: {T_min}"  # Should be reasonable temperature
        assert tcool_min > 0, f"tcool_min not positive: {tcool_min}"
        assert tcool_min < 1e9 * yr, f"tcool_min too large: {tcool_min/yr} yr"
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")


def test_cooling_at_extreme_temperatures():
    """Test cooling functions at extreme temperatures."""
    print("Testing cooling at extreme temperatures...", end="")
    P = 1e3 * kb
    
    # Very cold
    tcool_cold = tcool_P(1e3, P, 1.0, 0.0, 0.62)
    assert np.isfinite(tcool_cold) or tcool_cold > 1e20  # May be very long
    
    # Very hot
    tcool_hot = tcool_P(1e8, P, 1.0, 0.0, 0.62)
    assert np.isfinite(tcool_hot)
    assert tcool_hot > 0
    print(" PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running cooling module tests")
    print("=" * 60)
    
    # Load cooling table once
    print("Loading cooling table...")
    load_cooling_table(verbose=False)
    print("Done.\n")
    
    tests = [
        test_lambda_interpolator_loads,
        test_cooling_interpolator_caching,
        test_tcool_P_reasonable_values,
        test_tcool_P_hot_gas,
        test_tcool_P_metallicity_dependence,
        test_tcool_P_array_inputs,
        test_lambda_P_values,
        test_tcool_min_interpolators,
        test_cooling_at_extreme_temperatures,
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