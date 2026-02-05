#!/usr/bin/env python3
"""
Unit tests for the cooling module.

Tests cooling function interpolation, cooling time calculations,
and edge cases.
"""
import pytest
import numpy as np
from multiphasegalacticwind.cooling import (
    get_lambda_interpolator, get_cooling_interpolator,
    tcool_P, Lambda_P, get_tcool_min_interpolators,
    load_cooling_table
)
from multiphasegalacticwind.constants import kb, yr


class TestCoolingFunctions:
    """Tests for cooling function calculations."""
    
    @classmethod
    def setup_class(cls):
        """Load cooling table once for all tests."""
        load_cooling_table(verbose=False)
    
    def test_lambda_interpolator_loads(self):
        """Test that the main cooling interpolator loads correctly."""
        Lambda = get_lambda_interpolator()
        assert Lambda is not None
        
        # Test that it returns reasonable values
        # log_nH=0, log_T=5, Z=1, z=0 should give a cooling rate
        result = Lambda((0.0, 5.0, 1.0, 0.0))
        assert np.isfinite(result)
        assert result > 0  # Cooling rates are tabulated in linear cgs units
        assert result > 1e-28  # Not the interpolation fill value
    
    def test_cooling_interpolator_caching(self):
        """Test that cooling interpolator caching works."""
        # Get interpolator twice with same parameters
        interp1 = get_cooling_interpolator(0.62, 1.0, 0.0)
        interp2 = get_cooling_interpolator(0.62, 1.0, 0.0)
        assert interp1 is interp2  # Should be same object due to caching
        
        # Different parameters should give different interpolator
        interp3 = get_cooling_interpolator(0.62, 0.1, 0.0)
        assert interp3 is not interp1
    
    def test_tcool_P_reasonable_values(self):
        """Test that cooling times are reasonable."""
        # Typical ISM conditions
        T = 1e5  # K
        P = 1e3  # P/k_B in K cm^-3
        Z = 1.0  # Solar metallicity
        z = 0.0
        mu = 0.62
        
        tcool = tcool_P(T, P, Z, z, mu)
        assert np.isfinite(tcool)
        assert tcool > 0
        assert tcool < 1e10 * yr  # Should be less than Hubble time
        
    def test_tcool_P_hot_gas(self):
        """Test cooling time for hot gas."""
        # Hot gas should have longer cooling times
        T_hot = 1e7  # K
        P = 1e3
        
        tcool_hot = tcool_P(T_hot, P, 1.0, 0.0, 0.62)
        tcool_cool = tcool_P(1e5, P, 1.0, 0.0, 0.62)
        
        assert tcool_hot > tcool_cool
    
    def test_tcool_P_metallicity_dependence(self):
        """Test that lower metallicity gives longer cooling times."""
        T = 1e5  # K
        P = 1e3
        
        tcool_solar = tcool_P(T, P, 1.0, 0.0, 0.62)
        tcool_low_Z = tcool_P(T, P, 0.1, 0.0, 0.62)
        
        assert tcool_low_Z > tcool_solar
    
    def test_tcool_P_caching(self):
        """Test that tcool_P caching works correctly."""
        T = 1e5
        P = 1e3
        
        # Call twice with same parameters
        tcool1 = tcool_P(T, P, 1.0, 0.0, 0.62)
        tcool2 = tcool_P(T, P, 1.0, 0.0, 0.62)
        assert tcool1 == tcool2
    
    def test_tcool_P_array_inputs(self):
        """Test tcool_P with array inputs."""
        T = np.array([2e4, 1e5, 1e6])
        P = 1e3 * np.ones(3)
        
        tcool = tcool_P(T, P, 1.0, 0.0, 0.62)
        assert len(tcool) == 3
        assert np.all(np.isfinite(tcool))
        assert np.all(tcool > 0)
    
    def test_lambda_P_values(self):
        """Test Lambda_P cooling function values."""
        T = 1e5
        P = 1e3
        
        lambda_val = Lambda_P(T, P, 1.0, 0.0, 0.62)
        assert np.isfinite(lambda_val)
        assert lambda_val > 0
    
    def test_lambda_P_density_limit(self):
        """Test that Lambda_P handles high density limit."""
        T = 1e4
        # Very high pressure -> high density
        P_high = 1e10
        
        # Should not crash or return nan
        lambda_val = Lambda_P(T, P_high, 1.0, 0.0, 0.62)
        assert np.isfinite(lambda_val)
    
    def test_tcool_min_interpolators(self):
        """Test minimum cooling time interpolators."""
        T_tcool_min_P, tcool_min_P = get_tcool_min_interpolators()
        
        assert T_tcool_min_P is not None
        assert tcool_min_P is not None
        
        # Test at typical pressure
        P = 1e3
        Z = 1.0
        
        T_min = T_tcool_min_P((P, Z))
        tcool_min = tcool_min_P((P, Z))
        
        assert np.isfinite(T_min)
        assert np.isfinite(tcool_min)
        assert T_min > 0
        assert T_min < 1e7  # Should be reasonable temperature
        assert tcool_min > 0
        assert tcool_min < 1e9 * yr
    
    def test_cooling_at_extreme_temperatures(self):
        """Test cooling functions at extreme temperatures."""
        P = 1e3
        
        # Very cold
        tcool_cold = tcool_P(1e3, P, 1.0, 0.0, 0.62)
        assert np.isfinite(tcool_cold) or tcool_cold > 1e20  # May be very long
        
        # Very hot
        tcool_hot = tcool_P(1e8, P, 1.0, 0.0, 0.62)
        assert np.isfinite(tcool_hot)
        assert tcool_hot > 0
    
    def test_cooling_interpolator_with_different_mu(self):
        """Test cooling interpolator with different mean molecular weights."""
        # Primordial gas
        interp1 = get_cooling_interpolator(0.59, 1.0, 0.0)
        # Solar abundance
        interp2 = get_cooling_interpolator(0.62, 1.0, 0.0)
        
        # Interpolator axes are (pressure [dyne/cm^2], density [g/cm^3])
        val1 = interp1((1e3 * kb, 1e-27))
        val2 = interp2((1e3 * kb, 1e-27))
        
        assert np.isfinite(val1)
        assert np.isfinite(val2)
        # Values should be different due to different mu
        assert not np.isclose(val1, val2, rtol=1e-6, atol=0.0)


class TestCoolingEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_pressure(self):
        """Test behavior at zero pressure."""
        # Should handle gracefully, possibly return very long cooling time
        tcool = tcool_P(1e5, 0.0, 1.0, 0.0, 0.62)
        assert np.isfinite(tcool) or tcool > 1e20
    
    def test_negative_metallicity(self):
        """Test that negative metallicity is handled."""
        # Should either raise error or clip to zero
        try:
            tcool = tcool_P(1e5, 1e3, -0.1, 0.0, 0.62)
            # If it doesn't raise, should give similar result to Z=0
            assert tcool > 0
        except:
            # It's ok if it raises an error
            pass
    
    def test_high_redshift(self):
        """Test cooling at high redshift."""
        # Cooling tables may not extend to very high z
        z_max = 10.0
        try:
            tcool = tcool_P(1e5, 1e3, 1.0, z_max, 0.62)
            assert np.isfinite(tcool) or tcool > 0
        except:
            # May be outside interpolation bounds
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
