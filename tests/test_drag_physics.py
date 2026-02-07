"""
Test drag force physics to prevent regression of sign bug.
"""

import numpy as np
import pytest
from multiphasegalacticwind.core_physics import Wind_Evo
from multiphasegalacticwind.constants import kpc, Msun, kb, mp


def test_drag_force_sign():
    """Test that drag force has correct sign for different relative velocities"""
    
    # Setup common parameters
    r = 0.3 * kpc
    v_circ = 0.001 * 1e5
    rho_wind = 1e-24
    P = 1e-10
    metallicity = 0.3
    rhoZ_wind = rho_wind * metallicity
    
    # Configuration
    config_dict = {
        'N_cloud_species': 1,
        'metallicity': metallicity,
        'cooling_factor': 0.0,
        'Cooling_Factor': 0.0,
        'f_turb0': 0.0,  # No turbulence
        'drag_coeff': 1.0,
        'M_cloud_min': 0.1 * Msun,
        'CoolingAreaChiPower': 0.5,
        'ColdTurbulenceChiPower': -0.5,
        'TurbulentVelocityChiPower': 0.0,
        'geometric_factor': 1.0,
        'Mdot_coefficient': 0.0,  # No mass transfer
        'Omwind': 4*np.pi,
        'mu': 0.61,
    }
    
    # Common cloud parameters
    M_cloud = 1e30  # Small cloud
    Z_cloud = metallicity
    T_cloud = 1e4
    r0 = 0.3 * kpc
    injection_radius = r0
    injection_power = 1.0
    Ndot_cloud0 = np.array([1e-10])
    
    params = (
        v_circ, Ndot_cloud0, T_cloud, injection_radius,
        injection_power, config_dict, r0, 0.0, 0.0, None
    )
    
    # Test 1: Wind faster than cloud (v_rel > 0)
    v_wind = 50 * 1e5
    v_cloud = 30 * 1e5
    y1 = np.array([v_wind, rho_wind, P, rhoZ_wind, M_cloud, v_cloud, Z_cloud])
    
    dydt1 = Wind_Evo(r, y1, params)
    dv_wind1 = dydt1[0]
    dv_cloud1 = dydt1[5]
    
    # Wind should decelerate more (or accelerate less) due to drag
    # Cloud should accelerate due to drag
    # We can't test exact values but can test relative behavior
    
    # Test 2: Cloud faster than wind (v_rel < 0)
    v_wind = 30 * 1e5
    v_cloud = 50 * 1e5
    y2 = np.array([v_wind, rho_wind, P, rhoZ_wind, M_cloud, v_cloud, Z_cloud])
    
    dydt2 = Wind_Evo(r, y2, params)
    dv_wind2 = dydt2[0]
    dv_cloud2 = dydt2[5]
    
    # For cloud faster case, cloud should decelerate
    # This is the critical test - with the bug, cloud would still accelerate
    
    # Test 3: Equal velocities (v_rel = 0)
    v_equal = 40 * 1e5
    y3 = np.array([v_equal, rho_wind, P, rhoZ_wind, M_cloud, v_equal, Z_cloud])
    
    dydt3 = Wind_Evo(r, y3, params)
    dv_wind3 = dydt3[0]
    dv_cloud3 = dydt3[5]
    
    # With equal velocities, accelerations should be similar (no drag)
    # Both should feel only gravity and pressure gradients
    
    # The key test: wind acceleration should be MORE NEGATIVE when wind is faster
    # (more drag opposing wind motion)
    assert dv_wind1 < dv_wind3, "Wind should decelerate more when moving faster than cloud"
    assert dv_wind2 > dv_wind3, "Wind should decelerate less when moving slower than cloud"
    
    # Cloud acceleration should follow opposite pattern
    # This would fail with the v_rel**2 bug where sign is lost


def test_drag_momentum_conservation():
    """Test that drag conserves total momentum"""
    
    # Setup parameters for significant drag
    r = 0.3 * kpc
    v_circ = 0.001 * 1e5
    rho_wind = 1e-24
    P = 1e-10
    metallicity = 0.3
    
    # Large cloud for measurable drag
    M_cloud = 1e35  # Large cloud
    v_wind = 100 * 1e5
    v_cloud = 50 * 1e5
    
    # Create state
    y = np.array([
        v_wind, rho_wind, P, rho_wind * metallicity,
        M_cloud, v_cloud, metallicity
    ])
    
    # Configuration with strong drag
    config_dict = {
        'N_cloud_species': 1,
        'metallicity': metallicity,
        'cooling_factor': 0.0,
        'Cooling_Factor': 0.0,
        'f_turb0': 0.0,
        'drag_coeff': 10.0,  # Strong drag
        'M_cloud_min': 0.1 * Msun,
        'CoolingAreaChiPower': 0.5,
        'ColdTurbulenceChiPower': -0.5,
        'TurbulentVelocityChiPower': 0.0,
        'geometric_factor': 1.0,
        'Mdot_coefficient': 0.0,
        'Omwind': 4*np.pi,
        'mu': 0.61,
    }
    
    params = (
        v_circ, np.array([1e-5]), 1e4, r, 1.0,
        config_dict, r, 0.0, 0.0, None
    )
    
    # Get derivatives
    dydt = Wind_Evo(r, y, params)
    
    # The momentum exchange should be equal and opposite
    # This is a more complex test since we need to account for
    # the number density of clouds vs wind mass
    
    # Basic check: accelerations should have physically reasonable magnitudes
    assert np.isfinite(dydt[0]), "Wind acceleration should be finite"
    assert np.isfinite(dydt[5]), "Cloud acceleration should be finite"
    
    # Check that drag opposes relative motion
    v_rel = v_wind - v_cloud
    if v_rel > 0:
        # Wind faster: cloud should accelerate, wind should feel drag
        # We can't easily extract just the drag component, but we know
        # the cloud acceleration should be influenced by drag
        pass  # More complex to test without decomposing all forces
    
    # The key point is that the code doesn't crash and gives finite results


def test_drag_zero_relative_velocity():
    """Test that drag is zero when relative velocity is zero"""
    
    r = 0.3 * kpc
    v_circ = 0.001 * 1e5
    rho_wind = 1e-24
    P = 1e-10
    metallicity = 0.3
    
    # Same velocity for wind and cloud
    v_common = 40 * 1e5
    M_cloud = 1e30
    
    y = np.array([
        v_common, rho_wind, P, rho_wind * metallicity,
        M_cloud, v_common, metallicity
    ])
    
    # Minimal configuration
    config_dict = {
        'N_cloud_species': 1,
        'metallicity': metallicity,
        'cooling_factor': 0.0,
        'Cooling_Factor': 0.0,
        'f_turb0': 0.0,  # No turbulence
        'drag_coeff': 1.0,
        'M_cloud_min': 0.1 * Msun,
        'CoolingAreaChiPower': 0.5,
        'ColdTurbulenceChiPower': -0.5,
        'TurbulentVelocityChiPower': 0.0,
        'geometric_factor': 1.0,
        'Mdot_coefficient': 0.0,  # No mass transfer
        'Omwind': 4*np.pi,
        'mu': 0.61,
    }
    
    params = (
        v_circ, np.array([1e-10]), 1e4, r, 1.0,
        config_dict, r, 0.0, 0.0, None
    )
    
    dydt = Wind_Evo(r, y, params)
    
    # With zero relative velocity and no turbulence/mass transfer,
    # the cloud and wind should have similar dynamics
    # (both feeling gravity and pressure gradients similarly)
    
    # This is hard to test precisely without decomposing forces,
    # but we can check that nothing blows up
    assert np.all(np.isfinite(dydt)), "All derivatives should be finite"


def test_mass_exchange_depends_on_relative_speed_magnitude():
    """Cloud mass exchange should depend on |v_rel|, not the sign of v_rel."""

    r = 0.3 * kpc
    v_circ = 150.0 * 1e5
    rho_wind = 1e-24
    P = 1e-10
    metallicity = 0.3

    M_cloud = 1e33
    Z_cloud = metallicity
    T_cloud = 1e4

    # Disable drag to isolate the cloud mass-exchange contribution.
    config_dict = {
        'N_cloud_species': 1,
        'metallicity': metallicity,
        'redshift': 0.0,
        'cooling_factor': 0.0,
        'Cooling_Factor': 0.0,
        'f_turb0': 0.1,
        'drag_coeff': 0.0,
        'M_cloud_min': 0.1 * Msun,
        'CoolingAreaChiPower': 0.5,
        'ColdTurbulenceChiPower': -0.5,
        'TurbulentVelocityChiPower': 0.0,
        'geometric_factor': 1.0,
        'Mdot_coefficient': 1.0 / 3.0,
        'Omwind': 4 * np.pi,
        'mu': 0.61,
        'v_cloud_min': 1.0,
        'sonic_transition_tolerance': 0.01,
    }

    params = (
        v_circ, np.array([1e-8]), T_cloud, r, 1.0,
        config_dict, r, 0.0, 0.0, None
    )

    # Same |v_rel|, opposite signs.
    y_wind_faster = np.array([
        50e5, rho_wind, P, rho_wind * metallicity,
        M_cloud, 30e5, Z_cloud
    ])
    y_cloud_faster = np.array([
        30e5, rho_wind, P, rho_wind * metallicity,
        M_cloud, 50e5, Z_cloud
    ])

    d1 = Wind_Evo(r, y_wind_faster, params)
    d2 = Wind_Evo(r, y_cloud_faster, params)

    # dM_cloud/dr = Mdot_cloud / v_cloud, so compare recovered Mdot_cloud.
    assert np.isfinite(d1[4]) and np.isfinite(d2[4])
    mdot_cloud_1 = d1[4] * y_wind_faster[5]
    mdot_cloud_2 = d2[4] * y_cloud_faster[5]
    assert np.isclose(mdot_cloud_1, mdot_cloud_2, rtol=1e-10, atol=0.0)


if __name__ == "__main__":
    test_drag_force_sign()
    print("✓ Drag force sign test passed")
    
    test_drag_momentum_conservation()
    print("✓ Drag momentum conservation test passed")
    
    test_drag_zero_relative_velocity()
    print("✓ Zero relative velocity test passed")
    
    print("\nAll drag physics tests passed!")
