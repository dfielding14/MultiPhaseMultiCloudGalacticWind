"""
Test that Wind_Evo converges to Hot_Wind_Evo as η_M_cold → 0.
This validates the ODE implementation consistency.
"""

import numpy as np
import pytest
from multiphasegalacticwind.core_physics import Wind_Evo, Hot_Wind_Evo
from multiphasegalacticwind.constants import kpc, Msun, kb, mp


def test_ode_convergence():
    """Test that full model converges to hot-only as cold mass → 0"""
    
    # Test parameters
    r = 0.3 * kpc  # Test radius
    v_wind = 30.0 * 1e5  # 30 km/s
    rho_wind = 1e-24  # g/cm³
    P = 1e-10  # dyne/cm²
    metallicity = 0.3
    rhoZ_wind = rho_wind * metallicity
    
    # Hot-only state and parameters
    y_hot = np.array([v_wind, rho_wind, P, rhoZ_wind])
    
    v_circ = 0.001 * 1e5  # Nearly zero
    r0 = 0.3 * kpc
    Edot_per_Vol = 1e-10
    Mdot_per_Vol = 1e-30
    
    params_hot = (v_circ, True, r0, Edot_per_Vol, Mdot_per_Vol)
    
    # Evaluate hot-only ODE
    dydt_hot = Hot_Wind_Evo(r, y_hot, params_hot)
    
    # Full model with tiny cloud
    M_cloud = 1e20  # Very small (~ 1e-13 M☉)
    v_cloud = v_wind  # Same as wind
    Z_cloud = metallicity
    
    y_full = np.array([
        v_wind, rho_wind, P, rhoZ_wind,
        M_cloud, v_cloud, Z_cloud
    ])
    
    # Full model config
    config_dict = {
        'N_cloud_species': 1,
        'metallicity': metallicity,
        'cooling_factor': 0.0,
        'Cooling_Factor': 0.0,
        'f_turb0': 0.1,
        'drag_coeff': 0.475,
        'M_cloud_min': 0.1 * Msun,
        'CoolingAreaChiPower': 0.5,
        'ColdTurbulenceChiPower': -0.5,
        'TurbulentVelocityChiPower': 0.0,
        'geometric_factor': 1.0,
        'Mdot_coefficient': 1.0/3.0,
        'Omwind': 4*np.pi,
        'mu': 0.61,
    }
    
    # Full model parameters
    T_cloud = 1e4
    injection_radius = r0
    injection_power = 1.0
    Ndot_cloud0 = np.array([1e-10])
    
    params_full = (
        v_circ, Ndot_cloud0, T_cloud, injection_radius,
        injection_power, config_dict, r0, Edot_per_Vol,
        Mdot_per_Vol, None  # No cooling interpolator
    )
    
    # Evaluate full ODE
    dydt_full = Wind_Evo(r, y_full, params_full)
    
    # Compare hot phase derivatives (first 4 components)
    dydt_full_hot = dydt_full[:4]
    
    # Check convergence (should be very close)
    rel_diff_v = abs(dydt_full_hot[0] - dydt_hot[0]) / (abs(dydt_hot[0]) + 1e-20)
    rel_diff_rho = abs(dydt_full_hot[1] - dydt_hot[1]) / (abs(dydt_hot[1]) + 1e-20)
    rel_diff_P = abs(dydt_full_hot[2] - dydt_hot[2]) / (abs(dydt_hot[2]) + 1e-20)
    
    # Assert convergence (within 1% for tiny cloud)
    assert rel_diff_v < 0.01, f"Velocity derivative differs by {rel_diff_v:.1%}"
    assert rel_diff_rho < 0.01, f"Density derivative differs by {rel_diff_rho:.1%}"
    assert rel_diff_P < 0.01, f"Pressure derivative differs by {rel_diff_P:.1%}"


def test_parameter_viability():
    """Test that we can identify viable parameters for low-SFR galaxies"""
    
    from multiphasegalacticwind import WindModel, WindConfig
    
    # J0021+0052 parameters
    SFR = 3.0  # M☉/yr
    v_circ = 0.001  # km/s
    
    config = WindConfig(
        N_cloud_species=5,
        cooling_factor=0.0,
        rtol=1e-6,
        atol=1e-8
    )
    
    # Test viable parameters (should succeed)
    model_viable = WindModel(
        config=config,
        SFR=SFR,
        v_circ=v_circ,
        eta_M=1.0,
        eta_M_cold=0.1,
        eta_E=2.0,
        r_max_kpc=10.0
    )
    
    solution_viable = model_viable.run()
    assert solution_viable.r[-1] > 5.0, "Viable parameters should reach > 5 kpc"
    
    # Test non-viable parameters (should fail early)
    model_nonviable = WindModel(
        config=config,
        SFR=SFR,
        v_circ=v_circ,
        eta_M=0.1,
        eta_M_cold=1.0,  # Too much cold
        eta_E=0.1,  # Too little energy
        r_max_kpc=10.0
    )
    
    solution_nonviable = model_nonviable.run()
    # Non-viable parameters should produce a much weaker outflow even if
    # the integrator reaches r_max.
    assert solution_nonviable.v[-1] < 0.5 * solution_viable.v[-1]
    assert solution_nonviable.mass_loading_at_10kpc < solution_viable.mass_loading_at_10kpc


def test_known_parameter_outcomes():
    """Test that known parameter combinations behave as expected"""
    
    from multiphasegalacticwind import WindModel, WindConfig
    
    # Configuration for tests
    config = WindConfig(
        N_cloud_species=5,
        cooling_factor=0.0,
        rtol=1e-4,  # Relaxed for faster testing
        atol=1e-6
    )
    
    # Known good parameters for J0021+0052
    # These were shown to work in our diagnostics
    model_good = WindModel(
        config=config,
        SFR=3.0,
        v_circ=0.001,
        eta_M=1.0,
        eta_M_cold=0.1,
        eta_E=2.0,
        r_max_kpc=5.0  # Short for testing
    )
    
    solution_good = model_good.run()
    
    # Should reach at least 4 kpc with these parameters
    assert solution_good.r[-1] > 4.0, f"Good parameters only reached {solution_good.r[-1]:.1f} kpc"
    
    # Should achieve reasonable velocity
    assert solution_good.v[-1] > 500.0, f"Final velocity too low: {solution_good.v[-1]:.0f} km/s"


def test_single_cloud_species_supported():
    """Test that N_cloud_species=1 is a valid configuration."""

    from multiphasegalacticwind import WindModel

    model = WindModel(
        SFR=8.0,
        v_circ=120.0,
        eta_M=0.2,
        eta_M_cold=0.1,
        eta_E=1.0,
        N_cloud_species=1,
        cloud_mass_range=(100.0, 1e5),
        r_max_kpc=5.0,
        rtol=1e-6,
        atol=1e-8,
    )

    solution = model.run()
    assert solution.r[-1] > 3.0, "Single-species run should integrate to at least 3 kpc"
    assert solution.M_clouds.shape[0] == 1, "Solution should contain exactly one cloud species"


def test_mass_flux_uses_configured_solid_angle():
    """Mass flux diagnostics should respect the configured wind solid angle."""

    from multiphasegalacticwind import WindModel, WindConfig
    from multiphasegalacticwind.constants import Msun, yr

    config = WindConfig(half_opening_angle=np.pi / 4)
    model = WindModel(
        config=config,
        SFR=6.0,
        v_circ=120.0,
        eta_M=0.25,
        eta_M_cold=0.1,
        eta_E=1.0,
        r_max_kpc=6.0,
        N_cloud_species=4,
        rtol=1e-6,
        atol=1e-8,
    )
    solution = model.run()

    idx = -1
    expected_mdot = (
        config.Omwind * solution.sol.t[idx]**2 * solution.rho[idx] * solution.sol.y[0, idx] / (Msun / yr)
    )
    assert np.isclose(solution.Mdot[idx], expected_mdot, rtol=1e-12, atol=0.0)


def test_solver_max_step_tuning_preserves_key_observables():
    """
    Looser max-step caps should keep key observables near conservative settings.
    """
    from multiphasegalacticwind import WindModel

    common = dict(
        SFR=20.0,
        v_circ=150.0,
        eta_M=0.1,
        eta_M_cold=0.2,
        eta_E=1.0,
        cloud_mass_range=(1.0, 1.0e6),
        cloud_alpha=2.0,
        N_cloud_species=13,
        r_max_kpc=30.0,
        rtol=1e-6,
        atol=1e-8,
    )
    conservative = WindModel(**common, solver_max_step_kpc=0.1).run()
    tuned = WindModel(**common, solver_max_step_kpc=0.3).run()

    assert np.isclose(tuned.v_at_10kpc, conservative.v_at_10kpc, rtol=5e-4, atol=0.0)
    assert np.isclose(
        tuned.mass_loading_at_10kpc,
        conservative.mass_loading_at_10kpc,
        rtol=5e-4,
        atol=0.0,
    )


if __name__ == "__main__":
    test_ode_convergence()
    print("✓ ODE convergence test passed")
    
    test_parameter_viability()
    print("✓ Parameter viability test passed")
    
    test_known_parameter_outcomes()
    print("✓ Known parameter outcomes test passed")

    test_single_cloud_species_supported()
    print("✓ Single cloud species test passed")

    test_mass_flux_uses_configured_solid_angle()
    print("✓ Solid-angle mass flux test passed")
    
    print("\nAll validation tests passed!")
