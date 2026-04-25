"""
Regression tests for analysis helper consistency.
"""

import numpy as np
import pytest

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.analysis_helpers import (
    Cooling_and_Acceleration,
    Gradient_Components,
    calculate_cloud_moments,
    calculate_radiative_cooling_losses,
    cloud_ksi,
)
from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import Msun, Z_solar, kb, kpc, mp


@pytest.fixture(scope="module")
def multiphase_solution():
    """Representative multiphase solution for cooling-loss diagnostics."""
    model = WindModel(
        SFR=8.0,
        v_circ=120.0,
        eta_M=0.2,
        eta_M_cold=0.1,
        eta_E=1.0,
        cloud_mass_range=(1e2, 1e4),
        N_cloud_species=3,
        r_max_kpc=6.0,
        rtol=1e-6,
        atol=1e-8,
        cooling_backend="topaz",
    )
    return model.run()


def test_cloud_mass_flux_is_geometry_consistent():
    """
    Cloud mass flux diagnostics should be independent of chosen solid angle
    when number density is computed from the same geometry.
    """
    r = 5.0 * kpc
    state = np.array([
        5.0e7, 1.0e-24, 1.0e-10, 3.0e-25,  # wind
        1.0 * Msun,                         # M_cloud
        2.0e7,                              # v_cloud
        0.3,                                # Z_cloud
    ])
    ndot = np.array([1.0e-8])

    config_wide = WindConfig(half_opening_angle=np.pi / 2)
    config_narrow = WindConfig(half_opening_angle=np.pi / 4)

    m_wide = calculate_cloud_moments(r, state, config=config_wide, Ndot_cloud0=ndot)
    m_narrow = calculate_cloud_moments(r, state, config=config_narrow, Ndot_cloud0=ndot)

    assert np.isclose(
        m_wide["Mdot_cloud_tot"],
        m_narrow["Mdot_cloud_tot"],
        rtol=1e-12,
        atol=0.0,
    )


def test_gradient_components_raises_until_implemented():
    state = np.array([5.0e7, 1.0e-24, 1.0e-10, 3.0e-25, 1.0, 1.0, 0.3])
    with pytest.raises(NotImplementedError):
        Gradient_Components(5.0 * kpc, state)


def test_radiative_cooling_losses_contract_and_shapes(multiphase_solution):
    losses = calculate_radiative_cooling_losses(multiphase_solution)

    required_keys = {
        "r_kpc",
        "q_hot_cgs",
        "q_interface_cgs",
        "q_total_cgs",
        "q_interface_species_cgs",
        "dLdr_hot_cgs",
        "dLdr_interface_cgs",
        "dLdr_total_cgs",
        "L_hot_cgs",
        "L_interface_cgs",
        "L_total_cgs",
        "L_hot_cumulative_cgs",
        "L_interface_cumulative_cgs",
        "L_total_cumulative_cgs",
    }
    assert required_keys.issubset(losses.keys())

    n_r = losses["r_kpc"].size
    n_species = multiphase_solution.model.N_cloud_species
    assert n_r > 1
    assert losses["q_interface_species_cgs"].shape == (n_species, n_r)

    profile_keys = [
        "q_hot_cgs",
        "q_interface_cgs",
        "q_total_cgs",
        "dLdr_hot_cgs",
        "dLdr_interface_cgs",
        "dLdr_total_cgs",
        "L_hot_cumulative_cgs",
        "L_interface_cumulative_cgs",
        "L_total_cumulative_cgs",
    ]
    for key in profile_keys:
        arr = losses[key]
        assert arr.shape == (n_r,)
        assert np.all(np.isfinite(arr))
        assert np.all(arr >= 0.0)

    assert np.all(np.isfinite(losses["q_interface_species_cgs"]))
    assert np.all(losses["q_interface_species_cgs"] >= 0.0)
    assert losses["L_hot_cgs"] >= 0.0
    assert losses["L_interface_cgs"] >= 0.0
    assert losses["L_total_cgs"] >= 0.0

    assert np.allclose(losses["q_total_cgs"], losses["q_hot_cgs"] + losses["q_interface_cgs"])
    assert np.allclose(losses["dLdr_total_cgs"], losses["dLdr_hot_cgs"] + losses["dLdr_interface_cgs"])
    assert losses["L_total_cgs"] == pytest.approx(losses["L_hot_cgs"] + losses["L_interface_cgs"])
    assert losses["L_hot_cgs"] == pytest.approx(losses["L_hot_cumulative_cgs"][-1])
    assert losses["L_interface_cgs"] == pytest.approx(losses["L_interface_cumulative_cgs"][-1])
    assert losses["L_total_cgs"] == pytest.approx(losses["L_total_cumulative_cgs"][-1])


def test_solution_method_matches_helper(multiphase_solution):
    r_start = float(multiphase_solution.r[0])
    r_stop = float(multiphase_solution.r[-1])
    r_min = r_start + 0.21 * (r_stop - r_start)
    r_max = r_start + 0.79 * (r_stop - r_start)

    expected = calculate_radiative_cooling_losses(
        multiphase_solution,
        r_min_kpc=r_min,
        r_max_kpc=r_max,
    )
    actual = multiphase_solution.calculate_radiative_cooling_losses(
        r_min_kpc=r_min,
        r_max_kpc=r_max,
    )

    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, np.ndarray):
            assert np.allclose(actual_value, expected_value)
        else:
            assert actual_value == pytest.approx(expected_value)


def test_hot_cooling_zero_when_cooling_factor_zero():
    model = WindModel(
        SFR=8.0,
        v_circ=120.0,
        eta_M=0.2,
        eta_M_cold=0.1,
        eta_E=1.0,
        cloud_mass_range=(1e2, 1e4),
        N_cloud_species=3,
        r_max_kpc=6.0,
        rtol=1e-6,
        atol=1e-8,
        cooling_backend="topaz",
        Cooling_Factor=0.0,
    )
    solution = model.run()
    losses = calculate_radiative_cooling_losses(solution)

    assert np.allclose(losses["q_hot_cgs"], 0.0)
    assert np.allclose(losses["dLdr_hot_cgs"], 0.0)
    assert losses["L_hot_cgs"] == pytest.approx(0.0)
    assert np.allclose(losses["q_total_cgs"], losses["q_interface_cgs"])
    assert losses["L_total_cgs"] == pytest.approx(losses["L_interface_cgs"])


def test_interface_zero_when_no_cold_injection():
    model = WindModel(
        SFR=8.0,
        v_circ=120.0,
        eta_M=0.2,
        eta_M_cold=0.0,
        eta_E=1.0,
        cloud_mass_range=(1e2, 1e4),
        N_cloud_species=3,
        r_max_kpc=6.0,
        rtol=1e-6,
        atol=1e-8,
        cooling_backend="topaz",
    )
    solution = model.run()
    losses = calculate_radiative_cooling_losses(solution)

    assert np.allclose(losses["q_interface_cgs"], 0.0)
    assert np.allclose(losses["q_interface_species_cgs"], 0.0)
    assert np.allclose(losses["dLdr_interface_cgs"], 0.0)
    assert losses["L_interface_cgs"] == pytest.approx(0.0)
    assert np.allclose(losses["q_total_cgs"], losses["q_hot_cgs"])
    assert losses["L_total_cgs"] == pytest.approx(losses["L_hot_cgs"])


def test_radius_window_integration_behavior(multiphase_solution):
    full = calculate_radiative_cooling_losses(multiphase_solution)
    r_start = float(full["r_kpc"][0])
    r_stop = float(full["r_kpc"][-1])
    r_min = r_start + 0.17 * (r_stop - r_start)
    r_max = r_start + 0.73 * (r_stop - r_start)

    sub = calculate_radiative_cooling_losses(
        multiphase_solution,
        r_min_kpc=r_min,
        r_max_kpc=r_max,
    )

    assert sub["r_kpc"][0] == pytest.approx(r_min)
    assert sub["r_kpc"][-1] == pytest.approx(r_max)
    assert sub["L_total_cgs"] <= full["L_total_cgs"] * (1.0 + 1e-12) + 1e-30


def test_cloud_ksi_uses_core_chi_definition():
    rho_wind = 1.0e-26
    t_wind = 1.0e6
    t_cloud = 1.0e4
    mu = 0.62
    pressure = rho_wind * kb * t_wind / (mu * mp)
    state = np.array([
        5.0e7,
        rho_wind,
        pressure,
        rho_wind * Z_solar,
        1.0 * Msun,
        2.0e7,
        0.3 * Z_solar,
    ])
    r = 1.0 * kpc
    config_no_chi_power = WindConfig(
        mu=mu,
        T_cl=t_cloud,
        M_cloud_min=1e-3 * Msun,
        TurbulentVelocityChiPower=0.0,
    )
    config_chi_power = WindConfig(
        mu=mu,
        T_cl=t_cloud,
        M_cloud_min=1e-3 * Msun,
        TurbulentVelocityChiPower=1.0,
    )

    ksi_no_chi_power = cloud_ksi(r, state, config_no_chi_power, N_cloud_species=1)[0]
    ksi_chi_power = cloud_ksi(r, state, config_chi_power, N_cloud_species=1)[0]

    chi = t_wind / t_cloud
    assert ksi_chi_power / ksi_no_chi_power == pytest.approx(1.0 / chi)


def test_cooling_and_acceleration_uses_configurable_v_circ():
    rho_wind = 1.0e-26
    t_wind = 1.0e6
    mu = 0.62
    pressure = rho_wind * kb * t_wind / (mu * mp)
    state = np.array([
        5.0e7,
        rho_wind,
        pressure,
        rho_wind * Z_solar,
    ])
    r = 1.0 * kpc
    config = WindConfig(mu=mu)

    accel_150 = Cooling_and_Acceleration(r, state, config, v_circ_kms=150.0)
    accel_300 = Cooling_and_Acceleration(r, state, config, v_circ_kms=300.0)

    assert accel_300["dv_dr_grav"] == pytest.approx(4.0 * accel_150["dv_dr_grav"])
