"""
Regression tests for analysis helper consistency.
"""

import numpy as np
import pytest

from multiphasegalacticwind.analysis_helpers import (
    Cooling_and_Acceleration,
    Gradient_Components,
    calculate_cloud_moments,
    cloud_ksi,
)
from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import Msun, Z_solar, kb, kpc, mp


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


def test_cloud_ksi_uses_core_chi_definition():
    rho_wind = 1.0e-26
    t_wind = 1.0e6
    t_cloud = 1.0e4
    mu = 0.62
    pressure = rho_wind * kb * t_wind / (mu * mp)
    r = 1.0 * kpc
    state = np.array([
        8.0e7,
        rho_wind,
        pressure,
        rho_wind * Z_solar,
        1.0e30,
        1.0e7,
        0.3 * Z_solar,
    ])

    config_no_chi_power = WindConfig(
        mu=mu,
        T_cl=t_cloud,
        f_turb0=0.1,
        TurbulentVelocityChiPower=0.0,
    )
    config_chi_power = WindConfig(
        mu=mu,
        T_cl=t_cloud,
        f_turb0=0.1,
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
        8.0e7,
        rho_wind,
        pressure,
        rho_wind * Z_solar,
    ])
    r = 1.0 * kpc
    config = WindConfig(mu=mu)

    accel_150 = Cooling_and_Acceleration(r, state, config, v_circ_kms=150.0)
    accel_300 = Cooling_and_Acceleration(r, state, config, v_circ_kms=300.0)

    assert accel_300["dv_dr_grav"] == pytest.approx(4.0 * accel_150["dv_dr_grav"])
