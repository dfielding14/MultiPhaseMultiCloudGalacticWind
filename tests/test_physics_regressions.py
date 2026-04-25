"""Focused regression tests for wind RHS physics corrections."""

import numpy as np
import pytest

from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import Msun, Z_solar, gamma, kb, kpc, mp
from multiphasegalacticwind.core_physics import _cloud_exchange_kernel_numba
from multiphasegalacticwind.jax_physics import build_jax_wind_params, wind_evo_jax


def _jax_rhs_for_cooling_factor(cooling_factor: float) -> np.ndarray:
    config = WindConfig(Cooling_Factor=cooling_factor)
    rho_wind = 1.0e-26
    t_wind = 1.0e6
    pressure = rho_wind * kb * t_wind / (config.mu * mp)
    state = np.array([
        8.0e7,
        rho_wind,
        pressure,
        rho_wind * Z_solar,
        1.0e-4 * Msun,
        1.0e7,
        0.3 * Z_solar,
    ])
    params = build_jax_wind_params(
        v_circ=1.0e7,
        Ndot_cloud0=np.array([0.0]),
        T_cloud=config.T_cl,
        injection_radius=2.0 * kpc,
        injection_power=0.0,
        config_dict=config.to_dict(),
        r0=0.1 * kpc,
        Edot_per_Vol=0.0,
        Mdot_per_Vol=0.0,
    )
    return np.asarray(wind_evo_jax(state, 1.0 * kpc, params))


def test_jax_rhs_cooling_factor_scales_hot_cooling_term():
    rhs_no_cooling = _jax_rhs_for_cooling_factor(0.0)
    rhs_half = _jax_rhs_for_cooling_factor(0.5)
    rhs_one = _jax_rhs_for_cooling_factor(1.0)
    rhs_two = _jax_rhs_for_cooling_factor(2.0)

    cooling_delta_one = rhs_one[:4] - rhs_no_cooling[:4]
    assert np.linalg.norm(cooling_delta_one) > 0.0
    assert np.allclose(rhs_half[:4] - rhs_no_cooling[:4], 0.5 * cooling_delta_one)
    assert np.allclose(rhs_two[:4] - rhs_no_cooling[:4], 2.0 * cooling_delta_one)


def test_cloud_loss_enthalpy_uses_sound_speed_squared_once():
    r = 1.0 * kpc
    v_wind = 1.0e7
    rho_wind = 1.0e-26
    rho_cloud = 1.0e-24
    chi = rho_cloud / rho_wind
    m_cloud = 1.0 * Msun
    v_cloud = 4.0e6
    t_cloud = 1.0e4
    mu = 0.62
    cs_cl_sq = gamma * kb * t_cloud / (mu * mp)

    (
        _sum_mass,
        _sum_momentum,
        sum_energy,
        _sum_metals,
        dM_cloud_dr,
        _dv_cloud_dr,
        _dZ_cloud_dr,
    ) = _cloud_exchange_kernel_numba(
        r,
        0.0,
        v_wind,
        rho_wind,
        Z_solar,
        0.0,
        0.0,
        cs_cl_sq,
        rho_cloud,
        chi,
        np.array([1.0e-6]),
        np.array([m_cloud]),
        np.array([v_cloud]),
        np.array([0.3 * Z_solar]),
        np.array([1.0e30]),
        0.5 * kpc,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / 3.0,
        0.0,
        0.1,
        4.0 * np.pi,
        1.0e5,
    )

    number_density_cloud = 1.0e-6 / (4.0 * np.pi * r * r * v_cloud)
    mdot_loss = dM_cloud_dr[0] * v_cloud
    recovered_cloud_bernoulli = sum_energy / (number_density_cloud * mdot_loss)

    expected = 0.5 * v_cloud * v_cloud + cs_cl_sq / (gamma - 1.0)
    overcounted = 0.5 * v_cloud * v_cloud + (gamma / (gamma - 1.0)) * cs_cl_sq
    assert recovered_cloud_bernoulli == pytest.approx(expected)
    assert recovered_cloud_bernoulli != pytest.approx(overcounted)
