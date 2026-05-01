#!/usr/bin/env python3
"""
Unit tests for WindConfig defaults, validation, and helpers.
"""

import numpy as np
import pytest

from multiphasegalacticwind.config import WindConfig, get_default_config
from multiphasegalacticwind.constants import Msun
from multiphasegalacticwind.wind_model import WindModel


def test_default_initialization():
    config = WindConfig()

    assert config.mu == 0.62
    assert config.Z_hot_over_Z_solar == 10**-0.5
    assert config.redshift == 0.0
    assert config.cooling_backend == "topaz"
    assert config.topaz_cooling_table_path is None
    assert config.half_opening_angle == np.pi / 2
    assert config.v_cloud_init == 100.0
    assert config.v_cloud_min == 1.0
    assert config.solver_max_step_kpc == 0.3
    assert config.solver_first_step_kpc == 1e-12
    assert config.A_mix == 1.0
    assert config.beta_chi_mix == 0.0
    assert config.mixing_chi_pivot == 100.0


def test_custom_initialization():
    config = WindConfig(
        mu=0.59,
        Z_hot_over_Z_solar=0.1,
        redshift=2.0,
        v_cloud_init=200.0,
        v_cloud_min=10.0,
        f_turb0=0.2,
    )

    assert config.mu == 0.59
    assert config.Z_hot_over_Z_solar == 0.1
    assert config.redshift == 2.0
    assert config.v_cloud_init == 200.0
    assert config.v_cloud_min == 10.0
    assert config.f_turb0 == 0.2


def test_backward_compatibility_metallicity_alias():
    config = WindConfig(metallicity=0.5)
    assert config.Z_hot_over_Z_solar == 0.5
    assert config.metallicity == 0.5


def test_backward_compatibility_cooling_factor_alias():
    config = WindConfig(cooling_factor=0.5)
    assert config.Cooling_Factor == 0.5
    assert config.to_dict()["Cooling_Factor"] == 0.5


def test_conflicting_alias_values_raise():
    with pytest.raises(ValueError):
        WindConfig(Cooling_Factor=1.0, cooling_factor=0.5)

    with pytest.raises(ValueError):
        WindConfig(Z_hot_over_Z_solar=1.0, metallicity=0.5)


def test_to_dict_contains_expected_keys_and_values():
    config = WindConfig()
    config_dict = config.to_dict()

    assert config_dict["mu"] == config.mu
    assert config_dict["Z_hot_over_Z_solar"] == config.Z_hot_over_Z_solar
    assert config_dict["metallicity"] == config.metallicity
    assert config_dict["v_cloud_init"] == config.v_cloud_init
    assert config_dict["v_cloud_min"] == config.v_cloud_min
    assert config_dict["sonic_point_offset"] == config.sonic_point_offset
    assert config_dict["A_mix"] == config.A_mix
    assert config_dict["beta_chi_mix"] == config.beta_chi_mix
    assert config_dict["mixing_chi_pivot"] == config.mixing_chi_pivot


def test_get_default_config_returns_new_instances():
    config1 = get_default_config()
    config2 = get_default_config()

    assert config1 is not config2
    assert config1.to_dict() == config2.to_dict()


def test_omwind_from_half_opening_angle():
    config1 = WindConfig(half_opening_angle=np.pi / 2)
    expected1 = 4 * np.pi * (1.0 - np.cos(np.pi / 2))
    assert np.isclose(config1.Omwind, expected1)

    config2 = WindConfig(half_opening_angle=np.pi / 4)
    expected2 = 4 * np.pi * (1.0 - np.cos(np.pi / 4))
    assert np.isclose(config2.Omwind, expected2)

    # Solid angle must never exceed full sphere.
    config3 = WindConfig(half_opening_angle=np.pi)
    assert np.isclose(config3.Omwind, 4 * np.pi)


def test_physical_defaults():
    config = WindConfig()

    assert config.E_SN == 1e51
    assert config.mstar == 100.0
    assert config.M_cloud_min == 1e-2 * Msun
    assert np.isclose(config.cold_cloud_injection_radial_extent_frac, 1.33)


def test_parameter_ranges_and_validation():
    config = WindConfig(
        Z_hot_over_Z_solar=1e-3,
        redshift=10.0,
        half_opening_angle=np.pi * 0.99,
        sonic_point_offset=1e-8,
    )
    config.validate()  # should not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu": 0.0},
        {"redshift": -1.0},
        {"half_opening_angle": 0.0},
        {"f_turb0": 1.0},
        {"drag_coeff": 0.0},
        {"sonic_point_offset": 0.0},
        {"solver_max_step_kpc": 0.0},
        {"solver_first_step_kpc": 0.0},
        {"cooling_backend": "legacy"},
        {"cooling_backend": "invalid"},
        {"Cooling_Factor": -1.0},
        {"A_mix": 0.0},
        {"beta_chi_mix": np.nan},
        {"mixing_chi_pivot": 0.0},
    ],
)
def test_validate_rejects_invalid_values(kwargs):
    config = WindConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()


def test_set_defaults_rejects_unknown_parameter():
    with pytest.raises(ValueError):
        WindConfig.set_defaults(does_not_exist=1)


def test_wind_config_init_rejects_unknown_parameter():
    with pytest.raises(ValueError):
        WindConfig(does_not_exist=1)


def test_wind_model_config_kwargs_reject_unknown_parameter():
    with pytest.raises(ValueError):
        WindModel(does_not_exist=1)


def test_wind_model_supports_cooling_factor_alias():
    model = WindModel(cooling_factor=0.0, eta_M_cold=0.0, r_max_kpc=1.0)
    assert model.config.Cooling_Factor == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"N_cloud_species": 0},
        {"N_cloud_species": 1.5},
        {"cloud_mass_range": (-1.0, 10.0)},
        {"cloud_mass_range": (0.0, 10.0)},
        {"cloud_mass_range": (np.nan, 10.0)},
        {"cloud_mass_range": (1.0, np.inf)},
        {"cloud_mass_range": (10.0, 1.0)},
        {"cloud_mass_range": (10.0, 10.0), "N_cloud_species": 1},
    ],
)
def test_wind_model_rejects_invalid_cloud_distribution_inputs(kwargs):
    with pytest.raises(ValueError):
        WindModel(**kwargs)
