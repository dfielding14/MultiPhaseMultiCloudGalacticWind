#!/usr/bin/env python3
"""
Unit tests for WindConfig defaults, validation, and helpers.
"""

import numpy as np
import pytest

from multiphasegalacticwind.config import WindConfig, get_default_config
from multiphasegalacticwind.constants import Msun


def test_default_initialization():
    config = WindConfig()

    assert config.mu == 0.62
    assert config.Z_hot_over_Z_solar == 10**-0.5
    assert config.redshift == 0.0
    assert config.half_opening_angle == np.pi / 2
    assert config.v_cloud_init == 100.0
    assert config.v_cloud_min == 1.0


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


def test_to_dict_contains_expected_keys_and_values():
    config = WindConfig()
    config_dict = config.to_dict()

    assert config_dict["mu"] == config.mu
    assert config_dict["Z_hot_over_Z_solar"] == config.Z_hot_over_Z_solar
    assert config_dict["metallicity"] == config.metallicity
    assert config_dict["v_cloud_init"] == config.v_cloud_init
    assert config_dict["v_cloud_min"] == config.v_cloud_min
    assert config_dict["sonic_point_offset"] == config.sonic_point_offset


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
    ],
)
def test_validate_rejects_invalid_values(kwargs):
    config = WindConfig(**kwargs)
    with pytest.raises(ValueError):
        config.validate()


def test_set_defaults_rejects_unknown_parameter():
    with pytest.raises(ValueError):
        WindConfig.set_defaults(does_not_exist=1)
