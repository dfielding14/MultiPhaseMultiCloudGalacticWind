"""
Regression tests for integration event logic.
"""

import numpy as np

from multiphasegalacticwind.core_physics import (
    create_cloud_velocity_low_event,
    create_step_size_event,
)
from multiphasegalacticwind.constants import Msun


def _base_params(config_overrides=None):
    config_dict = {
        "M_cloud_min": 0.1 * Msun,
        "v_cloud_min": 1.0,  # km/s
    }
    if config_overrides:
        config_dict.update(config_overrides)
    return (0.0, np.array([1.0]), 1e4, 1.0, 1.0, config_dict, 1.0, 0.0, 0.0, None)


def test_cloud_velocity_event_ignores_inactive_clouds():
    params = _base_params()
    event = create_cloud_velocity_low_event(params)

    # Two cloud species:
    # - Species 0 is inactive (mass below threshold) and very slow.
    # - Species 1 is active and above velocity threshold.
    state_ok = np.array([
        1.0, 1.0, 1.0, 1.0,         # wind state
        1e-4 * Msun, 1.0 * Msun,    # M_cloud
        0.1e5, 2.0e5,               # v_cloud [cm/s]
        1.0, 1.0,                   # Z_cloud
    ])
    assert event(1.0, state_ok) > 0.0

    # Now make the active cloud sub-threshold: event must trigger.
    state_trigger = state_ok.copy()
    state_trigger[7] = 0.5e5  # 0.5 km/s
    assert event(1.0, state_trigger) < 0.0


def test_step_size_event_uses_relative_progress_and_resets():
    params = _base_params()
    event = create_step_size_event(params, min_relative_step=1e-6, n_small_steps=3)

    y = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Initialization call.
    assert event(1.0, y) > 0.0

    # Small relative steps accumulate.
    assert event(1.0 + 1e-7, y) > 0.0
    assert event(1.0 + 2e-7, y) > 0.0
    assert event(1.0 + 3e-7, y) == 0.0

    # A fresh monitor should reset on a large enough step.
    event2 = create_step_size_event(params, min_relative_step=1e-6, n_small_steps=3)
    assert event2(1.0, y) > 0.0
    assert event2(1.0 + 1e-7, y) > 0.0
    assert event2(1.0 + 1e-3, y) > 0.0  # large relative step clears counter
    assert event2(1.0 + 1e-3 + 1e-7, y) > 0.0
