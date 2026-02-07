"""
Regression tests for analysis helper consistency.
"""

import numpy as np
import pytest

from multiphasegalacticwind.analysis_helpers import calculate_cloud_moments, Gradient_Components
from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import Msun, kpc


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
