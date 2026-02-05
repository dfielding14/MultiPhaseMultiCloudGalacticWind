"""
Regression tests for observable mappings.
"""

import numpy as np
import pytest

from multiphasegalacticwind import WindModel
from multiphasegalacticwind.observables import (
    calculate_column_density_distribution,
    calculate_column_density_by_species,
)


@pytest.fixture(scope="module")
def m82_solution():
    """A representative multi-species solution for observable tests."""
    model = WindModel(
        SFR=20.0,
        v_circ=150.0,
        eta_M=0.1,
        eta_M_cold=0.2,
        eta_E=1.0,
        cloud_mass_range=(1.0, 1.0e6),
        cloud_alpha=2.0,
        N_cloud_species=6,
        r_max_kpc=20.0,
        rtol=1e-6,
        atol=1e-8,
    )
    return model.run()


def test_column_density_total_matches_species_sum(m82_solution):
    """
    Total dN/dv should equal the sum of per-species contributions on one grid.
    """
    r_max = min(20.0, float(m82_solution.r[-1]))
    v_total, dN_dv_total = calculate_column_density_distribution(
        m82_solution, r_min_kpc=0.3, r_max_kpc=r_max
    )
    v_species, dN_dv_dict = calculate_column_density_by_species(
        m82_solution, r_min_kpc=0.3, r_max_kpc=r_max
    )

    species = np.vstack(dN_dv_dict["species"])
    dN_dv_sum = np.sum(species, axis=0)

    assert np.allclose(v_total, v_species)
    assert species.shape[1] == v_total.size
    assert np.all(np.isfinite(dN_dv_total))
    assert np.all(np.isfinite(species))
    assert np.all(dN_dv_total >= 0.0)
    assert np.all(species >= 0.0)

    denom = np.trapezoid(np.abs(dN_dv_total), v_total)
    rel_error = np.trapezoid(np.abs(dN_dv_total - dN_dv_sum), v_total) / denom
    assert rel_error < 1e-12


def test_single_species_column_density_is_ordered_and_finite():
    """Single-species dN/dv should be finite and evaluated on an ordered v-grid."""
    model = WindModel(
        SFR=10.0,
        v_circ=120.0,
        eta_M=0.2,
        eta_M_cold=0.05,
        eta_E=1.0,
        cloud_mass_range=(1e4, 1e4),
        N_cloud_species=1,
        r_max_kpc=10.0,
        rtol=1e-6,
        atol=1e-8,
    )
    solution = model.run()

    r_max = min(10.0, float(solution.r[-1]))
    v_cloud, dN_dv = calculate_column_density_distribution(
        solution, cloud_index=0, r_min_kpc=0.3, r_max_kpc=r_max
    )

    assert v_cloud.size > 2
    assert np.all(np.diff(v_cloud) > 0.0)
    assert np.all(np.isfinite(dN_dv))
    assert np.all(dN_dv >= 0.0)

