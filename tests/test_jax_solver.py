"""Regression tests for the JAX solver path and Jacobian diagnostics."""

import numpy as np

from multiphasegalacticwind import WindModel


def test_jax_rhs_jacobian_shape_and_finite():
    model = WindModel(
        SFR=8.0,
        v_circ=120.0,
        eta_M=0.2,
        eta_M_cold=0.1,
        eta_E=1.0,
        N_cloud_species=3,
        cloud_mass_range=(1e2, 1e5),
        r_max_kpc=6.0,
        rtol=1e-6,
        atol=1e-8,
        cooling_backend="topaz",
    )
    solution = model.run()

    state0 = solution.sol.y[:, 0]
    jac = model.jacobian_rhs(float(solution.r[0]), state0)
    trajectory = model.run_differentiable()

    n = state0.size
    assert jac.shape == (n, n)
    assert np.all(np.isfinite(jac))
    assert trajectory.shape[0] >= solution.r.size
    assert trajectory.shape[1] == n
