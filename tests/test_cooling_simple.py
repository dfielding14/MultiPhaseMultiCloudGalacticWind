#!/usr/bin/env python3
"""
Additional smoke tests for cooling utilities.
"""

import numpy as np

from multiphasegalacticwind.cooling import (
    get_lambda_interpolator,
    get_cooling_interpolator,
    tcool_P,
    Lambda_P,
    get_tcool_min_interpolators,
)


def test_lambda_interpolator_returns_finite_values():
    Lambda = get_lambda_interpolator()
    val = Lambda((0.0, 5.0, 1.0, 0.0))
    assert np.isfinite(val)
    assert val > 0


def test_cooling_interpolator_is_cached_by_parameters():
    interp1 = get_cooling_interpolator(0.62, 1.0, 0.0)
    interp2 = get_cooling_interpolator(0.62, 1.0, 0.0)
    interp3 = get_cooling_interpolator(0.62, 0.1, 0.0)

    assert interp1 is interp2
    assert interp3 is not interp1


def test_tcool_and_lambda_behave_reasonably():
    T = 1e5
    P_over_kb = 1e3

    tcool = tcool_P(T, P_over_kb, 1.0, 0.0, 0.62)
    lam = Lambda_P(T, P_over_kb, 1.0, 0.0, 0.62)

    assert np.isfinite(tcool)
    assert tcool > 0
    assert np.isfinite(lam)
    assert lam > 0


def test_tcool_min_interpolators_return_positive_values():
    T_tcool_min_P, tcool_min_P = get_tcool_min_interpolators()

    T_min = T_tcool_min_P((1e3, 1.0))
    tcool_min = tcool_min_P((1e3, 1.0))

    assert np.isfinite(T_min)
    assert np.isfinite(tcool_min)
    assert T_min > 0
    assert tcool_min > 0


def test_tcool_array_inputs_are_finite():
    T = np.array([2e4, 1e5, 1e6])
    P_over_kb = 1e3 * np.ones_like(T)
    tcool = tcool_P(T, P_over_kb, 1.0, 0.0, 0.62)

    assert tcool.shape == T.shape
    assert np.all(np.isfinite(tcool))
    assert np.all(tcool > 0)
