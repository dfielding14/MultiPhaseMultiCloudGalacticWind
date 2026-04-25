"""Tests for normalized CLASSY observational data loaders."""

from __future__ import annotations

import numpy as np
import pytest

from multiphasegalacticwind.classy_observations import (
    build_classy_inference_inputs,
    build_binned_dndv_observation,
    get_classy_observation,
    load_classy_observations,
)


def test_classy_object_counts():
    all_obs = load_classy_observations(include_missing_profiles=True)
    prof_obs = load_classy_observations(include_missing_profiles=False)

    assert len(all_obs) == 43
    assert len(prof_obs) == 43


def test_classy_profile_record_has_required_fields():
    obs = get_classy_observation("J0021+0052")

    assert obs.has_profile
    assert obs.sfr_msun_per_yr > 0.0
    assert obs.r_gal_kpc > 0.0
    assert obs.r50_kpc > 0.0
    assert obs.r_star_kpc > 0.0
    assert obs.v_circ_kms > 0.0
    assert obs.v_circ_kms == pytest.approx(72.1, rel=0.0, abs=1e-12)
    assert obs.velocity_kms.size == obs.dndv_cm2_per_kms.size
    assert obs.velocity_kms.size >= 10
    assert np.all(np.isfinite(obs.velocity_kms))
    assert np.all(np.isfinite(obs.dndv_cm2_per_kms))
    assert np.all(obs.dndv_cm2_per_kms > 0.0)
    assert np.all(np.diff(obs.velocity_kms) >= 0.0)


def test_removed_object_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown CLASSY object"):
        get_classy_observation("J1612+0817")


def test_build_binned_dndv_observation_is_inference_ready():
    obs = get_classy_observation("J1429+0643")
    binned = build_binned_dndv_observation(
        obs,
        num_bins=20,
        vmin_kms=0.0,
        vmax_kms=800.0,
        fractional_error=0.10,
        min_error_floor_fraction=0.03,
        ar1_rho=0.5,
    )

    assert binned.velocity_bins_kms.shape == (20,)
    assert binned.observed_dndv.shape == (20,)
    assert binned.covariance_dndv.shape == (20, 20)
    assert np.all(np.isfinite(binned.observed_dndv))
    assert np.all(binned.observed_dndv > 0.0)
    assert np.all(np.linalg.eigvalsh(binned.covariance_dndv) > 0.0)


def test_build_classy_inference_inputs_shape5_wires_vcirc_and_radius():
    obs = get_classy_observation("J0021+0052")
    inputs = build_classy_inference_inputs(obs)

    assert inputs.observable_set == "logm0_mean_sigma_skew_kurt"
    assert inputs.model_kwargs["sfr"] == pytest.approx(obs.sfr_msun_per_yr)
    assert inputs.model_kwargs["r_star_kpc"] == pytest.approx(obs.r_gal_kpc)
    assert inputs.model_kwargs["v_circ"] == pytest.approx(obs.v_circ_kms)
    assert inputs.observed.shape == (5,)
    assert inputs.covariance.shape == (5, 5)
    assert inputs.velocity_bins_kms is None
    assert len(inputs.observable_names) == 5
    assert np.all(np.isfinite(inputs.observed))
    assert np.all(np.linalg.eigvalsh(inputs.covariance) > 0.0)


def test_build_classy_inference_inputs_dndv_mode_has_expected_bins():
    obs = get_classy_observation("J1429+0643")
    inputs = build_classy_inference_inputs(
        obs,
        observable_set="dndv_binned",
        radius_field="r_star_kpc",
        dndv_num_bins=20,
    )

    assert inputs.observable_set == "dndv_binned"
    assert inputs.model_kwargs["r_star_kpc"] == pytest.approx(obs.r_star_kpc)
    assert inputs.model_kwargs["v_circ"] == pytest.approx(obs.v_circ_kms)
    assert inputs.velocity_bins_kms is not None
    assert inputs.velocity_bins_kms.shape == (20,)
    assert inputs.observed.shape == (20,)
    assert inputs.covariance.shape == (20, 20)
    assert np.all(np.isfinite(inputs.observed))
    assert np.all(inputs.observed > 0.0)
    assert np.all(np.linalg.eigvalsh(inputs.covariance) > 0.0)
