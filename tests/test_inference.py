"""Tests for inference utilities and MAP/HMC fitting path."""

import importlib.util
import numpy as np
import pytest
import sys
from pathlib import Path

from multiphasegalacticwind.inference import (
    MomentInferenceModel,
    build_covariance,
    covariance_to_correlation,
    resolve_nuts_chain_method,
)
from multiphasegalacticwind.constants import kpc
from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.jax_physics import has_diffrax


def test_covariance_builder_and_correlation_roundtrip():
    sigma = np.array([2.0, 3.0, 5.0], dtype=float)
    corr = np.array(
        [
            [1.0, 0.4, -0.2],
            [0.4, 1.0, 0.1],
            [-0.2, 0.1, 1.0],
        ],
        dtype=float,
    )
    cov = build_covariance(sigma, corr)

    assert cov.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)

    corr_back = covariance_to_correlation(cov)
    assert np.allclose(np.diag(corr_back), 1.0)
    assert np.allclose(corr_back, corr_back.T)


def test_map_and_hmc_smoke_on_synthetic_moments():
    model = MomentInferenceModel(
        sfr=4.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=8.0,
        step_kpc=0.05,
        n_cloud_species=4,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
    )

    theta_true = np.array([0.25, 0.15, 0.92], dtype=float)
    moments_true = model.predict_moments(theta_true)

    sigma = 0.10 * moments_true
    covariance = build_covariance(sigma, np.eye(3))

    observed = moments_true * np.array([1.03, 0.98, 1.02], dtype=float)

    fit = model.fit_posterior(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.15, 0.25, 0.85),
        map_max_iter=12,
        map_num_starts=2,
        hmc_num_warmup=15,
        hmc_num_samples=25,
        hmc_step_size=0.02,
        hmc_leapfrog_steps=8,
        sampler="hmc",
        seed=7,
    )

    assert fit.map.theta_map.shape == (3,)
    assert np.all(np.isfinite(fit.map.theta_map))
    assert np.all(fit.map.theta_map > 0.0)
    assert fit.map.theta_map[2] < 1.0

    assert fit.hmc.samples_theta.shape == (25, 3)
    assert np.all(np.isfinite(fit.hmc.samples_theta))
    assert np.all(fit.hmc.samples_theta[:, 2] < 1.0)
    assert 0.0 <= fit.hmc.acceptance_rate <= 1.0
    assert fit.hmc.sampler == "hmc"
    assert fit.hmc.num_chains == 1

    pred_map = fit.map.predicted_moments
    rel_err = np.abs(pred_map - observed) / np.maximum(np.abs(observed), 1e-30)
    assert np.all(rel_err < 0.35)


def test_fit_posterior_reports_runtime_and_status_callback():
    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=3,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
    )

    theta_true = np.array([0.20, 0.12, 0.90], dtype=float)
    moments_true = model.predict_moments(theta_true)
    covariance = build_covariance(0.10 * moments_true, np.eye(3))

    observed = moments_true * np.array([1.01, 0.99, 1.02], dtype=float)
    status_messages: list[str] = []

    fit = model.fit_posterior(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.2, 0.2, 0.8),
        map_max_iter=8,
        map_num_starts=2,
        hmc_num_warmup=8,
        hmc_num_samples=10,
        sampler="hmc",
        seed=123,
        status_callback=status_messages.append,
    )

    assert fit.runtime_seconds is not None
    assert set(fit.runtime_seconds.keys()) == {"map", "posterior_sampling", "total"}
    assert fit.runtime_seconds["map"] >= 0.0
    assert fit.runtime_seconds["posterior_sampling"] >= 0.0
    assert fit.runtime_seconds["total"] >= fit.runtime_seconds["map"]
    assert fit.runtime_seconds["total"] >= fit.runtime_seconds["posterior_sampling"]

    assert len(status_messages) >= 4
    assert any("MAP optimization" in msg for msg in status_messages)
    assert any("posterior sampling" in msg for msg in status_messages)


def test_shape_observable_mode_predicts_and_fits():
    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=3,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
        observable_set="logm0_mean_sigma_skew_kurt",
    )

    theta_true = np.array([0.20, 0.12, 0.90], dtype=float)
    obs_true = model.predict_observables(theta_true)
    raw_true = model.predict_raw_moments(theta_true)

    assert obs_true.shape == (5,)
    assert raw_true.shape == (5,)
    assert np.all(np.isfinite(obs_true))
    assert np.all(np.isfinite(raw_true))
    assert obs_true[2] > 0.0

    sigma = np.array(
        [
            0.10,
            0.10 * abs(obs_true[1]),
            0.10 * abs(obs_true[2]),
            0.20,
            0.40,
        ],
        dtype=float,
    )
    covariance = build_covariance(sigma, np.eye(5))
    observed = obs_true * np.array([1.01, 0.98, 1.02, 1.05, 0.95], dtype=float)

    fit = model.fit_posterior(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.2, 0.2, 0.8),
        map_max_iter=8,
        map_num_starts=2,
        hmc_num_warmup=8,
        hmc_num_samples=10,
        sampler="hmc",
        seed=9,
    )

    assert fit.map.predicted_moments.shape == (5,)
    assert np.all(np.isfinite(fit.map.predicted_moments))
    assert fit.hmc.samples_theta.shape == (10, 3)


def test_exact_shape5_map_objective_prefers_truth_over_offset():
    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=3.0,
        step_kpc=0.20,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        cloud_alpha=2.0,
        observable_set="logm0_mean_sigma_skew_kurt",
    )

    theta_true = np.array([0.20, 0.20, 0.80], dtype=float)
    theta_offset = np.array([0.28, 0.15, 0.67], dtype=float)
    obs_true = model.predict_observables(theta_true)

    sigma = np.array(
        [
            np.log1p(0.10),
            0.10 * abs(obs_true[1]),
            0.10 * abs(obs_true[2]),
            0.20,
            0.40,
        ],
        dtype=float,
    )
    covariance = build_covariance(sigma, np.eye(5))
    nlp = model.make_negative_log_posterior(
        observed_moments=obs_true,
        covariance_moments=covariance,
        prior_mean_log=np.log(np.array([0.20, 0.20, 0.70], dtype=float)),
        prior_sigma_log=(1.4, 1.4, 0.45),
        include_transform_jacobian=False,
    )

    truth_u = model._unconstrained_from_theta_numpy(theta_true)
    offset_u = model._unconstrained_from_theta_numpy(theta_offset)

    assert float(nlp(truth_u)) < float(nlp(offset_u))


def test_eta_e_softcap_transform_and_penalty():
    model_bounded = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
    )
    bounded_u = model_bounded._unconstrained_from_theta_numpy(np.array([0.2, 0.1, 1.2], dtype=float))
    bounded_theta = model_bounded._theta_from_unconstrained_numpy(bounded_u)
    assert bounded_theta[2] < 1.0

    model_softcap = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        eta_e_parameterization="softcap",
    )
    softcap_u = model_softcap._unconstrained_from_theta_numpy(np.array([0.2, 0.1, 1.2], dtype=float))
    softcap_theta = model_softcap._theta_from_unconstrained_numpy(softcap_u)
    assert np.isclose(softcap_theta[2], 1.2)
    assert model_softcap._theta_from_unconstrained_numpy(np.array([0.0, 0.0, 2.0], dtype=float))[2] > 1.0

    penalties = model_softcap.eta_e_softcap_penalty(np.array([0.90, 1.00, 1.10, 1.20], dtype=float))
    assert penalties[0] < 1e-8
    assert penalties[3] > penalties[2] > penalties[1] >= 0.0


def test_ratio_energy_coordinate_transform_roundtrip_and_validation():
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        eta_e_parameterization="softcap",
        energy_coordinate="eta_e_over_eta_m",
    )

    theta = np.array([0.10, 0.35, 0.98], dtype=float)
    u = model._unconstrained_from_theta_numpy(theta)
    theta_roundtrip = model._theta_from_unconstrained_numpy(u)
    energy_coords = model.energy_coordinates_from_theta_numpy(theta_roundtrip)

    assert np.allclose(theta_roundtrip, theta)
    assert np.isclose(energy_coords[2], theta[2] / theta[0])

    theta_jax, log_theta_jax, jac_log_u_jax = model._theta_log_and_jac_log_u_jax(u)
    assert np.allclose(np.asarray(theta_jax), theta)
    assert np.allclose(np.asarray(log_theta_jax), np.log(theta))
    assert np.linalg.det(np.asarray(jac_log_u_jax)) > 0.0

    with pytest.raises(ValueError, match="requires eta_e_parameterization='softcap'"):
        MomentInferenceModel(
            sfr=2.0,
            r_star_kpc=0.12,
            v_circ=100.0,
            r_max_kpc=2.0,
            step_kpc=0.2,
            n_cloud_species=2,
            cloud_mass_range=(10.0, 1e3),
            energy_coordinate="eta_e_over_eta_m",
        )


def test_loading_ratio_coordinate_transform_roundtrip_and_validation():
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        eta_e_parameterization="softcap",
        energy_coordinate="loading_ratios",
    )

    theta = np.array([0.10, 0.35, 0.98], dtype=float)
    u = model._unconstrained_from_theta_numpy(theta)
    theta_roundtrip = model._theta_from_unconstrained_numpy(u)
    energy_coords = model.energy_coordinates_from_theta_numpy(theta_roundtrip)

    assert np.allclose(theta_roundtrip, theta)
    assert np.allclose(energy_coords, np.array([0.10, 3.50, 9.80]))

    theta_jax, log_theta_jax, jac_log_u_jax = model._theta_log_and_jac_log_u_jax(u)
    assert np.allclose(np.asarray(theta_jax), theta)
    assert np.allclose(np.asarray(log_theta_jax), np.log(theta))
    assert np.linalg.det(np.asarray(jac_log_u_jax)) > 0.0

    with pytest.raises(ValueError, match="requires eta_e_parameterization='softcap'"):
        MomentInferenceModel(
            sfr=2.0,
            r_star_kpc=0.12,
            v_circ=100.0,
            r_max_kpc=2.0,
            step_kpc=0.2,
            n_cloud_species=2,
            cloud_mass_range=(10.0, 1e3),
            energy_coordinate="loading_ratios",
        )


@pytest.mark.parametrize("energy_coordinate", ["eta_e_over_eta_m", "loading_ratios"])
def test_ratio_energy_coordinate_objective_is_finite(energy_coordinate):
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        observable_set="logm0_mean_sigma_skew_kurt",
        eta_e_parameterization="softcap",
        energy_coordinate=energy_coordinate,
    )

    theta_true = np.array([0.10, 0.25, 0.98], dtype=float)
    obs_true = model.predict_observables(theta_true)
    sigma = np.array([np.log1p(0.15), 0.15 * obs_true[1], 0.15 * obs_true[2], 0.25, 0.50], dtype=float)
    covariance = build_covariance(sigma, np.eye(5))
    nlp = model.make_negative_log_posterior(
        observed_moments=obs_true,
        covariance_moments=covariance,
        prior_mean_log=np.log(np.array([0.20, 0.20, 0.90], dtype=float)),
        prior_sigma_log=(1.4, 1.4, 1.2),
        include_transform_jacobian=True,
    )

    assert np.isfinite(float(nlp(model._unconstrained_from_theta_numpy(theta_true))))


@pytest.mark.parametrize("energy_coordinate", ["eta_e_over_eta_m", "loading_ratios"])
def test_ratio_energy_coordinate_hmc_smoke(energy_coordinate):
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        eta_e_parameterization="softcap",
        energy_coordinate=energy_coordinate,
    )

    theta_true = np.array([0.12, 0.10, 1.05], dtype=float)
    moments_true = model.predict_moments(theta_true)
    covariance = build_covariance(0.15 * moments_true, np.eye(3))

    fit = model.fit_posterior(
        observed_moments=moments_true,
        covariance_moments=covariance,
        initial_theta=theta_true,
        prior_mean_log=np.log(np.array([0.12, 0.10, 0.95], dtype=float)),
        prior_sigma_log=(1.4, 1.4, 1.2),
        map_max_iter=4,
        map_num_starts=1,
        hmc_num_warmup=4,
        hmc_num_samples=6,
        hmc_step_size=0.01,
        hmc_leapfrog_steps=4,
        sampler="hmc",
        seed=19,
    )

    assert np.all(np.isfinite(fit.map.theta_map))
    assert np.all(np.isfinite(fit.hmc.samples_theta))
    coordinate_samples = model.energy_coordinates_from_theta_numpy(fit.hmc.samples_theta)
    assert np.all(np.isfinite(coordinate_samples))
    assert np.all(coordinate_samples[:, 2] > 0.0)
    if energy_coordinate == "loading_ratios":
        assert np.all(coordinate_samples[:, 1] > 0.0)


def test_a_mix_and_beta_chi_forward_defaults_preserve_baseline():
    common = dict(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        observable_set="logm0_mean_sigma_skew_kurt",
    )
    theta3 = np.array([0.20, 0.15, 0.85], dtype=float)
    baseline = MomentInferenceModel(**common)
    expanded = MomentInferenceModel(**common, expanded_parameters="a_mix_beta_chi")

    obs3 = baseline.predict_observables(theta3)
    obs5 = expanded.predict_observables(np.array([0.20, 0.15, 0.85, 1.0, 0.0], dtype=float))
    assert np.allclose(obs5, obs3, rtol=1e-10, atol=1e-40)

    changed = expanded.predict_observables(np.array([0.20, 0.15, 0.85, 1.4, 0.25], dtype=float))
    assert np.all(np.isfinite(changed))
    assert not np.allclose(changed, obs3, rtol=1e-4, atol=1e-12)

    config_model = MomentInferenceModel(**common, config=WindConfig(A_mix=1.4, beta_chi_mix=0.25))
    config_changed = config_model.predict_observables(theta3)
    assert np.all(np.isfinite(config_changed))
    assert not np.allclose(config_changed, obs3, rtol=1e-4, atol=1e-12)


def test_expanded_parameter_transform_roundtrip_prior_and_validation():
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        expanded_parameters="a_mix_beta_chi",
        beta_chi_max_abs=0.75,
    )
    theta = np.array([0.20, 0.12, 0.82, 1.35, -0.20], dtype=float)
    u = model._unconstrained_from_theta_numpy(theta)
    roundtrip = model._theta_from_unconstrained_numpy(u)
    prior_coordinate = model._prior_coordinate_from_theta_numpy(roundtrip)

    assert model.parameter_names() == ("eta_M", "eta_M_cold", "eta_E", "A_mix", "beta_chi_mix")
    assert np.allclose(roundtrip, theta)
    assert np.allclose(prior_coordinate[:4], np.log(theta[:4]))
    assert np.isclose(prior_coordinate[4], theta[4])

    theta_jax, coord_jax, jac_jax = model._theta_log_and_jac_log_u_jax(u)
    assert np.allclose(np.asarray(theta_jax), theta)
    assert np.allclose(np.asarray(coord_jax), prior_coordinate)
    assert np.linalg.det(np.asarray(jac_jax)) > 0.0

    with pytest.raises(ValueError, match="expanded_parameters"):
        MomentInferenceModel(sfr=2.0, r_star_kpc=0.12, expanded_parameters="all_microphysics")


def test_a_mix_expanded_map_and_hmc_smoke():
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        expanded_parameters="a_mix",
    )

    theta_true = np.array([0.20, 0.12, 0.85, 1.15], dtype=float)
    moments_true = model.predict_moments(theta_true)
    covariance = build_covariance(0.18 * moments_true, np.eye(3))
    prior_mean = model._prior_coordinate_from_theta_numpy(np.array([0.20, 0.12, 0.80, 1.0], dtype=float))

    fit = model.fit_posterior(
        observed_moments=moments_true,
        covariance_moments=covariance,
        initial_theta=theta_true,
        prior_mean_log=prior_mean,
        prior_sigma_log=(1.4, 1.4, 0.8, 0.35),
        map_max_iter=4,
        map_num_starts=1,
        hmc_num_warmup=4,
        hmc_num_samples=6,
        hmc_step_size=0.01,
        hmc_leapfrog_steps=4,
        sampler="hmc",
        seed=29,
    )

    assert fit.map.theta_map.shape == (4,)
    assert fit.hmc.samples_theta.shape == (6, 4)
    assert fit.hmc.samples_log.shape == (6, 4)
    assert np.all(np.isfinite(fit.hmc.samples_theta))
    assert np.all(fit.hmc.samples_theta[:, 3] > 0.0)


def test_map_whitened_nuts_smoke_components_and_npz_diagnostics(tmp_path):
    pytest.importorskip("numpyro")

    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        eta_e_parameterization="softcap",
        energy_coordinate="eta_e_over_eta_m",
    )

    theta_true = np.array([0.12, 0.10, 1.02], dtype=float)
    moments_true = model.predict_moments(theta_true)
    covariance = build_covariance(0.18 * moments_true, np.eye(3))
    prior_mean_log = np.log(np.array([0.12, 0.10, 0.95], dtype=float))
    prior_sigma_log = (1.4, 1.4, 1.2)

    fit = model.fit_posterior(
        observed_moments=moments_true,
        covariance_moments=covariance,
        initial_theta=theta_true,
        prior_mean_log=prior_mean_log,
        prior_sigma_log=prior_sigma_log,
        map_max_iter=4,
        map_num_starts=1,
        hmc_num_warmup=4,
        hmc_num_samples=6,
        hmc_step_size=0.01,
        hmc_target_accept=0.8,
        sampler="nuts",
        nuts_coordinate="map_whitened",
        nuts_dense_mass=True,
        nuts_max_tree_depth=5,
        seed=23,
    )

    assert fit.hmc.sampler == "nuts"
    assert fit.hmc.nuts_coordinate == "map_whitened"
    assert fit.hmc.samples_theta.shape == (6, 3)
    assert fit.hmc.samples_unconstrained.shape == (6, 3)
    assert fit.hmc.samples_nuts_coordinate is not None
    assert fit.hmc.samples_nuts_coordinate.shape == (6, 3)
    assert fit.hmc.sample_diverging is not None
    assert fit.hmc.sample_diverging.shape == (6,)
    assert fit.hmc.sample_accept_prob is not None
    assert fit.hmc.sample_accept_prob.shape == (6,)
    assert np.all(np.isfinite(fit.hmc.samples_theta))

    component_fn = model.make_negative_log_posterior_components(
        observed_moments=moments_true,
        covariance_moments=covariance,
        prior_mean_log=prior_mean_log,
        prior_sigma_log=prior_sigma_log,
        include_transform_jacobian=True,
    )
    truth_components = np.asarray(component_fn(model._unconstrained_from_theta_numpy(theta_true)), dtype=float)
    map_components = np.asarray(component_fn(fit.map.unconstrained_theta_map), dtype=float)
    sample_components = np.asarray(component_fn(fit.hmc.samples_unconstrained[0]), dtype=float)
    assert truth_components.shape == (len(model.POSTERIOR_COMPONENT_NAMES),)
    assert np.all(np.isfinite(truth_components))
    assert np.all(np.isfinite(map_components))
    assert np.all(np.isfinite(sample_components))

    recovery_path = Path(__file__).resolve().parents[1] / "examples" / "inference_synthetic_recovery.py"
    spec = importlib.util.spec_from_file_location("local_inference_synthetic_recovery", recovery_path)
    assert spec is not None and spec.loader is not None
    recovery = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = recovery
    spec.loader.exec_module(recovery)

    posterior_components = recovery.evaluate_posterior_components(
        model=model,
        observed=moments_true,
        covariance=covariance,
        prior_mean_log=prior_mean_log,
        prior_sigma_log=prior_sigma_log,
        theta_true=theta_true,
        unconstrained_theta_map=fit.map.unconstrained_theta_map,
        samples_unconstrained=fit.hmc.samples_unconstrained,
    )
    diagnostics = {
        "sampler": fit.hmc.sampler,
        "nuts_coordinate": fit.hmc.nuts_coordinate,
        "acceptance_rate": fit.hmc.acceptance_rate,
        "num_divergent": fit.hmc.num_divergent,
        "max_tree_depth_hits": 0,
        "max_tree_depth_fraction": 0.0,
        "r_hat": fit.hmc.r_hat,
        "ess_bulk": fit.hmc.ess_bulk,
        "correlation_theta": fit.hmc.correlation_theta,
        "posterior_prob_eta_E_gt_1": float(np.mean(fit.hmc.samples_theta[:, 2] > 1.0)),
        "samples_unconstrained": fit.hmc.samples_unconstrained,
        "samples_nuts_coordinate": fit.hmc.samples_nuts_coordinate,
        "sample_diverging": fit.hmc.sample_diverging,
        "sample_accept_prob": fit.hmc.sample_accept_prob,
        "sample_num_steps": fit.hmc.sample_num_steps,
        "sample_energy": fit.hmc.sample_energy,
        "sample_potential_energy": fit.hmc.sample_potential_energy,
        **posterior_components,
    }
    metrics = recovery.summarize_recovery_metrics(theta_true, fit.map.theta_map, fit.hmc.samples_theta)
    result = recovery.RealizationResult(
        truth_case="unit",
        realization_id=0,
        theta_true=theta_true,
        true_observables=moments_true,
        true_raw_moments=model.predict_raw_moments(theta_true),
        observed=moments_true,
        sigma=np.sqrt(np.diag(covariance)),
        covariance=covariance,
        truth_valid=True,
        first_invalid_r_kpc=np.nan,
        success=True,
        error="",
        map_success=fit.map.success,
        map_message=fit.map.message,
        theta_map=fit.map.theta_map,
        map_predicted_observables=fit.map.predicted_moments,
        chi2=fit.map.chi2,
        nlp=fit.map.nlp,
        samples_theta=fit.hmc.samples_theta,
        samples_log=fit.hmc.samples_log,
        posterior_predictive=np.empty((0, model.observable_dim), dtype=float),
        metrics=metrics,
        diagnostics=diagnostics,
        runtime_seconds={"total": 0.0},
    )
    npz_path = recovery.write_npz(str(tmp_path), model, [result])
    with np.load(npz_path, allow_pickle=True) as data:
        for key in (
            "samples_unconstrained",
            "samples_nuts_coordinate",
            "sample_diverging",
            "sample_accept_prob",
            "sample_num_steps",
            "sample_energy",
            "sample_potential_energy",
            "posterior_component_names",
            "posterior_components_truth",
            "posterior_components_map",
            "posterior_components_samples",
            "log_dndv_low_signal_policy",
            "log_dndv_censored_mask",
        ):
            assert key in data.files
        assert data["sample_diverging"].shape == (1, 6)
        assert data["posterior_components_samples"].shape[2] == len(model.POSTERIOR_COMPONENT_NAMES)
        for key in (
            "failure_policy",
            "truth_trajectory_status_code",
            "map_trajectory_status_code",
            "sample_trajectory_status_code",
            "map_soft_reach_radius_kpc",
            "map_stall_penalty",
            "map_numerical_failure_penalty",
        ):
            assert key in data.files


def test_stalled_wind_policy_makes_known_bad_leaf_finite():
    common_kwargs = dict(
        sfr=20.0,
        r_star_kpc=0.3,
        v_circ=150.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=4,
        cloud_mass_range=(10.0, 1.0e4),
        observable_set="dndv_binned",
        dndv_num_bins=32,
        dndv_vmin_kms=0.0,
        dndv_vmax_kms=1600.0,
        eta_e_parameterization="softcap",
        energy_coordinate="eta_e_over_eta_m",
    )
    model = MomentInferenceModel(**common_kwargs)
    legacy_model = MomentInferenceModel(**common_kwargs, failure_policy="hard_invalid")

    theta_truth = np.array([0.1, 0.35, 0.98], dtype=float)
    theta_bad_leaf = np.array([0.09144991046377486, 0.3797144686587329, 1.0082027618085596], dtype=float)
    theta_previous_leaf = np.array([0.09145198540982885, 0.3791091179062511, 1.0092752170880355], dtype=float)

    prediction = model._predict_theta_with_valid_fn(theta_bad_leaf)
    observables, raw, valid, _barrier, first_invalid_r = prediction[:5]
    trajectory_status_code = float(np.asarray(prediction[5]))
    first_invalid_r_kpc = float(np.asarray(first_invalid_r) / kpc)
    soft_reach_radius_kpc = float(np.asarray(prediction[6]) / kpc)
    min_hot_velocity_kms = float(np.asarray(prediction[7]))
    stall_penalty = float(np.asarray(prediction[8]))

    assert float(np.asarray(valid)) == 0.0
    assert trajectory_status_code == 1.0
    assert 0.70 < first_invalid_r_kpc < 0.80
    assert 0.60 < soft_reach_radius_kpc < 0.80
    assert min_hot_velocity_kms < 0.0
    assert np.isfinite(stall_penalty)
    assert stall_penalty > 0.0
    assert np.all(np.isfinite(np.asarray(observables)))
    assert np.all(np.isfinite(np.asarray(raw)))

    truth_obs = model.predict_observables(theta_truth)
    legacy_truth_obs = legacy_model.predict_observables(theta_truth)
    assert np.allclose(truth_obs, legacy_truth_obs, rtol=1e-10, atol=1e-40)

    previous_prediction = model._predict_theta_with_valid_fn(theta_previous_leaf)
    assert float(np.asarray(previous_prediction[2])) == 1.0
    assert float(np.asarray(previous_prediction[5])) == 0.0
    assert float(np.asarray(previous_prediction[8])) == 0.0

    covariance = build_covariance(np.maximum(0.1 * truth_obs, 1e-30), np.eye(truth_obs.size))
    components = model.make_negative_log_posterior_components(
        observed_moments=truth_obs,
        covariance_moments=covariance,
        prior_mean_log=np.log(np.array([0.2, 0.2, 0.9], dtype=float)),
        prior_sigma_log=(1.4, 1.4, 1.2),
    )(model._unconstrained_from_theta_numpy(theta_bad_leaf))
    component = dict(zip(model.POSTERIOR_COMPONENT_NAMES, np.asarray(components, dtype=float)))
    assert np.isfinite(component["total_objective"])
    assert component["trajectory_status_code"] == 1.0
    assert component["hard_invalid_penalty"] == 0.0
    assert component["numerical_failure_penalty"] == 0.0
    assert component["stall_penalty"] > 0.0

    legacy_components = legacy_model.make_negative_log_posterior_components(
        observed_moments=legacy_truth_obs,
        covariance_moments=covariance,
        prior_mean_log=np.log(np.array([0.2, 0.2, 0.9], dtype=float)),
        prior_sigma_log=(1.4, 1.4, 1.2),
    )(legacy_model._unconstrained_from_theta_numpy(theta_bad_leaf))
    legacy_component = dict(zip(legacy_model.POSTERIOR_COMPONENT_NAMES, np.asarray(legacy_components, dtype=float)))
    assert legacy_component["trajectory_status_code"] == 1.0
    assert legacy_component["hard_invalid_penalty"] == 1.0e6


def test_softcap_posterior_smoke_and_eta_e_tail_probability():
    model = MomentInferenceModel(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        eta_e_parameterization="softcap",
    )

    theta_true = np.array([0.20, 0.10, 1.05], dtype=float)
    moments_true = model.predict_moments(theta_true)
    covariance = build_covariance(0.15 * moments_true, np.eye(3))

    fit = model.fit_posterior(
        observed_moments=moments_true,
        covariance_moments=covariance,
        initial_theta=(0.2, 0.1, 1.05),
        prior_mean_log=np.log(np.array([0.2, 0.1, 0.95], dtype=float)),
        prior_sigma_log=(1.4, 1.4, 1.2),
        map_max_iter=4,
        map_num_starts=1,
        hmc_num_warmup=4,
        hmc_num_samples=6,
        hmc_step_size=0.01,
        hmc_leapfrog_steps=4,
        sampler="hmc",
        seed=17,
    )

    assert np.all(np.isfinite(fit.map.theta_map))
    assert np.all(np.isfinite(fit.hmc.samples_theta))
    tail_probability = float(np.mean(fit.hmc.samples_theta[:, 2] > 1.0))
    assert 0.0 <= tail_probability <= 1.0


def test_dndv_binned_observable_mode_predicts_and_fits():
    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=3,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
        observable_set="dndv_binned",
        dndv_num_bins=12,
        dndv_vmin_kms=0.0,
        dndv_vmax_kms=900.0,
        dndv_kernel_sigma_kms=35.0,
        dndv_kernel="truncated_gaussian",
        dndv_kernel_truncate_sigma=3.0,
    )

    theta_true = np.array([0.20, 0.12, 0.90], dtype=float)
    obs_true = model.predict_observables(theta_true)
    v_bins = model.get_dndv_velocity_bins()

    assert obs_true.shape == (12,)
    assert v_bins.shape == (12,)
    assert np.all(np.isfinite(obs_true))
    assert np.all(obs_true > 0.0)

    idx = np.arange(obs_true.size)
    sigma = np.maximum(0.12 * obs_true, 0.03 * np.max(obs_true))
    corr = 0.6 ** np.abs(idx[:, None] - idx[None, :])
    covariance = build_covariance(sigma, corr)

    observed = obs_true * (1.0 + 0.03 * np.cos(0.4 * idx))
    observed = np.maximum(observed, 1e-40)

    fit = model.fit_posterior(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.2, 0.2, 0.8),
        map_max_iter=8,
        map_num_starts=2,
        hmc_num_warmup=8,
        hmc_num_samples=10,
        sampler="hmc",
        seed=13,
    )

    assert fit.map.predicted_moments.shape == (12,)
    assert np.all(np.isfinite(fit.map.predicted_moments))
    assert fit.hmc.samples_theta.shape == (10, 3)


def test_support_aware_dndv_kernels_suppress_far_tails():
    r_grid = np.array([0.0, 1.0], dtype=float)
    v_kms = np.array([[100.0], [100.0]], dtype=float)
    n_h_eff = np.ones((2, 1), dtype=float)
    velocity_bins = np.array([100.0, 250.0], dtype=float)
    sigma = 25.0

    gaussian = np.asarray(
        MomentInferenceModel._observables_dndv_binned(
            v_kms,
            n_h_eff,
            r_grid,
            velocity_bins,
            sigma,
            kernel_kind=MomentInferenceModel._DNDV_KERNEL_GAUSSIAN,
        )
    )
    truncated = np.asarray(
        MomentInferenceModel._observables_dndv_binned(
            v_kms,
            n_h_eff,
            r_grid,
            velocity_bins,
            sigma,
            kernel_kind=MomentInferenceModel._DNDV_KERNEL_TRUNCATED_GAUSSIAN,
            truncate_sigma=3.0,
        )
    )
    compact = np.asarray(
        MomentInferenceModel._observables_dndv_binned(
            v_kms,
            n_h_eff,
            r_grid,
            velocity_bins,
            sigma,
            kernel_kind=MomentInferenceModel._DNDV_KERNEL_COMPACT_COSINE,
        )
    )

    assert gaussian[0] > 0.0
    assert gaussian[1] > 0.0
    assert truncated[0] > 0.0
    assert truncated[1] == 0.0
    assert compact[0] > 0.0
    assert compact[1] == 0.0

    with pytest.raises(ValueError, match="Unsupported dndv_kernel"):
        MomentInferenceModel(
            sfr=3.0,
            r_star_kpc=0.20,
            observable_set="dndv_binned",
            dndv_kernel="not_a_kernel",
        )


def test_log_dndv_binned_observable_mode_is_finite_and_map_fits():
    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=3,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
        observable_set="log_dndv_binned",
        dndv_num_bins=12,
        dndv_vmin_kms=0.0,
        dndv_vmax_kms=900.0,
        dndv_kernel_sigma_kms=35.0,
        dndv_kernel="truncated_gaussian",
        dndv_kernel_truncate_sigma=3.0,
    )
    linear_model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=3,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
        observable_set="dndv_binned",
        dndv_num_bins=12,
        dndv_vmin_kms=0.0,
        dndv_vmax_kms=900.0,
        dndv_kernel_sigma_kms=35.0,
        dndv_kernel="truncated_gaussian",
        dndv_kernel_truncate_sigma=3.0,
    )

    theta_true = np.array([0.20, 0.12, 0.90], dtype=float)
    obs_true = model.predict_observables(theta_true)
    linear_obs_true = linear_model.predict_observables(theta_true)

    assert obs_true.shape == (12,)
    assert np.all(np.isfinite(obs_true))
    assert np.allclose(np.exp(obs_true), linear_obs_true, rtol=1e-10, atol=1e-40)
    assert model.get_dndv_velocity_bins().shape == (12,)

    idx = np.arange(obs_true.size)
    sigma = np.full(obs_true.size, np.log1p(0.12), dtype=float)
    corr = 0.6 ** np.abs(idx[:, None] - idx[None, :])
    covariance = build_covariance(sigma, corr)
    observed = obs_true + 0.03 * np.cos(0.4 * idx)

    fit = model.fit_map(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.2, 0.2, 0.8),
        max_iter=8,
        map_num_starts=2,
        seed=13,
    )
    assert fit.predicted_moments.shape == (12,)
    assert np.all(np.isfinite(fit.predicted_moments))
    assert np.all(fit.theta_map > 0.0)


def test_log_dndv_censored_upper_likelihood_masks_low_signal_bins():
    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=6.0,
        step_kpc=0.08,
        n_cloud_species=3,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
        observable_set="log_dndv_binned",
        dndv_num_bins=12,
        dndv_vmin_kms=0.0,
        dndv_vmax_kms=900.0,
        dndv_kernel_sigma_kms=35.0,
        log_dndv_low_signal_policy="censored_upper",
        log_dndv_censor_delta_log=12.0,
        log_dndv_censor_transition=0.10,
        log_dndv_censor_sigma=0.50,
        log_dndv_censor_upper_margin=1.0,
    )
    theta_true = np.array([0.20, 0.12, 0.90], dtype=float)
    obs_true = model.predict_observables(theta_true)
    mask = model.log_dndv_censored_mask(obs_true)
    assert mask.shape == obs_true.shape
    assert np.any(mask)
    assert not np.all(mask)

    idx = np.arange(obs_true.size)
    sigma = np.full(obs_true.size, np.log1p(0.12), dtype=float)
    covariance = build_covariance(sigma, 0.4 ** np.abs(idx[:, None] - idx[None, :]))
    component_fn = model.make_negative_log_posterior_components(
        observed_moments=obs_true,
        covariance_moments=covariance,
        include_transform_jacobian=False,
    )
    truth_components = dict(
        zip(
            model.POSTERIOR_COMPONENT_NAMES,
            np.asarray(component_fn(model._unconstrained_from_theta_numpy(theta_true)), dtype=float),
        )
    )
    assert truth_components["censored_chi2"] < 1.0e-2
    assert truth_components["gaussian_chi2"] < 1.0e-8

    stricter_upper_limits = obs_true.copy()
    stricter_upper_limits[mask] -= 2.0
    component_fn = model.make_negative_log_posterior_components(
        observed_moments=stricter_upper_limits,
        covariance_moments=covariance,
        include_transform_jacobian=False,
    )
    strict_components = dict(
        zip(
            model.POSTERIOR_COMPONENT_NAMES,
            np.asarray(component_fn(model._unconstrained_from_theta_numpy(theta_true)), dtype=float),
        )
    )
    assert strict_components["censored_chi2"] > truth_components["censored_chi2"] + 1.0
    assert strict_components["chi2"] >= strict_components["censored_chi2"]

    gaussian_model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        observable_set="log_dndv_binned",
        log_dndv_low_signal_policy="gaussian",
    )
    assert not np.any(gaussian_model.log_dndv_censored_mask(np.zeros(gaussian_model.observable_dim)))
    with pytest.raises(ValueError, match="Unsupported log_dndv_low_signal_policy"):
        MomentInferenceModel(
            sfr=3.0,
            r_star_kpc=0.20,
            observable_set="log_dndv_binned",
            log_dndv_low_signal_policy="not_a_policy",
        )


def test_map_and_nuts_smoke_on_synthetic_moments():
    pytest.importorskip("numpyro")

    model = MomentInferenceModel(
        sfr=3.0,
        r_star_kpc=0.20,
        v_circ=120.0,
        r_max_kpc=8.0,
        step_kpc=0.06,
        n_cloud_species=4,
        cloud_mass_range=(10.0, 1e4),
        cloud_alpha=2.0,
    )

    theta_true = np.array([0.22, 0.18, 0.95], dtype=float)
    moments_true = model.predict_moments(theta_true)

    sigma = 0.12 * moments_true
    covariance = build_covariance(sigma, np.eye(3))
    observed = moments_true * np.array([1.02, 0.99, 1.03], dtype=float)

    fit = model.fit_posterior(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.20, 0.20, 1.00),
        map_max_iter=8,
        map_num_starts=2,
        hmc_num_warmup=10,
        hmc_num_samples=12,
        hmc_step_size=0.02,
        hmc_target_accept=0.7,
        sampler="nuts",
        num_chains=2,
        nuts_dense_mass=True,
        nuts_max_tree_depth=8,
        seed=11,
    )

    assert fit.hmc.sampler == "nuts"
    assert fit.hmc.nuts_coordinate == "native"
    assert fit.hmc.num_chains == 2
    assert fit.hmc.samples_log.shape == (24, 3)
    assert fit.hmc.samples_theta.shape == (24, 3)
    assert fit.hmc.samples_nuts_coordinate is not None
    assert fit.hmc.samples_nuts_coordinate.shape == fit.hmc.samples_unconstrained.shape
    assert np.allclose(fit.hmc.samples_nuts_coordinate, fit.hmc.samples_unconstrained)
    assert fit.hmc.sample_diverging is not None
    assert fit.hmc.sample_diverging.shape == (24,)
    assert fit.hmc.sample_accept_prob is not None
    assert fit.hmc.sample_accept_prob.shape == (24,)
    assert fit.hmc.acceptance_rate_per_chain is not None
    assert fit.hmc.acceptance_rate_per_chain.shape == (2,)
    assert fit.hmc.nuts_chain_method in {"parallel", "vectorized", "sequential"}
    assert fit.hmc.nuts_dense_mass is True
    assert fit.hmc.nuts_max_tree_depth == 8
    assert fit.hmc.num_divergent_per_chain is not None
    assert fit.hmc.num_divergent_per_chain.shape == (2,)
    assert fit.hmc.bfmi_per_chain is not None
    assert fit.hmc.bfmi_per_chain.shape == (2,)
    assert fit.hmc.r_hat is not None
    assert fit.hmc.ess_bulk is not None
    assert np.all(np.isfinite(fit.hmc.r_hat))
    assert np.all(np.isfinite(fit.hmc.ess_bulk))
    assert fit.hmc.num_divergent >= 0
    assert np.all(fit.hmc.samples_theta[:, 2] < 1.0)


def test_resolve_nuts_chain_method_auto_and_validation():
    assert resolve_nuts_chain_method(1, "auto") == "sequential"
    assert resolve_nuts_chain_method(2, "sequential") == "sequential"
    assert resolve_nuts_chain_method(2, "vectorized") == "vectorized"
    assert resolve_nuts_chain_method(2, "parallel") == "parallel"

    auto = resolve_nuts_chain_method(2, "auto")
    assert auto in {"parallel", "vectorized"}

    with pytest.raises(ValueError):
        resolve_nuts_chain_method(2, "bad-mode")


def test_tsit5_mode_requires_diffrax_or_matches_rk4():
    common_kwargs = dict(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        cloud_alpha=2.0,
    )
    theta = np.array([0.2, 0.1, 0.8], dtype=float)

    model_rk4 = MomentInferenceModel(**common_kwargs, integrator_mode="rk4")
    moments_rk4 = model_rk4.predict_moments(theta)
    assert np.all(np.isfinite(moments_rk4))

    if not has_diffrax():
        with pytest.raises(ModuleNotFoundError):
            MomentInferenceModel(**common_kwargs, integrator_mode="tsit5")
        return

    model_tsit5 = MomentInferenceModel(**common_kwargs, integrator_mode="tsit5")
    moments_tsit5 = model_tsit5.predict_moments(theta)
    assert np.all(np.isfinite(moments_tsit5))
    rel = np.abs(moments_tsit5 - moments_rk4) / np.maximum(np.abs(moments_rk4), 1e-30)
    assert np.all(rel < 0.15)

    covariance = build_covariance(0.15 * moments_tsit5, np.eye(3))
    observed = moments_tsit5 * np.array([1.01, 0.99, 1.02], dtype=float)
    fit = model_tsit5.fit_map(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(0.2, 0.1, 0.8),
        max_iter=3,
        map_num_starts=1,
        seed=3,
    )
    assert fit.theta_map.shape == (3,)
    assert np.all(np.isfinite(fit.theta_map))


def test_rk2_rk3_modes_are_finite_and_track_rk4():
    common_kwargs = dict(
        sfr=2.0,
        r_star_kpc=0.12,
        v_circ=100.0,
        r_max_kpc=2.0,
        step_kpc=0.2,
        n_cloud_species=2,
        cloud_mass_range=(10.0, 1e3),
        cloud_alpha=2.0,
    )
    theta = np.array([0.2, 0.1, 0.8], dtype=float)

    moments_rk4 = MomentInferenceModel(**common_kwargs, integrator_mode="rk4").predict_moments(theta)
    moments_rk3 = MomentInferenceModel(**common_kwargs, integrator_mode="rk3").predict_moments(theta)
    moments_rk2 = MomentInferenceModel(**common_kwargs, integrator_mode="rk2").predict_moments(theta)

    assert np.all(np.isfinite(moments_rk4))
    assert np.all(np.isfinite(moments_rk3))
    assert np.all(np.isfinite(moments_rk2))

    rel_rk3 = np.abs(moments_rk3 - moments_rk4) / np.maximum(np.abs(moments_rk4), 1e-30)
    rel_rk2 = np.abs(moments_rk2 - moments_rk4) / np.maximum(np.abs(moments_rk4), 1e-30)

    assert np.all(rel_rk3 < 0.10)
    assert np.all(rel_rk2 < 0.35)
    assert float(np.mean(rel_rk3)) <= float(np.mean(rel_rk2))
