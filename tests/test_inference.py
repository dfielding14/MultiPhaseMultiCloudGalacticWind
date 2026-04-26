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


def test_ratio_energy_coordinate_objective_is_finite():
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
        energy_coordinate="eta_e_over_eta_m",
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


def test_ratio_energy_coordinate_hmc_smoke():
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
    ratio_samples = model.energy_coordinates_from_theta_numpy(fit.hmc.samples_theta)[:, 2]
    assert np.all(np.isfinite(ratio_samples))
    assert np.all(ratio_samples > 0.0)


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
        ):
            assert key in data.files
        assert data["sample_diverging"].shape == (1, 6)
        assert data["posterior_components_samples"].shape[2] == len(model.POSTERIOR_COMPONENT_NAMES)


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
