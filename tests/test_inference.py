"""Tests for inference utilities and MAP/HMC fitting path."""

import numpy as np
import pytest

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
        seed=11,
    )

    assert fit.hmc.sampler == "nuts"
    assert fit.hmc.num_chains == 2
    assert fit.hmc.samples_log.shape == (24, 3)
    assert fit.hmc.samples_theta.shape == (24, 3)
    assert fit.hmc.acceptance_rate_per_chain is not None
    assert fit.hmc.acceptance_rate_per_chain.shape == (2,)
    assert fit.hmc.nuts_chain_method in {"parallel", "vectorized", "sequential"}
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
