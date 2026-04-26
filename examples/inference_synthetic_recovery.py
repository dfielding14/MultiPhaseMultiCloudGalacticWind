#!/usr/bin/env python3
"""Synthetic input-recovery harness for the baseline three-parameter model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "inference_synthetic_recovery")
PARAM_NAMES = ("eta_M", "eta_M_cold", "eta_E")
OBSERVABLE_SETS = ("m0_m1_m2", "logm0_mean_sigma_skew_kurt", "dndv_binned")
ETA_E_MAX = 0.999


@dataclass(frozen=True)
class TruthCase:
    """Named synthetic truth in the current three-parameter inference surface."""

    name: str
    theta: tuple[float, float, float]
    description: str


TRUTH_CASES: dict[str, TruthCase] = {
    "fiducial": TruthCase(
        "fiducial",
        (0.20, 0.20, 0.80),
        "Moderate hot/cold mass loading with high but not extreme energy loading.",
    ),
    "low_eta_m_high_eta_e": TruthCase(
        "low_eta_m_high_eta_e",
        (0.07, 0.12, 0.96),
        "Fast low-mass hot wind with high specific energy.",
    ),
    "high_eta_m_low_eta_e": TruthCase(
        "high_eta_m_low_eta_e",
        (0.90, 0.20, 0.25),
        "Mass-loaded hot wind with relatively weak hot energy loading.",
    ),
    "low_eta_m_cold": TruthCase(
        "low_eta_m_cold",
        (0.22, 0.01, 0.80),
        "Nearly hot-only baseline with a small cold-cloud mass flux.",
    ),
    "high_eta_m_cold": TruthCase(
        "high_eta_m_cold",
        (0.35, 1.00, 0.85),
        "Cold-mass-rich wind where cool material dominates the column budget.",
    ),
    "near_failure_boundary": TruthCase(
        "near_failure_boundary",
        (0.06, 1.50, 0.40),
        "Cold-loaded, lower-energy case intended to sit near invalid prior regions.",
    ),
    "strong_wings": TruthCase(
        "strong_wings",
        (0.10, 0.35, 0.98),
        "High-energy case designed to test recovery of broad velocity profiles.",
    ),
    "narrow_profile": TruthCase(
        "narrow_profile",
        (0.70, 0.05, 0.45),
        "Higher hot mass loading and low cold loading, producing compact profiles.",
    ),
}


@dataclass
class RealizationResult:
    """One noisy synthetic recovery realization."""

    truth_case: str
    realization_id: int
    theta_true: np.ndarray
    true_observables: np.ndarray
    true_raw_moments: np.ndarray
    observed: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    truth_valid: bool
    first_invalid_r_kpc: float
    success: bool
    error: str
    map_success: bool
    map_message: str
    theta_map: np.ndarray
    map_predicted_observables: np.ndarray
    chi2: float
    nlp: float
    samples_theta: np.ndarray
    samples_log: np.ndarray
    posterior_predictive: np.ndarray
    metrics: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    runtime_seconds: dict[str, float]


def configure_jax_platform(platform: str) -> str:
    """Configure JAX backend environment variables before JAX-heavy imports."""
    requested = platform.strip().lower()
    if requested != "cpu":
        raise ValueError("Only --jax-platform cpu is supported on this machine for this harness.")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    return requested


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    truth_choices = ["all", *TRUTH_CASES.keys()]
    parser.add_argument("--truth-case", choices=truth_choices, default="fiducial")
    parser.add_argument("--observable-set", choices=OBSERVABLE_SETS, default="logm0_mean_sigma_skew_kurt")
    parser.add_argument("--noise-fraction", type=float, default=0.10)
    parser.add_argument(
        "--use-truth-observables",
        action="store_true",
        help="Use exact truth observables instead of drawing a noisy realization.",
    )
    parser.add_argument("--num-noise-realizations", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=200, help="Posterior samples per realization")
    parser.add_argument("--num-warmup", type=int, default=200, help="Posterior warmup steps per realization")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jax-platform", default="cpu", choices=["cpu"])

    parser.add_argument("--sfr", type=float, default=20.0, help="Star formation rate [Msun/yr]")
    parser.add_argument("--v-circ", type=float, default=150.0, help="Circular velocity [km/s]")
    parser.add_argument("--r-star-kpc", type=float, default=0.3, help="Launch/sonic radius [kpc]")
    parser.add_argument("--r-max-kpc", type=float, default=30.0)
    parser.add_argument("--step-kpc", type=float, default=0.02)
    parser.add_argument("--first-step-kpc", type=float, default=1.0e-12)
    parser.add_argument("--n-cloud-species", type=int, default=13)
    parser.add_argument("--cloud-mass-min", type=float, default=1.0)
    parser.add_argument("--cloud-mass-max", type=float, default=1.0e6)
    parser.add_argument("--cloud-alpha", type=float, default=2.0)
    parser.add_argument("--integrator-mode", choices=["rk2", "rk3", "rk4", "tsit5"], default="rk4")
    parser.add_argument("--integrator-rtol", type=float, default=1.0e-5)
    parser.add_argument("--integrator-atol", type=float, default=1.0e-8)
    parser.add_argument("--integrator-max-steps", type=int, default=131072)

    parser.add_argument("--dndv-num-bins", type=int, default=25)
    parser.add_argument("--dndv-vmin-kms", type=float, default=0.0)
    parser.add_argument("--dndv-vmax-kms", type=float, default=1200.0)
    parser.add_argument("--dndv-kernel-sigma-kms", type=float, default=None)
    parser.add_argument("--dndv-sigma-floor-frac", type=float, default=0.03)
    parser.add_argument("--dndv-bin-corr", type=float, default=0.60)

    parser.add_argument("--eta-e-parameterization", choices=["bounded", "softcap"], default="bounded")
    parser.add_argument("--eta-e-softcap-center", type=float, default=1.0)
    parser.add_argument("--eta-e-softcap-sigma", type=float, default=0.10)
    parser.add_argument("--eta-e-softcap-transition", type=float, default=0.01)

    parser.add_argument("--shape-skew-sigma", type=float, default=0.20)
    parser.add_argument("--shape-kurt-sigma", type=float, default=0.40)

    parser.add_argument("--sampler", choices=["hmc", "nuts", "none"], default="hmc")
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--nuts-chain-method", choices=["auto", "sequential", "parallel", "vectorized"], default="auto")
    parser.add_argument("--nuts-dense-mass", action="store_true", help="Use dense mass-matrix adaptation for NUTS.")
    parser.add_argument("--nuts-max-tree-depth", type=int, default=10)
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--map-max-iter", type=int, default=25)
    parser.add_argument("--map-num-starts", type=int, default=4)
    parser.add_argument("--hmc-step-size", type=float, default=0.02)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=12)
    parser.add_argument("--hmc-target-accept", type=float, default=0.70)
    parser.add_argument("--posterior-predictive-samples", type=int, default=256)
    parser.add_argument(
        "--no-diagnostic-plots",
        action="store_true",
        help="Skip per-realization corner and observable-fit diagnostic plots.",
    )

    parser.add_argument("--initial-eta-m", type=float, default=0.20)
    parser.add_argument("--initial-eta-m-cold", type=float, default=0.20)
    parser.add_argument("--initial-eta-e", type=float, default=0.80)
    parser.add_argument("--prior-eta-m", type=float, default=0.20)
    parser.add_argument("--prior-eta-m-cold", type=float, default=0.20)
    parser.add_argument("--prior-eta-e", type=float, default=0.70)
    parser.add_argument("--prior-sigma-log-eta-m", type=float, default=1.40)
    parser.add_argument("--prior-sigma-log-eta-m-cold", type=float, default=1.40)
    parser.add_argument("--prior-sigma-log-eta-e", type=float, default=0.45)

    args = parser.parse_args(argv)
    if args.noise_fraction <= 0.0:
        parser.error("--noise-fraction must be positive")
    if args.num_noise_realizations < 1:
        parser.error("--num-noise-realizations must be >= 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.num_warmup < 0:
        parser.error("--num-warmup must be >= 0")
    if args.nuts_max_tree_depth < 1:
        parser.error("--nuts-max-tree-depth must be >= 1")
    if args.posterior_predictive_samples < 1:
        parser.error("--posterior-predictive-samples must be >= 1")
    if args.eta_e_softcap_center <= 0.0:
        parser.error("--eta-e-softcap-center must be positive")
    if args.eta_e_softcap_sigma <= 0.0:
        parser.error("--eta-e-softcap-sigma must be positive")
    if args.eta_e_softcap_transition <= 0.0:
        parser.error("--eta-e-softcap-transition must be positive")
    return args


def selected_truth_cases(truth_case: str) -> list[TruthCase]:
    """Return truth cases requested by CLI."""
    if truth_case == "all":
        return [TRUTH_CASES[name] for name in TRUTH_CASES]
    return [TRUTH_CASES[truth_case]]


def build_model(args: argparse.Namespace):
    """Construct the baseline three-parameter inference model."""
    from multiphasegalacticwind.inference import MomentInferenceModel

    return MomentInferenceModel(
        sfr=args.sfr,
        r_star_kpc=args.r_star_kpc,
        v_circ=args.v_circ,
        r_max_kpc=args.r_max_kpc,
        step_kpc=args.step_kpc,
        first_step_kpc=args.first_step_kpc,
        n_cloud_species=args.n_cloud_species,
        cloud_mass_range=(args.cloud_mass_min, args.cloud_mass_max),
        cloud_alpha=args.cloud_alpha,
        integrator_mode=args.integrator_mode,
        integrator_rtol=args.integrator_rtol,
        integrator_atol=args.integrator_atol,
        integrator_max_steps=args.integrator_max_steps,
        observable_set=args.observable_set,
        dndv_num_bins=args.dndv_num_bins,
        dndv_vmin_kms=args.dndv_vmin_kms,
        dndv_vmax_kms=args.dndv_vmax_kms,
        dndv_kernel_sigma_kms=args.dndv_kernel_sigma_kms,
        eta_e_parameterization=args.eta_e_parameterization,
        eta_e_softcap_center=args.eta_e_softcap_center,
        eta_e_softcap_sigma=args.eta_e_softcap_sigma,
        eta_e_softcap_transition=args.eta_e_softcap_transition,
    )


def _predict_with_validity(model: Any, theta: Sequence[float]) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Evaluate truth observables and expose the model validity flag."""
    import jax.numpy as jnp

    theta_arr = np.asarray(theta, dtype=float)
    predictor = getattr(model, "_predict_theta_with_valid_fn", None)
    if predictor is None:
        observables = np.asarray(model.predict_observables(theta_arr), dtype=float)
        raw = np.asarray(model.predict_raw_moments(theta_arr), dtype=float)
        valid = bool(np.all(np.isfinite(observables)) and np.all(np.isfinite(raw)) and raw[0] > 0.0)
        return observables, raw, valid, np.nan

    observables_jax, raw_jax, valid_jax, _barrier, first_invalid_r = predictor(
        jnp.asarray(theta_arr, dtype=jnp.float64)
    )
    observables = np.asarray(observables_jax, dtype=float)
    raw = np.asarray(raw_jax, dtype=float)

    from multiphasegalacticwind.constants import kpc

    valid = bool(float(np.asarray(valid_jax)) > 0.5)
    valid = valid and bool(np.all(np.isfinite(observables)) and np.all(np.isfinite(raw)))
    return observables, raw, valid, float(np.asarray(first_invalid_r, dtype=float) / kpc)


def build_synthetic_covariance(
    model: Any,
    true_observables: Sequence[float],
    noise_fraction: float,
    shape_skew_sigma: float = 0.20,
    shape_kurt_sigma: float = 0.40,
    dndv_sigma_floor_frac: float = 0.03,
    dndv_bin_corr: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the synthetic covariance model for the configured observable set."""
    from multiphasegalacticwind.inference import build_covariance

    y = np.asarray(true_observables, dtype=float)
    frac = float(noise_fraction)
    if frac <= 0.0:
        raise ValueError("noise_fraction must be positive")

    if model.observable_set == "m0_m1_m2":
        sigma = np.maximum(frac * np.abs(y), 1.0e-30)
        corr = np.asarray(
            [
                [1.0, 0.55, 0.35],
                [0.55, 1.0, 0.65],
                [0.35, 0.65, 1.0],
            ],
            dtype=float,
        )
        return sigma, build_covariance(sigma, corr)

    if model.observable_set == "dndv_binned":
        amp = max(float(np.nanmax(np.abs(y))), 1.0e-30)
        sigma_floor = max(float(dndv_sigma_floor_frac), 1.0e-6) * amp
        sigma = np.maximum(frac * np.abs(y), sigma_floor)
        rho = float(np.clip(dndv_bin_corr, -0.95, 0.95))
        idx = np.arange(y.size)
        corr = rho ** np.abs(idx[:, None] - idx[None, :])
        return sigma, build_covariance(sigma, corr)

    sigma = np.asarray(
        [
            max(np.log1p(frac), 1.0e-6),
            max(frac * abs(y[1]), 1.0e-3),
            max(frac * abs(y[2]), 1.0e-3),
            max(float(shape_skew_sigma), 1.0e-3),
            max(float(shape_kurt_sigma), 1.0e-3),
        ],
        dtype=float,
    )
    corr = np.asarray(
        [
            [1.0, 0.20, 0.10, 0.00, 0.00],
            [0.20, 1.0, 0.35, 0.10, 0.05],
            [0.10, 0.35, 1.0, 0.10, 0.05],
            [0.00, 0.10, 0.10, 1.0, 0.20],
            [0.00, 0.05, 0.05, 0.20, 1.0],
        ],
        dtype=float,
    )
    return sigma, build_covariance(sigma, corr)


def draw_synthetic_observation(
    rng: np.random.Generator,
    model: Any,
    true_observables: Sequence[float],
    covariance: np.ndarray,
) -> np.ndarray:
    """Draw one noisy observable vector and enforce basic physical floors."""
    observed = rng.multivariate_normal(np.asarray(true_observables, dtype=float), np.asarray(covariance, dtype=float))
    if model.observable_set == "m0_m1_m2":
        observed = np.maximum(observed, 1.0e-24)
    elif model.observable_set == "dndv_binned":
        observed = np.maximum(observed, 1.0e-40)
    else:
        observed[2] = max(float(observed[2]), 1.0e-3)
    return np.asarray(observed, dtype=float)


def summarize_recovery_metrics(
    theta_true: Sequence[float],
    theta_map: Sequence[float],
    samples_theta: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute recovery metrics for one realization."""
    truth = np.asarray(theta_true, dtype=float)
    map_theta = np.asarray(theta_map, dtype=float)
    samples = np.asarray(samples_theta, dtype=float)

    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != truth.size:
        nan = np.full_like(truth, np.nan, dtype=float)
        false = np.zeros_like(truth, dtype=bool)
        return {
            "truth_in_68pct_interval": false,
            "truth_in_95pct_interval": false,
            "posterior_mean_bias": nan.copy(),
            "posterior_median_bias": nan.copy(),
            "posterior_width": nan.copy(),
            "map_error": np.abs(map_theta - truth) if map_theta.shape == truth.shape else nan.copy(),
            "q16": nan.copy(),
            "q50": nan.copy(),
            "q84": nan.copy(),
            "q025": nan.copy(),
            "q975": nan.copy(),
        }

    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    q025, q975 = np.percentile(samples, [2.5, 97.5], axis=0)
    return {
        "truth_in_68pct_interval": (truth >= q16) & (truth <= q84),
        "truth_in_95pct_interval": (truth >= q025) & (truth <= q975),
        "posterior_mean_bias": np.mean(samples, axis=0) - truth,
        "posterior_median_bias": q50 - truth,
        "posterior_width": q84 - q16,
        "map_error": np.abs(map_theta - truth),
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "q025": q025,
        "q975": q975,
    }


def _empty_result(
    truth_case: TruthCase,
    realization_id: int,
    model: Any,
    observed_dim: int,
    theta_true: np.ndarray,
    true_observables: np.ndarray,
    true_raw_moments: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    covariance: np.ndarray,
    truth_valid: bool,
    first_invalid_r_kpc: float,
    error: str,
    runtime_seconds: dict[str, float] | None = None,
) -> RealizationResult:
    metrics = summarize_recovery_metrics(theta_true, np.full((3,), np.nan), np.empty((0, 3)))
    return RealizationResult(
        truth_case=truth_case.name,
        realization_id=int(realization_id),
        theta_true=theta_true,
        true_observables=true_observables,
        true_raw_moments=true_raw_moments,
        observed=observed,
        sigma=sigma,
        covariance=covariance,
        truth_valid=truth_valid,
        first_invalid_r_kpc=first_invalid_r_kpc,
        success=False,
        error=error,
        map_success=False,
        map_message="",
        theta_map=np.full((3,), np.nan, dtype=float),
        map_predicted_observables=np.full((observed_dim,), np.nan, dtype=float),
        chi2=np.nan,
        nlp=np.nan,
        samples_theta=np.empty((0, 3), dtype=float),
        samples_log=np.empty((0, 3), dtype=float),
        posterior_predictive=np.empty((0, observed_dim), dtype=float),
        metrics=metrics,
        diagnostics={
            "sampler": getattr(model, "sampler", ""),
            "acceptance_rate": np.nan,
            "num_divergent": np.nan,
            "r_hat": None,
            "ess_bulk": None,
            "correlation_theta": np.full((3, 3), np.nan, dtype=float),
            "posterior_prob_eta_E_gt_1": np.nan,
        },
        runtime_seconds=runtime_seconds or {},
    )


def run_realization(
    args: argparse.Namespace,
    model: Any,
    truth_case: TruthCase,
    realization_id: int,
    rng: np.random.Generator,
) -> RealizationResult:
    """Run one noisy synthetic recovery realization."""
    start = time.perf_counter()
    theta_true = np.asarray(truth_case.theta, dtype=float)
    true_observables, true_raw_moments, truth_valid, first_invalid_r_kpc = _predict_with_validity(model, theta_true)
    sigma, covariance = build_synthetic_covariance(
        model=model,
        true_observables=true_observables,
        noise_fraction=args.noise_fraction,
        shape_skew_sigma=args.shape_skew_sigma,
        shape_kurt_sigma=args.shape_kurt_sigma,
        dndv_sigma_floor_frac=args.dndv_sigma_floor_frac,
        dndv_bin_corr=args.dndv_bin_corr,
    )
    if args.use_truth_observables:
        observed = np.asarray(true_observables, dtype=float).copy()
    else:
        observed = draw_synthetic_observation(rng, model, true_observables, covariance)

    if not truth_valid:
        return _empty_result(
            truth_case=truth_case,
            realization_id=realization_id,
            model=model,
            observed_dim=model.observable_dim,
            theta_true=theta_true,
            true_observables=true_observables,
            true_raw_moments=true_raw_moments,
            observed=observed,
            sigma=sigma,
            covariance=covariance,
            truth_valid=truth_valid,
            first_invalid_r_kpc=first_invalid_r_kpc,
            error="Truth case produced invalid model observables",
            runtime_seconds={"total": time.perf_counter() - start},
        )

    initial_theta = (args.initial_eta_m, args.initial_eta_m_cold, args.initial_eta_e)
    prior_mean_log = np.log(np.asarray([args.prior_eta_m, args.prior_eta_m_cold, args.prior_eta_e], dtype=float))
    prior_sigma_log = (
        args.prior_sigma_log_eta_m,
        args.prior_sigma_log_eta_m_cold,
        args.prior_sigma_log_eta_e,
    )

    try:
        if args.sampler == "none":
            t_fit = time.perf_counter()
            fit_map = model.fit_map(
                observed_moments=observed,
                covariance_moments=covariance,
                initial_theta=initial_theta,
                prior_mean_log=prior_mean_log,
                prior_sigma_log=prior_sigma_log,
                max_iter=args.map_max_iter,
                map_num_starts=args.map_num_starts,
                seed=args.seed + 1009 * realization_id,
            )
            fit_seconds = time.perf_counter() - t_fit
            samples_theta = np.empty((0, 3), dtype=float)
            samples_log = np.empty((0, 3), dtype=float)
            posterior_predictive = np.empty((0, model.observable_dim), dtype=float)
            diagnostics = {
                "sampler": "none",
                "acceptance_rate": np.nan,
                "num_divergent": np.nan,
                "r_hat": None,
                "ess_bulk": None,
                "correlation_theta": fit_map.correlation_theta,
                "posterior_prob_eta_E_gt_1": np.nan,
            }
            runtime_fit = {"map": fit_seconds, "posterior_sampling": 0.0, "total": fit_seconds}
        else:
            t_fit = time.perf_counter()
            fit = model.fit_posterior(
                observed_moments=observed,
                covariance_moments=covariance,
                initial_theta=initial_theta,
                prior_mean_log=prior_mean_log,
                prior_sigma_log=prior_sigma_log,
                map_max_iter=args.map_max_iter,
                map_num_starts=args.map_num_starts,
                hmc_num_warmup=args.num_warmup,
                hmc_num_samples=args.num_samples,
                hmc_step_size=args.hmc_step_size,
                hmc_leapfrog_steps=args.hmc_leapfrog_steps,
                hmc_target_accept=args.hmc_target_accept,
                sampler=args.sampler,
                num_chains=args.num_chains,
                nuts_chain_method=args.nuts_chain_method,
                nuts_dense_mass=args.nuts_dense_mass,
                nuts_max_tree_depth=args.nuts_max_tree_depth,
                nuts_progress_bar=not args.disable_progress_bar,
                seed=args.seed + 1009 * realization_id,
            )
            fit_seconds = time.perf_counter() - t_fit
            fit_map = fit.map
            samples_theta = fit.hmc.samples_theta
            samples_log = fit.hmc.samples_log
            posterior_predictive = model.predict_observables_for_log_samples(
                samples_log,
                max_samples=args.posterior_predictive_samples,
            )
            diagnostics = {
                "sampler": fit.hmc.sampler,
                "num_chains": int(fit.hmc.num_chains),
                "nuts_chain_method": fit.hmc.nuts_chain_method,
                "nuts_dense_mass": fit.hmc.nuts_dense_mass,
                "nuts_max_tree_depth": fit.hmc.nuts_max_tree_depth,
                "acceptance_rate": float(fit.hmc.acceptance_rate),
                "acceptance_rate_per_chain": None
                if fit.hmc.acceptance_rate_per_chain is None
                else fit.hmc.acceptance_rate_per_chain,
                "num_divergent": int(fit.hmc.num_divergent),
                "num_divergent_per_chain": None
                if fit.hmc.num_divergent_per_chain is None
                else fit.hmc.num_divergent_per_chain,
                "r_hat": fit.hmc.r_hat,
                "ess_bulk": fit.hmc.ess_bulk,
                "bfmi": fit.hmc.bfmi,
                "bfmi_per_chain": fit.hmc.bfmi_per_chain,
                "correlation_theta": fit.hmc.correlation_theta,
                "posterior_prob_eta_E_gt_1": float(np.mean(samples_theta[:, 2] > 1.0))
                if samples_theta.size > 0
                else np.nan,
            }
            runtime_fit = fit.runtime_seconds or {"total": fit_seconds}

        metrics = summarize_recovery_metrics(theta_true, fit_map.theta_map, samples_theta)
        runtime = {
            "fit_wall": float(fit_seconds),
            "total": float(time.perf_counter() - start),
            **{key: float(value) for key, value in runtime_fit.items()},
        }
        return RealizationResult(
            truth_case=truth_case.name,
            realization_id=int(realization_id),
            theta_true=theta_true,
            true_observables=true_observables,
            true_raw_moments=true_raw_moments,
            observed=observed,
            sigma=sigma,
            covariance=covariance,
            truth_valid=truth_valid,
            first_invalid_r_kpc=first_invalid_r_kpc,
            success=True,
            error="",
            map_success=bool(fit_map.success),
            map_message=fit_map.message,
            theta_map=np.asarray(fit_map.theta_map, dtype=float),
            map_predicted_observables=np.asarray(fit_map.predicted_moments, dtype=float),
            chi2=float(fit_map.chi2),
            nlp=float(fit_map.nlp),
            samples_theta=np.asarray(samples_theta, dtype=float),
            samples_log=np.asarray(samples_log, dtype=float),
            posterior_predictive=np.asarray(posterior_predictive, dtype=float),
            metrics=metrics,
            diagnostics=diagnostics,
            runtime_seconds=runtime,
        )
    except Exception as exc:  # noqa: BLE001 - failures are a recorded output of this harness.
        return _empty_result(
            truth_case=truth_case,
            realization_id=realization_id,
            model=model,
            observed_dim=model.observable_dim,
            theta_true=theta_true,
            true_observables=true_observables,
            true_raw_moments=true_raw_moments,
            observed=observed,
            sigma=sigma,
            covariance=covariance,
            truth_valid=truth_valid,
            first_invalid_r_kpc=first_invalid_r_kpc,
            error=str(exc),
            runtime_seconds={"total": time.perf_counter() - start},
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _pad_stack(arrays: Sequence[np.ndarray], trailing_shape: tuple[int, ...]) -> np.ndarray:
    max_len = max((arr.shape[0] for arr in arrays), default=0)
    out = np.full((len(arrays), max_len, *trailing_shape), np.nan, dtype=float)
    for i, arr in enumerate(arrays):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        out[i, : arr.shape[0], ...] = arr
    return out


def write_npz(output_dir: str, model: Any, results: Sequence[RealizationResult]) -> str:
    """Write machine-readable recovery arrays."""
    path = os.path.join(output_dir, "synthetic_recovery_results.npz")
    np.savez(
        path,
        truth_case=np.asarray([r.truth_case for r in results], dtype=str),
        realization_id=np.asarray([r.realization_id for r in results], dtype=int),
        success=np.asarray([r.success for r in results], dtype=bool),
        truth_valid=np.asarray([r.truth_valid for r in results], dtype=bool),
        first_invalid_r_kpc=np.asarray([r.first_invalid_r_kpc for r in results], dtype=float),
        errors=np.asarray([r.error for r in results], dtype=str),
        theta_names=np.asarray(PARAM_NAMES, dtype=str),
        observable_names=np.asarray(model.observable_names, dtype=str),
        observable_set=np.asarray(model.observable_set, dtype=str),
        theta_true=np.vstack([r.theta_true for r in results]),
        true_observables=np.vstack([r.true_observables for r in results]),
        true_raw_moments=np.vstack([r.true_raw_moments for r in results]),
        observed=np.vstack([r.observed for r in results]),
        sigma=np.vstack([r.sigma for r in results]),
        covariance=np.stack([r.covariance for r in results]),
        theta_map=np.vstack([r.theta_map for r in results]),
        map_predicted_observables=np.vstack([r.map_predicted_observables for r in results]),
        chi2=np.asarray([r.chi2 for r in results], dtype=float),
        nlp=np.asarray([r.nlp for r in results], dtype=float),
        map_success=np.asarray([r.map_success for r in results], dtype=bool),
        samples_theta=_pad_stack([r.samples_theta for r in results], (len(PARAM_NAMES),)),
        samples_log=_pad_stack([r.samples_log for r in results], (len(PARAM_NAMES),)),
        posterior_predictive=_pad_stack([r.posterior_predictive for r in results], (model.observable_dim,)),
        truth_in_68pct_interval=np.vstack([r.metrics["truth_in_68pct_interval"] for r in results]),
        truth_in_95pct_interval=np.vstack([r.metrics["truth_in_95pct_interval"] for r in results]),
        posterior_q16=np.vstack([r.metrics["q16"] for r in results]),
        posterior_q50=np.vstack([r.metrics["q50"] for r in results]),
        posterior_q84=np.vstack([r.metrics["q84"] for r in results]),
        posterior_q025=np.vstack([r.metrics["q025"] for r in results]),
        posterior_q975=np.vstack([r.metrics["q975"] for r in results]),
        posterior_mean_bias=np.vstack([r.metrics["posterior_mean_bias"] for r in results]),
        posterior_median_bias=np.vstack([r.metrics["posterior_median_bias"] for r in results]),
        posterior_width=np.vstack([r.metrics["posterior_width"] for r in results]),
        map_error=np.vstack([r.metrics["map_error"] for r in results]),
        parameter_correlation=np.stack([np.asarray(r.diagnostics["correlation_theta"], dtype=float) for r in results]),
        posterior_prob_eta_E_gt_1=np.asarray(
            [r.diagnostics.get("posterior_prob_eta_E_gt_1", np.nan) for r in results],
            dtype=float,
        ),
    )
    return path


def write_csv_summary(output_dir: str, results: Sequence[RealizationResult]) -> str:
    """Write compact row-per-realization recovery summary."""
    path = os.path.join(output_dir, "synthetic_recovery_summary.csv")
    fieldnames = [
        "truth_case",
        "realization_id",
        "truth_valid",
        "success",
        "map_success",
        "chi2",
        "acceptance_rate",
        "num_divergent",
        "max_r_hat",
        "min_ess_bulk",
        "posterior_prob_eta_E_gt_1",
        "runtime_total_seconds",
        "error",
    ]
    for name in PARAM_NAMES:
        fieldnames.extend(
            [
                f"true_{name}",
                f"map_{name}",
                f"map_error_{name}",
                f"posterior_mean_bias_{name}",
                f"posterior_median_bias_{name}",
                f"posterior_width_{name}",
                f"truth_in_68_{name}",
                f"truth_in_95_{name}",
            ]
        )

    with open(path, "w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            r_hat = result.diagnostics.get("r_hat")
            ess = result.diagnostics.get("ess_bulk")
            row: dict[str, Any] = {
                "truth_case": result.truth_case,
                "realization_id": result.realization_id,
                "truth_valid": result.truth_valid,
                "success": result.success,
                "map_success": result.map_success,
                "chi2": result.chi2,
                "acceptance_rate": result.diagnostics.get("acceptance_rate", np.nan),
                "num_divergent": result.diagnostics.get("num_divergent", np.nan),
                "max_r_hat": np.nan if r_hat is None else float(np.nanmax(r_hat)),
                "min_ess_bulk": np.nan if ess is None else float(np.nanmin(ess)),
                "posterior_prob_eta_E_gt_1": result.diagnostics.get("posterior_prob_eta_E_gt_1", np.nan),
                "runtime_total_seconds": result.runtime_seconds.get("total", np.nan),
                "error": result.error,
            }
            for idx, name in enumerate(PARAM_NAMES):
                row[f"true_{name}"] = result.theta_true[idx]
                row[f"map_{name}"] = result.theta_map[idx]
                row[f"map_error_{name}"] = result.metrics["map_error"][idx]
                row[f"posterior_mean_bias_{name}"] = result.metrics["posterior_mean_bias"][idx]
                row[f"posterior_median_bias_{name}"] = result.metrics["posterior_median_bias"][idx]
                row[f"posterior_width_{name}"] = result.metrics["posterior_width"][idx]
                row[f"truth_in_68_{name}"] = bool(result.metrics["truth_in_68pct_interval"][idx])
                row[f"truth_in_95_{name}"] = bool(result.metrics["truth_in_95pct_interval"][idx])
            writer.writerow(row)
    return path


def _diagnostic_slug(result: RealizationResult) -> str:
    return f"{result.truth_case}_realization_{result.realization_id:03d}"


def make_diagnostic_plots(output_dir: str, model: Any, results: Sequence[RealizationResult]) -> dict[str, list[str]]:
    """Write per-realization corner and observable-fit diagnostics."""
    from multiphasegalacticwind.inference import plot_corner, plot_dndv_fit, plot_observable_fit

    plot_dir = os.path.join(output_dir, "diagnostic_plots")
    os.makedirs(plot_dir, exist_ok=True)

    corner_paths: list[str] = []
    fit_paths: list[str] = []
    for result in results:
        if not result.success:
            continue

        slug = _diagnostic_slug(result)
        if result.samples_theta.size > 0:
            corner_path = os.path.join(plot_dir, f"{slug}_corner.png")
            plot_corner(
                result.samples_theta,
                labels=PARAM_NAMES,
                output_path=corner_path,
                truths=result.theta_true,
                map_theta=result.theta_map,
            )
            corner_paths.append(corner_path)

        fit_path = os.path.join(plot_dir, f"{slug}_observable_fit.png")
        if model.observable_set == "dndv_binned":
            plot_dndv_fit(
                velocity_bins_kms=model.get_dndv_velocity_bins(),
                observed_dndv=result.observed,
                covariance_dndv=result.covariance,
                map_dndv=result.map_predicted_observables,
                posterior_dndv_samples=result.posterior_predictive
                if result.posterior_predictive.size > 0
                else None,
                output_path=fit_path,
            )
        else:
            plot_observable_fit(
                observed_values=result.observed,
                covariance_values=result.covariance,
                map_values=result.map_predicted_observables,
                posterior_samples=result.posterior_predictive
                if result.posterior_predictive.size > 0
                else None,
                labels=model.observable_names,
                output_path=fit_path,
                yscale=model.default_observable_yscale,
            )
        fit_paths.append(fit_path)

    return {"corner_plots": corner_paths, "observable_fit_plots": fit_paths}


def write_metadata(
    output_dir: str,
    args: argparse.Namespace,
    model: Any,
    results: Sequence[RealizationResult],
    output_files: dict[str, Any],
    runtime_seconds: float,
) -> str:
    """Write run metadata and truth-case documentation."""
    success = np.asarray([result.success for result in results], dtype=bool)
    metadata = {
        "created_by": "examples/inference_synthetic_recovery.py",
        "runtime_seconds": float(runtime_seconds),
        "arguments": vars(args),
        "theta_names": list(PARAM_NAMES),
        "truth_cases": {
            name: {"theta": list(case.theta), "description": case.description} for name, case in TRUTH_CASES.items()
        },
        "observable_set": model.observable_set,
        "observable_names": list(model.observable_names),
        "eta_e_parameterization": model.eta_e_parameterization,
        "eta_e_softcap": {
            "center": model.eta_e_softcap_center,
            "sigma": model.eta_e_softcap_sigma,
            "transition": model.eta_e_softcap_transition,
        },
        "num_realizations": int(len(results)),
        "num_success": int(np.sum(success)),
        "success_fraction": float(np.mean(success)) if success.size else np.nan,
        "output_files": output_files,
        "notes": [
            "Fitted parameters are fixed to eta_M, eta_M_cold, and eta_E.",
            "Failures are retained as rows with success=false and NaN posterior arrays.",
            "sampler=none runs MAP only and leaves posterior interval metrics as NaN/false.",
            "Diagnostic corner plots use green dashed truth lines and red MAP lines.",
            "--use-truth-observables disables the random noise draw but keeps the configured covariance.",
            "Reported MAP values optimize the log-parameter posterior; posterior samplers include the unconstrained-transform Jacobian.",
            "eta_e_parameterization='softcap' is a diagnostic mode that allows eta_E > 1 with a smooth upper-tail penalty.",
        ],
    }
    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w", encoding="ascii") as fh:
        json.dump(_jsonable(metadata), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_jax_platform(args.jax_platform)

    start = time.perf_counter()
    os.makedirs(args.output, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    model = build_model(args)
    truth_cases = selected_truth_cases(args.truth_case)

    results: list[RealizationResult] = []
    realization_counter = 0
    for truth_case in truth_cases:
        for _ in range(args.num_noise_realizations):
            result = run_realization(args, model, truth_case, realization_counter, rng)
            results.append(result)
            status = "success" if result.success else "failed"
            print(
                f"[{len(results):4d}/{len(truth_cases) * args.num_noise_realizations}] "
                f"{truth_case.name} realization={result.realization_id}: {status}",
                flush=True,
            )
            if result.error:
                print(f"  error: {result.error}", flush=True)
            realization_counter += 1

    output_files: dict[str, Any] = {}
    output_files["npz"] = write_npz(args.output, model, results)
    output_files["csv"] = write_csv_summary(args.output, results)
    if not args.no_diagnostic_plots:
        output_files.update(make_diagnostic_plots(args.output, model, results))
    output_files["metadata"] = write_metadata(
        args.output,
        args,
        model,
        results,
        output_files,
        runtime_seconds=time.perf_counter() - start,
    )

    success_count = sum(result.success for result in results)
    print(f"Saved synthetic recovery results to {args.output}")
    print(f"Successful realizations: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
