#!/usr/bin/env python3
"""End-to-end publication-style synthetic inference for an M82-like wind."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from multiphasegalacticwind.inference import (
    MomentInferenceModel,
    build_covariance,
    plot_dndv_fit,
    plot_observable_fit,
    plot_corner,
    summarize_parameter_degeneracies,
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "m82_publication_inference")


class StatusLogger:
    """Simple elapsed-time logger writing to stdout and a text file."""

    def __init__(self, log_path: str) -> None:
        self._start = time.perf_counter()
        self._fh = open(log_path, "w", encoding="ascii")

    def close(self) -> None:
        self._fh.close()

    def log(self, message: str) -> None:
        elapsed = time.perf_counter() - self._start
        line = f"[+{elapsed:9.2f}s] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--sfr", type=float, default=20.0, help="Star formation rate [Msun/yr]")
    parser.add_argument("--r-star-kpc", type=float, default=0.3, help="Launch/sonic radius [kpc]")
    parser.add_argument("--v-circ", type=float, default=150.0, help="Circular velocity [km/s]")

    parser.add_argument("--eta-m", type=float, default=0.1, help="True hot mass-loading factor")
    parser.add_argument("--eta-m-cold", type=float, default=0.15, help="True cold mass-loading factor")
    parser.add_argument("--eta-e", type=float, default=0.9, help="True hot energy-loading factor")

    parser.add_argument(
        "--observable-set",
        choices=["m0_m1_m2", "logm0_mean_sigma_skew_kurt", "dndv_binned"],
        default="logm0_mean_sigma_skew_kurt",
        help="Observable vector used in the likelihood.",
    )
    parser.add_argument("--frac-error", type=float, default=0.10, help="Fractional 1-sigma error scale")
    parser.add_argument("--shape-skew-sigma", type=float, default=0.20, help="Absolute 1-sigma for skewness")
    parser.add_argument("--shape-kurt-sigma", type=float, default=0.40, help="Absolute 1-sigma for kurtosis")
    parser.add_argument("--dndv-num-bins", type=int, default=25, help="Number of velocity bins for dN/dv mode")
    parser.add_argument("--dndv-vmin-kms", type=float, default=0.0, help="Minimum velocity bin edge [km/s]")
    parser.add_argument("--dndv-vmax-kms", type=float, default=1200.0, help="Maximum velocity bin edge [km/s]")
    parser.add_argument(
        "--dndv-kernel-sigma-kms",
        type=float,
        default=None,
        help="Gaussian kernel width for dN/dv projection [km/s]. Default: half-bin width.",
    )
    parser.add_argument(
        "--dndv-sigma-floor-frac",
        type=float,
        default=0.03,
        help="Minimum bin uncertainty as a fraction of max(true dN/dv).",
    )
    parser.add_argument(
        "--dndv-bin-corr",
        type=float,
        default=0.60,
        help="AR(1) bin-to-bin correlation coefficient for synthetic dN/dv covariance.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--r-max-kpc", type=float, default=30.0)
    parser.add_argument("--step-kpc", type=float, default=0.02)
    parser.add_argument("--first-step-kpc", type=float, default=1e-12)
    parser.add_argument("--n-cloud-species", type=int, default=13)
    parser.add_argument("--integrator-mode", choices=["rk2", "rk3", "rk4", "tsit5"], default="rk4")
    parser.add_argument("--integrator-rtol", type=float, default=1e-5)
    parser.add_argument("--integrator-atol", type=float, default=1e-8)
    parser.add_argument("--integrator-max-steps", type=int, default=131072)

    parser.add_argument("--map-max-iter", type=int, default=35)
    parser.add_argument("--map-num-starts", type=int, default=6)
    parser.add_argument("--hmc-warmup", type=int, default=1800)
    parser.add_argument("--hmc-samples", type=int, default=3200)
    parser.add_argument("--hmc-step-size", type=float, default=0.02)
    parser.add_argument("--hmc-target-accept", type=float, default=0.9)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument(
        "--nuts-chain-method",
        choices=["auto", "sequential", "parallel", "vectorized"],
        default="auto",
    )
    parser.add_argument("--sampler", choices=["nuts", "hmc"], default="nuts")
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=12)
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable live NumPyro progress bar.",
    )

    parser.add_argument(
        "--posterior-moment-samples",
        type=int,
        default=1200,
        help="Max posterior draws used to evaluate observable predictive summaries.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _summarize_recovery(theta_true: np.ndarray, theta_map: np.ndarray, samples_theta: np.ndarray) -> dict[str, Any]:
    p16 = np.percentile(samples_theta, 16, axis=0)
    p50 = np.percentile(samples_theta, 50, axis=0)
    p84 = np.percentile(samples_theta, 84, axis=0)
    p025 = np.percentile(samples_theta, 2.5, axis=0)
    p975 = np.percentile(samples_theta, 97.5, axis=0)

    map_abs = np.abs(theta_map - theta_true)
    map_rel = map_abs / np.maximum(theta_true, 1e-30)
    med_abs = np.abs(p50 - theta_true)
    med_rel = med_abs / np.maximum(theta_true, 1e-30)

    return {
        "q16": p16,
        "q50": p50,
        "q84": p84,
        "q025": p025,
        "q975": p975,
        "truth_in_68": (theta_true >= p16) & (theta_true <= p84),
        "truth_in_95": (theta_true >= p025) & (theta_true <= p975),
        "map_abs_error": map_abs,
        "map_rel_error": map_rel,
        "median_abs_error": med_abs,
        "median_rel_error": med_rel,
    }


def _format_observables(labels: tuple[str, ...], values: np.ndarray) -> str:
    entries = [f"{name}={values[i]:.6e}" for i, name in enumerate(labels)]
    return ", ".join(entries)


def _build_synthetic_covariance(
    model: MomentInferenceModel,
    true_observables: np.ndarray,
    frac_error: float,
    shape_skew_sigma: float,
    shape_kurt_sigma: float,
    dndv_sigma_floor_frac: float,
    dndv_bin_corr: float,
) -> tuple[np.ndarray, np.ndarray]:
    if model.observable_set == "m0_m1_m2":
        sigma = np.maximum(frac_error * np.abs(true_observables), 1e-30)
        corr = np.asarray(
            [
                [1.0, 0.55, 0.35],
                [0.55, 1.0, 0.65],
                [0.35, 0.65, 1.0],
            ],
            dtype=float,
        )
        covariance = build_covariance(sigma, corr)
        return sigma, covariance

    if model.observable_set == "dndv_binned":
        amp = max(np.max(np.abs(true_observables)), 1e-30)
        sigma_floor = max(dndv_sigma_floor_frac, 1e-6) * amp
        sigma = np.maximum(frac_error * np.abs(true_observables), sigma_floor)
        rho = float(np.clip(dndv_bin_corr, -0.95, 0.95))
        idx = np.arange(model.observable_dim)
        corr = rho ** np.abs(idx[:, None] - idx[None, :])
        covariance = build_covariance(sigma, corr)
        return sigma, covariance

    sigma = np.asarray(
        [
            max(np.log1p(frac_error), 1e-6),
            max(frac_error * abs(true_observables[1]), 1e-3),
            max(frac_error * abs(true_observables[2]), 1e-3),
            max(shape_skew_sigma, 1e-3),
            max(shape_kurt_sigma, 1e-3),
        ],
        dtype=float,
    )
    corr = np.asarray(
        [
            [1.0, 0.20, 0.10, 0.0, 0.0],
            [0.20, 1.0, 0.35, 0.10, 0.05],
            [0.10, 0.35, 1.0, 0.10, 0.05],
            [0.0, 0.10, 0.10, 1.0, 0.20],
            [0.0, 0.05, 0.05, 0.20, 1.0],
        ],
        dtype=float,
    )
    covariance = build_covariance(sigma, corr)
    return sigma, covariance


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "run_status.log")
    logger = StatusLogger(log_path)

    try:
        logger.log("Starting M82 publication-style synthetic inference run.")
        logger.log(
            "Truth theta=[eta_M, eta_M_cold, eta_E]="
            f"[{args.eta_m:.4f}, {args.eta_m_cold:.4f}, {args.eta_e:.4f}]"
        )
        logger.log(
            f"Model setup: SFR={args.sfr:.3f} Msun/yr, r_star={args.r_star_kpc:.3f} kpc, v_circ={args.v_circ:.1f} km/s"
        )
        logger.log(f"Observable set: {args.observable_set}")
        logger.log(
            "Sampling setup: "
            f"{args.sampler.upper()} warmup={args.hmc_warmup}, samples={args.hmc_samples}, "
            f"chains={args.num_chains}, target_accept={args.hmc_target_accept:.2f}"
        )

        t0 = time.perf_counter()
        model = MomentInferenceModel(
            sfr=args.sfr,
            r_star_kpc=args.r_star_kpc,
            v_circ=args.v_circ,
            r_max_kpc=args.r_max_kpc,
            step_kpc=args.step_kpc,
            first_step_kpc=args.first_step_kpc,
            n_cloud_species=args.n_cloud_species,
            integrator_mode=args.integrator_mode,
            integrator_rtol=args.integrator_rtol,
            integrator_atol=args.integrator_atol,
            integrator_max_steps=args.integrator_max_steps,
            observable_set=args.observable_set,
            dndv_num_bins=args.dndv_num_bins,
            dndv_vmin_kms=args.dndv_vmin_kms,
            dndv_vmax_kms=args.dndv_vmax_kms,
            dndv_kernel_sigma_kms=args.dndv_kernel_sigma_kms,
        )
        t_model = time.perf_counter() - t0
        logger.log(f"MomentInferenceModel constructed in {t_model:.2f} s.")

        theta_true = np.asarray([args.eta_m, args.eta_m_cold, args.eta_e], dtype=float)

        t0 = time.perf_counter()
        true_observables = model.predict_observables(theta_true)
        true_raw_moments = model.predict_raw_moments(theta_true)
        t_truth = time.perf_counter() - t0
        logger.log(
            f"Forward truth observables computed in {t_truth:.2f} s: "
            + _format_observables(model.observable_names, true_observables)
        )
        logger.log(
            "Raw moments [M0..M4]: "
            + ", ".join(f"M{i}={true_raw_moments[i]:.6e}" for i in range(true_raw_moments.size))
        )

        sigma, covariance = _build_synthetic_covariance(
            model=model,
            true_observables=true_observables,
            frac_error=args.frac_error,
            shape_skew_sigma=args.shape_skew_sigma,
            shape_kurt_sigma=args.shape_kurt_sigma,
            dndv_sigma_floor_frac=args.dndv_sigma_floor_frac,
            dndv_bin_corr=args.dndv_bin_corr,
        )

        rng = np.random.default_rng(args.seed)
        observed = rng.multivariate_normal(true_observables, covariance)
        if model.observable_set == "m0_m1_m2":
            observed = np.maximum(observed, 1e-24)
        elif model.observable_set == "dndv_binned":
            observed = np.maximum(observed, 1e-40)
        else:
            observed[2] = max(observed[2], 1e-3)  # keep sampled velocity dispersion physical
        logger.log("Synthetic observation generated: " + _format_observables(model.observable_names, observed))

        def inference_status(message: str) -> None:
            logger.log(f"[inference] {message}")

        logger.log("Launching MAP + posterior sampling.")
        t0 = time.perf_counter()
        fit = model.fit_posterior(
            observed_moments=observed,
            covariance_moments=covariance,
            initial_theta=(0.2, 0.2, 0.8),
            prior_mean_log=np.log(np.asarray([0.2, 0.2, 0.7], dtype=float)),
            prior_sigma_log=(1.4, 1.4, 0.45),
            map_max_iter=args.map_max_iter,
            map_num_starts=args.map_num_starts,
            hmc_num_warmup=args.hmc_warmup,
            hmc_num_samples=args.hmc_samples,
            hmc_step_size=args.hmc_step_size,
            hmc_leapfrog_steps=args.hmc_leapfrog_steps,
            hmc_target_accept=args.hmc_target_accept,
            sampler=args.sampler,
            num_chains=args.num_chains,
            nuts_chain_method=args.nuts_chain_method,
            nuts_progress_bar=not args.disable_progress_bar,
            status_callback=inference_status,
            seed=args.seed,
        )
        t_fit = time.perf_counter() - t0
        logger.log(f"MAP + posterior sampling returned in {t_fit:.2f} s.")

        t0 = time.perf_counter()
        posterior_observable_samples = model.predict_observables_for_log_samples(
            fit.hmc.samples_log,
            max_samples=args.posterior_moment_samples,
        )
        t_pred = time.perf_counter() - t0
        logger.log(f"Posterior predictive observables computed in {t_pred:.2f} s.")

        corner_path = os.path.join(args.output_dir, "posterior_corner.png")
        if model.observable_set == "dndv_binned":
            observable_plot_path = os.path.join(args.output_dir, "dndv_fit.png")
            observable_plot_key = "plot_dndv_fit"
        else:
            observable_plot_path = os.path.join(args.output_dir, "observable_fit.png")
            observable_plot_key = "plot_observable_fit"

        t0 = time.perf_counter()
        plot_corner(
            fit.hmc.samples_theta,
            labels=MomentInferenceModel.PARAM_NAMES,
            output_path=corner_path,
            truths=theta_true,
            map_theta=fit.map.theta_map,
        )
        t_corner = time.perf_counter() - t0
        logger.log(f"Corner plot written in {t_corner:.2f} s -> {corner_path}")

        t0 = time.perf_counter()
        if model.observable_set == "dndv_binned":
            plot_dndv_fit(
                velocity_bins_kms=model.get_dndv_velocity_bins(),
                observed_dndv=observed,
                covariance_dndv=covariance,
                map_dndv=fit.map.predicted_moments,
                posterior_dndv_samples=posterior_observable_samples,
                output_path=observable_plot_path,
            )
        else:
            plot_observable_fit(
                observed_values=observed,
                covariance_values=covariance,
                map_values=fit.map.predicted_moments,
                posterior_samples=posterior_observable_samples,
                labels=model.observable_names,
                output_path=observable_plot_path,
                yscale=model.default_observable_yscale,
            )
        t_moment_plot = time.perf_counter() - t0
        logger.log(f"Observable-fit plot written in {t_moment_plot:.2f} s -> {observable_plot_path}")

        recovery = _summarize_recovery(theta_true, fit.map.theta_map, fit.hmc.samples_theta)
        runtime_fit = fit.runtime_seconds or {}
        runtime = {
            "construct_model": float(t_model),
            "forward_truth": float(t_truth),
            "fit_total_wall": float(t_fit),
            "posterior_predictive_eval": float(t_pred),
            "plot_corner": float(t_corner),
            observable_plot_key: float(t_moment_plot),
            "fit_map": float(runtime_fit.get("map", float("nan"))),
            "fit_posterior_sampling": float(runtime_fit.get("posterior_sampling", float("nan"))),
            "fit_total_reported": float(runtime_fit.get("total", float("nan"))),
            "run_total": float(logger.elapsed),
        }

        summary = {
            "input": {
                "sfr": float(args.sfr),
                "r_star_kpc": float(args.r_star_kpc),
                "v_circ": float(args.v_circ),
                "theta_true": theta_true.tolist(),
                "theta_labels": list(MomentInferenceModel.PARAM_NAMES),
                "observable_set": model.observable_set,
                "observable_labels": list(model.observable_names),
                "frac_error": float(args.frac_error),
                "dndv_num_bins": int(args.dndv_num_bins),
                "dndv_vmin_kms": float(args.dndv_vmin_kms),
                "dndv_vmax_kms": float(args.dndv_vmax_kms),
                "dndv_kernel_sigma_kms": float(model.dndv_kernel_sigma_kms),
                "seed": int(args.seed),
            },
            "observables": {
                "true": true_observables.tolist(),
                "observed": observed.tolist(),
                "sigma": sigma.tolist(),
                "covariance": covariance.tolist(),
                "raw_moments_true": true_raw_moments.tolist(),
                "velocity_bins_kms": None
                if model.observable_set != "dndv_binned"
                else model.get_dndv_velocity_bins().tolist(),
            },
            "map": {
                "theta_map": fit.map.theta_map.tolist(),
                "chi2": float(fit.map.chi2),
                "nlp": float(fit.map.nlp),
                "success": bool(fit.map.success),
                "message": fit.map.message,
                "predicted_observables": fit.map.predicted_moments.tolist(),
            },
            "posterior": {
                "sampler": fit.hmc.sampler,
                "num_chains": int(fit.hmc.num_chains),
                "nuts_chain_method": fit.hmc.nuts_chain_method,
                "num_samples_total": int(fit.hmc.samples_theta.shape[0]),
                "acceptance_rate": float(fit.hmc.acceptance_rate),
                "num_divergent": int(fit.hmc.num_divergent),
                "r_hat": None if fit.hmc.r_hat is None else fit.hmc.r_hat.tolist(),
                "ess_bulk": None if fit.hmc.ess_bulk is None else fit.hmc.ess_bulk.tolist(),
                "bfmi": None if fit.hmc.bfmi is None else float(fit.hmc.bfmi),
            },
            "recovery": {
                "q16": recovery["q16"].tolist(),
                "q50": recovery["q50"].tolist(),
                "q84": recovery["q84"].tolist(),
                "q025": recovery["q025"].tolist(),
                "q975": recovery["q975"].tolist(),
                "truth_in_68": recovery["truth_in_68"].astype(bool).tolist(),
                "truth_in_95": recovery["truth_in_95"].astype(bool).tolist(),
                "map_abs_error": recovery["map_abs_error"].tolist(),
                "map_rel_error": recovery["map_rel_error"].tolist(),
                "median_abs_error": recovery["median_abs_error"].tolist(),
                "median_rel_error": recovery["median_rel_error"].tolist(),
                "correlation_theta": fit.hmc.correlation_theta.tolist(),
                "degeneracy_summary": summarize_parameter_degeneracies(
                    fit.hmc.correlation_theta,
                    MomentInferenceModel.PARAM_NAMES,
                ),
            },
            "runtime_seconds": runtime,
            "outputs": {
                "corner_plot": corner_path,
                "observable_fit_plot": observable_plot_path,
                "status_log": log_path,
            },
        }

        summary_path = os.path.join(args.output_dir, "inference_summary.json")
        with open(summary_path, "w", encoding="ascii") as fh:
            json.dump(summary, fh, indent=2)
        logger.log(f"Summary JSON written -> {summary_path}")

        samples_path = os.path.join(args.output_dir, "posterior_samples.npz")
        np.savez(
            samples_path,
            theta_true=theta_true,
            observables_true=true_observables,
            observables_observed=observed,
            raw_moments_true=true_raw_moments,
            covariance=covariance,
            map_theta=fit.map.theta_map,
            map_observables=fit.map.predicted_moments,
            samples_theta=fit.hmc.samples_theta,
            samples_log=fit.hmc.samples_log,
            posterior_observable_samples=posterior_observable_samples,
        )
        logger.log(f"Posterior sample bundle written -> {samples_path}")

        logger.log("Final recovery summary:")
        for idx, name in enumerate(MomentInferenceModel.PARAM_NAMES):
            logger.log(
                f"  {name}: true={theta_true[idx]:.5f}, map={fit.map.theta_map[idx]:.5f}, "
                f"median={recovery['q50'][idx]:.5f}, 68%=[{recovery['q16'][idx]:.5f}, {recovery['q84'][idx]:.5f}]"
            )
        if fit.hmc.r_hat is None:
            rhat_text = "n/a"
        else:
            rhat_text = f"{np.max(fit.hmc.r_hat):.4f}"
        logger.log(
            "Sampler diagnostics: "
            f"acceptance={fit.hmc.acceptance_rate:.3f}, divergences={fit.hmc.num_divergent}, max_rhat={rhat_text}"
        )
        logger.log(f"Run complete. Total elapsed wall time: {logger.elapsed:.2f} s.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
