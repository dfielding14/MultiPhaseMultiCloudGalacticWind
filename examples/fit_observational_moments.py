#!/usr/bin/env python
"""Fit (eta_M, eta_M_cold, eta_E) to observed [M0, M1, M2] moments."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from multiphasegalacticwind.inference import (
    MomentInferenceModel,
    build_covariance,
    plot_corner,
    plot_moment_fit,
    summarize_parameter_degeneracies,
)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "fit_observational_moments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit wind parameters to observed dN/dv moments")

    parser.add_argument("--sfr", type=float, required=True, help="Star formation rate [Msun/yr]")
    parser.add_argument("--r-star-kpc", type=float, required=True, help="Launch/sonic radius [kpc]")
    parser.add_argument("--v-circ", type=float, default=150.0, help="Circular velocity [km/s]")
    parser.add_argument("--r-max-kpc", type=float, default=30.0)

    parser.add_argument("--m0", type=float, required=True, help="Observed M0")
    parser.add_argument("--m1", type=float, required=True, help="Observed M1")
    parser.add_argument("--m2", type=float, required=True, help="Observed M2")

    parser.add_argument("--sigma-m0", type=float, required=True, help="1-sigma uncertainty on M0")
    parser.add_argument("--sigma-m1", type=float, required=True, help="1-sigma uncertainty on M1")
    parser.add_argument("--sigma-m2", type=float, required=True, help="1-sigma uncertainty on M2")

    parser.add_argument("--corr-m0-m1", type=float, default=0.0)
    parser.add_argument("--corr-m0-m2", type=float, default=0.0)
    parser.add_argument("--corr-m1-m2", type=float, default=0.0)

    parser.add_argument("--initial-eta-m", type=float, default=0.2)
    parser.add_argument("--initial-eta-m-cold", type=float, default=0.2)
    parser.add_argument("--initial-eta-e", type=float, default=1.0)

    parser.add_argument("--n-cloud-species", type=int, default=13)
    parser.add_argument("--cloud-mass-min", type=float, default=1.0)
    parser.add_argument("--cloud-mass-max", type=float, default=1e6)
    parser.add_argument("--cloud-alpha", type=float, default=2.0)

    parser.add_argument("--step-kpc", type=float, default=0.02)
    parser.add_argument("--first-step-kpc", type=float, default=1e-12)
    parser.add_argument("--integrator-mode", choices=["rk2", "rk3", "rk4", "tsit5"], default="rk4")
    parser.add_argument("--integrator-rtol", type=float, default=1e-5)
    parser.add_argument("--integrator-atol", type=float, default=1e-8)
    parser.add_argument("--integrator-max-steps", type=int, default=131072)

    parser.add_argument("--map-max-iter", type=int, default=25)
    parser.add_argument("--hmc-warmup", type=int, default=1000)
    parser.add_argument("--hmc-samples", type=int, default=1500)
    parser.add_argument("--hmc-step-size", type=float, default=0.02)
    parser.add_argument("--hmc-leapfrog-steps", type=int, default=12)
    parser.add_argument("--hmc-target-accept", type=float, default=0.8)
    parser.add_argument("--sampler", choices=["nuts", "hmc"], default="nuts")
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument(
        "--nuts-chain-method",
        choices=["auto", "sequential", "parallel", "vectorized"],
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    observed = np.asarray([args.m0, args.m1, args.m2], dtype=float)
    sigma = np.asarray([args.sigma_m0, args.sigma_m1, args.sigma_m2], dtype=float)
    corr = np.asarray(
        [
            [1.0, args.corr_m0_m1, args.corr_m0_m2],
            [args.corr_m0_m1, 1.0, args.corr_m1_m2],
            [args.corr_m0_m2, args.corr_m1_m2, 1.0],
        ],
        dtype=float,
    )
    covariance = build_covariance(sigma, corr)

    model = MomentInferenceModel(
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
    )

    fit = model.fit_posterior(
        observed_moments=observed,
        covariance_moments=covariance,
        initial_theta=(args.initial_eta_m, args.initial_eta_m_cold, args.initial_eta_e),
        map_max_iter=args.map_max_iter,
        hmc_num_warmup=args.hmc_warmup,
        hmc_num_samples=args.hmc_samples,
        hmc_step_size=args.hmc_step_size,
        hmc_leapfrog_steps=args.hmc_leapfrog_steps,
        hmc_target_accept=args.hmc_target_accept,
        sampler=args.sampler,
        num_chains=args.num_chains,
        nuts_chain_method=args.nuts_chain_method,
        seed=args.seed,
    )

    posterior_moment_samples = model.predict_moments_for_log_samples(fit.hmc.samples_log, max_samples=400)

    corner_path = os.path.join(args.output_dir, "posterior_corner.png")
    moments_path = os.path.join(args.output_dir, "moment_fit.png")
    plot_corner(
        fit.hmc.samples_theta,
        labels=MomentInferenceModel.PARAM_NAMES,
        output_path=corner_path,
        map_theta=fit.map.theta_map,
    )
    plot_moment_fit(
        observed_moments=fit.observed_moments,
        covariance_moments=fit.covariance_moments,
        map_moments=fit.map.predicted_moments,
        posterior_moment_samples=posterior_moment_samples,
        output_path=moments_path,
    )

    theta_map = fit.map.theta_map
    theta_std = np.sqrt(np.diag(fit.hmc.covariance_theta))
    corr = fit.hmc.correlation_theta

    print("Best-fit parameters (MAP):")
    print(f"  eta_M      = {theta_map[0]:.5f}")
    print(f"  eta_M_cold = {theta_map[1]:.5f}")
    print(f"  eta_E      = {theta_map[2]:.5f}")
    print("Approx posterior 1-sigma from HMC samples:")
    print(f"  sigma(eta_M)      = {theta_std[0]:.5f}")
    print(f"  sigma(eta_M_cold) = {theta_std[1]:.5f}")
    print(f"  sigma(eta_E)      = {theta_std[2]:.5f}")
    print(f"MAP chi2 = {fit.map.chi2:.4f}")
    print(f"Sampler = {fit.hmc.sampler} ({fit.hmc.num_chains} chain(s))")
    if fit.hmc.sampler == "nuts":
        print(f"NUTS chain method = {fit.hmc.nuts_chain_method}")
    print(f"Sampler acceptance rate = {fit.hmc.acceptance_rate:.3f}")
    print(f"Divergent transitions = {fit.hmc.num_divergent}")
    if fit.hmc.r_hat is not None:
        print(f"max Rhat(log_theta) = {np.max(fit.hmc.r_hat):.4f}")
    if fit.hmc.ess_bulk is not None:
        print(f"min ESS(log_theta) = {np.min(fit.hmc.ess_bulk):.1f}")

    print("\nDegeneracy summary (HMC correlation):")
    for line in summarize_parameter_degeneracies(corr, MomentInferenceModel.PARAM_NAMES):
        print(f"  {line}")

    results = {
        "input": {
            "sfr": args.sfr,
            "r_star_kpc": args.r_star_kpc,
            "v_circ": args.v_circ,
            "r_max_kpc": args.r_max_kpc,
            "observed_moments": observed.tolist(),
            "sigma_moments": sigma.tolist(),
            "covariance": covariance.tolist(),
        },
        "map": {
            "theta_map": fit.map.theta_map.tolist(),
            "log_theta_map": fit.map.log_theta_map.tolist(),
            "predicted_moments": fit.map.predicted_moments.tolist(),
            "chi2": float(fit.map.chi2),
            "nlp": float(fit.map.nlp),
            "success": bool(fit.map.success),
            "message": fit.map.message,
        },
        "hmc": {
            "sampler": fit.hmc.sampler,
            "num_chains": int(fit.hmc.num_chains),
            "nuts_chain_method": fit.hmc.nuts_chain_method,
            "acceptance_rate": float(fit.hmc.acceptance_rate),
            "acceptance_rate_per_chain": None
            if fit.hmc.acceptance_rate_per_chain is None
            else fit.hmc.acceptance_rate_per_chain.tolist(),
            "final_step_size": float(fit.hmc.final_step_size),
            "num_divergent": int(fit.hmc.num_divergent),
            "num_divergent_per_chain": None
            if fit.hmc.num_divergent_per_chain is None
            else fit.hmc.num_divergent_per_chain.tolist(),
            "num_steps_mean": float(fit.hmc.num_steps_mean),
            "bfmi_per_chain": None if fit.hmc.bfmi_per_chain is None else fit.hmc.bfmi_per_chain.tolist(),
            "r_hat": None if fit.hmc.r_hat is None else fit.hmc.r_hat.tolist(),
            "ess_bulk": None if fit.hmc.ess_bulk is None else fit.hmc.ess_bulk.tolist(),
            "covariance_theta": fit.hmc.covariance_theta.tolist(),
            "correlation_theta": fit.hmc.correlation_theta.tolist(),
        },
    }

    with open(os.path.join(args.output_dir, "fit_summary.json"), "w", encoding="ascii") as fh:
        json.dump(results, fh, indent=2)

    np.savez(
        os.path.join(args.output_dir, "fit_posterior_samples.npz"),
        samples_log=fit.hmc.samples_log,
        samples_theta=fit.hmc.samples_theta,
        observed=fit.observed_moments,
        covariance=fit.covariance_moments,
        map_theta=fit.map.theta_map,
        map_predicted_moments=fit.map.predicted_moments,
    )

    print(f"\nSaved outputs to {args.output_dir}")
    print(f"  - {corner_path}")
    print(f"  - {moments_path}")


if __name__ == "__main__":
    main()
