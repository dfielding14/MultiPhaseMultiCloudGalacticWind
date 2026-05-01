"""Inspect observed-window divergent transitions in synthetic recovery output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import gamma, kb, mp
from multiphasegalacticwind.inference import MomentInferenceModel
from multiphasegalacticwind.wind_model import WindModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", help="Path to synthetic_recovery_results.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--realization-index", type=int, default=0)
    parser.add_argument("--pre", type=int, default=6, help="Samples before each divergence to show")
    parser.add_argument("--post", type=int, default=3, help="Samples after each divergence to show")
    return parser.parse_args()


def _load_metadata(npz_path: Path) -> dict[str, Any]:
    meta_path = npz_path.with_name("run_metadata.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata next to {npz_path}: {meta_path}")
    with meta_path.open("r", encoding="ascii") as fh:
        return json.load(fh)


def _build_inference_model(arguments: dict[str, Any]) -> MomentInferenceModel:
    return MomentInferenceModel(
        sfr=arguments["sfr"],
        r_star_kpc=arguments["r_star_kpc"],
        v_circ=arguments["v_circ"],
        r_max_kpc=arguments["r_max_kpc"],
        step_kpc=arguments["step_kpc"],
        first_step_kpc=arguments["first_step_kpc"],
        n_cloud_species=arguments["n_cloud_species"],
        cloud_mass_range=(arguments["cloud_mass_min"], arguments["cloud_mass_max"]),
        cloud_alpha=arguments["cloud_alpha"],
        integrator_mode=arguments["integrator_mode"],
        integrator_rtol=arguments["integrator_rtol"],
        integrator_atol=arguments["integrator_atol"],
        integrator_max_steps=arguments["integrator_max_steps"],
        observable_set=arguments["observable_set"],
        dndv_num_bins=arguments["dndv_num_bins"],
        dndv_vmin_kms=arguments["dndv_vmin_kms"],
        dndv_vmax_kms=arguments["dndv_vmax_kms"],
        dndv_kernel_sigma_kms=arguments["dndv_kernel_sigma_kms"],
        dndv_kernel=arguments["dndv_kernel"],
        dndv_kernel_truncate_sigma=arguments["dndv_kernel_truncate_sigma"],
        eta_e_parameterization=arguments["eta_e_parameterization"],
        energy_coordinate=arguments["energy_coordinate"],
        eta_e_softcap_center=arguments["eta_e_softcap_center"],
        eta_e_softcap_sigma=arguments["eta_e_softcap_sigma"],
        eta_e_softcap_transition=arguments["eta_e_softcap_transition"],
        failure_policy=arguments["failure_policy"],
        stall_velocity_floor_kms=arguments["stall_velocity_floor_kms"],
        stall_velocity_transition_kms=arguments["stall_velocity_transition_kms"],
        stall_radius_sigma_kpc=arguments["stall_radius_sigma_kpc"],
        stall_radius_transition_kpc=arguments["stall_radius_transition_kpc"],
    )


def _run_wind_solution(theta: np.ndarray, arguments: dict[str, Any]):
    config = WindConfig(
        solver_max_step_kpc=float(arguments["step_kpc"]),
        solver_first_step_kpc=float(arguments["first_step_kpc"]),
    )
    model = WindModel(
        v_circ=float(arguments["v_circ"]),
        SFR=float(arguments["sfr"]),
        eta_M=float(theta[0]),
        eta_M_cold=float(theta[1]),
        eta_E=float(theta[2]),
        r_star_kpc=float(arguments["r_star_kpc"]),
        r_max_kpc=float(arguments["r_max_kpc"]),
        cloud_mass_range=(float(arguments["cloud_mass_min"]), float(arguments["cloud_mass_max"])),
        cloud_alpha=float(arguments["cloud_alpha"]),
        N_cloud_species=int(arguments["n_cloud_species"]),
        config=config,
    )
    return model.run()


def _velocity_bins(names: np.ndarray) -> np.ndarray:
    bins = []
    for name in names:
        text = str(name)
        bins.append(float(text.split("@", 1)[1].split("km/s", 1)[0]))
    return np.asarray(bins, dtype=float)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_chain_context(
    path: Path,
    div_idx: int,
    window: np.ndarray,
    velocity_bins: np.ndarray,
    residuals: np.ndarray,
    theta: np.ndarray,
    energy_coordinate: np.ndarray,
    divergent: np.ndarray,
    accept_prob: np.ndarray,
    num_steps: np.ndarray,
    energy: np.ndarray,
    potential_energy: np.ndarray,
    components: np.ndarray,
    component_index: dict[str, int],
    nuts_coordinate: np.ndarray,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), constrained_layout=True)
    ax = axes[0]
    ax.plot(window, theta[window, 0], marker="o", label=r"$\eta_M$")
    ax.plot(window, energy_coordinate[window, 1], marker="o", label=r"$\eta_{M,cold}/\eta_M$")
    ax.plot(window, energy_coordinate[window, 2], marker="o", label=r"$\eta_E/\eta_M$")
    ax.axvline(div_idx, color="tab:red", lw=1.5, alpha=0.75)
    ax.set_ylabel("physical / ratio coordinates")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.plot(window, components[window, component_index["chi2"]], marker="o", label=r"$\chi^2$")
    ax.plot(window, components[window, component_index["prior_chi2"]], marker="o", label="prior chi2")
    ax.plot(window, components[window, component_index["total_objective"]], marker="o", label="objective")
    ax.axvline(div_idx, color="tab:red", lw=1.5, alpha=0.75)
    ax.set_ylabel("posterior components")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2]
    dz = np.full(window.shape, np.nan, dtype=float)
    for out_i, sample_i in enumerate(window):
        if sample_i > 0:
            dz[out_i] = np.linalg.norm(nuts_coordinate[sample_i] - nuts_coordinate[sample_i - 1])
    ax.plot(window, num_steps[window], marker="o", label="NUTS steps")
    ax2 = ax.twinx()
    ax2.plot(window, accept_prob[window], marker="s", color="tab:green", label="accept prob")
    ax2.plot(window, dz, marker="^", color="tab:purple", label=r"$|\Delta z|$")
    ax.axvline(div_idx, color="tab:red", lw=1.5, alpha=0.75)
    ax.set_ylabel("NUTS steps")
    ax2.set_ylabel("accept / whitened step")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    ax = axes[3]
    image = residuals[window].T
    vmax = max(1.0, float(np.nanpercentile(np.abs(image), 98.0)))
    mesh = ax.pcolormesh(
        window,
        velocity_bins,
        image,
        shading="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.axvline(div_idx, color="black", lw=1.2, alpha=0.7)
    ax.set_xlabel("post-warmup sample index")
    ax.set_ylabel("velocity bin [km/s]")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("normalized residual")
    fig.suptitle(f"Observed-window chain context around divergent transition {div_idx}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_physical_tracks(
    path: Path,
    div_idx: int,
    window: np.ndarray,
    solutions: dict[int, Any],
    velocity_bins: np.ndarray,
    observed_log_profile: np.ndarray,
    residuals: np.ndarray,
    predicted_log_profiles: np.ndarray,
    divergent: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    color_values = np.linspace(0.15, 0.85, window.size)
    cmap = plt.get_cmap("viridis")

    for color_value, sample_i in zip(color_values, window):
        sol = solutions[sample_i]
        is_div = bool(divergent[sample_i])
        color = "tab:red" if sample_i == div_idx else cmap(color_value)
        lw = 2.5 if sample_i == div_idx else 1.1
        alpha = 1.0 if sample_i == div_idx else 0.62
        label = f"{sample_i}{' div' if is_div else ''}"

        axes[0, 0].plot(sol.r, sol.v, color=color, lw=lw, alpha=alpha, label=label)
        axes[0, 1].plot(sol.r, sol.T, color=color, lw=lw, alpha=alpha)
        for species in range(sol.v_cl.shape[0]):
            axes[1, 0].plot(sol.r, sol.v_cl[species], color=color, lw=0.8, alpha=alpha * 0.55)
        axes[1, 1].plot(sol.r, sol.M_cloud_tot, color=color, lw=lw, alpha=alpha)
        axes[2, 0].plot(velocity_bins, predicted_log_profiles[sample_i], color=color, lw=lw, alpha=alpha)
        axes[2, 1].plot(velocity_bins, residuals[sample_i], color=color, lw=lw, alpha=alpha)

    axes[2, 0].plot(velocity_bins, observed_log_profile, color="black", lw=2.0, ls="--", label="truth")
    axes[2, 1].axhline(0.0, color="black", lw=0.8)
    axes[0, 0].set_ylabel("hot velocity [km/s]")
    axes[0, 1].set_ylabel("hot temperature [K]")
    axes[1, 0].set_ylabel("cloud velocity [km/s]")
    axes[1, 1].set_ylabel(r"total cloud mass [$M_\odot$]")
    axes[2, 0].set_ylabel(r"log binned $dN/dv$")
    axes[2, 1].set_ylabel("normalized residual")
    for ax in axes[:2].flat:
        ax.set_xlabel("radius [kpc]")
        ax.set_xlim(left=0.3)
    for ax in axes[2]:
        ax.set_xlabel("velocity bin [km/s]")
    axes[0, 1].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(loc="best", fontsize=8, ncol=2)
    axes[2, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Physical tracks and log-profile sequence around divergence {div_idx}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _launch_temperature(theta: np.ndarray, config: WindConfig) -> np.ndarray:
    eta_m = np.maximum(theta[:, 0], 1e-30)
    eta_e = np.maximum(theta[:, 2], 1e-30)
    mach0 = 1.0 + float(config.sonic_point_offset)
    specific_energy = (eta_e / eta_m) * (config.E_SN / (config.mstar * 1.98847e33))
    v_star = np.sqrt(specific_energy) * (1.0 / ((gamma - 1.0) * mach0) + 0.5) ** (-0.5)
    cs_sq = v_star * v_star / (mach0 * mach0)
    return config.mu * mp * cs_sq / (gamma * kb)


def main() -> None:
    args = parse_args()
    npz_path = Path(args.input_npz)
    output_dir = Path(args.output_dir) if args.output_dir else npz_path.with_name("observed_window_divergence_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(npz_path)
    arguments = metadata["arguments"]
    model = _build_inference_model(arguments)

    with np.load(npz_path, allow_pickle=True) as data:
        realization_index = int(args.realization_index)
        theta = np.asarray(data["samples_theta"][realization_index], dtype=float)
        energy_coordinate = np.asarray(data["samples_energy_coordinate"][realization_index], dtype=float)
        nuts_coordinate = np.asarray(data["samples_nuts_coordinate"][realization_index], dtype=float)
        unconstrained = np.asarray(data["samples_unconstrained"][realization_index], dtype=float)
        divergent = np.asarray(data["sample_diverging"][realization_index], dtype=bool)
        accept_prob = np.asarray(data["sample_accept_prob"][realization_index], dtype=float)
        num_steps = np.asarray(data["sample_num_steps"][realization_index], dtype=float)
        energy = np.asarray(data["sample_energy"][realization_index], dtype=float)
        potential_energy = np.asarray(data["sample_potential_energy"][realization_index], dtype=float)
        observed = np.asarray(data["observed"][realization_index], dtype=float)
        sigma = np.asarray(data["sigma"][realization_index], dtype=float)
        components = np.asarray(data["posterior_components_samples"][realization_index], dtype=float)
        component_names = [str(name) for name in data["posterior_component_names"]]
        observable_names = np.asarray(data["observable_names"])

    velocity_bins = _velocity_bins(observable_names)
    component_index = {name: idx for idx, name in enumerate(component_names)}
    predictions = np.asarray(jax.vmap(model._predict_theta_fn)(jnp.asarray(theta, dtype=jnp.float64)))
    residuals = (predictions - observed[None, :]) / np.maximum(sigma[None, :], 1e-300)
    divergent_indices = np.flatnonzero(divergent)

    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    config = WindConfig()
    launch_temperature = _launch_temperature(theta, config)
    for idx in range(theta.shape[0]):
        max_residual_bin = int(np.nanargmax(np.abs(residuals[idx])))
        previous = idx - 1
        row = {
            "sample_index": int(idx),
            "divergent": bool(divergent[idx]),
            "eta_M": float(theta[idx, 0]),
            "eta_M_cold": float(theta[idx, 1]),
            "eta_E": float(theta[idx, 2]),
            "eta_M_cold_over_eta_M": float(energy_coordinate[idx, 1]),
            "eta_E_over_eta_M": float(energy_coordinate[idx, 2]),
            "T_star_K": float(launch_temperature[idx]),
            "accept_prob": float(accept_prob[idx]),
            "num_steps": float(num_steps[idx]),
            "energy": float(energy[idx]),
            "potential_energy": float(potential_energy[idx]),
            "kinetic_energy": float(energy[idx] - potential_energy[idx]),
            "chi2": float(components[idx, component_index["chi2"]]),
            "prior_chi2": float(components[idx, component_index["prior_chi2"]]),
            "total_objective": float(components[idx, component_index["total_objective"]]),
            "trajectory_status_code": float(components[idx, component_index["trajectory_status_code"]]),
            "stall_penalty": float(components[idx, component_index["stall_penalty"]]),
            "numerical_failure_penalty": float(components[idx, component_index["numerical_failure_penalty"]]),
            "hard_invalid_penalty": float(components[idx, component_index["hard_invalid_penalty"]]),
            "rms_residual": float(np.sqrt(np.nanmean(residuals[idx] ** 2))),
            "max_abs_residual": float(np.nanmax(np.abs(residuals[idx]))),
            "max_abs_residual_velocity_kms": float(velocity_bins[max_residual_bin]),
            "max_abs_residual_value": float(residuals[idx, max_residual_bin]),
            "delta_nuts_coordinate_norm": float(np.linalg.norm(nuts_coordinate[idx] - nuts_coordinate[previous]))
            if previous >= 0
            else np.nan,
            "delta_unconstrained_norm": float(np.linalg.norm(unconstrained[idx] - unconstrained[previous]))
            if previous >= 0
            else np.nan,
            "delta_theta_relative_norm": float(
                np.linalg.norm((theta[idx] - theta[previous]) / np.maximum(theta[previous], 1e-30))
            )
            if previous >= 0
            else np.nan,
        }
        rows.append(row)
        if previous >= 0:
            delta_residual = residuals[idx] - residuals[previous]
            delta_bin = int(np.nanargmax(np.abs(delta_residual)))
            transition_rows.append(
                {
                    "sample_index": int(idx),
                    "divergent": bool(divergent[idx]),
                    "previous_index": int(previous),
                    "delta_nuts_coordinate_norm": row["delta_nuts_coordinate_norm"],
                    "delta_unconstrained_norm": row["delta_unconstrained_norm"],
                    "delta_theta_relative_norm": row["delta_theta_relative_norm"],
                    "delta_total_objective": float(
                        components[idx, component_index["total_objective"]]
                        - components[previous, component_index["total_objective"]]
                    ),
                    "delta_chi2": float(
                        components[idx, component_index["chi2"]] - components[previous, component_index["chi2"]]
                    ),
                    "max_delta_residual_sigma": float(delta_residual[delta_bin]),
                    "max_delta_residual_velocity_kms": float(velocity_bins[delta_bin]),
                }
            )

    _write_csv(output_dir / "chain_sample_metrics.csv", rows)
    _write_csv(output_dir / "chain_transition_metrics.csv", transition_rows)

    figure_paths: dict[str, list[str]] = {"chain_context": [], "physical_tracks": []}
    sequence_summaries: list[dict[str, Any]] = []
    for div_idx in divergent_indices:
        start = max(0, int(div_idx) - int(args.pre))
        stop = min(theta.shape[0], int(div_idx) + int(args.post) + 1)
        window = np.arange(start, stop, dtype=int)
        chain_path = output_dir / f"divergence_{div_idx:03d}_chain_context.png"
        _plot_chain_context(
            chain_path,
            int(div_idx),
            window,
            velocity_bins,
            residuals,
            theta,
            energy_coordinate,
            divergent,
            accept_prob,
            num_steps,
            energy,
            potential_energy,
            components,
            component_index,
            nuts_coordinate,
        )
        figure_paths["chain_context"].append(str(chain_path))

        solutions = {int(sample_idx): _run_wind_solution(theta[sample_idx], arguments) for sample_idx in window}
        physical_path = output_dir / f"divergence_{div_idx:03d}_physical_tracks.png"
        _plot_physical_tracks(
            physical_path,
            int(div_idx),
            window,
            solutions,
            velocity_bins,
            observed,
            residuals,
            predictions,
            divergent,
        )
        figure_paths["physical_tracks"].append(str(physical_path))

        pre = int(div_idx) - 1
        if pre >= 0:
            delta_residual = residuals[div_idx] - residuals[pre]
            delta_bin = int(np.nanargmax(np.abs(delta_residual)))
            sequence_summaries.append(
                {
                    "divergent_index": int(div_idx),
                    "window": [int(window[0]), int(window[-1])],
                    "previous_index": int(pre),
                    "delta_nuts_coordinate_norm": float(np.linalg.norm(nuts_coordinate[div_idx] - nuts_coordinate[pre])),
                    "delta_unconstrained_norm": float(np.linalg.norm(unconstrained[div_idx] - unconstrained[pre])),
                    "delta_theta_relative_norm": float(
                        np.linalg.norm((theta[div_idx] - theta[pre]) / np.maximum(theta[pre], 1e-30))
                    ),
                    "max_delta_residual_sigma": float(delta_residual[delta_bin]),
                    "max_delta_residual_velocity_kms": float(velocity_bins[delta_bin]),
                    "divergent_theta": theta[div_idx].tolist(),
                    "divergent_energy_coordinate": energy_coordinate[div_idx].tolist(),
                    "previous_theta": theta[pre].tolist(),
                    "previous_energy_coordinate": energy_coordinate[pre].tolist(),
                    "divergent_objective": float(components[div_idx, component_index["total_objective"]]),
                    "previous_objective": float(components[pre, component_index["total_objective"]]),
                    "divergent_chi2": float(components[div_idx, component_index["chi2"]]),
                    "previous_chi2": float(components[pre, component_index["chi2"]]),
                    "divergent_num_steps": float(num_steps[div_idx]),
                    "divergent_accept_prob": float(accept_prob[div_idx]),
                }
            )

    summary = {
        "input_npz": str(npz_path),
        "output_dir": str(output_dir),
        "num_samples": int(theta.shape[0]),
        "divergent_indices": divergent_indices.astype(int).tolist(),
        "notes": [
            "These figures inspect retained post-warmup Markov-chain states.",
            "The internal NUTS leapfrog/tree states that triggered each divergence were not saved in this run.",
        ],
        "sequence_summaries": sequence_summaries,
        "figure_paths": figure_paths,
        "csv": {
            "sample_metrics": str(output_dir / "chain_sample_metrics.csv"),
            "transition_metrics": str(output_dir / "chain_transition_metrics.csv"),
        },
    }
    with (output_dir / "divergence_inspection_summary.json").open("w", encoding="ascii") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
