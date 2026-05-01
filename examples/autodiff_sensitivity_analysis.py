#!/usr/bin/env python
"""
Autodiff-based parameter sensitivity analysis using JAX.

This script computes local sensitivities (Jacobians) of dN/dv moments (M0, M1, M2) and
profiles with respect to (eta_M, eta_M_cold, eta_E) around a fiducial model.

Unlike brute-force grid sweeps, this uses JAX automatic differentiation.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind.constants import Msun, Z_solar, gamma, kb, kpc, mp, yr
from multiphasegalacticwind.jax_physics import JaxWindParams, integrate_wind_rk4_scan
from multiphasegalacticwind.topaz_cooling import get_jax_table_arrays, load_cooling_table

jax.config.update("jax_enable_x64", True)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "autodiff_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autodiff sensitivities for eta_M, eta_M_cold, eta_E")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--r-max-kpc", type=float, default=30.0)
    parser.add_argument("--step-kpc", type=float, default=0.02)
    parser.add_argument("--first-step-kpc", type=float, default=1e-12)

    parser.add_argument("--sfr", type=float, default=20.0)
    parser.add_argument("--v-circ", type=float, default=150.0)
    parser.add_argument("--r-star-kpc", type=float, default=0.3)

    parser.add_argument("--eta-m", type=float, default=0.1)
    parser.add_argument("--eta-m-cold", type=float, default=0.2)
    parser.add_argument("--eta-e", type=float, default=1.0)

    parser.add_argument("--n-cloud-species", type=int, default=13)
    parser.add_argument("--cloud-mass-min", type=float, default=1.0)
    parser.add_argument("--cloud-mass-max", type=float, default=1e6)
    parser.add_argument("--cloud-alpha", type=float, default=2.0)

    parser.add_argument("--t-cloud", type=float, default=1e4)
    parser.add_argument("--z-hot-over-z-solar", type=float, default=10 ** -0.5)
    parser.add_argument("--z-cloud-over-z-solar", type=float, default=0.3)

    parser.add_argument("--half-opening-angle", type=float, default=float(np.pi / 2.0))
    parser.add_argument("--v-cloud-init", type=float, default=100.0)
    parser.add_argument("--v-cloud-min", type=float, default=1.0)

    parser.add_argument("--cooling-area-chi-power", type=float, default=0.5)
    parser.add_argument("--cold-turbulence-chi-power", type=float, default=-0.5)
    parser.add_argument("--turbulent-velocity-chi-power", type=float, default=0.0)
    parser.add_argument("--geometric-factor", type=float, default=1.0)
    parser.add_argument("--mdot-coefficient", type=float, default=1.0 / 3.0)
    parser.add_argument("--cooling-factor", type=float, default=1.0)
    parser.add_argument("--drag-coeff", type=float, default=0.5)
    parser.add_argument("--f-turb0", type=float, default=0.1)
    parser.add_argument("--m-cloud-min", type=float, default=1e-2)

    parser.add_argument("--mu", type=float, default=0.62)
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--sonic-point-offset", type=float, default=1e-6)

    parser.add_argument("--cold-cloud-injection-radial-power", type=float, default=6.0)
    parser.add_argument("--cold-cloud-injection-radial-extent-frac", type=float, default=1.33)

    parser.add_argument("--e-sn", type=float, default=1e51)
    parser.add_argument("--mstar", type=float, default=100.0)

    parser.add_argument("--topaz-cooling-table-path", default=None)
    return parser.parse_args()


def build_radius_grid(r_start: float, r_end: float, first_step: float, step_cap: float) -> np.ndarray:
    points = [float(r_start)]
    h = max(first_step, 1e-12 * kpc)
    h_cap = max(step_cap, h)

    while h < h_cap and points[-1] + h < r_end:
        points.append(points[-1] + h)
        h = min(h * 2.0, h_cap)

    current = points[-1]
    if current >= r_end:
        return np.asarray(points, dtype=float)

    n_uniform = max(1, int(np.ceil((r_end - current) / h_cap)))
    tail = np.linspace(current, r_end, n_uniform + 1, dtype=float)[1:]
    return np.concatenate([np.asarray(points, dtype=float), tail])


def setup_cloud_distribution(
    n_cloud_species: int,
    cloud_mass_min: float,
    cloud_mass_max: float,
    cloud_alpha: float,
):
    log_min = np.log10(cloud_mass_min)
    log_max = np.log10(cloud_mass_max)

    M_cloud0 = jnp.logspace(log_min, log_max, n_cloud_species) * Msun
    dN_dlogM = M_cloud0 ** (1.0 - cloud_alpha)

    if n_cloud_species == 1:
        mass_weights = jnp.asarray([1.0], dtype=jnp.float64)
    else:
        dlogM = (log_max - log_min) / (n_cloud_species - 1)
        N_rel = dN_dlogM * dlogM
        M_in_bin = M_cloud0 * N_rel
        mass_weights = M_in_bin / jnp.sum(M_in_bin)

    return M_cloud0, mass_weights


def make_sensitivity_functions(args: argparse.Namespace):
    r_star = args.r_star_kpc * kpc
    r_end = args.r_max_kpc * kpc
    r_grid = jnp.asarray(
        build_radius_grid(
            r_star,
            r_end,
            args.first_step_kpc * kpc,
            args.step_kpc * kpc,
        ),
        dtype=jnp.float64,
    )

    mu = float(args.mu)
    Omwind = min(4.0 * np.pi * (1.0 - np.cos(args.half_opening_angle)), 4.0 * np.pi)
    Mach0 = 1.0 + float(args.sonic_point_offset)

    M_cloud0, mass_weights = setup_cloud_distribution(
        args.n_cloud_species,
        args.cloud_mass_min,
        args.cloud_mass_max,
        args.cloud_alpha,
    )

    table = load_cooling_table(args.topaz_cooling_table_path)
    topaz_log10_temperature, topaz_primordial_cooling_cgs, topaz_metal_cooling_cgs = get_jax_table_arrays(
        table=table
    )

    SFR_cgs = float(args.sfr) * Msun / yr
    v_circ_cgs = float(args.v_circ) * 1e5

    def build_state_and_params(theta: jnp.ndarray):
        eta_M, eta_M_cold, eta_E = theta

        Mdot_hot = eta_M * SFR_cgs
        Edot_hot = eta_E * (args.e_sn / (args.mstar * Msun)) * SFR_cgs

        v_star = jnp.sqrt(Edot_hot / Mdot_hot) * (
            1.0 / ((gamma - 1.0) * Mach0) + 0.5
        ) ** (-0.5)
        rho_star = Mdot_hot / (Omwind * r_star * r_star * v_star)
        P_star = rho_star * v_star * v_star / (Mach0 * Mach0 * gamma)

        eta_M_cold_array = eta_M_cold * mass_weights
        Mdot_cold0 = eta_M_cold_array * SFR_cgs
        Ndot_cloud0 = Mdot_cold0 / M_cloud0

        y0 = jnp.zeros((4 + 3 * args.n_cloud_species,), dtype=jnp.float64)
        y0 = y0.at[0].set(v_star)
        y0 = y0.at[1].set(rho_star)
        y0 = y0.at[2].set(P_star)
        y0 = y0.at[3].set(rho_star * args.z_hot_over_z_solar * Z_solar)
        y0 = y0.at[4 : 4 + args.n_cloud_species].set(M_cloud0)
        y0 = y0.at[4 + args.n_cloud_species : 4 + 2 * args.n_cloud_species].set(args.v_cloud_init * 1e5)
        y0 = y0.at[4 + 2 * args.n_cloud_species :].set(args.z_cloud_over_z_solar * Z_solar)

        source_volume = 4.0 / 3.0 * np.pi * r_star**3
        params = JaxWindParams(
            v_circ=v_circ_cgs,
            Ndot_cloud0=Ndot_cloud0,
            T_cloud=float(args.t_cloud),
            injection_radius=float(args.cold_cloud_injection_radial_extent_frac) * r_star,
            injection_power=float(args.cold_cloud_injection_radial_power),
            M_cloud_min=float(args.m_cloud_min) * Msun,
            CoolingAreaChiPower=float(args.cooling_area_chi_power),
            ColdTurbulenceChiPower=float(args.cold_turbulence_chi_power),
            TurbulentVelocityChiPower=float(args.turbulent_velocity_chi_power),
            geometric_factor=float(args.geometric_factor),
            Mdot_coefficient=float(args.mdot_coefficient),
            A_mix=1.0,
            beta_chi_mix=0.0,
            mixing_chi_pivot=100.0,
            Cooling_Factor=float(args.cooling_factor),
            drag_coeff=float(args.drag_coeff),
            f_turb0=float(args.f_turb0),
            Omwind=float(Omwind),
            mu=float(args.mu),
            redshift=float(args.redshift),
            Z_hot_over_Z_solar=float(args.z_hot_over_z_solar),
            v_cloud_min=float(args.v_cloud_min),
            r0=r_star,
            Edot_per_Vol=Edot_hot / source_volume,
            Mdot_per_Vol=Mdot_hot / source_volume,
            topaz_log10_temperature=topaz_log10_temperature,
            topaz_primordial_cooling_cgs=topaz_primordial_cooling_cgs,
            topaz_metal_cooling_cgs=topaz_metal_cooling_cgs,
        )
        return y0, params

    @jax.jit
    def trajectory(theta: jnp.ndarray):
        y0, params = build_state_and_params(theta)
        return integrate_wind_rk4_scan(r_grid, y0, params)

    def summary_observables(theta: jnp.ndarray):
        """
        Return [M0, M1, M2] of the dN/dv distribution.

        Moments are computed from the monotonic mapping integral:
        M_n = ∫ v^n (dN/dv) dv = ∫ v^n n_H dr.
        """
        y = trajectory(theta)
        _, params = build_state_and_params(theta)

        n_species = args.n_cloud_species
        r_kpc = r_grid / kpc

        M_cloud = y[:, 4 : 4 + n_species]
        v_cloud = y[:, 4 + n_species : 4 + 2 * n_species]

        injection = jnp.where(
            r_grid < params.injection_radius,
            (r_grid / jnp.maximum(params.injection_radius, 1e-30)) ** params.injection_power,
            1.0,
        )
        Ndot = params.Ndot_cloud0[None, :] * injection[:, None]
        v_cloud_safe = jnp.maximum(v_cloud, params.v_cloud_min * 1e5)
        denom = params.Omwind * (r_grid[:, None] ** 2) * v_cloud_safe
        rho_cloud = Ndot * M_cloud / jnp.maximum(denom, 1e-60)
        n_H = rho_cloud / (1.4 * mp)

        active = M_cloud >= params.M_cloud_min
        radial_window = ((r_kpc >= 0.5) & (r_kpc <= args.r_max_kpc))[:, None]
        weight = (active & radial_window).astype(jnp.float64)

        v_kms = v_cloud / 1e5
        nH_eff = n_H * weight

        M0 = jnp.sum(jnp.trapezoid(nH_eff, r_grid, axis=0))
        M1 = jnp.sum(jnp.trapezoid(nH_eff * v_kms, r_grid, axis=0))
        M2 = jnp.sum(jnp.trapezoid(nH_eff * v_kms * v_kms, r_grid, axis=0))

        return jnp.asarray([M0, M1, M2], dtype=jnp.float64)

    def velocity_profile(theta: jnp.ndarray):
        y = trajectory(theta)
        return y[:, 0] / 1e5

    def temperature_profile(theta: jnp.ndarray):
        y = trajectory(theta)
        return y[:, 2] / (y[:, 1] / (mu * mp)) / kb

    return r_grid, trajectory, summary_observables, velocity_profile, temperature_profile


def plot_summary_sensitivities(
    output_dir: str,
    outputs0: np.ndarray,
    jacobian_outputs: np.ndarray,
    theta0: np.ndarray,
):
    output_labels = ["M0(dN/dv)", "M1(dN/dv)", "M2(dN/dv)"]
    param_labels = ["eta_M", "eta_M_cold", "eta_E"]

    denom = np.where(np.abs(outputs0) > 1e-60, outputs0, np.nan)
    elasticity = jacobian_outputs * theta0[np.newaxis, :] / denom[:, np.newaxis]
    finite = np.isfinite(elasticity)
    vmax = np.max(np.abs(elasticity[finite])) if np.any(finite) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    im = ax.imshow(elasticity, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(param_labels)))
    ax.set_xticklabels(param_labels)
    ax.set_yticks(np.arange(len(output_labels)))
    ax.set_yticklabels(output_labels)

    for i in range(elasticity.shape[0]):
        for j in range(elasticity.shape[1]):
            ax.text(j, i, f"{elasticity[i, j]:+.3f}", ha="center", va="center", color="black")

    ax.set_title("Local log-sensitivities: d ln(output) / d ln(parameter)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("elasticity")

    base = os.path.join(output_dir, "autodiff_summary_elasticity_matrix")
    fig.savefig(base + ".png", dpi=220)
    plt.close(fig)


def plot_profile_sensitivities(
    output_dir: str,
    r_grid: np.ndarray,
    baseline_v: np.ndarray,
    baseline_T: np.ndarray,
    jac_v: np.ndarray,
    jac_T: np.ndarray,
    theta0: np.ndarray,
):
    r_kpc = r_grid / kpc
    param_labels = ["eta_M", "eta_M_cold", "eta_E"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    sens_v = jac_v * theta0[np.newaxis, :] / np.maximum(np.abs(baseline_v[:, np.newaxis]), 1e-30)
    sens_T = jac_T * theta0[np.newaxis, :] / np.maximum(np.abs(baseline_T[:, np.newaxis]), 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(r_kpc, baseline_v, color="black", lw=1.8)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title("Baseline velocity profile")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(r_kpc, baseline_T, color="black", lw=1.8)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("T [K]")
    ax.set_yscale("log")
    ax.set_title("Baseline temperature profile")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for i, (label, color) in enumerate(zip(param_labels, colors)):
        ax.plot(r_kpc, sens_v[:, i], label=label, lw=1.5, color=color)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"$\partial \ln v / \partial \ln p$")
    ax.set_title("Velocity sensitivity profiles")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for i, (label, color) in enumerate(zip(param_labels, colors)):
        ax.plot(r_kpc, sens_T[:, i], label=label, lw=1.5, color=color)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel(r"$\partial \ln T / \partial \ln p$")
    ax.set_title("Temperature sensitivity profiles")
    ax.grid(alpha=0.25)

    base = os.path.join(output_dir, "autodiff_profile_sensitivities")
    fig.savefig(base + ".png", dpi=220)
    plt.close(fig)


def plot_linearization_check(
    output_dir: str,
    summary_fn,
    jacobian_outputs: np.ndarray,
    theta0: np.ndarray,
):
    labels = ["M0", "M1", "M2"]

    perturb = np.array(
        [
            [0.05, 0.00, 0.00],
            [0.00, 0.05, 0.00],
            [0.00, 0.00, 0.05],
            [-0.05, 0.00, 0.00],
            [0.00, -0.05, 0.00],
            [0.00, 0.00, -0.05],
            [0.04, 0.03, -0.02],
        ],
        dtype=float,
    )

    y0 = np.asarray(summary_fn(jnp.asarray(theta0)))
    actual = []
    linear = []

    for frac in perturb:
        delta_theta = theta0 * frac
        y_actual = np.asarray(summary_fn(jnp.asarray(theta0 + delta_theta)))
        y_lin = y0 + jacobian_outputs @ delta_theta
        actual.append(y_actual)
        linear.append(y_lin)

    actual = np.asarray(actual)
    linear = np.asarray(linear)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.scatter(actual[:, i], linear[:, i], s=40, alpha=0.9)
        low = min(np.min(actual[:, i]), np.min(linear[:, i]))
        high = max(np.max(actual[:, i]), np.max(linear[:, i]))
        ax.plot([low, high], [low, high], "k--", lw=1.0)
        ax.set_xlabel(f"actual {label}")
        ax.set_ylabel(f"linearized {label}")
        ax.set_title(label)
        ax.grid(alpha=0.25)

    fig.suptitle("Local linearization check (autodiff Jacobian)")
    base = os.path.join(output_dir, "autodiff_linearization_check")
    fig.savefig(base + ".png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    r_grid, trajectory_fn, summary_fn, velocity_profile_fn, temperature_profile_fn = make_sensitivity_functions(args)

    theta0 = jnp.asarray([args.eta_m, args.eta_m_cold, args.eta_e], dtype=jnp.float64)

    # Warm-up compile for runtime clarity.
    _ = trajectory_fn(theta0).block_until_ready()

    y0 = np.asarray(trajectory_fn(theta0))
    outputs0 = np.asarray(summary_fn(theta0))

    jac_outputs = np.asarray(jax.jacrev(summary_fn)(theta0))
    jac_v = np.asarray(jax.jacrev(velocity_profile_fn)(theta0))
    jac_T = np.asarray(jax.jacrev(temperature_profile_fn)(theta0))

    baseline_v = y0[:, 0] / 1e5
    baseline_T = y0[:, 2] / (y0[:, 1] / (args.mu * mp)) / kb

    print("Baseline outputs (dN/dv moments):")
    print(f"  M0 = {outputs0[0]:.6e}")
    print(f"  M1 = {outputs0[1]:.6e}")
    print(f"  M2 = {outputs0[2]:.6e}")

    print("\nLocal derivatives d(output)/d(parameter):")
    print("rows: [M0, M1, M2], cols: [eta_M, eta_M_cold, eta_E]")
    print(jac_outputs)

    plot_summary_sensitivities(args.output_dir, outputs0, jac_outputs, np.asarray(theta0))
    plot_profile_sensitivities(
        args.output_dir,
        np.asarray(r_grid),
        baseline_v,
        baseline_T,
        jac_v,
        jac_T,
        np.asarray(theta0),
    )
    plot_linearization_check(args.output_dir, summary_fn, jac_outputs, np.asarray(theta0))

    np.savez(
        os.path.join(args.output_dir, "autodiff_sensitivity_data.npz"),
        r_grid=np.asarray(r_grid),
        theta0=np.asarray(theta0),
        baseline_state=y0,
        baseline_outputs=outputs0,
        jac_outputs=jac_outputs,
        jac_velocity_profile=jac_v,
        jac_temperature_profile=jac_T,
    )

    print(f"\nSaved outputs to {args.output_dir}/")


if __name__ == "__main__":
    main()
