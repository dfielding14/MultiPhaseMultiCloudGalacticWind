#!/usr/bin/env python3
"""Generate model-method intuition figures for Paper 2."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiphasegalacticwind.config import WindConfig
from multiphasegalacticwind.constants import Myr, Msun, Z_solar, gamma, kb, km, kpc, mp, muH, yr
from multiphasegalacticwind.core_physics import setup_cloud_powerlaw_distribution
from multiphasegalacticwind.observables import calculate_column_density_by_species
from multiphasegalacticwind.topaz_cooling import load_cooling_table, tcool_P_topaz


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "paper", "paper2_inference_validation", "figures")


@dataclass(frozen=True)
class FiducialCloudSetup:
    """Fiducial cloud-spectrum parameters used in the Paper 2 method figures."""

    M_cl_min_msun: float = 1.0
    M_cl_max_msun: float = 1.0e6
    N_cl: int = 13
    alpha_cl: float = 2.0
    eta_M_cold: float = 1.0
    sfr_msun_per_yr: float = 20.0
    r_star_kpc: float = 0.3
    v_cl0_kms: float = 100.0


@dataclass(frozen=True)
class FiducialWindSetup(FiducialCloudSetup):
    """Fiducial solved wind used for the remaining Section 2 method figures."""

    v_circ_kms: float = 150.0
    eta_M: float = 0.1
    eta_M_cold: float = 0.2
    eta_E: float = 1.0
    r_max_kpc: float = 30.0
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    solver_max_step_kpc: float = 0.3


def configure_matplotlib_for_paper(use_tex: bool = True) -> None:
    """Apply the paper-style Matplotlib configuration used by the atlas figures."""
    if use_tex and shutil.which("latex") is None:
        raise RuntimeError("LaTeX rendering requested, but no 'latex' executable was found on PATH.")

    matplotlib.rcParams.update(
        {
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "lines.dash_capstyle": "round",
            "text.usetex": bool(use_tex),
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    if use_tex:
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible figure generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for generated paper figures")
    parser.add_argument("--no-usetex", action="store_true", help="Disable Matplotlib LaTeX text rendering")
    return parser.parse_args()


def build_fiducial_distribution(setup: FiducialCloudSetup) -> dict[str, np.ndarray]:
    """Return the fiducial cloud mass grid and injection rates."""
    M_cloud0, eta_M_cold_i, Mdot_cold0, Ndot_cloud0 = setup_cloud_powerlaw_distribution(
        np.log10(setup.M_cl_min_msun),
        np.log10(setup.M_cl_max_msun),
        setup.N_cl,
        alpha_cloud=setup.alpha_cl,
        eta_M_cold_tot=setup.eta_M_cold,
        SFR=setup.sfr_msun_per_yr * Msun / yr,
    )
    return {
        "M_cloud0": M_cloud0,
        "eta_M_cold_i": eta_M_cold_i,
        "Mdot_cold0": Mdot_cold0,
        "Ndot_cloud0": Ndot_cloud0,
    }


def injection_profile(r_kpc: np.ndarray, setup: FiducialCloudSetup, config: WindConfig) -> np.ndarray:
    """Compute the implemented cumulative radial cloud injection factor."""
    r_inj_kpc = config.cold_cloud_injection_radial_extent_frac * setup.r_star_kpc
    power = config.cold_cloud_injection_radial_power
    return np.where(r_kpc < r_inj_kpc, (r_kpc / r_inj_kpc) ** power, 1.0)


def plot_cloud_mass_spectrum(output_dir: str, setup: FiducialCloudSetup) -> str:
    """Plot the initial cloud mass bins, mass fluxes, and number injection rates."""
    arrays = build_fiducial_distribution(setup)
    M_msun = arrays["M_cloud0"] / Msun
    eta_i = arrays["eta_M_cold_i"]
    Mdot_msun_per_yr = arrays["Mdot_cold0"] / Msun * yr
    Ndot_per_yr = arrays["Ndot_cloud0"] * yr
    indices = np.arange(1, setup.N_cl + 1)

    path = os.path.join(output_dir, "cloud_mass_spectrum_initialization.png")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(indices, M_msun, marker="o", color="tab:blue", lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("cloud species index")
    ax.set_ylabel(r"$M_{{\rm cl},i}\ [M_\odot]$")
    ax.set_title("representative masses")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.bar(indices, eta_i / setup.eta_M_cold, color="tab:orange", alpha=0.78)
    ax.axhline(1.0 / setup.N_cl, color="black", lw=1.2, ls="--", label=r"$1/N_{\rm cl}$")
    ax.set_xlabel("cloud species index")
    ax.set_ylabel(r"$f_{M,i}$")
    ax.set_title("cold mass fraction")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(indices, Mdot_msun_per_yr, color="tab:red", marker="s", ms=3.5, lw=1.1, alpha=0.8)
    ax2.set_ylabel(r"$\dot{M}_{{\rm cold},i}\ [M_\odot\,{\rm yr}^{-1}]$")

    ax = axes[2]
    ax.plot(M_msun, Ndot_per_yr, marker="o", color="tab:green", lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{{\rm cl},i}\ [M_\odot]$")
    ax.set_ylabel(r"$\dot{N}_{{\rm cl},i}\ [{\rm yr}^{-1}]$")
    ax.set_title("cloud number injection")
    ax.grid(alpha=0.25)

    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def plot_cloud_injection_number_density(output_dir: str, setup: FiducialCloudSetup) -> str:
    """Plot the radial injection factor and initial cloud number densities."""
    config = WindConfig()
    arrays = build_fiducial_distribution(setup)
    M_msun = arrays["M_cloud0"] / Msun
    Ndot_cloud0 = arrays["Ndot_cloud0"]

    r_kpc = np.logspace(np.log10(0.05), np.log10(30.0), 500)
    profile = injection_profile(r_kpc, setup, config)
    r_cm = r_kpc * kpc
    v_cl0 = setup.v_cl0_kms * km
    n_cloud_cm3 = Ndot_cloud0[None, :] * profile[:, None] / (config.Omwind * r_cm[:, None] ** 2 * v_cl0)
    n_cloud_kpc3 = n_cloud_cm3 * kpc**3
    representative_indices = [0, setup.N_cl // 2, setup.N_cl - 1]
    colors = ["tab:blue", "tab:purple", "tab:red"]

    r_over_rstar = r_kpc / setup.r_star_kpc

    path = os.path.join(output_dir, "cloud_injection_number_density.png")
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.45), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(r_over_rstar, profile, color="black", lw=1.8)
    ax.axvline(
        config.cold_cloud_injection_radial_extent_frac,
        color="tab:red",
        lw=1.2,
        ls="--",
        label=rf"$r_{{\rm inj}}={config.cold_cloud_injection_radial_extent_frac:.2f}r_\star$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$I(r)$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    mass_labels = [r"1", r"10^3", r"10^6"]
    for index, color, mass_label in zip(representative_indices, colors, mass_labels):
        ax.plot(
            r_over_rstar,
            n_cloud_kpc3[:, index],
            color=color,
            lw=1.7,
            label=rf"$M_{{\rm cl}}={mass_label}\,M_\odot$",
        )
    ax.axvline(config.cold_cloud_injection_radial_extent_frac, color="0.35", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r/r_\star$")
    ax.set_ylabel(r"$n_{\rm cl}\ [{\rm kpc}^{-3}]$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def representative_species_indices(n_species: int) -> list[int]:
    """Return low-, intermediate-, and high-mass species indices."""
    return [0, n_species // 2, n_species - 1]


def representative_species_styles(solution) -> tuple[list[int], list[str], list[str]]:
    """Return representative species indices, colors, and TeX mass labels."""
    indices = representative_species_indices(solution.model.N_cloud_species)
    colors = ["tab:blue", "tab:purple", "tab:red"]
    masses = solution.model.M_cloud0[indices] / Msun
    labels = [rf"$M_{{\rm cl,0}}={mass:.0e}\,M_\odot$" for mass in masses]
    labels = [label.replace("e+00", "").replace("e+03", r"\times 10^3").replace("e+06", r"\times 10^6") for label in labels]
    return indices, colors, labels


def build_fiducial_solution(setup: FiducialWindSetup):
    """Run the fiducial solved wind used by the dynamic Section 2 figures."""
    from multiphasegalacticwind import WindModel

    model = WindModel(
        SFR=setup.sfr_msun_per_yr,
        v_circ=setup.v_circ_kms,
        eta_M=setup.eta_M,
        eta_M_cold=setup.eta_M_cold,
        eta_E=setup.eta_E,
        r_star_kpc=setup.r_star_kpc,
        r_max_kpc=setup.r_max_kpc,
        cloud_mass_range=(setup.M_cl_min_msun, setup.M_cl_max_msun),
        cloud_alpha=setup.alpha_cl,
        N_cloud_species=setup.N_cl,
        rtol=setup.rtol,
        atol=setup.atol,
        solver_max_step_kpc=setup.solver_max_step_kpc,
    )
    solution = model.run()
    if not solution.sol.success:
        raise RuntimeError(f"Fiducial Section 2 model failed: {solution.sol.message}")
    return solution


def compute_cloud_exchange_diagnostics(solution) -> dict[str, np.ndarray]:
    """Compute cloud exchange quantities using formulas matched to the JAX RHS."""
    model = solution.model
    config = model.config
    state = np.asarray(solution.sol.y, dtype=float)
    n_species = model.N_cloud_species

    r = np.asarray(solution.sol.t, dtype=float)
    v_wind = state[0]
    rho_wind = state[1]
    pressure = state[2]
    rhoz_wind = state[3]
    M_cloud = state[4 : 4 + n_species]
    v_cloud = state[4 + n_species : 4 + 2 * n_species]
    Z_cloud = state[4 + 2 * n_species : 4 + 3 * n_species]

    safe_r = np.maximum(r, 1.0e-30)
    safe_rho_wind = np.maximum(rho_wind, 1.0e-60)
    z_wind = rhoz_wind / safe_rho_wind
    t_wind = (pressure / kb) * (config.mu * mp / safe_rho_wind)
    t_mix = np.sqrt(np.maximum(t_wind[None, :] * config.T_cl, 1.0e-30))
    z_mix = np.sqrt(np.maximum(z_wind[None, :] * Z_cloud, 0.0))

    table = load_cooling_table(config.topaz_cooling_table_path)
    t_cool_layer = np.asarray(
        tcool_P_topaz(t_mix, pressure[None, :] / kb, z_mix / Z_solar, config.mu, table=table),
        dtype=float,
    )
    t_cool_layer = np.where(t_cool_layer < 0.0, 1.0e10 * Myr, t_cool_layer)

    rho_cloud = pressure * (config.mu * mp) / (kb * config.T_cl)
    chi = rho_cloud / safe_rho_wind
    chi_safe = np.maximum(chi, 1.0e-30)

    rho_cloud_safe = np.maximum(rho_cloud, 1.0e-60)
    r_cloud = np.where(
        M_cloud > 0.0,
        np.power(np.maximum(M_cloud / ((4.0 * np.pi / 3.0) * rho_cloud_safe[None, :]), 0.0), 1.0 / 3.0),
        0.0,
    )
    r_cloud_safe = np.where(r_cloud > 0.0, r_cloud, np.inf)

    v_rel = v_wind[None, :] - v_cloud
    v_turb = config.f_turb0 * np.abs(v_rel) * np.power(chi_safe[None, :], config.TurbulentVelocityChiPower)
    v_turb_cold = v_turb * np.power(chi_safe[None, :], config.ColdTurbulenceChiPower)
    ksi = r_cloud / (np.maximum(v_turb, 1.0e-10) * np.maximum(t_cool_layer, 1.0e-30))

    area_boost = config.geometric_factor * np.power(chi_safe[None, :], config.CoolingAreaChiPower)
    ksi_factor = np.where(ksi < 1.0, np.power(np.maximum(ksi, 0.0), 0.5), np.power(np.maximum(ksi, 0.0), 0.25))
    cloud_active = M_cloud > config.M_cloud_min

    Mdot_grow = np.where(
        cloud_active,
        config.Mdot_coefficient
        * 3.0
        * M_cloud
        * v_turb
        * area_boost
        / (r_cloud_safe * chi_safe[None, :])
        * ksi_factor,
        0.0,
    )
    Mdot_loss = np.where(
        cloud_active,
        config.Mdot_coefficient * 3.0 * (-M_cloud) * v_turb_cold / r_cloud_safe,
        0.0,
    )
    Mdot_cloud = Mdot_grow + Mdot_loss

    r0 = model.r_star_kpc * kpc
    injection_radius = config.cold_cloud_injection_radial_extent_frac * r0
    injection_factor = np.where(
        r < injection_radius,
        np.power(np.maximum(safe_r / max(injection_radius, 1.0e-30), 0.0), config.cold_cloud_injection_radial_power),
        1.0,
    )
    v_cloud_floor = max(config.v_cloud_min * km, 1.0e-10)
    v_cloud_safe = np.maximum(v_cloud, v_cloud_floor)
    Ndot_cloud = np.asarray(model.Ndot_cloud0, dtype=float)[:, None] * injection_factor[None, :]
    number_density_cloud = Ndot_cloud / (config.Omwind * safe_r[None, :] ** 2 * v_cloud_safe)

    p_dot_ram = 0.5 * config.drag_coeff * rho_wind[None, :] * np.pi * v_rel * np.abs(v_rel) * r_cloud * r_cloud
    p_dot_transfer = v_wind[None, :] * Mdot_grow + v_cloud * Mdot_loss

    Phir = (model.v_circ * km) ** 2 * np.log(safe_r)
    vBsq_wind = 0.5 * v_wind * v_wind + (gamma / (gamma - 1.0)) * pressure / safe_rho_wind + Phir
    cs_cl_sq = gamma * kb * config.T_cl / (config.mu * mp)
    vBsq_cl = 0.5 * v_cloud * v_cloud + cs_cl_sq / (gamma - 1.0) + Phir[None, :]
    e_dot_transfer = vBsq_wind[None, :] * Mdot_grow + vBsq_cl * Mdot_loss

    mass_source_species = -number_density_cloud * Mdot_cloud
    momentum_source_species = -number_density_cloud * (p_dot_transfer + p_dot_ram)
    energy_source_species = -number_density_cloud * (e_dot_transfer + p_dot_ram * v_wind[None, :])

    return {
        "r_kpc": r / kpc,
        "r_over_rstar": r / (model.r_star_kpc * kpc),
        "M_cloud_msun": M_cloud / Msun,
        "v_cloud_kms": v_cloud / km,
        "rho_cloud": rho_cloud,
        "chi": chi,
        "t_wind": t_wind,
        "t_mix": t_mix,
        "z_mix": z_mix,
        "t_cool_layer": t_cool_layer,
        "r_cloud_cm": r_cloud,
        "v_turb": v_turb,
        "ksi": ksi,
        "Mdot_grow": Mdot_grow,
        "Mdot_loss": Mdot_loss,
        "Mdot_cloud": Mdot_cloud,
        "number_density_cloud": number_density_cloud,
        "mass_source_species": mass_source_species,
        "momentum_source_species": momentum_source_species,
        "energy_source_species": energy_source_species,
    }


def plot_mixing_layer_cooling_parameter(output_dir: str, solution, diagnostics: dict[str, np.ndarray]) -> str:
    """Plot the cooling parameter xi for representative cloud species."""
    indices, colors, labels = representative_species_styles(solution)
    x = diagnostics["r_kpc"]
    path = os.path.join(output_dir, "mixing_layer_cooling_parameter.png")
    fig, ax = plt.subplots(figsize=(3.45, 3.0), constrained_layout=True)
    for index, color, label in zip(indices, colors, labels):
        y = np.where(diagnostics["ksi"][index] > 0.0, diagnostics["ksi"][index], np.nan)
        ax.plot(x, y, color=color, lw=1.7, label=label)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r\ [{\rm kpc}]$")
    ax.set_ylabel(r"$\xi$")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def plot_cloud_species_mass_velocity_evolution(output_dir: str, solution, diagnostics: dict[str, np.ndarray]) -> str:
    """Plot cloud mass survival and acceleration for representative species."""
    indices, colors, labels = representative_species_styles(solution)
    x = diagnostics["r_kpc"]
    path = os.path.join(output_dir, "cloud_species_mass_velocity_evolution.png")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)

    for index, color, label in zip(indices, colors, labels):
        mass_ratio = diagnostics["M_cloud_msun"][index] / max(diagnostics["M_cloud_msun"][index, 0], 1.0e-30)
        mass_ratio = np.where(mass_ratio > 0.0, mass_ratio, np.nan)
        axes[0].plot(x, mass_ratio, color=color, lw=1.7, label=label)
        axes[1].plot(x, diagnostics["v_cloud_kms"][index], color=color, lw=1.7, label=label)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$r\ [{\rm kpc}]$")
    axes[0].set_ylabel(r"$M_{\rm cl}/M_{\rm cl,0}$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5, loc="best")

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$r\ [{\rm kpc}]$")
    axes[1].set_ylabel(r"$v_{\rm cl}\ [{\rm km\,s}^{-1}]$")
    axes[1].grid(alpha=0.25)

    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def _plot_signed_source_axis(ax: plt.Axes, x: np.ndarray, total: np.ndarray, species: np.ndarray, indices, colors, labels, ylabel: str) -> None:
    """Plot one signed source-term axis with a robust symlog scale."""
    finite = np.abs(np.concatenate([total[np.isfinite(total)], species[indices][np.isfinite(species[indices])]]))
    positive = finite[finite > 0.0]
    max_abs = float(np.nanmax(positive)) if positive.size else 1.0
    exponent = int(np.floor(np.log10(max_abs))) if max_abs > 0.0 else 0
    scale = 10.0**exponent
    total_scaled = total / scale
    species_scaled = species / scale
    max_abs_scaled = max_abs / scale
    linthresh = max(max_abs_scaled * 1.0e-2, 1.0e-6)
    ax.plot(x, total_scaled, color="black", lw=1.9, label="total")
    for index, color, label in zip(indices, colors, labels):
        ax.plot(x, species_scaled[index], color=color, lw=1.2, alpha=0.85, label=label)
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(-1.2 * max_abs_scaled, 1.2 * max_abs_scaled)
    ax.set_xlabel(r"$r\ [{\rm kpc}]$")
    ax.set_ylabel(ylabel.replace("@EXP@", str(exponent)))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25)


def plot_cloud_coupling_source_terms(output_dir: str, solution, diagnostics: dict[str, np.ndarray]) -> str:
    """Plot signed cloud back-reaction source terms for the hot phase."""
    indices, colors, labels = representative_species_styles(solution)
    x = diagnostics["r_kpc"]
    path = os.path.join(output_dir, "cloud_coupling_source_terms.png")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=True)

    source_specs = [
        ("mass_source_species", r"$S_\rho\ [10^{@EXP@}\,{\rm g\,cm}^{-3}\,{\rm s}^{-1}]$"),
        ("momentum_source_species", r"$S_p\ [10^{@EXP@}\,{\rm dyn\,cm}^{-3}]$"),
        ("energy_source_species", r"$S_e\ [10^{@EXP@}\,{\rm erg\,cm}^{-3}\,{\rm s}^{-1}]$"),
    ]
    for ax, (key, ylabel) in zip(axes, source_specs):
        species = diagnostics[key]
        total = np.sum(species, axis=0)
        _plot_signed_source_axis(ax, x, total, species, indices, colors, labels, ylabel)
    axes[0].legend(frameon=False, fontsize=7.0, loc="best")

    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def plot_cloud_species_dndv_contributions(output_dir: str, solution) -> str:
    """Plot total and species-resolved column-density velocity profiles."""
    indices, colors, labels = representative_species_styles(solution)
    r_max = min(30.0, float(solution.r[-1]))
    v_grid, dndv = calculate_column_density_by_species(
        solution,
        r_min_kpc=solution.model.r_star_kpc,
        r_max_kpc=r_max,
    )
    path = os.path.join(output_dir, "cloud_species_dndv_contributions.png")
    fig, ax = plt.subplots(figsize=(3.45, 3.0), constrained_layout=True)
    for index, color, label in zip(indices, colors, labels):
        y = np.where(dndv["species"][index] > 0.0, dndv["species"][index], np.nan)
        ax.plot(v_grid, y, color=color, lw=1.2, alpha=0.85, label=label)
    total = np.where(dndv["total"] > 0.0, dndv["total"], np.nan)
    ax.plot(v_grid, total, color="black", lw=1.9, label="total")
    ax.set_yscale("log")
    ax.set_xlabel(r"$v_{\rm cl}\ [{\rm km\,s}^{-1}]$")
    ax.set_ylabel(r"$dN/dv\ [{\rm cm}^{-2}({\rm km\,s}^{-1})^{-1}]$")
    ax.legend(frameon=False, fontsize=7.0, loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def main() -> None:
    """Generate all Paper 2 model-method figures."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configure_matplotlib_for_paper(use_tex=not args.no_usetex)
    setup = FiducialCloudSetup()
    wind_setup = FiducialWindSetup()
    solution = build_fiducial_solution(wind_setup)
    diagnostics = compute_cloud_exchange_diagnostics(solution)
    paths = [
        plot_cloud_mass_spectrum(args.output_dir, setup),
        plot_cloud_injection_number_density(args.output_dir, setup),
        plot_mixing_layer_cooling_parameter(args.output_dir, solution, diagnostics),
        plot_cloud_species_mass_velocity_evolution(args.output_dir, solution, diagnostics),
        plot_cloud_coupling_source_terms(args.output_dir, solution, diagnostics),
        plot_cloud_species_dndv_contributions(args.output_dir, solution),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
