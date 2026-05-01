"""Inference utilities for fitting wind parameters to observed dN/dv moments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import WindConfig
from .constants import Msun, Z_solar, gamma, kb, kpc, mp, yr
from .jax_physics import (
    JaxWindParams,
    has_diffrax,
    integrate_wind_rk2_scan,
    integrate_wind_rk3_scan,
    integrate_wind_rk4_scan,
    integrate_wind_tsit5,
)
from .topaz_cooling import get_jax_table_arrays, load_cooling_table

jax.config.update("jax_enable_x64", True)


def build_radius_grid(r_start_cgs: float, r_end_cgs: float, first_step_cgs: float, step_cap_cgs: float) -> np.ndarray:
    """Construct a radial grid with geometric warm-up near launch and fixed outer spacing."""
    points = [float(r_start_cgs)]
    h = max(float(first_step_cgs), 1e-12 * kpc)
    h_cap = max(float(step_cap_cgs), h)

    while h < h_cap and points[-1] + h < r_end_cgs:
        points.append(points[-1] + h)
        h = min(h * 2.0, h_cap)

    current = points[-1]
    if current >= r_end_cgs:
        return np.asarray(points, dtype=float)

    n_uniform = max(1, int(np.ceil((r_end_cgs - current) / h_cap)))
    tail = np.linspace(current, r_end_cgs, n_uniform + 1, dtype=float)[1:]
    return np.concatenate([np.asarray(points, dtype=float), tail])


def build_covariance(std: Sequence[float], corr: np.ndarray | None = None) -> np.ndarray:
    """Build a covariance matrix from 1-sigma errors and optional correlation matrix."""
    sigma = np.asarray(std, dtype=float)
    if sigma.ndim != 1:
        raise ValueError(f"std must be a 1D array of standard deviations, got shape {sigma.shape}")
    n_obs = int(sigma.size)
    if n_obs < 1:
        raise ValueError("std must contain at least one entry")
    if np.any(sigma <= 0.0):
        raise ValueError("All standard deviations must be positive")

    if corr is None:
        corr_m = np.eye(n_obs, dtype=float)
    else:
        corr_m = np.asarray(corr, dtype=float)
        if corr_m.shape != (n_obs, n_obs):
            raise ValueError(f"corr must be shape ({n_obs}, {n_obs}), got {corr_m.shape}")
        if not np.allclose(corr_m, corr_m.T, atol=1e-12):
            raise ValueError("corr must be symmetric")
        if not np.allclose(np.diag(corr_m), 1.0, atol=1e-12):
            raise ValueError("corr diagonal must be exactly 1")

    cov = np.outer(sigma, sigma) * corr_m
    return stabilize_covariance(cov)


def stabilize_covariance(cov: np.ndarray, min_eig: float = 1e-20) -> np.ndarray:
    """Project covariance to symmetric positive definite with eigenvalue floor."""
    cov = np.asarray(cov, dtype=float)
    sym = 0.5 * (cov + cov.T)
    sym = np.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)
    n = sym.shape[0]
    scale = max(float(np.max(np.abs(sym))), 1.0)
    sym_scaled = sym / scale
    min_eig_scaled = float(min_eig) / scale

    jitter = max(min_eig_scaled, 1e-16)
    for _ in range(6):
        try:
            eigvals, eigvecs = np.linalg.eigh(sym_scaled)
            eigvals_clipped = np.clip(np.nan_to_num(eigvals, nan=min_eig_scaled, posinf=1e200), min_eig_scaled, 1e200)
            eigvecs = np.nan_to_num(eigvecs, nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
                spd_scaled = (eigvecs * eigvals_clipped[np.newaxis, :]) @ eigvecs.T
            spd_scaled = np.nan_to_num(spd_scaled, nan=0.0, posinf=1e200, neginf=0.0)
            spd = spd_scaled * scale
            return 0.5 * (spd + spd.T)
        except np.linalg.LinAlgError:
            sym_scaled = sym_scaled + jitter * np.eye(n)
            jitter *= 10.0

    return np.eye(n) * max(min_eig, 1e-12)


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation matrix."""
    cov = np.asarray(cov, dtype=float)
    diag = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass
class MAPFitResult:
    success: bool
    message: str
    n_iter: int
    start_index: int
    num_starts: int
    unconstrained_theta_map: np.ndarray
    log_theta_map: np.ndarray
    theta_map: np.ndarray
    predicted_moments: np.ndarray
    observed_moments: np.ndarray
    nlp: float
    chi2: float
    gradient_norm_inf: float
    hessian_log: np.ndarray
    covariance_log: np.ndarray
    covariance_theta: np.ndarray
    correlation_theta: np.ndarray
    hessian_unconstrained: np.ndarray
    covariance_unconstrained: np.ndarray


@dataclass
class HMCResult:
    samples_unconstrained: np.ndarray
    samples_log: np.ndarray
    samples_theta: np.ndarray
    samples_nuts_coordinate: np.ndarray | None
    sample_diverging: np.ndarray | None
    sample_accept_prob: np.ndarray | None
    sample_num_steps: np.ndarray | None
    sample_energy: np.ndarray | None
    sample_potential_energy: np.ndarray | None
    sampler: str
    num_chains: int
    acceptance_rate: float
    acceptance_rate_per_chain: np.ndarray | None
    final_step_size: float
    num_divergent: int
    num_steps_mean: float
    num_steps_max: float
    tree_depth_mean: float
    mean_log_posterior: float
    energy_mean: float
    energy_var: float
    potential_energy_mean: float
    bfmi: float | None
    r_hat: np.ndarray | None
    ess_bulk: np.ndarray | None
    covariance_log: np.ndarray
    covariance_theta: np.ndarray
    correlation_theta: np.ndarray
    nuts_chain_method: str | None = None
    nuts_dense_mass: bool | None = None
    nuts_max_tree_depth: int | None = None
    nuts_coordinate: str | None = None
    num_divergent_per_chain: np.ndarray | None = None
    bfmi_per_chain: np.ndarray | None = None


@dataclass
class PosteriorFitResult:
    map: MAPFitResult
    hmc: HMCResult
    observed_moments: np.ndarray
    covariance_moments: np.ndarray
    runtime_seconds: dict[str, float] | None = None


def summarize_parameter_degeneracies(correlation_theta: np.ndarray, names: Sequence[str]) -> list[str]:
    """Return pairwise degeneracy summary lines sorted by absolute correlation."""
    corr = np.asarray(correlation_theta, dtype=float)
    lines: list[tuple[float, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c_ij = float(corr[i, j])
            lines.append((abs(c_ij), f"{names[i]} vs {names[j]}: corr={c_ij:+.3f}"))
    lines.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in lines]


def resolve_nuts_chain_method(num_chains: int, chain_method: str) -> str:
    """Resolve NumPyro chain method with a robust auto mode."""
    method = chain_method.strip().lower()
    allowed = {"auto", "sequential", "parallel", "vectorized"}
    if method not in allowed:
        raise ValueError(f"Unsupported nuts_chain_method '{chain_method}'. Expected one of {sorted(allowed)}.")

    n_chains = max(1, int(num_chains))
    if method != "auto":
        return method
    if n_chains == 1:
        return "sequential"
    if jax.local_device_count() >= n_chains:
        return "parallel"
    return "vectorized"


class MomentInferenceModel:
    """Autodiff-enabled inference model for moment-based dN/dv observables."""

    PARAM_NAMES = ("eta_M", "eta_M_cold", "eta_E")
    EXPANDED_PARAMETER_NAMES = {
        "none": PARAM_NAMES,
        "a_mix": (*PARAM_NAMES, "A_mix"),
        "a_mix_beta_chi": (*PARAM_NAMES, "A_mix", "beta_chi_mix"),
    }
    POSTERIOR_COMPONENT_NAMES = (
        "chi2",
        "gaussian_chi2",
        "censored_chi2",
        "prior_chi2",
        "log_transform_jacobian",
        "transform_term",
        "eta_e_softcap_penalty",
        "validity_barrier",
        "early_fail_penalty",
        "stall_penalty",
        "numerical_failure_penalty",
        "hard_invalid_penalty",
        "total_objective",
        "valid",
        "trajectory_status_code",
        "soft_reach_radius_kpc",
        "first_invalid_r_kpc",
        "min_hot_velocity_kms",
    )
    _THETA_FLOOR = 1e-12
    _ETA_E_MAX = 0.999
    _OBS_MOMENTS3 = "m0_m1_m2"
    _OBS_SHAPE5 = "logm0_mean_sigma_skew_kurt"
    _OBS_DNDV_BINNED = "dndv_binned"
    _OBS_LOG_DNDV_BINNED = "log_dndv_binned"
    _DNDV_KERNEL_GAUSSIAN = 0
    _DNDV_KERNEL_TRUNCATED_GAUSSIAN = 1
    _DNDV_KERNEL_COMPACT_COSINE = 2
    _OBSERVABLE_SETS = {
        _OBS_MOMENTS3: ("M0", "M1", "M2"),
        _OBS_SHAPE5: ("logM0", "mean_v", "sigma_v", "skewness", "kurtosis"),
    }

    def __init__(
        self,
        sfr: float,
        r_star_kpc: float,
        v_circ: float = 150.0,
        r_max_kpc: float = 30.0,
        r_obs_min_kpc: float = 0.5,
        step_kpc: float = 0.02,
        first_step_kpc: float = 1e-12,
        n_cloud_species: int = 13,
        cloud_mass_range: tuple[float, float] = (1.0, 1e6),
        cloud_alpha: float = 2.0,
        integrator_mode: str = "rk4",
        integrator_rtol: float = 1e-5,
        integrator_atol: float = 1e-8,
        integrator_max_steps: int = 131072,
        observable_set: str = "m0_m1_m2",
        dndv_num_bins: int = 25,
        dndv_vmin_kms: float = 0.0,
        dndv_vmax_kms: float = 1200.0,
        dndv_kernel_sigma_kms: float | None = None,
        dndv_kernel: str = "gaussian",
        dndv_kernel_truncate_sigma: float = 3.0,
        log_dndv_low_signal_policy: str = "gaussian",
        log_dndv_censor_delta_log: float = 20.0,
        log_dndv_censor_transition: float = 0.25,
        log_dndv_censor_sigma: float = 0.50,
        log_dndv_censor_upper_margin: float = 1.0,
        eta_e_parameterization: str = "bounded",
        energy_coordinate: str = "eta_e",
        expanded_parameters: str = "none",
        eta_e_softcap_center: float = 1.0,
        eta_e_softcap_sigma: float = 0.10,
        eta_e_softcap_transition: float = 0.01,
        beta_chi_max_abs: float = 0.75,
        mixing_chi_pivot: float | None = None,
        failure_policy: str = "stalled_wind",
        stall_velocity_floor_kms: float = 0.0,
        stall_velocity_transition_kms: float = 50.0,
        stall_radius_sigma_kpc: float = 0.25,
        stall_radius_transition_kpc: float = 0.05,
        config: WindConfig | None = None,
        topaz_cooling_table_path: str | None = None,
    ) -> None:
        if sfr <= 0.0:
            raise ValueError("sfr must be positive")
        if r_star_kpc <= 0.0:
            raise ValueError("r_star_kpc must be positive")
        if r_max_kpc <= r_star_kpc:
            raise ValueError("r_max_kpc must exceed r_star_kpc")
        if n_cloud_species < 1:
            raise ValueError("n_cloud_species must be >= 1")
        if integrator_rtol <= 0.0 or integrator_atol <= 0.0:
            raise ValueError("integrator_rtol and integrator_atol must be positive")
        if integrator_max_steps < 8:
            raise ValueError("integrator_max_steps must be >= 8")

        self.sfr = float(sfr)
        self.r_star_kpc = float(r_star_kpc)
        self.v_circ = float(v_circ)
        self.r_max_kpc = float(r_max_kpc)
        self.r_obs_min_kpc = float(r_obs_min_kpc)
        self.integrator_mode = integrator_mode.strip().lower()
        self.integrator_rtol = float(integrator_rtol)
        self.integrator_atol = float(integrator_atol)
        self.integrator_max_steps = int(integrator_max_steps)
        self.dndv_num_bins = int(dndv_num_bins)
        self.dndv_vmin_kms = float(dndv_vmin_kms)
        self.dndv_vmax_kms = float(dndv_vmax_kms)
        self.dndv_kernel = self._resolve_dndv_kernel(dndv_kernel)
        self.dndv_kernel_truncate_sigma = float(dndv_kernel_truncate_sigma)
        self.log_dndv_low_signal_policy = self._resolve_log_dndv_low_signal_policy(log_dndv_low_signal_policy)
        self.log_dndv_censor_delta_log = float(log_dndv_censor_delta_log)
        self.log_dndv_censor_transition = float(log_dndv_censor_transition)
        self.log_dndv_censor_sigma = float(log_dndv_censor_sigma)
        self.log_dndv_censor_upper_margin = float(log_dndv_censor_upper_margin)
        self.observable_set = self._resolve_observable_set(observable_set)
        self.eta_e_parameterization = self._resolve_eta_e_parameterization(eta_e_parameterization)
        self.energy_coordinate = self._resolve_energy_coordinate(energy_coordinate)
        self.expanded_parameters = self._resolve_expanded_parameters(expanded_parameters)
        self.param_names = self.EXPANDED_PARAMETER_NAMES[self.expanded_parameters]
        self.theta_dim = len(self.param_names)
        self.beta_chi_max_abs = float(beta_chi_max_abs)
        self.mixing_chi_pivot = None if mixing_chi_pivot is None else float(mixing_chi_pivot)
        self.eta_e_softcap_center = float(eta_e_softcap_center)
        self.eta_e_softcap_sigma = float(eta_e_softcap_sigma)
        self.eta_e_softcap_transition = float(eta_e_softcap_transition)
        self.failure_policy = self._resolve_failure_policy(failure_policy)
        self.stall_velocity_floor_kms = float(stall_velocity_floor_kms)
        self.stall_velocity_transition_kms = float(stall_velocity_transition_kms)
        self.stall_radius_sigma_kpc = float(stall_radius_sigma_kpc)
        self.stall_radius_transition_kpc = float(stall_radius_transition_kpc)

        if self.integrator_mode not in {"rk2", "rk3", "rk4", "tsit5"}:
            raise ValueError("integrator_mode must be one of {'rk2', 'rk3', 'rk4', 'tsit5'}")
        if self.dndv_num_bins < 6:
            raise ValueError("dndv_num_bins must be >= 6")
        if self.dndv_vmax_kms <= self.dndv_vmin_kms:
            raise ValueError("dndv_vmax_kms must exceed dndv_vmin_kms")
        if self.energy_coordinate in {"eta_e_over_eta_m", "loading_ratios"} and self.eta_e_parameterization != "softcap":
            raise ValueError(f"energy_coordinate='{self.energy_coordinate}' requires eta_e_parameterization='softcap'")
        if self.beta_chi_max_abs <= 0.0:
            raise ValueError("beta_chi_max_abs must be positive")
        if self.mixing_chi_pivot is not None and (self.mixing_chi_pivot <= 0.0 or not np.isfinite(self.mixing_chi_pivot)):
            raise ValueError("mixing_chi_pivot must be positive and finite")
        if self.eta_e_softcap_center <= 0.0:
            raise ValueError("eta_e_softcap_center must be positive")
        if self.eta_e_softcap_sigma <= 0.0:
            raise ValueError("eta_e_softcap_sigma must be positive")
        if self.eta_e_softcap_transition <= 0.0:
            raise ValueError("eta_e_softcap_transition must be positive")
        if self.stall_velocity_transition_kms <= 0.0:
            raise ValueError("stall_velocity_transition_kms must be positive")
        if self.stall_radius_sigma_kpc <= 0.0:
            raise ValueError("stall_radius_sigma_kpc must be positive")
        if self.stall_radius_transition_kpc <= 0.0:
            raise ValueError("stall_radius_transition_kpc must be positive")

        self.dndv_bin_edges_kms = np.linspace(
            self.dndv_vmin_kms,
            self.dndv_vmax_kms,
            self.dndv_num_bins + 1,
            dtype=float,
        )
        self.dndv_bin_centers_kms = 0.5 * (self.dndv_bin_edges_kms[:-1] + self.dndv_bin_edges_kms[1:])
        if dndv_kernel_sigma_kms is None:
            dndv_kernel_sigma_kms = 0.5 * (self.dndv_bin_edges_kms[1] - self.dndv_bin_edges_kms[0])
        self.dndv_kernel_sigma_kms = float(dndv_kernel_sigma_kms)
        if self.dndv_kernel_sigma_kms <= 0.0:
            raise ValueError("dndv_kernel_sigma_kms must be positive")
        if self.dndv_kernel_truncate_sigma <= 0.0:
            raise ValueError("dndv_kernel_truncate_sigma must be positive")
        if self.log_dndv_censor_delta_log <= 0.0:
            raise ValueError("log_dndv_censor_delta_log must be positive")
        if self.log_dndv_censor_transition <= 0.0:
            raise ValueError("log_dndv_censor_transition must be positive")
        if self.log_dndv_censor_sigma <= 0.0:
            raise ValueError("log_dndv_censor_sigma must be positive")
        if self.log_dndv_censor_upper_margin < 0.0:
            raise ValueError("log_dndv_censor_upper_margin must be non-negative")

        if self.observable_set in {self._OBS_DNDV_BINNED, self._OBS_LOG_DNDV_BINNED}:
            prefix = "log_dN/dv" if self.observable_set == self._OBS_LOG_DNDV_BINNED else "dN/dv"
            self.observable_names = tuple(f"{prefix}@{v:.0f}km/s" for v in self.dndv_bin_centers_kms)
            self.default_observable_yscale = "log" if self.observable_set == self._OBS_DNDV_BINNED else "linear"
        else:
            self.observable_names = self._OBSERVABLE_SETS[self.observable_set]
            self.default_observable_yscale = "log" if self.observable_set == self._OBS_MOMENTS3 else "linear"
        self.observable_dim = len(self.observable_names)

        if self.integrator_mode == "tsit5" and not has_diffrax():
            raise ModuleNotFoundError(
                "integrator_mode='tsit5' requires Diffrax. Install with `python -m pip install diffrax`."
            )

        self.config = WindConfig() if config is None else config
        self.config.validate()
        if self.mixing_chi_pivot is None:
            self.mixing_chi_pivot = float(self.config.mixing_chi_pivot)

        self.r_star_cgs = self.r_star_kpc * kpc
        self.r_grid = jnp.asarray(
            build_radius_grid(
                self.r_star_cgs,
                self.r_max_kpc * kpc,
                first_step_kpc * kpc,
                step_kpc * kpc,
            ),
            dtype=jnp.float64,
        )

        log_m_min = np.log10(float(cloud_mass_range[0]))
        log_m_max = np.log10(float(cloud_mass_range[1]))
        m_cloud0 = np.logspace(log_m_min, log_m_max, int(n_cloud_species), dtype=float) * Msun
        dndlogm = m_cloud0 ** (1.0 - float(cloud_alpha))
        if int(n_cloud_species) == 1:
            mass_weights = np.array([1.0], dtype=float)
        else:
            dlogm = (log_m_max - log_m_min) / (int(n_cloud_species) - 1)
            n_rel = dndlogm * dlogm
            m_in_bin = m_cloud0 * n_rel
            mass_weights = m_in_bin / np.sum(m_in_bin)

        self.m_cloud0 = jnp.asarray(m_cloud0, dtype=jnp.float64)
        self.mass_weights = jnp.asarray(mass_weights, dtype=jnp.float64)

        table = load_cooling_table(topaz_cooling_table_path or self.config.topaz_cooling_table_path)
        self._topaz_arrays = get_jax_table_arrays(table=table)

        self._predict_theta_with_valid_fn = self._build_predictor()
        if self.integrator_mode in {"rk2", "rk3", "rk4"}:
            self._predict_theta_fn = jax.jit(lambda theta: self._predict_theta_with_valid_fn(theta)[0])
            self._predict_raw_theta_fn = jax.jit(lambda theta: self._predict_theta_with_valid_fn(theta)[1])
            self._predict_log_theta_fn = jax.jit(
                lambda prior_coordinate: self._predict_theta_fn(self._theta_from_prior_coordinate_jax(prior_coordinate))
            )
        else:
            self._predict_theta_fn = lambda theta: self._predict_theta_with_valid_fn(theta)[0]
            self._predict_raw_theta_fn = lambda theta: self._predict_theta_with_valid_fn(theta)[1]
            self._predict_log_theta_fn = lambda prior_coordinate: self._predict_theta_fn(
                self._theta_from_prior_coordinate_jax(prior_coordinate)
            )

    @classmethod
    def _resolve_observable_set(cls, observable_set: str) -> str:
        key = observable_set.strip().lower()
        aliases = {
            "m0_m1_m2": cls._OBS_MOMENTS3,
            "moments3": cls._OBS_MOMENTS3,
            "raw_moments": cls._OBS_MOMENTS3,
            "logm0_mean_sigma_skew_kurt": cls._OBS_SHAPE5,
            "shape5": cls._OBS_SHAPE5,
            "transformed_moments": cls._OBS_SHAPE5,
            "log_shape_moments": cls._OBS_SHAPE5,
            "dndv_binned": cls._OBS_DNDV_BINNED,
            "dndv": cls._OBS_DNDV_BINNED,
            "binned_dndv": cls._OBS_DNDV_BINNED,
            "full_dndv": cls._OBS_DNDV_BINNED,
            "log_dndv_binned": cls._OBS_LOG_DNDV_BINNED,
            "log_dndv": cls._OBS_LOG_DNDV_BINNED,
            "binned_log_dndv": cls._OBS_LOG_DNDV_BINNED,
            "log_profile": cls._OBS_LOG_DNDV_BINNED,
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported observable_set. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{observable_set}'."
            )
        return aliases[key]

    @staticmethod
    def _resolve_dndv_kernel(dndv_kernel: str) -> str:
        key = dndv_kernel.strip().lower()
        aliases = {
            "gaussian": "gaussian",
            "normal": "gaussian",
            "truncated_gaussian": "truncated_gaussian",
            "truncated-gaussian": "truncated_gaussian",
            "trunc_gaussian": "truncated_gaussian",
            "compact_cosine": "compact_cosine",
            "compact-cosine": "compact_cosine",
            "raised_cosine": "compact_cosine",
            "cosine": "compact_cosine",
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported dndv_kernel. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{dndv_kernel}'."
            )
        return aliases[key]

    @staticmethod
    def _resolve_log_dndv_low_signal_policy(policy: str) -> str:
        key = policy.strip().lower()
        aliases = {
            "gaussian": "gaussian",
            "none": "gaussian",
            "two_sided": "gaussian",
            "censored_upper": "censored_upper",
            "censored-upper": "censored_upper",
            "upper_limit": "censored_upper",
            "upper-limit": "censored_upper",
            "censored": "censored_upper",
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported log_dndv_low_signal_policy. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{policy}'."
            )
        return aliases[key]

    def log_dndv_censored_mask(self, observed_moments: Sequence[float]) -> np.ndarray:
        """Return low-signal log-profile bins treated as censored upper limits."""
        y_obs = np.asarray(observed_moments, dtype=float)
        if y_obs.shape != (self.observable_dim,):
            raise ValueError(
                "observed_moments must match observable_set size "
                f"{self.observable_dim} ({self.observable_names}), got shape {y_obs.shape}"
            )
        if self.observable_set != self._OBS_LOG_DNDV_BINNED or self.log_dndv_low_signal_policy != "censored_upper":
            return np.zeros((self.observable_dim,), dtype=bool)
        finite = np.isfinite(y_obs)
        if not np.any(finite):
            return np.zeros((self.observable_dim,), dtype=bool)
        threshold = float(np.nanmax(y_obs[finite]) - self.log_dndv_censor_delta_log)
        return finite & (y_obs <= threshold)

    def log_dndv_censor_upper_limits(self, observed_moments: Sequence[float]) -> np.ndarray:
        """Return log upper limits used by the censored low-signal likelihood."""
        y_obs = np.asarray(observed_moments, dtype=float)
        if y_obs.shape != (self.observable_dim,):
            raise ValueError(
                "observed_moments must match observable_set size "
                f"{self.observable_dim} ({self.observable_names}), got shape {y_obs.shape}"
            )
        return y_obs + self.log_dndv_censor_upper_margin

    def _observable_likelihood_setup(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
    ) -> dict[str, np.ndarray | bool]:
        """Prepare static Gaussian/censored likelihood arrays."""
        y_obs_np = np.asarray(observed_moments, dtype=float)
        cov = stabilize_covariance(np.asarray(covariance_moments, dtype=float))
        censored_mask = self.log_dndv_censored_mask(y_obs_np)
        use_censored = bool(np.any(censored_mask))

        gaussian_cov_inv = np.zeros_like(cov, dtype=float)
        if use_censored:
            detected_mask = ~censored_mask
            if np.any(detected_mask):
                detected_cov = stabilize_covariance(cov[np.ix_(detected_mask, detected_mask)])
                detected_cov_inv = np.linalg.inv(detected_cov)
                gaussian_cov_inv[np.ix_(detected_mask, detected_mask)] = detected_cov_inv
        else:
            gaussian_cov_inv = np.linalg.inv(cov)

        return {
            "covariance": cov,
            "gaussian_cov_inv": gaussian_cov_inv,
            "censored_mask": censored_mask.astype(float),
            "censored_upper": self.log_dndv_censor_upper_limits(y_obs_np),
            "use_censored": use_censored,
        }

    @staticmethod
    def _resolve_eta_e_parameterization(eta_e_parameterization: str) -> str:
        key = eta_e_parameterization.strip().lower()
        aliases = {
            "bounded": "bounded",
            "hardcap": "bounded",
            "sigmoid": "bounded",
            "softcap": "softcap",
            "eta_e_softcap": "softcap",
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported eta_e_parameterization. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{eta_e_parameterization}'."
            )
        return aliases[key]

    @staticmethod
    def _resolve_energy_coordinate(energy_coordinate: str) -> str:
        key = energy_coordinate.strip().lower()
        aliases = {
            "eta_e": "eta_e",
            "direct": "eta_e",
            "direct_eta_e": "eta_e",
            "eta_e_over_eta_m": "eta_e_over_eta_m",
            "specific_energy": "eta_e_over_eta_m",
            "q_e": "eta_e_over_eta_m",
            "loading_ratios": "loading_ratios",
            "cold_and_energy_ratios": "loading_ratios",
            "eta_m_cold_over_eta_m_and_eta_e_over_eta_m": "loading_ratios",
            "eta_e_over_eta_m_and_eta_m_cold_over_eta_m": "loading_ratios",
            "q_cold_q_e": "loading_ratios",
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported energy_coordinate. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{energy_coordinate}'."
            )
        return aliases[key]

    @staticmethod
    def _resolve_expanded_parameters(expanded_parameters: str) -> str:
        key = expanded_parameters.strip().lower()
        aliases = {
            "none": "none",
            "baseline": "none",
            "three_parameter": "none",
            "3param": "none",
            "a_mix": "a_mix",
            "amix": "a_mix",
            "mixing_amplitude": "a_mix",
            "a_mix_beta_chi": "a_mix_beta_chi",
            "amix_beta_chi": "a_mix_beta_chi",
            "a_mix+beta_chi": "a_mix_beta_chi",
            "mixing_amplitude_chi": "a_mix_beta_chi",
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported expanded_parameters. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{expanded_parameters}'."
            )
        return aliases[key]

    @staticmethod
    def _resolve_failure_policy(failure_policy: str) -> str:
        key = failure_policy.strip().lower()
        aliases = {
            "stalled_wind": "stalled_wind",
            "stalled": "stalled_wind",
            "smooth_stall": "stalled_wind",
            "hard_invalid": "hard_invalid",
            "legacy": "hard_invalid",
        }
        if key not in aliases:
            raise ValueError(
                "Unsupported failure_policy. "
                f"Expected one of {sorted(set(aliases.keys()))}, got '{failure_policy}'."
            )
        return aliases[key]

    @staticmethod
    def _observables_shape5_from_raw_moments(raw_moments):
        """Return [logM0, mean_v, sigma_v, skewness, kurtosis] from raw moments [M0..M4]."""
        raw = jnp.asarray(raw_moments, dtype=jnp.float64)
        m0 = jnp.maximum(raw[0], 1e-300)
        mean_v = raw[1] / m0
        second = raw[2] / m0
        var_v = second - mean_v * mean_v
        sigma_v = jnp.sqrt(jnp.maximum(var_v, 1e-24))

        third = raw[3] / m0
        fourth = raw[4] / m0
        mu3 = third - 3.0 * mean_v * second + 2.0 * mean_v**3
        mu4 = fourth - 4.0 * mean_v * third + 6.0 * mean_v * mean_v * second - 3.0 * mean_v**4

        sigma3 = jnp.maximum(sigma_v**3, 1e-24)
        sigma4 = jnp.maximum(sigma_v**4, 1e-24)
        skewness = mu3 / sigma3
        kurtosis = mu4 / sigma4
        return jnp.asarray([jnp.log(m0), mean_v, sigma_v, skewness, kurtosis], dtype=jnp.float64)

    @staticmethod
    def _observables_dndv_binned(
        v_kms,
        n_h_eff,
        r_grid,
        velocity_bins_kms,
        kernel_sigma_kms,
        kernel_kind: int = _DNDV_KERNEL_GAUSSIAN,
        truncate_sigma: float = 3.0,
    ):
        """Return kernel-binned dN/dv over a fixed velocity grid."""
        sigma = jnp.maximum(jnp.asarray(kernel_sigma_kms, dtype=jnp.float64), 1e-12)
        vb = jnp.asarray(velocity_bins_kms, dtype=jnp.float64)

        dv = (v_kms[:, :, None] - vb[None, None, :]) / sigma
        if int(kernel_kind) == MomentInferenceModel._DNDV_KERNEL_TRUNCATED_GAUSSIAN:
            cutoff = jnp.asarray(truncate_sigma, dtype=jnp.float64)
            norm = jax.lax.erf(cutoff / jnp.sqrt(2.0))
            kernel = jnp.exp(-0.5 * dv * dv) / (sigma * jnp.sqrt(2.0 * jnp.pi) * jnp.maximum(norm, 1e-12))
            kernel = jnp.where(jnp.abs(dv) <= cutoff, kernel, 0.0)
        elif int(kernel_kind) == MomentInferenceModel._DNDV_KERNEL_COMPACT_COSINE:
            abs_dv = jnp.abs(dv)
            kernel = 0.5 * (1.0 + jnp.cos(jnp.pi * dv)) / sigma
            kernel = jnp.where(abs_dv <= 1.0, kernel, 0.0)
        else:
            kernel = jnp.exp(-0.5 * dv * dv) / (sigma * jnp.sqrt(2.0 * jnp.pi))
        dndv_species = jnp.trapezoid(n_h_eff[:, :, None] * kernel, r_grid, axis=0)
        return jnp.sum(dndv_species, axis=0)

    def _default_theta_numpy(self) -> np.ndarray:
        """Return default physical parameters for the active inference dimension."""
        theta = [0.2, 0.2, 0.8]
        if self.expanded_parameters in {"a_mix", "a_mix_beta_chi"}:
            theta.append(1.0)
        if self.expanded_parameters == "a_mix_beta_chi":
            theta.append(0.0)
        return np.asarray(theta, dtype=float)

    def _default_prior_sigma_numpy(self) -> np.ndarray:
        """Return default prior widths in the active mixed prior coordinate."""
        sigma = [1.5, 1.5, 0.8]
        if self.expanded_parameters in {"a_mix", "a_mix_beta_chi"}:
            sigma.append(0.35)
        if self.expanded_parameters == "a_mix_beta_chi":
            sigma.append(0.25)
        return np.asarray(sigma, dtype=float)

    def _coerce_theta_numpy(self, theta: Sequence[float], *, name: str = "theta") -> np.ndarray:
        """Return a physical theta vector matching the active inference dimension."""
        arr = np.asarray(theta, dtype=float)
        if arr.shape == (self.theta_dim,):
            return arr
        if arr.shape == (3,) and self.theta_dim > 3:
            default = self._default_theta_numpy()
            default[:3] = arr
            return default
        raise ValueError(f"{name} must be length-{self.theta_dim}")

    def _prior_coordinate_from_theta_numpy(self, theta: Sequence[float]) -> np.ndarray:
        """Map physical theta to the mixed prior coordinate used by MAP/HMC."""
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.shape[-1] != self.theta_dim:
            if theta_arr.shape == (3,) and self.theta_dim > 3:
                theta_arr = self._coerce_theta_numpy(theta_arr)
            else:
                raise ValueError(f"theta must have length-{self.theta_dim} trailing dimension")
        coord = np.empty_like(theta_arr, dtype=float)
        coord[..., :3] = np.log(np.maximum(theta_arr[..., :3], self._THETA_FLOOR))
        if self.theta_dim >= 4:
            coord[..., 3] = np.log(np.maximum(theta_arr[..., 3], self._THETA_FLOOR))
        if self.theta_dim >= 5:
            coord[..., 4] = theta_arr[..., 4]
        return coord

    def _theta_from_prior_coordinate_numpy(self, prior_coordinate: np.ndarray) -> np.ndarray:
        """Map mixed prior coordinates back to physical theta."""
        coord = np.asarray(prior_coordinate, dtype=float)
        if coord.shape[-1] != self.theta_dim:
            raise ValueError(f"prior_coordinate must have length-{self.theta_dim} trailing dimension")
        out = np.empty_like(coord, dtype=float)
        out[..., :3] = np.exp(coord[..., :3])
        if self.theta_dim >= 4:
            out[..., 3] = np.exp(coord[..., 3])
        if self.theta_dim >= 5:
            out[..., 4] = coord[..., 4]
        return out

    def _theta_from_prior_coordinate_jax(self, prior_coordinate):
        """JAX map from mixed prior coordinates to physical theta."""
        coord = jnp.asarray(prior_coordinate, dtype=jnp.float64)
        values = [jnp.exp(coord[0]), jnp.exp(coord[1]), jnp.exp(coord[2])]
        if self.theta_dim >= 4:
            values.append(jnp.exp(coord[3]))
        if self.theta_dim >= 5:
            values.append(coord[4])
        return jnp.asarray(values, dtype=jnp.float64)

    def _coerce_prior_coordinate_numpy(
        self,
        prior_coordinate: Sequence[float] | None,
        *,
        default_theta: Sequence[float] | None = None,
        name: str = "prior_mean_log",
    ) -> np.ndarray:
        """Coerce a prior-coordinate vector, extending old length-3 inputs in expanded modes."""
        if prior_coordinate is None:
            theta = self._default_theta_numpy() if default_theta is None else self._coerce_theta_numpy(default_theta)
            return self._prior_coordinate_from_theta_numpy(theta)
        arr = np.asarray(prior_coordinate, dtype=float)
        if arr.shape == (self.theta_dim,):
            return arr
        if arr.shape == (3,) and self.theta_dim > 3:
            default = self._prior_coordinate_from_theta_numpy(
                self._default_theta_numpy() if default_theta is None else self._coerce_theta_numpy(default_theta)
            )
            default[:3] = arr
            return default
        raise ValueError(f"{name} must be length-{self.theta_dim}")

    def _coerce_prior_sigma_numpy(self, prior_sigma_log: Sequence[float], *, name: str = "prior_sigma_log") -> np.ndarray:
        """Coerce prior widths in the active mixed prior coordinate."""
        arr = np.asarray(prior_sigma_log, dtype=float)
        if arr.shape == (self.theta_dim,):
            return arr
        if arr.shape == (3,) and self.theta_dim > 3:
            default = self._default_prior_sigma_numpy()
            default[:3] = arr
            return default
        raise ValueError(f"{name} must be length-{self.theta_dim}")

    def _jac_theta_prior_coordinate_numpy(self, theta: Sequence[float]) -> np.ndarray:
        """Jacobian d theta / d prior-coordinate at a physical theta vector."""
        theta_arr = self._coerce_theta_numpy(theta)
        diag = np.ones((self.theta_dim,), dtype=float)
        diag[:3] = theta_arr[:3]
        if self.theta_dim >= 4:
            diag[3] = theta_arr[3]
        return np.diag(diag)

    def parameter_names(self) -> tuple[str, ...]:
        """Return physical parameter names for the active inference mode."""
        return tuple(self.param_names)

    def energy_coordinate_names(self) -> tuple[str, ...]:
        """Return labels after applying any active loading-ratio diagnostic coordinates."""
        names = list(self.param_names)
        if self.energy_coordinate == "eta_e_over_eta_m":
            names[2] = "eta_E_over_eta_M"
        elif self.energy_coordinate == "loading_ratios":
            names[1] = "eta_M_cold_over_eta_M"
            names[2] = "eta_E_over_eta_M"
        return tuple(names)

    def _theta_from_unconstrained_jax(self, unconstrained_theta):
        """Map unconstrained variables to physical theta."""
        u = jnp.asarray(unconstrained_theta, dtype=jnp.float64)
        eta_m = jax.nn.softplus(u[0]) + self._THETA_FLOOR
        eta_m_cold_coordinate = jax.nn.softplus(u[1]) + self._THETA_FLOOR
        if self.energy_coordinate == "loading_ratios":
            eta_m_cold_over_eta_m = eta_m_cold_coordinate
            eta_e_over_eta_m = jax.nn.softplus(u[2]) + self._THETA_FLOOR
            eta_m_cold = eta_m * eta_m_cold_over_eta_m
            eta_e = eta_m * eta_e_over_eta_m
            theta = [eta_m, eta_m_cold, eta_e]
            if self.theta_dim >= 4:
                theta.append(jax.nn.softplus(u[3]) + self._THETA_FLOOR)
            if self.theta_dim >= 5:
                theta.append(self.beta_chi_max_abs * jnp.tanh(u[4]))
            return jnp.asarray(theta, dtype=jnp.float64)
        if self.energy_coordinate == "eta_e_over_eta_m":
            eta_e_over_eta_m = jax.nn.softplus(u[2]) + self._THETA_FLOOR
            eta_e = eta_m * eta_e_over_eta_m
            theta = [eta_m, eta_m_cold_coordinate, eta_e]
            if self.theta_dim >= 4:
                theta.append(jax.nn.softplus(u[3]) + self._THETA_FLOOR)
            if self.theta_dim >= 5:
                theta.append(self.beta_chi_max_abs * jnp.tanh(u[4]))
            return jnp.asarray(theta, dtype=jnp.float64)

        if self.eta_e_parameterization == "bounded":
            eta_e = self._ETA_E_MAX * jax.nn.sigmoid(u[2]) + self._THETA_FLOOR
        else:
            eta_e = jax.nn.softplus(u[2]) + self._THETA_FLOOR
        theta = [eta_m, eta_m_cold_coordinate, eta_e]
        if self.theta_dim >= 4:
            theta.append(jax.nn.softplus(u[3]) + self._THETA_FLOOR)
        if self.theta_dim >= 5:
            theta.append(self.beta_chi_max_abs * jnp.tanh(u[4]))
        return jnp.asarray(theta, dtype=jnp.float64)

    def _theta_log_and_jac_log_u_jax(self, unconstrained_theta):
        """
        Return theta, log(theta), and dlog(theta)/du on the unconstrained manifold.

        The Jacobian term is used so sampling in unconstrained variables preserves
        the original prior defined in log(theta) space.
        """
        u = jnp.asarray(unconstrained_theta, dtype=jnp.float64)
        sig = jax.nn.sigmoid(u)
        theta = self._theta_from_unconstrained_jax(u)
        if self.energy_coordinate == "loading_ratios":
            eta_m_cold_over_eta_m = jax.nn.softplus(u[1]) + self._THETA_FLOOR
            eta_e_over_eta_m = jax.nn.softplus(u[2]) + self._THETA_FLOOR
            dlog_eta_m_du = sig[0] / jnp.maximum(theta[0], self._THETA_FLOOR)
            dlog_cold_ratio_du = sig[1] / jnp.maximum(eta_m_cold_over_eta_m, self._THETA_FLOOR)
            dlog_energy_ratio_du = sig[2] / jnp.maximum(eta_e_over_eta_m, self._THETA_FLOOR)
            jac_log_u = jnp.asarray(
                [
                    [dlog_eta_m_du, 0.0, 0.0],
                    [dlog_eta_m_du, dlog_cold_ratio_du, 0.0],
                    [dlog_eta_m_du, 0.0, dlog_energy_ratio_du],
                ],
                dtype=jnp.float64,
            )
            prior_coordinate = jnp.log(theta[:3])
            if self.theta_dim >= 4:
                dlog_a_mix_du = sig[3] / jnp.maximum(theta[3], self._THETA_FLOOR)
                jac_log_u = jnp.pad(jac_log_u, ((0, 1), (0, 1)))
                jac_log_u = jac_log_u.at[3, 3].set(dlog_a_mix_du)
                prior_coordinate = jnp.concatenate([prior_coordinate, jnp.log(theta[3:4])])
            if self.theta_dim >= 5:
                dbeta_du = self.beta_chi_max_abs * (1.0 - jnp.tanh(u[4]) ** 2)
                jac_log_u = jnp.pad(jac_log_u, ((0, 1), (0, 1)))
                jac_log_u = jac_log_u.at[4, 4].set(dbeta_du)
                prior_coordinate = jnp.concatenate([prior_coordinate, theta[4:5]])
            return theta, prior_coordinate, jac_log_u

        if self.energy_coordinate == "eta_e_over_eta_m":
            eta_e_over_eta_m = jax.nn.softplus(u[2]) + self._THETA_FLOOR
            dlog_eta_m_du = sig[0] / jnp.maximum(theta[0], self._THETA_FLOOR)
            dlog_eta_m_cold_du = sig[1] / jnp.maximum(theta[1], self._THETA_FLOOR)
            dlog_ratio_du = sig[2] / jnp.maximum(eta_e_over_eta_m, self._THETA_FLOOR)
            jac_log_u = jnp.asarray(
                [
                    [dlog_eta_m_du, 0.0, 0.0],
                    [0.0, dlog_eta_m_cold_du, 0.0],
                    [dlog_eta_m_du, 0.0, dlog_ratio_du],
                ],
                dtype=jnp.float64,
            )
            prior_coordinate = jnp.log(theta[:3])
            if self.theta_dim >= 4:
                dlog_a_mix_du = sig[3] / jnp.maximum(theta[3], self._THETA_FLOOR)
                jac_log_u = jnp.pad(jac_log_u, ((0, 1), (0, 1)))
                jac_log_u = jac_log_u.at[3, 3].set(dlog_a_mix_du)
                prior_coordinate = jnp.concatenate([prior_coordinate, jnp.log(theta[3:4])])
            if self.theta_dim >= 5:
                dbeta_du = self.beta_chi_max_abs * (1.0 - jnp.tanh(u[4]) ** 2)
                jac_log_u = jnp.pad(jac_log_u, ((0, 1), (0, 1)))
                jac_log_u = jac_log_u.at[4, 4].set(dbeta_du)
                prior_coordinate = jnp.concatenate([prior_coordinate, theta[4:5]])
            return theta, prior_coordinate, jac_log_u

        if self.eta_e_parameterization == "bounded":
            deta_e_du = self._ETA_E_MAX * sig[2] * (1.0 - sig[2])
        else:
            deta_e_du = sig[2]

        dcoord_du = [
            sig[0] / jnp.maximum(theta[0], self._THETA_FLOOR),
            sig[1] / jnp.maximum(theta[1], self._THETA_FLOOR),
            deta_e_du / jnp.maximum(theta[2], self._THETA_FLOOR),
        ]
        prior_coordinate_values = [jnp.log(theta[0]), jnp.log(theta[1]), jnp.log(theta[2])]
        if self.theta_dim >= 4:
            dcoord_du.append(sig[3] / jnp.maximum(theta[3], self._THETA_FLOOR))
            prior_coordinate_values.append(jnp.log(theta[3]))
        if self.theta_dim >= 5:
            dcoord_du.append(self.beta_chi_max_abs * (1.0 - jnp.tanh(u[4]) ** 2))
            prior_coordinate_values.append(theta[4])
        jac_log_u = jnp.diag(jnp.asarray(dcoord_du, dtype=jnp.float64))
        prior_coordinate = jnp.asarray(prior_coordinate_values, dtype=jnp.float64)
        return theta, prior_coordinate, jac_log_u

    _theta_log_and_dlog_du_jax = _theta_log_and_jac_log_u_jax

    def _unconstrained_from_theta_numpy(self, theta):
        """Inverse map for initializing unconstrained optimization/sampling state."""
        theta_arr = np.asarray(theta, dtype=float)
        theta_arr = self._coerce_theta_numpy(theta_arr)

        # Guard against nonphysical values in user-provided initial points.
        eta_m = np.clip(theta_arr[0], self._THETA_FLOOR * 10.0, None)
        eta_m_cold = np.clip(theta_arr[1], self._THETA_FLOOR * 10.0, None)
        if self.energy_coordinate == "loading_ratios":
            eta_e = np.clip(theta_arr[2], self._THETA_FLOOR * 10.0, None)
            eta_m_cold_over_eta_m = np.clip(eta_m_cold / eta_m, self._THETA_FLOOR * 10.0, None)
            eta_e_over_eta_m = np.clip(eta_e / eta_m, self._THETA_FLOOR * 10.0, None)
            u = np.zeros(self.theta_dim, dtype=float)
            u[0] = np.log(np.expm1(max(eta_m - self._THETA_FLOOR, 1e-12)))
            u[1] = np.log(np.expm1(max(eta_m_cold_over_eta_m - self._THETA_FLOOR, 1e-12)))
            u[2] = np.log(np.expm1(max(eta_e_over_eta_m - self._THETA_FLOOR, 1e-12)))
            if self.theta_dim >= 4:
                a_mix = np.clip(theta_arr[3], self._THETA_FLOOR * 10.0, None)
                u[3] = np.log(np.expm1(max(a_mix - self._THETA_FLOOR, 1e-12)))
            if self.theta_dim >= 5:
                beta_frac = np.clip(theta_arr[4] / self.beta_chi_max_abs, -1.0 + 1e-10, 1.0 - 1e-10)
                u[4] = np.arctanh(beta_frac)
            return u

        if self.energy_coordinate == "eta_e_over_eta_m":
            eta_e = np.clip(theta_arr[2], self._THETA_FLOOR * 10.0, None)
            eta_e_over_eta_m = np.clip(eta_e / eta_m, self._THETA_FLOOR * 10.0, None)
            u = np.zeros(self.theta_dim, dtype=float)
            u[0] = np.log(np.expm1(max(eta_m - self._THETA_FLOOR, 1e-12)))
            u[1] = np.log(np.expm1(max(eta_m_cold - self._THETA_FLOOR, 1e-12)))
            u[2] = np.log(np.expm1(max(eta_e_over_eta_m - self._THETA_FLOOR, 1e-12)))
            if self.theta_dim >= 4:
                a_mix = np.clip(theta_arr[3], self._THETA_FLOOR * 10.0, None)
                u[3] = np.log(np.expm1(max(a_mix - self._THETA_FLOOR, 1e-12)))
            if self.theta_dim >= 5:
                beta_frac = np.clip(theta_arr[4] / self.beta_chi_max_abs, -1.0 + 1e-10, 1.0 - 1e-10)
                u[4] = np.arctanh(beta_frac)
            return u

        if self.eta_e_parameterization == "bounded":
            eta_e = np.clip(theta_arr[2], self._THETA_FLOOR * 10.0, self._ETA_E_MAX - 1e-9)
        else:
            eta_e = np.clip(theta_arr[2], self._THETA_FLOOR * 10.0, None)

        u = np.zeros(self.theta_dim, dtype=float)
        u[0] = np.log(np.expm1(max(eta_m - self._THETA_FLOOR, 1e-12)))
        u[1] = np.log(np.expm1(max(eta_m_cold - self._THETA_FLOOR, 1e-12)))

        if self.eta_e_parameterization == "bounded":
            frac = np.clip((eta_e - self._THETA_FLOOR) / self._ETA_E_MAX, 1e-10, 1.0 - 1e-10)
            u[2] = np.log(frac / (1.0 - frac))
        else:
            u[2] = np.log(np.expm1(max(eta_e - self._THETA_FLOOR, 1e-12)))
        if self.theta_dim >= 4:
            a_mix = np.clip(theta_arr[3], self._THETA_FLOOR * 10.0, None)
            u[3] = np.log(np.expm1(max(a_mix - self._THETA_FLOOR, 1e-12)))
        if self.theta_dim >= 5:
            beta_frac = np.clip(theta_arr[4] / self.beta_chi_max_abs, -1.0 + 1e-10, 1.0 - 1e-10)
            u[4] = np.arctanh(beta_frac)
        return u

    def _theta_from_unconstrained_numpy(self, unconstrained_theta):
        """NumPy helper for reporting and posterior sample conversion."""
        u = np.asarray(unconstrained_theta, dtype=float)
        sig = 1.0 / (1.0 + np.exp(-u))
        eta_m = np.log1p(np.exp(-np.abs(u[..., 0]))) + np.maximum(u[..., 0], 0.0) + self._THETA_FLOOR
        eta_m_cold_coordinate = (
            np.log1p(np.exp(-np.abs(u[..., 1]))) + np.maximum(u[..., 1], 0.0) + self._THETA_FLOOR
        )
        if self.energy_coordinate == "loading_ratios":
            eta_m_cold_over_eta_m = eta_m_cold_coordinate
            eta_e_over_eta_m = (
                np.log1p(np.exp(-np.abs(u[..., 2]))) + np.maximum(u[..., 2], 0.0) + self._THETA_FLOOR
            )
            eta_m_cold = eta_m * eta_m_cold_over_eta_m
            eta_e = eta_m * eta_e_over_eta_m
            theta = [eta_m, eta_m_cold, eta_e]
            if self.theta_dim >= 4:
                theta.append(np.log1p(np.exp(-np.abs(u[..., 3]))) + np.maximum(u[..., 3], 0.0) + self._THETA_FLOOR)
            if self.theta_dim >= 5:
                theta.append(self.beta_chi_max_abs * np.tanh(u[..., 4]))
            return np.stack(theta, axis=-1)

        if self.energy_coordinate == "eta_e_over_eta_m":
            eta_e_over_eta_m = (
                np.log1p(np.exp(-np.abs(u[..., 2]))) + np.maximum(u[..., 2], 0.0) + self._THETA_FLOOR
            )
            eta_e = eta_m * eta_e_over_eta_m
            theta = [eta_m, eta_m_cold_coordinate, eta_e]
            if self.theta_dim >= 4:
                theta.append(np.log1p(np.exp(-np.abs(u[..., 3]))) + np.maximum(u[..., 3], 0.0) + self._THETA_FLOOR)
            if self.theta_dim >= 5:
                theta.append(self.beta_chi_max_abs * np.tanh(u[..., 4]))
            return np.stack(theta, axis=-1)

        if self.eta_e_parameterization == "bounded":
            eta_e = self._ETA_E_MAX * sig[..., 2] + self._THETA_FLOOR
        else:
            eta_e = np.log1p(np.exp(-np.abs(u[..., 2]))) + np.maximum(u[..., 2], 0.0) + self._THETA_FLOOR
        theta = [eta_m, eta_m_cold_coordinate, eta_e]
        if self.theta_dim >= 4:
            theta.append(np.log1p(np.exp(-np.abs(u[..., 3]))) + np.maximum(u[..., 3], 0.0) + self._THETA_FLOOR)
        if self.theta_dim >= 5:
            theta.append(self.beta_chi_max_abs * np.tanh(u[..., 4]))
        return np.stack(theta, axis=-1)

    def energy_coordinates_from_theta_numpy(self, theta):
        """Return reporting coordinates for theta in the active diagnostic ratio mode."""
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.shape[-1] != self.theta_dim:
            raise ValueError(f"theta must have length-{self.theta_dim} trailing dimension")
        if self.energy_coordinate == "loading_ratios":
            out = np.array(theta_arr, dtype=float, copy=True)
            out[..., 1] = out[..., 1] / np.maximum(out[..., 0], self._THETA_FLOOR)
            out[..., 2] = out[..., 2] / np.maximum(out[..., 0], self._THETA_FLOOR)
            return out
        if self.energy_coordinate != "eta_e_over_eta_m":
            return np.array(theta_arr, dtype=float, copy=True)
        out = np.array(theta_arr, dtype=float, copy=True)
        out[..., 2] = out[..., 2] / np.maximum(out[..., 0], self._THETA_FLOOR)
        return out

    def eta_e_softcap_penalty(self, eta_e: float | np.ndarray) -> np.ndarray:
        """Return the diagnostic soft-cap penalty for eta_E values."""
        eta_e_arr = np.asarray(eta_e, dtype=float)
        transition = self.eta_e_softcap_transition
        excess = transition * np.logaddexp(0.0, (eta_e_arr - self.eta_e_softcap_center) / transition)
        return 0.5 * (excess / self.eta_e_softcap_sigma) ** 2

    def _build_predictor(self):
        r_grid = self.r_grid
        r_star = self.r_star_cgs
        sfr_cgs = self.sfr * Msun / yr
        v_circ_cgs = self.v_circ * 1e5
        config = self.config

        m_cloud0 = self.m_cloud0
        mass_weights = self.mass_weights
        n_species = int(m_cloud0.shape[0])

        topaz_log10_temperature, topaz_primordial_cooling_cgs, topaz_metal_cooling_cgs = self._topaz_arrays

        source_volume = 4.0 / 3.0 * np.pi * r_star**3
        mach0 = 1.0 + float(config.sonic_point_offset)
        injection_radius = float(config.cold_cloud_injection_radial_extent_frac) * r_star
        injection_power = float(config.cold_cloud_injection_radial_power)
        eta_e_max = self._ETA_E_MAX
        bounded_eta_e = self.eta_e_parameterization == "bounded"

        r_obs_min_kpc = self.r_obs_min_kpc
        r_obs_max_kpc = self.r_max_kpc
        r_kpc = r_grid / kpc
        radial_window_col = ((r_kpc >= r_obs_min_kpc) & (r_kpc <= r_obs_max_kpc))[:, None]
        dndv_velocity_bins = jnp.asarray(self.dndv_bin_centers_kms, dtype=jnp.float64)
        dndv_kernel_sigma = jnp.asarray(self.dndv_kernel_sigma_kms, dtype=jnp.float64)
        dndv_kernel_kind = {
            "gaussian": self._DNDV_KERNEL_GAUSSIAN,
            "truncated_gaussian": self._DNDV_KERNEL_TRUNCATED_GAUSSIAN,
            "compact_cosine": self._DNDV_KERNEL_COMPACT_COSINE,
        }[self.dndv_kernel]
        dndv_kernel_truncate_sigma = float(self.dndv_kernel_truncate_sigma)
        use_stalled_wind_policy = self.failure_policy == "stalled_wind"
        stall_velocity_floor = jnp.asarray(self.stall_velocity_floor_kms * 1e5, dtype=jnp.float64)
        stall_velocity_transition = jnp.asarray(self.stall_velocity_transition_kms * 1e5, dtype=jnp.float64)
        stall_radius_sigma = jnp.asarray(self.stall_radius_sigma_kpc * kpc, dtype=jnp.float64)
        stall_radius_transition = jnp.asarray(self.stall_radius_transition_kpc * kpc, dtype=jnp.float64)
        required_radius = jnp.asarray(self.r_max_kpc * kpc, dtype=jnp.float64)
        injection_profile_col = jnp.where(
            r_grid < injection_radius,
            (r_grid / jnp.maximum(injection_radius, 1e-30)) ** injection_power,
            1.0,
        )[:, None]
        solid_angle_r2_col = (config.Omwind * r_grid * r_grid)[:, None]
        integrator_mode = self.integrator_mode
        integrator_rtol = self.integrator_rtol
        integrator_atol = self.integrator_atol
        integrator_max_steps = self.integrator_max_steps
        use_a_mix_parameter = self.expanded_parameters in {"a_mix", "a_mix_beta_chi"}
        use_beta_chi_parameter = self.expanded_parameters == "a_mix_beta_chi"
        static_a_mix = float(self.config.A_mix)
        static_beta_chi_mix = float(self.config.beta_chi_mix)
        static_mixing_chi_pivot = float(self.mixing_chi_pivot)

        @jax.jit
        def build_state_and_params(theta):
            eta_m = jnp.maximum(theta[0], 1e-12)
            eta_m_cold = jnp.maximum(theta[1], 1e-12)
            if bounded_eta_e:
                eta_e = jnp.clip(theta[2], 1e-12, eta_e_max)
            else:
                eta_e = jnp.maximum(theta[2], 1e-12)
            a_mix = jnp.maximum(theta[3], 1e-12) if use_a_mix_parameter else jnp.asarray(static_a_mix, dtype=jnp.float64)
            beta_chi_mix = theta[4] if use_beta_chi_parameter else jnp.asarray(static_beta_chi_mix, dtype=jnp.float64)

            mdot_hot = eta_m * sfr_cgs
            edot_hot = eta_e * (config.E_SN / (config.mstar * Msun)) * sfr_cgs

            v_star = jnp.sqrt(edot_hot / mdot_hot) * (1.0 / ((gamma - 1.0) * mach0) + 0.5) ** (-0.5)
            rho_star = mdot_hot / (config.Omwind * r_star * r_star * v_star)
            p_star = rho_star * v_star * v_star / (mach0 * mach0 * gamma)

            eta_m_cold_array = eta_m_cold * mass_weights
            mdot_cold0 = eta_m_cold_array * sfr_cgs
            ndot_cloud0 = mdot_cold0 / m_cloud0

            y0 = jnp.zeros((4 + 3 * n_species,), dtype=jnp.float64)
            y0 = y0.at[0].set(v_star)
            y0 = y0.at[1].set(rho_star)
            y0 = y0.at[2].set(p_star)
            y0 = y0.at[3].set(rho_star * config.Z_hot_over_Z_solar * Z_solar)
            y0 = y0.at[4 : 4 + n_species].set(m_cloud0)
            y0 = y0.at[4 + n_species : 4 + 2 * n_species].set(config.v_cloud_init * 1e5)
            y0 = y0.at[4 + 2 * n_species :].set(config.Z_cloud_over_Z_solar * Z_solar)

            params = JaxWindParams(
                v_circ=v_circ_cgs,
                Ndot_cloud0=ndot_cloud0,
                T_cloud=float(config.T_cl),
                injection_radius=injection_radius,
                injection_power=injection_power,
                M_cloud_min=float(config.M_cloud_min),
                CoolingAreaChiPower=float(config.CoolingAreaChiPower),
                ColdTurbulenceChiPower=float(config.ColdTurbulenceChiPower),
                TurbulentVelocityChiPower=float(config.TurbulentVelocityChiPower),
                geometric_factor=float(config.geometric_factor),
                Mdot_coefficient=float(config.Mdot_coefficient),
                A_mix=a_mix,
                beta_chi_mix=beta_chi_mix,
                mixing_chi_pivot=static_mixing_chi_pivot,
                Cooling_Factor=float(config.Cooling_Factor),
                drag_coeff=float(config.drag_coeff),
                f_turb0=float(config.f_turb0),
                Omwind=float(config.Omwind),
                mu=float(config.mu),
                redshift=float(config.redshift),
                Z_hot_over_Z_solar=float(config.Z_hot_over_Z_solar),
                v_cloud_min=float(config.v_cloud_min),
                r0=r_star,
                Edot_per_Vol=edot_hot / source_volume,
                Mdot_per_Vol=mdot_hot / source_volume,
                topaz_log10_temperature=topaz_log10_temperature,
                topaz_primordial_cooling_cgs=topaz_primordial_cooling_cgs,
                topaz_metal_cooling_cgs=topaz_metal_cooling_cgs,
            )
            return y0, params

        if integrator_mode == "rk2":

            @jax.jit
            def integrate_state(y0, params):
                return integrate_wind_rk2_scan(r_grid, y0, params)

        elif integrator_mode == "rk3":

            @jax.jit
            def integrate_state(y0, params):
                return integrate_wind_rk3_scan(r_grid, y0, params)

        elif integrator_mode == "rk4":

            @jax.jit
            def integrate_state(y0, params):
                return integrate_wind_rk4_scan(r_grid, y0, params)

        else:

            def integrate_state(y0, params):
                return integrate_wind_tsit5(
                    r_grid,
                    y0,
                    params,
                    rtol=integrator_rtol,
                    atol=integrator_atol,
                    max_steps=integrator_max_steps,
                )

        def predict_theta_with_valid(theta):
            y0, params = build_state_and_params(theta)
            y = integrate_state(y0, params)

            finite_matrix = jnp.isfinite(y)
            finite = jnp.all(finite_matrix, axis=1)
            v_wind = y[:, 0]
            rho_wind = y[:, 1]
            pressure = y[:, 2]
            physical = (v_wind > 0.0) & (rho_wind > 0.0) & (pressure > 0.0)
            valid_state = jnp.all(finite & physical)
            invalid_row = ~(finite & physical)
            has_invalid = jnp.any(invalid_row)
            first_invalid_idx = jnp.where(has_invalid, jnp.argmax(invalid_row), y.shape[0] - 1)
            first_invalid_r = r_grid[first_invalid_idx]
            first_finite = finite[first_invalid_idx]
            # Finite nonphysical rows are failed/stalled wind outcomes in the
            # inference layer; reserve status 2 for genuinely nonfinite solver
            # failures where no local physical diagnostics can be trusted.
            first_invalid_is_finite_failure = has_invalid & first_finite
            trajectory_status_code = jnp.where(
                first_invalid_is_finite_failure,
                1.0,
                jnp.where(has_invalid, 2.0, 0.0),
            )
            min_hot_velocity_kms = jnp.min(
                jnp.nan_to_num(v_wind / 1e5, nan=-1e12, posinf=1e12, neginf=-1e12)
            )

            finite_alive = (finite & (rho_wind > 0.0) & (pressure > 0.0)).astype(jnp.float64)
            v_alive = jax.nn.sigmoid((v_wind - stall_velocity_floor) / stall_velocity_transition)
            alive_step = jnp.clip(finite_alive * v_alive, 0.0, 1.0)
            alive_weight = jnp.cumprod(alive_step)
            if not use_stalled_wind_policy:
                alive_weight = jnp.ones_like(v_wind)
            soft_reach_radius = r_grid[0] + jnp.trapezoid(alive_weight, r_grid)
            reach_deficit = stall_radius_transition * jax.nn.softplus(
                (required_radius - soft_reach_radius) / stall_radius_transition
            )
            stall_penalty = jnp.where(
                (trajectory_status_code > 0.5) & (trajectory_status_code < 1.5),
                0.5 * (reach_deficit / stall_radius_sigma) ** 2,
                0.0,
            )

            m_cloud = y[:, 4 : 4 + n_species]
            v_cloud = y[:, 4 + n_species : 4 + 2 * n_species]
            if use_stalled_wind_policy:
                m_cloud_obs = jnp.maximum(
                    jnp.nan_to_num(m_cloud, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                )
                v_cloud_obs = jnp.nan_to_num(v_cloud, nan=0.0, posinf=1e10, neginf=-1e10)
                alive_col = alive_weight[:, None]
            else:
                m_cloud_obs = m_cloud
                v_cloud_obs = v_cloud
                alive_col = jnp.ones_like(m_cloud)

            ndot = params.Ndot_cloud0[None, :] * injection_profile_col
            v_cloud_safe = jnp.maximum(v_cloud_obs, params.v_cloud_min * 1e5)
            denom = solid_angle_r2_col * v_cloud_safe
            rho_cloud = ndot * m_cloud_obs / jnp.maximum(denom, 1e-60)
            n_h = rho_cloud / (1.4 * mp)

            active = m_cloud_obs >= params.M_cloud_min
            weight = (active & radial_window_col).astype(jnp.float64) * alive_col

            v_kms = v_cloud_obs / 1e5
            n_h_eff = n_h * weight

            m0 = jnp.sum(jnp.trapezoid(n_h_eff, r_grid, axis=0))
            m1 = jnp.sum(jnp.trapezoid(n_h_eff * v_kms, r_grid, axis=0))
            v2 = v_kms * v_kms
            m2 = jnp.sum(jnp.trapezoid(n_h_eff * v2, r_grid, axis=0))
            m3 = jnp.sum(jnp.trapezoid(n_h_eff * v2 * v_kms, r_grid, axis=0))
            m4 = jnp.sum(jnp.trapezoid(n_h_eff * v2 * v2, r_grid, axis=0))
            raw_moments = jnp.asarray([m0, m1, m2, m3, m4], dtype=jnp.float64)

            if self.observable_set == self._OBS_MOMENTS3:
                observables = raw_moments[:3]
                valid_observables = jnp.all(jnp.isfinite(observables)) & jnp.all(observables > 0.0)
                fallback_obs = jnp.asarray([1e-30, 1e-20, 1e-10], dtype=jnp.float64)
                obs_finite = jnp.nan_to_num(observables, nan=0.0, posinf=1e100, neginf=-1e100)
                obs_floor = jnp.asarray([1e-30, 1e-20, 1e-10], dtype=jnp.float64)
                obs_barrier = jnp.sum(jax.nn.softplus((obs_floor - obs_finite) / obs_floor))
            elif self.observable_set == self._OBS_SHAPE5:
                observables = self._observables_shape5_from_raw_moments(raw_moments)
                var_v = (raw_moments[2] / jnp.maximum(raw_moments[0], 1e-300)) - observables[1] * observables[1]
                valid_observables = jnp.all(jnp.isfinite(observables)) & (var_v > 0.0) & (raw_moments[0] > 0.0)
                fallback_obs = jnp.asarray([jnp.log(1e-30), 300.0, 100.0, 0.0, 3.0], dtype=jnp.float64)
                obs_finite_violation = jnp.mean(jnp.where(jnp.isfinite(observables), 0.0, 1.0))
                variance_barrier = jax.nn.softplus((1e-8 - var_v) / 1e-8)
                obs_barrier = obs_finite_violation + variance_barrier
            else:
                dndv_floor = 1e-40
                dndv_binned = self._observables_dndv_binned(
                    v_kms=v_kms,
                    n_h_eff=n_h_eff,
                    r_grid=r_grid,
                    velocity_bins_kms=dndv_velocity_bins,
                    kernel_sigma_kms=dndv_kernel_sigma,
                    kernel_kind=dndv_kernel_kind,
                    truncate_sigma=dndv_kernel_truncate_sigma,
                )
                dndv_safe = jnp.maximum(dndv_binned, dndv_floor)
                if self.observable_set == self._OBS_LOG_DNDV_BINNED:
                    observables = jnp.log(dndv_safe)
                    valid_observables = jnp.all(jnp.isfinite(observables))
                    fallback_obs = jnp.full((self.observable_dim,), jnp.log(dndv_floor), dtype=jnp.float64)
                    obs_barrier = jnp.mean(jnp.where(jnp.isfinite(observables), 0.0, 1.0))
                else:
                    observables = dndv_safe
                    valid_observables = jnp.all(jnp.isfinite(observables)) & jnp.all(observables > 0.0)
                    fallback_obs = jnp.full((self.observable_dim,), dndv_floor, dtype=jnp.float64)
                    obs_finite = jnp.nan_to_num(observables, nan=0.0, posinf=1e100, neginf=-1e100)
                    obs_barrier = jnp.sum(jax.nn.softplus((dndv_floor - obs_finite) / dndv_floor))

            valid_raw = jnp.all(jnp.isfinite(raw_moments)) & (raw_moments[0] > 0.0) & (raw_moments[2] > 0.0)
            valid = valid_state & valid_raw & valid_observables

            v_scale = jnp.maximum(jnp.abs(y0[0]), 1.0)
            rho_scale = jnp.maximum(jnp.abs(y0[1]), 1e-30)
            p_scale = jnp.maximum(jnp.abs(y0[2]), 1e-30)
            def nonpositive_barrier(values, scale):
                rel = jnp.nan_to_num(values / scale, nan=-1.0, posinf=1e6, neginf=-1e6)
                return jnp.where(rel > 0.0, 0.0, jax.nn.softplus(-rel))

            if use_stalled_wind_policy:
                soft_neg_v = jax.nn.softplus((stall_velocity_floor - v_wind) / stall_velocity_transition)
            else:
                soft_neg_v = nonpositive_barrier(v_wind, v_scale)
            soft_neg_rho = nonpositive_barrier(rho_wind, rho_scale)
            soft_neg_p = nonpositive_barrier(pressure, p_scale)
            finite_violation = jnp.mean(jnp.where(finite_matrix, 0.0, 1.0))

            barrier_value = (
                jnp.mean(soft_neg_v + soft_neg_rho + soft_neg_p) + 10.0 * finite_violation + obs_barrier
            )
            usable_observables = valid_observables & valid_raw & jnp.isfinite(raw_moments[0])
            if use_stalled_wind_policy:
                usable_observables = usable_observables & (trajectory_status_code < 1.5)
            else:
                usable_observables = usable_observables & valid_state
            observables_safe = jnp.where(usable_observables, observables, fallback_obs)

            raw_fallback = jnp.asarray([1e-30, 1e-20, 1e-10, 1e-5, 1e0], dtype=jnp.float64)
            raw_moments_safe = jnp.where(usable_observables, raw_moments, raw_fallback)
            return (
                observables_safe,
                raw_moments_safe,
                jnp.where(valid, 1.0, 0.0),
                barrier_value,
                first_invalid_r,
                trajectory_status_code,
                soft_reach_radius,
                min_hot_velocity_kms,
                stall_penalty,
            )

        if integrator_mode in {"rk2", "rk3", "rk4"}:
            return jax.jit(predict_theta_with_valid)
        return predict_theta_with_valid

    def predict_observables(self, theta: Sequence[float]) -> np.ndarray:
        """Predict configured observables for active physical parameters."""
        theta_arr = jnp.asarray(self._coerce_theta_numpy(theta), dtype=jnp.float64)
        return np.asarray(self._predict_theta_fn(theta_arr), dtype=float)

    def predict_moments(self, theta: Sequence[float]) -> np.ndarray:
        """Predict raw [M0, M1, M2] for linear-space parameters."""
        return self.predict_raw_moments(theta)[:3]

    def predict_observables_log(self, log_theta: Sequence[float]) -> np.ndarray:
        """Predict configured observables from the active mixed prior coordinate."""
        log_theta_arr = jnp.asarray(log_theta, dtype=jnp.float64)
        return np.asarray(self._predict_log_theta_fn(log_theta_arr), dtype=float)

    def predict_moments_log(self, log_theta: Sequence[float]) -> np.ndarray:
        """Predict raw [M0, M1, M2] from the active mixed prior coordinate."""
        log_theta_arr = np.asarray(log_theta, dtype=float)
        return self.predict_moments(self._theta_from_prior_coordinate_numpy(log_theta_arr))

    def predict_raw_moments(self, theta: Sequence[float]) -> np.ndarray:
        """Predict raw velocity moments [M0, M1, M2, M3, M4] regardless of observable_set."""
        theta_arr = jnp.asarray(self._coerce_theta_numpy(theta), dtype=jnp.float64)
        return np.asarray(self._predict_raw_theta_fn(theta_arr), dtype=float)

    def get_dndv_velocity_bins(self) -> np.ndarray:
        """Return binned dN/dv velocity centers [km/s] for binned profile observable sets."""
        if self.observable_set not in {self._OBS_DNDV_BINNED, self._OBS_LOG_DNDV_BINNED}:
            raise ValueError("Velocity bins are only defined for binned dN/dv observable sets")
        return np.asarray(self.dndv_bin_centers_kms, dtype=float)

    def make_negative_log_posterior(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 0.8),
        invalid_penalty: float = 1e6,
        include_transform_jacobian: bool = True,
    ):
        """
        Create JAX-jitted negative log posterior over unconstrained parameters.

        The prior is defined in log(theta) space. Set
        ``include_transform_jacobian=True`` when sampling in unconstrained
        variables so the transformed target preserves the log(theta) posterior
        measure. Use ``False`` for coordinate-independent MAP reporting in
        log(theta) space.
        """
        y_obs = jnp.asarray(observed_moments, dtype=jnp.float64)
        if y_obs.shape != (self.observable_dim,):
            raise ValueError(
                "observed_moments must match observable_set size "
                f"{self.observable_dim} ({self.observable_names}), got shape {y_obs.shape}"
            )

        likelihood_setup = self._observable_likelihood_setup(observed_moments, covariance_moments)
        gaussian_cov_inv = jnp.asarray(likelihood_setup["gaussian_cov_inv"], dtype=jnp.float64)
        censored_mask = jnp.asarray(likelihood_setup["censored_mask"], dtype=jnp.float64)
        censored_upper = jnp.asarray(likelihood_setup["censored_upper"], dtype=jnp.float64)
        use_censored_likelihood = bool(likelihood_setup["use_censored"])
        censor_transition = jnp.asarray(self.log_dndv_censor_transition, dtype=jnp.float64)
        censor_sigma = jnp.asarray(self.log_dndv_censor_sigma, dtype=jnp.float64)

        prior_mean_log_arr = self._coerce_prior_coordinate_numpy(prior_mean_log, name="prior_mean_log")
        prior_sigma_log_arr = self._coerce_prior_sigma_numpy(prior_sigma_log, name="prior_sigma_log")

        if np.any(prior_sigma_log_arr <= 0.0):
            raise ValueError("All prior_sigma_log entries must be positive")

        prior_mean_log_jax = jnp.asarray(prior_mean_log_arr, dtype=jnp.float64)
        prior_sigma_log_jax = jnp.asarray(prior_sigma_log_arr, dtype=jnp.float64)
        r_max_cgs = float(self.r_max_kpc * kpc)
        use_stalled_wind_policy = self.failure_policy == "stalled_wind"
        use_eta_e_softcap = self.eta_e_parameterization == "softcap"
        softcap_center = jnp.asarray(self.eta_e_softcap_center, dtype=jnp.float64)
        softcap_sigma = jnp.asarray(self.eta_e_softcap_sigma, dtype=jnp.float64)
        softcap_transition = jnp.asarray(self.eta_e_softcap_transition, dtype=jnp.float64)

        predict_theta_with_valid = self._predict_theta_with_valid_fn

        def likelihood_chi2_terms(observables):
            resid = observables - y_obs
            gaussian_chi2 = resid @ gaussian_cov_inv @ resid
            if use_censored_likelihood:
                censor_excess = censor_transition * jax.nn.softplus(
                    (observables - censored_upper) / censor_transition
                )
                censored_chi2 = jnp.sum(censored_mask * (censor_excess / censor_sigma) ** 2)
            else:
                censored_chi2 = jnp.asarray(0.0, dtype=jnp.float64)
            chi2 = gaussian_chi2 + censored_chi2
            return chi2, gaussian_chi2, censored_chi2

        @jax.jit
        def nlp(unconstrained_theta):
            theta, log_theta, jac_log_u = self._theta_log_and_jac_log_u_jax(unconstrained_theta)
            prediction = predict_theta_with_valid(theta)
            observables, _raw_moments, valid, barrier_value, first_invalid_r = prediction[:5]
            trajectory_status_code = prediction[5]
            stall_penalty = prediction[8]
            chi2, _gaussian_chi2, _censored_chi2 = likelihood_chi2_terms(observables)
            prior_chi2 = jnp.sum(((log_theta - prior_mean_log_jax) / prior_sigma_log_jax) ** 2)
            jac_sign, log_jacobian_raw = jnp.linalg.slogdet(jac_log_u)
            log_jacobian = jnp.where(jac_sign != 0.0, log_jacobian_raw, -jnp.inf)
            transform_term = -log_jacobian if include_transform_jacobian else 0.0
            if use_eta_e_softcap:
                eta_e_excess = softcap_transition * jax.nn.softplus(
                    (theta[2] - softcap_center) / softcap_transition
                )
                eta_e_softcap_penalty = 0.5 * (eta_e_excess / softcap_sigma) ** 2
            else:
                eta_e_softcap_penalty = 0.0
            smooth_penalty = 100.0 * barrier_value
            early_fail_penalty = jnp.where(
                (valid > 0.5) | use_stalled_wind_policy,
                0.0,
                10.0 * jax.nn.softplus((r_max_cgs - first_invalid_r) / jnp.maximum(r_max_cgs, 1e-30)),
            )
            numerical_failure_penalty = jnp.where(
                use_stalled_wind_policy & (trajectory_status_code > 1.5),
                invalid_penalty,
                0.0,
            )
            hard_penalty = jnp.where(
                use_stalled_wind_policy | (valid > 0.5),
                0.0,
                invalid_penalty,
            )
            return (
                0.5 * (chi2 + prior_chi2)
                + transform_term
                + eta_e_softcap_penalty
                + smooth_penalty
                + early_fail_penalty
                + stall_penalty
                + numerical_failure_penalty
                + hard_penalty
            )

        return nlp

    def make_negative_log_posterior_components(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 0.8),
        invalid_penalty: float = 1e6,
        include_transform_jacobian: bool = True,
    ):
        """Create a JAX-jitted posterior component evaluator for diagnostics."""
        y_obs = jnp.asarray(observed_moments, dtype=jnp.float64)
        if y_obs.shape != (self.observable_dim,):
            raise ValueError(
                "observed_moments must match observable_set size "
                f"{self.observable_dim} ({self.observable_names}), got shape {y_obs.shape}"
            )

        likelihood_setup = self._observable_likelihood_setup(observed_moments, covariance_moments)
        gaussian_cov_inv = jnp.asarray(likelihood_setup["gaussian_cov_inv"], dtype=jnp.float64)
        censored_mask = jnp.asarray(likelihood_setup["censored_mask"], dtype=jnp.float64)
        censored_upper = jnp.asarray(likelihood_setup["censored_upper"], dtype=jnp.float64)
        use_censored_likelihood = bool(likelihood_setup["use_censored"])
        censor_transition = jnp.asarray(self.log_dndv_censor_transition, dtype=jnp.float64)
        censor_sigma = jnp.asarray(self.log_dndv_censor_sigma, dtype=jnp.float64)

        prior_mean_log_arr = self._coerce_prior_coordinate_numpy(prior_mean_log, name="prior_mean_log")
        prior_sigma_log_arr = self._coerce_prior_sigma_numpy(prior_sigma_log, name="prior_sigma_log")

        if np.any(prior_sigma_log_arr <= 0.0):
            raise ValueError("All prior_sigma_log entries must be positive")

        prior_mean_log_jax = jnp.asarray(prior_mean_log_arr, dtype=jnp.float64)
        prior_sigma_log_jax = jnp.asarray(prior_sigma_log_arr, dtype=jnp.float64)
        r_max_cgs = float(self.r_max_kpc * kpc)
        use_stalled_wind_policy = self.failure_policy == "stalled_wind"
        use_eta_e_softcap = self.eta_e_parameterization == "softcap"
        softcap_center = jnp.asarray(self.eta_e_softcap_center, dtype=jnp.float64)
        softcap_sigma = jnp.asarray(self.eta_e_softcap_sigma, dtype=jnp.float64)
        softcap_transition = jnp.asarray(self.eta_e_softcap_transition, dtype=jnp.float64)

        predict_theta_with_valid = self._predict_theta_with_valid_fn

        def likelihood_chi2_terms(observables):
            resid = observables - y_obs
            gaussian_chi2 = resid @ gaussian_cov_inv @ resid
            if use_censored_likelihood:
                censor_excess = censor_transition * jax.nn.softplus(
                    (observables - censored_upper) / censor_transition
                )
                censored_chi2 = jnp.sum(censored_mask * (censor_excess / censor_sigma) ** 2)
            else:
                censored_chi2 = jnp.asarray(0.0, dtype=jnp.float64)
            chi2 = gaussian_chi2 + censored_chi2
            return chi2, gaussian_chi2, censored_chi2

        @jax.jit
        def components(unconstrained_theta):
            theta, log_theta, jac_log_u = self._theta_log_and_jac_log_u_jax(unconstrained_theta)
            prediction = predict_theta_with_valid(theta)
            observables, _raw_moments, valid, barrier_value, first_invalid_r = prediction[:5]
            trajectory_status_code = prediction[5]
            soft_reach_radius = prediction[6]
            min_hot_velocity_kms = prediction[7]
            stall_penalty = prediction[8]
            chi2, gaussian_chi2, censored_chi2 = likelihood_chi2_terms(observables)
            prior_chi2 = jnp.sum(((log_theta - prior_mean_log_jax) / prior_sigma_log_jax) ** 2)
            jac_sign, log_jacobian_raw = jnp.linalg.slogdet(jac_log_u)
            log_jacobian = jnp.where(jac_sign != 0.0, log_jacobian_raw, -jnp.inf)
            transform_term = -log_jacobian if include_transform_jacobian else 0.0
            if use_eta_e_softcap:
                eta_e_excess = softcap_transition * jax.nn.softplus(
                    (theta[2] - softcap_center) / softcap_transition
                )
                eta_e_softcap_penalty = 0.5 * (eta_e_excess / softcap_sigma) ** 2
            else:
                eta_e_softcap_penalty = 0.0
            smooth_penalty = 100.0 * barrier_value
            early_fail_penalty = jnp.where(
                (valid > 0.5) | use_stalled_wind_policy,
                0.0,
                10.0 * jax.nn.softplus((r_max_cgs - first_invalid_r) / jnp.maximum(r_max_cgs, 1e-30)),
            )
            numerical_failure_penalty = jnp.where(
                use_stalled_wind_policy & (trajectory_status_code > 1.5),
                invalid_penalty,
                0.0,
            )
            hard_penalty = jnp.where(
                use_stalled_wind_policy | (valid > 0.5),
                0.0,
                invalid_penalty,
            )
            total_objective = (
                0.5 * (chi2 + prior_chi2)
                + transform_term
                + eta_e_softcap_penalty
                + smooth_penalty
                + early_fail_penalty
                + stall_penalty
                + numerical_failure_penalty
                + hard_penalty
            )
            return jnp.asarray(
                [
                    chi2,
                    gaussian_chi2,
                    censored_chi2,
                    prior_chi2,
                    log_jacobian,
                    transform_term,
                    eta_e_softcap_penalty,
                    smooth_penalty,
                    early_fail_penalty,
                    stall_penalty,
                    numerical_failure_penalty,
                    hard_penalty,
                    total_objective,
                    valid,
                    trajectory_status_code,
                    soft_reach_radius / kpc,
                    first_invalid_r / kpc,
                    min_hot_velocity_kms,
                ],
                dtype=jnp.float64,
            )

        return components

    def _fit_map_single_from_nlp(
        self,
        nlp,
        observed_moments: np.ndarray,
        covariance_moments: np.ndarray,
        initial_unconstrained_theta: np.ndarray,
        max_iter: int,
        grad_tol: float,
        start_index: int,
        num_starts: int,
    ) -> MAPFitResult:
        """Single-start damped-Newton MAP solve in unconstrained coordinates."""
        value_grad_fn = jax.jit(jax.value_and_grad(nlp))
        grad_fn = jax.jit(jax.grad(nlp))
        use_fd_hessian = self.integrator_mode == "tsit5"
        hess_fn = None if use_fd_hessian else jax.jit(jax.hessian(nlp))

        def compute_hessian(theta_u: np.ndarray) -> np.ndarray:
            if hess_fn is not None:
                return np.asarray(hess_fn(jnp.asarray(theta_u, dtype=jnp.float64)), dtype=float)

            # Diffrax traces in tsit5 mode do not support second-order autodiff.
            # Use finite differences of first derivatives in the 3D parameter space.
            u = np.asarray(theta_u, dtype=float)
            n = u.size
            h_cols = np.zeros((n, n), dtype=float)
            for i in range(n):
                h_i = 1e-3 * max(1.0, abs(u[i]))
                e_i = np.zeros_like(u)
                e_i[i] = h_i
                g_plus = np.asarray(grad_fn(jnp.asarray(u + e_i, dtype=jnp.float64)), dtype=float)
                g_minus = np.asarray(grad_fn(jnp.asarray(u - e_i, dtype=jnp.float64)), dtype=float)
                h_cols[:, i] = (g_plus - g_minus) / (2.0 * h_i)
            return 0.5 * (h_cols + h_cols.T)

        unconstrained_theta = np.asarray(initial_unconstrained_theta, dtype=float)
        if unconstrained_theta.shape != (self.theta_dim,):
            raise ValueError(f"initial_unconstrained_theta must be length-{self.theta_dim}")

        success = False
        message = "Maximum iterations reached before convergence"
        n_iter = 0

        for it in range(max_iter):
            n_iter = it + 1
            val, grad = value_grad_fn(jnp.asarray(unconstrained_theta, dtype=jnp.float64))
            val = float(val)
            grad = np.asarray(grad, dtype=float)

            if not np.isfinite(val) or not np.all(np.isfinite(grad)):
                message = "Encountered non-finite objective/gradient"
                break

            grad_norm = float(np.linalg.norm(grad, ord=np.inf))
            if grad_norm < grad_tol:
                success = True
                message = "Converged: gradient infinity norm below tolerance"
                break

            hess = compute_hessian(unconstrained_theta)
            hess = 0.5 * (hess + hess.T)
            if not np.all(np.isfinite(hess)):
                hess = np.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)
                hess = hess + 1e-6 * np.eye(self.theta_dim)

            accepted = False
            damping = 1e-6
            for _ in range(12):
                try:
                    step = np.linalg.solve(hess + damping * np.eye(self.theta_dim), grad)
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue

                candidate = unconstrained_theta - step
                cand_val = float(nlp(jnp.asarray(candidate, dtype=jnp.float64)))
                if np.isfinite(cand_val) and cand_val < val:
                    unconstrained_theta = candidate
                    accepted = True
                    break

                alpha = 0.5
                for _ in range(8):
                    candidate = unconstrained_theta - alpha * step
                    cand_val = float(nlp(jnp.asarray(candidate, dtype=jnp.float64)))
                    if np.isfinite(cand_val) and cand_val < val:
                        unconstrained_theta = candidate
                        accepted = True
                        break
                    alpha *= 0.5
                if accepted:
                    break
                damping *= 10.0

            if not accepted:
                gnorm = np.linalg.norm(grad)
                if gnorm == 0.0:
                    success = True
                    message = "Converged with zero gradient"
                    break
                candidate = unconstrained_theta - 0.05 * (grad / gnorm)
                cand_val = float(nlp(jnp.asarray(candidate, dtype=jnp.float64)))
                if np.isfinite(cand_val) and cand_val < val:
                    unconstrained_theta = candidate
                else:
                    message = "Line search stalled: unable to reduce objective"
                    break

        final_nlp, final_grad = value_grad_fn(jnp.asarray(unconstrained_theta, dtype=jnp.float64))
        final_nlp = float(final_nlp)
        final_grad = np.asarray(final_grad, dtype=float)
        final_grad_norm = float(np.linalg.norm(final_grad, ord=np.inf))

        hessian_unconstrained = compute_hessian(unconstrained_theta)
        hessian_unconstrained = np.nan_to_num(hessian_unconstrained, nan=0.0, posinf=0.0, neginf=0.0)
        hessian_unconstrained = stabilize_covariance(hessian_unconstrained, min_eig=1e-12)
        try:
            covariance_unconstrained = np.linalg.inv(hessian_unconstrained)
        except np.linalg.LinAlgError:
            covariance_unconstrained = np.linalg.pinv(hessian_unconstrained, hermitian=True)

        theta_map_jax, log_theta_map_jax, jac_log_u_jax = self._theta_log_and_jac_log_u_jax(
            jnp.asarray(unconstrained_theta, dtype=jnp.float64)
        )
        theta_map = np.asarray(theta_map_jax, dtype=float)
        log_theta_map = np.asarray(log_theta_map_jax, dtype=float)
        jac_log_u = np.asarray(jac_log_u_jax, dtype=float)

        covariance_log = jac_log_u @ covariance_unconstrained @ jac_log_u.T
        covariance_log = stabilize_covariance(covariance_log, min_eig=1e-14)
        try:
            hessian_log = np.linalg.inv(covariance_log)
        except np.linalg.LinAlgError:
            hessian_log = np.linalg.pinv(covariance_log, hermitian=True)

        jac_theta_log = self._jac_theta_prior_coordinate_numpy(theta_map)
        covariance_theta = jac_theta_log @ covariance_log @ jac_theta_log
        covariance_theta = stabilize_covariance(covariance_theta, min_eig=1e-20)
        correlation_theta = covariance_to_correlation(covariance_theta)

        predicted = self.predict_observables(theta_map)
        likelihood_setup = self._observable_likelihood_setup(observed_moments, covariance_moments)
        resid = predicted - np.asarray(observed_moments, dtype=float)
        gaussian_chi2 = float(resid @ likelihood_setup["gaussian_cov_inv"] @ resid)
        if bool(likelihood_setup["use_censored"]):
            censor_excess = self.log_dndv_censor_transition * np.logaddexp(
                0.0,
                (predicted - likelihood_setup["censored_upper"]) / self.log_dndv_censor_transition,
            )
            censored_chi2 = float(
                np.sum(
                    likelihood_setup["censored_mask"]
                    * (censor_excess / self.log_dndv_censor_sigma) ** 2
                )
            )
        else:
            censored_chi2 = 0.0
        chi2 = gaussian_chi2 + censored_chi2

        return MAPFitResult(
            success=success,
            message=message,
            n_iter=n_iter,
            start_index=int(start_index),
            num_starts=int(num_starts),
            unconstrained_theta_map=np.asarray(unconstrained_theta, dtype=float),
            log_theta_map=np.asarray(log_theta_map, dtype=float),
            theta_map=np.asarray(theta_map, dtype=float),
            predicted_moments=np.asarray(predicted, dtype=float),
            observed_moments=np.asarray(observed_moments, dtype=float),
            nlp=final_nlp,
            chi2=chi2,
            gradient_norm_inf=final_grad_norm,
            hessian_log=np.asarray(hessian_log, dtype=float),
            covariance_log=np.asarray(covariance_log, dtype=float),
            covariance_theta=np.asarray(covariance_theta, dtype=float),
            correlation_theta=np.asarray(correlation_theta, dtype=float),
            hessian_unconstrained=np.asarray(hessian_unconstrained, dtype=float),
            covariance_unconstrained=np.asarray(covariance_unconstrained, dtype=float),
        )

    def _build_map_start_points(
        self,
        initial_theta: Sequence[float],
        prior_mean_log: Sequence[float] | None,
        map_num_starts: int,
        seed: int,
    ) -> np.ndarray:
        """Construct deterministic MAP initial points in unconstrained coordinates."""
        n_starts = max(1, int(map_num_starts))
        starts: list[np.ndarray] = [self._unconstrained_from_theta_numpy(np.asarray(initial_theta, dtype=float))]

        prior_theta = (
            self._theta_from_prior_coordinate_numpy(self._coerce_prior_coordinate_numpy(prior_mean_log))
            if prior_mean_log is not None
            else self._default_theta_numpy()
        )
        starts.append(self._unconstrained_from_theta_numpy(prior_theta))

        rng = np.random.default_rng(int(seed))
        base = starts[0]
        jitter_scale = np.full((self.theta_dim,), 0.5, dtype=float)
        jitter_scale[:3] = np.asarray([0.8, 0.8, 0.7], dtype=float)
        if self.theta_dim >= 5:
            jitter_scale[4] = 0.25
        while len(starts) < n_starts:
            starts.append(base + rng.normal(loc=0.0, scale=jitter_scale, size=self.theta_dim))

        return np.asarray(starts[:n_starts], dtype=float)

    def _fit_map_from_nlp(
        self,
        nlp,
        observed_moments: np.ndarray,
        covariance_moments: np.ndarray,
        initial_theta: Sequence[float],
        max_iter: int,
        grad_tol: float,
        map_num_starts: int,
        seed: int,
        prior_mean_log: Sequence[float] | None,
    ) -> MAPFitResult:
        """Run multi-start MAP and select the best valid minimum."""
        starts = self._build_map_start_points(
            initial_theta=initial_theta,
            prior_mean_log=prior_mean_log,
            map_num_starts=map_num_starts,
            seed=seed,
        )

        results: list[MAPFitResult] = []
        failed_starts = 0
        for i, start in enumerate(starts):
            try:
                results.append(
                    self._fit_map_single_from_nlp(
                        nlp=nlp,
                        observed_moments=observed_moments,
                        covariance_moments=covariance_moments,
                        initial_unconstrained_theta=start,
                        max_iter=max_iter,
                        grad_tol=grad_tol,
                        start_index=i,
                        num_starts=int(starts.shape[0]),
                    )
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                failed_starts += 1

        if not results:
            raise np.linalg.LinAlgError("All MAP starts failed before producing a finite diagnostic result")

        finite = np.asarray([np.isfinite(r.nlp) and np.all(np.isfinite(r.theta_map)) for r in results], dtype=bool)
        pd = np.asarray(
            [
                np.all(np.linalg.eigvalsh(0.5 * (r.hessian_unconstrained + r.hessian_unconstrained.T)) > 0.0)
                for r in results
            ],
            dtype=bool,
        )

        valid_pd = finite & pd
        selection_mask = valid_pd if np.any(valid_pd) else finite
        if not np.any(selection_mask):
            selection_mask = np.ones(len(results), dtype=bool)

        candidate_indices = np.where(selection_mask)[0]
        best_local_idx = candidate_indices[np.argmin([results[i].nlp for i in candidate_indices])]
        best = results[int(best_local_idx)]

        quality = "PD minimum" if valid_pd[int(best_local_idx)] else "finite minimum (non-PD Hessian fallback)"
        if failed_starts:
            quality = f"{quality}; skipped {failed_starts} failed start(s)"
        best.message = f"{best.message}; selected start {best.start_index + 1}/{best.num_starts} [{quality}]"
        return best

    def fit_map(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
        initial_theta: Sequence[float] = (0.2, 0.2, 0.8),
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 0.8),
        max_iter: int = 25,
        grad_tol: float = 1e-5,
        map_num_starts: int = 4,
        seed: int = 0,
    ) -> MAPFitResult:
        """Fit the log-parameter MAP estimate with multi-start damped-Newton iterations."""
        nlp = self.make_negative_log_posterior(
            observed_moments=observed_moments,
            covariance_moments=covariance_moments,
            prior_mean_log=prior_mean_log,
            prior_sigma_log=prior_sigma_log,
            include_transform_jacobian=False,
        )
        return self._fit_map_from_nlp(
            nlp=nlp,
            observed_moments=np.asarray(observed_moments, dtype=float),
            covariance_moments=np.asarray(covariance_moments, dtype=float),
            initial_theta=initial_theta,
            max_iter=max_iter,
            grad_tol=grad_tol,
            map_num_starts=map_num_starts,
            seed=seed,
            prior_mean_log=prior_mean_log,
        )

    def _sample_hmc_from_nlp(
        self,
        nlp,
        initial_log_theta: np.ndarray,
        mass_diag: np.ndarray,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        leapfrog_steps: int,
        target_accept: float,
        seed: int,
    ) -> HMCResult:
        """Legacy single-chain HMC fallback sampler."""
        value_grad_fn = jax.jit(jax.value_and_grad(nlp))

        rng = np.random.default_rng(seed)
        q = np.asarray(initial_log_theta, dtype=float).copy()
        inv_mass_diag = 1.0 / np.clip(np.asarray(mass_diag, dtype=float), 1e-12, None)

        def potential_and_grad(q_vec: np.ndarray) -> tuple[float, np.ndarray]:
            val, grad = value_grad_fn(jnp.asarray(q_vec, dtype=jnp.float64))
            return float(val), np.asarray(grad, dtype=float)

        def kinetic(p_vec: np.ndarray) -> float:
            return 0.5 * float(np.sum(p_vec * p_vec * inv_mass_diag))

        total_steps = int(num_warmup) + int(num_samples)
        samples_log: list[np.ndarray] = []
        accepted = 0
        log_probs: list[float] = []

        eps = float(step_size)
        for i in range(total_steps):
            u0, grad_u0 = potential_and_grad(q)
            p0 = rng.normal(loc=0.0, scale=np.sqrt(1.0 / inv_mass_diag), size=q.shape[0])

            q_prop = q.copy()
            p_prop = p0.copy()

            if np.all(np.isfinite(grad_u0)):
                p_prop = p_prop - 0.5 * eps * grad_u0
            else:
                p_prop[:] = np.nan

            valid_path = np.all(np.isfinite(p_prop))
            u_prop = np.inf
            grad_u_prop = np.zeros_like(grad_u0)

            for leap_idx in range(int(leapfrog_steps)):
                if not valid_path:
                    break
                q_prop = q_prop + eps * inv_mass_diag * p_prop
                u_prop, grad_u_prop = potential_and_grad(q_prop)
                if not np.isfinite(u_prop) or not np.all(np.isfinite(grad_u_prop)):
                    valid_path = False
                    break
                if leap_idx != int(leapfrog_steps) - 1:
                    p_prop = p_prop - eps * grad_u_prop

            if valid_path:
                p_prop = p_prop - 0.5 * eps * grad_u_prop
                p_prop = -p_prop

            h0 = u0 + kinetic(p0)
            h1 = u_prop + kinetic(p_prop) if valid_path else np.inf
            log_alpha = min(0.0, h0 - h1)
            alpha = float(np.exp(log_alpha)) if np.isfinite(log_alpha) else 0.0

            if rng.random() < alpha:
                q = q_prop
                u0 = u_prop
                accepted += 1

            if i < num_warmup:
                adapt_gain = 1.0 / np.sqrt(float(i) + 1.0)
                eps = float(np.clip(eps * np.exp(adapt_gain * (alpha - target_accept)), 1e-5, 0.5))
            else:
                samples_log.append(q.copy())
                log_probs.append(-u0)

        samples_u_arr = np.asarray(samples_log, dtype=float)
        samples_theta = self._theta_from_unconstrained_numpy(samples_u_arr)
        samples_log_arr = self._prior_coordinate_from_theta_numpy(samples_theta)
        n_kept = int(samples_u_arr.shape[0])

        if samples_u_arr.shape[0] > 1:
            covariance_log = stabilize_covariance(np.cov(samples_log_arr.T), min_eig=1e-12)
            covariance_theta = stabilize_covariance(np.cov(samples_theta.T), min_eig=1e-12)
        else:
            covariance_log = np.eye(self.theta_dim)
            covariance_theta = np.eye(self.theta_dim)

        return HMCResult(
            samples_unconstrained=samples_u_arr,
            samples_log=samples_log_arr,
            samples_theta=samples_theta,
            samples_nuts_coordinate=samples_u_arr,
            sample_diverging=np.zeros((n_kept,), dtype=bool),
            sample_accept_prob=np.full((n_kept,), np.nan, dtype=float),
            sample_num_steps=np.full((n_kept,), float(leapfrog_steps), dtype=float),
            sample_energy=np.full((n_kept,), np.nan, dtype=float),
            sample_potential_energy=np.full((n_kept,), np.nan, dtype=float),
            sampler="hmc",
            num_chains=1,
            acceptance_rate=float(accepted / max(total_steps, 1)),
            acceptance_rate_per_chain=np.asarray([accepted / max(total_steps, 1)], dtype=float),
            final_step_size=float(eps),
            num_divergent=0,
            num_steps_mean=float(leapfrog_steps),
            num_steps_max=float(leapfrog_steps),
            tree_depth_mean=float(np.log2(max(leapfrog_steps, 1))),
            mean_log_posterior=float(np.mean(log_probs)) if log_probs else float("nan"),
            energy_mean=float("nan"),
            energy_var=float("nan"),
            potential_energy_mean=float("nan"),
            bfmi=None,
            r_hat=None,
            ess_bulk=None,
            covariance_log=covariance_log,
            covariance_theta=covariance_theta,
            correlation_theta=covariance_to_correlation(covariance_theta),
            nuts_chain_method="hmc",
            nuts_dense_mass=None,
            nuts_max_tree_depth=None,
            nuts_coordinate="native",
            num_divergent_per_chain=np.asarray([0], dtype=float),
            bfmi_per_chain=None,
        )

    def _sample_nuts_from_nlp(
        self,
        nlp,
        initial_log_theta: np.ndarray,
        whitening_center: np.ndarray,
        whitening_scale: np.ndarray,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        target_accept: float,
        num_chains: int,
        chain_method: str,
        dense_mass: bool,
        max_tree_depth: int,
        progress_bar: bool,
        nuts_coordinate: str,
        seed: int,
    ) -> HMCResult:
        """Sample posterior with NumPyro NUTS."""
        try:
            import numpyro
            from numpyro.diagnostics import summary as numpyro_summary
            from numpyro.infer import MCMC, NUTS
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "NumPyro is required for sampler='nuts'. Install it with `python -m pip install numpyro`."
            ) from exc

        n_chains = max(1, int(num_chains))
        resolved_chain_method = resolve_nuts_chain_method(n_chains, chain_method)
        coordinate = nuts_coordinate.strip().lower()
        if coordinate not in {"native", "map_whitened"}:
            raise ValueError("nuts_coordinate must be one of {'native', 'map_whitened'}")

        center = jnp.asarray(whitening_center, dtype=jnp.float64)
        scale = jnp.asarray(whitening_scale, dtype=jnp.float64)

        def coordinate_to_unconstrained(q_theta):
            if coordinate == "map_whitened":
                return center + scale @ q_theta
            return q_theta

        def potential_fn(params):
            return nlp(coordinate_to_unconstrained(params["q_theta"]))

        nuts_kernel = NUTS(
            potential_fn=potential_fn,
            target_accept_prob=float(target_accept),
            step_size=float(step_size),
            dense_mass=bool(dense_mass),
            max_tree_depth=int(max_tree_depth),
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=n_chains,
            progress_bar=bool(progress_bar),
            chain_method=resolved_chain_method,
        )
        rng_key = jax.random.PRNGKey(int(seed))
        init_log = jnp.asarray(initial_log_theta, dtype=jnp.float64)
        init_q = jnp.zeros_like(init_log) if coordinate == "map_whitened" else init_log
        if n_chains > 1:
            init_q = jnp.broadcast_to(init_q, (n_chains, init_q.shape[0]))
        mcmc.run(
            rng_key,
            init_params={"q_theta": init_q},
            extra_fields=("accept_prob", "num_steps", "diverging", "energy", "potential_energy"),
        )

        samples_q_by_chain = mcmc.get_samples(group_by_chain=True)["q_theta"]
        samples_q_by_chain = np.asarray(samples_q_by_chain, dtype=float)
        if coordinate == "map_whitened":
            center_np = np.asarray(whitening_center, dtype=float)
            scale_np = np.asarray(whitening_scale, dtype=float)
            samples_u_by_chain = center_np + np.einsum("csd,ed->cse", samples_q_by_chain, scale_np)
        else:
            samples_u_by_chain = samples_q_by_chain
        samples_unconstrained = samples_u_by_chain.reshape(-1, samples_u_by_chain.shape[-1])
        samples_nuts_coordinate = samples_q_by_chain.reshape(-1, samples_q_by_chain.shape[-1])
        samples_theta = self._theta_from_unconstrained_numpy(samples_unconstrained)
        samples_log = self._prior_coordinate_from_theta_numpy(samples_theta)

        extra = mcmc.get_extra_fields(group_by_chain=True)
        accept_prob = np.asarray(extra.get("accept_prob"), dtype=float)
        diverging = np.asarray(extra.get("diverging"), dtype=bool)
        num_steps = np.asarray(extra.get("num_steps"), dtype=float)
        energy = np.asarray(extra.get("energy"), dtype=float) if "energy" in extra else np.asarray([])
        potential_energy = (
            np.asarray(extra.get("potential_energy"), dtype=float) if "potential_energy" in extra else np.asarray([])
        )
        sample_accept_prob = accept_prob.reshape(-1) if accept_prob.size else np.full((samples_theta.shape[0],), np.nan)
        sample_diverging = diverging.reshape(-1) if diverging.size else np.zeros((samples_theta.shape[0],), dtype=bool)
        sample_num_steps = num_steps.reshape(-1) if num_steps.size else np.full((samples_theta.shape[0],), np.nan)
        sample_energy = energy.reshape(-1) if energy.size else np.full((samples_theta.shape[0],), np.nan)
        sample_potential_energy = (
            potential_energy.reshape(-1) if potential_energy.size else np.full((samples_theta.shape[0],), np.nan)
        )

        acceptance_rate_per_chain = np.mean(accept_prob, axis=1) if accept_prob.size else np.full((n_chains,), np.nan)
        acceptance_rate = float(np.mean(accept_prob)) if accept_prob.size else float("nan")
        num_divergent = int(np.sum(diverging)) if diverging.size else 0
        num_divergent_per_chain = (
            np.sum(diverging, axis=1).astype(float) if diverging.ndim == 2 else np.full((n_chains,), np.nan)
        )
        num_steps_mean = float(np.mean(num_steps)) if num_steps.size else float("nan")
        num_steps_max = float(np.max(num_steps)) if num_steps.size else float("nan")
        tree_depth_mean = float(np.mean(np.log2(np.maximum(num_steps, 1.0)))) if num_steps.size else float("nan")

        step_state = getattr(getattr(mcmc.last_state, "adapt_state", None), "step_size", np.nan)
        final_step_size = float(np.mean(np.asarray(step_state, dtype=float)))

        sample_dict = {"q_theta": samples_q_by_chain}
        diag = numpyro_summary(sample_dict, group_by_chain=True)["q_theta"]
        r_hat = np.asarray(diag.get("r_hat"), dtype=float)
        ess_bulk = np.asarray(diag.get("n_eff"), dtype=float)

        if samples_log.shape[0] > 1:
            covariance_log = stabilize_covariance(np.cov(samples_log.T), min_eig=1e-12)
            covariance_theta = stabilize_covariance(np.cov(samples_theta.T), min_eig=1e-12)
        else:
            covariance_log = np.eye(self.theta_dim)
            covariance_theta = np.eye(self.theta_dim)

        nlp_batch = jax.vmap(lambda x: nlp(x))
        mean_log_posterior = float(-jnp.mean(nlp_batch(jnp.asarray(samples_unconstrained, dtype=jnp.float64))))

        if energy.size:
            energy_mean = float(np.mean(energy))
            energy_var = float(np.var(energy))
            deltas = np.diff(energy, axis=1) if energy.ndim == 2 and energy.shape[1] > 1 else np.asarray([])
            if deltas.size and np.all(np.isfinite(deltas)) and energy_var > 0.0:
                bfmi_chain = np.mean(deltas * deltas, axis=1) / (np.var(energy, axis=1) + 1e-30)
                bfmi = float(np.mean(bfmi_chain))
            else:
                bfmi_chain = np.full((n_chains,), np.nan)
                bfmi = float("nan")
        else:
            energy_mean = float("nan")
            energy_var = float("nan")
            bfmi_chain = np.full((n_chains,), np.nan)
            bfmi = float("nan")

        potential_energy_mean = float(np.mean(potential_energy)) if potential_energy.size else float("nan")

        return HMCResult(
            samples_unconstrained=samples_unconstrained,
            samples_log=samples_log,
            samples_theta=samples_theta,
            samples_nuts_coordinate=samples_nuts_coordinate,
            sample_diverging=np.asarray(sample_diverging, dtype=bool),
            sample_accept_prob=np.asarray(sample_accept_prob, dtype=float),
            sample_num_steps=np.asarray(sample_num_steps, dtype=float),
            sample_energy=np.asarray(sample_energy, dtype=float),
            sample_potential_energy=np.asarray(sample_potential_energy, dtype=float),
            sampler="nuts",
            num_chains=n_chains,
            acceptance_rate=acceptance_rate,
            acceptance_rate_per_chain=acceptance_rate_per_chain,
            final_step_size=final_step_size,
            num_divergent=num_divergent,
            num_steps_mean=num_steps_mean,
            num_steps_max=num_steps_max,
            tree_depth_mean=tree_depth_mean,
            mean_log_posterior=mean_log_posterior,
            energy_mean=energy_mean,
            energy_var=energy_var,
            potential_energy_mean=potential_energy_mean,
            bfmi=bfmi,
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            covariance_log=covariance_log,
            covariance_theta=covariance_theta,
            correlation_theta=covariance_to_correlation(covariance_theta),
            nuts_chain_method=resolved_chain_method,
            nuts_dense_mass=bool(dense_mass),
            nuts_max_tree_depth=int(max_tree_depth),
            nuts_coordinate=coordinate,
            num_divergent_per_chain=num_divergent_per_chain,
            bfmi_per_chain=bfmi_chain,
        )

    def _sample_posterior(
        self,
        nlp,
        initial_log_theta: np.ndarray,
        mass_diag: np.ndarray,
        covariance_unconstrained: np.ndarray,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        leapfrog_steps: int,
        target_accept: float,
        num_chains: int,
        nuts_chain_method: str,
        nuts_dense_mass: bool,
        nuts_max_tree_depth: int,
        nuts_progress_bar: bool,
        nuts_coordinate: str,
        seed: int,
        sampler: str,
    ) -> HMCResult:
        """Dispatch posterior sampling backend."""
        sampler_key = sampler.strip().lower()
        if sampler_key == "nuts":
            covariance_u = stabilize_covariance(np.asarray(covariance_unconstrained, dtype=float), min_eig=1e-10)
            whitening_scale = np.linalg.cholesky(covariance_u)
            return self._sample_nuts_from_nlp(
                nlp=nlp,
                initial_log_theta=initial_log_theta,
                whitening_center=np.asarray(initial_log_theta, dtype=float),
                whitening_scale=whitening_scale,
                num_warmup=num_warmup,
                num_samples=num_samples,
                step_size=step_size,
                target_accept=target_accept,
                num_chains=max(1, int(num_chains)),
                chain_method=nuts_chain_method,
                dense_mass=bool(nuts_dense_mass),
                max_tree_depth=int(nuts_max_tree_depth),
                progress_bar=nuts_progress_bar,
                nuts_coordinate=nuts_coordinate,
                seed=seed,
            )
        if sampler_key == "hmc":
            return self._sample_hmc_from_nlp(
                nlp=nlp,
                initial_log_theta=initial_log_theta,
                mass_diag=mass_diag,
                num_warmup=num_warmup,
                num_samples=num_samples,
                step_size=step_size,
                leapfrog_steps=leapfrog_steps,
                target_accept=target_accept,
                seed=seed,
            )
        raise ValueError(f"Unknown sampler '{sampler}'. Expected 'nuts' or 'hmc'.")

    def fit_posterior(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
        initial_theta: Sequence[float] = (0.2, 0.2, 0.8),
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 0.8),
        map_max_iter: int = 25,
        map_grad_tol: float = 1e-5,
        map_num_starts: int = 4,
        hmc_num_warmup: int = 250,
        hmc_num_samples: int = 500,
        hmc_step_size: float = 0.02,
        hmc_leapfrog_steps: int = 12,
        hmc_target_accept: float = 0.70,
        sampler: str = "nuts",
        num_chains: int = 1,
        nuts_chain_method: str = "auto",
        nuts_dense_mass: bool = False,
        nuts_max_tree_depth: int = 10,
        nuts_coordinate: str = "native",
        nuts_progress_bar: bool = False,
        status_callback: Callable[[str], None] | None = None,
        seed: int = 0,
    ) -> PosteriorFitResult:
        """Run MAP + posterior sampling workflow for configured observables."""
        if int(nuts_max_tree_depth) < 1:
            raise ValueError("nuts_max_tree_depth must be >= 1")
        nuts_coordinate_key = nuts_coordinate.strip().lower()
        if nuts_coordinate_key not in {"native", "map_whitened"}:
            raise ValueError("nuts_coordinate must be one of {'native', 'map_whitened'}")
        total_start = time.perf_counter()
        if status_callback is not None:
            status_callback("Building MAP and sampler negative log-posteriors.")

        nlp_map = self.make_negative_log_posterior(
            observed_moments=observed_moments,
            covariance_moments=covariance_moments,
            prior_mean_log=prior_mean_log,
            prior_sigma_log=prior_sigma_log,
            include_transform_jacobian=False,
        )
        nlp_sampler = self.make_negative_log_posterior(
            observed_moments=observed_moments,
            covariance_moments=covariance_moments,
            prior_mean_log=prior_mean_log,
            prior_sigma_log=prior_sigma_log,
            include_transform_jacobian=True,
        )

        if status_callback is not None:
            status_callback("Starting MAP optimization.")
        map_start = time.perf_counter()
        map_result = self._fit_map_from_nlp(
            nlp=nlp_map,
            observed_moments=np.asarray(observed_moments, dtype=float),
            covariance_moments=np.asarray(covariance_moments, dtype=float),
            initial_theta=initial_theta,
            max_iter=map_max_iter,
            grad_tol=map_grad_tol,
            map_num_starts=map_num_starts,
            seed=seed + 17,
            prior_mean_log=prior_mean_log,
        )
        map_elapsed = time.perf_counter() - map_start

        mass_diag = np.diag(stabilize_covariance(map_result.covariance_unconstrained, min_eig=1e-10))

        if status_callback is not None:
            sampler_name = sampler.strip().lower()
            status_callback(
                f"MAP complete in {map_elapsed:.2f} s. Starting {sampler_name.upper()} posterior sampling."
            )
        sample_start = time.perf_counter()
        hmc_result = self._sample_posterior(
            nlp=nlp_sampler,
            initial_log_theta=map_result.unconstrained_theta_map,
            mass_diag=mass_diag,
            covariance_unconstrained=map_result.covariance_unconstrained,
            num_warmup=hmc_num_warmup,
            num_samples=hmc_num_samples,
            step_size=hmc_step_size,
            leapfrog_steps=hmc_leapfrog_steps,
            target_accept=hmc_target_accept,
            num_chains=num_chains,
            nuts_chain_method=nuts_chain_method,
            nuts_dense_mass=nuts_dense_mass,
            nuts_max_tree_depth=nuts_max_tree_depth,
            nuts_progress_bar=nuts_progress_bar,
            nuts_coordinate=nuts_coordinate_key,
            seed=seed,
            sampler=sampler,
        )
        sample_elapsed = time.perf_counter() - sample_start
        total_elapsed = time.perf_counter() - total_start

        runtimes = {
            "map": float(map_elapsed),
            "posterior_sampling": float(sample_elapsed),
            "total": float(total_elapsed),
        }
        if status_callback is not None:
            status_callback(
                "Posterior sampling complete in "
                f"{sample_elapsed:.2f} s (total {total_elapsed:.2f} s)."
            )

        return PosteriorFitResult(
            map=map_result,
            hmc=hmc_result,
            observed_moments=np.asarray(observed_moments, dtype=float),
            covariance_moments=stabilize_covariance(np.asarray(covariance_moments, dtype=float)),
            runtime_seconds=runtimes,
        )

    def predict_observables_for_log_samples(self, samples_log: np.ndarray, max_samples: int = 512) -> np.ndarray:
        """Evaluate configured observables for a subset of posterior log-parameter samples."""
        arr = np.asarray(samples_log, dtype=float)
        if arr.ndim == 3 and arr.shape[-1] == self.theta_dim:
            arr = arr.reshape(-1, self.theta_dim)
        if arr.ndim != 2 or arr.shape[1] != self.theta_dim:
            raise ValueError(
                f"samples_log must have shape (N, {self.theta_dim}) or (chains, N, {self.theta_dim})"
            )

        if arr.shape[0] > max_samples:
            idx = np.linspace(0, arr.shape[0] - 1, max_samples, dtype=int)
            arr = arr[idx]

        batch = jnp.asarray(arr, dtype=jnp.float64)
        predict_batch = jax.jit(jax.vmap(self._predict_log_theta_fn, in_axes=0))
        return np.asarray(predict_batch(batch), dtype=float)

    def predict_moments_for_log_samples(self, samples_log: np.ndarray, max_samples: int = 512) -> np.ndarray:
        """Backward-compatible alias for `predict_observables_for_log_samples`."""
        return self.predict_observables_for_log_samples(samples_log=samples_log, max_samples=max_samples)


def plot_corner(
    samples_theta: np.ndarray,
    labels: Sequence[str],
    output_path: str,
    truths: Sequence[float] | None = None,
    map_theta: Sequence[float] | None = None,
) -> None:
    """Create a corner plot with dense occupancy shading and optional KDE contours."""
    x = np.asarray(samples_theta, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(labels):
        raise ValueError("samples_theta must have shape (N, D) matching labels")

    d = x.shape[1]
    n_points = x.shape[0]
    fig, axes = plt.subplots(d, d, figsize=(3.1 * d, 3.1 * d), constrained_layout=True)
    kde_ready = False
    gaussian_kde = None
    if n_points >= 1000:
        try:
            from scipy.stats import gaussian_kde as scipy_gaussian_kde
        except Exception:
            kde_ready = False
        else:
            gaussian_kde = scipy_gaussian_kde
            kde_ready = True

    truths_arr = None if truths is None else np.asarray(truths, dtype=float)
    map_arr = None if map_theta is None else np.asarray(map_theta, dtype=float)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue

            if i == j:
                ax.hist(x[:, j], bins=45, color="tab:blue", alpha=0.85, density=True)
                if truths_arr is not None:
                    ax.axvline(truths_arr[j], color="tab:green", lw=1.5, ls="--")
                if map_arr is not None:
                    ax.axvline(map_arr[j], color="tab:red", lw=1.5)
            else:
                counts, xedges, yedges, _ = ax.hist2d(
                    x[:, j],
                    x[:, i],
                    bins=45,
                    cmap="Blues",
                    cmin=1,
                )
                if np.any(np.isfinite(counts)) and np.nanmax(counts) > 0.0:
                    xmid = 0.5 * (xedges[:-1] + xedges[1:])
                    ymid = 0.5 * (yedges[:-1] + yedges[1:])
                    level_hi = 0.6 * np.nanmax(counts)
                    level_mid = 0.3 * np.nanmax(counts)
                    levels = [lvl for lvl in (level_mid, level_hi) if lvl > 0.0]
                    if levels:
                        ax.contour(
                            xmid,
                            ymid,
                            counts.T,
                            levels=levels,
                            colors="tab:blue",
                            linewidths=1.0,
                        )
                if kde_ready and gaussian_kde is not None:
                    n_kde = min(n_points, 5000)
                    idx_kde = np.linspace(0, n_points - 1, n_kde, dtype=int)
                    xy = np.vstack([x[idx_kde, j], x[idx_kde, i]])
                    x_min, x_max = float(np.min(xy[0])), float(np.max(xy[0]))
                    y_min, y_max = float(np.min(xy[1])), float(np.max(xy[1]))
                    if x_max > x_min and y_max > y_min:
                        pad_x = 0.05 * (x_max - x_min)
                        pad_y = 0.05 * (y_max - y_min)
                        x_eval = np.linspace(x_min - pad_x, x_max + pad_x, 70, dtype=float)
                        y_eval = np.linspace(y_min - pad_y, y_max + pad_y, 70, dtype=float)
                        grid_x, grid_y = np.meshgrid(x_eval, y_eval)
                        try:
                            kde = gaussian_kde(xy)
                            grid_z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
                            z_flat = grid_z[np.isfinite(grid_z)]
                            if z_flat.size > 0 and np.nanmax(z_flat) > 0.0:
                                q_levels = np.quantile(z_flat, [0.70, 0.88, 0.97])
                                q_levels = np.unique(q_levels[q_levels > 0.0])
                                if q_levels.size > 0:
                                    ax.contour(
                                        grid_x,
                                        grid_y,
                                        grid_z,
                                        levels=q_levels,
                                        colors="navy",
                                        linewidths=1.1,
                                    )
                        except Exception:
                            pass
                if n_points > 1200:
                    idx = np.linspace(0, n_points - 1, 1200, dtype=int)
                    overlay = x[idx]
                else:
                    overlay = x
                ax.scatter(overlay[:, j], overlay[:, i], s=4, alpha=0.10, color="black", edgecolors="none")
                if truths_arr is not None:
                    ax.plot(truths_arr[j], truths_arr[i], marker="x", color="tab:green", ms=7, mew=1.5)
                if map_arr is not None:
                    ax.plot(map_arr[j], map_arr[i], marker="o", color="tab:red", ms=4)

            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.set_yticklabels([])

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_observable_fit(
    observed_values: np.ndarray,
    covariance_values: np.ndarray,
    map_values: np.ndarray,
    posterior_samples: np.ndarray | None,
    labels: Sequence[str],
    output_path: str,
    yscale: str = "linear",
) -> None:
    """Plot observed observables with uncertainties against MAP and posterior predictive summaries."""
    obs = np.asarray(observed_values, dtype=float)
    cov = np.asarray(covariance_values, dtype=float)
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    map_pred = np.asarray(map_values, dtype=float)
    labels_arr = list(labels)

    if obs.shape != map_pred.shape:
        raise ValueError("observed_values and map_values must have the same shape")
    if obs.ndim != 1:
        raise ValueError("observed_values must be 1D")
    if cov.shape != (obs.size, obs.size):
        raise ValueError("covariance_values shape must match observable dimension")
    if len(labels_arr) != obs.size:
        raise ValueError("labels length must match observable dimension")

    x = np.arange(obs.size)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.errorbar(x, obs, yerr=sigma, fmt="o", color="black", lw=1.5, capsize=4, label="Observed")
    ax.scatter(x, map_pred, marker="s", s=45, color="tab:red", label="MAP prediction")

    if posterior_samples is not None and posterior_samples.size > 0:
        samp = np.asarray(posterior_samples, dtype=float)
        if samp.ndim != 2 or samp.shape[1] != obs.size:
            raise ValueError("posterior_samples must have shape (N, D) where D=len(labels)")
        p16 = np.percentile(samp, 16, axis=0)
        p50 = np.percentile(samp, 50, axis=0)
        p84 = np.percentile(samp, 84, axis=0)
        yerr_low = p50 - p16
        yerr_high = p84 - p50
        ax.errorbar(
            x,
            p50,
            yerr=np.vstack([yerr_low, yerr_high]),
            fmt="^",
            color="tab:blue",
            lw=1.4,
            capsize=4,
            label="Posterior predictive",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_arr, rotation=18 if len(labels_arr) > 4 else 0, ha="right" if len(labels_arr) > 4 else "center")
    if yscale == "log" and np.all(obs > 0.0) and np.all(map_pred > 0.0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
    ax.set_ylabel("Observable value")
    ax.set_title("Observed vs fitted observables")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dndv_fit(
    velocity_bins_kms: np.ndarray,
    observed_dndv: np.ndarray,
    covariance_dndv: np.ndarray,
    map_dndv: np.ndarray,
    posterior_dndv_samples: np.ndarray | None,
    output_path: str,
) -> None:
    """Plot binned dN/dv with uncertainty, MAP, and posterior predictive envelope."""
    v = np.asarray(velocity_bins_kms, dtype=float)
    obs = np.asarray(observed_dndv, dtype=float)
    cov = np.asarray(covariance_dndv, dtype=float)
    map_pred = np.asarray(map_dndv, dtype=float)

    if v.ndim != 1 or obs.ndim != 1 or map_pred.ndim != 1:
        raise ValueError("velocity_bins_kms, observed_dndv, and map_dndv must be 1D")
    if obs.shape != v.shape or map_pred.shape != v.shape:
        raise ValueError("velocity_bins_kms, observed_dndv, and map_dndv must have matching shapes")
    if cov.shape != (v.size, v.size):
        raise ValueError("covariance_dndv shape must match velocity bin count")

    sigma = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    obs_floor = np.maximum(obs, 1e-40)
    map_floor = np.maximum(map_pred, 1e-40)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(v, obs_floor, color="black", lw=1.3, marker="o", ms=3.5, label="Observed")
    ax.fill_between(
        v,
        np.maximum(obs_floor - sigma, 1e-40),
        np.maximum(obs_floor + sigma, 1e-40),
        color="black",
        alpha=0.14,
        linewidth=0.0,
        label="Observed 1-sigma",
    )
    ax.plot(v, map_floor, color="tab:red", lw=1.8, label="MAP prediction")

    if posterior_dndv_samples is not None and posterior_dndv_samples.size > 0:
        samp = np.asarray(posterior_dndv_samples, dtype=float)
        if samp.ndim != 2 or samp.shape[1] != v.size:
            raise ValueError("posterior_dndv_samples must have shape (N, num_bins)")
        p16 = np.maximum(np.percentile(samp, 16, axis=0), 1e-40)
        p50 = np.maximum(np.percentile(samp, 50, axis=0), 1e-40)
        p84 = np.maximum(np.percentile(samp, 84, axis=0), 1e-40)
        ax.fill_between(v, p16, p84, color="tab:blue", alpha=0.20, linewidth=0.0, label="Posterior 16-84%")
        ax.plot(v, p50, color="tab:blue", lw=1.4, ls="--", label="Posterior median")

    ax.set_yscale("log")
    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel(r"dN/dv [cm$^{-2}$ / (km s$^{-1}$)]")
    ax.set_title("Binned dN/dv: observed vs fitted")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_moment_fit(
    observed_moments: np.ndarray,
    covariance_moments: np.ndarray,
    map_moments: np.ndarray,
    posterior_moment_samples: np.ndarray | None,
    output_path: str,
) -> None:
    """Backward-compatible moment-fit helper for [M0, M1, M2]."""
    plot_observable_fit(
        observed_values=observed_moments,
        covariance_values=covariance_moments,
        map_values=map_moments,
        posterior_samples=posterior_moment_samples,
        labels=("M0", "M1", "M2"),
        output_path=output_path,
        yscale="log",
    )
