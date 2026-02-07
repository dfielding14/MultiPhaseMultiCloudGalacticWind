"""Inference utilities for fitting wind parameters to observed dN/dv moments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .config import WindConfig
from .constants import Msun, Z_solar, gamma, kb, kpc, mp, yr
from .jax_physics import JaxWindParams, integrate_wind_rk4_scan
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
    if sigma.shape != (3,):
        raise ValueError(f"std must be length-3 for [M0, M1, M2], got shape {sigma.shape}")
    if np.any(sigma <= 0.0):
        raise ValueError("All standard deviations must be positive")

    if corr is None:
        corr_m = np.eye(3, dtype=float)
    else:
        corr_m = np.asarray(corr, dtype=float)
        if corr_m.shape != (3, 3):
            raise ValueError(f"corr must be shape (3, 3), got {corr_m.shape}")
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

    jitter = max(min_eig, 1e-16)
    for _ in range(6):
        try:
            eigvals, eigvecs = np.linalg.eigh(sym)
            eigvals_clipped = np.clip(eigvals, min_eig, None)
            return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        except np.linalg.LinAlgError:
            sym = sym + jitter * np.eye(n)
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


@dataclass
class HMCResult:
    samples_log: np.ndarray
    samples_theta: np.ndarray
    sampler: str
    num_chains: int
    acceptance_rate: float
    acceptance_rate_per_chain: np.ndarray | None
    final_step_size: float
    num_divergent: int
    num_steps_mean: float
    mean_log_posterior: float
    r_hat: np.ndarray | None
    ess_bulk: np.ndarray | None
    covariance_log: np.ndarray
    covariance_theta: np.ndarray
    correlation_theta: np.ndarray


@dataclass
class PosteriorFitResult:
    map: MAPFitResult
    hmc: HMCResult
    observed_moments: np.ndarray
    covariance_moments: np.ndarray


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


class MomentInferenceModel:
    """Autodiff-enabled inference model for [M0, M1, M2] observables."""

    PARAM_NAMES = ("eta_M", "eta_M_cold", "eta_E")

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

        self.sfr = float(sfr)
        self.r_star_kpc = float(r_star_kpc)
        self.v_circ = float(v_circ)
        self.r_max_kpc = float(r_max_kpc)
        self.r_obs_min_kpc = float(r_obs_min_kpc)

        self.config = WindConfig() if config is None else config
        self.config.validate()

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
        self._predict_theta_fn = jax.jit(lambda theta: self._predict_theta_with_valid_fn(theta)[0])
        self._predict_log_theta_fn = jax.jit(lambda log_theta: self._predict_theta_fn(jnp.exp(log_theta)))

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

        r_obs_min_kpc = self.r_obs_min_kpc
        r_obs_max_kpc = self.r_max_kpc

        @jax.jit
        def build_state_and_params(theta):
            eta_m = jnp.maximum(theta[0], 1e-12)
            eta_m_cold = jnp.maximum(theta[1], 1e-12)
            eta_e = jnp.maximum(theta[2], 1e-12)

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

        @jax.jit
        def predict_theta_with_valid(theta):
            y0, params = build_state_and_params(theta)
            y = integrate_wind_rk4_scan(r_grid, y0, params)

            finite = jnp.all(jnp.isfinite(y), axis=1)
            physical = (y[:, 0] > 0.0) & (y[:, 1] > 0.0) & (y[:, 2] > 0.0)
            valid_state = jnp.all(finite & physical)

            m_cloud = y[:, 4 : 4 + n_species]
            v_cloud = y[:, 4 + n_species : 4 + 2 * n_species]

            injection = jnp.where(
                r_grid < params.injection_radius,
                (r_grid / jnp.maximum(params.injection_radius, 1e-30)) ** params.injection_power,
                1.0,
            )
            ndot = params.Ndot_cloud0[None, :] * injection[:, None]
            v_cloud_safe = jnp.maximum(v_cloud, params.v_cloud_min * 1e5)
            denom = params.Omwind * (r_grid[:, None] ** 2) * v_cloud_safe
            rho_cloud = ndot * m_cloud / jnp.maximum(denom, 1e-60)
            n_h = rho_cloud / (1.4 * mp)

            active = m_cloud >= params.M_cloud_min
            r_kpc = r_grid / kpc
            radial_window = ((r_kpc >= r_obs_min_kpc) & (r_kpc <= r_obs_max_kpc))[:, None]
            weight = (active & radial_window).astype(jnp.float64)

            v_kms = v_cloud / 1e5
            n_h_eff = n_h * weight

            m0 = jnp.sum(jnp.trapezoid(n_h_eff, r_grid, axis=0))
            m1 = jnp.sum(jnp.trapezoid(n_h_eff * v_kms, r_grid, axis=0))
            m2 = jnp.sum(jnp.trapezoid(n_h_eff * v_kms * v_kms, r_grid, axis=0))
            moments = jnp.asarray([m0, m1, m2], dtype=jnp.float64)

            valid_moments = jnp.all(jnp.isfinite(moments)) & jnp.all(moments > 0.0)
            valid = valid_state & valid_moments

            fallback = jnp.asarray([1e-30, 1e-20, 1e-10], dtype=jnp.float64)
            moments_safe = jnp.where(valid, moments, fallback)
            return moments_safe, jnp.where(valid, 1.0, 0.0)

        return predict_theta_with_valid

    def predict_moments(self, theta: Sequence[float]) -> np.ndarray:
        """Predict [M0, M1, M2] for linear-space parameters [eta_M, eta_M_cold, eta_E]."""
        theta_arr = jnp.asarray(theta, dtype=jnp.float64)
        return np.asarray(self._predict_theta_fn(theta_arr), dtype=float)

    def predict_moments_log(self, log_theta: Sequence[float]) -> np.ndarray:
        """Predict [M0, M1, M2] for log-space parameters log([eta_M, eta_M_cold, eta_E])."""
        log_theta_arr = jnp.asarray(log_theta, dtype=jnp.float64)
        return np.asarray(self._predict_log_theta_fn(log_theta_arr), dtype=float)

    def make_negative_log_posterior(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 1.0),
        invalid_penalty: float = 1e6,
    ):
        """Create JAX-jitted negative log posterior over log-parameters."""
        y_obs = jnp.asarray(observed_moments, dtype=jnp.float64)
        if y_obs.shape != (3,):
            raise ValueError("observed_moments must be length-3 [M0, M1, M2]")

        cov = stabilize_covariance(np.asarray(covariance_moments, dtype=float))
        cov_inv = jnp.asarray(np.linalg.inv(cov), dtype=jnp.float64)

        if prior_mean_log is None:
            prior_mean_log_arr = np.log(np.asarray([0.2, 0.2, 1.0], dtype=float))
        else:
            prior_mean_log_arr = np.asarray(prior_mean_log, dtype=float)
        prior_sigma_log_arr = np.asarray(prior_sigma_log, dtype=float)

        if prior_mean_log_arr.shape != (3,) or prior_sigma_log_arr.shape != (3,):
            raise ValueError("prior_mean_log and prior_sigma_log must be length-3")
        if np.any(prior_sigma_log_arr <= 0.0):
            raise ValueError("All prior_sigma_log entries must be positive")

        prior_mean_log_jax = jnp.asarray(prior_mean_log_arr, dtype=jnp.float64)
        prior_sigma_log_jax = jnp.asarray(prior_sigma_log_arr, dtype=jnp.float64)

        predict_theta_with_valid = self._predict_theta_with_valid_fn

        @jax.jit
        def nlp(log_theta):
            theta = jnp.exp(log_theta)
            moments, valid = predict_theta_with_valid(theta)
            resid = moments - y_obs
            chi2 = resid @ cov_inv @ resid
            prior_chi2 = jnp.sum(((log_theta - prior_mean_log_jax) / prior_sigma_log_jax) ** 2)
            penalty = jnp.where(valid > 0.5, 0.0, invalid_penalty)
            return 0.5 * (chi2 + prior_chi2) + penalty

        return nlp

    def _fit_map_from_nlp(
        self,
        nlp,
        observed_moments: np.ndarray,
        covariance_moments: np.ndarray,
        initial_theta: Sequence[float],
        max_iter: int,
        grad_tol: float,
    ) -> MAPFitResult:
        value_grad_fn = jax.jit(jax.value_and_grad(nlp))
        hess_fn = jax.jit(jax.hessian(nlp))

        theta0 = np.asarray(initial_theta, dtype=float)
        if theta0.shape != (3,) or np.any(theta0 <= 0.0):
            raise ValueError("initial_theta must be positive length-3 array")

        log_theta = np.log(theta0)
        success = False
        message = "Maximum iterations reached before convergence"
        n_iter = 0

        for it in range(max_iter):
            n_iter = it + 1
            val, grad = value_grad_fn(jnp.asarray(log_theta, dtype=jnp.float64))
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

            hess = np.asarray(hess_fn(jnp.asarray(log_theta, dtype=jnp.float64)), dtype=float)
            hess = 0.5 * (hess + hess.T)
            if not np.all(np.isfinite(hess)):
                hess = np.nan_to_num(hess, nan=0.0, posinf=0.0, neginf=0.0)
                hess = hess + 1e-6 * np.eye(3)

            accepted = False
            damping = 1e-6
            for _ in range(12):
                try:
                    step = np.linalg.solve(hess + damping * np.eye(3), grad)
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue

                candidate = log_theta - step
                cand_val = float(nlp(jnp.asarray(candidate, dtype=jnp.float64)))
                if np.isfinite(cand_val) and cand_val < val:
                    log_theta = candidate
                    accepted = True
                    break

                alpha = 0.5
                for _ in range(8):
                    candidate = log_theta - alpha * step
                    cand_val = float(nlp(jnp.asarray(candidate, dtype=jnp.float64)))
                    if np.isfinite(cand_val) and cand_val < val:
                        log_theta = candidate
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
                candidate = log_theta - 0.05 * (grad / gnorm)
                cand_val = float(nlp(jnp.asarray(candidate, dtype=jnp.float64)))
                if np.isfinite(cand_val) and cand_val < val:
                    log_theta = candidate
                else:
                    message = "Line search stalled: unable to reduce objective"
                    break

        final_nlp, final_grad = value_grad_fn(jnp.asarray(log_theta, dtype=jnp.float64))
        final_nlp = float(final_nlp)
        final_grad = np.asarray(final_grad, dtype=float)
        final_grad_norm = float(np.linalg.norm(final_grad, ord=np.inf))

        hessian_log = np.asarray(hess_fn(jnp.asarray(log_theta, dtype=jnp.float64)), dtype=float)
        hessian_log = np.nan_to_num(hessian_log, nan=0.0, posinf=0.0, neginf=0.0)
        hessian_log = stabilize_covariance(hessian_log, min_eig=1e-12)
        covariance_log = np.linalg.inv(hessian_log)

        theta_map = np.exp(log_theta)
        jac = np.diag(theta_map)
        covariance_theta = jac @ covariance_log @ jac
        correlation_theta = covariance_to_correlation(covariance_theta)

        predicted = self.predict_moments(theta_map)
        cov_obs = stabilize_covariance(np.asarray(covariance_moments, dtype=float))
        resid = predicted - np.asarray(observed_moments, dtype=float)
        chi2 = float(resid @ np.linalg.inv(cov_obs) @ resid)

        return MAPFitResult(
            success=success,
            message=message,
            n_iter=n_iter,
            log_theta_map=np.asarray(log_theta, dtype=float),
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
        )

    def fit_map(
        self,
        observed_moments: Sequence[float],
        covariance_moments: np.ndarray,
        initial_theta: Sequence[float] = (0.2, 0.2, 1.0),
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 1.0),
        max_iter: int = 25,
        grad_tol: float = 1e-5,
    ) -> MAPFitResult:
        """Fit MAP estimate in log-parameter space using damped Newton iterations."""
        nlp = self.make_negative_log_posterior(
            observed_moments=observed_moments,
            covariance_moments=covariance_moments,
            prior_mean_log=prior_mean_log,
            prior_sigma_log=prior_sigma_log,
        )
        return self._fit_map_from_nlp(
            nlp=nlp,
            observed_moments=np.asarray(observed_moments, dtype=float),
            covariance_moments=np.asarray(covariance_moments, dtype=float),
            initial_theta=initial_theta,
            max_iter=max_iter,
            grad_tol=grad_tol,
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
            p0 = rng.normal(loc=0.0, scale=np.sqrt(1.0 / inv_mass_diag), size=3)

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

        samples_log_arr = np.asarray(samples_log, dtype=float)
        samples_theta = np.exp(samples_log_arr)

        if samples_log_arr.shape[0] > 1:
            covariance_log = stabilize_covariance(np.cov(samples_log_arr.T), min_eig=1e-12)
            covariance_theta = stabilize_covariance(np.cov(samples_theta.T), min_eig=1e-12)
        else:
            covariance_log = np.eye(3)
            covariance_theta = np.eye(3)

        return HMCResult(
            samples_log=samples_log_arr,
            samples_theta=samples_theta,
            sampler="hmc",
            num_chains=1,
            acceptance_rate=float(accepted / max(total_steps, 1)),
            acceptance_rate_per_chain=np.asarray([accepted / max(total_steps, 1)], dtype=float),
            final_step_size=float(eps),
            num_divergent=0,
            num_steps_mean=float(leapfrog_steps),
            mean_log_posterior=float(np.mean(log_probs)) if log_probs else float("nan"),
            r_hat=None,
            ess_bulk=None,
            covariance_log=covariance_log,
            covariance_theta=covariance_theta,
            correlation_theta=covariance_to_correlation(covariance_theta),
        )

    def _sample_nuts_from_nlp(
        self,
        nlp,
        initial_log_theta: np.ndarray,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        target_accept: float,
        num_chains: int,
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

        def potential_fn(params):
            return nlp(params["log_theta"])

        nuts_kernel = NUTS(
            potential_fn=potential_fn,
            target_accept_prob=float(target_accept),
            step_size=float(step_size),
        )
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=int(num_warmup),
            num_samples=int(num_samples),
            num_chains=int(num_chains),
            progress_bar=False,
            chain_method="sequential",
        )
        rng_key = jax.random.PRNGKey(int(seed))
        init_log = jnp.asarray(initial_log_theta, dtype=jnp.float64)
        if int(num_chains) > 1:
            init_log = jnp.broadcast_to(init_log, (int(num_chains), init_log.shape[0]))
        mcmc.run(
            rng_key,
            init_params={"log_theta": init_log},
            extra_fields=("accept_prob", "num_steps", "diverging"),
        )

        samples_by_chain = mcmc.get_samples(group_by_chain=True)["log_theta"]
        samples_by_chain = np.asarray(samples_by_chain, dtype=float)
        samples_log = samples_by_chain.reshape(-1, samples_by_chain.shape[-1])
        samples_theta = np.exp(samples_log)

        extra = mcmc.get_extra_fields(group_by_chain=True)
        accept_prob = np.asarray(extra.get("accept_prob"), dtype=float)
        diverging = np.asarray(extra.get("diverging"), dtype=bool)
        num_steps = np.asarray(extra.get("num_steps"), dtype=float)

        acceptance_rate_per_chain = np.mean(accept_prob, axis=1) if accept_prob.size else np.full((num_chains,), np.nan)
        acceptance_rate = float(np.mean(accept_prob)) if accept_prob.size else float("nan")
        num_divergent = int(np.sum(diverging)) if diverging.size else 0
        num_steps_mean = float(np.mean(num_steps)) if num_steps.size else float("nan")

        step_state = getattr(getattr(mcmc.last_state, "adapt_state", None), "step_size", np.nan)
        final_step_size = float(np.mean(np.asarray(step_state, dtype=float)))

        sample_dict = {"log_theta": samples_by_chain}
        diag = numpyro_summary(sample_dict, group_by_chain=True)["log_theta"]
        r_hat = np.asarray(diag.get("r_hat"), dtype=float)
        ess_bulk = np.asarray(diag.get("n_eff"), dtype=float)

        if samples_log.shape[0] > 1:
            covariance_log = stabilize_covariance(np.cov(samples_log.T), min_eig=1e-12)
            covariance_theta = stabilize_covariance(np.cov(samples_theta.T), min_eig=1e-12)
        else:
            covariance_log = np.eye(3)
            covariance_theta = np.eye(3)

        nlp_batch = jax.vmap(lambda x: nlp(x))
        mean_log_posterior = float(-jnp.mean(nlp_batch(jnp.asarray(samples_log, dtype=jnp.float64))))

        return HMCResult(
            samples_log=samples_log,
            samples_theta=samples_theta,
            sampler="nuts",
            num_chains=int(num_chains),
            acceptance_rate=acceptance_rate,
            acceptance_rate_per_chain=acceptance_rate_per_chain,
            final_step_size=final_step_size,
            num_divergent=num_divergent,
            num_steps_mean=num_steps_mean,
            mean_log_posterior=mean_log_posterior,
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            covariance_log=covariance_log,
            covariance_theta=covariance_theta,
            correlation_theta=covariance_to_correlation(covariance_theta),
        )

    def _sample_posterior(
        self,
        nlp,
        initial_log_theta: np.ndarray,
        mass_diag: np.ndarray,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        leapfrog_steps: int,
        target_accept: float,
        num_chains: int,
        seed: int,
        sampler: str,
    ) -> HMCResult:
        """Dispatch posterior sampling backend."""
        sampler_key = sampler.strip().lower()
        if sampler_key == "nuts":
            return self._sample_nuts_from_nlp(
                nlp=nlp,
                initial_log_theta=initial_log_theta,
                num_warmup=num_warmup,
                num_samples=num_samples,
                step_size=step_size,
                target_accept=target_accept,
                num_chains=max(1, int(num_chains)),
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
        initial_theta: Sequence[float] = (0.2, 0.2, 1.0),
        prior_mean_log: Sequence[float] | None = None,
        prior_sigma_log: Sequence[float] = (1.5, 1.5, 1.0),
        map_max_iter: int = 25,
        map_grad_tol: float = 1e-5,
        hmc_num_warmup: int = 250,
        hmc_num_samples: int = 500,
        hmc_step_size: float = 0.02,
        hmc_leapfrog_steps: int = 12,
        hmc_target_accept: float = 0.70,
        sampler: str = "nuts",
        num_chains: int = 1,
        seed: int = 0,
    ) -> PosteriorFitResult:
        """Run MAP + posterior sampling workflow for observed [M0, M1, M2]."""
        nlp = self.make_negative_log_posterior(
            observed_moments=observed_moments,
            covariance_moments=covariance_moments,
            prior_mean_log=prior_mean_log,
            prior_sigma_log=prior_sigma_log,
        )

        map_result = self._fit_map_from_nlp(
            nlp=nlp,
            observed_moments=np.asarray(observed_moments, dtype=float),
            covariance_moments=np.asarray(covariance_moments, dtype=float),
            initial_theta=initial_theta,
            max_iter=map_max_iter,
            grad_tol=map_grad_tol,
        )

        mass_diag = np.diag(stabilize_covariance(map_result.covariance_log, min_eig=1e-10))

        hmc_result = self._sample_posterior(
            nlp=nlp,
            initial_log_theta=map_result.log_theta_map,
            mass_diag=mass_diag,
            num_warmup=hmc_num_warmup,
            num_samples=hmc_num_samples,
            step_size=hmc_step_size,
            leapfrog_steps=hmc_leapfrog_steps,
            target_accept=hmc_target_accept,
            num_chains=num_chains,
            seed=seed,
            sampler=sampler,
        )

        return PosteriorFitResult(
            map=map_result,
            hmc=hmc_result,
            observed_moments=np.asarray(observed_moments, dtype=float),
            covariance_moments=stabilize_covariance(np.asarray(covariance_moments, dtype=float)),
        )

    def predict_moments_for_log_samples(self, samples_log: np.ndarray, max_samples: int = 512) -> np.ndarray:
        """Evaluate model moments for a subset of posterior log-parameter samples."""
        arr = np.asarray(samples_log, dtype=float)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("samples_log must have shape (N, 3) or (chains, N, 3)")

        if arr.shape[0] > max_samples:
            idx = np.linspace(0, arr.shape[0] - 1, max_samples, dtype=int)
            arr = arr[idx]

        batch = jnp.asarray(arr, dtype=jnp.float64)
        predict_batch = jax.jit(jax.vmap(self._predict_log_theta_fn, in_axes=0))
        return np.asarray(predict_batch(batch), dtype=float)


def plot_corner(
    samples_theta: np.ndarray,
    labels: Sequence[str],
    output_path: str,
    truths: Sequence[float] | None = None,
    map_theta: Sequence[float] | None = None,
) -> None:
    """Create a lightweight corner plot for 3 parameters."""
    x = np.asarray(samples_theta, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(labels):
        raise ValueError("samples_theta must have shape (N, D) matching labels")

    d = x.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(3.1 * d, 3.1 * d), constrained_layout=True)

    truths_arr = None if truths is None else np.asarray(truths, dtype=float)
    map_arr = None if map_theta is None else np.asarray(map_theta, dtype=float)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue

            if i == j:
                ax.hist(x[:, j], bins=35, color="tab:blue", alpha=0.8, density=True)
                if truths_arr is not None:
                    ax.axvline(truths_arr[j], color="tab:green", lw=1.5, ls="--")
                if map_arr is not None:
                    ax.axvline(map_arr[j], color="tab:red", lw=1.5)
            else:
                ax.scatter(x[:, j], x[:, i], s=6, alpha=0.22, color="tab:blue", edgecolors="none")
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


def plot_moment_fit(
    observed_moments: np.ndarray,
    covariance_moments: np.ndarray,
    map_moments: np.ndarray,
    posterior_moment_samples: np.ndarray | None,
    output_path: str,
) -> None:
    """Plot observed moments with uncertainty against MAP and posterior predictive summary."""
    obs = np.asarray(observed_moments, dtype=float)
    cov = np.asarray(covariance_moments, dtype=float)
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    map_pred = np.asarray(map_moments, dtype=float)

    labels = ["M0", "M1", "M2"]
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.errorbar(x, obs, yerr=sigma, fmt="o", color="black", lw=1.5, capsize=4, label="Observed")
    ax.scatter(x, map_pred, marker="s", s=45, color="tab:red", label="MAP prediction")

    if posterior_moment_samples is not None and posterior_moment_samples.size > 0:
        samp = np.asarray(posterior_moment_samples, dtype=float)
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
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Moment value")
    ax.set_title("Observed vs fitted dN/dv moments")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
