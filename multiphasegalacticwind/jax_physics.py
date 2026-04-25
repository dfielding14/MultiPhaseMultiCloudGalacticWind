"""JAX-native physics kernels and ODE integration helpers."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .constants import Myr, Z_solar, gamma, kb, mp, muH
from .topaz_cooling import (
    get_jax_table_arrays,
    lambda_p_rho_topaz_scalar_jax,
    load_cooling_table,
    tcool_P_topaz_jax,
)

_FOUR_PI_OVER_THREE = 4.0 * np.pi / 3.0


class JaxWindParams(NamedTuple):
    v_circ: float
    Ndot_cloud0: jnp.ndarray
    T_cloud: float
    injection_radius: float
    injection_power: float
    M_cloud_min: float
    CoolingAreaChiPower: float
    ColdTurbulenceChiPower: float
    TurbulentVelocityChiPower: float
    geometric_factor: float
    Mdot_coefficient: float
    Cooling_Factor: float
    drag_coeff: float
    f_turb0: float
    Omwind: float
    mu: float
    redshift: float
    Z_hot_over_Z_solar: float
    v_cloud_min: float
    r0: float
    Edot_per_Vol: float
    Mdot_per_Vol: float
    topaz_log10_temperature: jnp.ndarray
    topaz_primordial_cooling_cgs: jnp.ndarray
    topaz_metal_cooling_cgs: jnp.ndarray


def build_jax_wind_params(
    v_circ: float,
    Ndot_cloud0,
    T_cloud: float,
    injection_radius: float,
    injection_power: float,
    config_dict: dict[str, Any],
    r0: float,
    Edot_per_Vol: float,
    Mdot_per_Vol: float,
    table_path: str | None = None,
):
    """Build JAX parameter bundle for the multiphase RHS."""
    topaz_table = config_dict.get("_topaz_table")
    if topaz_table is None:
        topaz_table = load_cooling_table(table_path or config_dict.get("topaz_cooling_table_path"))
    topaz_log10_temperature, topaz_primordial_cooling_cgs, topaz_metal_cooling_cgs = get_jax_table_arrays(
        table=topaz_table
    )

    return JaxWindParams(
        v_circ=float(v_circ),
        Ndot_cloud0=jnp.asarray(Ndot_cloud0, dtype=jnp.float64),
        T_cloud=float(T_cloud),
        injection_radius=float(injection_radius),
        injection_power=float(injection_power),
        M_cloud_min=float(config_dict["M_cloud_min"]),
        CoolingAreaChiPower=float(config_dict["CoolingAreaChiPower"]),
        ColdTurbulenceChiPower=float(config_dict["ColdTurbulenceChiPower"]),
        TurbulentVelocityChiPower=float(config_dict["TurbulentVelocityChiPower"]),
        geometric_factor=float(config_dict["geometric_factor"]),
        Mdot_coefficient=float(config_dict["Mdot_coefficient"]),
        Cooling_Factor=float(config_dict["Cooling_Factor"]),
        drag_coeff=float(config_dict["drag_coeff"]),
        f_turb0=float(config_dict["f_turb0"]),
        Omwind=float(config_dict["Omwind"]),
        mu=float(config_dict["mu"]),
        redshift=float(config_dict.get("redshift", 0.0)),
        Z_hot_over_Z_solar=float(config_dict.get("Z_hot_over_Z_solar", config_dict.get("metallicity", 1.0))),
        v_cloud_min=float(config_dict.get("v_cloud_min", 1.0)),
        r0=float(r0),
        Edot_per_Vol=float(Edot_per_Vol),
        Mdot_per_Vol=float(Mdot_per_Vol),
        topaz_log10_temperature=topaz_log10_temperature,
        topaz_primordial_cooling_cgs=topaz_primordial_cooling_cgs,
        topaz_metal_cooling_cgs=topaz_metal_cooling_cgs,
    )


def _safe_power(base, exponent):
    return jnp.power(jnp.maximum(base, 0.0), exponent)


def has_diffrax() -> bool:
    """Return True when Diffrax is importable for adaptive integration."""
    try:
        import diffrax  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _load_diffrax():
    try:
        import diffrax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Diffrax is required for integrator_mode='tsit5'. Install with `python -m pip install diffrax`."
        ) from exc
    return diffrax


def wind_evo_jax(state, r, params: JaxWindParams):
    """JAX-native multiphase RHS used for integration and autodiff."""
    n_cloud_species = (state.shape[0] - 4) // 3

    v_wind = state[0]
    rho_wind = state[1]
    Pressure = state[2]
    rhoZ_wind = state[3]
    M_cloud = state[4 : 4 + n_cloud_species]
    v_cloud = state[4 + n_cloud_species : 4 + 2 * n_cloud_species]
    Z_cloud = state[4 + 2 * n_cloud_species : 4 + 3 * n_cloud_species]

    safe_r = jnp.maximum(r, 1e-30)
    valid_wind = (Pressure > 0.0) & (rho_wind > 0.0) & (v_wind > 0.0)

    cs_sq_wind = gamma * Pressure / jnp.maximum(rho_wind, 1e-60)
    Mach_sq_wind = v_wind * v_wind / jnp.maximum(cs_sq_wind, 1e-60)
    Z_wind = rhoZ_wind / jnp.maximum(rho_wind, 1e-60)
    Phir = params.v_circ * params.v_circ * jnp.log(safe_r)
    vBsq_wind = 0.5 * v_wind * v_wind + (gamma / (gamma - 1.0)) * Pressure / jnp.maximum(rho_wind, 1e-60) + Phir

    rho_cloud = Pressure * (params.mu * mp) / (kb * params.T_cloud)
    chi = rho_cloud / jnp.maximum(rho_wind, 1e-60)
    chi_safe = jnp.maximum(chi, 1e-30)

    T_wind = (Pressure / kb) * (params.mu * mp / jnp.maximum(rho_wind, 1e-60))
    T_mix = jnp.sqrt(jnp.maximum(T_wind * params.T_cloud, 1e-30))
    Z_mix = jnp.sqrt(jnp.maximum(Z_wind * Z_cloud, 0.0))

    t_cool_layer = tcool_P_topaz_jax(
        T_mix,
        Pressure / kb,
        Z_mix / Z_solar,
        params.mu,
        params.topaz_log10_temperature,
        params.topaz_primordial_cooling_cgs,
        params.topaz_metal_cooling_cgs,
    )
    t_cool_layer = jnp.where(t_cool_layer < 0.0, 1e10 * Myr, t_cool_layer)

    injection_factor = jnp.where(
        r < params.injection_radius,
        _safe_power(r / jnp.maximum(params.injection_radius, 1e-30), params.injection_power),
        1.0,
    )
    Ndot_cloud = params.Ndot_cloud0 * injection_factor

    v_cloud_floor = jnp.maximum(params.v_cloud_min * 1e5, 1e-10)
    v_cloud_safe = jnp.maximum(v_cloud, v_cloud_floor)
    number_density_cloud = Ndot_cloud / (params.Omwind * safe_r * safe_r * v_cloud_safe)

    rho_cloud_safe = jnp.maximum(rho_cloud, 1e-60)
    r_cloud = jnp.where(M_cloud > 0.0, _safe_power(M_cloud / (_FOUR_PI_OVER_THREE * rho_cloud_safe), 1.0 / 3.0), 0.0)
    r_cloud_safe = jnp.where(r_cloud > 0.0, r_cloud, jnp.inf)

    v_rel = v_wind - v_cloud
    v_turb = params.f_turb0 * jnp.abs(v_rel) * _safe_power(chi_safe, params.TurbulentVelocityChiPower)
    ksi = r_cloud / (jnp.maximum(v_turb, 1e-10) * jnp.maximum(t_cool_layer, 1e-30))
    area_boost = params.geometric_factor * _safe_power(chi_safe, params.CoolingAreaChiPower)
    v_turb_cold = v_turb * _safe_power(chi_safe, params.ColdTurbulenceChiPower)

    cloud_active = M_cloud > params.M_cloud_min
    ksi_factor = jnp.where(ksi < 1.0, _safe_power(ksi, 0.5), _safe_power(ksi, 0.25))

    Mdot_grow = jnp.where(
        cloud_active,
        params.Mdot_coefficient
        * 3.0
        * M_cloud
        * v_turb
        * area_boost
        / (r_cloud_safe * chi_safe)
        * ksi_factor,
        0.0,
    )
    Mdot_loss = jnp.where(
        cloud_active,
        params.Mdot_coefficient * 3.0 * (-M_cloud) * v_turb_cold / r_cloud_safe,
        0.0,
    )
    Mdot_cloud = Mdot_grow + Mdot_loss

    Mdot_SN = jnp.where(r < params.r0, params.Mdot_per_Vol, 0.0)
    Edot_SN = jnp.where(r < params.r0, params.Edot_per_Vol, 0.0)

    drhodt = Mdot_SN - jnp.sum(number_density_cloud * Mdot_cloud)

    p_dot_ram = 0.5 * params.drag_coeff * rho_wind * jnp.pi * v_rel * jnp.abs(v_rel) * r_cloud * r_cloud
    p_dot_transfer = v_wind * Mdot_grow + v_cloud * Mdot_loss
    dpdt = -jnp.sum(number_density_cloud * (p_dot_transfer + p_dot_ram))

    lambda_hot = lambda_p_rho_topaz_scalar_jax(
        Pressure,
        rho_wind,
        params.mu,
        params.Z_hot_over_Z_solar,
        params.topaz_log10_temperature,
        params.topaz_primordial_cooling_cgs,
        params.topaz_metal_cooling_cgs,
    )
    e_dot_cool = -params.Cooling_Factor * (rho_wind / (muH * mp)) ** 2 * lambda_hot

    cs_cl_sq = gamma * kb * params.T_cloud / (params.mu * mp)
    vBsq_cl = 0.5 * v_cloud * v_cloud + cs_cl_sq / (gamma - 1.0) + Phir
    e_dot_transfer = vBsq_wind * Mdot_grow + vBsq_cl * Mdot_loss
    dedt = Edot_SN - jnp.sum(number_density_cloud * (e_dot_transfer + p_dot_ram * v_wind)) + e_dot_cool

    drhoZdt = -jnp.sum(number_density_cloud * (Z_wind * Mdot_grow + Z_cloud * Mdot_loss))

    sonic_regularization_width = 0.01
    epsilon = Mach_sq_wind - 1.0
    epsilon_safe = jnp.where(
        jnp.abs(epsilon) < 1e-10,
        jnp.where(epsilon == 0.0, 1e-10, jnp.sign(epsilon) * 1e-10),
        epsilon,
    )
    denominator = jnp.where(
        jnp.abs(Mach_sq_wind - 1.0) < sonic_regularization_width,
        epsilon_safe,
        1.0 - (1.0 / jnp.maximum(Mach_sq_wind, 1e-60)),
    )

    rho_v_over_r = rho_wind * v_wind / safe_r
    source_term = (
        drhodt * (gamma + 1.0) / 2.0
        - gamma * dpdt / jnp.maximum(v_wind, 1e-60)
        + (gamma - 1.0) * dedt / jnp.maximum(v_wind * v_wind, 1e-60)
        - (gamma - 1.0) * (Phir / jnp.maximum(v_wind * v_wind, 1e-60)) * drhodt
    )

    dv_dr = (v_wind / safe_r) / denominator * (
        2.0 / jnp.maximum(Mach_sq_wind, 1e-60)
        - (params.v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
        - source_term / jnp.maximum(rho_v_over_r, 1e-60)
    )

    drho_dr = (rho_wind / safe_r) / denominator * (
        -2.0
        + (params.v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
        + (
            drhodt * (gamma + 3.0) / 2.0
            - gamma * dpdt / jnp.maximum(v_wind, 1e-60)
            + (gamma - 1.0) * dedt / jnp.maximum(v_wind * v_wind, 1e-60)
            - drhodt / jnp.maximum(Mach_sq_wind, 1e-60)
            + (gamma - 1.0) * (Phir / jnp.maximum(v_wind * v_wind, 1e-60)) * drhodt
        )
        / jnp.maximum(rho_v_over_r, 1e-60)
    )

    Z_wind_safe = jnp.where(jnp.abs(Z_wind) > 1e-30, Z_wind, 1e-30)
    drhoZ_dr = (
        (rhoZ_wind / safe_r)
        / denominator
        * (
            -2.0
            + (params.v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
            + (
                drhodt * (gamma + 3.0) / 2.0
                - gamma * dpdt / jnp.maximum(v_wind, 1e-60)
                + (gamma - 1.0) * dedt / jnp.maximum(v_wind * v_wind, 1e-60)
                - drhodt / jnp.maximum(Mach_sq_wind, 1e-60)
                + (gamma - 1.0) * (Phir / jnp.maximum(v_wind * v_wind, 1e-60)) * drhodt
            )
            / jnp.maximum(rho_v_over_r, 1e-60)
        )
        + (rhoZ_wind / safe_r)
        * (1.0 / jnp.maximum(rho_v_over_r, 1e-60))
        * ((drhoZdt / Z_wind_safe) - drhodt)
    )

    dP_dr = (Pressure / safe_r) * gamma / denominator * (
        -2.0
        + (params.v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
        + (
            drhodt
            + drhodt
            * (gamma - 1.0)
            / 2.0
            * Mach_sq_wind
            * (1.0 - 2.0 * Phir / jnp.maximum(v_wind * v_wind, 1e-60))
            - dpdt / jnp.maximum(v_wind, 1e-60)
            + (gamma - 1.0)
            * Mach_sq_wind
            * (dedt - v_wind * dpdt)
            / jnp.maximum(v_wind * v_wind, 1e-60)
        )
        / jnp.maximum(rho_v_over_r, 1e-60)
    )

    dM_cloud_dr = jnp.where(cloud_active, Mdot_cloud / v_cloud_safe, 0.0)
    dv_cloud_dr = jnp.where(
        cloud_active,
        (p_dot_ram + v_rel * Mdot_grow - M_cloud * params.v_circ * params.v_circ / safe_r)
        / jnp.maximum(M_cloud * v_cloud_safe, 1e-60),
        0.0,
    )
    dZ_cloud_dr = jnp.where(
        cloud_active,
        (Z_wind - Z_cloud) * Mdot_grow / jnp.maximum(M_cloud * v_cloud_safe, 1e-60),
        0.0,
    )

    derivatives = jnp.concatenate(
        [
            jnp.asarray([dv_dr, drho_dr, dP_dr, drhoZ_dr]),
            dM_cloud_dr,
            dv_cloud_dr,
            dZ_cloud_dr,
        ]
    )

    ok = valid_wind & (chi > 0.0) & jnp.all(jnp.isfinite(derivatives))
    return jnp.where(ok, derivatives, jnp.zeros_like(derivatives))


def hot_wind_evo_jax(state, r, v_circ: float, include_source_terms: bool, r0: float, Edot_per_Vol: float, Mdot_per_Vol: float):
    """JAX-native hot-wind RHS."""
    v_wind = state[0]
    rho_wind = state[1]
    Pressure = state[2]

    safe_r = jnp.maximum(r, 1e-30)
    cs_sq_wind = gamma * Pressure / jnp.maximum(rho_wind, 1e-60)
    Mach_sq_wind = v_wind * v_wind / jnp.maximum(cs_sq_wind, 1e-60)
    Phir = v_circ * v_circ * jnp.log(safe_r)

    source_on = jnp.where(jnp.asarray(include_source_terms), 1.0, 0.0)
    source_mask = source_on * jnp.where(r < r0, 1.0, 0.0)
    Edot_SN = Edot_per_Vol * source_mask
    Mdot_SN = Mdot_per_Vol * source_mask

    drhodt = Mdot_SN
    dpdt = 0.0
    dedt = Edot_SN

    sonic_denom = 1.0 - (1.0 / jnp.maximum(Mach_sq_wind, 1e-60))
    sonic_denom = jnp.where(jnp.abs(sonic_denom) < 1e-5, jnp.sign(sonic_denom + 1e-30) * 1e-5, sonic_denom)

    rho_v_over_r = rho_wind * v_wind / safe_r

    dv_dr = (v_wind / safe_r) / sonic_denom * (
        2.0 / jnp.maximum(Mach_sq_wind, 1e-60)
        - (v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
        - (drhodt * (gamma + 1.0) / 2.0 + (gamma - 1.0) * dedt / jnp.maximum(v_wind * v_wind, 1e-60))
        / jnp.maximum(rho_v_over_r, 1e-60)
    )

    drho_dr = (rho_wind / safe_r) / sonic_denom * (
        -2.0
        + (v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
        + (
            drhodt * (gamma + 3.0) / 2.0
            + (gamma - 1.0) * dedt / jnp.maximum(v_wind * v_wind, 1e-60)
            - drhodt / jnp.maximum(Mach_sq_wind, 1e-60)
        )
        / jnp.maximum(rho_v_over_r, 1e-60)
    )

    dP_dr = (Pressure / safe_r) * gamma / sonic_denom * (
        -2.0
        + (v_circ / jnp.maximum(v_wind, 1e-60)) ** 2
        + (
            drhodt
            + drhodt * (gamma - 1.0) / 2.0 * Mach_sq_wind
            + (gamma - 1.0) * Mach_sq_wind * dedt / jnp.maximum(v_wind * v_wind, 1e-60)
        )
        / jnp.maximum(rho_v_over_r, 1e-60)
    )

    derivatives = jnp.asarray([dv_dr, drho_dr, dP_dr])
    ok = (v_wind > 0.0) & (rho_wind > 0.0) & (Pressure > 0.0) & jnp.all(jnp.isfinite(derivatives))
    return jnp.where(ok, derivatives, jnp.zeros_like(derivatives))


@jax.jit
def integrate_wind_rk4_scan(r_grid, y0, params: JaxWindParams):
    """
    Integrate the full multiphase wind with fixed-grid RK4 in JAX.

    This avoids the high overhead of `jax.experimental.ode.odeint` while
    remaining fully differentiable and JIT-compiled.
    """
    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)

    def step(y, inputs):
        r0, r1 = inputs
        h = r1 - r0
        k1 = wind_evo_jax(y, r0, params)
        k2 = wind_evo_jax(y + 0.5 * h * k1, r0 + 0.5 * h, params)
        k3 = wind_evo_jax(y + 0.5 * h * k2, r0 + 0.5 * h, params)
        k4 = wind_evo_jax(y + h * k3, r1, params)
        y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0_jax, (r_jax[:-1], r_jax[1:]))
    return jnp.vstack([y0_jax[None, :], ys])


@jax.jit
def integrate_wind_rk2_scan(r_grid, y0, params: JaxWindParams):
    """Integrate the full multiphase wind with fixed-grid RK2 midpoint."""
    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)

    def step(y, inputs):
        r0, r1 = inputs
        h = r1 - r0
        k1 = wind_evo_jax(y, r0, params)
        k2 = wind_evo_jax(y + 0.5 * h * k1, r0 + 0.5 * h, params)
        y_next = y + h * k2
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0_jax, (r_jax[:-1], r_jax[1:]))
    return jnp.vstack([y0_jax[None, :], ys])


@jax.jit
def integrate_wind_rk3_scan(r_grid, y0, params: JaxWindParams):
    """Integrate the full multiphase wind with fixed-grid classical RK3."""
    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)

    def step(y, inputs):
        r0, r1 = inputs
        h = r1 - r0
        k1 = wind_evo_jax(y, r0, params)
        k2 = wind_evo_jax(y + 0.5 * h * k1, r0 + 0.5 * h, params)
        k3 = wind_evo_jax(y - h * k1 + 2.0 * h * k2, r1, params)
        y_next = y + (h / 6.0) * (k1 + 4.0 * k2 + k3)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0_jax, (r_jax[:-1], r_jax[1:]))
    return jnp.vstack([y0_jax[None, :], ys])


def integrate_wind_tsit5(
    r_grid,
    y0,
    params: JaxWindParams,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_steps: int = 131072,
):
    """
    Integrate the full multiphase wind with adaptive Tsit5 (Diffrax).

    The solution is saved exactly on the provided `r_grid` points.
    """
    diffrax = _load_diffrax()

    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)
    if r_jax.size < 2:
        return y0_jax[None, :]

    dt0 = jnp.maximum(r_jax[1] - r_jax[0], 1e-30)
    term = diffrax.ODETerm(lambda r, y, args: wind_evo_jax(y, r, args))
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=r_jax)
    controller = diffrax.PIDController(rtol=float(rtol), atol=float(atol))

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=r_jax[0],
        t1=r_jax[-1],
        dt0=dt0,
        y0=y0_jax,
        args=params,
        saveat=saveat,
        stepsize_controller=controller,
        max_steps=int(max_steps),
        throw=False,
    )
    success = sol.result == diffrax.RESULTS.successful
    invalid = jnp.full_like(sol.ys, jnp.nan)
    return jnp.where(success, sol.ys, invalid)


@jax.jit
def integrate_hot_wind_rk4_scan(
    r_grid,
    y0,
    v_circ,
    include_source_terms,
    r0,
    Edot_per_Vol,
    Mdot_per_Vol,
):
    """Integrate the hot-only control model with fixed-grid RK4 in JAX."""
    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)

    def rhs(y, r):
        return hot_wind_evo_jax(y, r, v_circ, include_source_terms, r0, Edot_per_Vol, Mdot_per_Vol)

    def step(y, inputs):
        r0_loc, r1_loc = inputs
        h = r1_loc - r0_loc
        k1 = rhs(y, r0_loc)
        k2 = rhs(y + 0.5 * h * k1, r0_loc + 0.5 * h)
        k3 = rhs(y + 0.5 * h * k2, r0_loc + 0.5 * h)
        k4 = rhs(y + h * k3, r1_loc)
        y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0_jax, (r_jax[:-1], r_jax[1:]))
    return jnp.vstack([y0_jax[None, :], ys])


@jax.jit
def integrate_hot_wind_rk2_scan(
    r_grid,
    y0,
    v_circ,
    include_source_terms,
    r0,
    Edot_per_Vol,
    Mdot_per_Vol,
):
    """Integrate the hot-only control model with fixed-grid RK2 midpoint."""
    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)

    def rhs(y, r):
        return hot_wind_evo_jax(y, r, v_circ, include_source_terms, r0, Edot_per_Vol, Mdot_per_Vol)

    def step(y, inputs):
        r0_loc, r1_loc = inputs
        h = r1_loc - r0_loc
        k1 = rhs(y, r0_loc)
        k2 = rhs(y + 0.5 * h * k1, r0_loc + 0.5 * h)
        y_next = y + h * k2
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0_jax, (r_jax[:-1], r_jax[1:]))
    return jnp.vstack([y0_jax[None, :], ys])


@jax.jit
def integrate_hot_wind_rk3_scan(
    r_grid,
    y0,
    v_circ,
    include_source_terms,
    r0,
    Edot_per_Vol,
    Mdot_per_Vol,
):
    """Integrate the hot-only control model with fixed-grid classical RK3."""
    y0_jax = jnp.asarray(y0, dtype=jnp.float64)
    r_jax = jnp.asarray(r_grid, dtype=jnp.float64)

    def rhs(y, r):
        return hot_wind_evo_jax(y, r, v_circ, include_source_terms, r0, Edot_per_Vol, Mdot_per_Vol)

    def step(y, inputs):
        r0_loc, r1_loc = inputs
        h = r1_loc - r0_loc
        k1 = rhs(y, r0_loc)
        k2 = rhs(y + 0.5 * h * k1, r0_loc + 0.5 * h)
        k3 = rhs(y - h * k1 + 2.0 * h * k2, r1_loc)
        y_next = y + (h / 6.0) * (k1 + 4.0 * k2 + k3)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0_jax, (r_jax[:-1], r_jax[1:]))
    return jnp.vstack([y0_jax[None, :], ys])


def jacobian_wind_rhs_state(r: float, state, params: JaxWindParams):
    """Compute Jacobian ∂f/∂y of the multiphase RHS at (r, state)."""
    state_jax = jnp.asarray(state, dtype=jnp.float64)
    jac = jax.jacfwd(lambda y: wind_evo_jax(y, r, params))(state_jax)
    return jac
