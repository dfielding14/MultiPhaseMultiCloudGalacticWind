"""Topaz cooling-table loading and JAX interpolation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .constants import kb, mp, muH


@dataclass(frozen=True)
class CoolingTable:
    log10_temperature: np.ndarray
    primordial_cooling_cgs: np.ndarray
    metal_cooling_cgs: np.ndarray


_TABLE_CACHE: dict[str, CoolingTable] = {}
_JAX_TABLE_CACHE: dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = {}
_DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "topaz_cooling_high_nh.csv"
_DEFAULT_TABLE_CACHE_KEY = "__default_topaz_table__"


def _cache_key_for_path(path: str | Path | None) -> tuple[str, Path]:
    if path is None:
        return _DEFAULT_TABLE_CACHE_KEY, _DEFAULT_TABLE_PATH
    table_path = Path(path)
    return str(table_path), table_path


def _infer_cache_key_from_table(table: CoolingTable) -> str | None:
    for key, value in _TABLE_CACHE.items():
        if value is table:
            return key
    return None


def load_cooling_table(path: str | Path | None = None) -> CoolingTable:
    """Load and cache the Topaz cooling table from CSV."""
    cache_key, table_path = _cache_key_for_path(path)
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = np.genfromtxt(table_path, delimiter=",", comments="#", names=True, skip_header=3)
    table = CoolingTable(
        log10_temperature=np.ascontiguousarray(data["log10_T"], dtype=float),
        primordial_cooling_cgs=np.ascontiguousarray(data["primordial_cooling_cgs"], dtype=float),
        metal_cooling_cgs=np.ascontiguousarray(data["metal_cooling_cgs"], dtype=float),
    )
    _TABLE_CACHE[cache_key] = table
    return table


def get_jax_table_arrays(
    table: CoolingTable | None = None,
    table_path: str | Path | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return cached JAX arrays for the cooling table."""
    if table is None:
        cache_key, _ = _cache_key_for_path(table_path)
        table_obj = load_cooling_table(table_path)
    else:
        table_obj = table
        cache_key = _infer_cache_key_from_table(table_obj)
        if cache_key is None:
            cache_key = f"table-id:{id(table_obj)}"

    cached = _JAX_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    arrays = (
        jnp.asarray(table_obj.log10_temperature),
        jnp.asarray(table_obj.primordial_cooling_cgs),
        jnp.asarray(table_obj.metal_cooling_cgs),
    )
    _JAX_TABLE_CACHE[cache_key] = arrays
    return arrays


def lambda_total_cgs_jax(
    temperature_K,
    metallicity_solar,
    log10_temperature,
    primordial_cooling_cgs,
    metal_cooling_cgs,
):
    """Return total cooling function Λ(T, Z) in cgs (JAX-native)."""
    t_min = jnp.power(10.0, log10_temperature[0])
    t_max = jnp.power(10.0, log10_temperature[-1])
    t_clip = jnp.clip(jnp.asarray(temperature_K), t_min, t_max)
    log10_t = jnp.log10(t_clip)
    lambda_prim = jnp.interp(log10_t, log10_temperature, primordial_cooling_cgs)
    lambda_metal = jnp.interp(log10_t, log10_temperature, metal_cooling_cgs)
    return lambda_prim + jnp.asarray(metallicity_solar) * lambda_metal


def tcool_P_topaz_jax(
    temperature_K,
    pressure_over_kb,
    metallicity_solar,
    mu: float,
    log10_temperature,
    primordial_cooling_cgs,
    metal_cooling_cgs,
):
    """Cooling time t_cool(T, P/k_B, Z) using Topaz tables (JAX-native)."""
    t = jnp.asarray(temperature_K)
    p = jnp.asarray(pressure_over_kb)

    lambda_val = lambda_total_cgs_jax(
        t,
        metallicity_solar,
        log10_temperature,
        primordial_cooling_cgs,
        metal_cooling_cgs,
    )
    n_h_actual = p / t * (mu / muH)
    denominator = n_h_actual * lambda_val

    result = 1.5 * (muH / mu) * kb * t / denominator
    return jnp.where(denominator == 0.0, jnp.inf, result)


def lambda_p_rho_topaz_scalar_jax(
    pressure,
    rho,
    mu: float,
    metallicity_solar: float,
    log10_temperature,
    primordial_cooling_cgs,
    metal_cooling_cgs,
):
    """Return Λ(P, rho) scalar for hot-phase cooling in RHS (JAX-native)."""
    safe_rho = jnp.where(rho > 0.0, rho, jnp.inf)
    temperature = (pressure / kb) * (mu * mp / safe_rho)
    lambda_val = lambda_total_cgs_jax(
        temperature,
        metallicity_solar,
        log10_temperature,
        primordial_cooling_cgs,
        metal_cooling_cgs,
    )
    return jnp.where(rho > 0.0, lambda_val, 0.0)


def lambda_total_cgs(
    temperature_K: float | np.ndarray,
    metallicity_solar: float | np.ndarray,
    table: CoolingTable | None = None,
    table_path: str | Path | None = None,
):
    """Compatibility wrapper for Λ(T, Z) using JAX kernels."""
    log10_temperature, primordial_cooling_cgs, metal_cooling_cgs = get_jax_table_arrays(
        table=table,
        table_path=table_path,
    )
    return lambda_total_cgs_jax(
        temperature_K,
        metallicity_solar,
        log10_temperature,
        primordial_cooling_cgs,
        metal_cooling_cgs,
    )


def tcool_P_topaz(
    temperature_K: float | np.ndarray,
    pressure_over_kb: float | np.ndarray,
    metallicity_solar: float | np.ndarray,
    mu: float,
    table: CoolingTable | None = None,
    table_path: str | Path | None = None,
):
    """Compatibility wrapper for t_cool(T, P/k_B, Z) using JAX kernels."""
    log10_temperature, primordial_cooling_cgs, metal_cooling_cgs = get_jax_table_arrays(
        table=table,
        table_path=table_path,
    )
    return tcool_P_topaz_jax(
        temperature_K,
        pressure_over_kb,
        metallicity_solar,
        mu,
        log10_temperature,
        primordial_cooling_cgs,
        metal_cooling_cgs,
    )


def tcool_P_topaz_vector(
    temperature_K: float | np.ndarray,
    pressure_over_kb: float,
    metallicity_solar: np.ndarray,
    mu: float,
    table: CoolingTable,
):
    """Fast vector cooling call for ODE RHS (JAX-native)."""
    log10_temperature, primordial_cooling_cgs, metal_cooling_cgs = get_jax_table_arrays(table=table)
    z = jnp.asarray(metallicity_solar)
    t_in = jnp.asarray(temperature_K)
    t = jnp.where(jnp.ndim(t_in) == 0, jnp.ones_like(z) * t_in, t_in)
    p = jnp.ones_like(z) * pressure_over_kb
    return tcool_P_topaz_jax(
        t,
        p,
        z,
        mu,
        log10_temperature,
        primordial_cooling_cgs,
        metal_cooling_cgs,
    )


def get_lambda_p_rho_callable_topaz(
    mu: float,
    metallicity_solar: float,
    table_path: str | Path | None = None,
) -> Callable[[tuple[float, float] | np.ndarray], float | np.ndarray]:
    """
    Build Λ(P, rho) callable compatible with the legacy interpolation interface.

    Input pressure must be in dyne cm^-2, density in g cm^-3.
    """
    table = load_cooling_table(table_path)
    log10_temperature, primordial_cooling_cgs, metal_cooling_cgs = get_jax_table_arrays(table=table)

    def lambda_p_rho(point: tuple[float, float] | np.ndarray):
        arr = jnp.asarray(point)
        pressure = arr[..., 0]
        rho = arr[..., 1]
        out = lambda_p_rho_topaz_scalar_jax(
            pressure,
            rho,
            mu,
            metallicity_solar,
            log10_temperature,
            primordial_cooling_cgs,
            metal_cooling_cgs,
        )
        return out

    return lambda_p_rho
