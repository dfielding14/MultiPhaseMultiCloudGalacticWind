"""Topaz cooling-table loading and interpolation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numba import njit

from .constants import kb, mp, muH


@dataclass(frozen=True)
class CoolingTable:
    log10_temperature: np.ndarray
    primordial_cooling_cgs: np.ndarray
    metal_cooling_cgs: np.ndarray

    def primordial(self, log10_temperature: float | np.ndarray) -> float | np.ndarray:
        values = np.interp(log10_temperature, self.log10_temperature, self.primordial_cooling_cgs)
        if np.ndim(log10_temperature) == 0:
            return float(values)
        return values

    def metal(self, log10_temperature: float | np.ndarray) -> float | np.ndarray:
        values = np.interp(log10_temperature, self.log10_temperature, self.metal_cooling_cgs)
        if np.ndim(log10_temperature) == 0:
            return float(values)
        return values


_TABLE_CACHE: dict[str, CoolingTable] = {}
_DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "topaz_cooling_high_nh.csv"
_DEFAULT_TABLE_CACHE_KEY = "__default_topaz_table__"


@njit(cache=True)
def _interp_clamped_scalar(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    n = xp.shape[0]
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0 = xp[lo]
    x1 = xp[lo + 1]
    y0 = fp[lo]
    y1 = fp[lo + 1]
    if x1 == x0:
        return y0
    frac = (x - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


@njit(cache=True)
def _lambda_total_scalar_numba(
    temperature_K: float,
    metallicity_solar: float,
    log10_temperature: np.ndarray,
    primordial_cooling_cgs: np.ndarray,
    metal_cooling_cgs: np.ndarray,
) -> float:
    t_min = 10.0 ** log10_temperature[0]
    t_max = 10.0 ** log10_temperature[-1]
    t = temperature_K
    if t < t_min:
        t = t_min
    elif t > t_max:
        t = t_max

    log10_t = np.log10(t)
    lambda_prim = _interp_clamped_scalar(log10_t, log10_temperature, primordial_cooling_cgs)
    lambda_metal = _interp_clamped_scalar(log10_t, log10_temperature, metal_cooling_cgs)
    return lambda_prim + metallicity_solar * lambda_metal


@njit(cache=True)
def _lambda_total_array_numba(
    temperature_K: np.ndarray,
    metallicity_solar: np.ndarray,
    log10_temperature: np.ndarray,
    primordial_cooling_cgs: np.ndarray,
    metal_cooling_cgs: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(temperature_K)
    for i in range(temperature_K.size):
        out[i] = _lambda_total_scalar_numba(
            temperature_K[i],
            metallicity_solar[i],
            log10_temperature,
            primordial_cooling_cgs,
            metal_cooling_cgs,
        )
    return out


@njit(cache=True)
def _tcool_array_numba(
    temperature_K: np.ndarray,
    pressure_over_kb: np.ndarray,
    metallicity_solar: np.ndarray,
    mu: float,
    log10_temperature: np.ndarray,
    primordial_cooling_cgs: np.ndarray,
    metal_cooling_cgs: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(temperature_K)
    prefactor = 1.5 * (muH / mu) * kb
    mu_ratio = mu / muH

    for i in range(temperature_K.size):
        t = temperature_K[i]
        if t <= 0.0:
            out[i] = np.inf
            continue

        lambda_val = _lambda_total_scalar_numba(
            t,
            metallicity_solar[i],
            log10_temperature,
            primordial_cooling_cgs,
            metal_cooling_cgs,
        )
        n_h_actual = pressure_over_kb[i] / t * mu_ratio
        denominator = n_h_actual * lambda_val
        if denominator == 0.0:
            out[i] = np.inf
        else:
            out[i] = prefactor * t / denominator
    return out


@njit(cache=True)
def lambda_p_rho_topaz_scalar_numba(
    pressure: float,
    rho: float,
    mu: float,
    metallicity_solar: float,
    log10_temperature: np.ndarray,
    primordial_cooling_cgs: np.ndarray,
    metal_cooling_cgs: np.ndarray,
) -> float:
    if rho <= 0.0:
        return 0.0
    temperature = (pressure / kb) * (mu * mp / rho)
    return _lambda_total_scalar_numba(
        temperature,
        metallicity_solar,
        log10_temperature,
        primordial_cooling_cgs,
        metal_cooling_cgs,
    )


def _default_table_path() -> Path:
    return _DEFAULT_TABLE_PATH


def load_cooling_table(path: str | Path | None = None) -> CoolingTable:
    if path is None:
        cache_key = _DEFAULT_TABLE_CACHE_KEY
        table_path = _DEFAULT_TABLE_PATH
    else:
        table_path = Path(path)
        cache_key = str(table_path)
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


def lambda_total_cgs(
    temperature_K: float | np.ndarray,
    metallicity_solar: float | np.ndarray,
    table: CoolingTable | None = None,
    table_path: str | Path | None = None,
) -> float | np.ndarray:
    """Return total cooling function Λ(T, Z) in cgs from the Topaz table."""
    table_obj = table if table is not None else load_cooling_table(table_path)
    temp_b, z_b = np.broadcast_arrays(
        np.asarray(temperature_K, dtype=float),
        np.asarray(metallicity_solar, dtype=float),
    )
    temp_flat = np.ascontiguousarray(temp_b.ravel(), dtype=float)
    z_flat = np.ascontiguousarray(z_b.ravel(), dtype=float)
    total_flat = _lambda_total_array_numba(
        temp_flat,
        z_flat,
        table_obj.log10_temperature,
        table_obj.primordial_cooling_cgs,
        table_obj.metal_cooling_cgs,
    )
    total = total_flat.reshape(temp_b.shape)
    if total.ndim == 0:
        return float(total)
    return total


def tcool_P_topaz(
    temperature_K: float | np.ndarray,
    pressure_over_kb: float | np.ndarray,
    metallicity_solar: float | np.ndarray,
    mu: float,
    table: CoolingTable | None = None,
    table_path: str | Path | None = None,
) -> float | np.ndarray:
    """Cooling time t_cool(T, P/k_B, Z) using the Topaz cooling table."""
    table_obj = table if table is not None else load_cooling_table(table_path)
    t_b, p_b, z_b = np.broadcast_arrays(
        np.asarray(temperature_K, dtype=float),
        np.asarray(pressure_over_kb, dtype=float),
        np.asarray(metallicity_solar, dtype=float),
    )
    t_flat = np.ascontiguousarray(t_b.ravel(), dtype=float)
    p_flat = np.ascontiguousarray(p_b.ravel(), dtype=float)
    z_flat = np.ascontiguousarray(z_b.ravel(), dtype=float)
    result_flat = _tcool_array_numba(
        t_flat,
        p_flat,
        z_flat,
        mu,
        table_obj.log10_temperature,
        table_obj.primordial_cooling_cgs,
        table_obj.metal_cooling_cgs,
    )
    result = result_flat.reshape(t_b.shape)
    if result.ndim == 0:
        return float(result)
    return result


def tcool_P_topaz_vector(
    temperature_K: float | np.ndarray,
    pressure_over_kb: float,
    metallicity_solar: np.ndarray,
    mu: float,
    table: CoolingTable,
) -> np.ndarray:
    """Fast path for vector cooling-time calls used in ODE RHS."""
    z = np.ascontiguousarray(np.asarray(metallicity_solar, dtype=float))
    t_in = np.asarray(temperature_K, dtype=float)
    if t_in.ndim == 0:
        t = np.full(z.shape, float(t_in), dtype=float)
    else:
        t = np.ascontiguousarray(t_in, dtype=float)
        if t.shape != z.shape:
            raise ValueError("temperature_K and metallicity_solar must have identical shapes")
    p = np.full(t.shape, float(pressure_over_kb), dtype=float)
    return _tcool_array_numba(
        t,
        p,
        z,
        mu,
        table.log10_temperature,
        table.primordial_cooling_cgs,
        table.metal_cooling_cgs,
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

    def lambda_p_rho(point: tuple[float, float] | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(point, dtype=float)
        if arr.ndim == 1:
            pressure = float(arr[0])
            rho = float(arr[1])
            return float(
                lambda_p_rho_topaz_scalar_numba(
                    pressure,
                    rho,
                    mu,
                    metallicity_solar,
                    table.log10_temperature,
                    table.primordial_cooling_cgs,
                    table.metal_cooling_cgs,
                )
            )

        pressure = arr[..., 0]
        rho = arr[..., 1]
        out = np.zeros_like(pressure, dtype=float)
        it = np.nditer(
            [pressure, rho, out],
            flags=["refs_ok", "multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for p_i, r_i, out_i in it:
            out_i[...] = lambda_p_rho_topaz_scalar_numba(
                float(p_i),
                float(r_i),
                mu,
                metallicity_solar,
                table.log10_temperature,
                table.primordial_cooling_cgs,
                table.metal_cooling_cgs,
            )
        return out

    return lambda_p_rho
