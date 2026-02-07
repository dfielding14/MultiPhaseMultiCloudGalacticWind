"""Topaz cooling-table loading and interpolation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

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


def _default_table_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "topaz_cooling_high_nh.csv"


def load_cooling_table(path: str | Path | None = None) -> CoolingTable:
    table_path = Path(path) if path is not None else _default_table_path()
    cache_key = str(table_path.resolve())
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = np.genfromtxt(table_path, delimiter=",", comments="#", names=True, skip_header=3)
    table = CoolingTable(
        log10_temperature=np.asarray(data["log10_T"], dtype=float),
        primordial_cooling_cgs=np.asarray(data["primordial_cooling_cgs"], dtype=float),
        metal_cooling_cgs=np.asarray(data["metal_cooling_cgs"], dtype=float),
    )
    _TABLE_CACHE[cache_key] = table
    return table


def lambda_total_cgs(
    temperature_K: float | np.ndarray,
    metallicity_solar: float | np.ndarray,
    table_path: str | Path | None = None,
) -> float | np.ndarray:
    """Return total cooling function Λ(T, Z) in cgs from the Topaz table."""
    table = load_cooling_table(table_path)
    temperature_arr = np.asarray(temperature_K, dtype=float)
    metallicity_arr = np.asarray(metallicity_solar, dtype=float)

    # Clamp to table support to avoid extrapolation artifacts.
    t_min = 10 ** float(table.log10_temperature[0])
    t_max = 10 ** float(table.log10_temperature[-1])
    t_clip = np.clip(temperature_arr, t_min, t_max)
    log10_t = np.log10(t_clip)

    lambda_prim = np.asarray(table.primordial(log10_t), dtype=float)
    lambda_metal = np.asarray(table.metal(log10_t), dtype=float)
    total = lambda_prim + metallicity_arr * lambda_metal
    if np.ndim(temperature_K) == 0 and np.ndim(metallicity_solar) == 0:
        return float(total)
    return total


def tcool_P_topaz(
    temperature_K: float | np.ndarray,
    pressure_over_kb: float | np.ndarray,
    metallicity_solar: float | np.ndarray,
    mu: float,
    table_path: str | Path | None = None,
) -> float | np.ndarray:
    """Cooling time t_cool(T, P/k_B, Z) using the Topaz cooling table."""
    t = np.asarray(temperature_K, dtype=float)
    p = np.asarray(pressure_over_kb, dtype=float)
    n_h_actual = p / t * (mu / muH)
    lambda_val = np.asarray(lambda_total_cgs(t, metallicity_solar, table_path), dtype=float)
    denominator = n_h_actual * lambda_val
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1.5 * (muH / mu) * kb * t / denominator

    if np.ndim(result) == 0:
        if float(denominator) == 0.0:
            return float(np.inf)
        return float(result)
    return np.where(denominator == 0.0, np.inf, result)


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
    t_min = 10 ** float(table.log10_temperature[0])
    t_max = 10 ** float(table.log10_temperature[-1])

    def lambda_p_rho(point: tuple[float, float] | np.ndarray) -> float | np.ndarray:
        arr = np.asarray(point, dtype=float)
        if arr.ndim == 1:
            pressure = float(arr[0])
            rho = float(arr[1])
            if rho <= 0.0:
                return 0.0
            temperature = np.clip((pressure / kb) * (mu * mp / rho), t_min, t_max)
            log10_t = np.log10(temperature)
            return float(
                table.primordial(log10_t) + metallicity_solar * table.metal(log10_t)
            )

        pressure = arr[..., 0]
        rho = arr[..., 1]
        safe_rho = np.where(rho > 0.0, rho, np.inf)
        temperature = np.clip((pressure / kb) * (mu * mp / safe_rho), t_min, t_max)
        log10_t = np.log10(temperature)
        lambda_total = (
            np.asarray(table.primordial(log10_t), dtype=float)
            + metallicity_solar * np.asarray(table.metal(log10_t), dtype=float)
        )
        return np.where(rho > 0.0, lambda_total, 0.0)

    return lambda_p_rho
