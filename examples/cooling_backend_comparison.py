#!/usr/bin/env python3
"""
Compare legacy vs Topaz cooling backends and explore Topaz table reduction.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from multiphasegalacticwind import WindModel


DEFAULT_TOPAZ_TABLE = (
    Path(__file__).resolve().parent.parent
    / "multiphasegalacticwind"
    / "data"
    / "topaz_cooling_high_nh.csv"
)


def run_case(params: dict, backend: str, table_path: str | None = None) -> dict[str, float]:
    model = WindModel(
        **params,
        rtol=1e-6,
        atol=1e-8,
        cooling_backend=backend,
        topaz_cooling_table_path=table_path,
    )
    start = time.perf_counter()
    sol = model.run()
    wall = time.perf_counter() - start
    obs_r_max = max(0.3, min(10.0, float(sol.r[-1])))
    moments = sol.calculate_velocity_moments(r_min_kpc=0.3, r_max_kpc=obs_r_max)
    return {
        "time_s": wall,
        "r_end_kpc": float(sol.r[-1]),
        "v_10kpc_kms": float(sol.v_at_10kpc),
        "eta_10kpc": float(sol.mass_loading_at_10kpc),
        "v_mean_kms": float(moments.get("mean", np.nan)),
        "v_disp_kms": float(moments.get("dispersion", np.nan)),
    }


def pct_delta(ref: float, val: float) -> float:
    if not np.isfinite(ref) or ref == 0.0:
        return np.nan
    return 100.0 * (val - ref) / ref


def load_topaz_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", comments="#", names=True, skip_header=3)
    return (
        np.asarray(data["log10_T"], dtype=float),
        np.asarray(data["primordial_cooling_cgs"], dtype=float),
        np.asarray(data["metal_cooling_cgs"], dtype=float),
    )


def write_topaz_table(path: Path, log_t: np.ndarray, prim: np.ndarray, metal: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# source_file=topaz_cooling_high_nh.csv\n")
        f.write("# selected_nH_index=80\n")
        f.write("# selected_log10_nH=0.00000000\n")
        f.write("log10_T,primordial_cooling_cgs,metal_cooling_cgs\n")
        for x, p, m in zip(log_t, prim, metal):
            f.write(f"{x:.17e},{p:.17e},{m:.17e}\n")


def ensure_last(indices: np.ndarray, n: int) -> np.ndarray:
    if indices.size == 0 or indices[-1] != n - 1:
        return np.append(indices, n - 1)
    return indices


def main() -> None:
    cases = [
        (
            "m82_baseline",
            dict(
                SFR=20.0,
                v_circ=150.0,
                eta_M=0.1,
                eta_M_cold=0.2,
                eta_E=1.0,
                N_cloud_species=13,
                cloud_mass_range=(1.0, 1.0e6),
                cloud_alpha=2.0,
                r_max_kpc=30.0,
            ),
        ),
        (
            "colder_heavy",
            dict(
                SFR=20.0,
                v_circ=150.0,
                eta_M=0.1,
                eta_M_cold=0.5,
                eta_E=1.0,
                N_cloud_species=13,
                cloud_mass_range=(1.0, 1.0e6),
                cloud_alpha=2.0,
                r_max_kpc=30.0,
            ),
        ),
    ]

    print("Legacy vs Topaz backend comparison")
    for name, params in cases:
        legacy = run_case(params, backend="legacy")
        topaz = run_case(params, backend="topaz")
        print(f"\nCASE {name}")
        print(f"  legacy: {legacy}")
        print(f"  topaz : {topaz}")
        print(
            "  delta% topaz-vs-legacy:"
            f" time={pct_delta(legacy['time_s'], topaz['time_s']):.3f},"
            f" v10={pct_delta(legacy['v_10kpc_kms'], topaz['v_10kpc_kms']):.3f},"
            f" eta10={pct_delta(legacy['eta_10kpc'], topaz['eta_10kpc']):.3f},"
            f" mean={pct_delta(legacy['v_mean_kms'], topaz['v_mean_kms']):.3f},"
            f" disp={pct_delta(legacy['v_disp_kms'], topaz['v_disp_kms']):.3f}"
        )

    print("\nTopaz table-reduction exploration")
    log_t, prim, metal = load_topaz_arrays(DEFAULT_TOPAZ_TABLE)
    log_t_min = np.log10(3000.0)
    trunc_mask = log_t >= log_t_min
    workdir = Path(tempfile.mkdtemp(prefix="topaz_cooling_"))

    full_path = workdir / "topaz_full.csv"
    write_topaz_table(full_path, log_t, prim, metal)

    trunc_path = workdir / "topaz_trunc3000.csv"
    write_topaz_table(trunc_path, log_t[trunc_mask], prim[trunc_mask], metal[trunc_mask])

    idx_half = ensure_last(np.arange(log_t.size)[::2], log_t.size)
    half_path = workdir / "topaz_half_uniform.csv"
    write_topaz_table(half_path, log_t[idx_half], prim[idx_half], metal[idx_half])

    log_t_tr, prim_tr, metal_tr = log_t[trunc_mask], prim[trunc_mask], metal[trunc_mask]
    idx_tr_half = ensure_last(np.arange(log_t_tr.size)[::2], log_t_tr.size)
    trunc_half_path = workdir / "topaz_trunc3000_half_uniform.csv"
    write_topaz_table(
        trunc_half_path, log_t_tr[idx_tr_half], prim_tr[idx_tr_half], metal_tr[idx_tr_half]
    )

    variants = [
        ("topaz_full", str(full_path)),
        ("trunc3000", str(trunc_path)),
        ("half_uniform", str(half_path)),
        ("trunc3000_half", str(trunc_half_path)),
    ]

    for name, params in cases:
        print(f"\nCASE {name}")
        outputs: dict[str, dict[str, float]] = {}
        for variant_name, variant_path in variants:
            outputs[variant_name] = run_case(params, backend="topaz", table_path=variant_path)
            print(f"  {variant_name}: {outputs[variant_name]}")

        ref = outputs["topaz_full"]
        print("  delta% vs topaz_full:")
        for variant_name in ("trunc3000", "half_uniform", "trunc3000_half"):
            row = outputs[variant_name]
            print(
                f"    {variant_name}:"
                f" time={pct_delta(ref['time_s'], row['time_s']):.3f},"
                f" v10={pct_delta(ref['v_10kpc_kms'], row['v_10kpc_kms']):.3f},"
                f" eta10={pct_delta(ref['eta_10kpc'], row['eta_10kpc']):.3f},"
                f" mean={pct_delta(ref['v_mean_kms'], row['v_mean_kms']):.3f},"
                f" disp={pct_delta(ref['v_disp_kms'], row['v_disp_kms']):.3f}"
            )

    print(f"\nTemporary generated tables written to: {workdir}")


if __name__ == "__main__":
    main()
