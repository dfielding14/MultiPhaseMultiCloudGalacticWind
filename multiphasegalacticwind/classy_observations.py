"""CLASSY observational data loaders for inference workflows.

This module provides a normalized interface to the processed CLASSY dataset:
- one catalog row per galaxy (profile-complete subset) with SFR and galaxy-size fields,
- ragged per-object dN/dv profiles,
- convenience helper to build binned dN/dv vectors and covariance matrices
  suitable for ``MomentInferenceModel(observable_set='dndv_binned')``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_CATALOG_PATH = _DATA_DIR / "classy_catalog.csv"
_DEFAULT_PROFILES_PATH = _DATA_DIR / "classy_profiles.npz"
__all__ = [
    "ClassyObservation",
    "BinnedDndvObservation",
    "ClassyInferenceInputs",
    "load_classy_observations",
    "get_classy_observation",
    "build_binned_dndv_observation",
    "build_classy_inference_inputs",
]

_OBS_MOMENTS3 = "m0_m1_m2"
_OBS_SHAPE5 = "logm0_mean_sigma_skew_kurt"
_OBS_DNDV_BINNED = "dndv_binned"


def _safe_float(text: str) -> float:
    value = str(text).strip()
    if value == "":
        return float("nan")
    return float(value)


@dataclass(frozen=True)
class ClassyObservation:
    """Normalized observational record for one CLASSY galaxy.

    Attributes
    ----------
    object_id
        Canonical galaxy identifier.
    sfr_msun_per_yr
        Star-formation rate [Msun/yr].
    r_gal_kpc
        Galaxy size proxy [kpc]. In this catalog, this is set to R50.
    r50_kpc
        Half-light radius R50 [kpc] from ancillary table.
    r_star_kpc
        Characteristic stellar radius r_* [kpc] derived from Logr_*.
    v_circ_kms
        Circular velocity [km/s] from ancillary table.
    z
        Redshift.
    mean_v_cloud_kms, hwhm_kms, total_lognh_cm2
        Profile summary fields from the CLASSY NH profile table.
    nh_int_left_kms, nh_int_right_kms
        NH integration boundaries [km/s] from CLASSY profile table.
    velocity_kms
        Observed velocity grid [km/s], typically blueshifted (negative).
    dndv_cm2_per_kms
        Observed profile dN/dv [cm^-2 / (km/s)] on ``velocity_kms``.
    """

    object_id: str
    sfr_msun_per_yr: float
    r_gal_kpc: float
    r50_kpc: float
    r_star_kpc: float
    v_circ_kms: float
    z: float
    mean_v_cloud_kms: float
    hwhm_kms: float
    total_lognh_cm2: float
    nh_int_left_kms: float
    nh_int_right_kms: float
    velocity_kms: np.ndarray
    dndv_cm2_per_kms: np.ndarray

    @property
    def has_profile(self) -> bool:
        """Return True when this object has a usable dN/dv profile."""
        return self.velocity_kms.size > 0 and self.dndv_cm2_per_kms.size > 0


@dataclass(frozen=True)
class BinnedDndvObservation:
    """Inference-ready binned dN/dv product."""

    object_id: str
    velocity_bins_kms: np.ndarray
    observed_dndv: np.ndarray
    covariance_dndv: np.ndarray


@dataclass(frozen=True)
class ClassyInferenceInputs:
    """Inference-ready per-galaxy inputs derived from CLASSY observations.

    Attributes
    ----------
    object_id
        Canonical galaxy identifier.
    observable_set
        One of ``m0_m1_m2``, ``logm0_mean_sigma_skew_kurt``, or ``dndv_binned``.
    model_kwargs
        Keyword arguments for ``MomentInferenceModel`` construction.
        Contains ``sfr``, ``r_star_kpc``, and ``v_circ``.
    config_kwargs
        Keyword arguments for ``WindConfig`` to set fixed metallicities.
    observed
        Observable vector used in likelihood evaluation.
    covariance
        Observable covariance matrix.
    observable_names
        Observable labels matching ``observed`` order.
    velocity_bins_kms
        Velocity-bin centers for ``dndv_binned`` mode; otherwise ``None``.
    """

    object_id: str
    observable_set: str
    model_kwargs: Dict[str, float]
    config_kwargs: Dict[str, float]
    observed: np.ndarray
    covariance: np.ndarray
    observable_names: tuple[str, ...]
    velocity_bins_kms: np.ndarray | None = None


def _stabilize_covariance(cov: np.ndarray, min_eig: float = 1e-20) -> np.ndarray:
    """Project covariance to symmetric positive-definite with eigenvalue floor."""
    cov = np.asarray(cov, dtype=float)
    sym = 0.5 * (cov + cov.T)
    sym = np.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)
    scale = max(float(np.max(np.abs(sym))), 1.0)
    sym_scaled = sym / scale
    min_eig_scaled = float(min_eig) / scale

    eigvals = np.linalg.eigvalsh(sym_scaled)
    eigvals = np.nan_to_num(eigvals, nan=min_eig_scaled, posinf=min_eig_scaled, neginf=min_eig_scaled)
    shift = max(min_eig_scaled - float(np.min(eigvals)), 0.0)
    spd_scaled = sym_scaled + (shift + 1e-14) * np.eye(sym_scaled.shape[0], dtype=float)
    spd = spd_scaled * scale
    return 0.5 * (spd + spd.T)


def _load_catalog_rows(catalog_path: Path) -> list[dict[str, str]]:
    with catalog_path.open("r", encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_profiles_npz(profiles_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(profiles_path, allow_pickle=False) as data:
        object_ids = np.asarray(data["object_ids"], dtype=str)
        offsets = np.asarray(data["profile_offsets"], dtype=np.int64)
        velocity = np.asarray(data["velocity_kms"], dtype=float)
        dndv = np.asarray(data["dndv_cm2_per_kms"], dtype=float)
    return object_ids, offsets, velocity, dndv


def load_classy_observations(
    include_missing_profiles: bool = True,
    catalog_path: str | Path | None = None,
    profiles_path: str | Path | None = None,
) -> Dict[str, ClassyObservation]:
    """Load normalized CLASSY observations indexed by object id.

    Parameters
    ----------
    include_missing_profiles
        If False, only return objects with non-empty dN/dv profiles.
        The default processed dataset is already profile-complete, so this flag
        matters only for custom catalog/profile overrides.
    catalog_path, profiles_path
        Optional overrides for processed CLASSY files.
    """
    catalog_file = Path(catalog_path) if catalog_path is not None else _DEFAULT_CATALOG_PATH
    profiles_file = Path(profiles_path) if profiles_path is not None else _DEFAULT_PROFILES_PATH

    rows = _load_catalog_rows(catalog_file)
    object_ids_npz, offsets, velocity_flat, dndv_flat = _load_profiles_npz(profiles_file)

    offset_lookup = {obj: i for i, obj in enumerate(object_ids_npz.tolist())}
    observations: Dict[str, ClassyObservation] = {}

    for row in rows:
        obj = row["object_id"].strip()
        if obj not in offset_lookup:
            continue

        i = offset_lookup[obj]
        lo = int(offsets[i])
        hi = int(offsets[i + 1])
        velocity = velocity_flat[lo:hi].copy()
        dndv = dndv_flat[lo:hi].copy()

        obs = ClassyObservation(
            object_id=obj,
            sfr_msun_per_yr=_safe_float(row["sfr_msun_per_yr"]),
            r_gal_kpc=_safe_float(row["r_gal_kpc"]),
            r50_kpc=_safe_float(row["r50_kpc"]),
            r_star_kpc=_safe_float(row["r_star_kpc"]),
            v_circ_kms=_safe_float(row.get("v_circ_kms", "nan")),
            z=_safe_float(row["z"]),
            mean_v_cloud_kms=_safe_float(row["mean_v_cloud_kms"]),
            hwhm_kms=_safe_float(row["hwhm_kms"]),
            total_lognh_cm2=_safe_float(row["total_lognh_cm2"]),
            nh_int_left_kms=_safe_float(row["nh_int_left_kms"]),
            nh_int_right_kms=_safe_float(row["nh_int_right_kms"]),
            velocity_kms=velocity,
            dndv_cm2_per_kms=dndv,
        )

        if include_missing_profiles or obs.has_profile:
            observations[obj] = obs

    return observations


def get_classy_observation(
    object_id: str,
    catalog_path: str | Path | None = None,
    profiles_path: str | Path | None = None,
) -> ClassyObservation:
    """Return one CLASSY observation by canonical object id."""
    observations = load_classy_observations(
        include_missing_profiles=True,
        catalog_path=catalog_path,
        profiles_path=profiles_path,
    )
    key = object_id.strip()
    if key not in observations:
        available = sorted(observations.keys())
        raise KeyError(f"Unknown CLASSY object '{object_id}'. Available count={len(available)}")
    return observations[key]


def build_binned_dndv_observation(
    observation: ClassyObservation,
    num_bins: int = 20,
    vmin_kms: float = 0.0,
    vmax_kms: float | None = None,
    fractional_error: float = 0.10,
    min_error_floor_fraction: float = 0.03,
    ar1_rho: float = 0.0,
    use_absolute_velocity: bool = True,
) -> BinnedDndvObservation:
    """Convert an observed profile to fixed-bin dN/dv + covariance.

    The returned vectors are shaped for
    ``MomentInferenceModel(observable_set='dndv_binned')``.

    Parameters
    ----------
    observation
        One CLASSY object with profile data.
    num_bins
        Number of target velocity bins.
    vmin_kms, vmax_kms
        Bin-edge range in km/s. ``vmax_kms`` defaults to max observed |v|.
    fractional_error
        Relative error model applied to each binned dN/dv point.
    min_error_floor_fraction
        Floor for errors relative to peak binned dN/dv.
    ar1_rho
        AR(1) adjacent-bin correlation coefficient.
    use_absolute_velocity
        If True (default), map observed blueshifted velocities to |v|.
    """
    if not observation.has_profile:
        raise ValueError(f"Object {observation.object_id} has no dN/dv profile")
    if num_bins < 6:
        raise ValueError("num_bins must be >= 6")

    v_obs = np.asarray(observation.velocity_kms, dtype=float)
    d_obs = np.asarray(observation.dndv_cm2_per_kms, dtype=float)

    if use_absolute_velocity:
        v_obs = np.abs(v_obs)

    valid = np.isfinite(v_obs) & np.isfinite(d_obs) & (d_obs > 0.0)
    v_obs = v_obs[valid]
    d_obs = d_obs[valid]
    if v_obs.size < 2:
        raise ValueError(f"Object {observation.object_id} does not have enough valid profile points")

    order = np.argsort(v_obs)
    v_obs = v_obs[order]
    d_obs = d_obs[order]

    v_unique, inv = np.unique(v_obs, return_inverse=True)
    d_unique = np.zeros_like(v_unique)
    counts = np.zeros_like(v_unique)
    for i, idx in enumerate(inv):
        d_unique[idx] += d_obs[i]
        counts[idx] += 1.0
    d_unique = d_unique / np.maximum(counts, 1.0)

    vmax = float(np.max(v_unique)) if vmax_kms is None else float(vmax_kms)
    if vmax <= vmin_kms:
        raise ValueError("vmax_kms must exceed vmin_kms")

    edges = np.linspace(float(vmin_kms), vmax, int(num_bins) + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    log_floor = np.log(1e-80)
    log_profile = np.log(np.clip(d_unique, 1e-80, None))
    log_interp = np.interp(centers, v_unique, log_profile, left=log_floor, right=log_floor)
    observed = np.exp(log_interp)
    observed = np.maximum(observed, 1e-40)

    amp = max(float(np.max(observed)), 1e-40)
    sigma = np.maximum(float(fractional_error) * observed, float(min_error_floor_fraction) * amp)

    rho = float(np.clip(ar1_rho, -0.95, 0.95))
    idx = np.arange(observed.size)
    corr = rho ** np.abs(idx[:, None] - idx[None, :])
    cov = np.outer(sigma, sigma) * corr
    cov = _stabilize_covariance(cov)

    return BinnedDndvObservation(
        object_id=observation.object_id,
        velocity_bins_kms=centers,
        observed_dndv=observed,
        covariance_dndv=cov,
    )


def _resolve_observable_set(observable_set: str) -> str:
    key = observable_set.strip().lower()
    aliases = {
        "m0_m1_m2": _OBS_MOMENTS3,
        "moments3": _OBS_MOMENTS3,
        "raw_moments": _OBS_MOMENTS3,
        "logm0_mean_sigma_skew_kurt": _OBS_SHAPE5,
        "shape5": _OBS_SHAPE5,
        "transformed_moments": _OBS_SHAPE5,
        "log_shape_moments": _OBS_SHAPE5,
        "dndv_binned": _OBS_DNDV_BINNED,
        "dndv": _OBS_DNDV_BINNED,
        "binned_dndv": _OBS_DNDV_BINNED,
        "full_dndv": _OBS_DNDV_BINNED,
    }
    if key not in aliases:
        allowed = sorted(set(aliases.keys()))
        raise ValueError(f"Unsupported observable_set '{observable_set}'. Expected one of {allowed}.")
    return aliases[key]


def _extract_radius_kpc(observation: ClassyObservation, radius_field: str) -> float:
    key = radius_field.strip().lower()
    options = {
        "r_gal_kpc": observation.r_gal_kpc,
        "r50_kpc": observation.r50_kpc,
        "r_star_kpc": observation.r_star_kpc,
    }
    if key not in options:
        allowed = sorted(options.keys())
        raise ValueError(f"Unsupported radius_field '{radius_field}'. Expected one of {allowed}.")
    value = float(options[key])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"radius_field '{radius_field}' is non-finite or non-positive for {observation.object_id}")
    return value


def _profile_raw_moments(
    observation: ClassyObservation,
    max_order: int = 4,
    use_absolute_velocity: bool = True,
) -> np.ndarray:
    """Compute raw velocity moments from one CLASSY dN/dv profile."""
    if not observation.has_profile:
        raise ValueError(f"Object {observation.object_id} has no dN/dv profile")
    if max_order < 2:
        raise ValueError("max_order must be >= 2")

    velocity = np.asarray(observation.velocity_kms, dtype=float)
    dndv = np.asarray(observation.dndv_cm2_per_kms, dtype=float)

    if use_absolute_velocity:
        velocity = np.abs(velocity)

    valid = np.isfinite(velocity) & np.isfinite(dndv) & (dndv > 0.0)
    velocity = velocity[valid]
    dndv = dndv[valid]
    if velocity.size < 2:
        raise ValueError(f"Object {observation.object_id} does not have enough valid profile samples")

    order = np.argsort(velocity)
    velocity = velocity[order]
    dndv = dndv[order]

    velocity_unique, inverse = np.unique(velocity, return_inverse=True)
    dndv_unique = np.zeros_like(velocity_unique)
    counts = np.zeros_like(velocity_unique)
    for i, idx in enumerate(inverse):
        dndv_unique[idx] += dndv[i]
        counts[idx] += 1.0
    dndv_unique = dndv_unique / np.maximum(counts, 1.0)

    moments = np.asarray(
        [np.trapezoid(dndv_unique * velocity_unique**n, velocity_unique) for n in range(max_order + 1)],
        dtype=float,
    )
    return moments


def _shape5_from_raw_moments(raw_moments: np.ndarray) -> np.ndarray:
    """Return [logM0, mean_v, sigma_v, skewness, kurtosis] from raw [M0..M4]."""
    raw = np.asarray(raw_moments, dtype=float)
    if raw.shape[0] < 5:
        raise ValueError("raw_moments must contain at least M0..M4")

    m0 = max(float(raw[0]), 1e-300)
    mean_v = float(raw[1]) / m0
    second = float(raw[2]) / m0
    variance = max(second - mean_v * mean_v, 1e-24)
    sigma_v = np.sqrt(variance)

    third = float(raw[3]) / m0
    fourth = float(raw[4]) / m0
    mu3 = third - 3.0 * mean_v * second + 2.0 * mean_v**3
    mu4 = fourth - 4.0 * mean_v * third + 6.0 * mean_v * mean_v * second - 3.0 * mean_v**4

    skewness = mu3 / max(sigma_v**3, 1e-24)
    kurtosis = mu4 / max(sigma_v**4, 1e-24)
    return np.asarray([np.log(m0), mean_v, sigma_v, skewness, kurtosis], dtype=float)


def _build_diagonal_covariance(values: np.ndarray, fractional_error: float) -> np.ndarray:
    """Build a diagonal covariance using a fractional error with a unit floor."""
    if fractional_error <= 0.0:
        raise ValueError("fractional_error must be positive")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be 1D")
    sigma = fractional_error * np.maximum(np.abs(values), 1.0)
    sigma = np.maximum(sigma, 1e-12)
    return np.diag(sigma * sigma)


def build_classy_inference_inputs(
    observation: ClassyObservation,
    observable_set: str = _OBS_SHAPE5,
    radius_field: str = "r_gal_kpc",
    fractional_error: float = 0.10,
    dndv_num_bins: int = 20,
    dndv_vmin_kms: float = 0.0,
    dndv_vmax_kms: float | None = None,
    dndv_min_error_floor_fraction: float = 0.03,
    dndv_ar1_rho: float = 0.0,
    use_absolute_velocity: bool = True,
    z_hot_over_z_solar: float = 10**-0.5,
    z_cloud_over_z_solar: float = 0.3,
) -> ClassyInferenceInputs:
    """Build per-galaxy inference inputs using CLASSY observables.

    This helper wires ``sfr``, launch radius, and ``v_circ`` into
    ``MomentInferenceModel`` kwargs and returns an observed vector + covariance
    for the requested observable set.
    """
    obs_key = _resolve_observable_set(observable_set)
    r_star_kpc = _extract_radius_kpc(observation, radius_field)
    v_circ = float(observation.v_circ_kms)
    if not np.isfinite(v_circ) or v_circ <= 0.0:
        raise ValueError(f"v_circ_kms is non-finite or non-positive for {observation.object_id}")

    model_kwargs: Dict[str, float] = {
        "sfr": float(observation.sfr_msun_per_yr),
        "r_star_kpc": r_star_kpc,
        "v_circ": v_circ,
    }
    config_kwargs: Dict[str, float] = {
        "Z_hot_over_Z_solar": float(z_hot_over_z_solar),
        "Z_cloud_over_Z_solar": float(z_cloud_over_z_solar),
    }

    if obs_key == _OBS_DNDV_BINNED:
        dndv = build_binned_dndv_observation(
            observation,
            num_bins=dndv_num_bins,
            vmin_kms=dndv_vmin_kms,
            vmax_kms=dndv_vmax_kms,
            fractional_error=fractional_error,
            min_error_floor_fraction=dndv_min_error_floor_fraction,
            ar1_rho=dndv_ar1_rho,
            use_absolute_velocity=use_absolute_velocity,
        )
        names = tuple(f"dN/dv@{v:.0f}km/s" for v in dndv.velocity_bins_kms)
        return ClassyInferenceInputs(
            object_id=observation.object_id,
            observable_set=obs_key,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            observed=np.asarray(dndv.observed_dndv, dtype=float),
            covariance=np.asarray(dndv.covariance_dndv, dtype=float),
            observable_names=names,
            velocity_bins_kms=np.asarray(dndv.velocity_bins_kms, dtype=float),
        )

    raw = _profile_raw_moments(observation, max_order=4, use_absolute_velocity=use_absolute_velocity)
    if obs_key == _OBS_MOMENTS3:
        observed = np.asarray(raw[:3], dtype=float)
        names = ("M0", "M1", "M2")
    else:
        observed = _shape5_from_raw_moments(raw)
        names = ("logM0", "mean_v", "sigma_v", "skewness", "kurtosis")

    covariance = _build_diagonal_covariance(observed, fractional_error=fractional_error)
    return ClassyInferenceInputs(
        object_id=observation.object_id,
        observable_set=obs_key,
        model_kwargs=model_kwargs,
        config_kwargs=config_kwargs,
        observed=observed,
        covariance=covariance,
        observable_names=names,
        velocity_bins_kms=None,
    )
