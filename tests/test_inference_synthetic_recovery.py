"""Tests for the synthetic recovery inference harness."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "examples" / "inference_synthetic_recovery.py"


def load_harness_module():
    spec = importlib.util.spec_from_file_location("inference_synthetic_recovery", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_truth_cases_cover_workplan_names():
    harness = load_harness_module()

    expected = {
        "fiducial",
        "low_eta_m_high_eta_e",
        "high_eta_m_low_eta_e",
        "low_eta_m_cold",
        "high_eta_m_cold",
        "near_failure_boundary",
        "strong_wings",
        "narrow_profile",
    }

    assert expected.issubset(harness.TRUTH_CASES)
    for case in harness.TRUTH_CASES.values():
        theta = np.asarray(case.theta, dtype=float)
        assert theta.shape == (3,)
        assert np.all(theta > 0.0)
        assert theta[2] < harness.ETA_E_MAX


def test_covariance_builder_supports_all_observable_modes():
    harness = load_harness_module()

    class FakeModel:
        def __init__(self, observable_set, observable_dim):
            self.observable_set = observable_set
            self.observable_dim = observable_dim

    cases = [
        (FakeModel("m0_m1_m2", 3), np.array([1.0e19, 2.0e22, 3.0e25])),
        (FakeModel("logm0_mean_sigma_skew_kurt", 5), np.array([44.0, 300.0, 90.0, 0.2, 3.1])),
        (FakeModel("dndv_binned", 8), np.geomspace(1.0e12, 1.0e15, 8)),
    ]

    for model, values in cases:
        sigma, covariance = harness.build_synthetic_covariance(model, values, noise_fraction=0.1)
        assert sigma.shape == (model.observable_dim,)
        assert covariance.shape == (model.observable_dim, model.observable_dim)
        assert np.all(sigma > 0.0)
        assert np.allclose(covariance, covariance.T)
        assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_recovery_metric_summary_intervals_and_biases():
    harness = load_harness_module()

    theta_true = np.array([0.2, 0.1, 0.8])
    theta_map = np.array([0.22, 0.11, 0.78])
    samples = np.array(
        [
            [0.18, 0.08, 0.75],
            [0.20, 0.10, 0.80],
            [0.22, 0.12, 0.85],
        ],
        dtype=float,
    )

    metrics = harness.summarize_recovery_metrics(theta_true, theta_map, samples)

    assert metrics["truth_in_68pct_interval"].shape == (3,)
    assert metrics["truth_in_95pct_interval"].shape == (3,)
    assert np.all(metrics["truth_in_95pct_interval"])
    assert np.allclose(metrics["posterior_median_bias"], 0.0)
    assert np.allclose(metrics["map_error"], np.abs(theta_map - theta_true))
    assert np.all(metrics["posterior_width"] > 0.0)


def test_synthetic_recovery_smoke_writes_outputs(tmp_path):
    harness = load_harness_module()
    output_dir = tmp_path / "synthetic_recovery"

    harness.main(
        [
            "--truth-case",
            "fiducial",
            "--observable-set",
            "logm0_mean_sigma_skew_kurt",
            "--num-noise-realizations",
            "1",
            "--num-samples",
            "4",
            "--num-warmup",
            "2",
            "--seed",
            "321",
            "--output",
            str(output_dir),
            "--r-max-kpc",
            "2.0",
            "--step-kpc",
            "0.25",
            "--n-cloud-species",
            "2",
            "--cloud-mass-min",
            "10.0",
            "--cloud-mass-max",
            "1.0e3",
            "--sampler",
            "none",
            "--map-max-iter",
            "2",
            "--map-num-starts",
            "1",
            "--jax-platform",
            "cpu",
        ]
    )

    npz_path = output_dir / "synthetic_recovery_results.npz"
    csv_path = output_dir / "synthetic_recovery_summary.csv"
    metadata_path = output_dir / "run_metadata.json"

    assert npz_path.exists()
    assert csv_path.exists()
    assert metadata_path.exists()

    data = np.load(npz_path)
    assert data["theta_true"].shape == (1, 3)
    assert data["true_observables"].shape == (1, 5)
    assert data["observed"].shape == (1, 5)
    assert data["covariance"].shape == (1, 5, 5)
    assert data["theta_map"].shape == (1, 3)
    assert data["samples_theta"].shape[0] == 1
    assert data["posterior_predictive"].shape[0] == 1

    with open(csv_path, newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["truth_case"] == "fiducial"
    assert {"true_eta_M", "map_eta_M", "map_error_eta_M", "success"}.issubset(rows[0])
