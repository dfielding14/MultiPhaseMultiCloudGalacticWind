"""Tests for the prior predictive inference harness."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "examples" / "inference_prior_predictive.py"


def load_harness_module():
    spec = importlib.util.spec_from_file_location("inference_prior_predictive", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sample_prior_shape_and_bounds():
    harness = load_harness_module()
    rng = np.random.default_rng(123)

    samples = harness.sample_prior(rng, 128)

    assert samples.shape == (128, 3)
    for idx, name in enumerate(harness.PARAM_NAMES):
        lo, hi = harness.DEFAULT_PRIOR_BOUNDS[name]
        assert np.all(samples[:, idx] >= lo)
        assert np.all(samples[:, idx] <= hi)
    assert np.all(samples[:, 2] < harness.ETA_E_MAX)


def test_prior_predictive_smoke_writes_outputs(tmp_path):
    harness = load_harness_module()
    output_dir = tmp_path / "prior_predictive"

    harness.main(
        [
            "--num-samples",
            "2",
            "--seed",
            "321",
            "--output",
            str(output_dir),
            "--r-max-kpc",
            "2.0",
            "--step-kpc",
            "0.2",
            "--n-cloud-species",
            "2",
            "--cloud-mass-min",
            "10.0",
            "--cloud-mass-max",
            "1.0e3",
            "--jax-platform",
            "cpu",
            "--no-usetex",
        ]
    )

    npz_path = output_dir / "prior_predictive_samples.npz"
    csv_path = output_dir / "prior_predictive_summary.csv"
    metadata_path = output_dir / "run_metadata.json"

    assert npz_path.exists()
    assert csv_path.exists()
    assert metadata_path.exists()
    assert (output_dir / "prior_predictive_validity.png").exists()
    assert (output_dir / "prior_predictive_observable_histograms.png").exists()
    assert (output_dir / "prior_predictive_parameter_scatter.png").exists()

    data = np.load(npz_path)
    assert data["theta_samples"].shape == (2, 3)
    assert data["observables"].shape[0] == 2
    assert data["raw_moments"].shape == (2, 5)
    assert data["valid"].shape == (2,)

    with open(csv_path, newline="", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {"eta_M", "eta_M_cold", "eta_E", "valid", "logM0", "mean_v", "sigma_v"}.issubset(rows[0])


def test_evaluate_sample_records_exception_without_crashing():
    harness = load_harness_module()

    class FailingModel:
        observable_dim = 5
        observable_names = ("a", "b", "c", "d", "e")

        def _predict_theta_with_valid_fn(self, theta):
            raise RuntimeError("intentional failure")

    result = harness.evaluate_sample(FailingModel(), np.array([0.2, 0.1, 0.8]), sample_id=7)

    assert result.sample_id == 7
    assert not result.valid
    assert "intentional failure" in result.error
    assert result.observables.shape == (5,)
    assert result.raw_moments.shape == (5,)
    assert np.all(np.isnan(result.observables))
    assert np.all(np.isnan(result.raw_moments))
