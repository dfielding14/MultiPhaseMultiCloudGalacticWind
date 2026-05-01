"""Tests for the TRML/cloud sensitivity-screen harness."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "examples" / "trml_sensitivity_screen.py"
EXAMPLES_DIR = REPO_ROOT / "examples"


def load_harness_module():
    sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location("trml_sensitivity_screen", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_effective_mixing_parameters_are_screenable():
    harness = load_harness_module()

    assert "A_mix" in harness.PARAMETER_SPECS
    assert "beta_chi_mix" in harness.PARAMETER_SPECS
    assert harness.PARAMETER_SPECS["A_mix"].target == "config"
    assert harness.PARAMETER_SPECS["A_mix"].default == 1.0
    assert harness.PARAMETER_SPECS["beta_chi_mix"].target == "config"
    assert harness.PARAMETER_SPECS["beta_chi_mix"].default == 0.0

    args = harness.parse_args(
        [
            "--parameters",
            "A_mix,beta_chi_mix",
            "--truth-cases",
            "fiducial,strong_wings",
        ]
    )
    assert harness.selected_parameter_names(args) == ["A_mix", "beta_chi_mix"]
    assert harness.selected_truth_case_names(args) == ["fiducial", "strong_wings"]


def test_a_mix_sensitivity_smoke_writes_outputs(tmp_path):
    harness = load_harness_module()
    output_dir = tmp_path / "trml_a_mix_smoke"

    harness.main(
        [
            "--parameters",
            "A_mix",
            "--values",
            "1.0",
            "--truth-cases",
            "fiducial",
            "--observable-set",
            "logm0_mean_sigma_skew_kurt",
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
            "--failure-policy",
            "stalled_wind",
            "--skip-loading-degeneracy",
            "--no-plots",
            "--jax-platform",
            "cpu",
        ]
    )

    rows_path = output_dir / "trml_sensitivity_rows.csv"
    summary_path = output_dir / "trml_sensitivity_summary.csv"
    npz_path = output_dir / "trml_sensitivity_samples.npz"
    metadata_path = output_dir / "run_metadata.json"
    assert rows_path.exists()
    assert summary_path.exists()
    assert npz_path.exists()
    assert metadata_path.exists()

    with rows_path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["parameter_name"] == "A_mix"
    assert rows[0]["truth_case"] == "fiducial"
    assert rows[0]["error"] == ""
