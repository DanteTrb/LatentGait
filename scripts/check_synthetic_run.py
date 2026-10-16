"""Check final-protocol integration output structure without printing fitted values."""

import argparse
import csv
import json
from pathlib import Path


def check(root):
    root = Path(root)
    meta = json.loads((root / "provenance.json").read_text())
    assert meta["mode"] == "synthetic_execution_check"
    assert meta["status"] == "synthetic_check_complete_no_inference"
    assert meta["input_sha256"] == "generated_from_scratch"
    assert meta["protocol"]["protocol_version"] == 3
    assert meta["protocol"]["orientation"] == {
        "kinematics": [1, 1, 1],
        "rhythmicity": "largest_absolute_loading_positive",
        "complexity": "largest_absolute_loading_positive",
    }
    assert meta["protocol"]["bootstrap_scaling"] == "refit"
    assert meta["protocol"]["cumulative_rule"] == "central_90"
    with (root / "pca_loadings.csv").open() as handle:
        loading_rows = list(csv.DictReader(handle))
    groups = {}
    for row in loading_rows:
        groups.setdefault((row["domain"], int(row["component"])), []).append(float(row["loading"]))
    assert len(groups) == 15
    for key, values in groups.items():
        if key == ("kinematics", 1):
            assert sum(values) > 0
        else:
            assert max(values, key=abs) > 0
    for name in ("fixed", "free", "spline"):
        with (root / f"{name}_parameters.csv").open() as handle:
            fields = csv.DictReader(handle).fieldnames
        assert not any(c.startswith("or_") for c in fields)
    with (root / "logistic_associations.csv").open() as handle:
        assert "or_mean" in csv.DictReader(handle).fieldnames
    for suffix, expected in [("bootstrap", 8), ("bootstrap_stability", 16), ("component_diagnostics", 16)]:
        with (root / f"pca_{suffix}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows and all(int(r["n_resamples"]) == expected for r in rows)
    summary = json.loads((root / "aggregate_subject_contrasts.json").read_text())
    assert summary["cumulative_rule"] == "central_90"
    if summary["cumulative_interval"] is not None:
        assert summary["cumulative_interval_weight_fraction"] >= 0.9 - 1e-12
    regions = json.loads((root / "curve_region_sensitivity.json").read_text())
    assert set(regions) == {"fixed", "free", "spline"}
    for model in regions.values():
        assert len(model["checks"]) == 6
        assert model["primary_fraction"] == 0.8
        assert model["primary_grid_points"] == 1000
        assert model["refined_grid_points"] == 2000
    for path in root.glob("*.csv"):
        with path.open() as handle:
            fields = csv.DictReader(handle).fieldnames
        assert not {"participant_id", "trial_id"}.intersection(fields)
    assert (root / "prior_probability_curves.csv").exists()
    print("PASS: final-protocol synthetic reports, native coefficient scales and private output schema")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    check(parser.parse_args().output_directory)
