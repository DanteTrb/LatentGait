"""Optional private raw-signal stage: metadata and full-stride events are user supplied."""

import json

import numpy as np
import pandas as pd
import yaml

from .features import extract_axis, trim_boundaries


def extract_manifest(manifest_path, protocol_path, destination):
    from .workflow import external_path

    protocol = yaml.safe_load(protocol_path.read_text())
    if (
        protocol.get("sampling_hz") != 100
        or not protocol.get("axis_mapping_reference")
        or not protocol.get("bts_export_version_and_units")
    ):
        raise ValueError("Resolve sensor axis, 100-Hz acquisition and vendor export metadata")
    rbf, rqa = protocol["rbf"], protocol["rqa"]
    if (
        rbf.get("kernel") is None
        or rbf.get("smoothing") is None
        or any(rqa.get(k) is None for k in ("ami_bins", "rtol", "atol", "radius_fraction", "lmin", "theiler"))
    ):
        raise ValueError("Unresolved RBF/RQA protocol choices; extraction stopped")
    records = json.loads(manifest_path.read_text())["records"]
    if not records:
        raise ValueError("Empty signal manifest")
    rows, seen = [], set()
    for record in records:
        if not isinstance(record["participant_id"], str) or not isinstance(record["trial_id"], str):
            raise ValueError("Signal manifest identifiers must be explicit strings")
        subject, trial = record["participant_id"].strip(), record["trial_id"].strip()
        if not subject or not trial or subject in seen:
            raise ValueError("One externally verified trial per participant is required")
        seen.add(subject)
        signal = pd.read_csv(external_path(record["signal_csv"]))
        if not set(("time_s", "v", "ml", "ap")).issubset(signal.columns):
            raise ValueError("Raw signal schema: time_s,v,ml,ap")
        if not np.isfinite(signal[["time_s", "v", "ml", "ap"]].to_numpy()).all():
            raise ValueError("Nonfinite signal sample")
        if not np.allclose(np.diff(signal.time_s), 0.01, rtol=1e-5, atol=1e-8):
            raise ValueError("Signal is not uniformly sampled at 100 Hz")
        boundaries = np.asarray(record["stride_boundaries"])
        kept = trim_boundaries(signal.v, boundaries)
        row = dict(participant_id=subject, trial_id=trial, retained_strides=len(kept) - 1)
        for axis in ("v", "ml", "ap"):
            values = extract_axis(signal[axis].to_numpy(), boundaries, axis, rbf=rbf, rqa=rqa)
            row.update({f"{family}_{axis}": values[family] for family in ("hr", "ihr", "det", "mse")})
            row[f"rqa_delay_{axis}"] = values["delay"]
            row[f"rqa_embedding_dimension_{axis}"] = values["dimension"]
        rows.append(row)
    # This private output necessarily has linkage IDs. Never write or publish it in Git.
    with destination.open("x") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)
