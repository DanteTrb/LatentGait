"""Auditable CLI for local, authorised data; it never writes into version control."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml
from .cohort import lock_subject_cohort
from .pca import fit_domain_pcas

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--output", required=True)
    a=ap.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text())
    cohort=lock_subject_cohort(pd.read_csv(cfg["input_table"]), id_col=cfg["id_column"], fall_col=cfg["fall_column"], expected_n=cfg.get("expected_unique_subjects",269))
    events=int(cohort[cfg["fall_column"]].sum())
    if cfg.get("expected_fallers") is not None and events != cfg["expected_fallers"]: raise ValueError("Faller count does not match configuration")
    if cfg.get("expected_nonfallers") is not None and len(cohort)-events != cfg["expected_nonfallers"]: raise ValueError("Non-faller count does not match configuration")
    r=fit_domain_pcas(cohort, cfg["domains"], n_bootstrap=500, seed=cfg.get("seed",42))
    out=Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for name, value in [("domain_pc1_scores",r.scores),("pc1_loadings",r.loadings),("pc1_variance",r.variance),("pc1_bootstrap_ci",r.bootstrap)]: value.to_csv(out/f"{name}.csv",index=False)
if __name__ == "__main__": main()
