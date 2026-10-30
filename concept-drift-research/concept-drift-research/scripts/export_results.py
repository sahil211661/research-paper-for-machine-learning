#!/usr/bin/env python
"""Rebuild the CSV result tables and a Markdown results summary from saved raw results.

Usage: python scripts/export_results.py --tag main
Requires results/raw/<tag>_runs.csv etc. produced by run_experiments.py or the notebook.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                           # noqa: E402
from src.experiments import StudyResults                     # noqa: E402
from src.reporting import build_tables, save_tables, narrate_h1  # noqa: E402


def load_saved(tag: str) -> StudyResults:
    raw, logs = ROOT / "results" / "raw", ROOT / "results" / "logs"
    z = np.load(logs / f"{tag}_prequential_errors.npz", allow_pickle=False)
    errors = {}
    for k, e in zip(z["keys"], z["errors"]):
        sc, pol, seed = str(k).split("|")
        errors[(sc, pol, int(seed))] = e
    mem = raw / f"{tag}_memory.csv"
    return StudyResults(pd.read_csv(raw / f"{tag}_runs.csv"), pd.read_csv(raw / f"{tag}_drift_events.csv"),
                        pd.read_csv(logs / f"{tag}_event_log.csv"), errors,
                        pd.read_csv(raw / f"{tag}_monitor_runs.csv"), pd.DataFrame(),
                        pd.read_csv(mem) if mem.exists() else pd.DataFrame())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--profile", default="full")
    args = ap.parse_args()
    cfg = load_config(args.profile, root=ROOT)
    res = load_saved(args.tag)
    tables = build_tables(res, cfg)
    paths = save_tables(tables, ROOT)
    summary = ["# Results summary (auto-generated from executed runs)\n",
               f"Tag: `{args.tag}`; runs: {len(res.runs)}; seeds per cell: {res.runs['seed'].nunique()}.\n",
               "## H1: accuracy drop of the no-adaptation model\n", narrate_h1(tables["h1_degradation"]), ""]
    (ROOT / "reports" / "results_summary_auto.md").write_text("\n".join(summary))
    print(f"wrote {len(paths)} tables and reports/results_summary_auto.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
