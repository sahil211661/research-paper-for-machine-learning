#!/usr/bin/env python
"""Run the full experimental study outside the notebook.

Usage:
    python scripts/run_experiments.py --profile smoke --tag smoke   # ~1 minute pipeline check ONLY
    python scripts/run_experiments.py --profile full  --tag main    # full study (~10-15 min on 1 CPU)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                                     # noqa: E402
from src.experiments import (build_policy_specs, measure_memory, run_study, save_study)  # noqa: E402
from src.reporting import build_tables, save_tables                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="full", choices=["full", "smoke"])
    ap.add_argument("--tag", default=None, help="file tag for raw outputs (default: profile name)")
    ap.add_argument("--n-seeds", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.profile, root=ROOT)
    if args.n_seeds:
        from dataclasses import replace
        cfg = replace(cfg, run=replace(cfg.run, n_seeds=args.n_seeds))
    tag = args.tag or args.profile
    t0 = time.time()
    print(f"profile={cfg.run.profile} seeds={cfg.run.n_seeds} stream_length={cfg.stream.length}")
    res = run_study(cfg)
    prim = build_policy_specs(cfg)
    scen = "abrupt_swap" if "abrupt_swap" in cfg.run.scenarios else cfg.run.scenarios[0]
    import pandas as pd
    res.memory = measure_memory(cfg, scen, prim)
    paths = save_study(res, ROOT, tag)
    tables = build_tables(res, cfg)
    save_tables(tables, ROOT)
    print("failed runs:", len(res.failed))
    for k, p in paths.items():
        print(f"  saved {k}: {p.relative_to(ROOT)}")
    print(f"done in {time.time() - t0:.1f}s; {len(tables)} tables written to results/tables/")
    return 0 if not res.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
