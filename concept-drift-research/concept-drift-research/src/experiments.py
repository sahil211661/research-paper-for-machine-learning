"""Prequential experiment runner, adaptation policies and grid execution.

Per-observation protocol (identical for every policy; see ``run_prequential``)
-----------------------------------------------------------------------------
For t = 0 .. N-1, with (x_t, y_t) the t-th observation of the stream:

 1. yhat_t = model.predict_one(x_t)     -- BEFORE the model has seen y_t.
                                           (A model that has learned nothing returns None;
                                            this is recorded as class 0, the default.)
 2. e_t = 1[yhat_t != y_t]              -- label arrives; log (yhat_t, y_t, e_t).
 3. detector.update(e_t)                -- only for drift-triggered policies.
 4. buffer.append((x_t, y_t))           -- only for ``replay`` actions (already-seen data).
 5. Adaptation decision (using only information up to and including t):
      * drift policy : accepted detector alarm at t
      * fixed policy : (t + 1) is a positive multiple of W and t + 1 < N
    If adapting, the model is REPLACED by a fresh instance which then
        - ``reset``  : learns only (x_t, y_t);
        - ``replay`` : learns the last R buffered observations (including t) in order.
    The detector (drift policies) is re-initialised after adapting.
    Otherwise the model learns (x_t, y_t) incrementally.
 6. Log timing, alarms and adaptation events.

No policy sees x_{t+1}, ... or y_{t+1}, ... before step t is complete.
"""
from __future__ import annotations

import time
import tracemalloc
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import AdaptationConfig, ExperimentConfig
from .data_streams import ScenarioSpec, StreamData, generate_stream, make_scenario_specs
from .detectors import DetectorSpec, detector_specs, make_detector, run_detector_on_errors
from .evaluation import summarize_run
from .models import make_model_factory


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicySpec:
    name: str
    trigger: str                      # none | fixed | drift
    action: str | None = None         # reset | replay
    detector: DetectorSpec | None = None

    def describe(self) -> dict[str, str]:
        return {"policy": self.name, "trigger": self.trigger, "action": self.action or "-",
                "detector": self.detector.kind if self.detector else "-",
                "detector_level": self.detector.level if self.detector else "-"}


def build_policy_specs(cfg: ExperimentConfig) -> dict[str, PolicySpec]:
    dets = detector_specs(cfg.detector)
    p: dict[str, PolicySpec] = {"none": PolicySpec("none", "none")}
    for act in ("reset", "replay"):
        p[f"fixed_{act}"] = PolicySpec(f"fixed_{act}", "fixed", act)
    for kind in ("adwin", "ph"):
        for act in ("reset", "replay"):
            p[f"{kind}_{act}"] = PolicySpec(f"{kind}_{act}", "drift", act, dets[f"{kind}_default"])
    return p


def build_sensitivity_policy_specs(cfg: ExperimentConfig) -> dict[str, PolicySpec]:
    dets = detector_specs(cfg.detector)
    out = {}
    for kind in ("adwin", "ph"):
        for level in ("sensitive", "conservative"):
            nm = f"{kind}_{level}_replay"
            out[nm] = PolicySpec(nm, "drift", "replay", dets[f"{kind}_{level}"])
    return out


PRIMARY_POLICY_ORDER = ["none", "fixed_reset", "fixed_replay", "adwin_reset", "adwin_replay",
                        "ph_reset", "ph_replay"]


# --------------------------------------------------------------------------- #
# Prequential run
# --------------------------------------------------------------------------- #
@dataclass
class RunLog:
    policy: PolicySpec
    y_true: np.ndarray
    y_pred: np.ndarray
    alarms: np.ndarray               # accepted in-loop alarm positions (drift policies)
    adaptations: np.ndarray          # positions at which the model was rebuilt
    n_raw_alarms: int
    timing: dict[str, float]
    n_obs: int

    @property
    def errors(self) -> np.ndarray:
        return (self.y_true != self.y_pred).astype(np.uint8)


def run_prequential(records: Sequence[dict], labels: np.ndarray, model_factory: Callable[[], object],
                    policy: PolicySpec, adaptation: AdaptationConfig) -> RunLog:
    n = len(labels)
    pc = time.perf_counter
    model = model_factory()
    det = make_detector(policy.detector) if policy.trigger == "drift" else None
    buf: deque | None = deque(maxlen=adaptation.replay_size) if policy.action == "replay" else None
    y_pred = np.zeros(n, dtype=np.int8)
    alarms: list[int] = []
    adapts: list[int] = []
    raw_alarms = 0
    t_pred = t_learn = t_det = t_adapt = 0.0
    t_start = pc()
    for t in range(n):
        x = records[t]
        y = int(labels[t])
        a = pc()
        yp = model.predict_one(x)                      # 1. predict BEFORE learning
        b = pc()
        t_pred += b - a
        y_pred[t] = 0 if yp is None else int(yp)
        e = int(y_pred[t] != y)                        # 2. error after the label arrives
        alarm = False
        if det is not None:
            a = pc()
            alarm = det.update(float(e))               # 3. detector update
            t_det += pc() - a
            if alarm:
                alarms.append(t)
        if buf is not None:
            buf.append((x, y))                         # 4. replay buffer (past data only)
        adapt = False
        if policy.trigger == "drift":
            adapt = alarm
        elif policy.trigger == "fixed":
            adapt = ((t + 1) % adaptation.fixed_window == 0) and (t + 1 < n)
        if adapt:                                      # 5. adaptation
            a = pc()
            model = model_factory()
            if policy.action == "replay":
                for xb, yb in buf:                     # type: ignore[union-attr]
                    model.learn_one(xb, yb)
            else:
                model.learn_one(x, y)
            if det is not None:
                raw_alarms += det.n_raw_alarms
                det = make_detector(policy.detector)   # type: ignore[arg-type]
            adapts.append(t)
            t_adapt += pc() - a
        else:
            a = pc()
            model.learn_one(x, y)
            t_learn += pc() - a
    total = pc() - t_start
    if det is not None:
        raw_alarms += det.n_raw_alarms
    return RunLog(policy, np.asarray(labels, dtype=np.int8), y_pred, np.asarray(alarms, dtype=int),
                  np.asarray(adapts, dtype=int), raw_alarms,
                  {"total_s": total, "predict_s": t_pred, "learn_s": t_learn,
                   "detector_s": t_det, "adapt_s": t_adapt}, n)


# --------------------------------------------------------------------------- #
# Study execution
# --------------------------------------------------------------------------- #
@dataclass
class StudyResults:
    runs: pd.DataFrame
    events: pd.DataFrame
    alarm_log: pd.DataFrame
    errors: dict[tuple[str, str, int], np.ndarray]
    monitor_runs: pd.DataFrame
    monitor_events: pd.DataFrame
    memory: pd.DataFrame
    failed: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def _run_row(log: RunLog, spec: ScenarioSpec, seed: int, model_name: str, ecfg) -> tuple[dict, list[dict]]:
    summ, ev = summarize_run(log.y_true, log.y_pred, log.alarms, spec, ecfg)
    row = {"scenario": spec.name, "scenario_kind": spec.kind, "seed": seed, "model": model_name,
           **log.policy.describe(), **summ,
           "n_adaptations": len(log.adaptations), "n_raw_alarms": log.n_raw_alarms,
           "time_total_s": log.timing["total_s"], "time_predict_s": log.timing["predict_s"],
           "time_learn_s": log.timing["learn_s"], "time_detector_s": log.timing["detector_s"],
           "time_adapt_s": log.timing["adapt_s"],
           "us_per_obs": 1e6 * log.timing["total_s"] / log.n_obs}
    if log.policy.trigger == "fixed":
        for k in ("n_alarms", "n_detected", "n_missed", "n_duplicates", "n_false_alarms",
                  "detection_recall", "detection_delay", "detection_delay_from_end",
                  "false_alarms_per_1000", "alarm_precision"):
            row[k] = np.nan
    if log.policy.trigger == "none":
        for k in ("n_alarms", "n_detected", "n_missed", "n_duplicates", "n_false_alarms",
                  "detection_recall", "detection_delay", "detection_delay_from_end",
                  "false_alarms_per_1000", "alarm_precision"):
            row[k] = np.nan
    ev_rows = [{"scenario": spec.name, "seed": seed, "policy": log.policy.name, **r} for r in ev]
    return row, ev_rows


def run_study(cfg: ExperimentConfig, scenarios: Iterable[str] | None = None,
              policies: dict[str, PolicySpec] | None = None, seeds: Sequence[int] | None = None,
              model_name: str | None = None, monitor: bool = True, progress: bool = True) -> StudyResults:
    """Run the full factorial (scenario x seed x policy) plus the monitor-only detector study."""
    specs = make_scenario_specs(cfg.stream)
    scenarios = list(scenarios or cfg.run.scenarios)
    policies = policies or {**build_policy_specs(cfg), **build_sensitivity_policy_specs(cfg)}
    seeds = list(seeds or cfg.seeds())
    model_name = model_name or cfg.model.primary
    factory = make_model_factory(model_name, cfg.model)
    det_specs = detector_specs(cfg.detector)

    run_rows, ev_rows, alarm_rows, errors = [], [], [], {}
    mon_rows, mon_ev = [], []
    failed: list[dict] = []
    t_begin = time.perf_counter()
    for sc in scenarios:
        spec = specs[sc]
        for seed in seeds:
            sd = generate_stream(spec, cfg.stream, seed)
            records, labels = sd.records(), sd.y
            logs: dict[str, RunLog] = {}
            for pname, pol in policies.items():
                try:
                    log = run_prequential(records, labels, factory, pol, cfg.adaptation)
                except Exception as exc:     # never hide failures
                    failed.append({"scenario": sc, "seed": seed, "policy": pname,
                                   "error": f"{type(exc).__name__}: {exc}"})
                    continue
                logs[pname] = log
                row, evs = _run_row(log, spec, seed, model_name, cfg.eval)
                run_rows.append(row)
                ev_rows.extend(evs)
                errors[(sc, pname, seed)] = log.errors
                for a in log.alarms:
                    alarm_rows.append({"scenario": sc, "policy": pname, "seed": seed, "t": int(a), "event": "alarm"})
                for a in log.adaptations:
                    alarm_rows.append({"scenario": sc, "policy": pname, "seed": seed, "t": int(a), "event": "adaptation"})
            if monitor and "none" in logs:
                base = logs["none"]
                err = base.errors
                for dname, dspec in det_specs.items():
                    try:
                        alarms, n_raw, t_det = run_detector_on_errors(dspec, err)
                    except Exception as exc:
                        failed.append({"scenario": sc, "seed": seed, "policy": f"monitor:{dname}",
                                       "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    summ, evs = summarize_run(base.y_true, base.y_pred, alarms, spec, cfg.eval)
                    mon_rows.append({"scenario": sc, "scenario_kind": spec.kind, "seed": seed,
                                     "detector": dspec.kind, "level": dspec.level, "detector_name": dname,
                                     **{k: v for k, v in summ.items() if k in (
                                         "n_alarms", "n_events", "n_detected", "n_missed", "n_duplicates",
                                         "n_false_alarms", "detection_recall", "detection_delay",
                                         "detection_delay_from_end", "false_alarms_per_1000",
                                         "alarm_precision", "alarms_after_shift_window")},
                                     "n_raw_alarms": n_raw, "time_detector_s": t_det,
                                     "us_per_update": 1e6 * t_det / len(err)})
                    for a in alarms:
                        alarm_rows.append({"scenario": sc, "policy": f"monitor:{dname}", "seed": seed,
                                           "t": int(a), "event": "alarm"})
        if progress:
            print(f"[{time.perf_counter() - t_begin:7.1f}s] finished scenario {sc} "
                  f"({len(seeds)} seeds x {len(policies)} policies)")
    return StudyResults(pd.DataFrame(run_rows), pd.DataFrame(ev_rows), pd.DataFrame(alarm_rows),
                        errors, pd.DataFrame(mon_rows), pd.DataFrame(mon_ev), pd.DataFrame(),
                        failed, cfg.to_dict())


def measure_memory(cfg: ExperimentConfig, scenario: str, policies: dict[str, PolicySpec],
                   n_runs: int | None = None) -> pd.DataFrame:
    """Peak Python-heap allocation (tracemalloc) during ``run_prequential`` only.

    Measured in a separate pass (tracemalloc slows execution, so timing and memory are never
    measured together).  Scope: Python allocations traced by tracemalloc; native allocations
    made inside compiled extensions are not necessarily captured."""
    spec = make_scenario_specs(cfg.stream)[scenario]
    factory = make_model_factory(cfg.model.primary, cfg.model)
    rows = []
    for seed in cfg.seeds()[: (n_runs or cfg.run.memory_runs)]:
        sd = generate_stream(spec, cfg.stream, seed)
        records, labels = sd.records(), sd.y
        for pname, pol in policies.items():
            tracemalloc.start()
            run_prequential(records, labels, factory, pol, cfg.adaptation)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"scenario": scenario, "seed": seed, "policy": pname, "peak_kib": peak / 1024.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Real-world (unknown change points) runner
# --------------------------------------------------------------------------- #
def run_realworld(records: Sequence[dict], labels: np.ndarray, cfg: ExperimentConfig,
                  policies: dict[str, PolicySpec] | None = None) -> dict[str, RunLog]:
    policies = policies or {k: v for k, v in build_policy_specs(cfg).items()
                            if k in ("none", "fixed_replay", "adwin_replay", "ph_replay")}
    factory = make_model_factory(cfg.model.primary, cfg.model)
    return {name: run_prequential(records, labels, factory, pol, cfg.adaptation)
            for name, pol in policies.items()}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_study(res: StudyResults, root: Path, tag: str = "main") -> dict[str, Path]:
    raw = root / "results" / "raw"
    logs = root / "results" / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": raw / f"{tag}_runs.csv", "events": raw / f"{tag}_drift_events.csv",
        "monitor": raw / f"{tag}_monitor_runs.csv", "alarms": logs / f"{tag}_event_log.csv",
        "memory": raw / f"{tag}_memory.csv",
    }
    res.runs.to_csv(paths["runs"], index=False)
    res.events.to_csv(paths["events"], index=False)
    res.monitor_runs.to_csv(paths["monitor"], index=False)
    res.alarm_log.to_csv(paths["alarms"], index=False)
    res.memory.to_csv(paths["memory"], index=False)
    keys = list(res.errors)
    np.savez_compressed(logs / f"{tag}_prequential_errors.npz",
                        keys=np.array(["|".join(map(str, k)) for k in keys]),
                        errors=np.array([res.errors[k] for k in keys], dtype=np.uint8))
    pd.DataFrame(res.failed).to_csv(logs / f"{tag}_failed_runs.csv", index=False)
    import json
    (logs / f"{tag}_config.json").write_text(json.dumps(res.config, indent=2, default=str))
    return paths
