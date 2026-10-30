"""Streaming evaluation metrics, alarm matching and recovery-time definitions.

Conventions (fixed BEFORE running the main experiments)
-------------------------------------------------------
* Positions are 0-based observation indices.  An alarm "at position t" is raised
  after the label of observation t arrived (it can only influence t+1 onward).
* ``onset`` o_i: first observation generated (even partly) by the new concept.
  ``end`` e_i: first observation after the transition (e_i == o_i for abrupt).
* Detection window of event i: [o_i, min(e_i + tolerance, o_{i+1}, N)).
  The FIRST alarm in the window is the detection; detection delay = alarm - o_i
  (also reported relative to e_i).  Further alarms in the same window are
  *duplicates*: not detections, not false alarms.  Alarms outside every window
  (including alarms before the first onset) are false alarms.  An event with no
  alarm in its window is *missed*; its delay is NaN (never imputed).
* Recovery time (per event): with acc_t = 1 - error_t, reference
  ``ref`` = mean accuracy on [o - pre_window, o); threshold = frac * ref
  (frac = 0.95).  Using rolling windows of length w that lie entirely at/after
  the onset, recovery is the end t* of the first window from which the rolling
  accuracy stays >= threshold for ``hold`` consecutive positions; recovery time
  = t* - o  (so the minimum attainable value is w - 1).  Search horizon: until
  the next onset or the end of the stream.  If never recovered, the event is
  right-censored: ``recovered=False`` and the capped time = horizon - o.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EvalConfig
from .data_streams import ScenarioSpec


# --------------------------------------------------------------------------- #
# Basic predictive metrics
# --------------------------------------------------------------------------- #
def rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing rolling mean; the first w-1 values use the available (shorter) window."""
    a = np.asarray(a, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(1, len(a) + 1)
    lo = np.maximum(0, idx - w)
    return (cs[idx] - cs[lo]) / (idx - lo)


def cumulative_mean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.cumsum(a) / np.arange(1, len(a) + 1)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> dict[str, int]:
    yt = np.asarray(y_true) == positive
    yp = np.asarray(y_pred) == positive
    return {"tp": int((yt & yp).sum()), "fp": int((~yt & yp).sum()),
            "fn": int((yt & ~yp).sum()), "tn": int((~yt & ~yp).sum())}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> dict[str, float]:
    c = confusion_counts(y_true, y_pred, positive)
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else np.nan
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else np.nan
    f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan
    return {"precision": p, "recall": r, "f1": f1}


def window_accuracy(err: np.ndarray, lo: int, hi: int) -> float:
    lo, hi = max(0, lo), min(len(err), hi)
    return float(1.0 - err[lo:hi].mean()) if hi > lo else float("nan")


# --------------------------------------------------------------------------- #
# Alarm matching
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    detections: list[dict] = field(default_factory=list)   # one per detected event
    missed: list[int] = field(default_factory=list)        # event indices
    duplicates: int = 0
    false_alarms: list[int] = field(default_factory=list)  # alarm positions
    windows: list[tuple[int, int]] = field(default_factory=list)
    n_alarms: int = 0
    exposure_outside: int = 0                               # observations outside windows


def detection_windows(onsets, ends, tolerance: int, n: int) -> list[tuple[int, int]]:
    wins = []
    for i, (o, e) in enumerate(zip(onsets, ends)):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else n
        wins.append((int(o), int(min(e + tolerance, nxt, n))))
    return wins


def match_alarms(alarms, onsets, ends, tolerance: int, n: int) -> MatchResult:
    alarms = np.sort(np.asarray(alarms, dtype=int))
    res = MatchResult(n_alarms=len(alarms))
    res.windows = detection_windows(onsets, ends, tolerance, n)
    used = np.zeros(len(alarms), dtype=bool)
    for i, ((lo, hi), o, e) in enumerate(zip(res.windows, onsets, ends)):
        inside = np.where((alarms >= lo) & (alarms < hi))[0]
        if len(inside) == 0:
            res.missed.append(i)
            continue
        first = int(alarms[inside[0]])
        res.detections.append({"event": i, "onset": int(o), "end": int(e), "alarm": first,
                               "delay": first - int(o), "delay_from_end": first - int(e)})
        res.duplicates += len(inside) - 1
        used[inside] = True
    res.false_alarms = [int(a) for a in alarms[~used]]
    res.exposure_outside = int(n - sum(hi - lo for lo, hi in res.windows))
    return res


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
def recovery_time(err: np.ndarray, onset: int, horizon_end: int, w: int,
                  ref_window: int, frac: float, hold: int) -> dict[str, float | bool]:
    acc = 1.0 - np.asarray(err, dtype=float)
    out: dict[str, float | bool] = {"ref_acc": np.nan, "threshold": np.nan, "recovered": False,
                                    "recovery_time": np.nan, "recovery_time_capped": np.nan,
                                    "degraded": False}
    if onset < min(w, ref_window) or onset >= len(acc):
        return out  # not enough pre-drift data for a reference
    ref = float(acc[max(0, onset - ref_window):onset].mean())
    thr = frac * ref
    out.update(ref_acc=ref, threshold=thr)
    horizon_end = min(horizon_end, len(acc))
    cs = np.concatenate([[0.0], np.cumsum(acc)])
    idx = np.arange(onset + w - 1, horizon_end)
    out["recovery_time_capped"] = float(horizon_end - onset)
    if len(idx) < hold:
        return out
    ra = (cs[idx + 1] - cs[idx + 1 - w]) / w
    ok = ra >= thr
    out["degraded"] = bool((~ok).any())
    run = np.convolve(ok.astype(int), np.ones(hold, dtype=int), mode="valid") == hold
    hits = np.where(run)[0]
    if len(hits):
        t_star = int(idx[hits[0]])
        out.update(recovered=True, recovery_time=float(t_star - onset),
                   recovery_time_capped=float(t_star - onset))
    return out


# --------------------------------------------------------------------------- #
# Run-level summary
# --------------------------------------------------------------------------- #
def evaluation_points(spec: ScenarioSpec, n: int) -> tuple[list[int], list[int], str]:
    """Reference points for accuracy-window metrics.  Concept-drift scenarios use their
    onsets/ends; the covariate-shift control uses its shift point; the stationary
    stream uses the stream midpoint as a pseudo change point (negative control)."""
    if spec.has_concept_drift:
        return list(spec.onsets), list(spec.ends), "concept_drift"
    if spec.shift_points:
        return list(spec.shift_points), list(spec.shift_points), "covariate_shift"
    return [n // 2], [n // 2], "pseudo_midpoint"


def event_metrics(err: np.ndarray, spec: ScenarioSpec, ecfg: EvalConfig,
                  recovery_fractions: tuple[float, ...] | None = None) -> list[dict]:
    n = len(err)
    pts, ends, _ = evaluation_points(spec, n)
    rows = []
    fracs = recovery_fractions or (ecfg.recovery_fraction,)
    for i, (o, e) in enumerate(zip(pts, ends)):
        nxt = pts[i + 1] if i + 1 < len(pts) else n
        row = {
            "event": i, "onset": o, "end": e,
            "pre_acc": window_accuracy(err, o - ecfg.pre_window, o),
            "during_acc": window_accuracy(err, o, e) if e > o else np.nan,
            "post_acc": window_accuracy(err, e, min(e + ecfg.post_window, nxt)),
            "drift_window_acc": window_accuracy(err, o, min(e + ecfg.post_window, nxt)),
        }
        row["acc_drop"] = row["pre_acc"] - row["drift_window_acc"]
        for fr in fracs:
            rec = recovery_time(err, o, nxt, ecfg.rolling_window, ecfg.pre_window, fr, ecfg.recovery_hold)
            suffix = "" if len(fracs) == 1 else f"_f{fr:.2f}"
            row[f"recovery_time{suffix}"] = rec["recovery_time"]
            row[f"recovery_time_capped{suffix}"] = rec["recovery_time_capped"]
            row[f"recovered{suffix}"] = rec["recovered"]
            row[f"degraded{suffix}"] = rec["degraded"]
        rows.append(row)
    return rows


def summarize_run(y_true: np.ndarray, y_pred: np.ndarray, alarms, spec: ScenarioSpec,
                  ecfg: EvalConfig) -> tuple[dict, list[dict]]:
    """Return (run-level metrics dict, per-event rows).  Run-level event metrics are
    means over the scenario's events (NaN-aware)."""
    n = len(y_true)
    err = (np.asarray(y_true) != np.asarray(y_pred)).astype(float)
    s: dict[str, float] = {
        "n_obs": n,
        "overall_acc": float(1.0 - err.mean()),
        "overall_err": float(err.mean()),
    }
    s.update(precision_recall_f1(y_true, y_pred, ecfg.positive_class))
    ev = event_metrics(err, spec, ecfg)
    for key in ("pre_acc", "during_acc", "post_acc", "drift_window_acc", "acc_drop",
                "recovery_time", "recovery_time_capped"):
        vals = np.array([r[key] for r in ev], dtype=float)
        s[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    s["recovery_rate"] = float(np.mean([r["recovered"] for r in ev]))
    s["degraded_rate"] = float(np.mean([r["degraded"] for r in ev]))

    alarms = np.asarray(alarms, dtype=int)
    m = match_alarms(alarms, spec.onsets, spec.ends, ecfg.detection_tolerance, n)
    n_ev = len(spec.onsets)
    delays = np.array([d["delay"] for d in m.detections], dtype=float)
    delays_end = np.array([d["delay_from_end"] for d in m.detections], dtype=float)
    s.update({
        "n_alarms": m.n_alarms,
        "n_events": n_ev,
        "n_detected": len(m.detections),
        "n_missed": len(m.missed),
        "n_duplicates": m.duplicates,
        "n_false_alarms": len(m.false_alarms),
        "detection_recall": (len(m.detections) / n_ev) if n_ev else np.nan,
        "detection_delay": float(delays.mean()) if len(delays) else np.nan,
        "detection_delay_from_end": float(delays_end.mean()) if len(delays_end) else np.nan,
        "false_alarms_per_1000": 1000.0 * len(m.false_alarms) / max(1, m.exposure_outside),
        "alarm_precision": (len(m.detections) / (len(m.detections) + len(m.false_alarms))
                            if (len(m.detections) + len(m.false_alarms)) else np.nan),
    })
    if spec.shift_points:
        lo = spec.shift_points[0]
        s["alarms_after_shift_window"] = int(((alarms >= lo) & (alarms < lo + ecfg.detection_tolerance)).sum())
    else:
        s["alarms_after_shift_window"] = np.nan
    rows = []
    det_by_event = {d["event"]: d for d in m.detections}
    for r in ev:
        d = det_by_event.get(r["event"])
        if spec.has_concept_drift:
            r["detected"] = d is not None
            r["delay"] = d["delay"] if d else np.nan
            r["alarm"] = d["alarm"] if d else np.nan
        rows.append(r)
    return s, rows
