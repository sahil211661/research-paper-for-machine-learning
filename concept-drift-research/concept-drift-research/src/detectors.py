"""Drift detector wrappers around River's ADWIN and Page-Hinkley.

Detection signal (all experiments): the binary prequential error indicator
e_t = 1[yhat_t != y_t], available only AFTER the label of observation t arrives.

Alarm semantics (verified empirically, see notebook Section 5 and tests):
* river.drift.ADWIN raises ``drift_detected`` for a significant change of the
  window mean in EITHER direction.  For an error stream only an *increase*
  indicates degradation, so :class:`ADWINDetector` accepts an alarm only if the
  window mean after the cut is larger than before it (``direction='increase'``).
  Rejected (decrease) alarms are counted in ``n_raw_alarms`` but not accepted.
* river.drift.PageHinkley(mode='up') detects increases of the mean natively and
  resets its own statistics after an alarm.

Detector parameters are NOT equivalent: ADWIN's ``delta`` is a confidence
parameter of a statistical test on window means (larger = more sensitive);
PH's ``threshold`` is a bound on a cumulative sum of deviations (lower = more
sensitive) whose scale depends on ``delta`` (tolerated magnitude) and on the
signal's range.  Sensitivity levels are therefore labelled, not numerically
matched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from river import drift

from .config import DetectorConfig


@dataclass(frozen=True)
class DetectorSpec:
    kind: str                 # "adwin" | "ph"
    level: str                # "sensitive" | "default" | "conservative"
    params: Mapping[str, float]
    direction: str = "increase"

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.level}"


def detector_specs(cfg: DetectorConfig) -> dict[str, DetectorSpec]:
    specs: dict[str, DetectorSpec] = {}
    for level, p in cfg.adwin_levels.items():
        specs[f"adwin_{level}"] = DetectorSpec("adwin", level, dict(p), cfg.adwin_direction)
    for level, p in cfg.ph_levels.items():
        params = dict(p)
        params.update(min_instances=cfg.ph_min_instances, delta=cfg.ph_delta, alpha=cfg.ph_alpha)
        specs[f"ph_{level}"] = DetectorSpec("ph", level, params, cfg.ph_mode)
    return specs


class ADWINDetector:
    """ADWIN on the error indicator with a direction filter."""

    def __init__(self, delta: float = 0.002, direction: str = "increase", **_: float) -> None:
        if direction not in ("increase", "both"):
            raise ValueError("direction must be 'increase' or 'both'")
        self._d = drift.ADWIN(delta=delta)
        self.direction = direction
        self.n_raw_alarms = 0
        self.n_accepted = 0
        self.last_raw = False

    def update(self, error: float) -> bool:
        before = self._d.estimation
        self._d.update(error)
        self.last_raw = bool(self._d.drift_detected)
        if not self.last_raw:
            return False
        self.n_raw_alarms += 1
        if self.direction == "both" or self._d.estimation > before:
            self.n_accepted += 1
            return True
        return False


class PageHinkleyDetector:
    """Page-Hinkley (upward mean change) on the error indicator."""

    def __init__(self, threshold: float = 50.0, min_instances: int = 30, delta: float = 0.005,
                 alpha: float = 0.9999, mode: str = "up", **_: float) -> None:
        self._d = drift.PageHinkley(min_instances=min_instances, delta=delta,
                                    threshold=threshold, alpha=alpha, mode=mode)
        self.n_raw_alarms = 0
        self.n_accepted = 0
        self.last_raw = False

    def update(self, error: float) -> bool:
        self._d.update(error)
        self.last_raw = bool(self._d.drift_detected)
        if self.last_raw:
            self.n_raw_alarms += 1
            self.n_accepted += 1
        return self.last_raw


def make_detector(spec: DetectorSpec):
    if spec.kind == "adwin":
        return ADWINDetector(direction=spec.direction, **spec.params)
    if spec.kind == "ph":
        return PageHinkleyDetector(mode=spec.direction if spec.direction in ("up", "down", "both") else "up",
                                   **spec.params)
    raise KeyError(spec.kind)


def run_detector_on_errors(spec: DetectorSpec, errors: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Feed a recorded error stream to a fresh detector (monitor-only mode).

    Returns (accepted alarm positions, raw alarm count, wall-clock seconds spent
    inside ``update`` only)."""
    det = make_detector(spec)
    alarms: list[int] = []
    t_det = 0.0
    pc = time.perf_counter
    for t, e in enumerate(errors):
        t0 = pc()
        acc = det.update(float(e))
        t_det += pc() - t0
        if acc:
            alarms.append(t)
    return np.asarray(alarms, dtype=int), det.n_raw_alarms, t_det
