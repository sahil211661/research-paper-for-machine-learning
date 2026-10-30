import numpy as np
import pytest

from src.config import EvalConfig
from src.data_streams import ScenarioSpec
from src.evaluation import (confusion_counts, cumulative_mean, match_alarms, precision_recall_f1,
                            recovery_time, rolling_mean, summarize_run, window_accuracy)

ECFG = EvalConfig(rolling_window=50, pre_window=200, post_window=200, detection_tolerance=100,
                  recovery_fraction=0.95, recovery_hold=50)


def test_rolling_and_cumulative_mean_consistent():
    a = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    r = rolling_mean(a, 3)
    assert r[2] == pytest.approx(2 / 3) and r[3] == pytest.approx(2 / 3) and r[0] == 1.0
    assert cumulative_mean(a)[-1] == pytest.approx(a.mean())
    assert rolling_mean(a, len(a))[-1] == pytest.approx(a.mean())


def test_precision_recall_f1_and_confusion():
    yt = np.array([1, 1, 0, 0, 1]); yp = np.array([1, 0, 0, 1, 1])
    assert confusion_counts(yt, yp) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    m = precision_recall_f1(yt, yp)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_accuracy_equals_one_minus_error():
    yt = np.array([1, 0, 1, 0]); yp = np.array([1, 1, 1, 0])
    err = (yt != yp).astype(float)
    assert window_accuracy(err, 0, 4) == pytest.approx((yt == yp).mean())


def test_alarm_matching_rules():
    # events at 500 (abrupt) and 1500; tolerance 100; n = 2000
    alarms = [100, 520, 530, 700, 1510]
    m = match_alarms(alarms, [500, 1500], [500, 1500], 100, 2000)
    assert [d["delay"] for d in m.detections] == [20, 10]
    assert m.duplicates == 1            # 530 duplicates the detection at 520
    assert m.false_alarms == [100, 700]  # before first onset, and after tolerance window
    assert m.missed == []
    assert m.exposure_outside == 2000 - 200


def test_missed_event_has_no_delay_and_late_alarm_is_false():
    m = match_alarms([650], [500], [500], 100, 1000)
    assert m.detections == [] and m.missed == [0] and m.false_alarms == [650]


def test_gradual_window_includes_transition():
    m = match_alarms([900], [800], [1000], 100, 2000)     # alarm inside transition
    assert m.detections[0]["delay"] == 100 and m.detections[0]["delay_from_end"] == -100


def test_recovery_time_never_recovers_is_censored_not_invented():
    err = np.r_[np.zeros(400), np.ones(600) * 0.5]  # 100% acc then 50%
    err = np.where(np.arange(1000) < 400, 0.0, np.tile([0.0, 1.0], 500)[:1000])
    r = recovery_time(err, 400, 1000, 50, 200, 0.95, 50)
    assert r["recovered"] is False and np.isnan(r["recovery_time"]) and r["recovery_time_capped"] == 600


def test_recovery_time_detects_sustained_recovery():
    err = np.zeros(1000)
    err[400:600] = np.tile([0.0, 1.0], 100)             # degraded for 200 observations
    r = recovery_time(err, 400, 1000, 50, 200, 0.95, 50)
    assert r["recovered"] and 200 <= r["recovery_time"] <= 200 + 50 + 1 and r["degraded"]


def test_no_degradation_gives_floor_recovery_time():
    r = recovery_time(np.zeros(1000), 400, 1000, 50, 200, 0.95, 50)
    assert r["recovered"] and r["recovery_time"] == 49 and not r["degraded"]


def test_summarize_run_stationary_counts_all_alarms_as_false():
    spec = ScenarioSpec("s", "stationary", "none")
    yt = np.zeros(1000, dtype=int); yp = yt.copy()
    s, _ = summarize_run(yt, yp, [100, 300], spec, ECFG)
    assert s["n_false_alarms"] == 2 and s["n_events"] == 0 and np.isnan(s["detection_delay"])
    assert s["false_alarms_per_1000"] == pytest.approx(2.0)
