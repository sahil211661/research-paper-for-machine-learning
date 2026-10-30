import numpy as np

from src.config import DetectorConfig
from src.detectors import ADWINDetector, PageHinkleyDetector, detector_specs, make_detector, run_detector_on_errors


def _step(p1, p2, n1=3000, n2=3000, seed=0):
    rng = np.random.default_rng(seed)
    return np.r_[rng.random(n1) < p1, rng.random(n2) < p2].astype(float)


def test_adwin_alarm_on_error_increase_after_change_point():
    det = ADWINDetector(delta=0.002)
    alarms = [t for t, e in enumerate(_step(0.1, 0.4)) if det.update(e)]
    assert alarms and 3000 <= alarms[0] < 3600


def test_adwin_direction_filter_rejects_error_decrease():
    s = _step(0.4, 0.1)
    up, both = ADWINDetector(direction="increase"), ADWINDetector(direction="both")
    a_up = [t for t, e in enumerate(s) if up.update(e)]
    a_both = [t for t, e in enumerate(s) if both.update(e)]
    assert up.n_raw_alarms >= 1 and a_up == [] and len(a_both) >= 1


def test_page_hinkley_up_alarms_on_increase_only():
    inc = PageHinkleyDetector(); dec = PageHinkleyDetector()
    assert [t for t, e in enumerate(_step(0.1, 0.4)) if inc.update(e)]
    # an error DECREASE must not raise 'up' alarms (alarms before t=3000 are ordinary false alarms
    # of the stationary segment and are not what this test is about)
    assert not [t for t, e in enumerate(_step(0.4, 0.1)) if dec.update(e) and t >= 3000]


def test_no_alarm_on_constant_signal():
    for det in (ADWINDetector(), PageHinkleyDetector()):
        assert not any(det.update(0.0) for _ in range(5000))


def test_alarm_signal_logged_correctly_in_monitor_mode():
    specs = detector_specs(DetectorConfig())
    s = _step(0.1, 0.5)
    alarms, n_raw, t_det = run_detector_on_errors(specs["adwin_default"], s)
    det = make_detector(specs["adwin_default"])
    manual = [t for t, e in enumerate(s) if det.update(e)]
    assert list(alarms) == manual and n_raw == det.n_raw_alarms and t_det > 0


def test_sensitivity_levels_ordered_for_false_alarms_on_stationary_noise():
    specs = detector_specs(DetectorConfig())
    counts = {}
    for k in ("adwin", "ph"):
        c = []
        for lvl in ("sensitive", "conservative"):
            tot = 0
            for seed in range(5):
                tot += len(run_detector_on_errors(specs[f"{k}_{lvl}"], _step(0.1, 0.1, 5000, 5000, seed))[0])
            c.append(tot)
        counts[k] = c
    assert counts["adwin"][0] >= counts["adwin"][1] and counts["ph"][0] >= counts["ph"][1]
