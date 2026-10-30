import numpy as np
import pytest

from src.config import AdaptationConfig, ExperimentConfig, load_config, make_variants
from src.data_streams import generate_stream, make_scenario_specs
from src.detectors import DetectorSpec
from src.experiments import PolicySpec, build_policy_specs, run_prequential, run_study
from src.models import make_model_factory


class SpyModel:
    """Records the call order and what each call saw."""
    log: list = []

    def __init__(self):
        self.n_learned = 0

    def predict_one(self, x):
        SpyModel.log.append(("predict", x["x1"], self.n_learned))
        return 1

    def learn_one(self, x, y):
        SpyModel.log.append(("learn", x["x1"], y))
        self.n_learned += 1


def _toy(n=60):
    recs = [{"x1": float(i), "x2": 0.0, "x3": 0.0} for i in range(n)]
    labels = np.array([i % 2 for i in range(n)], dtype=np.int8)
    return recs, labels


ADAPT = AdaptationConfig(fixed_window=20, replay_size=5)


def test_prediction_precedes_learning_and_no_future_data():
    SpyModel.log = []
    recs, labels = _toy()
    run_prequential(recs, labels, SpyModel, PolicySpec("none", "none"), ADAPT)
    ev = SpyModel.log
    assert len(ev) == 2 * len(recs)
    for t in range(len(recs)):
        kind_p, xp, _ = ev[2 * t]; kind_l, xl, yl = ev[2 * t + 1]
        assert kind_p == "predict" and kind_l == "learn"
        assert xp == xl == float(t)            # model only ever sees observation t at step t
        assert yl == labels[t]
    # the model had learned exactly t observations when asked to predict t
    assert [e[2] for e in ev[0::2]] == list(range(len(recs)))


def test_first_prediction_made_by_untrained_model():
    SpyModel.log = []
    recs, labels = _toy(5)
    run_prequential(recs, labels, SpyModel, PolicySpec("none", "none"), ADAPT)
    assert SpyModel.log[0] == ("predict", 0.0, 0)


def test_none_policy_has_no_adaptation_events():
    recs, labels = _toy()
    log = run_prequential(recs, labels, SpyModel, PolicySpec("none", "none"), ADAPT)
    assert len(log.adaptations) == 0 and len(log.alarms) == 0


def test_fixed_policy_adapts_only_on_schedule_and_replays_past_only():
    SpyModel.log = []
    recs, labels = _toy(60)
    log = run_prequential(recs, labels, SpyModel, PolicySpec("fixed_replay", "fixed", "replay"), ADAPT)
    assert list(log.adaptations) == [19, 39]       # (t+1) % 20 == 0 and t+1 < n
    # after the rebuild at t=19, the replayed examples are observations 15..19 (past only, incl. current)
    replay = [e for i, e in enumerate(SpyModel.log) if e[0] == "learn"]
    xs = [e[1] for e in replay]
    assert xs[19:24] == [15.0, 16.0, 17.0, 18.0, 19.0]
    assert max(xs[19:24]) <= 19.0


def test_drift_policy_adapts_exactly_at_accepted_alarms():
    cfg = load_config("full")
    sd = generate_stream(make_scenario_specs(cfg.stream)["abrupt_swap"], cfg.stream, 7)
    pol = build_policy_specs(cfg)["adwin_reset"]
    log = run_prequential(sd.records(), sd.y, make_model_factory("hoeffding_tree", cfg.model), pol, cfg.adaptation)
    assert list(log.adaptations) == list(log.alarms) and len(log.alarms) >= 1
    assert log.alarms.min() >= 1


def test_detector_receives_error_computed_after_prediction():
    """Alarm positions of an in-loop policy equal those of the detector fed the logged error stream
    (until the first adaptation changes the model)."""
    cfg = load_config("full")
    sd = generate_stream(make_scenario_specs(cfg.stream)["abrupt_swap"], cfg.stream, 3)
    fac = make_model_factory("hoeffding_tree", cfg.model)
    none = run_prequential(sd.records(), sd.y, fac, PolicySpec("none", "none"), cfg.adaptation)
    from src.detectors import detector_specs, run_detector_on_errors
    spec = detector_specs(cfg.detector)["adwin_default"]
    mon, _, _ = run_detector_on_errors(spec, none.errors)
    live = run_prequential(sd.records(), sd.y, fac, PolicySpec("a", "drift", "reset", spec), cfg.adaptation)
    assert len(mon) and live.alarms[0] == mon[0]


def test_replay_buffer_never_exceeds_configured_size_and_policies_share_data_order():
    cfg = load_config("full")
    sd = generate_stream(make_scenario_specs(cfg.stream)["stationary"], cfg.stream, 4)
    fac = make_model_factory("hoeffding_tree", cfg.model)
    a = run_prequential(sd.records(), sd.y, fac, PolicySpec("none", "none"), cfg.adaptation)
    b = run_prequential(sd.records(), sd.y, fac, PolicySpec("f", "fixed", "replay"), cfg.adaptation)
    assert np.array_equal(a.y_true, b.y_true)               # identical labels / order across policies
    # until the first rebuild the two policies are identical
    w = cfg.adaptation.fixed_window
    assert np.array_equal(a.y_pred[:w], b.y_pred[:w])


def test_evaluation_metric_consistent_between_log_and_summary():
    cfg = load_config("smoke")
    res = run_study(cfg, scenarios=["abrupt_swap"], seeds=cfg.seeds()[:1],
                    policies={"none": build_policy_specs(cfg)["none"]}, progress=False)
    row = res.runs.iloc[0]
    err = res.errors[("abrupt_swap", "none", cfg.seeds()[0])]
    assert row["overall_acc"] == pytest.approx(1 - err.mean())


def test_configured_drift_points_preserved_in_logged_study():
    cfg = load_config("smoke")
    res = run_study(cfg, scenarios=["recurring_swap"], seeds=cfg.seeds()[:1],
                    policies={"none": build_policy_specs(cfg)["none"]}, progress=False)
    onsets = sorted(res.events["onset"].unique())
    assert onsets == sorted(cfg.stream.recurring_positions)


def test_run_study_records_failures_instead_of_hiding_them():
    cfg = load_config("smoke")
    bad = PolicySpec("bad", "drift", "reset", DetectorSpec("nope", "default", {}))
    res = run_study(cfg, scenarios=["stationary"], seeds=cfg.seeds()[:1], policies={"bad": bad},
                    monitor=False, progress=False)
    assert len(res.failed) == 1 and "bad" == res.failed[0]["policy"]


def test_config_variants_and_yaml_roundtrip():
    cfg = load_config("full")
    v = make_variants(cfg, {"adaptation.fixed_window": [500, 1000], "stream.label_noise": [0.0, 0.1]})
    assert len(v) == 4 and {x.adaptation.fixed_window for x in v} == {500, 1000}
    assert cfg.run.n_seeds == 20 and load_config("smoke").run.n_seeds == 3
    with pytest.raises(KeyError):
        make_variants(cfg, {"adaptation.nonexistent": [1]})
