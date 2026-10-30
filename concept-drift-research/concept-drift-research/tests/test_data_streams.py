import numpy as np
import pandas as pd
import pytest

from src.config import StreamConfig
from src.data_streams import (DatasetUnavailableError, concept_disagreement_rate, elec2_to_stream,
                              generate_stream, load_elec2, make_scenario_specs, validate_stream, ELEC2_COLUMNS)

CFG = StreamConfig(length=6000, abrupt_position=3000, gradual_center=3000, gradual_widths=(600, 1500),
                   recurring_positions=(1500, 3000, 4500), covariate_position=3000)
SPECS = make_scenario_specs(CFG)


@pytest.mark.parametrize("name", list(SPECS))
def test_generator_follows_declared_configuration(name):
    sd = generate_stream(SPECS[name], CFG, seed=11)
    checks = validate_stream(sd, CFG)
    bools = {k: v for k, v in checks.items() if isinstance(v, bool)}
    assert all(bools.values()), bools


def test_stationary_has_no_hidden_change_point():
    sd = generate_stream(SPECS["stationary"], CFG, seed=3)
    assert sd.concept.sum() == 0 and SPECS["stationary"].onsets == () and SPECS["stationary"].shift_points == ()
    assert np.array_equal(sd.y_clean, ((sd.X[:, 0] + sd.X[:, 1]) > CFG.theta_a).astype(np.int8))


def test_abrupt_drift_point_is_preserved_and_swap_changes_only_label_function():
    sd = generate_stream(SPECS["abrupt_swap"], CFG, seed=3)
    o = SPECS["abrupt_swap"].onsets[0]
    assert sd.concept[o - 1] == 0 and sd.concept[o] == 1
    st = generate_stream(SPECS["stationary"], CFG, seed=3)
    assert np.array_equal(sd.X, st.X)                 # same P(X) draws (common random numbers)
    assert np.array_equal(sd.y[:o], st.y[:o])          # identical before the drift


def test_gradual_transition_is_a_monotone_mixture():
    sd = generate_stream(SPECS["gradual_swap_w1500"], CFG, seed=5)
    s, e = SPECS["gradual_swap_w1500"].onsets[0], SPECS["gradual_swap_w1500"].ends[0]
    assert e - s == 1500
    thirds = [sd.concept[s + i * 500: s + (i + 1) * 500].mean() for i in range(3)]
    assert thirds[0] < thirds[1] < thirds[2]
    assert sd.concept[:s].sum() == 0 and sd.concept[e:].all()


def test_covariate_shift_changes_px_not_label_rule():
    sd = generate_stream(SPECS["covariate_shift"], CFG, seed=2)
    p = SPECS["covariate_shift"].shift_points[0]
    assert sd.X[p:, 0].min() >= CFG.covariate_x1_low_after
    assert np.array_equal(sd.y_clean, ((sd.X[:, 0] + sd.X[:, 1]) > CFG.theta_a).astype(np.int8))


def test_concept_swap_preserves_prevalence_threshold_does_not():
    assert concept_disagreement_rate(CFG, "swap") > 0.2
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (200000, 3))
    pa = ((X[:, 0] + X[:, 1]) > CFG.theta_a).mean()
    pswap = ((X[:, 1] + X[:, 2]) > CFG.theta_a).mean()
    pthr = ((X[:, 0] + X[:, 1]) > CFG.theta_threshold_shift).mean()
    assert abs(pa - pswap) < 0.01 and abs(pa - pthr) > 0.05


def test_same_seed_reproducible_and_different_seed_differs():
    a = generate_stream(SPECS["abrupt_swap"], CFG, 1); b = generate_stream(SPECS["abrupt_swap"], CFG, 1)
    c = generate_stream(SPECS["abrupt_swap"], CFG, 2)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y) and not np.array_equal(a.X, c.X)


def test_matches_river_sea_variant0():
    from river import datasets
    ds = list(datasets.synth.SEA(variant=0, seed=1).take(500))
    X = np.array([[x[0], x[1]] for x, _ in ds]); y = np.array([int(v) for _, v in ds])
    assert np.array_equal(((X[:, 0] + X[:, 1]) > 8).astype(int), y)


def test_elec2_loader_reports_clear_error_when_unavailable(tmp_path):
    with pytest.raises(DatasetUnavailableError) as ei:
        load_elec2(tmp_path, allow_download=False)
    assert "data/raw/electricity.csv" in str(ei.value)


def test_elec2_local_file_schema_and_chronology(tmp_path):
    # TOY FIXTURE with the Elec2 schema -- used only to test the code path, never as data.
    (tmp_path / "data" / "raw").mkdir(parents=True)
    df = pd.DataFrame({c: np.arange(20, dtype=float) for c in ELEC2_COLUMNS if c != "class"})
    df["class"] = ["UP", "DOWN"] * 10
    df.to_csv(tmp_path / "data" / "raw" / "electricity.csv", index=False)
    out, prov = load_elec2(tmp_path, allow_download=False)
    recs, y = elec2_to_stream(out)
    assert len(recs) == 20 and list(y[:4]) == [1, 0, 1, 0] and recs[3]["date"] == 3.0   # order preserved
    assert "local file" in prov
