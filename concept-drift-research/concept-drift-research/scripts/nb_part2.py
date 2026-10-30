"""Notebook cells: Sections 5-9."""
from _nbhelpers import code, md

CELLS = []
A = CELLS.append

A(md(r"""
---
## Section 5 — Environment setup and dependency validation

**What the code does.** Records interpreter and library versions, verifies that every required import works, and runs a lightweight compatibility check of each API the study relies on (stream generation, `predict_one`/`learn_one`, detector `update`/`drift_detected`, alarm direction semantics, metric calculation).
**Why.** Method names and defaults differ across River versions; the study must use the *installed, documented* API rather than assumed names.
**Output.** A version table, a pass/fail compatibility table, and `results/logs/environment.txt`.

Dependencies are pinned in `requirements.txt`. If a package is missing, run `pip install -r requirements.txt` once in a terminal (the notebook does not reinstall anything automatically).
"""))

A(code(r"""
import os, sys, json, time, platform, warnings, inspect, subprocess, importlib
from pathlib import Path

# Robust project-root detection: independent of the notebook's working directory.
_here = Path.cwd().resolve()
ROOT = next(p for p in [_here, *_here.parents] if (p / "pyproject.toml").exists() and (p / "src").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
matplotlib.rcParams["figure.dpi"] = 100          # inline size; files are saved at 300 DPI

def observed(text: str) -> None:
    # Display result-derived text produced by code (never typed in advance).
    display(Markdown("**Observed pattern (auto-generated from the executed results):** " + text))

print("project root:", ROOT)
"""))

A(code(r"""
import importlib.metadata as md
required = ["river", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn", "statsmodels", "PyYAML",
            "ipython", "pytest", "nbformat", "psutil"]
rows = []
for p in required:
    try:
        rows.append({"package": p, "version": md.version(p), "import_ok": True})
    except md.PackageNotFoundError:
        rows.append({"package": p, "version": "MISSING", "import_ok": False})
env = pd.DataFrame(rows)
display(env)
assert env["import_ok"].all(), "Missing packages: run `pip install -r requirements.txt` in a terminal."
PYTHON_VERSION = platform.python_version()
print("Python", PYTHON_VERSION, "|", platform.platform(), "| CPU cores:", os.cpu_count())
(ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "logs" / "environment.txt").write_text(
    f"python {PYTHON_VERSION}\nplatform {platform.platform()}\n" + "\n".join(f"{r.package}=={r.version}" for r in env.itertuples()))
"""))

A(code(r"""
import src
from src.config import load_config, make_variants, find_project_root
from src.data_streams import *
from src.models import make_model_factory, check_online_api
from src.detectors import *
from src.evaluation import *
from src.experiments import *
from src.reporting import *
from src.statistical_analysis import *
from src import visualization as viz

from river import drift, tree, datasets, naive_bayes

# ---- lightweight compatibility checks of every API used in the study ----
checks = []
def check(name, fn):
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    checks.append({"check": name, "passed": bool(ok), "detail": detail})

def _gen():
    cfg_ = load_config("smoke", root=ROOT)
    sd = generate_stream(make_scenario_specs(cfg_.stream)["abrupt_swap"], cfg_.stream, 1)
    return len(sd) == cfg_.stream.length, f"X{sd.X.shape}, y mean={sd.y.mean():.2f}"
def _river_sea():
    rows = list(datasets.synth.SEA(variant=0, seed=1).take(5)); return len(rows) == 5, f"keys={list(rows[0][0])}"
def _model():
    m = tree.HoeffdingTreeClassifier(); check_online_api(m)
    first = m.predict_one({"x1": 1.0, "x2": 2.0, "x3": 3.0}); m.learn_one({"x1": 1.0, "x2": 2.0, "x3": 3.0}, 1)
    return True, f"untrained predict_one -> {first!r} (recorded as class 0); learn_one OK"
def _adwin():
    a = drift.ADWIN(); rng = np.random.default_rng(0)
    s = np.r_[rng.random(2000) < .1, rng.random(2000) < .4].astype(float); up = dn = 0
    for v in s:
        before = a.estimation; a.update(v)
        if a.drift_detected: up += a.estimation > before; dn += a.estimation <= before
    return a.drift_detected in (True, False), f"alarms with error increase={up}, decrease={dn} (attribute drift_detected exists)"
def _adwin_dir():
    s = np.r_[np.random.default_rng(1).random(3000) < .4, np.random.default_rng(2).random(3000) < .1].astype(float)
    raw = ADWINDetector(direction="both"); flt = ADWINDetector(direction="increase")
    for v in s: raw.update(v); flt.update(v)
    return raw.n_raw_alarms >= 1 and flt.n_accepted < raw.n_accepted + 1, \
        f"on an error DECREASE: raw ADWIN alarms={raw.n_raw_alarms}; accepted by 'increase' filter={flt.n_accepted}"
def _ph():
    out = {}
    for mode in ("up", "down", "both"):
        d = drift.PageHinkley(mode=mode); s = np.r_[np.random.default_rng(0).random(3000) < .1, np.random.default_rng(3).random(3000) < .4].astype(float)
        out[mode] = sum(1 for v in s if (d.update(v) or d.drift_detected))
    return out["up"] >= 1 and out["down"] == 0, f"alarms on an error INCREASE by mode: {out}"
def _ph_reset():
    d = drift.PageHinkley(mode="up"); s = np.r_[np.zeros(500), np.ones(1500)]
    n = sum(1 for v in s if (d.update(v) or d.drift_detected)); return n == 1, f"alarms on a single 0->1 step: {n} (detector resets itself after an alarm)"
def _metrics():
    yt, yp = np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1]); m = precision_recall_f1(yt, yp)
    return abs(m["recall"] - 2 / 3) < 1e-9 and window_accuracy((yt != yp).astype(float), 0, 4) == 0.75, f"precision={m['precision']:.2f} recall={m['recall']:.2f}"
def _spec():
    s = detector_specs(load_config("smoke", root=ROOT).detector); return len(s) == 6, f"{sorted(s)}"
for n, f in [("synthetic generator (src.data_streams)", _gen), ("River SEA generator", _river_sea),
             ("Hoeffding tree predict_one/learn_one", _model), ("ADWIN update/drift_detected/estimation", _adwin),
             ("ADWIN direction semantics (decrease is NOT degradation)", _adwin_dir),
             ("Page-Hinkley modes (up detects increases only)", _ph), ("Page-Hinkley self-reset after alarm", _ph_reset),
             ("metric calculations", _metrics), ("detector specifications", _spec)]:
    check(n, f)
compat = pd.DataFrame(checks); display(compat)
assert compat["passed"].all(), "API compatibility problem: see table above"
"""))

A(md(r"""
**Reading the table.** ADWIN raises an alarm for a significant change of the window mean in *either* direction, so on an error stream a *decrease* (the model improving) also triggers it. The study therefore uses a documented direction filter (`ADWINDetector(direction="increase")`): an alarm is accepted only if the window mean after the cut is larger than before it. Rejected alarms are still counted (`n_raw_alarms`) so the effect of the filter is visible. Page-Hinkley is run in `mode="up"`, which only detects increases of the mean.
"""))

A(md(r"""
---
## Section 6 — Configuration and reproducibility

**What the code does.** Loads `configs/default.yaml` into typed dataclasses (`src/config.py`). Every experimental constant — seeds, stream length, drift positions, transition widths, model and detector parameters, fixed-window size, evaluation windows, repetitions, output directories — is a named configuration value. `make_variants` creates configuration variants (used here only to demonstrate the helper; sensitivity analyses are run through the same mechanism).
**Seeds.** The stream seed is `base_seed + i`. All scenarios with the same seed share the same random draws (common random numbers), so differences between scenarios and between policies are not confounded by sampling noise. Models and detectors are deterministic given the data.
**Profiles.** `full` (the study) and `smoke` (a quick pipeline check; never reported as a result). Select via the environment variable `DRIFT_PROFILE` before starting Jupyter.
"""))

A(code(r"""
PROFILE = os.environ.get("DRIFT_PROFILE", "full")
TAG = "main" if PROFILE == "full" else PROFILE
cfg = load_config(PROFILE, root=ROOT)
print(f"PROFILE={PROFILE!r}  seeds={cfg.run.n_seeds}  stream length={cfg.stream.length}  base seed={cfg.run.base_seed}")
if PROFILE != "full":
    display(Markdown("> **WARNING: smoke profile — numbers produced in this run are only a pipeline check and must not be reported.**"))
print(json.dumps({k: v for k, v in cfg.to_dict().items() if k != "paths"}, indent=1, default=str)[:3500])

variants = make_variants(cfg, {"adaptation.fixed_window": [cfg.adaptation.fixed_window // 2, cfg.adaptation.fixed_window],
                               "stream.label_noise": [0.0, cfg.stream.label_noise]})
print("\nconfiguration variants generated by make_variants:", [(v.adaptation.fixed_window, v.stream.label_noise) for v in variants])
"""))

A(md(r"""
**Reproducibility checklist** (executable parts are checked in Section 22)

- [x] Fixed seeds (`run.base_seed + i`), recorded in every result row.
- [x] All constants in one YAML file; the full configuration is written to `results/logs/<tag>_config.json` for every run.
- [x] Versions of Python and all libraries are recorded (Section 5) and pinned in `requirements.txt`.
- [x] Identical data order and identical stream objects for all policies within a (scenario, seed).
- [x] No hidden state: all cells run top-to-bottom after a kernel restart.
- [x] Deterministic models and detectors (same seed ⇒ identical predictions; verified below).
- [x] Data are generated or obtained from a documented source; no data are fabricated.
"""))

A(code(r"""
# Determinism check: the same seed must give identical streams and identical prequential predictions.
specs = make_scenario_specs(cfg.stream)
factory = make_model_factory(cfg.model.primary, cfg.model)
pol_none = build_policy_specs(cfg)["none"]
sd1 = generate_stream(specs["abrupt_swap"], cfg.stream, cfg.seeds()[0]); sd2 = generate_stream(specs["abrupt_swap"], cfg.stream, cfg.seeds()[0])
r1 = run_prequential(sd1.records(), sd1.y, factory, pol_none, cfg.adaptation); r2 = run_prequential(sd2.records(), sd2.y, factory, pol_none, cfg.adaptation)
DETERMINISM_OK = bool(np.array_equal(sd1.X, sd2.X) and np.array_equal(r1.y_pred, r2.y_pred))
print("identical streams and identical predictions for the same seed:", DETERMINISM_OK)
assert DETERMINISM_OK
"""))

A(md(r"""
---
## Section 7 — Synthetic data generation

**What the code does.** Generates SEA-style streams (Street & Kim [9]): three features $x_1,x_2,x_3\sim U[0,10]$; concept A labels an instance positive if $x_1+x_2>\theta$ with $\theta=8$ (identical to River's `SEA(variant=0)`, verified by a unit test); each label is flipped with probability 0.05.

**Scenarios and the mechanism that changes in each**

| Scenario | Mechanism | Changes $P(X)$? | Changes $P(Y\mid X)$ (concept drift)? | Changes $P(Y)$? |
|---|---|---|---|---|
| `stationary` | none (no hidden change point) | no | no | no |
| `covariate_shift` | $x_1\sim U[3,10]$ after the change point, same labelling rule | **yes** | no | yes (side effect) |
| `abrupt_swap` | concept A → B: label rule $x_2+x_3>\theta$ at the change point | no | **yes** | no (by symmetry) |
| `abrupt_threshold` | concept A → B′: $\theta$ 8 → 9.5 | no | **yes** | yes (side effect) |
| `gradual_swap_w*` | A → B (swap) through a linear mixture ramp of width $w$: for $t$ in the transition, each example comes from B with probability $(t-\text{start}+0.5)/w$, otherwise from A | no | **yes** | no |
| `recurring_swap` | A→B→A→B at three change points | no | **yes** (4 segments) | no |

The `swap` concept B uses the *irrelevant* feature $x_3$ and ignores $x_1$, which changes P(Y|X) strongly (disagreement with A ≈ 0.30, estimated below) while leaving the feature distribution and the class prevalence unchanged. The `covariate_shift` scenario is a **control**: if a detector fires there it reacts to a change that is not concept drift.

**What the visualisations show and do not show.** They show *that the generator produces the designed change at the designed place*. They do not show that any learner is affected — that is tested in Section 10.
"""))

A(code(r"""
spec_table = pd.DataFrame([{"scenario": s.name, "kind": s.kind, "concept_B": s.concept_b, "onsets": s.onsets, "ends": s.ends,
                            "covariate_shift_points": s.shift_points, "description": s.description} for s in specs.values()])
display(spec_table[["scenario", "kind", "concept_B", "onsets", "ends", "covariate_shift_points"]])

# ---- validation: the generator follows its intended configuration (all scenarios x first 5 seeds) ----
val_rows = []
for sc in cfg.run.scenarios:
    for sd_ in cfg.seeds()[:5]:
        checks_ = validate_stream(generate_stream(specs[sc], cfg.stream, sd_), cfg.stream)
        bools = {k: v for k, v in checks_.items() if isinstance(v, bool)}
        val_rows.append({"scenario": sc, "seed": sd_, "n_checks": len(bools), "all_passed": all(bools.values())})
val = pd.DataFrame(val_rows); display(val.groupby("scenario").agg(runs=("seed", "count"), checks_per_run=("n_checks", "first"), all_passed=("all_passed", "all")))
assert val["all_passed"].all()
GENERATOR_VALID = bool(val["all_passed"].all())

# magnitude and type of each change (Monte-Carlo disagreement of the labelling rules, and empirical shifts)
mag = pd.DataFrame({"concept change": ["swap (A→B)", "threshold (A→B′)"],
                    "P(label_A ≠ label_B)": [concept_disagreement_rate(cfg.stream, "swap"), concept_disagreement_rate(cfg.stream, "threshold")]})
display(mag)
demo = {n: generate_stream(specs[n], cfg.stream, cfg.seeds()[0]) for n in cfg.run.scenarios}
dsum = []
for n in ("covariate_shift", "abrupt_swap", "abrupt_threshold"):
    pt = (specs[n].onsets or specs[n].shift_points)[0]
    dsum.append({"scenario": n, **distribution_summary(demo[n], pt, 1000)})
display(pd.DataFrame(dsum).set_index("scenario"))
"""))

A(code(r"""
viz.set_style()
fig = viz.fig_stream_overview({n: demo[n] for n in ("stationary", "abrupt_swap", "gradual_swap_w%d" % cfg.stream.gradual_widths[0], "recurring_swap")}, cfg)
viz.save_figure(fig, "fig01_stream_overview", ROOT); plt.show()
"""))

A(md(r"""
**Figure 1 — what is plotted / why / axes / limits.** *Plotted:* the rolling share of concept-B examples (blue) and the rolling positive-class proportion (orange) for four generated streams, with the known drift points marked. *Why it matters:* it verifies that the stationary stream has no hidden change, that the abrupt change is a step, that the gradual change is a ramp over the stated interval and that the recurring change alternates. *Axes:* x = observation index; y = proportion in a 200-observation trailing window. *Limits:* it documents the generator only, not the learner's behaviour; the positive-class proportion is flat for `swap` drifts by construction.
"""))

A(code(r"""
observed("; ".join(
    f"{n}: share of concept-B examples in the last 200 observations = {demo[n].concept[-200:].mean():.2f}, in the first 200 = {demo[n].concept[:200].mean():.2f}"
    for n in ("stationary", "abrupt_swap", "gradual_swap_w%d" % cfg.stream.gradual_widths[0], "recurring_swap")) + ".")
fig = viz.fig_distributions(demo["abrupt_swap"], demo["covariate_shift"], cfg)
viz.save_figure(fig, "fig02_distributions_before_after", ROOT); plt.show()
"""))

A(md(r"""
**Figure 2 — what is plotted / why / axes / limits.** *Plotted:* histograms of $x_1$ and $x_3$ before and after the change point, and the empirical $P(Y=1\mid x_1+x_2)$ before and after, for the concept-swap stream (top row) and the covariate-shift control (bottom row). *Why it matters:* it separates a change of $P(X)$ from a change of $P(Y\mid X)$: for `abrupt_swap` the feature histograms coincide but the conditional curve moves; for `covariate_shift` the histogram of $x_1$ moves but the conditional curve stays. *Axes:* densities (left, middle); empirical positive rate per bin of $x_1+x_2$ (right). *Limits:* bins with <10 observations are omitted; the plot is a descriptive check, not a hypothesis test.
"""))

A(code(r"""
sw, cv = dsum[1], dsum[0]
observed(f"for abrupt_swap the largest KS statistic on the feature marginals is {sw['max KS stat on P(X) marginals']:.3f} (no feature-distribution change) while "
         f"{sw['label disagreement with concept A rule after (noise-free labels)']:.2f} of noise-free labels disagree with the old rule; for covariate_shift the largest KS statistic is "
         f"{cv['max KS stat on P(X) marginals']:.3f} while label disagreement with the old rule is {cv['label disagreement with concept A rule after (noise-free labels)']:.2f}.")
"""))

A(md(r"""
---
## Section 8 — Dataset inspection and preprocessing (Elec2)

**Documented source.** `river.datasets.Elec2` (https://riverml.xyz/dev/api/datasets/Elec2/). The installed River docstring describes it as electricity prices in New South Wales with a binary target (price goes UP or DOWN), 45,312 observations and 8 features (`date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer`) plus the label; River's documentation cites *SPLICE-2 comparative evaluation: Electricity pricing* [13].
**Loader behaviour.** `load_elec2` tries (1) `data/raw/electricity.csv`, (2) River's dataset API (download), and otherwise raises `DatasetUnavailableError` with an alternative workflow. **It never substitutes another dataset and never fabricates data.**
**Planned preprocessing.** None beyond type conversion: the original chronological order is kept (no shuffling), `UP` is the positive class, no scaling or encoding is fitted (the source file is already min-max normalised by its publishers over the *whole* period, which is itself a possible (mild) look-ahead; see Section 19).
"""))

A(code(r"""
ELEC2_AVAILABLE, ELEC2_ERROR = False, None
try:
    elec_df, elec_src = load_elec2(ROOT)
    ELEC2_AVAILABLE = True
    print("Elec2 loaded from:", elec_src)
except DatasetUnavailableError as exc:
    ELEC2_ERROR = str(exc)
    print("ELEC2 NOT AVAILABLE IN THIS EXECUTION ENVIRONMENT\n" + "-" * 60 + "\n" + ELEC2_ERROR[:1800])
"""))

A(code(r"""
if ELEC2_AVAILABLE:
    print("rows, columns:", elec_df.shape, "(expected rows:", ELEC2_EXPECTED_ROWS, ")")
    display(elec_df.dtypes.to_frame("dtype").T)
    print("missing values per column:\n", elec_df.isna().sum().to_dict())
    print("duplicate rows:", int(elec_df.duplicated().sum()))
    print("class distribution:\n", elec_df["class"].value_counts(normalize=True).round(4).to_dict())
    print("'date' non-decreasing in file order:", bool(elec_df["date"].is_monotonic_increasing))
    display(elec_df.describe().T)
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.5))
    for ax, c in zip(axes.ravel(), [c for c in elec_df.columns if c != "class"]):
        ax.hist(elec_df[c], bins=40, color="#0072b2"); ax.set_title(c, fontsize=8)
    viz.save_figure(fig, "figS_elec2_feature_distributions", ROOT); plt.show()
else:
    display(Markdown("**Section 8 could not be executed here** (dataset unreachable). Nothing is displayed because nothing was loaded; no placeholder statistics are shown. "
                     "To enable Sections 8 and 17: place `electricity.csv` in `data/raw/` as described in the error message above and re-run the notebook."))
"""))

A(md(r"""
---
## Section 9 — Prequential evaluation framework

**What the code does.** `src/experiments.py::run_prequential` implements one protocol for every policy. For each observation $t$:

1. $\hat y_t$ = `model.predict_one(x_t)` — **before** the model has seen $y_t$ (an untrained model returns `None`, recorded as class 0).
2. Log $\hat y_t, y_t, e_t=\mathbb 1[\hat y_t\ne y_t]$ (the error exists only after the label arrives).
3. Detector update with $e_t$ (drift-triggered policies only).
4. Append $(x_t,y_t)$ to the replay buffer (replay actions only; buffer holds past, already-labelled data).
5. Adaptation decision using only information up to $t$: *drift policy* — accepted alarm at $t$; *fixed policy* — $(t+1)$ is a multiple of $W$ (and $t+1<N$). On adaptation the model is replaced by a fresh one that either learns only $(x_t,y_t)$ (**reset**) or the last $R$ buffered observations including $t$ (**replay**); the detector is re-initialised. Otherwise the model learns $(x_t,y_t)$.
6. Log timing, alarms and adaptation positions.

**Information that is lost.** A *reset* loses everything learned before $t$ (the new model starts empty). *Replay* keeps only the last $R$ labelled observations; whatever lies further back is lost, and the replay window may still contain pre-drift data when the alarm is late. **Memory/time cost.** Replay needs a buffer of $R$ observations and $R$ learning steps per rebuild.

**Fairness.** All policies in a (scenario, seed) receive the same stream object in the same order; no policy sees $x_{t+1},y_{t+1}$ before step $t$ completes. This is verified by unit tests (a spy model records the call order) and by the identical-prefix check in Section 12.
"""))

A(code(r"""
print(inspect.getsource(run_prequential))
"""))

A(code(r"""
# Run the unit-test suite (toy inputs only; no downloads).  The tests verify the protocol rules listed above.
proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
print(proc.stdout[-1500:] + proc.stderr[-500:])
TESTS_PASSED = proc.returncode == 0
print("unit tests passed:", TESTS_PASSED)
assert TESTS_PASSED
"""))
