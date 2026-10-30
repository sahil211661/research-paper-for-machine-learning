"""Notebook cells: Sections 10-17."""
from _nbhelpers import code, md

CELLS = []
A = CELLS.append

A(md(r"""
---
## Section 10 — Baseline experiment (and execution of the full experimental grid)

**What the code does.** Runs the complete grid **once**: every scenario × every seed × every policy (7 primary + 4 detector-sensitivity policies), then replays each scenario's no-adaptation error stream through all six detector settings in *monitor-only* mode (Section 11), measures memory in a separate pass, and saves all raw outputs. Sections 10–14 analyse these same results; nothing is re-tuned after looking at them.

**Pre-registered definitions** (fixed in `configs/default.yaml` before running): rolling window $w=200$; reference window = 1000 observations before the change point; drift window = [onset, end + 1000); recovery = rolling accuracy ≥ 95 % of the pre-drift reference for 200 consecutive observations (sensitivity: 90 % and 98 %, Section 13); detection tolerance = 1000 observations after the end of the transition; first alarm in a window = detection, later alarms in the same window = duplicates (neither detection nor false alarm), alarms outside all windows = false alarms, no alarm = missed (delay undefined).

**Why a stationary baseline.** Rolling accuracy fluctuates even when nothing changes (binomial noise with $\sqrt{p(1-p)/w}\approx 0.02$ for $w=200$, $p\approx0.9$) and detectors can fire on noise. The stationary stream calibrates both: a dip in one window is not evidence of drift.
"""))

A(code(r"""
t0 = time.time()
res = run_study(cfg)
res.memory = measure_memory(cfg, "abrupt_swap" if "abrupt_swap" in cfg.run.scenarios else cfg.run.scenarios[0], build_policy_specs(cfg))
RUNTIME_MAIN_S = time.time() - t0
paths = save_study(res, ROOT, TAG)
tables = build_tables(res, cfg)
table_paths = save_tables(tables, ROOT)
print(f"\nmain study finished in {RUNTIME_MAIN_S/60:.1f} min | runs: {len(res.runs)} | monitor rows: {len(res.monitor_runs)} | failed runs: {len(res.failed)}")
assert len(res.failed) == 0, res.failed
display(res.runs.groupby("scenario", sort=False).agg(runs=("seed", "count"), seeds=("seed", "nunique")))
"""))

A(md(r"""
### Baseline: no-adaptation model
**Figure 3 — stationary baseline.** *Plotted:* rolling (w=200) and cumulative prequential accuracy of the non-adapting Hoeffding tree on the stationary stream, mean ± SD over seeds. *Why:* it shows the learning curve and the size of ordinary fluctuations that any detector must tolerate. *Axes:* x = observation index, y = accuracy. *Limits:* with 5 % label noise the best attainable accuracy is about 0.95; the curve describes this learner and generator only.
"""))

A(code(r"""
fig = viz.fig_stationary_baseline(res, cfg); viz.save_figure(fig, "fig03_stationary_baseline", ROOT); plt.show()
st_none = res.runs[(res.runs.scenario == "stationary") & (res.runs.policy == "none")]
M = np.vstack([rolling_mean(1 - e.astype(float), cfg.eval.rolling_window) for (sc, p, _), e in res.errors.items() if sc == "stationary" and p == "none"])
late = M[:, cfg.stream.length // 4:]
observed(f"on the stationary stream the final cumulative accuracy is {st_none['overall_acc'].mean():.3f} (SD across {len(st_none)} seeds {st_none['overall_acc'].std():.3f}); "
         f"after the first quarter of the stream the rolling accuracy of a single run varies with SD {late.std(axis=1).mean():.3f} around its own mean, "
         f"and individual rolling values range from {late.min():.3f} to {late.max():.3f} although nothing changes.")
"""))

A(md(r"""
**Figure 4 — cumulative accuracy with known drift locations.** *Plotted:* cumulative prequential accuracy of the **no-adaptation** model for every scenario, mean ± SD over seeds, with the known change points (dashed red), transition intervals (shaded) and the covariate-shift point (dotted purple). *Why:* it shows when and how strongly performance bends away from the learning curve. *Axes:* x = observation index, y = accuracy since the start. *Limits:* cumulative accuracy reacts slowly (it averages the whole history), so the *timing* of degradation is read from Figure 5.

**Figure 5 — rolling error rate.** *Plotted:* trailing 200-observation error rate for the same runs. *Why:* local view of degradation and recovery used for the recovery definition. *Axes:* y = error rate in the window (lower is better). *Limits:* the window length trades noise for delay.
"""))

A(code(r"""
fig = viz.fig_accuracy_with_drift(res, cfg, "accuracy"); viz.save_figure(fig, "fig04_accuracy_with_drift", ROOT); plt.show()
fig = viz.fig_accuracy_with_drift(res, cfg, "error"); viz.save_figure(fig, "fig05_rolling_error_with_drift", ROOT); plt.show()
ap = tables["adaptation_performance"]; ap0 = ap[ap["policy"] == "none"].set_index("scenario")
display(ap0[["pre_acc_mean", "pre_acc_sd", "drift_window_acc_mean", "drift_window_acc_sd", "post_acc_mean", "acc_drop_mean", "n_runs"]])
observed("; ".join(f"{sc}: pre-drift {r.pre_acc_mean:.3f} → drift window {r.drift_window_acc_mean:.3f} (mean drop {r.acc_drop_mean:+.3f})" for sc, r in ap0.iterrows()) +
         ". The first two rows labelled stationary and covariate_shift are control streams: their 'change point' is the stream midpoint / the covariate-shift point, not a concept change.")
"""))

A(md(r"""
---
## Section 11 — Drift detector experiments

**What the code does.** The no-adaptation error stream of each (scenario, seed) is fed to **fresh** ADWIN and Page-Hinkley instances at three a-priori sensitivity levels (*monitor-only* mode: one shared prequential error stream, detectors attached to a separate monitoring component). Because the model never changes in this mode, all six detectors see exactly the same signal.

**Detector signal.** Binary error indicator $e_t$ (1 = misclassified), computed after the label arrives.

**Parameters (not equivalent).**
| Detector | Parameter | Meaning | sensitive / **default** / conservative |
|---|---|---|---|
| ADWIN | `delta` | confidence parameter of the sub-window mean test; larger ⇒ alarms on weaker evidence | 0.02 / **0.002** / 0.0002 |
| Page-Hinkley | `threshold` ($\lambda$) | bound on the cumulative deviation statistic; smaller ⇒ alarms sooner | 25 / **50** / 100 |
| Page-Hinkley | `delta`, `alpha`, `min_instances`, `mode` | tolerated drift per step (0.005), forgetting factor (0.9999), warm-up (30), `up` (increases only) | fixed |

Defaults are River's library defaults and were fixed *before* any result was seen; the other two levels bracket them by an order of magnitude. No threshold was tuned on outcomes.

**Alarm-matching rule** (see `src/evaluation.py`): window of event $i$ = [onset$_i$, min(end$_i$ + 1000, onset$_{i+1}$, N)); first alarm in a window ⇒ detection with delay = alarm − onset$_i$ (and alarm − end$_i$, negative if inside the transition); further alarms in the window = duplicates (not counted twice); alarms elsewhere = false alarms; windows without alarm = missed (delay undefined, **never imputed**). False alarms are normalised per 1000 observations outside all windows.
"""))

A(code(r"""
display(tables["detector_performance"].round(3))
display(tables["detection_delay"].round(1))
display(tables["false_alarms_and_missed"].round(3))
"""))

A(md(r"""
**Figures 6 and 7 — detection timelines.** *Plotted:* for one seed (the first), the rolling error rate of the non-adapting model on `abrupt_swap` and the accepted alarms (coloured vertical lines) of ADWIN (Fig. 6) and Page-Hinkley (Fig. 7) at each sensitivity level; the red dashed line is the known drift point. *Why:* it makes alarm timing, duplicates and false alarms visible for a concrete run. *Axes:* x = observation index, y = rolling error. *Limits:* a single seed illustrates mechanics; statements about detector behaviour must rely on the multi-seed tables and figures that follow.
"""))

A(code(r"""
sc_demo = "abrupt_swap" if "abrupt_swap" in cfg.run.scenarios else cfg.run.scenarios[0]
fig = viz.fig_detector_timeline(res, cfg, "adwin", sc_demo); viz.save_figure(fig, "fig06_adwin_timeline", ROOT); plt.show()
fig = viz.fig_detector_timeline(res, cfg, "ph", sc_demo); viz.save_figure(fig, "fig07_pagehinkley_timeline", ROOT); plt.show()
sd0 = cfg.seeds()[0]; onset0 = specs[sc_demo].onsets[0]
al = res.alarm_log[(res.alarm_log.scenario == sc_demo) & (res.alarm_log.seed == sd0) & res.alarm_log.policy.str.startswith("monitor:")]
observed(f"for {sc_demo}, seed {sd0}, drift onset at t={onset0}: " + "; ".join(
    f"{p.replace('monitor:', '')} alarms at {sorted(g['t'].tolist())}" for p, g in al.groupby('policy')) + ".")
"""))

A(md(r"""
**Figure 8 — detector comparison across drift types.** *Plotted:* detection rate (fraction of known events detected) and number of accepted alarms per run, default settings, for each scenario with concept drift; bars = mean, whiskers = 95 % CI over seeds. *Why:* separates *whether* a drift is found from *how noisy* the detector is. *Limits:* the detection rate is bounded by the tolerance window; recurring drift contains three events per run.

**Figure 9 — detection delay.** *Plotted:* box plots of detection delay (observations after onset) for detected events only, for all six detector settings. *Why:* delay is only defined when detection happens, so Figure 8 must be read alongside it. *Limits:* survivorship — settings that miss hard cases can look fast.

**Figure 10 — false-alarm comparison.** *Plotted:* false alarms per 1000 observations outside the detection windows, all scenarios including the stationary and covariate-shift controls. *Why:* the stationary stream isolates pure noise-triggered alarms; the covariate-shift stream shows reactions to a non-concept change. *Limits:* rates are small; confidence intervals matter.
"""))

A(code(r"""
fig = viz.fig_detector_comparison(res, cfg); viz.save_figure(fig, "fig08_detector_comparison", ROOT); plt.show()
fig = viz.fig_detection_delay(res, cfg); viz.save_figure(fig, "fig09_detection_delay", ROOT); plt.show()
fig = viz.fig_false_alarms(res, cfg); viz.save_figure(fig, "fig10_false_alarms", ROOT); plt.show()
m = res.monitor_runs
g = m.groupby(["detector", "level"]).agg(rate=("false_alarms_per_1000", "mean"), delay=("detection_delay", "mean"), det=("detection_recall", "mean")).reset_index()
stat = m[m.scenario == "stationary"].groupby("detector_name")["n_alarms"].mean()
cov = m[m.scenario == "covariate_shift"].groupby("detector_name")["alarms_after_shift_window"].mean() if "covariate_shift" in set(m.scenario) else pd.Series(dtype=float)
observed("alarms per run on the stationary stream (all are false alarms by construction): " + ", ".join(f"{k}: {v:.2f}" for k, v in stat.items()) +
         ("; mean alarms within the tolerance window after the covariate-shift point (a non-concept change): " + ", ".join(f"{k}: {v:.2f}" for k, v in cov.items()) if len(cov) else "") + ".")
"""))

A(md(r"""
---
## Section 12 — Adaptation and retraining experiments

**Policies compared** (identical streams, order and evaluation rules):

| Policy | Trigger | Action on trigger | State kept | Cost drivers |
|---|---|---|---|---|
| `none` | – | – | everything the incremental learner has learned | learner only |
| `fixed_reset` | every $W$ observations | fresh empty model | nothing | $\lfloor N/W\rfloor$ resets regardless of drift |
| `fixed_replay` | every $W$ observations | fresh model trained on last $R$ observations | last $R$ labelled observations | buffer + $R$ updates per rebuild |
| `adwin_reset`, `ph_reset` | accepted detector alarm | fresh empty model; detector re-initialised | nothing | detector update per observation; resets on every (also false) alarm |
| `adwin_replay`, `ph_replay` | accepted detector alarm | fresh model trained on last $R$ observations | last $R$ observations | as above + buffer |

Fixed window $W=2000$ and replay size $R=500$ were set a priori. The replay buffer contains only observations with index ≤ the current one (no look-ahead). For drift-triggered replay the buffer may still contain pre-drift data when the alarm is late — this is part of the policy being evaluated, not a defect.

**Fair-comparison protocol check.** Below, for every (scenario, seed) we verify that each policy's prediction errors are *identical* to the no-adaptation errors up to its first adaptation (same model, same data, same order): if any policy had seen different data or future information, the prefixes would differ.
"""))

A(code(r"""
# Fair-comparison / leakage check: identical prediction-error prefix up to the first adaptation of each policy.
ev = res.alarm_log[res.alarm_log.event == "adaptation"]
first_adapt = ev.groupby(["scenario", "policy", "seed"])["t"].min().to_dict()
bad, n_checked = [], 0
for (sc, pol, seed), err in res.errors.items():
    if pol == "none": continue
    base = res.errors[(sc, "none", seed)]
    k = first_adapt.get((sc, pol, seed), len(err) - 1)       # errors up to and including step k are produced before adaptation takes effect
    n_checked += 1
    if not np.array_equal(err[:k + 1], base[:k + 1]): bad.append((sc, pol, seed))
print(f"checked {n_checked} policy runs; runs with differing pre-adaptation prefix: {len(bad)}")
assert not bad
FAIRNESS_OK = not bad
# adaptation events only at documented trigger conditions
W = cfg.adaptation.fixed_window
fx = ev[ev.policy.str.startswith("fixed")]
assert ((fx["t"] + 1) % W == 0).all(), "fixed-window adaptation off schedule"
dr = ev[~ev.policy.str.startswith("fixed")]
al_ = res.alarm_log[res.alarm_log.event == "alarm"]
merged = dr.merge(al_, on=["scenario", "policy", "seed", "t"], how="left", suffixes=("", "_a"))
assert merged["event_a"].notna().all(), "drift adaptation without an accepted alarm"
TRIGGERS_OK = True
print("fixed-window adaptations only at multiples of W; every drift-triggered adaptation coincides with an accepted alarm: True")
"""))

A(code(r"""
display(tables["adaptation_performance"].round(3))
display(tables["recovery_time"].round(2))
display(tables["computational_cost"].round(3))
"""))

A(md(r"""
**Figure 11 — adaptation-policy performance curves.** *Plotted:* trailing 200-observation accuracy averaged over seeds, one panel per concept-drift scenario, one line per primary policy; known drift points and transitions are marked. *Why:* shows the shape of degradation and recovery for each policy on identical data. *Axes:* x = observation index, y = rolling accuracy (zoomed to 0.55–1). *Limits:* means over seeds hide run-to-run variability (see Figures 12 and 14); lines overlap before the first adaptation by construction.

**Figure 12 — recovery time.** *Plotted:* distribution over seeds of the (event-averaged) time from drift onset until the rolling accuracy has recovered to ≥95 % of its pre-drift reference and stayed there for 200 observations. Runs that never recover are **right-censored** at the horizon (next onset or end of stream) — they are not given an invented recovery time. *Why:* quantifies how long the model stays unreliable. *Axes:* y = observations (minimum attainable ≈ 199 because the rolling window must lie entirely after the onset). *Limits:* the value depends on the threshold and window (sensitivity in Section 13).
"""))

A(code(r"""
fig = viz.fig_policy_curves(res, cfg); viz.save_figure(fig, "fig11_policy_curves", ROOT); plt.show()
fig = viz.fig_recovery_time(res, cfg); viz.save_figure(fig, "fig12_recovery_time", ROOT); plt.show()
rc = tables["recovery_time"]
notrec = rc[rc.recovery_rate_mean < 1][["scenario", "policy", "recovery_rate_mean"]]
observed("fraction of runs in which the model did recover (recovery rate < 1 listed only): " +
         ("; ".join(f"{r.scenario}/{r.policy}: {r.recovery_rate_mean:.2f}" for r in notrec.itertuples()) if len(notrec) else "all runs recovered within their horizon") + ".")
"""))

A(md(r"""
---
## Section 13 — Controlled experiments and sensitivity analyses

**Grid actually executed (Section 10).** 7 scenarios (drift type × magnitude/transition width) × 11 policies (detector choice × sensitivity × action) × `n_seeds` independent seeds; the experiment parameters of every run (scenario, kind, seed, model, trigger, action, detector, level) are stored in `results/raw/<tag>_runs.csv`, together with the configuration snapshot in `results/logs/<tag>_config.json`. Drift **magnitude** is varied by the concept change (`swap` disagreement ≈ 0.30 vs `threshold` ≈ 0.13) and the **transition duration** by the gradual widths. Nothing in this grid was re-tuned after seeing test outcomes: all parameters are from the configuration fixed in advance; the sensitivity levels are reported as sensitivity levels, not as "selected best".

**Additional controlled analyses below:** (a) recovery-definition sensitivity (90 %, 95 %, 98 %), (b) model-effect sub-experiment (Gaussian Naive Bayes vs Hoeffding tree, two scenarios, none vs ADWIN-replay), (c) memory (separate tracemalloc pass).
"""))

A(code(r"""
display(tables["recovery_sensitivity"].round(2).pivot_table(index=["scenario", "policy"], columns="fraction", values="recovery_time_capped_mean").round(0))
rs = tables["recovery_sensitivity"]
tight = rs.pivot_table(index=["scenario", "policy"], columns="fraction", values="recovery_time_capped_mean")
order_ok = (tight[0.90] <= tight[0.95] + 1e-9).mean(), (tight[0.95] <= tight[0.98] + 1e-9).mean()
observed(f"stricter recovery thresholds give longer or equal censored recovery times in {order_ok[0]*100:.0f}% (90→95 %) and {order_ok[1]*100:.0f}% (95→98 %) of scenario/policy cells; "
         "differences between policies that persist across the three thresholds are less likely to be artefacts of the 95 % choice.")
"""))

A(code(r"""
# (b) model-effect sub-experiment: separate runs, same seeds, same streams
nb_seeds = cfg.seeds()[: (10 if PROFILE == "full" else 2)]
nb_scen = [s for s in ("stationary", "abrupt_swap") if s in cfg.run.scenarios]
pols = {k: v for k, v in build_policy_specs(cfg).items() if k in ("none", "adwin_replay")}
t0 = time.time()
res_nb = run_study(cfg, scenarios=nb_scen, policies=pols, seeds=nb_seeds, model_name="gaussian_nb", monitor=False, progress=False)
res_ht = res.runs[res.runs.scenario.isin(nb_scen) & res.runs.policy.isin(pols) & res.runs.seed.isin(nb_seeds)]
mc = pd.concat([res_ht, res_nb.runs]).groupby(["scenario", "model", "policy"]).agg(
    n_runs=("seed", "count"), overall_acc=("overall_acc", "mean"), drift_window_acc=("drift_window_acc", "mean"),
    n_adaptations=("n_adaptations", "mean"), us_per_obs=("us_per_obs", "mean")).reset_index()
display(mc.round(3)); mc.to_csv(ROOT / "results" / "tables" / "model_comparison.csv", index=False)
print(f"model sub-experiment finished in {time.time()-t0:.0f}s; failed: {len(res_nb.failed)}")
"""))

A(code(r"""
display(res.memory.groupby("policy").agg(n_runs=("seed", "count"), peak_kib_mean=("peak_kib", "mean"), peak_kib_sd=("peak_kib", "std")).round(1))
observed("peak traced Python-heap allocation during the prequential loop (scenario %s, %d seeds per policy) ranges from %.0f to %.0f KiB across policies; "
         "tracemalloc ignores memory allocated outside the Python allocator, so the numbers are indicative of relative, not absolute, footprint." %
         (res.memory.scenario.iloc[0], res.memory.seed.nunique(), res.memory.groupby("policy").peak_kib.mean().min(), res.memory.groupby("policy").peak_kib.mean().max()))
"""))

A(md(r"""
---
## Section 14 — Statistical analysis

**Unit of replication.** One independent stream run (one seed). Policies and detector settings are compared on the **same seeds**, so comparisons are *paired*; the per-seed difference is the observation. Thousands of correlated predictions within a run are never used as replicates.

**H1 control.** The learner keeps improving on a stationary stream, so a raw before/after drop understates degradation. The stationary stream of the *same seed* (identical features and noise draws) is evaluated with the *same windows* as each scenario, and the per-seed difference between the scenario's drop and this control drop (difference-in-differences) is tested.

**Reported:** number of runs, mean and SD, median and IQR, 95 % t-intervals (descriptive tables), for paired comparisons the mean difference with a **paired bootstrap** 95 % CI (5000 resamples of seeds), **Cohen's $d_z$** (mean difference / SD of differences) and the share of runs improved, a **Wilcoxon signed-rank** p-value (two-sided; zero differences dropped), and **Holm-adjusted** p-values over each family of comparisons (family = all rows produced by one call to `paired_table`). A Friedman test per scenario serves as an omnibus check across the seven primary policies. **Practical relevance** is flagged with pre-set thresholds (1 percentage point of drift-window accuracy, 0.5 pp of overall accuracy, 100 observations of recovery time).

**Assumptions and limits.** Wilcoxon assumes symmetric differences around the median; bootstrap intervals with 20 seeds are approximate; with the same stream shared by all policies, seeds are blocks. The number of tests is large, so Holm correction matters, and p-values near the threshold should not be over-interpreted. Statistical significance concerns *these* generators, learner and parameters.
"""))

A(code(r"""
display(tables["h1_degradation"].round(4))
st = tables["statistical_comparisons"]
cols = ["hypothesis", "scenario", "metric", "condition", "baseline", "n_pairs", "mean_condition", "mean_baseline", "mean_diff", "ci_low", "ci_high", "cohens_dz", "p_wilcoxon", "p_holm", "significant_holm"]
display(st[(st.hypothesis == "H2") & (st.metric == "drift_window_acc")][cols].round(4))
"""))

A(code(r"""
display(st[(st.hypothesis == "H2") & (st.metric == "recovery_time_capped")][cols].round(3))
display(st[st.hypothesis.isin(["H3", "H3-inloop"])][["hypothesis", "scenario", "detector", "metric", "condition", "baseline", "n_pairs", "mean_diff", "ci_low", "ci_high", "cohens_dz", "p_holm", "significant_holm"]].round(4))
display(tables["friedman_omnibus"].round(4))
observed(narrate_comparisons(st, "H2", "drift_window_acc") + " " + narrate_comparisons(st, "H2", "overall_acc") + " " + narrate_comparisons(st, "H2", "recovery_time_capped", higher_better=False))
"""))

A(md(r"""
---
## Section 15 — Comprehensive visualization (remaining figures and inventory)

Figures 1–12 were produced next to the analysis they support. The remaining required figures are generated here. All figures come from the executed results; none is illustrative.

**Figure 13 — performance vs computational cost.** *Plotted:* per policy, mean overall accuracy (over the concept-drift scenarios) against mean processing time per observation; marker area ∝ mean number of adaptations. *Why:* shows what each policy costs. *Limits:* wall-clock timings on a shared single-core machine are noisy; differences of a few µs are within noise (see per-run SD in `computational_cost.csv`).

**Figure 14 — statistical uncertainty.** *Plotted:* paired difference to no adaptation in drift-window accuracy per scenario and policy with 95 % bootstrap CIs; an asterisk marks Holm-significant comparisons. *Limits:* differences are relative to `none`; they do not rank the other policies against each other.

**Figure 15 — heatmap.** *Plotted:* mean overall accuracy for every policy × scenario cell. *Limits:* means only; uncertainty is in Figures 12 and 14.

**Figure 17 — summary dashboard.** *Plotted:* four panels (drift-window accuracy, delay-vs-false-alarm trade-off, recovery, accuracy-vs-cost). *Limits:* a summary, not a replacement for the tables.
"""))

A(code(r"""
fig = viz.fig_performance_vs_cost(res, cfg); viz.save_figure(fig, "fig13_performance_vs_cost", ROOT); plt.show()
specs_ = make_scenario_specs(cfg.stream)
dsub = res.runs[res.runs.scenario.map(lambda s: specs_[s].has_concept_drift)]
pc = dsub.groupby("policy").agg(acc=("overall_acc", "mean"), us=("us_per_obs", "mean"), ad=("n_adaptations", "mean"))
observed(f"over concept-drift scenarios the highest mean overall accuracy is {pc.acc.idxmax()} ({pc.acc.max():.3f}) and the lowest {pc.acc.idxmin()} ({pc.acc.min():.3f}); "
         f"processing time per observation ranges from {pc.us.min():.1f} µs ({pc.us.idxmin()}) to {pc.us.max():.1f} µs ({pc.us.idxmax()}).")
"""))

A(code(r"""
fig = viz.fig_uncertainty(st, "drift_window_acc"); viz.save_figure(fig, "fig14_statistical_uncertainty", ROOT); plt.show()
h2 = st[(st.hypothesis == "H2") & (st.metric == "drift_window_acc")]
widest = h2.assign(w=h2.ci_high - h2.ci_low).sort_values("w").iloc[-1]; narrow = h2.assign(w=h2.ci_high - h2.ci_low).sort_values("w").iloc[0]
observed(f"{int(h2.significant_holm.sum())} of {len(h2)} policy-vs-none comparisons on drift-window accuracy are Holm-significant; CI widths range from {narrow.w:.3f} ({narrow.scenario}/{narrow.condition}) to {widest.w:.3f} ({widest.scenario}/{widest.condition}).")
"""))

A(code(r"""
fig = viz.fig_heatmap(res, cfg, "overall_acc"); viz.save_figure(fig, "fig15_heatmap_overall_accuracy", ROOT); plt.show()
hm = res.runs.pivot_table(index="policy", columns="scenario", values="overall_acc", aggfunc="mean")
best = hm.idxmax(); observed("best policy per scenario by mean overall accuracy: " + "; ".join(f"{c}: {best[c]} ({hm[c].max():.3f})" for c in hm.columns) + ".")
fig = viz.fig_dashboard(res, cfg, tables); viz.save_figure(fig, "fig17_summary_dashboard", ROOT); plt.show()
"""))

A(code(r"""
# Figure 16 (real-world stream) is produced in Section 17 only if Elec2 is available.  Inventory of generated figure files:
inv = pd.DataFrame([{"file": p.name, "size_kB": round(p.stat().st_size / 1024, 1)} for p in sorted((ROOT / "figures").glob("fig*.png"))])
display(inv)
required_figs = ["fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07", "fig08", "fig09", "fig10", "fig11", "fig12", "fig13", "fig14", "fig15", "fig16", "fig17"]
missing_figs = [f for f in required_figs if not any(p.name.startswith(f) for p in (ROOT / "figures").glob("*.png"))]
print("required figures not generated:", missing_figs)
"""))

A(md(r"""
---
## Section 16 — Results tables

All tables are exported to `results/tables/*.csv` (done in Section 10 via `save_tables`). Each table contains its grouping variables and `n_runs` (sample size per cell); parameters of the runs are in `results/raw/<tag>_runs.csv`. Tables report several metrics deliberately: **no single metric ranks the methods** — accuracy, recovery, false alarms, delay and cost pull in different directions.
"""))

A(code(r"""
exp = pd.DataFrame([{"table": p.name, "rows": len(pd.read_csv(p)), "columns": pd.read_csv(p, nrows=1).shape[1]} for p in sorted((ROOT / "results" / "tables").glob("*.csv"))])
display(exp)
display(tables["overall_model_performance"].round(3).head(14))
display(tables["results_by_drift_type"].round(3).head(12))
"""))

A(md(r"""
---
## Section 17 — Real-world validation (Elec2)

**Protocol (executed only when the dataset is available).** The same Hoeffding tree and the policies `none`, `fixed_replay`, `adwin_replay`, `ph_replay` (default settings) are run once over the chronological stream (the learner and detectors are deterministic, so there is no seed to replicate; results are descriptive and carry **no** significance claims). Because **no ground-truth drift timestamps exist**, no detection delay is computed: we report rolling accuracy, overall accuracy, alarm positions and adaptation counts only, and discuss seasonality, label timing and confounding in Section 19.
"""))

A(code(r"""
REALWORLD_EXECUTED = False
if ELEC2_AVAILABLE:
    recs, y_el = elec2_to_stream(elec_df)
    logs_rw = run_realworld(recs, y_el, cfg)
    rw = pd.DataFrame([{"policy": k, "n": v.n_obs, "overall_acc": 1 - v.errors.mean(), "alarms": len(v.alarms), "adaptations": len(v.adaptations),
                        "us_per_obs": 1e6 * v.timing["total_s"] / v.n_obs} for k, v in logs_rw.items()])
    display(rw.round(4)); rw.to_csv(ROOT / "results" / "tables" / "realworld_elec2.csv", index=False)
    fig = viz.fig_realworld(logs_rw); viz.save_figure(fig, "fig16_realworld_elec2", ROOT); plt.show()
    REALWORLD_EXECUTED = True
else:
    msg = ("Real-world validation NOT executed: Elec2 could not be loaded in this environment (documented source unreachable; no local copy in data/raw/). "
           "The synthetic study is the completed core experiment. To run Stage B, follow the alternative workflow printed in Section 8 and re-execute the notebook.")
    (ROOT / "results" / "logs" / "realworld_status.txt").write_text(msg + "\n\nLoader message:\n" + (ELEC2_ERROR or ""))
    display(Markdown("> **" + msg + "**\n>\n> Consequently Figure 16 and any statement about transfer of the synthetic findings to a real-world stream are **not** produced."))
"""))
