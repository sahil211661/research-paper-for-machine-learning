"""Build exportable result tables and result-driven text from executed experiments.

All sentences returned by the ``narrate_*`` helpers are assembled from the computed
tables; none of them contains a pre-determined conclusion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data_streams import make_scenario_specs
from .evaluation import event_metrics
from .experiments import PRIMARY_POLICY_ORDER, StudyResults
from .statistical_analysis import describe, friedman, holm, paired_bootstrap_ci, paired_table, cohens_dz, wilcoxon_p

DRIFT_SCENARIOS_ORDER = ["abrupt_swap", "abrupt_threshold", "gradual_swap_w1000",
                         "gradual_swap_w2500", "recurring_swap"]


def _wide(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_sd"] = (m, "std")
    out = df.groupby(by, sort=False).agg(**agg)
    out["n_runs"] = df.groupby(by, sort=False).size()
    return out.reset_index()


def _order(df: pd.DataFrame, scenarios: list[str], policies: list[str], pcol: str = "policy") -> pd.DataFrame:
    df = df.copy()
    df["scenario"] = pd.Categorical(df["scenario"], [s for s in scenarios if s in set(df["scenario"])], ordered=True)
    if pcol in df:
        order = [p for p in policies if p in set(df[pcol])] + [p for p in df[pcol].unique() if p not in policies]
        df[pcol] = pd.Categorical(df[pcol], order, ordered=True)
    return df.sort_values(["scenario", pcol] if pcol in df else ["scenario"]).reset_index(drop=True)


def h1_table(res: StudyResults, cfg: ExperimentConfig) -> pd.DataFrame:
    """H1: per-seed degradation = pre-drift accuracy - accuracy in the drift window, NO-adaptation policy.

    Two estimates are reported:
    * ``mean_drop``: raw paired drop (negative values mean accuracy ROSE, e.g. continued learning).
    * ``did_*``: difference-in-differences against the stationary stream of the SAME seed, evaluated with the
      SAME windows as the scenario (so learning-curve effects cancel; common random numbers make the pairing exact).
    """
    specs = make_scenario_specs(cfg.stream)
    base = res.runs[res.runs["policy"] == "none"]
    rows = []
    for sc, g in base.groupby("scenario", sort=False):
        g = g.sort_values("seed")
        drop = (g["pre_acc"] - g["drift_window_acc"]).to_numpy()
        ctrl = []
        for seed in g["seed"]:
            ev = event_metrics(res.errors[("stationary", "none", int(seed))].astype(float), specs[sc], cfg.eval)
            ctrl.append(np.nanmean([e["acc_drop"] for e in ev]))
        ctrl = np.asarray(ctrl)
        did = drop - ctrl
        lo, hi = paired_bootstrap_ci(drop, cfg.run.n_bootstrap, cfg.run.alpha)
        dlo, dhi = paired_bootstrap_ci(did, cfg.run.n_bootstrap, cfg.run.alpha)
        rows.append({"scenario": sc, "n_runs": len(drop), "mean_pre_acc": g["pre_acc"].mean(),
                     "mean_drift_window_acc": g["drift_window_acc"].mean(), "mean_drop": drop.mean(),
                     "ci_low": lo, "ci_high": hi, "cohens_dz": cohens_dz(drop), "p_wilcoxon": wilcoxon_p(drop),
                     "control_drop_same_windows": ctrl.mean(), "did_mean": did.mean(), "did_ci_low": dlo,
                     "did_ci_high": dhi, "did_dz": cohens_dz(did),
                     "did_p_wilcoxon": wilcoxon_p(did) if sc != "stationary" else np.nan})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_wilcoxon"].to_numpy())
    out["significant_holm"] = out["p_holm"] < cfg.run.alpha
    out["did_p_holm"] = holm(out["did_p_wilcoxon"].to_numpy())
    out["did_significant_holm"] = out["did_p_holm"] < cfg.run.alpha
    out["reference_type"] = out["scenario"].map(lambda s_: specs[s_].kind)
    return out


def recovery_sensitivity(res: StudyResults, cfg: ExperimentConfig, policies=("none", "adwin_replay", "ph_replay", "fixed_replay")) -> pd.DataFrame:
    specs = make_scenario_specs(cfg.stream)
    fr = cfg.eval.recovery_fraction_sensitivity
    rows = []
    for (sc, pol, seed), err in res.errors.items():
        if pol not in policies or not specs[sc].has_concept_drift:
            continue
        ev = event_metrics(err.astype(float), specs[sc], cfg.eval, recovery_fractions=fr)
        for f in fr:
            sfx = f"_f{f:.2f}"
            rows.append({"scenario": sc, "policy": pol, "seed": seed, "fraction": f,
                         "recovery_time_capped": np.mean([e[f"recovery_time_capped{sfx}"] for e in ev]),
                         "recovered": np.mean([e[f"recovered{sfx}"] for e in ev])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["scenario", "policy", "fraction"], sort=False)
              .agg(recovery_time_capped_mean=("recovery_time_capped", "mean"),
                   recovery_rate=("recovered", "mean"), n_runs=("seed", "count")).reset_index())


def build_tables(res: StudyResults, cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    runs = res.runs
    prim = runs[runs["policy"].isin(PRIMARY_POLICY_ORDER)]
    scen = list(cfg.run.scenarios)
    T: dict[str, pd.DataFrame] = {}

    T["overall_model_performance"] = _order(_wide(
        prim, ["scenario", "policy"], ["overall_acc", "overall_err", "precision", "recall", "f1"]), scen, PRIMARY_POLICY_ORDER)

    mon = res.monitor_runs
    mon_order = [f"{k}_{l}" for k in ("adwin", "ph") for l in ("sensitive", "default", "conservative")]
    T["detector_performance"] = _order(_wide(
        mon, ["scenario", "detector_name"],
        ["n_alarms", "n_raw_alarms", "n_detected", "detection_recall", "alarm_precision"]), scen, mon_order, "detector_name")

    det_rows = []
    for (sc, dn), g in mon[mon["n_events"] > 0].groupby(["scenario", "detector_name"], sort=False):
        d = describe(g["detection_delay"].to_numpy())
        de = describe(g["detection_delay_from_end"].to_numpy())
        det_rows.append({"scenario": sc, "detector_name": dn, "n_runs": len(g),
                         "runs_with_detection": int(g["detection_delay"].notna().sum()),
                         "delay_mean": d["mean"], "delay_sd": d["sd"], "delay_median": d["median"],
                         "delay_q25": d["q25"], "delay_q75": d["q75"], "delay_from_end_mean": de["mean"]})
    T["detection_delay"] = _order(pd.DataFrame(det_rows), scen, mon_order, "detector_name")

    T["false_alarms_and_missed"] = _order(_wide(
        mon, ["scenario", "detector_name"],
        ["n_false_alarms", "false_alarms_per_1000", "n_missed", "n_duplicates", "alarms_after_shift_window"]),
        scen, mon_order, "detector_name")

    T["adaptation_performance"] = _order(_wide(
        prim, ["scenario", "policy"],
        ["pre_acc", "drift_window_acc", "post_acc", "acc_drop", "overall_acc"]), scen, PRIMARY_POLICY_ORDER)

    rec = _wide(prim, ["scenario", "policy"], ["recovery_time_capped", "recovery_rate", "degraded_rate"])
    med = prim.groupby(["scenario", "policy"], sort=False)["recovery_time_capped"].median().rename("recovery_time_capped_median").reset_index()
    T["recovery_time"] = _order(rec.merge(med, on=["scenario", "policy"]), scen, PRIMARY_POLICY_ORDER)

    cost = _wide(runs, ["scenario", "policy"], ["us_per_obs", "time_total_s", "time_detector_s",
                                                "time_adapt_s", "n_adaptations"])
    if len(res.memory):
        mem = res.memory.groupby("policy", sort=False)["peak_kib"].mean().rename("peak_kib_mean").reset_index()
        cost = cost.merge(mem, on="policy", how="left")
    allp = PRIMARY_POLICY_ORDER + [p for p in runs["policy"].unique() if p not in PRIMARY_POLICY_ORDER]
    T["computational_cost"] = _order(cost, scen, allp)

    kind_runs = prim.copy()
    T["results_by_drift_type"] = _wide(kind_runs, ["scenario_kind", "policy"],
                                       ["overall_acc", "drift_window_acc", "recovery_time_capped"])
    T["results_by_seed"] = runs[["scenario", "policy", "seed", "overall_acc", "pre_acc", "drift_window_acc",
                                 "post_acc", "recovery_time_capped", "recovery_rate", "n_alarms",
                                 "n_false_alarms", "detection_delay", "n_adaptations", "us_per_obs"]].copy()

    # ---------------- statistical comparisons ----------------
    pols = [p for p in PRIMARY_POLICY_ORDER if p != "none"]
    st = []
    for metric, thr in (("drift_window_acc", 0.01), ("overall_acc", 0.005), ("recovery_time_capped", 100.0)):
        t = paired_table(prim, ["scenario"], "policy", "none", pols, metric, n_boot=cfg.run.n_bootstrap,
                         alpha=cfg.run.alpha, practical_threshold=thr)
        t.insert(0, "hypothesis", "H2")
        st.append(t)
    # H3: detector sensitivity, monitor-only
    for metric in ("false_alarms_per_1000", "detection_delay", "us_per_update"):
        for kind in ("adwin", "ph"):
            sub = mon[mon["detector"] == kind]
            if metric == "detection_delay":
                sub = sub[sub["n_events"] > 0]
            t = paired_table(sub, ["scenario", "detector"], "level", "conservative",
                             ["default", "sensitive"], metric, n_boot=cfg.run.n_bootstrap, alpha=cfg.run.alpha)
            if len(t):
                t.insert(0, "hypothesis", "H3")
                st.append(t)
    # H3 in-loop compute: adaptation count of sensitivity levels (replay policies)
    for kind in ("adwin", "ph"):
        names = [f"{kind}_{l}_replay" for l in ("conservative", "default", "sensitive")]
        names = [n if n != f"{kind}_default_replay" else f"{kind}_replay" for n in names]
        sub = runs[runs["policy"].isin(names)]
        for metric in ("n_adaptations", "us_per_obs"):
            t = paired_table(sub, ["scenario"], "policy", names[0], names[1:], metric,
                             n_boot=cfg.run.n_bootstrap, alpha=cfg.run.alpha)
            if len(t):
                t.insert(0, "hypothesis", "H3-inloop")
                st.append(t)
    T["statistical_comparisons"] = pd.concat([s for s in st if len(s)], ignore_index=True)
    T["h1_degradation"] = h1_table(res, cfg)
    T["recovery_sensitivity"] = recovery_sensitivity(res, cfg)
    fr = []
    for sc, g in prim.groupby("scenario", sort=False):
        for metric in ("drift_window_acc", "overall_acc"):
            f = friedman(g, "policy", metric, PRIMARY_POLICY_ORDER)
            fr.append({"scenario": sc, "metric": metric, **f})
    T["friedman_omnibus"] = pd.DataFrame(fr)
    return T


def save_tables(tables: dict[str, pd.DataFrame], root: Path) -> list[Path]:
    out = root / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Result-driven narrative helpers (numbers come from the tables)
# --------------------------------------------------------------------------- #
def narrate_ranking(table: pd.DataFrame, scenario: str, mean_col: str, label: str,
                    higher_better: bool = True, key: str = "policy", k: int = 3, fmt: str = "{:.3f}") -> str:
    t = table[table["scenario"] == scenario].dropna(subset=[mean_col])
    if t.empty:
        return f"No data for {scenario}."
    t = t.sort_values(mean_col, ascending=not higher_better)
    parts = [f"{r[key]} ({fmt.format(r[mean_col])})" for _, r in t.head(k).iterrows()]
    worst = t.iloc[-1]
    direction = "higher is better" if higher_better else "lower is better"
    return (f"{scenario}, {label} ({direction}): best -> " + ", ".join(parts) +
            f"; worst: {worst[key]} ({fmt.format(worst[mean_col])}).")


def narrate_h1(h1: pd.DataFrame) -> str:
    lines = []
    for _, r in h1.iterrows():
        line = (f"- {r['scenario']} ({r['reference_type']}): raw mean accuracy drop {r['mean_drop']:+.3f} "
                f"(95% bootstrap CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}], Holm p={r['p_holm']:.3g}, n={int(r['n_runs'])})")
        if r["scenario"] != "stationary":
            line += (f"; relative to the stationary stream evaluated on the same windows (control drop {r['control_drop_same_windows']:+.3f}) "
                     f"the excess drop is {r['did_mean']:+.3f} (CI [{r['did_ci_low']:+.3f}, {r['did_ci_high']:+.3f}], d_z={r['did_dz']:.2f}, Holm p={r['did_p_holm']:.3g})")
        lines.append(line + ".")
    return "\n".join(lines)


def narrate_comparisons(tab: pd.DataFrame, hypothesis: str, metric: str, higher_better: bool = True) -> str:
    t = tab[(tab["hypothesis"] == hypothesis) & (tab["metric"] == metric)]
    if t.empty:
        return "No comparisons available."
    sig = t[t["significant_holm"]]
    better = sig[(sig["mean_diff"] > 0) == higher_better]
    worse = sig[(sig["mean_diff"] > 0) != higher_better]
    return (f"{len(t)} paired comparisons on `{metric}`; {len(better)} significantly better and "
            f"{len(worse)} significantly worse than the baseline after Holm correction; "
            f"{len(t) - len(sig)} not significant.")


# --------------------------------------------------------------------------- #
# Research-question discussion (assembled from executed results only)
# --------------------------------------------------------------------------- #
def _f(x, spec=".3f"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def rq1_text(res: StudyResults, T: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> str:
    h1 = T["h1_degradation"]
    out = ["**Observation: degradation of the non-adapting model (H1).** Paired per-seed drop = "
           "pre-drift accuracy minus accuracy in the drift window "
           f"(n = {int(h1['n_runs'].min())} runs per scenario):", narrate_h1(h1), ""]
    mon = res.monitor_runs
    d = mon[(mon["level"] == "default") & (mon["n_events"] > 0)]
    g = d.groupby(["scenario", "detector"]).agg(
        recall=("detection_recall", "mean"), delay=("detection_delay", "mean"),
        delay_end=("detection_delay_from_end", "mean"), fa=("n_false_alarms", "mean")).reset_index()
    out.append("**Detection under default settings** (monitor-only; delay in observations after drift onset, "
               "undefined for missed events):")
    for _, r in g.iterrows():
        out.append(f"- {r['scenario']} / {r['detector'].upper()}: detection rate {_f(r['recall'], '.2f')}, "
                   f"mean delay from onset {_f(r['delay'], '.0f')} (from transition end {_f(r['delay_end'], '.0f')}), "
                   f"mean false alarms per run {_f(r['fa'], '.2f')}.")
    ab = g[g["scenario"].str.startswith("abrupt")]["delay"].mean()
    gr = g[g["scenario"].str.startswith("gradual")]["delay"].mean()
    out.append("")
    out.append(f"Averaged over detectors, mean delay from onset is {_f(ab, '.0f')} for abrupt and {_f(gr, '.0f')} for gradual "
               "scenarios; part of the gradual figure reflects that the new concept is only partially present early in the "
               "transition, so delay measured from onset is not directly comparable with abrupt delay.")
    n_det = d.groupby(["scenario", "detector"])["detection_recall"].mean()
    low = n_det[n_det < 0.8]
    out.append("")
    out.append("**Caution on delays:** delay exists only for detected events, so it is conditional on detection (survivorship); settings with low detection rates "
               "(" + (", ".join(f"{a}/{b.upper()} {v:.2f}" for (a, b), v in low.items()) if len(low) else "none below 0.80") + ") are summarised by few events and should be read together with the detection rate.")
    out.append("The negative raw 'drop' on control streams means accuracy *rose* after the midpoint/shift because the learner keeps improving; the excess drop relative to the stationary control (above) is the more appropriate estimate of drift-attributable degradation.")
    return "\n".join(out)


def rq2_text(res: StudyResults, T: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> str:
    ap, st = T["adaptation_performance"], T["statistical_comparisons"]
    h2 = st[st["hypothesis"] == "H2"]
    rc = T["recovery_time"]
    out = []
    specs = make_scenario_specs(cfg.stream)
    for sc in cfg.run.scenarios:
        if not specs[sc].has_concept_drift:
            continue
        out.append("- " + narrate_ranking(ap, sc, "drift_window_acc_mean", "mean drift-window accuracy", True, k=3))
        a = h2[(h2["scenario"] == sc) & (h2["metric"] == "drift_window_acc")]
        better = a[(a["significant_holm"]) & (a["mean_diff"] > 0)]["condition"].tolist()
        worse = a[(a["significant_holm"]) & (a["mean_diff"] < 0)]["condition"].tolist()
        out.append(f"  Significantly better than no adaptation (Holm p<{cfg.run.alpha}): {', '.join(better) or 'none'}; "
                   f"significantly worse: {', '.join(worse) or 'none'}.")
        out.append("  " + narrate_ranking(rc, sc, "recovery_time_capped_mean", "mean censored recovery time", False, k=3, fmt="{:.0f}"))
    s = h2[(h2["scenario"] == "stationary") & (h2["metric"] == "overall_acc")]
    if len(s):
        out.append("")
        out.append("**Cost of adapting when nothing changes (stationary stream, overall accuracy difference to no adaptation):** " +
                   "; ".join(f"{r['condition']} {r['mean_diff']:+.4f} (Holm p={r['p_holm']:.3g})" for _, r in s.iterrows()) + ".")
    wins = []
    for sc in cfg.run.scenarios:
        if not specs[sc].has_concept_drift:
            continue
        a_ = ap[ap["scenario"] == sc].set_index("policy")["drift_window_acc_mean"]
        for k in ("adwin", "ph", "fixed"):
            if f"{k}_reset" in a_ and f"{k}_replay" in a_:
                wins.append((sc, k, a_[f"{k}_reset"] - a_[f"{k}_replay"]))
    if wins:
        better_reset = [w for w in wins if w[2] > 0]
        out.append("")
        out.append(f"**Reset vs replay (drift-window accuracy, same trigger):** reset has the higher mean in {len(better_reset)} of {len(wins)} (scenario, trigger) pairs; "
                   "mean difference (reset - replay) by trigger: " + "; ".join(
                       f"{k} {np.mean([w[2] for w in wins if w[1] == k]):+.3f}" for k in ("adwin", "ph", "fixed")) +
                   ". A replay window can still contain mostly old-concept observations when an alarm is late, which is one candidate explanation (not tested separately here).")
    drift_pol = {"adwin_reset", "adwin_replay", "ph_reset", "ph_replay"}
    b = h2[(h2["metric"] == "drift_window_acc") & (h2["condition"].isin(drift_pol)) & h2["scenario"].map(lambda s_: specs[s_].has_concept_drift)]
    out.append("")
    out.append(f"Across the {len(b)} (drift scenario x drift-triggered policy) comparisons on drift-window accuracy, "
               f"{int(((b['significant_holm']) & (b['mean_diff'] > 0)).sum())} were significantly better than no adaptation after Holm correction and "
               f"{int(((b['significant_holm']) & (b['mean_diff'] < 0)).sum())} significantly worse.")
    return "\n".join(out)


def rq3_text(res: StudyResults, T: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> str:
    mon = res.monitor_runs
    out = []
    levels = ["sensitive", "default", "conservative"]
    for kind in ("adwin", "ph"):
        m = mon[mon["detector"] == kind]
        row = {}
        for lvl in levels:
            x = m[m["level"] == lvl]
            row[lvl] = (x[x["n_events"] > 0]["detection_delay"].mean(), x["false_alarms_per_1000"].mean(),
                        x[x["n_events"] > 0]["detection_recall"].mean(), x["us_per_update"].mean())
        delays = [row[l][0] for l in levels]
        fas = [row[l][1] for l in levels]
        mono_d = all(np.isfinite(delays)) and delays[0] <= delays[1] <= delays[2]
        mono_f = all(np.isfinite(fas)) and fas[0] >= fas[1] >= fas[2]
        out.append(f"- **{kind.upper()}** (pooled over scenarios; sensitive -> conservative): mean delay "
                   + " / ".join(_f(d, '.0f') for d in delays) + "; false alarms per 1000 obs "
                   + " / ".join(_f(f_, '.3f') for f_ in fas) + "; detection rate "
                   + " / ".join(_f(row[l][2], '.2f') for l in levels) + "; detector update cost (µs) "
                   + " / ".join(_f(row[l][3], '.1f') for l in levels)
                   + f". Delay non-decreasing with conservativeness: {mono_d}; false-alarm rate non-increasing: {mono_f}.")
    st = T["statistical_comparisons"]
    h3 = st[st["hypothesis"] == "H3"]
    for metric in ("detection_delay", "false_alarms_per_1000"):
        a = h3[(h3["metric"] == metric) & (h3["condition"] == "sensitive")]
        out.append(f"- Sensitive vs conservative, `{metric}`: {int(a['significant_holm'].sum())} of {len(a)} scenario/detector comparisons "
                   f"significant after Holm; mean paired difference ranges {_f(a['mean_diff'].min(), '.3g')} to {_f(a['mean_diff'].max(), '.3g')}.")
    il = st[st["hypothesis"] == "H3-inloop"]
    a = il[(il["metric"] == "n_adaptations")]
    out.append(f"- In-loop adaptation counts (sensitive vs conservative replay policies): {int(a['significant_holm'].sum())} of {len(a)} comparisons significant (Holm); "
               f"mean extra adaptations per run range {_f(a['mean_diff'].min(), '.2f')} to {_f(a['mean_diff'].max(), '.2f')}.")
    out.append("- *Note:* pooled delays are computed over *detected* events only; settings that detect fewer events (see detection rates) are summarised on different, typically easier, subsets. The paired tests below use only seeds where both settings detected.")
    cost = T["computational_cost"]
    c = cost.groupby("policy", observed=True)[["us_per_obs_mean", "time_detector_s_mean", "time_adapt_s_mean"]].mean()
    base = c.loc["none", "us_per_obs_mean"] if "none" in c.index else np.nan
    sd_run = res.runs.groupby("policy")["us_per_obs"].std().mean()
    det_us = res.monitor_runs["us_per_update"].max()
    out.append("")
    out.append("**Computational overhead per observation** (mean over scenarios; relative to no adaptation = "
               f"{_f(base, '.1f')} µs): " + "; ".join(f"{p} {r['us_per_obs_mean']:.1f} µs" for p, r in c.iterrows()) +
               f". Run-to-run SD of the per-observation time is about {sd_run:.1f} µs, and the time may also depend on tree size (a hypothesis not tested here), "
               f"so differences of a few µs between policies cannot be attributed to detector overhead; the measured detector update itself costs at most {det_us:.1f} µs per observation.")
    return "\n".join(out)


def conclusion_text(res: StudyResults, T: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> str:
    h1 = T["h1_degradation"]
    st = T["statistical_comparisons"]
    h2 = st[(st["hypothesis"] == "H2") & (st["metric"] == "drift_window_acc")]
    specs = make_scenario_specs(cfg.stream)
    cd = [s for s in cfg.run.scenarios if specs[s].has_concept_drift]
    h1c = h1[h1["scenario"].isin(cd)]
    ctrl = h1[~h1["scenario"].isin(cd)]
    n_sig = int(h1c["did_significant_holm"].sum())
    lines = [
        f"- **H1** (degradation): relative to the stationary stream evaluated on identical windows, the no-adaptation model lost accuracy in {n_sig} of {len(h1c)} concept-drift scenarios "
        f"(Holm-significant); excess drops range {h1c['did_mean'].min():+.3f} to {h1c['did_mean'].max():+.3f}. "
        "Control streams (raw drops; negative = accuracy rose): " +
        "; ".join(f"{r['scenario']} {r['mean_drop']:+.3f}" + (f" (excess vs stationary {r['did_mean']:+.3f}, Holm p={r['did_p_holm']:.3g})" if r['scenario'] != 'stationary' else "") for _, r in ctrl.iterrows()) + ".",
    ]
    dd = h2[h2["scenario"].isin(cd)]
    for pol, g in dd.groupby("condition"):
        lines.append(f"- **H2** `{pol}`: improves drift-window accuracy over no adaptation in {int((g['significant_holm'] & (g['mean_diff'] > 0)).sum())} "
                     f"of {len(g)} drift scenarios (Holm), worse in {int((g['significant_holm'] & (g['mean_diff'] < 0)).sum())}; "
                     f"mean paired difference {g['mean_diff'].mean():+.3f}.")
    lines.append("- **H3**: see the sensitivity statements under RQ3, which report whether delay fell and false alarms rose with sensitivity in these data (paired, Holm-corrected).")
    lines.append(f"- Evidence base: {res.runs['seed'].nunique()} independent seeds per cell, {len(res.runs)} prequential runs, "
                 f"{len(res.failed)} failed runs.")
    return "\n".join(lines)
