"""Figure functions.  Every figure is drawn from executed results; none uses illustrative data."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ExperimentConfig
from .data_streams import ScenarioSpec, StreamData, make_scenario_specs
from .evaluation import cumulative_mean, rolling_mean
from .experiments import PRIMARY_POLICY_ORDER, StudyResults

POLICY_COLORS = {
    "none": "#4d4d4d", "fixed_reset": "#e69f00", "fixed_replay": "#d55e00",
    "adwin_reset": "#56b4e9", "adwin_replay": "#0072b2",
    "ph_reset": "#66c2a5", "ph_replay": "#009e73",
    "adwin_sensitive_replay": "#9ecae1", "adwin_conservative_replay": "#08306b",
    "ph_sensitive_replay": "#a1d99b", "ph_conservative_replay": "#00441b",
}
DET_COLORS = {"adwin": "#0072b2", "ph": "#009e73"}
LEVEL_ORDER = ["sensitive", "default", "conservative"]
FIG_DPI = 300


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    matplotlib.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": FIG_DPI, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.constrained_layout.use": True,
    })


def save_figure(fig: plt.Figure, name: str, root: Path, formats: Sequence[str] = ("png", "pdf")) -> list[Path]:
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
        paths.append(p)
    return paths


def _mark_drift(ax, spec: ScenarioSpec, label: bool = True) -> None:
    for i, (o, e) in enumerate(zip(spec.onsets, spec.ends)):
        if e > o:
            ax.axvspan(o, e, color="crimson", alpha=0.12, label="drift transition" if (label and i == 0) else None)
            ax.axvline(o, color="crimson", ls="--", lw=0.8)
        else:
            ax.axvline(o, color="crimson", ls="--", lw=1.0, label="known drift point" if (label and i == 0) else None)
    for i, s in enumerate(spec.shift_points):
        ax.axvline(s, color="purple", ls=":", lw=1.0, label="covariate shift (not concept drift)" if (label and i == 0) else None)


def _curves(res: StudyResults, scenario: str, policy: str, fn) -> np.ndarray:
    arr = [fn(e.astype(float)) for (sc, p, _), e in res.errors.items() if sc == scenario and p == policy]
    return np.vstack(arr)


def _band(ax, x, M, color, label, lw=1.2):
    m, s = M.mean(0), M.std(0, ddof=1) if len(M) > 1 else np.zeros(M.shape[1])
    ax.plot(x, m, color=color, lw=lw, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)


# 1 ---------------------------------------------------------------------- #
def fig_stream_overview(streams: dict[str, StreamData], cfg: ExperimentConfig, w: int = 200) -> plt.Figure:
    names = list(streams)
    fig, axes = plt.subplots(len(names), 1, figsize=(7.5, 2.0 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, nm in zip(axes, names):
        sd = streams[nm]
        t = np.arange(len(sd))
        ax.plot(t, rolling_mean(sd.concept, w), color="#0072b2", lw=1.2, label=f"share of concept-B examples (w={w})")
        ax.plot(t, rolling_mean(sd.y, w), color="#d55e00", lw=0.9, label=f"P(Y=1) (w={w})")
        _mark_drift(ax, sd.spec)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{nm}: {sd.spec.description}", loc="left", fontsize=8)
        ax.set_ylabel("proportion")
    axes[0].legend(ncol=2, loc="lower right")
    axes[-1].set_xlabel("observation index t")
    return fig


# 2 ---------------------------------------------------------------------- #
def fig_distributions(sd_swap: StreamData, sd_cov: StreamData, cfg: ExperimentConfig, ref: int = 2000) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for row, (sd, ttl) in enumerate(((sd_swap, "abrupt_swap (P(Y|X) changes)"), (sd_cov, "covariate_shift (P(X) changes)"))):
        pt = sd.spec.onsets[0] if sd.spec.onsets else sd.spec.shift_points[0]
        a, b = slice(max(0, pt - ref), pt), slice(pt, pt + ref)
        ax = axes[row, 0]
        ax.hist(sd.X[a, 0], bins=30, density=True, alpha=0.55, label="before", color="#0072b2")
        ax.hist(sd.X[b, 0], bins=30, density=True, alpha=0.55, label="after", color="#d55e00")
        ax.set_title(f"{ttl}\nfeature x1", loc="left", fontsize=8); ax.set_xlabel("x1"); ax.set_ylabel("density")
        ax.legend()
        ax = axes[row, 1]
        ax.hist(sd.X[a, 2], bins=30, density=True, alpha=0.55, color="#0072b2")
        ax.hist(sd.X[b, 2], bins=30, density=True, alpha=0.55, color="#d55e00")
        ax.set_title("feature x3", loc="left", fontsize=8); ax.set_xlabel("x3")
        ax = axes[row, 2]
        s = sd.X[:, 0] + sd.X[:, 1]
        bins = np.linspace(0, 20, 21)
        for sl, c, lb in ((a, "#0072b2", "before"), (b, "#d55e00", "after")):
            idx = np.digitize(s[sl], bins)
            pr = [sd.y[sl][idx == k].mean() if (idx == k).sum() > 10 else np.nan for k in range(1, len(bins))]
            ax.plot(0.5 * (bins[1:] + bins[:-1]), pr, marker="o", ms=3, color=c, label=lb)
        ax.axvline(cfg.stream.theta_a, color="grey", ls=":", lw=0.8)
        ax.set_title("empirical P(Y=1 | x1+x2)", loc="left", fontsize=8)
        ax.set_xlabel("x1 + x2"); ax.set_ylabel("P(Y=1 | bin)")
    return fig


# 3 / 4 / 5 -------------------------------------------------------------- #
def fig_stationary_baseline(res: StudyResults, cfg: ExperimentConfig) -> plt.Figure:
    w = cfg.eval.rolling_window
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    acc = lambda e: 1 - e
    M_roll = _curves(res, "stationary", "none", lambda e: rolling_mean(1 - e, w))
    M_cum = _curves(res, "stationary", "none", lambda e: cumulative_mean(1 - e))
    t = np.arange(M_roll.shape[1])
    _band(axes[0], t, M_roll, POLICY_COLORS["none"], f"rolling accuracy (w={w})")
    _band(axes[1], t, M_cum, "#0072b2", "cumulative prequential accuracy")
    for ax, ttl in zip(axes, ("Rolling accuracy", "Cumulative accuracy")):
        ax.set_xlabel("observation index t"); ax.set_ylabel("accuracy"); ax.set_title(ttl + " (stationary, no adaptation): mean ± SD over seeds", loc="left", fontsize=8)
        ax.legend()
    return fig


def fig_accuracy_with_drift(res: StudyResults, cfg: ExperimentConfig, kind: str = "accuracy") -> plt.Figure:
    specs = make_scenario_specs(cfg.stream)
    scen = [s for s in cfg.run.scenarios]
    ncol = 2
    nrow = int(np.ceil(len(scen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.3 * nrow), sharey=True)
    w = cfg.eval.rolling_window
    for ax, sc in zip(axes.ravel(), scen):
        if kind == "accuracy":
            M = _curves(res, sc, "none", lambda e: cumulative_mean(1 - e))
            lbl, yl = "cumulative accuracy", "accuracy"
        else:
            M = _curves(res, sc, "none", lambda e: rolling_mean(e, w))
            lbl, yl = f"rolling error rate (w={w})", "error rate"
        _band(ax, np.arange(M.shape[1]), M, POLICY_COLORS["none"], lbl)
        _mark_drift(ax, specs[sc], label=(sc == scen[0]))
        ax.set_title(sc, loc="left", fontsize=8); ax.set_ylabel(yl)
    for ax in axes.ravel()[len(scen):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("observation index t")
    axes.ravel()[0].legend(loc="best")
    return fig


# 6 / 7 ------------------------------------------------------------------ #
def fig_detector_timeline(res: StudyResults, cfg: ExperimentConfig, kind: str, scenario: str = "abrupt_swap",
                          seed_index: int = 0) -> plt.Figure:
    specs = make_scenario_specs(cfg.stream)
    seed = cfg.seeds()[seed_index]
    err = res.errors[(scenario, "none", seed)].astype(float)
    w = cfg.eval.rolling_window
    t = np.arange(len(err))
    fig, axes = plt.subplots(len(LEVEL_ORDER), 1, figsize=(7.5, 6.0), sharex=True)
    for ax, lvl in zip(axes, LEVEL_ORDER):
        ax.plot(t, rolling_mean(err, w), color="#4d4d4d", lw=1.0, label=f"rolling error (w={w})")
        a = res.alarm_log[(res.alarm_log["policy"] == f"monitor:{kind}_{lvl}") &
                          (res.alarm_log["scenario"] == scenario) & (res.alarm_log["seed"] == seed)]["t"].to_numpy()
        ax.vlines(a, 0, 1, color=DET_COLORS[kind], lw=1.5, label="accepted alarm")
        _mark_drift(ax, specs[scenario], label=(lvl == "sensitive"))
        ax.set_ylim(0, max(0.5, rolling_mean(err, w).max() * 1.1))
        ax.set_ylabel("error rate"); ax.set_title(f"{kind.upper()} ({lvl}); {len(a)} alarm(s)", loc="left", fontsize=8)
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("observation index t")
    fig.suptitle(f"{kind.upper()} monitoring the no-adaptation model: {scenario}, seed={seed}", fontsize=9)
    return fig


# 8 / 9 / 10 ------------------------------------------------------------- #
def fig_detector_comparison(res: StudyResults, cfg: ExperimentConfig) -> plt.Figure:
    m = res.monitor_runs
    m = m[(m["level"] == "default") & (m["n_events"] > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    sns.barplot(data=m, x="scenario", y="detection_recall", hue="detector", palette=DET_COLORS, errorbar=("ci", 95),
                ax=axes[0], capsize=0.1, err_kws={"linewidth": 1})
    axes[0].set_ylabel("detection rate (events detected / events)"); axes[0].set_title("Detection rate, default settings", loc="left", fontsize=8)
    sns.barplot(data=m, x="scenario", y="n_alarms", hue="detector", palette=DET_COLORS, errorbar=("ci", 95),
                ax=axes[1], capsize=0.1, err_kws={"linewidth": 1})
    axes[1].set_ylabel("accepted alarms per run"); axes[1].set_title("Alarm volume, default settings", loc="left", fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30); ax.set_xlabel("")
    return fig


def fig_detection_delay(res: StudyResults, cfg: ExperimentConfig) -> plt.Figure:
    m = res.monitor_runs
    m = m[m["n_events"] > 0].dropna(subset=["detection_delay"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    m = m.assign(setting=m["detector"].str.upper() + " " + m["level"])
    order = [f"{k} {l}" for k in ("ADWIN", "PH") for l in LEVEL_ORDER]
    sns.boxplot(data=m, x="scenario", y="detection_delay", hue="setting", hue_order=order, ax=ax, fliersize=2, linewidth=0.7)
    ax.set_ylabel("detection delay (observations after onset)"); ax.set_xlabel("")
    ax.set_title("Detection delay of detected events (missed events excluded; see detection-rate figure)", loc="left", fontsize=8)
    ax.legend(ncol=2, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis="x", rotation=20)
    return fig


def fig_false_alarms(res: StudyResults, cfg: ExperimentConfig) -> plt.Figure:
    m = res.monitor_runs.copy()
    m["setting"] = m["detector"].str.upper() + " " + m["level"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    order = [f"{k} {l}" for k in ("ADWIN", "PH") for l in LEVEL_ORDER]
    sns.barplot(data=m, x="scenario", y="false_alarms_per_1000", hue="setting", hue_order=order, ax=ax,
                errorbar=("ci", 95), capsize=0.05, err_kws={"linewidth": 0.8})
    ax.set_ylabel("false alarms per 1000 observations\n(outside detection windows)"); ax.set_xlabel("")
    ax.set_title("False-alarm rate by detector setting and scenario (mean, 95% CI over seeds)", loc="left", fontsize=8)
    ax.legend(ncol=2, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.tick_params(axis="x", rotation=20)
    return fig


# 11 --------------------------------------------------------------------- #
def fig_policy_curves(res: StudyResults, cfg: ExperimentConfig, scenarios: Sequence[str] | None = None) -> plt.Figure:
    specs = make_scenario_specs(cfg.stream)
    scenarios = list(scenarios or [s for s in cfg.run.scenarios if specs[s].has_concept_drift])
    w = cfg.eval.rolling_window
    ncol = 2
    nrow = int(np.ceil(len(scenarios) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 2.8 * nrow), sharey=True)
    for ax, sc in zip(np.atleast_1d(axes).ravel(), scenarios):
        ylo = 1.0
        for pol in PRIMARY_POLICY_ORDER:
            M = _curves(res, sc, pol, lambda e: rolling_mean(1 - e, w))
            ylo = min(ylo, M.mean(0)[w:].min())
            ax.plot(np.arange(M.shape[1]), M.mean(0), color=POLICY_COLORS[pol], lw=1.0, label=pol)
        _mark_drift(ax, specs[sc], label=False)
        ax.set_title(sc, loc="left", fontsize=8); ax.set_ylabel(f"rolling accuracy (w={w})")
        ax.set_ylim(max(0.0, ylo - 0.03), 1.0)
    for ax in np.atleast_1d(axes).ravel()[len(scenarios):]:
        ax.axis("off")
    for ax in np.atleast_1d(axes)[-1] if nrow > 1 else np.atleast_1d(axes):
        ax.set_xlabel("observation index t")
    flat = np.atleast_1d(axes).ravel()
    handles, labels = flat[0].get_legend_handles_labels()
    if len(flat) > len(scenarios):
        flat[len(scenarios)].legend(handles, labels, loc="center", fontsize=8, title="policy", frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7)
    return fig


# 12 --------------------------------------------------------------------- #
def fig_recovery_time(res: StudyResults, cfg: ExperimentConfig) -> plt.Figure:
    specs = make_scenario_specs(cfg.stream)
    r = res.runs[res.runs["policy"].isin(PRIMARY_POLICY_ORDER) & res.runs["scenario"].map(lambda s: specs[s].has_concept_drift)]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    sns.boxplot(data=r, x="scenario", y="recovery_time_capped", hue="policy", hue_order=PRIMARY_POLICY_ORDER,
                palette=POLICY_COLORS, ax=ax, fliersize=1.5, linewidth=0.6)
    ax.set_ylabel("time to recovery (observations; censored at horizon)"); ax.set_xlabel("")
    ax.set_title(f"Recovery: rolling accuracy >= {cfg.eval.recovery_fraction:.0%} of pre-drift reference for {cfg.eval.recovery_hold} observations", loc="left", fontsize=8)
    ax.legend(ncol=1, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0)); ax.tick_params(axis="x", rotation=15)
    return fig


# 13 --------------------------------------------------------------------- #
def fig_performance_vs_cost(res: StudyResults, cfg: ExperimentConfig) -> plt.Figure:
    specs = make_scenario_specs(cfg.stream)
    r = res.runs[res.runs["scenario"].map(lambda s: specs[s].has_concept_drift)]
    g = r.groupby("policy").agg(acc=("overall_acc", "mean"), us=("us_per_obs", "mean"),
                                adapt=("n_adaptations", "mean"), us_sd=("us_per_obs", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for _, row in g.iterrows():
        ax.scatter(row["us"], row["acc"], s=40 + 40 * row["adapt"], color=POLICY_COLORS.get(row["policy"], "k"),
                   edgecolor="k", lw=0.5, alpha=0.9)
        ax.annotate(row["policy"], (row["us"], row["acc"]), textcoords="offset points", xytext=(4, 4), fontsize=6.5)
    ax.set_xlabel("processing time per observation (µs; mean over drift scenarios and seeds)")
    ax.set_ylabel("overall prequential accuracy (mean over drift scenarios)")
    ax.set_title("Predictive performance vs computational cost (marker area ~ mean number of adaptations)", loc="left", fontsize=8)
    return fig


# 14 --------------------------------------------------------------------- #
def fig_uncertainty(stat_tab: pd.DataFrame, metric: str = "drift_window_acc") -> plt.Figure:
    t = stat_tab[(stat_tab["hypothesis"] == "H2") & (stat_tab["metric"] == metric)]
    scen = list(dict.fromkeys(t["scenario"]))
    conds = list(dict.fromkeys(t["condition"]))
    ncol = 4
    nrow = int(np.ceil(len(scen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, sc in zip(axes, scen):
        s_ = t[t["scenario"] == sc].set_index("condition").reindex(conds)
        y = np.arange(len(conds))
        ax.errorbar(s_["mean_diff"], y, xerr=[s_["mean_diff"] - s_["ci_low"], s_["ci_high"] - s_["mean_diff"]],
                    fmt="o", color="k", ms=3, capsize=2, lw=0.9)
        ax.axvline(0, color="crimson", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(conds, fontsize=7); ax.invert_yaxis()
        ax.set_title(sc, fontsize=8, loc="left"); ax.set_xlabel(f"paired difference in {metric}", fontsize=7)
        for yi, (_, r) in zip(y, s_.iterrows()):
            if bool(r.get("significant_holm", False)):
                ax.annotate("*", (r["ci_high"], yi), xytext=(3, -3), textcoords="offset points", fontsize=11)
    for ax in axes[len(scen):]:
        ax.axis("off")
    fig.suptitle("Difference to the no-adaptation baseline: mean and 95% paired-bootstrap CI over seeds (* = Holm-adjusted p < 0.05)", fontsize=9)
    return fig


# 15 --------------------------------------------------------------------- #
def fig_heatmap(res: StudyResults, cfg: ExperimentConfig, metric: str = "overall_acc") -> plt.Figure:
    r = res.runs
    piv = r.pivot_table(index="policy", columns="scenario", values=metric, aggfunc="mean")
    rows = [p for p in dict.fromkeys(PRIMARY_POLICY_ORDER + list(r["policy"].unique())) if p in piv.index]
    piv = piv.reindex(index=rows, columns=[s for s in cfg.run.scenarios if s in piv.columns])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={"label": metric}, annot_kws={"fontsize": 6.5})
    ax.set_title(f"Mean {metric} by adaptation policy and scenario ({r['seed'].nunique()} seeds)", loc="left", fontsize=8)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(axis="x", rotation=25)
    return fig


# 16 --------------------------------------------------------------------- #
def fig_realworld(logs: dict, w: int = 1000, title: str = "Elec2") -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for name, lg in logs.items():
        axes[0].plot(rolling_mean(1 - lg.errors.astype(float), w), lw=0.9, label=name, color=POLICY_COLORS.get(name))
        if len(lg.alarms):
            axes[1].vlines(lg.alarms, 0, 1, lw=0.7, color=POLICY_COLORS.get(name), alpha=0.8)
    axes[0].set_ylabel(f"rolling accuracy (w={w})"); axes[0].legend(ncol=4)
    axes[1].set_yticks([]); axes[1].set_ylabel("accepted alarms"); axes[1].set_xlabel("observation index (chronological)")
    axes[0].set_title(f"{title}: chronological prequential evaluation (no ground-truth drift points)", loc="left", fontsize=8)
    return fig


# 17 --------------------------------------------------------------------- #
def fig_dashboard(res: StudyResults, cfg: ExperimentConfig, tables: dict[str, pd.DataFrame]) -> plt.Figure:
    specs = make_scenario_specs(cfg.stream)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # (a) post-drift accuracy by policy
    r = res.runs[res.runs["policy"].isin(PRIMARY_POLICY_ORDER) & res.runs["scenario"].map(lambda s: specs[s].has_concept_drift)]
    sns.barplot(data=r, x="scenario", y="drift_window_acc", hue="policy", hue_order=PRIMARY_POLICY_ORDER,
                palette=POLICY_COLORS, ax=axes[0, 0], errorbar=("ci", 95), err_kws={"linewidth": 0.6})
    axes[0, 0].set_ylim(max(0.0, r.groupby(["scenario", "policy"])["drift_window_acc"].mean().min() - 0.06), min(1.0, r.groupby(["scenario", "policy"])["drift_window_acc"].mean().max() + 0.04)); axes[0, 0].set_title("(a) accuracy in drift window", loc="left", fontsize=8)
    axes[0, 0].legend(fontsize=5.5, ncol=2); axes[0, 0].tick_params(axis="x", rotation=25); axes[0, 0].set_xlabel("")
    # (b) detection delay vs false alarms (default level)
    m = res.monitor_runs[res.monitor_runs["level"].isin(LEVEL_ORDER)]
    g = m[m["n_events"] > 0].groupby(["detector", "level"]).agg(delay=("detection_delay", "mean")).reset_index()
    fa = m.groupby(["detector", "level"]).agg(fa=("false_alarms_per_1000", "mean")).reset_index()
    g = g.merge(fa)
    for _, row in g.iterrows():
        axes[0, 1].scatter(row["fa"], row["delay"], color=DET_COLORS[row["detector"]], s=40, edgecolor="k", lw=0.5)
        axes[0, 1].annotate(f"{row['detector']}-{row['level'][:4]}", (row["fa"], row["delay"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    axes[0, 1].set_xlabel("false alarms / 1000 obs (all scenarios)"); axes[0, 1].set_ylabel("mean detection delay")
    axes[0, 1].set_title("(b) detection delay vs false alarms", loc="left", fontsize=8)
    # (c) recovery
    sns.barplot(data=r, x="scenario", y="recovery_time_capped", hue="policy", hue_order=PRIMARY_POLICY_ORDER,
                palette=POLICY_COLORS, ax=axes[1, 0], errorbar=("ci", 95), err_kws={"linewidth": 0.6}, legend=False)
    axes[1, 0].set_title("(c) recovery time (censored)", loc="left", fontsize=8); axes[1, 0].tick_params(axis="x", rotation=25); axes[1, 0].set_xlabel("")
    # (d) cost
    c = res.runs.groupby("policy").agg(us=("us_per_obs", "mean"), acc=("overall_acc", "mean")).reindex(PRIMARY_POLICY_ORDER)
    axes[1, 1].scatter(c["us"], c["acc"], c=[POLICY_COLORS[p] for p in c.index], edgecolor="k", s=45)
    for p, row in c.iterrows():
        axes[1, 1].annotate(p, (row["us"], row["acc"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    axes[1, 1].set_xlabel("µs per observation"); axes[1, 1].set_ylabel("overall accuracy"); axes[1, 1].set_title("(d) accuracy vs cost", loc="left", fontsize=8)
    fig.suptitle("Summary dashboard (all panels computed from executed runs)", fontsize=10)
    return fig
