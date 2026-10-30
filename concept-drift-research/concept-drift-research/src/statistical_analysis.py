"""Statistics for repeated stream runs.

The unit of replication is the *independent stream run* (one seed).  Policies are
compared on the SAME seeds (common random numbers), so comparisons are paired.
Thousands of temporally dependent predictions within a run are never treated as
independent replicates.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def describe(x: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return {k: np.nan for k in ("n", "mean", "sd", "median", "q25", "q75", "ci_low", "ci_high")}
    m = a.mean()
    sd = a.std(ddof=1) if n > 1 else np.nan
    if n > 1:
        h = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
    else:
        h = np.nan
    return {"n": n, "mean": m, "sd": sd, "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
            "ci_low": m - h, "ci_high": m + h}


def summarize(df: pd.DataFrame, by: list[str], metrics: Iterable[str], alpha: float = 0.05) -> pd.DataFrame:
    """Long table: one row per group x metric with mean/sd/median/IQR/t-CI over runs."""
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            rows.append({**dict(zip(by, key)), "metric": m, **describe(g[m].to_numpy(), alpha)})
    return pd.DataFrame(rows)


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def cohens_dz(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return np.nan
    return float(d.mean() / d.std(ddof=1))


def wilcoxon_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (exact for small n, zero differences dropped).
    Returns 1.0 if all differences are zero."""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) < 1:
        return 1.0
    try:
        return float(stats.wilcoxon(d, alternative="two-sided", method="auto").pvalue)
    except ValueError:
        return np.nan


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs ignored, returned as NaN)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def paired_table(df: pd.DataFrame, group_cols: list[str], cond_col: str, baseline: str,
                 conditions: Sequence[str], metric: str, seed_col: str = "seed",
                 n_boot: int = 5000, alpha: float = 0.05, practical_threshold: float | None = None,
                 holm_family: str = "all") -> pd.DataFrame:
    """Paired comparisons of each condition against ``baseline`` within every group.

    diff = condition - baseline (per seed).  Reports mean diff, bootstrap CI, Cohen's d_z,
    Wilcoxon p, and Holm-adjusted p over the whole family of rows produced by this call."""
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        piv = g.pivot_table(index=seed_col, columns=cond_col, values=metric, aggfunc="first")
        if baseline not in piv:
            continue
        for c in conditions:
            if c == baseline or c not in piv:
                continue
            pair = piv[[c, baseline]].dropna()
            diff = (pair[c] - pair[baseline]).to_numpy()
            lo, hi = paired_bootstrap_ci(diff, n_boot, alpha)
            row = {**dict(zip(group_cols, key)), "metric": metric, "condition": c, "baseline": baseline,
                   "n_pairs": len(diff), "mean_condition": float(pair[c].mean()),
                   "mean_baseline": float(pair[baseline].mean()),
                   "mean_diff": float(diff.mean()) if len(diff) else np.nan,
                   "ci_low": lo, "ci_high": hi, "cohens_dz": cohens_dz(diff),
                   "p_wilcoxon": wilcoxon_p(diff),
                   "frac_runs_improved": float(np.mean(diff > 0)) if len(diff) else np.nan}
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_wilcoxon"].to_numpy())
        out["significant_holm"] = out["p_holm"] < alpha
        if practical_threshold is not None:
            out["practically_relevant"] = out["mean_diff"].abs() >= practical_threshold
    return out


def friedman(df: pd.DataFrame, cond_col: str, metric: str, conditions: Sequence[str],
             seed_col: str = "seed") -> dict[str, float]:
    piv = df.pivot_table(index=seed_col, columns=cond_col, values=metric, aggfunc="first")
    piv = piv[[c for c in conditions if c in piv]].dropna()
    if piv.shape[1] < 3 or piv.shape[0] < 3:
        return {"n_blocks": piv.shape[0], "chi2": np.nan, "p": np.nan, "kendalls_w": np.nan}
    chi2, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv])
    k, n = piv.shape[1], piv.shape[0]
    return {"n_blocks": n, "chi2": float(chi2), "p": float(p), "kendalls_w": float(chi2 / (n * (k - 1)))}
