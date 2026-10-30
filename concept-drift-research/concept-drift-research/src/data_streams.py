"""Data streams: controlled SEA-style synthetic generators and the Elec2 loader.

Synthetic design (all scenarios share the same random draws for a given seed,
so scenarios differ ONLY in the documented mechanism -- common random numbers):

* Features  x1, x2, x3 ~ U[0, 10] i.i.d.  (x3 is irrelevant under concept A).
* Concept A : y = 1[x1 + x2 > theta_a]             (SEA as implemented in River, variant 0: theta_a = 8)
* Concept B (``swap``)      : y = 1[x2 + x3 > theta_a]
      -> changes P(Y|X); P(X) unchanged; marginal P(Y) unchanged by symmetry.
* Concept B'(``threshold``) : y = 1[x1 + x2 > theta']
      -> changes P(Y|X) AND the marginal P(Y) (class prevalence) as a side effect.
* ``covariate_shift``: x1 ~ U[3, 10] after the change point, labelling rule
      unchanged -> P(X) changes, P(Y|X) does NOT; P(Y) changes as a side effect.
      This is NOT concept drift (real drift) in the P(Y|X) sense; it is a control.
* Label noise: each label is flipped with probability ``label_noise``
  independently of the concept (so P(Y|X) is noisy but the noise rate is constant).

Gradual drift: for t in [start, end) each observation is generated from concept
B with probability p(t) = (t - start + 0.5) / width (linear ramp) and from
concept A otherwise (independent Bernoulli draws).  For t >= end all
observations come from B.  ``onset`` = start, ``end`` = start + width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import stats

from .config import StreamConfig

FEATURE_NAMES = ("x1", "x2", "x3")


# --------------------------------------------------------------------------- #
# Scenario specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str                       # stationary | covariate | abrupt | gradual | recurring
    concept_b: str                  # swap | threshold | none
    onsets: tuple[int, ...] = ()    # known concept-drift onsets (used for detection matching)
    ends: tuple[int, ...] = ()      # end of transition (== onset for abrupt)
    shift_points: tuple[int, ...] = ()   # non-concept distribution shifts (covariate control)
    description: str = ""

    @property
    def has_concept_drift(self) -> bool:
        return len(self.onsets) > 0


def make_scenario_specs(cfg: StreamConfig) -> dict[str, ScenarioSpec]:
    specs: dict[str, ScenarioSpec] = {}
    specs["stationary"] = ScenarioSpec(
        "stationary", "stationary", "none",
        description="Concept A throughout; no change point of any kind.")
    specs["covariate_shift"] = ScenarioSpec(
        "covariate_shift", "covariate", "none", shift_points=(cfg.covariate_position,),
        description="x1 ~ U[%.1f,%.1f] after t=%d; labelling rule unchanged (P(X) shift only)."
        % (cfg.covariate_x1_low_after, cfg.feature_high, cfg.covariate_position))
    specs["abrupt_swap"] = ScenarioSpec(
        "abrupt_swap", "abrupt", "swap", (cfg.abrupt_position,), (cfg.abrupt_position,),
        description="Abrupt change of P(Y|X): A -> B (swap) at t=%d; P(X), P(Y) preserved."
        % cfg.abrupt_position)
    specs["abrupt_threshold"] = ScenarioSpec(
        "abrupt_threshold", "abrupt", "threshold", (cfg.abrupt_position,), (cfg.abrupt_position,),
        description="Abrupt threshold shift theta %.1f -> %.1f at t=%d (P(Y|X) and P(Y) change)."
        % (cfg.theta_a, cfg.theta_threshold_shift, cfg.abrupt_position))
    for w in cfg.gradual_widths:
        start = cfg.gradual_center - w // 2
        specs[f"gradual_swap_w{w}"] = ScenarioSpec(
            f"gradual_swap_w{w}", "gradual", "swap", (start,), (start + w,),
            description="Gradual A -> B (swap): linear mixture ramp over [%d,%d)." % (start, start + w))
    specs["recurring_swap"] = ScenarioSpec(
        "recurring_swap", "recurring", "swap", tuple(cfg.recurring_positions),
        tuple(cfg.recurring_positions),
        description="Recurring abrupt drift A->B->A->B at t=%s." % (list(cfg.recurring_positions),))
    return specs


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass
class StreamData:
    spec: ScenarioSpec
    seed: int
    X: np.ndarray                  # (N, 3) float
    y: np.ndarray                  # (N,) int8 in {0,1}; observed (possibly noisy) label
    concept: np.ndarray            # (N,) int8: 0 = concept A, 1 = concept B / B'
    y_clean: np.ndarray            # (N,) int8: label before noise flipping
    flipped: np.ndarray            # (N,) bool: noise indicator
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def records(self) -> list[dict[str, float]]:
        """Feature dictionaries in River format (precomputed once per stream)."""
        return [dict(zip(FEATURE_NAMES, map(float, row))) for row in self.X]


def _concept_b_label(X: np.ndarray, kind: str, cfg: StreamConfig) -> np.ndarray:
    if kind == "swap":
        return (X[:, 1] + X[:, 2] > cfg.theta_a)
    if kind == "threshold":
        return (X[:, 0] + X[:, 1] > cfg.theta_threshold_shift)
    return (X[:, 0] + X[:, 1] > cfg.theta_a)  # "none": B == A


def generate_stream(spec: ScenarioSpec, cfg: StreamConfig, seed: int) -> StreamData:
    """Generate one stream.  The draw order is fixed so that all scenarios with the
    same seed share X, the Bernoulli ramp draws and the noise draws."""
    n = cfg.length
    rng = np.random.default_rng(seed)
    X = rng.uniform(cfg.feature_low, cfg.feature_high, size=(n, cfg.n_features))
    x1_alt = rng.uniform(cfg.covariate_x1_low_after, cfg.feature_high, size=n)
    u_ramp = rng.random(n)
    u_noise = rng.random(n)

    t = np.arange(n)
    if spec.kind == "covariate":
        X[t >= spec.shift_points[0], 0] = x1_alt[t >= spec.shift_points[0]]

    concept = np.zeros(n, dtype=np.int8)
    if spec.kind == "abrupt":
        concept[t >= spec.onsets[0]] = 1
    elif spec.kind == "gradual":
        start, end = spec.onsets[0], spec.ends[0]
        p = np.clip((t - start + 0.5) / (end - start), 0.0, 1.0)
        p[t < start] = 0.0
        concept = (u_ramp < p).astype(np.int8)
        concept[t >= end] = 1
    elif spec.kind == "recurring":
        concept = (np.searchsorted(np.array(spec.onsets), t, side="right") % 2).astype(np.int8)

    y_a = (X[:, 0] + X[:, 1] > cfg.theta_a)
    y_b = _concept_b_label(X, spec.concept_b, cfg)
    y_clean = np.where(concept == 1, y_b, y_a).astype(np.int8)
    flipped = u_noise < cfg.label_noise
    y = np.where(flipped, 1 - y_clean, y_clean).astype(np.int8)
    return StreamData(spec, seed, X, y, concept, y_clean, flipped,
                      meta={"onsets": spec.onsets, "ends": spec.ends,
                            "shift_points": spec.shift_points, "length": n})


# --------------------------------------------------------------------------- #
# Validation of the generator against its documented configuration
# --------------------------------------------------------------------------- #
def validate_stream(sd: StreamData, cfg: StreamConfig, tol_sigma: float = 4.0) -> dict[str, bool | float]:
    """Check that a generated stream follows its intended configuration."""
    n = len(sd)
    out: dict[str, bool | float] = {}
    y_a = (sd.X[:, 0] + sd.X[:, 1] > cfg.theta_a).astype(np.int8)
    y_b = _concept_b_label(sd.X, sd.spec.concept_b, cfg).astype(np.int8)
    out["labels_follow_declared_concept"] = bool(
        np.all(sd.y_clean == np.where(sd.concept == 1, y_b, y_a)))
    p = cfg.label_noise
    sd_noise = np.sqrt(n * p * (1 - p)) if n else 1.0
    out["noise_rate"] = float(sd.flipped.mean())
    out["noise_rate_within_tolerance"] = bool(abs(sd.flipped.sum() - n * p) <= tol_sigma * sd_noise)
    spec = sd.spec
    if spec.kind == "stationary" or spec.kind == "covariate":
        out["no_concept_change_points"] = bool(sd.concept.sum() == 0)
    if spec.kind == "abrupt":
        o = spec.onsets[0]
        out["abrupt_onset_exact"] = bool(sd.concept[:o].sum() == 0 and sd.concept[o:].all())
    if spec.kind == "gradual":
        s, e = spec.onsets[0], spec.ends[0]
        out["gradual_pure_before_start"] = bool(sd.concept[:s].sum() == 0)
        out["gradual_pure_after_end"] = bool(sd.concept[e:].all())
        third = (e - s) // 3
        frac = [sd.concept[s + i * third: s + (i + 1) * third].mean() for i in range(3)]
        out["gradual_mixture_increasing"] = bool(frac[0] < frac[1] < frac[2])
    if spec.kind == "recurring":
        ok = True
        bounds = [0, *spec.onsets, n]
        for i in range(len(bounds) - 1):
            seg = sd.concept[bounds[i]:bounds[i + 1]]
            ok &= bool(np.all(seg == i % 2))
        out["recurring_alternation_exact"] = ok
    if spec.kind == "covariate":
        sp = spec.shift_points[0]
        out["covariate_x1_min_after"] = float(sd.X[sp:, 0].min())
        out["covariate_x1_shifted"] = bool(sd.X[sp:, 0].min() >= cfg.covariate_x1_low_after - 1e-12)
    return out


def distribution_summary(sd: StreamData, split: int, ref_window: int) -> dict[str, float]:
    """Quantities that distinguish covariate / prior / concept shift around ``split``."""
    a = slice(max(0, split - ref_window), split)
    b = slice(split, min(len(sd), split + ref_window))
    X, y, yc = sd.X, sd.y, sd.y_clean
    ks = [stats.ks_2samp(X[a, j], X[b, j]).statistic for j in range(X.shape[1])]
    y_a = (X[b, 0] + X[b, 1] > 8.0)  # nominal concept-A rule on post window (for reference)
    return {
        "P(Y=1) before": float(y[a].mean()),
        "P(Y=1) after": float(y[b].mean()),
        "max KS stat on P(X) marginals": float(max(ks)),
        "label disagreement with concept A rule after (noise-free labels)":
            float(np.mean(yc[b] != y_a.astype(np.int8))),
    }


def concept_disagreement_rate(cfg: StreamConfig, kind: str, n: int = 400_000, seed: int = 0) -> float:
    """Monte-Carlo P(label_A != label_B): the 'magnitude' of a concept change."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(cfg.feature_low, cfg.feature_high, size=(n, cfg.n_features))
    ya = (X[:, 0] + X[:, 1] > cfg.theta_a)
    yb = _concept_b_label(X, kind, cfg)
    return float(np.mean(ya != yb))


# --------------------------------------------------------------------------- #
# Elec2 (real-world stream)
# --------------------------------------------------------------------------- #
ELEC2_COLUMNS = ["date", "day", "period", "nswprice", "nswdemand", "vicprice",
                 "vicdemand", "transfer", "class"]
ELEC2_EXPECTED_ROWS = 45_312


class DatasetUnavailableError(RuntimeError):
    """Raised when Elec2 cannot be obtained; message contains the alternative workflow."""


ELEC2_HELP = (
    "Elec2 could not be loaded.\n"
    "Documented source: River API  river.datasets.Elec2  "
    "(https://riverml.xyz/dev/api/datasets/Elec2/), which downloads "
    "https://maxhalford.github.io/files/datasets/electricity.zip.\n"
    "Alternative workflow: obtain the same dataset on a machine with internet access "
    "(e.g. run `python -c \"from river import datasets; datasets.Elec2().download()\"`; the file is "
    "stored as <river_data>/Elec2/electricity.csv), copy electricity.csv to "
    "data/raw/electricity.csv inside this project (columns: " + ", ".join(ELEC2_COLUMNS) + "), "
    "and re-run the notebook/script.  No substitute dataset is used."
)


def _read_elec2_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ELEC2_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetUnavailableError(f"{path} lacks expected Elec2 columns {missing}.\n" + ELEC2_HELP)
    return df[ELEC2_COLUMNS].copy()


def load_elec2(project_root: Path, allow_download: bool = True) -> tuple[pd.DataFrame, str]:
    """Return (dataframe in original chronological order, provenance string).

    Order of attempts: (1) data/raw/electricity.csv, (2) River's dataset API
    (downloads if needed), (3) raise :class:`DatasetUnavailableError`.
    The row order of the source file is never changed.
    """
    local = project_root / "data" / "raw" / "electricity.csv"
    if local.exists():
        return _read_elec2_csv(local), f"local file {local.relative_to(project_root)}"
    if allow_download:
        try:
            from river import datasets
            ds = datasets.Elec2()
            rows = []
            for x, y in ds:     # River downloads on first iteration
                r = dict(x)
                r["class"] = "UP" if y else "DOWN"
                rows.append(r)
            df = pd.DataFrame(rows)[ELEC2_COLUMNS]
            return df, "river.datasets.Elec2"
        except Exception as exc:  # network failure, missing cache, etc.
            raise DatasetUnavailableError(f"{type(exc).__name__}: {exc}\n" + ELEC2_HELP) from exc
    raise DatasetUnavailableError(ELEC2_HELP)


ELEC2_FEATURES = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]


def elec2_to_stream(df: pd.DataFrame) -> tuple[list[dict[str, float]], np.ndarray]:
    """Chronological (X records, y in {0,1}); 'UP' is the positive class (1).
    No shuffling, no scaling (the source is pre-normalised -- see limitations)."""
    y = (df["class"].astype(str).str.upper() == "UP").astype(np.int8).to_numpy()
    recs = df[ELEC2_FEATURES].astype(float).to_dict(orient="records")
    return recs, y


def iter_stream(sd: StreamData) -> Iterator[tuple[dict[str, float], int]]:
    for rec, yy in zip(sd.records(), sd.y):
        yield rec, int(yy)
