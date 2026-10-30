"""Central, typed configuration for the concept-drift study.

All experimental constants live here (or in ``configs/default.yaml``, which is
loaded into these dataclasses).  Nothing else in the code base should contain
magic numbers.  Use :func:`load_config` to build a config for a profile
(``full`` or ``smoke``) and :func:`make_variants` to create configuration
variants for sensitivity analyses.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


# --------------------------------------------------------------------------- #
# Project root detection (does not rely on the current working directory)
# --------------------------------------------------------------------------- #
def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` until a directory containing ``pyproject.toml``
    and ``src/`` is found.  Works from the notebook, scripts and tests."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    # Fallback: location of this file (src/config.py -> project root)
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StreamConfig:
    length: int = 10_000
    n_features: int = 3
    feature_low: float = 0.0
    feature_high: float = 10.0
    theta_a: float = 8.0                 # concept A: y = 1[x1 + x2 > theta_a]
    theta_threshold_shift: float = 9.5   # concept B' (threshold shift): y = 1[x1 + x2 > this]
    label_noise: float = 0.05            # symmetric label-flip probability
    abrupt_position: int = 5_000
    gradual_center: int = 5_000
    gradual_widths: tuple[int, ...] = (1_000, 2_500)
    recurring_positions: tuple[int, ...] = (2_500, 5_000, 7_500)
    covariate_position: int = 5_000
    covariate_x1_low_after: float = 3.0  # x1 ~ U[low_after, high] after the shift


@dataclass(frozen=True)
class ModelConfig:
    primary: str = "hoeffding_tree"
    ht_grace_period: int = 100
    ht_delta: float = 1e-5
    ht_leaf_prediction: str = "nba"


@dataclass(frozen=True)
class DetectorConfig:
    # sensitivity levels are fixed a priori (library default = "default")
    adwin_levels: Mapping[str, Mapping[str, float]] = field(default_factory=lambda: {
        "sensitive": {"delta": 0.02},
        "default": {"delta": 0.002},
        "conservative": {"delta": 0.0002},
    })
    adwin_direction: str = "increase"   # only error increases count as alarms
    ph_levels: Mapping[str, Mapping[str, float]] = field(default_factory=lambda: {
        "sensitive": {"threshold": 25.0},
        "default": {"threshold": 50.0},
        "conservative": {"threshold": 100.0},
    })
    ph_min_instances: int = 30
    ph_delta: float = 0.005
    ph_alpha: float = 0.9999
    ph_mode: str = "up"


@dataclass(frozen=True)
class AdaptationConfig:
    fixed_window: int = 2_000   # rebuild period W for the fixed-window policy
    replay_size: int = 500      # number of most recent labelled observations replayed


@dataclass(frozen=True)
class EvalConfig:
    rolling_window: int = 200
    pre_window: int = 1_000
    post_window: int = 1_000
    detection_tolerance: int = 1_000
    recovery_fraction: float = 0.95
    recovery_hold: int = 200
    recovery_fraction_sensitivity: tuple[float, ...] = (0.90, 0.95, 0.98)
    positive_class: int = 1


@dataclass(frozen=True)
class RunConfig:
    profile: str = "full"
    base_seed: int = 20260101
    n_seeds: int = 20
    n_bootstrap: int = 5_000
    alpha: float = 0.05
    memory_runs: int = 3
    scenarios: tuple[str, ...] = (
        "stationary", "covariate_shift", "abrupt_swap", "abrupt_threshold",
        "gradual_swap_w1000", "gradual_swap_w2500", "recurring_swap",
    )


@dataclass(frozen=True)
class PathConfig:
    root: Path = field(default_factory=find_project_root)
    results: str = "results"
    tables: str = "results/tables"
    raw: str = "results/raw"
    logs: str = "results/logs"
    figures: str = "figures"
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"

    def path(self, name: str) -> Path:
        p = self.root / getattr(self, name)
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class ExperimentConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def seeds(self) -> list[int]:
        return [self.run.base_seed + i for i in range(self.run.n_seeds)]

    def to_dict(self) -> dict[str, Any]:
        def conv(o: Any) -> Any:
            if is_dataclass(o):
                return {f.name: conv(getattr(o, f.name)) for f in fields(o)}
            if isinstance(o, Mapping):
                return {k: conv(v) for k, v in o.items()}
            if isinstance(o, (tuple, list)):
                return [conv(v) for v in o]
            if isinstance(o, Path):
                return str(o)
            return o
        return conv(self)


# --------------------------------------------------------------------------- #
# Loading and variants
# --------------------------------------------------------------------------- #
_SECTIONS = {
    "stream": StreamConfig, "model": ModelConfig, "detector": DetectorConfig,
    "adaptation": AdaptationConfig, "eval": EvalConfig, "run": RunConfig,
}


def _deep_update(base: dict, upd: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(raw: Mapping[str, Mapping[str, Any]], root: Path | None) -> ExperimentConfig:
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = dict(raw.get(name, {}))
        for f in fields(cls):  # YAML lists -> tuples where the dataclass expects tuples
            if f.name in section and isinstance(section[f.name], list):
                section[f.name] = tuple(section[f.name])
        kwargs[name] = cls(**section)
    paths = PathConfig(root=root) if root is not None else PathConfig()
    return ExperimentConfig(paths=paths, **kwargs)


def load_config(profile: str = "full", path: str | Path | None = None,
                root: Path | None = None) -> ExperimentConfig:
    """Load ``configs/default.yaml`` and apply the requested profile overrides."""
    root = root or find_project_root()
    path = Path(path) if path else root / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    base = {k: v for k, v in raw.items() if k != "profiles"}
    profiles = raw.get("profiles", {})
    if profile not in profiles and profile != "full":
        raise KeyError(f"Unknown profile {profile!r}; available: {sorted(profiles)} + 'full'")
    merged = _deep_update(base, profiles.get(profile, {}))
    merged.setdefault("run", {})["profile"] = profile
    return _build(merged, root)


def make_variants(base: ExperimentConfig, grid: Mapping[str, Iterable[Any]]) -> list[ExperimentConfig]:
    """Cartesian product of configuration variants.

    ``grid`` keys use ``section.field`` notation, e.g.
    ``{"adaptation.fixed_window": [1000, 2000], "stream.label_noise": [0.0, 0.05]}``.
    """
    keys = list(grid)
    variants = []
    for combo in itertools.product(*[list(grid[k]) for k in keys]):
        cfg = base
        for key, val in zip(keys, combo):
            section, fname = key.split(".", 1)
            if section not in _SECTIONS:
                raise KeyError(f"Unknown config section {section!r}")
            sec_obj = getattr(cfg, section)
            if fname not in {f.name for f in fields(sec_obj)}:
                raise KeyError(f"{section} has no field {fname!r}")
            cfg = replace(cfg, **{section: replace(sec_obj, **{fname: val})})
        variants.append(cfg)
    return variants
