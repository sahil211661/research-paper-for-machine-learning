"""Incremental classifiers (River).  Both expose predict_one / learn_one."""
from __future__ import annotations

from typing import Callable

from river import naive_bayes, tree

from .config import ModelConfig

ModelFactory = Callable[[], object]


def make_model_factory(name: str, cfg: ModelConfig) -> ModelFactory:
    """Return a zero-argument factory producing a *fresh, untrained* model."""
    if name == "hoeffding_tree":
        return lambda: tree.HoeffdingTreeClassifier(
            grace_period=cfg.ht_grace_period, delta=cfg.ht_delta,
            leaf_prediction=cfg.ht_leaf_prediction)
    if name == "gaussian_nb":
        return lambda: naive_bayes.GaussianNB()
    raise KeyError(f"unknown model {name!r}")


def check_online_api(model: object) -> None:
    for m in ("predict_one", "learn_one"):
        if not callable(getattr(model, m, None)):
            raise TypeError(f"{type(model).__name__} lacks required online method {m}()")
