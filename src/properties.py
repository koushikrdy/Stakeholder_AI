"""
Measurable properties of an explanation.

Five quantities, each mapped into [0, 1] so they can be combined by a weight
vector.  Four of them depend only on (explanation, instance, model); the fifth,
actionability, additionally depends on the stakeholder's downstream action `a`
and is therefore computed in ``utility.py``.

The two model-centric criteria named in the abstract -- fidelity and
completeness -- are separated here rather than conflated, because the whole
argument turns on them behaving differently:

  fidelity      Is the explanation pointing at the features the model is
                actually most sensitive to, *at the budget it chose*?
                Measured against a greedy oracle with the same budget, so a
                two-feature counterfactual is not punished for being short.

  completeness  How much of the model's total local effect do the cited
                features account for in aggregate?  Here a two-feature
                counterfactual *is* penalised, and should be.

Both use the same primitive -- replace a feature with its cohort reference
value and watch the prediction move -- so they are on one scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURE_NAMES
from .explanations import Explanation
from .model import RiskModel

FIDELITY_BUDGET = 3  # max features considered when scoring the pointing


@dataclass
class PropertyScores:
    fidelity: float
    completeness: float
    sparsity: float
    cost: float
    # actionability is stakeholder-specific and added later
    raw_runtime_s: float
    n_cited: int
    degenerate: bool = False

    def as_dict(self) -> dict:
        return {
            "fidelity": self.fidelity,
            "completeness": self.completeness,
            "sparsity": self.sparsity,
            "cost": self.cost,
            "raw_runtime_s": self.raw_runtime_s,
            "n_cited": self.n_cited,
            "degenerate": self.degenerate,
        }


class PropertyEvaluator:
    """Scores explanations for one model, caching the per-instance oracle."""

    def __init__(self, model: RiskModel):
        self.model = model
        self.baseline = model.baseline_values
        self._oracle_cache: dict[int, tuple[list[float], float]] = {}

    # ---------------- primitives ----------------

    def _delete(self, x: np.ndarray, feature_idx) -> np.ndarray:
        z = np.asarray(x, dtype=float).copy()
        for j in feature_idx:
            z[j] = self.baseline[j]
        return z

    def _p(self, rows: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(rows)
        return self.model.predict_proba(pd.DataFrame(rows, columns=FEATURE_NAMES))

    def _oracle(self, x: np.ndarray, instance_id: int):
        """Greedy best-possible deletion curve and the full-deletion effect.

        Returns (curve, full_effect) where curve[k-1] is the largest achievable
        |f(x) - f(x with k features deleted)| found by greedy forward selection.
        """
        if instance_id in self._oracle_cache:
            return self._oracle_cache[instance_id]

        p0 = float(self._p(x)[0])
        chosen: list[int] = []
        curve: list[float] = []
        remaining = list(range(len(FEATURE_NAMES)))

        for _ in range(FIDELITY_BUDGET):
            trials = np.array([self._delete(x, chosen + [j]) for j in remaining])
            effects = np.abs(p0 - self._p(trials))
            best = int(np.argmax(effects))
            chosen.append(remaining[best])
            curve.append(float(effects[best]))
            remaining.pop(best)

        full = float(abs(p0 - self._p(self._delete(x, range(len(FEATURE_NAMES))))[0]))
        self._oracle_cache[instance_id] = (curve, full)
        return curve, full

    # ---------------- properties ----------------

    def fidelity(self, exp: Explanation, x: np.ndarray) -> tuple[float, bool]:
        cited = exp.cited_features
        if not cited:
            return 0.0, False

        k = min(FIDELITY_BUDGET, len(cited))
        p0 = float(self._p(x)[0])
        idx = [FEATURE_NAMES.index(f) for f in cited[:k]]

        achieved = []
        for t in range(1, k + 1):
            z = self._delete(x, idx[:t])
            achieved.append(abs(p0 - float(self._p(z)[0])))

        oracle_curve, _ = self._oracle(x, exp.instance_id)
        oracle_mean = float(np.mean(oracle_curve[:k]))
        if oracle_mean < 1e-6:
            # The model is locally flat; there is no "right" set of levers to
            # point at, so fidelity is undefined rather than perfect.
            return 1.0, True
        return float(np.clip(np.mean(achieved) / oracle_mean, 0.0, 1.0)), False

    def completeness(self, exp: Explanation, x: np.ndarray) -> float:
        cited = exp.cited_features
        if not cited:
            return 0.0
        p0 = float(self._p(x)[0])
        idx = [FEATURE_NAMES.index(f) for f in cited]
        partial = abs(p0 - float(self._p(self._delete(x, idx))[0]))
        _, full = self._oracle(x, exp.instance_id)
        if full < 1e-6:
            return 1.0
        return float(np.clip(partial / full, 0.0, 1.0))

    @staticmethod
    def sparsity(exp: Explanation) -> float:
        n = len(FEATURE_NAMES)
        return float(np.clip(1.0 - exp.n_components / n, 0.0, 1.0))

    # ---------------- batch scoring ----------------

    @staticmethod
    def cost_by_rank(explanations: list[Explanation]) -> dict[str, float]:
        """Relative operational burden of each method, as a rank in [0, 1].

        Cost is measured, but not used raw.  Wall-clock time is machine- and
        load-dependent: scoring it directly made the whole experiment
        irreproducible, with utilities drifting in the third decimal between
        identical runs purely because the CPU was busier.  Since a stakeholder
        experiences cost comparatively -- "expensive next to the alternatives on
        the table" -- the criterion uses each method's *rank* by median runtime,
        evenly spaced from 0 (cheapest) to 1 (dearest).

        This is stable as long as the ordering is, and here the medians are
        separated by orders of magnitude (exact TreeSHAP in microseconds,
        LIME's 2000-sample surrogate in tens of milliseconds), so they are.
        Raw seconds are still recorded per explanation and reported.
        """
        by_method: dict[str, list[float]] = {}
        for e in explanations:
            by_method.setdefault(e.method, []).append(e.runtime_s)
        medians = {m: float(np.median(v)) for m, v in by_method.items()}

        order = sorted(medians, key=lambda m: medians[m])
        denom = max(len(order) - 1, 1)
        return {m: i / denom for i, m in enumerate(order)}

    def score_all(
        self, explanations: list[Explanation], instances: dict[int, np.ndarray]
    ) -> dict[tuple[int, str], PropertyScores]:
        """Score every explanation on the four intrinsic criteria."""
        cost = self.cost_by_rank(explanations)

        out: dict[tuple[int, str], PropertyScores] = {}
        for exp in explanations:
            x = instances[exp.instance_id]
            fid, degenerate = self.fidelity(exp, x)
            out[(exp.instance_id, exp.method)] = PropertyScores(
                fidelity=fid,
                completeness=self.completeness(exp, x),
                sparsity=self.sparsity(exp),
                cost=float(cost[exp.method]),
                raw_runtime_s=float(exp.runtime_s),
                n_cited=exp.n_components,
                degenerate=degenerate,
            )
        return out
