"""
The risk model being explained.

A gradient-boosted tree ensemble is used because (a) the generative process in
``data.py`` contains interactions and a threshold effect that a linear model
cannot represent, and (b) ``shap.TreeExplainer`` supports it exactly, which
removes sampling noise from one arm of the explanation space.

The model is deliberately *not* wrapped in a calibrator for explanation
purposes: SHAP's tree path-dependent algorithm explains the raw ensemble, and
introducing an isotonic layer on top would mean SHAP and LIME were explaining
different functions.  Calibration quality is still reported so the reader can
see the probabilities are usable as decision inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import FEATURE_NAMES, RANDOM_SEED


@dataclass
class RiskModel:
    """Thin wrapper giving the explainers one stable prediction interface."""

    estimator: GradientBoostingClassifier
    background: pd.DataFrame  # reference cohort used for perturbation baselines
    threshold: float = 0.5

    def predict_proba(self, X) -> np.ndarray:
        X = _as_frame(X)
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # scikit-learn-shaped callables for LIME, which wants a 2-column output.
    def predict_proba_2col(self, X) -> np.ndarray:
        X = _as_frame(X)
        return self.estimator.predict_proba(X)

    @property
    def baseline_values(self) -> np.ndarray:
        """Per-feature reference value used when a feature is 'removed'.

        Median for continuous features, mode for binary ones.  Replacing a
        feature with this value is the operational meaning of "delete" in the
        deletion-based fidelity metric in ``properties.py``.
        """
        vals = []
        for name in FEATURE_NAMES:
            col = self.background[name]
            if set(np.unique(col)) <= {0.0, 1.0}:
                vals.append(float(col.mode().iloc[0]))
            else:
                vals.append(float(col.median()))
        return np.asarray(vals, dtype=float)

    @property
    def base_rate(self) -> float:
        """Model output when every feature is set to its reference value."""
        row = pd.DataFrame([self.baseline_values], columns=FEATURE_NAMES)
        return float(self.predict_proba(row)[0])


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X[FEATURE_NAMES]
    arr = np.atleast_2d(np.asarray(X, dtype=float))
    return pd.DataFrame(arr, columns=FEATURE_NAMES)


def train_risk_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = RANDOM_SEED,
    action_quantile: float = 0.80,
) -> RiskModel:
    """Fit the ensemble and set the decision threshold where clinicians set it.

    A 0.5 cut-off would be wrong here.  Risk models are not used to guess who
    has an event; they are used to decide who gets escalated, and that cut-off
    is set by capacity and guideline, not by the argmax of the posterior.  The
    threshold is therefore the ``action_quantile`` of predicted risk on the
    training cohort -- the top 20% get flagged.  This matters for the framework
    because the counterfactual explainer's whole meaning is "what would move
    this patient across the line", and an unreachable line makes it vacuous.
    """
    clf = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.06,
        max_depth=3,
        subsample=0.85,
        random_state=seed,
    )
    X_train = X_train[FEATURE_NAMES]
    clf.fit(X_train, y_train)

    train_p = clf.predict_proba(X_train)[:, 1]
    threshold = float(np.quantile(train_p, action_quantile))

    return RiskModel(
        estimator=clf, background=X_train.copy(), threshold=threshold
    )


def evaluate(model: RiskModel, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    p = model.predict_proba(X_test)
    yhat = (p >= model.threshold).astype(int)
    tp = int(((yhat == 1) & (y_test == 1)).sum())
    fp = int(((yhat == 1) & (y_test == 0)).sum())
    fn = int(((yhat == 0) & (y_test == 1)).sum())
    return {
        "n_test": int(len(y_test)),
        "event_rate": float(y_test.mean()),
        "auc": float(roc_auc_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "accuracy": float((yhat == y_test).mean()),
        "decision_threshold": float(model.threshold),
        "flag_rate": float(yhat.mean()),
        "precision_at_threshold": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall_at_threshold": float(tp / (tp + fn)) if tp + fn else 0.0,
        "base_rate_at_reference": float(model.base_rate),
    }


def calibration_table(model: RiskModel, X_test: pd.DataFrame, y_test: np.ndarray, bins: int = 5):
    p = model.predict_proba(X_test)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": i + 1,
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()),
                "observed_rate": float(y_test[m].mean()),
            }
        )
    return pd.DataFrame(rows)
