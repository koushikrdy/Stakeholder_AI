

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 20260914

# --------------------------------------------------------------------------
# Feature schema
# --------------------------------------------------------------------------
# `agency` records which actor can move the feature in the real world.
#   "clinical"  -> a clinician can change it via treatment
#   "lifestyle" -> the patient can change it via behaviour
#   "both"      -> reachable by clinician and patient through different routes
#   "fixed"     -> nobody can change it (demographics, history)


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    units: str
    agency: str
    kind: str  # "continuous" or "binary"
    lo: float
    hi: float


FEATURES: tuple[Feature, ...] = (
    Feature("age", "Age", "years", "fixed", "continuous", 25, 70),
    Feature("male", "Male", "0/1", "fixed", "binary", 0, 1),
    Feature("cholesterol_level", "Cholesterol level", "1-3 ordinal", "clinical", "continuous", 1, 3),
    Feature("systolic_bp", "Systolic blood pressure", "mmHg", "both", "continuous", 80, 220),
    Feature("glucose_level", "Glucose level", "1-3 ordinal", "both", "continuous", 1, 3),
    Feature("bmi", "Body mass index", "kg/m2", "lifestyle", "continuous", 14, 55),
    Feature("smoker", "Current smoker", "0/1", "lifestyle", "binary", 0, 1),
    Feature("alcohol", "Alcohol intake", "0/1", "lifestyle", "binary", 0, 1),
    Feature("active", "Physically active", "0/1", "lifestyle", "binary", 0, 1),
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURE_INDEX: dict[str, int] = {f.name: i for i, f in enumerate(FEATURES)}
FEATURE_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}

# Ground-truth log-odds contributions.  Used only for reporting/validation,
# never shown to the explainers.
TRUE_LINEAR_EFFECTS: dict[str, float] = {
    "age": 0.045,
    "male": 0.30,
    "cholesterol_level": 0.55,
    "systolic_bp": 0.030,
    "glucose_level": 0.45,
    "bmi": 0.060,
    "smoker": 0.35,
    "alcohol": 0.10,
    "active": -0.45,          # the one genuinely protective-when-present feature
}

# Features that genuinely do nothing in the generative process would be useful
# as a negative control, but every feature above is a real CVD risk factor, so
# the "noise" control is provided by the interaction terms being absent for
# some features rather than by dead features.


def _truncated_normal(rng, mean, sd, lo, hi, n):
    x = rng.normal(mean, sd, n)
    return np.clip(x, lo, hi)


def generate_cohort(n_patients: int = 4000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Draw a synthetic patient cohort with correlated, clinically plausible features."""
    rng = np.random.default_rng(seed)
    n = n_patients

    age = _truncated_normal(rng, 53, 7, 25, 70, n)
    male = (rng.random(n) < 0.35).astype(float)  # this dataset skews female (~65%)

    # Metabolic cluster: BMI, cholesterol, glucose and BP move together.
    metabolic = rng.normal(0, 1, n)

    bmi = np.clip(26.0 + 3.6 * metabolic + rng.normal(0, 3.0, n) + 0.03 * (age - 53), 14, 55)
    cholesterol_level = np.clip(
        np.round(1.35 + 0.55 * metabolic + rng.normal(0, 0.4, n)), 1, 3
    )
    glucose_level = np.clip(
        np.round(1.20 + 0.45 * metabolic + rng.normal(0, 0.35, n)), 1, 3
    )
    systolic_bp = np.clip(
        122 + 9.0 * metabolic + 0.55 * (age - 53) + rng.normal(0, 12, n), 80, 220
    )

    # Smoking and drinking are far more common among men in this cohort.
    smoker = np.where(male == 1, (rng.random(n) < 0.35).astype(float),
                       (rng.random(n) < 0.07).astype(float))
    alcohol = np.where(male == 1, (rng.random(n) < 0.25).astype(float),
                        (rng.random(n) < 0.04).astype(float))

    # Health-seeking behaviour: being active is negatively coupled with
    # metabolic burden.
    behaviour = rng.normal(0, 1, n) - 0.30 * metabolic
    active = (behaviour > -0.35).astype(float)

    X = pd.DataFrame(
        {
            "age": age,
            "male": male,
            "cholesterol_level": cholesterol_level,
            "systolic_bp": systolic_bp,
            "glucose_level": glucose_level,
            "bmi": bmi,
            "smoker": smoker,
            "alcohol": alcohol,
            "active": active,
        }
    )[FEATURE_NAMES]

    logit = _true_logit(X, rng)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)

    X = X.copy()
    X["_true_risk"] = prob
    X["_event"] = y
    return X


def load_real_world_cohort(path: str | Path, target: str = "_event") -> pd.DataFrame:
    """Load a real cohort while preserving the pipeline's feature contract."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"real-world dataset not found: {source}")

    raw = pd.read_csv(source)
    required = [*FEATURE_NAMES, target]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(
            "real-world CSV is missing required columns: " + ", ".join(missing)
        )

    cohort = raw[required].copy()
    for name in required:
        cohort[name] = pd.to_numeric(cohort[name], errors="coerce")
    cohort = cohort.dropna().reset_index(drop=True)

    outcome = cohort.pop(target)
    unique_outcomes = set(outcome.unique())
    if not unique_outcomes <= {0, 1} or len(unique_outcomes) < 2:
        raise ValueError(
            f"outcome column {target!r} must contain both numeric values 0 and 1"
        )
    cohort["_event"] = outcome.astype(int)

    for feature in FEATURES:
        values = set(cohort[feature.name].unique())
        if feature.kind == "binary" and not values <= {0, 1}:
            raise ValueError(
                f"binary feature {feature.name!r} must contain only numeric 0/1 values"
            )
        if ((cohort[feature.name] < feature.lo) | (cohort[feature.name] > feature.hi)).any():
            raise ValueError(
                f"feature {feature.name!r} contains values outside [{feature.lo}, {feature.hi}]"
            )
    if len(cohort) < 20:
        raise ValueError("real-world dataset must contain at least 20 complete rows")

    return cohort


def _true_logit(X: pd.DataFrame, rng) -> np.ndarray:
    """Generative log-odds: linear main effects plus two non-linear terms.

    The non-linearities are what justify using a gradient-boosted model rather
    than logistic regression, and therefore what makes post-hoc explanation
    necessary at all.
    """
    z = np.full(len(X), -2.45)

    # Centred linear main effects.
    centres = {
        "age": 53.0,
        "male": 0.0,
        "cholesterol_level": 1.35,
        "systolic_bp": 122.0,
        "glucose_level": 1.20,
        "bmi": 26.0,
        "smoker": 0.0,
        "alcohol": 0.0,
        "active": 1.0,
    }
    for name, beta in TRUE_LINEAR_EFFECTS.items():
        z = z + beta * (X[name].to_numpy() - centres[name])

    # Interaction 1: smoking is far more dangerous at high blood pressure.
    z = z + 0.017 * X["smoker"].to_numpy() * (X["systolic_bp"].to_numpy() - 122.0)

    # Interaction 2: high cholesterol compounds elevated glucose.
    z = z + 0.20 * (X["cholesterol_level"].to_numpy() - 1.35) * np.maximum(0.0, X["glucose_level"].to_numpy() - 1.20)

    # Threshold effect: risk accelerates above stage-2 hypertension.
    z = z + 0.65 * (X["systolic_bp"].to_numpy() > 160).astype(float)

    z = z + rng.normal(0, 0.28, len(X))
    return z


def split_cohort(df: pd.DataFrame, test_frac: float = 0.3, seed: int = RANDOM_SEED):
    """Deterministic train/test split returning (X_train, y_train, X_test, y_test)."""
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(df))
    n_test = int(round(test_frac * len(df)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    feats = df[FEATURE_NAMES]
    y = df["_event"].to_numpy()

    return (
        feats.iloc[train_idx].reset_index(drop=True),
        y[train_idx],
        feats.iloc[test_idx].reset_index(drop=True),
        y[test_idx],
    )


def agency_sets() -> dict[str, set[str]]:
    """Map each agency tag to the set of feature names carrying it."""
    out: dict[str, set[str]] = {}
    for f in FEATURES:
        out.setdefault(f.agency, set()).add(f.name)
    return out
