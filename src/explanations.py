"""
The explanation space  E = {SHAP, LIME, Counterfactual, Rules}.

Every explainer returns the same ``Explanation`` object.  That is the whole
point of this module: the framework argmaxes a utility over a set of candidate
explanations, so the candidates must be commensurable.  Each method is reduced
to three things the utility function can read:

  * ``attributions`` -- a signed importance per cited feature
  * ``cited_features`` -- the features the reader is actually asked to look at,
    ordered most-important first
  * ``n_components`` / ``runtime_s`` -- the presentation and compute load

Reducing a counterfactual to an "attribution" needs a word of justification.
A counterfactual says "change X from a to b".  We record the induced change,
normalised by the feature's cohort spread, as its importance.  That is not the
same object as a Shapley value and we do not pretend it is -- but both answer
"which features does this explanation put in front of the stakeholder, and how
strongly", which is exactly what the downstream metrics consume.
"""

from __future__ import annotations

import time
import warnings
from itertools import combinations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from .data import FEATURE_BY_NAME, FEATURE_NAMES, RANDOM_SEED
from .model import RiskModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

METHODS = ("SHAP", "LIME", "Counterfactual", "Rules")


def _lede(label: str) -> str:
    """Lower-case a feature label for mid-sentence use, leaving acronyms alone."""
    head = label.split(" ")[0]
    # Leave acronyms (LDL) and mixed-case clinical terms (HbA1c) untouched.
    if any(c.isupper() for c in head[1:]):
        return label
    return label[0].lower() + label[1:]


@dataclass
class Explanation:
    method: str
    instance_id: int
    attributions: dict[str, float]
    runtime_s: float
    prediction: float
    payload: dict = field(default_factory=dict)

    @property
    def cited_features(self) -> list[str]:
        """Features carrying non-negligible weight, strongest first."""
        items = [(k, abs(v)) for k, v in self.attributions.items() if abs(v) > 1e-9]
        items.sort(key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in items]

    @property
    def n_components(self) -> int:
        return len(self.cited_features)

    def mass(self) -> float:
        return float(sum(abs(v) for v in self.attributions.values()))

    def mass_on(self, feature_set) -> float:
        """Share of total attribution mass sitting on a given feature set."""
        total = self.mass()
        if total <= 1e-12:
            return 0.0
        hit = sum(abs(v) for k, v in self.attributions.items() if k in feature_set)
        return float(hit / total)

    def describe(self) -> str:
        if self.method == "Counterfactual":
            return self.payload.get("text", "")
        if self.method == "Rules":
            return self.payload.get("text", "")
        parts = [
            f"{FEATURE_BY_NAME[f].label} {'+' if self.attributions[f] > 0 else '-'}"
            f"{abs(self.attributions[f]):.3f}"
            for f in self.cited_features[:5]
        ]
        return "; ".join(parts)


# --------------------------------------------------------------------------
# Local neighbourhood -- shared by LIME and the rule extractor so that the two
# surrogate-based methods are fitted on comparable data.
# --------------------------------------------------------------------------


def local_neighbourhood(
    x: np.ndarray,
    background: pd.DataFrame,
    n_samples: int = 600,
    scale: float = 0.45,
    seed: int = 0,
) -> np.ndarray:
    """Sample points around ``x`` by feature-wise perturbation.

    Continuous features get Gaussian noise proportional to their cohort
    standard deviation; binary features flip with a fixed probability.  This is
    the same neighbourhood used for the faithfulness metric, so no method is
    evaluated on a distribution it was not fitted to.
    """
    rng = np.random.default_rng(seed)
    sd = background[FEATURE_NAMES].std(ddof=0).to_numpy()
    Z = np.tile(x, (n_samples, 1)).astype(float)

    for j, name in enumerate(FEATURE_NAMES):
        feat = FEATURE_BY_NAME[name]
        if feat.kind == "binary":
            flip = rng.random(n_samples) < 0.18
            Z[flip, j] = 1.0 - Z[flip, j]
        else:
            Z[:, j] = Z[:, j] + rng.normal(0, scale * sd[j], n_samples)
            Z[:, j] = np.clip(Z[:, j], feat.lo, feat.hi)

    Z[0] = x  # keep the instance itself in the sample
    return Z


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


class ShapExplainer:
    name = "SHAP"

    def __init__(self, model: RiskModel):
        import shap

        self.model = model
        self._explainer = shap.TreeExplainer(model.estimator)

    def explain(self, x: np.ndarray, instance_id: int) -> Explanation:
        t0 = time.perf_counter()
        row = pd.DataFrame([x], columns=FEATURE_NAMES)
        vals = self._explainer.shap_values(row)
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[0, :, -1]
        else:
            vals = vals[0]
        runtime = time.perf_counter() - t0

        attributions = {name: float(vals[j]) for j, name in enumerate(FEATURE_NAMES)}
        return Explanation(
            method=self.name,
            instance_id=instance_id,
            attributions=attributions,
            runtime_s=runtime,
            prediction=float(self.model.predict_proba(row)[0]),
            payload={"expected_value": float(np.ravel(self._explainer.expected_value)[-1])},
        )


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------


class LimeExplainer:
    name = "LIME"

    def __init__(self, model: RiskModel, n_features: int = 6, n_samples: int = 2000):
        from lime.lime_tabular import LimeTabularExplainer

        self.model = model
        self.n_features = n_features
        self.n_samples = n_samples
        cat_idx = [
            j for j, n in enumerate(FEATURE_NAMES) if FEATURE_BY_NAME[n].kind == "binary"
        ]
        self._explainer = LimeTabularExplainer(
            training_data=model.background[FEATURE_NAMES].to_numpy(),
            feature_names=list(FEATURE_NAMES),
            class_names=["no event", "event"],
            categorical_features=cat_idx,
            discretize_continuous=True,
            random_state=RANDOM_SEED,
            mode="classification",
        )

    def explain(self, x: np.ndarray, instance_id: int) -> Explanation:
        t0 = time.perf_counter()
        exp = self._explainer.explain_instance(
            data_row=np.asarray(x, dtype=float),
            predict_fn=self.model.predict_proba_2col,
            num_features=self.n_features,
            num_samples=self.n_samples,
            labels=(1,),
        )
        runtime = time.perf_counter() - t0

        attributions = {name: 0.0 for name in FEATURE_NAMES}
        for feat_idx, weight in exp.as_map()[1]:
            attributions[FEATURE_NAMES[feat_idx]] = float(weight)

        row = pd.DataFrame([x], columns=FEATURE_NAMES)
        return Explanation(
            method=self.name,
            instance_id=instance_id,
            attributions=attributions,
            runtime_s=runtime,
            prediction=float(self.model.predict_proba(row)[0]),
            payload={"surrogate_r2": float(exp.score)},
        )


# --------------------------------------------------------------------------
# Counterfactual
# --------------------------------------------------------------------------


class CounterfactualExplainer:
    """Minimal-cardinality, cost-weighted counterfactual.

    Sparsity is imposed by the *search order*, not by pruning afterwards.  The
    search enumerates feature subsets of size 1, then 2, then 3, and stops at
    the first cardinality that admits a valid counterfactual.  This matters:
    an earlier implementation perturbed all mutable features at once and pruned
    back, which reliably produced dense counterfactuals, because once every
    feature has been shrunk to its minimum no single one can be reverted
    without losing validity.  Searching small-first avoids that trap entirely
    and is what makes the counterfactual arm genuinely sparse.

    Within a subset, candidates are sampled, the cheapest valid one is kept,
    and each changed feature is then binary-searched back toward the factual
    value for as long as validity holds.

    ``margin`` is a robustness requirement, and it is not cosmetic.  A tree
    ensemble is a step function, so a bare "cross the threshold" constraint is
    happily satisfied by nudging a feature a hair past one split point -- which
    yields advice like "reduce LDL by 1.8 mg/dL" that is an artefact of the
    model's geometry rather than a real intervention.  Requiring the
    counterfactual to land ``margin`` clear of the threshold rules those out.
    """

    name = "Counterfactual"
    MAX_CARDINALITY = 3
    max_robustness_checks = 40

    def __init__(
        self,
        model: RiskModel,
        mutable: set[str] | None = None,
        n_candidates_per_subset: int = 80,
        margin: float = 0.05,
        robust_frac: float = 0.90,
        probe_sd: float = 0.10,
        n_probes: int = 64,
        seed: int = RANDOM_SEED,
    ):
        self.model = model
        self.mutable = mutable or {
            n for n, f in FEATURE_BY_NAME.items() if f.agency != "fixed"
        }
        self.n_candidates_per_subset = n_candidates_per_subset
        self.margin = margin
        self.robust_frac = robust_frac
        self.probe_sd = probe_sd
        self.n_probes = n_probes
        self.seed = seed
        self._sd = model.background[FEATURE_NAMES].std(ddof=0).to_numpy()
        # Per-unit change cost: normalised so one cohort SD costs 1.0, with a
        # surcharge on changes that are clinically hard to achieve.
        self._difficulty = {
            "systolic_bp": 1.0,
            "glucose_level": 1.4,
            "bmi": 1.8,
            "smoker": 2.2,
            "alcohol": 1.3,
            "active": 1.5,
            "cholesterol_level": 3.0,
        }

    def _cost(self, x: np.ndarray, z: np.ndarray) -> float:
        total = 0.0
        for j, name in enumerate(FEATURE_NAMES):
            if name not in self.mutable:
                continue
            d = abs(z[j] - x[j]) / max(self._sd[j], 1e-9)
            total += self._difficulty.get(name, 1.0) * d
        return float(total)

    def _valid(self, probs: np.ndarray, want_below: bool) -> np.ndarray:
        t = self.model.threshold
        if want_below:
            return probs <= t - self.margin
        return probs >= t + self.margin

    def _sample_subset(self, x, subset, rng, want_below):
        """Candidate vectors varying only the features in ``subset``."""
        n = self.n_candidates_per_subset
        C = np.tile(x, (n, 1))
        # Features where a *higher* value lowers risk. Among the continuous
        # features (cholesterol_level, glucose_level, systolic_bp, bmi) none
        # are protective-when-higher. "active" IS protective, but it's binary
        # here so it's handled by the unconditional flip below, not this set.
        beneficial: set[str] = set()
        for j in subset:
            name = FEATURE_NAMES[j]
            feat = FEATURE_BY_NAME[name]
            if feat.kind == "binary":
                C[:, j] = 1.0 - C[:, j]
            else:
                # Magnitudes spread over 0.1 - 3 cohort SDs so both small and
                # large interventions are on the table.
                mag = rng.uniform(0.1, 3.0, n) * self._sd[j]
                helps_down = name in beneficial
                # want_below: reduce risk -> raise beneficial features, lower others
                direction = (1.0 if helps_down else -1.0) if want_below else (
                    -1.0 if helps_down else 1.0
                )
                C[:, j] = np.clip(C[:, j] + direction * mag, feat.lo, feat.hi)
        return C

    def _robustness(self, Z, subset, want_below, rng=None):
        """Fraction of a small ball around each candidate that stays valid.

        The margin alone does not defeat the knife-edge problem: a tree
        ensemble can have a genuine step, so a candidate can sit 5 percentage
        points clear of the threshold while being 2 mg/dL from the split that
        produced the drop.  Advice like that is unfollowable -- measurement
        noise alone would undo it.  So validity is evaluated over a
        neighbourhood rather than at a point, and a candidate must keep the
        decision on ``robust_frac`` of it.
        """
        n_probe = self.n_probes
        Z = np.atleast_2d(Z)
        probes = np.repeat(Z, n_probe, axis=0)
        # A fixed probe pattern, drawn once per instance. Re-drawing it on every
        # call would make the predicate a noisy coin flip: with 64 probes the
        # Monte Carlo error on a 0.90 acceptance rate is about 4 points, enough
        # for the same counterfactual to pass on acceptance and fail on report.
        noise_block = np.tile(self._probe_pattern, (len(Z), 1))
        for j in subset:
            feat = FEATURE_BY_NAME[FEATURE_NAMES[j]]
            if feat.kind == "binary":
                continue
            probes[:, j] = np.clip(
                probes[:, j] + noise_block[:, j] * self.probe_sd * self._sd[j],
                feat.lo,
                feat.hi,
            )
        p = self.model.predict_proba(pd.DataFrame(probes, columns=FEATURE_NAMES))
        ok = self._valid(p, want_below).reshape(len(Z), n_probe)
        return ok.mean(axis=1)

    def _is_robust(self, z, subset, want_below, rng) -> bool:
        return bool(self._robustness(z[None, :], subset, want_below, rng)[0]
                    >= self.robust_frac)

    def _shrink(self, x, z, subset, want_below, rng):
        """Pull each changed feature back toward x while robust validity holds."""
        z = z.copy()
        for j in subset:
            if abs(z[j] - x[j]) < 1e-12:
                continue
            if FEATURE_BY_NAME[FEATURE_NAMES[j]].kind == "binary":
                continue
            lo, hi = x[j], z[j]  # lo not robustly valid, hi robustly valid
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                trial = z.copy()
                trial[j] = mid
                if self._is_robust(trial, subset, want_below, rng):
                    hi = mid
                else:
                    lo = mid
            z[j] = hi
        return z

    def explain(self, x: np.ndarray, instance_id: int) -> Explanation:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed + instance_id)
        self._probe_pattern = np.random.default_rng(
            self.seed + 7919 * (instance_id + 1)
        ).standard_normal((self.n_probes, len(FEATURE_NAMES)))
        x = np.asarray(x, dtype=float)

        p0 = float(self.model.predict_proba(pd.DataFrame([x], columns=FEATURE_NAMES))[0])
        want_below = p0 >= self.model.threshold
        mut_idx = [j for j, n in enumerate(FEATURE_NAMES) if n in self.mutable]

        best_z = None
        best_subset = None
        for k in range(1, self.MAX_CARDINALITY + 1):
            subsets = list(combinations(mut_idx, k))
            blocks, owners = [], []
            for sub in subsets:
                blocks.append(self._sample_subset(x, sub, rng, want_below))
                owners.extend([sub] * self.n_candidates_per_subset)
            C = np.vstack(blocks)
            probs = self.model.predict_proba(pd.DataFrame(C, columns=FEATURE_NAMES))
            ok = self._valid(probs, want_below)
            if not ok.any():
                continue

            cand = C[ok]
            cand_subsets = [owners[i] for i in np.flatnonzero(ok)]
            costs = np.array([self._cost(x, z) for z in cand])

            # Cheapest first, then keep the first that survives the
            # robustness check. Capped so the probe cost stays bounded.
            for rank in np.argsort(costs)[: self.max_robustness_checks]:
                sub = cand_subsets[int(rank)]
                if self._is_robust(cand[int(rank)], sub, want_below, rng):
                    best_subset = sub
                    best_z = self._shrink(x, cand[int(rank)], sub, want_below, rng)
                    break
            if best_z is not None:
                break

        runtime = time.perf_counter() - t0

        if best_z is None:
            return Explanation(
                method=self.name,
                instance_id=instance_id,
                attributions={n: 0.0 for n in FEATURE_NAMES},
                runtime_s=runtime,
                prediction=p0,
                payload={
                    "valid": False,
                    "cardinality": None,
                    "text": (
                        f"No counterfactual clearing the decision threshold by "
                        f"{self.margin:.0%} was found by changing at most "
                        f"{self.MAX_CARDINALITY} actionable features."
                    ),
                },
            )

        z = best_z
        p_cf = float(self.model.predict_proba(pd.DataFrame([z], columns=FEATURE_NAMES))[0])

        attributions = {n: 0.0 for n in FEATURE_NAMES}
        changes = []
        for j, name in enumerate(FEATURE_NAMES):
            delta = z[j] - x[j]
            if abs(delta) <= 1e-9:
                continue
            attributions[name] = float(delta / max(self._sd[j], 1e-9))
            feat = FEATURE_BY_NAME[name]
            if feat.kind == "binary":
                changes.append(
                    f"{_lede(feat.label)} changes from "
                    f"{'yes' if x[j] else 'no'} to {'yes' if z[j] else 'no'}"
                )
            else:
                dp = 1
                changes.append(
                    f"{_lede(feat.label)} moves from {x[j]:.{dp}f} to "
                    f"{z[j]:.{dp}f} {feat.units}"
                )

        direction = "falls" if p_cf < p0 else "rises"
        joined = changes[0] if len(changes) == 1 else (
            " and ".join([", ".join(changes[:-1]), changes[-1]])
        )
        text = f"If {joined}, predicted risk {direction} from {p0:.0%} to {p_cf:.0%}."

        return Explanation(
            method=self.name,
            instance_id=instance_id,
            attributions=attributions,
            runtime_s=runtime,
            prediction=p0,
            payload={
                "valid": True,
                "cardinality": len(best_subset),
                "robustness": float(
                    self._robustness(z[None, :], best_subset, want_below, rng)[0]
                ),
                "cf_vector": z.tolist(),
                "cf_prediction": p_cf,
                "cost": self._cost(x, z),
                "text": text,
            },
        )


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------


class RuleExplainer:
    """Anchor-style local rule: the decision path of a shallow tree fitted to
    the neighbourhood of x, reported with its empirical precision and coverage.
    """

    name = "Rules"

    def __init__(self, model: RiskModel, max_depth: int = 4, n_samples: int = 800):
        self.model = model
        self.max_depth = max_depth
        self.n_samples = n_samples

    def explain(self, x: np.ndarray, instance_id: int) -> Explanation:
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=float)

        Z = local_neighbourhood(
            x, self.model.background, n_samples=self.n_samples, seed=RANDOM_SEED + instance_id
        )
        labels = self.model.predict(pd.DataFrame(Z, columns=FEATURE_NAMES))

        if len(np.unique(labels)) < 2:
            # Degenerate neighbourhood: the rule is simply "everything nearby
            # gets the same label".  Report it honestly rather than faking a split.
            runtime = time.perf_counter() - t0
            row = pd.DataFrame([x], columns=FEATURE_NAMES)
            return Explanation(
                method=self.name,
                instance_id=instance_id,
                attributions={n: 0.0 for n in FEATURE_NAMES},
                runtime_s=runtime,
                prediction=float(self.model.predict_proba(row)[0]),
                payload={
                    "precision": 1.0,
                    "coverage": 1.0,
                    "conditions": [],
                    "text": "Prediction is locally constant: no discriminating condition found.",
                },
            )

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth, min_samples_leaf=15, random_state=RANDOM_SEED
        )
        tree.fit(Z, labels)

        conditions, feature_depths = _decision_path(tree, x)
        mask = _apply_conditions(Z, conditions)
        own_label = int(self.model.predict(pd.DataFrame([x], columns=FEATURE_NAMES))[0])
        precision = float((labels[mask] == own_label).mean()) if mask.sum() else 0.0
        coverage = float(mask.mean())

        # Importance is depth-based: a condition tested earlier in the path
        # constrains a larger region and is therefore doing more work.
        attributions = {n: 0.0 for n in FEATURE_NAMES}
        for name, depth in feature_depths.items():
            attributions[name] = 1.0 / (1.0 + depth)

        runtime = time.perf_counter() - t0
        row = pd.DataFrame([x], columns=FEATURE_NAMES)
        text = (
            "IF " + " AND ".join(c["text"] for c in conditions)
            + f" THEN {'high' if own_label == 1 else 'low'} risk "
            f"(precision {precision:.0%}, covers {coverage:.0%} of the local region)."
        )
        return Explanation(
            method=self.name,
            instance_id=instance_id,
            attributions=attributions,
            runtime_s=runtime,
            prediction=float(self.model.predict_proba(row)[0]),
            payload={
                "precision": precision,
                "coverage": coverage,
                "conditions": conditions,
                "text": text,
            },
        )


def _decision_path(tree: DecisionTreeClassifier, x: np.ndarray):
    t = tree.tree_
    node, depth = 0, 0
    conditions, feature_depths = [], {}

    while t.children_left[node] != _tree.TREE_LEAF:
        j = int(t.feature[node])
        thr = float(t.threshold[node])
        name = FEATURE_NAMES[j]
        feat = FEATURE_BY_NAME[name]
        goes_left = x[j] <= thr
        if feat.kind == "binary":
            text = f"{feat.label} is {'no' if goes_left else 'yes'}"
        else:
            op = "<=" if goes_left else ">"
            text = f"{feat.label} {op} {thr:.1f} {feat.units}"
        conditions.append({"feature": name, "op": "<=" if goes_left else ">",
                           "threshold": thr, "text": text})
        feature_depths.setdefault(name, depth)
        node = t.children_left[node] if goes_left else t.children_right[node]
        depth += 1

    return conditions, feature_depths


def _apply_conditions(Z: np.ndarray, conditions) -> np.ndarray:
    mask = np.ones(len(Z), dtype=bool)
    for c in conditions:
        j = FEATURE_NAMES.index(c["feature"])
        if c["op"] == "<=":
            mask &= Z[:, j] <= c["threshold"]
        else:
            mask &= Z[:, j] > c["threshold"]
    return mask


def build_explainers(model: RiskModel) -> dict:
    return {
        "SHAP": ShapExplainer(model),
        "LIME": LimeExplainer(model),
        "Counterfactual": CounterfactualExplainer(model),
        "Rules": RuleExplainer(model),
    }
