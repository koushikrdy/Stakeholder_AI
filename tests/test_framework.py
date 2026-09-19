"""
Tests for the properties the framework's claims depend on.

These are not smoke tests.  Each one pins down a statement the write-up makes,
so that if the statement stops being true the suite says so:

  * the criteria really are bounded in [0, 1], or the weighted sum is not a
    comparable utility across stakeholders;
  * a counterfactual really does cross the decision threshold, or the
    explanation is false rather than merely unhelpful;
  * sparsity, actionability and completeness really do order the methods the
    way the argument assumes;
  * the argmax really is an argmax;
  * the pipeline is deterministic under a fixed seed.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FEATURE_NAMES, generate_cohort, load_real_world_cohort, split_cohort
from src.explanations import METHODS, build_explainers
from src.model import train_risk_model
from src.properties import PropertyEvaluator
from src.stakeholders import CRITERIA, MODEL_CENTRIC, PATIENT, STAKEHOLDERS
from src.utility import criterion_vector, select

N_TEST_INSTANCES = 6


@pytest.fixture(scope="module")
def fitted():
    cohort = generate_cohort(1200, seed=11)
    X_tr, y_tr, X_te, y_te = split_cohort(cohort, seed=11)
    model = train_risk_model(X_tr, y_tr, seed=11)
    explainers = build_explainers(model)

    probs = model.predict_proba(X_te)
    order = np.argsort(probs)
    picks = np.linspace(0, len(order) - 1, N_TEST_INSTANCES).astype(int)
    idx = order[picks]

    instances = {int(i): X_te.iloc[int(i)][FEATURE_NAMES].to_numpy(float) for i in idx}
    by_instance, flat = {}, []
    for inst_id, x in instances.items():
        exps = [explainers[m].explain(x, inst_id) for m in METHODS]
        by_instance[inst_id] = exps
        flat.extend(exps)

    evaluator = PropertyEvaluator(model)
    scores = evaluator.score_all(flat, instances)
    return dict(
        model=model, instances=instances, by_instance=by_instance,
        flat=flat, scores=scores,
    )


# --------------------------------------------------------------------------
# Data and model
# --------------------------------------------------------------------------


def test_cohort_is_plausible():
    c = generate_cohort(2000, seed=3)
    assert 0.10 < c["_event"].mean() < 0.35, "event rate should be clinically plausible"
    assert c[FEATURE_NAMES].notna().all().all()
    # Correlations the generative process puts in on purpose.
    assert c["bmi"].corr(c["systolic_bp"]) > 0.3
    assert c["bmi"].corr(c["cholesterol_level"]) > 0.3  # both driven by the metabolic factor


def test_real_world_loader_preserves_the_feature_contract(tmp_path):
    source = generate_cohort(40, seed=3).drop(columns="_true_risk")
    source = source.rename(columns={"_event": "outcome"})
    path = tmp_path / "cohort.csv"
    source.to_csv(path, index=False)

    loaded = load_real_world_cohort(path, target="outcome")

    assert loaded[FEATURE_NAMES].columns.tolist() == FEATURE_NAMES
    assert loaded["_event"].isin([0, 1]).all()
    assert len(loaded) == 40


def test_threshold_flags_the_intended_fraction(fitted):
    model = fitted["model"]
    p = model.predict_proba(model.background)
    flagged = (p >= model.threshold).mean()
    assert 0.15 < flagged < 0.25, "threshold should flag roughly the top 20%"


def test_deleting_all_features_lands_on_the_base_rate(fitted):
    model = fitted["model"]
    row = pd.DataFrame([model.baseline_values], columns=FEATURE_NAMES)
    assert abs(float(model.predict_proba(row)[0]) - model.base_rate) < 1e-12


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------


def test_every_method_returns_an_explanation(fitted):
    for exps in fitted["by_instance"].values():
        assert {e.method for e in exps} == set(METHODS)


def test_counterfactuals_actually_cross_the_threshold(fitted):
    """A counterfactual that does not flip the decision is simply wrong."""
    model = fitted["model"]
    checked = 0
    for inst_id, exps in fitted["by_instance"].items():
        cf = next(e for e in exps if e.method == "Counterfactual")
        if not cf.payload.get("valid"):
            continue
        x = fitted["instances"][inst_id]
        p0 = float(model.predict_proba(pd.DataFrame([x], columns=FEATURE_NAMES))[0])
        z = np.asarray(cf.payload["cf_vector"], dtype=float)
        p1 = float(model.predict_proba(pd.DataFrame([z], columns=FEATURE_NAMES))[0])
        assert (p0 >= model.threshold) != (p1 >= model.threshold), (
            f"counterfactual for {inst_id} did not cross the decision boundary"
        )
        checked += 1
    assert checked > 0, "no valid counterfactual was produced at all"


def test_counterfactuals_are_minimal_cardinality(fitted):
    """The search must stop at the smallest subset that works."""
    for exps in fitted["by_instance"].values():
        cf = next(e for e in exps if e.method == "Counterfactual")
        if cf.payload.get("valid"):
            assert 1 <= cf.n_components <= 3


def test_counterfactuals_never_touch_immutable_features(fitted):
    """Age and sex cannot be prescribed."""
    for exps in fitted["by_instance"].values():
        cf = next(e for e in exps if e.method == "Counterfactual")
        assert "age" not in cf.cited_features
        assert "male" not in cf.cited_features


def test_shap_explains_every_feature(fitted):
    for exps in fitted["by_instance"].values():
        shap_exp = next(e for e in exps if e.method == "SHAP")
        assert shap_exp.n_components >= len(FEATURE_NAMES) - 1


def test_rules_are_short_and_honest(fitted):
    for exps in fitted["by_instance"].values():
        r = next(e for e in exps if e.method == "Rules")
        assert r.n_components <= 4, "rule depth is capped at 4"
        assert 0.0 <= r.payload["precision"] <= 1.0
        assert 0.0 < r.payload["coverage"] <= 1.0


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


def test_all_criteria_are_bounded(fitted):
    """Utilities are only comparable if every criterion shares the [0,1] scale."""
    for s in fitted["scores"].values():
        for field in ("fidelity", "completeness", "sparsity", "cost"):
            v = getattr(s, field)
            assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"{field} out of range: {v}"


def test_actionability_is_bounded_and_stakeholder_relative(fitted):
    seen_difference = False
    for inst_id, exps in fitted["by_instance"].items():
        for exp in exps:
            sc = fitted["scores"][(inst_id, exp.method)]
            vals = {}
            for key, s in STAKEHOLDERS.items():
                a = criterion_vector(exp, sc, s)["actionability"]
                assert 0.0 <= a <= 1.0
                vals[key] = a
            if abs(vals["doctor"] - vals["patient"]) > 1e-6:
                seen_difference = True
    assert seen_difference, (
        "actionability must differ by stakeholder, or it is not stakeholder-relative"
    )


def test_sparsity_ordering_matches_the_argument(fitted):
    """SHAP is the densest explanation; counterfactuals and rules are the sparsest."""
    import statistics as st

    by_method = {m: [] for m in METHODS}
    for (_, method), s in fitted["scores"].items():
        by_method[method].append(s.sparsity)
    mean = {m: st.mean(v) for m, v in by_method.items()}
    assert mean["SHAP"] < mean["LIME"] < mean["Counterfactual"]
    assert mean["SHAP"] < mean["Rules"]


def test_completeness_ordering_matches_the_argument(fitted):
    """The trade-off the paper rests on: sparse explanations explain less."""
    import statistics as st

    by_method = {m: [] for m in METHODS}
    for (_, method), s in fitted["scores"].items():
        by_method[method].append(s.completeness)
    mean = {m: st.mean(v) for m, v in by_method.items()}
    assert mean["SHAP"] > mean["Counterfactual"], (
        "if a counterfactual were as complete as SHAP there would be no trade-off"
    )


def test_fidelity_of_an_empty_explanation_is_zero(fitted):
    from src.explanations import Explanation

    evaluator = PropertyEvaluator(fitted["model"])
    inst_id = next(iter(fitted["instances"]))
    empty = Explanation(
        method="Rules", instance_id=inst_id,
        attributions={n: 0.0 for n in FEATURE_NAMES},
        runtime_s=0.01, prediction=0.5,
    )
    fid, _ = evaluator.fidelity(empty, fitted["instances"][inst_id])
    assert fid == 0.0
    assert evaluator.completeness(empty, fitted["instances"][inst_id]) == 0.0


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_weights_sum_to_one_in_absolute_value():
    for s in list(STAKEHOLDERS.values()) + [MODEL_CENTRIC]:
        total = sum(abs(w) for w in s.weights.values())
        assert abs(total - 1.0) < 1e-9, f"{s.key} weights sum to {total}"
        assert set(s.weights) == set(CRITERIA)
        assert s.weights["cost"] <= 0.0, "cost must be a penalty"


def test_select_returns_the_true_argmax(fitted):
    for exps in fitted["by_instance"].values():
        for s in STAKEHOLDERS.values():
            sel = select(exps, fitted["scores"], s)
            assert sel.chosen == max(sel.utilities, key=sel.utilities.get)
            assert sel.margin >= -1e-12


def test_auditor_ignores_actionability(fitted):
    """The auditor's decision is about the model, so the criterion must not bite."""
    from src.stakeholders import AUDITOR

    for inst_id, exps in fitted["by_instance"].items():
        for exp in exps:
            sc = fitted["scores"][(inst_id, exp.method)]
            phi = criterion_vector(exp, sc, AUDITOR)
            assert AUDITOR.weights["actionability"] * phi["actionability"] == 0.0


def test_patient_prefers_actionable_and_sparse_explanations(fitted):
    """The headline claim, as a test: patients should not be handed SHAP."""
    picks = [
        select(exps, fitted["scores"], PATIENT).chosen
        for exps in fitted["by_instance"].values()
    ]
    assert picks.count("SHAP") < len(picks) / 2


def test_pipeline_is_deterministic():
    """Same seed, same numbers -- otherwise none of the reported figures mean anything."""
    def run():
        cohort = generate_cohort(600, seed=5)
        X_tr, y_tr, X_te, _ = split_cohort(cohort, seed=5)
        model = train_risk_model(X_tr, y_tr, seed=5)
        ex = build_explainers(model)
        x = X_te.iloc[0][FEATURE_NAMES].to_numpy(float)
        return [ex[m].explain(x, 0) for m in METHODS]

    a, b = run(), run()
    for ea, eb in zip(a, b):
        assert ea.method == eb.method
        for name in FEATURE_NAMES:
            assert ea.attributions[name] == pytest.approx(eb.attributions[name], abs=1e-9)


def test_counterfactuals_are_robust_not_knife_edge(fitted):
    """A counterfactual must survive small perturbations of its own values.

    Without this, the search returns advice like "reduce LDL by 1.8 mg/dL" --
    valid against the model, useless against reality, because it is sitting on
    a split point rather than in a region.
    """
    for exps in fitted["by_instance"].values():
        cf = next(e for e in exps if e.method == "Counterfactual")
        if cf.payload.get("valid"):
            assert cf.payload["robustness"] >= 0.90 - 1e-9, (
                f"knife-edge counterfactual: only {cf.payload['robustness']:.0%} "
                f"of its neighbourhood keeps the decision"
            )


def test_counterfactuals_clear_the_threshold_by_the_margin(fitted):
    model = fitted["model"]
    for inst_id, exps in fitted["by_instance"].items():
        cf = next(e for e in exps if e.method == "Counterfactual")
        if not cf.payload.get("valid"):
            continue
        p1 = cf.payload["cf_prediction"]
        p0 = cf.prediction
        if p0 >= model.threshold:
            assert p1 <= model.threshold - 0.05 + 1e-9
        else:
            assert p1 >= model.threshold + 0.05 - 1e-9


def test_cost_is_reproducible_not_wall_clock(fitted):
    """Cost must not carry machine load into the results.

    Scoring raw wall-clock time made identical runs disagree in the third
    decimal. Cost is a rank over median runtimes instead, so every explanation
    from one method shares a cost and the spacing is fixed.
    """
    by_method = {}
    for (_, method), s in fitted["scores"].items():
        by_method.setdefault(method, set()).add(round(s.cost, 12))
    for method, vals in by_method.items():
        assert len(vals) == 1, f"{method} has instance-varying cost: {vals}"
    all_costs = sorted(v.pop() for v in by_method.values())
    assert all_costs == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_cost_ranking_matches_the_known_ordering(fitted):
    """Exact TreeSHAP is the cheapest arm; LIME's 2000-sample surrogate the dearest."""
    cost = {m: s.cost for (_, m), s in fitted["scores"].items()}
    assert cost["SHAP"] == 0.0
    assert cost["LIME"] == 1.0
