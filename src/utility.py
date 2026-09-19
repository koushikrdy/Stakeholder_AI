"""
The selection framework itself.

    E*_s = argmax_{E in eps} U_s(E | x, y-hat, a)

Everything upstream of this module exists to make the argmax well-posed: a
common explanation interface, a common property scale, and a stakeholder whose
downstream decision is stated.  What is left here is the argmax, plus the two
diagnostics that turn it into an argument rather than a mechanism:

  * **divergence** -- how often the model-centric winner is not the
    stakeholder-optimal one.  If this were near zero, the framework would be
    an elaborate way to agree with existing practice.

  * **regret** -- what the stakeholder loses, in their own utility units, when
    they are handed the model-centric winner instead of their own.  Divergence
    counts disagreements; regret says whether they cost anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .explanations import Explanation
from .properties import PropertyScores
from .stakeholders import CRITERIA, MODEL_CENTRIC, Stakeholder


@dataclass
class Selection:
    instance_id: int
    stakeholder: str
    chosen: str
    utility: float
    utilities: dict  # method -> utility
    runner_up: str
    margin: float


def criterion_vector(
    exp: Explanation, scores: PropertyScores, stakeholder: Stakeholder
) -> dict:
    """Assemble phi_k(E, x, y-hat, a) for one (explanation, stakeholder) pair.

    Only actionability depends on the stakeholder; the rest are intrinsic.
    """
    if stakeholder.action_set:
        actionability = exp.mass_on(stakeholder.action_set)
    else:
        # No levers from this seat. The criterion carries zero weight for such
        # stakeholders, so the value is never used -- recorded as 0 for clarity.
        actionability = 0.0

    return {
        "fidelity": scores.fidelity,
        "completeness": scores.completeness,
        "sparsity": scores.sparsity,
        "actionability": actionability,
        "cost": scores.cost,
    }


def select(
    explanations: list[Explanation],
    scores: dict,
    stakeholder: Stakeholder,
) -> Selection:
    """Evaluate every candidate and return the argmax for one stakeholder."""
    if not explanations:
        raise ValueError("empty explanation space")

    instance_id = explanations[0].instance_id
    utilities = {}
    for exp in explanations:
        phi = criterion_vector(exp, scores[(exp.instance_id, exp.method)], stakeholder)
        utilities[exp.method] = stakeholder.utility(phi)

    ordered = sorted(utilities.items(), key=lambda kv: kv[1], reverse=True)
    best, best_u = ordered[0]
    runner_up, runner_u = ordered[1] if len(ordered) > 1 else (best, best_u)

    return Selection(
        instance_id=instance_id,
        stakeholder=stakeholder.key,
        chosen=best,
        utility=float(best_u),
        utilities={k: float(v) for k, v in utilities.items()},
        runner_up=runner_up,
        margin=float(best_u - runner_u),
    )


def build_long_table(
    explanations_by_instance: dict,
    scores: dict,
    stakeholders: dict,
) -> pd.DataFrame:
    """One row per (instance, stakeholder, method) with criteria and utility."""
    rows = []
    for inst_id, exps in explanations_by_instance.items():
        for s in stakeholders.values():
            for exp in exps:
                sc = scores[(inst_id, exp.method)]
                phi = criterion_vector(exp, sc, s)
                rows.append(
                    {
                        "instance_id": inst_id,
                        "stakeholder": s.key,
                        "method": exp.method,
                        **{f"phi_{k}": phi[k] for k in CRITERIA},
                        "utility": s.utility(phi),
                        "n_cited": sc.n_cited,
                        "runtime_s": sc.raw_runtime_s,
                        "prediction": exp.prediction,
                        "degenerate_fidelity": sc.degenerate,
                    }
                )
    return pd.DataFrame(rows)


def divergence_and_regret(
    explanations_by_instance: dict,
    scores: dict,
    stakeholders: dict,
) -> pd.DataFrame:
    """Per (instance, stakeholder): does the model-centric pick differ, and at what cost?"""
    rows = []
    for inst_id, exps in explanations_by_instance.items():
        mc = select(exps, scores, MODEL_CENTRIC)
        for s in stakeholders.values():
            sel = select(exps, scores, s)
            regret = sel.utility - sel.utilities[mc.chosen]
            rows.append(
                {
                    "instance_id": inst_id,
                    "stakeholder": s.key,
                    "model_centric_choice": mc.chosen,
                    "stakeholder_choice": sel.chosen,
                    "diverges": sel.chosen != mc.chosen,
                    "stakeholder_utility": sel.utility,
                    "utility_of_model_centric_choice": sel.utilities[mc.chosen],
                    "regret": float(regret),
                    "margin_over_runner_up": sel.margin,
                }
            )
    return pd.DataFrame(rows)


def weight_sensitivity(
    explanations_by_instance: dict,
    scores: dict,
    stakeholder: Stakeholder,
    n_draws: int = 200,
    noise: float = 0.15,
    seed: int = 7,
) -> pd.DataFrame:
    """How stable is E*_s under perturbation of the weight vector?

    The weights are an expert judgement, not a measurement, so a selection that
    flips under small reweighting is not a finding.  Each draw perturbs the
    weights multiplicatively with log-normal noise, renormalises to unit L1,
    preserves signs, and re-runs the argmax.
    """
    rng = np.random.default_rng(seed)
    base = np.array([stakeholder.weights[k] for k in CRITERIA], dtype=float)
    signs = np.sign(base)
    mag = np.abs(base)

    rows = []
    for draw in range(n_draws):
        jitter = rng.lognormal(mean=0.0, sigma=noise, size=len(base))
        w = mag * jitter
        total = w.sum()
        w = w / total if total > 0 else mag
        perturbed = Stakeholder(
            key=stakeholder.key,
            name=stakeholder.name,
            decision=stakeholder.decision,
            action_set=stakeholder.action_set,
            weights={k: float(signs[i] * w[i]) for i, k in enumerate(CRITERIA)},
        )
        for inst_id, exps in explanations_by_instance.items():
            sel = select(exps, scores, perturbed)
            rows.append({"draw": draw, "instance_id": inst_id, "chosen": sel.chosen})

    df = pd.DataFrame(rows)
    baseline = {
        inst_id: select(exps, scores, stakeholder).chosen
        for inst_id, exps in explanations_by_instance.items()
    }
    df["matches_baseline"] = df.apply(
        lambda r: r["chosen"] == baseline[r["instance_id"]], axis=1
    )
    return df
