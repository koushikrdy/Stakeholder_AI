"""
Stakeholders, their downstream decisions, and their utility weights.

The abstract's utility is written U_s(E | x, y-hat, a): it is indexed by the
stakeholder *and* conditioned on a downstream decision.  Those two things are
bundled here, because in practice a stakeholder is only well-defined once you
say what they are about to do with the explanation.

Two modelling choices are worth defending:

1.  ``action_set`` is the set of features the stakeholder can actually move.
    It differs by role: a clinician titrates blood pressure and lipids, a
    patient changes behaviour, and an auditor changes nothing about the
    patient at all.  Actionability is measured against this set, which is why
    it is a stakeholder-relative property rather than an intrinsic one.

2.  The auditor's action set is deliberately empty and their actionability
    weight is zero.  Their decision -- approve the model or flag it for review
    -- is about the model, not the patient.  Rather than inventing a fake
    actionability score for them, the criterion drops out of their utility.
    A framework that claims to be stakeholder-specific ought to allow a
    criterion to be irrelevant to someone.

Weights are on a common scale: the absolute values sum to 1 for every
stakeholder, so utilities are comparable across roles.  Cost carries a
negative weight; everything else is a benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import agency_sets

_AGENCY = agency_sets()

CRITERIA = ("fidelity", "completeness", "sparsity", "actionability", "cost")


@dataclass(frozen=True)
class Stakeholder:
    key: str
    name: str
    decision: str  # the downstream action `a`
    action_set: frozenset
    weights: dict = field(default_factory=dict)
    rationale: str = ""

    def utility(self, scores: dict) -> float:
        """U_s(E | x, y-hat, a) = sum_k w_{s,k} * phi_k(E)."""
        return float(sum(self.weights[k] * scores[k] for k in CRITERIA))

    def weight_vector(self) -> list[float]:
        return [self.weights[k] for k in CRITERIA]


DOCTOR = Stakeholder(
    key="doctor",
    name="Treating clinician",
    decision="adjust this patient's treatment plan",
    action_set=frozenset(_AGENCY.get("clinical", set()) | _AGENCY.get("both", set())),
    weights={
        "fidelity": 0.30,
        "completeness": 0.15,
        "sparsity": 0.15,
        "actionability": 0.30,
        "cost": -0.10,
    },
    rationale=(
        "Carries clinical liability, so needs the explanation to be faithful, but "
        "acts through a narrow set of prescribable levers and has roughly twelve "
        "minutes of consultation time."
    ),
)

PATIENT = Stakeholder(
    key="patient",
    name="Patient",
    decision="decide what to change in daily behaviour",
    action_set=frozenset(_AGENCY.get("lifestyle", set()) | _AGENCY.get("both", set())),
    weights={
        "fidelity": 0.10,
        "completeness": 0.05,
        "sparsity": 0.35,
        "actionability": 0.40,
        "cost": -0.10,
    },
    rationale=(
        "Needs to walk away with something doable. A complete attribution over ten "
        "features is worse than useless here; two concrete changes beat it."
    ),
)

AUDITOR = Stakeholder(
    key="auditor",
    name="Regulatory auditor",
    decision="approve the model or flag it for review",
    action_set=frozenset(),
    weights={
        "fidelity": 0.40,
        "completeness": 0.40,
        "sparsity": 0.05,
        "actionability": 0.00,
        "cost": -0.15,
    },
    rationale=(
        "Assesses the model, not the patient. Needs an account of the whole "
        "prediction that is demonstrably faithful; brevity is nearly irrelevant "
        "and nothing about the patient is actionable from this seat."
    ),
)

STAKEHOLDERS: dict[str, Stakeholder] = {
    s.key: s for s in (DOCTOR, PATIENT, AUDITOR)
}


# The status-quo criterion the abstract argues against: explanation quality
# judged purely on how well it tracks the model, with no reference to who is
# reading it or what they will do next.
MODEL_CENTRIC = Stakeholder(
    key="model_centric",
    name="Model-centric baseline",
    decision="none (evaluation criterion, not a decision)",
    action_set=frozenset(),
    weights={
        "fidelity": 0.50,
        "completeness": 0.50,
        "sparsity": 0.00,
        "actionability": 0.00,
        "cost": 0.00,
    },
    rationale="Fidelity and completeness only -- the criterion this work is measured against.",
)
