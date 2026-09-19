# Model Fidelity to Decision Utility

Implementation of the decision-theoretic framework in the term paper
*"From Model Fidelity to Decision Utility: A Decision-Theoretic Framework for
Stakeholder-Specific Explanation Selection in XAI"* (Tiparthi Koushik Reddy,
1863216, Universität Trier).

The framework picks, for a given prediction and a given stakeholder, the
explanation that maximises that stakeholder's utility:

```
E*_s = argmax_{E in eps} U_s(E | x, y-hat, a)

  eps  = {SHAP, LIME, Counterfactual, Rules}     candidate explanation space
  U_s  = stakeholder-specific utility
  x    = the patient's feature vector
  y-hat= the model's prediction
  a    = the downstream decision the stakeholder is about to take
```

---

## Data

Two cohorts are supported.

**Real (used for the paper).** The public Cardiovascular Disease dataset,
70,000 examinations, 69,598 after removing biologically implausible entries.
`cardio_train.csv` is the raw file; `cohort_ready.csv` is the prepared version.

```bash
python scripts/prepare_cardio_train.py cardio_train.csv cohort_ready.csv
python run_experiment.py --data-csv cohort_ready.csv
```

**Synthetic (a controlled pilot).** A generated cohort with a known generative
process, useful for sanity checks because the ground truth is known.

```bash
python run_experiment.py        # no --data-csv
```

Note the outcome in the real cohort is **prevalent** cardiovascular disease at
examination, not an incident event within a horizon. Nothing produced here is a
5- or 10-year risk estimate.

---

## What the experiment shows (real cohort)

A risk model is trained on 70% of the cohort, then 60 held-out patients spread
across the predicted-risk range each receive all four explanations. Every
explanation is scored on five criteria, and the argmax is run for three
stakeholders plus a **model-centric baseline** scoring fidelity and completeness
alone, the way current practice does.

Model: AUC 0.798, Brier 0.182, accuracy 0.642 on 20,879 held-out patients.
Decision threshold at the 80th percentile of predicted risk (tau = 0.822),
flagging 20.5%.

**Selection frequency** — share of the 60 patients where each method wins:

| Stakeholder | SHAP | LIME | Counterfactual | Rules |
|---|---|---|---|---|
| Clinician | **42%** | 2% | 35% | 22% |
| Patient | 3% | 2% | **55%** | 40% |
| Auditor | **82%** | 0% | 5% | 13% |

**Divergence and regret** — how often the model-centric winner is *not* the
stakeholder-optimal one, and what that costs:

| Stakeholder | Diverges on | Mean regret | Max regret |
|---|---|---|---|
| Clinician | 35% | 0.028 | 0.221 |
| Patient | 75% | 0.174 | 0.440 |
| Auditor | 12% | 0.004 | 0.057 |

Regret is in the stakeholder's own utility units, where the attainable range is
about 0.90. The auditor is a negative control: their decision is about the
model, so the framework agrees with current practice 88% of the time at
near-zero regret, exactly as it should.

**A negative result worth reading.** On this cohort actionability barely
separates the clinician from the patient — averaged over the four methods the
gap is only 0.06, against 0.27 on the synthetic pilot. Of nine features only two
(age, sex) are fixed, so almost everything is actionable by somebody and most of
it by both. The divergence above is real but is driven mainly by **sparsity**,
not by the action set. Testing the action-set mechanism properly needs a cohort
where the stakeholders command disjoint levers.

---

## Reproducibility

Seeded and deterministic **on a fixed machine**: repeated runs produce identical
output. Across machines it is not. Floating-point differences move the AUC in
the fifth decimal, which shifts the decision threshold, which reclassifies a few
patients near it. The `results/` and `figures/` committed here were regenerated
together with the paper, so the two agree exactly. An earlier run of identical
code and data on a different machine gave a patient divergence of 80.0% against
the 75.0% here. The direction and size of every effect are stable; the third
significant figure is not.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py --data-csv cohort_ready.csv
.venv/bin/python -m pytest tests/ -q          # 24 tests
```

---

## Layout

```
src/data.py           feature schema, synthetic generator, real-cohort loader
src/model.py          gradient-boosted risk model, decision threshold
src/explanations.py   the four explainers behind one interface
src/properties.py     the five criteria on a common [0,1] scale
src/stakeholders.py   action sets and weight vectors
src/utility.py        the argmax, divergence, regret, weight sensitivity
src/analysis.py       tables and figures
run_experiment.py     the pipeline
scripts/              one-off preprocessing for the raw Kaggle CSV
tests/                24 tests pinning the claims above
results/              CSVs + summary.json
figures/              five PNGs
```

`results/criteria_long.csv` is the main artefact: one row per
(patient x stakeholder x method) with all five criterion values and the
resulting utility. Everything else is an aggregate of it.
