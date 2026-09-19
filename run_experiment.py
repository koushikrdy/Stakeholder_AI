#!/usr/bin/env python3
"""
End-to-end experiment for

    "Model Fidelity to Decision Utility: Decision-Theoretic Foundations for
     Stakeholder-Specific Explanation Selection in XAI"

Pipeline
--------
  1. generate a synthetic clinical cohort with a known generative process
  2. train and evaluate the risk model that will be explained
  3. for each held-out patient, produce all four candidate explanations
  4. score every explanation on the five criteria
  5. run the argmax for each stakeholder, and for the model-centric baseline
  6. measure divergence, regret, and stability under reweighting
  7. write tables to results/ and figures to figures/

Usage
-----
    python run_experiment.py                 # default: 60 patients
    python run_experiment.py --n-explain 25  # quicker run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import analysis
from src.data import (
    FEATURE_NAMES,
    RANDOM_SEED,
    generate_cohort,
    load_real_world_cohort,
    split_cohort,
)
from src.explanations import METHODS, build_explainers
from src.model import calibration_table, evaluate, train_risk_model
from src.properties import PropertyEvaluator
from src.stakeholders import CRITERIA, MODEL_CENTRIC, STAKEHOLDERS
from src.utility import (
    build_long_table,
    divergence_and_regret,
    select,
    weight_sensitivity,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patients", type=int, default=40000)
    ap.add_argument("--n-explain", type=int, default=600,
                    help="held-out patients to explain (the expensive loop)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--sensitivity-draws", type=int, default=200)
    ap.add_argument(
        "--data-csv",
        type=Path,
        help="real-world CSV with the feature columns and a binary outcome column",
    )
    ap.add_argument(
        "--target-column",
        default="_event",
        help="outcome column in --data-csv (default: _event)",
    )
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    # ---------------------------------------------------------------- 1. data
    print("[1/7] loading cohort ...")
    if args.data_csv:
        cohort = load_real_world_cohort(args.data_csv, target=args.target_column)
        print(f"      source {args.data_csv}")
    else:
        cohort = generate_cohort(args.n_patients, seed=args.seed)
        print("      source synthetic cohort")
    X_tr, y_tr, X_te, y_te = split_cohort(cohort, seed=args.seed)
    print(f"      train {len(X_tr)}  test {len(X_te)}  event rate {cohort['_event'].mean():.1%}")

    # --------------------------------------------------------------- 2. model
    print("[2/7] training risk model ...")
    model = train_risk_model(X_tr, y_tr, seed=args.seed)
    metrics = evaluate(model, X_te, y_te)
    calib = calibration_table(model, X_te, y_te)
    print(f"      AUC {metrics['auc']:.3f}   Brier {metrics['brier']:.3f}   "
          f"acc {metrics['accuracy']:.3f}")

    # ------------------------------------------------- 3. candidate explanations
    print("[3/7] building explanation space ...")
    explainers = build_explainers(model)

    rng = np.random.default_rng(args.seed + 99)
    probs_te = model.predict_proba(X_te)
    # Stratify the explained sample across the risk spectrum so the results are
    # not driven by one end of the distribution.
    order = np.argsort(probs_te)
    picks = np.linspace(0, len(order) - 1, args.n_explain).astype(int)
    chosen_idx = order[picks]
    rng.shuffle(chosen_idx)

    instances = {int(i): X_te.iloc[int(i)][FEATURE_NAMES].to_numpy(dtype=float)
                 for i in chosen_idx}

    explanations_by_instance: dict[int, list] = {}
    flat: list = []
    for n, (inst_id, x) in enumerate(instances.items(), 1):
        exps = [explainers[m].explain(x, inst_id) for m in METHODS]
        explanations_by_instance[inst_id] = exps
        flat.extend(exps)
        if n % 10 == 0 or n == len(instances):
            print(f"      explained {n}/{len(instances)} patients")

    # ----------------------------------------------------------- 4. properties
    print("[4/7] scoring explanation properties ...")
    evaluator = PropertyEvaluator(model)
    scores = evaluator.score_all(flat, instances)

    n_degenerate = sum(1 for s in scores.values() if s.degenerate)
    if n_degenerate:
        print(f"      note: {n_degenerate} explanation(s) sat in a locally flat "
              f"region; fidelity undefined there")

    # ------------------------------------------------------------ 5. selection
    print("[5/7] running argmax selection ...")
    long = build_long_table(explanations_by_instance, scores, STAKEHOLDERS)

    selections = []
    for inst_id, exps in explanations_by_instance.items():
        for s in list(STAKEHOLDERS.values()) + [MODEL_CENTRIC]:
            sel = select(exps, scores, s)
            selections.append(
                {
                    "instance_id": inst_id,
                    "stakeholder": sel.stakeholder,
                    "chosen": sel.chosen,
                    "utility": sel.utility,
                    "runner_up": sel.runner_up,
                    "margin": sel.margin,
                    **{f"U_{m}": sel.utilities[m] for m in METHODS},
                }
            )
    selections = pd.DataFrame(selections)

    # ------------------------------------------------- 6. divergence / stability
    print("[6/7] measuring divergence, regret and stability ...")
    div = divergence_and_regret(explanations_by_instance, scores, STAKEHOLDERS)

    sens = {}
    for key, s in STAKEHOLDERS.items():
        sens[key] = weight_sensitivity(
            explanations_by_instance, scores, s, n_draws=args.sensitivity_draws
        )

    # ---------------------------------------------------------- 7. write output
    print("[7/7] writing tables and figures ...")
    freq = analysis.selection_frequency(long)
    util = analysis.mean_utility_table(long)
    prof = analysis.property_profile(long)
    act = analysis.actionability_by_stakeholder(long)

    long.to_csv(RESULTS / "criteria_long.csv", index=False)
    selections.to_csv(RESULTS / "selections.csv", index=False)
    div.to_csv(RESULTS / "divergence_regret.csv", index=False)
    freq.to_csv(RESULTS / "selection_frequency.csv")
    util.to_csv(RESULTS / "mean_utility.csv")
    prof.to_csv(RESULTS / "property_profile.csv")
    act.to_csv(RESULTS / "actionability.csv")
    calib.to_csv(RESULTS / "calibration.csv", index=False)

    analysis.fig_selection_frequency(long, FIGURES / "selection_frequency.png")
    analysis.fig_utility_heatmap(long, FIGURES / "utility_heatmap.png")
    analysis.fig_property_profile(long, FIGURES / "property_profile.png")
    analysis.fig_regret(div, FIGURES / "regret.png")
    analysis.fig_sensitivity(sens, FIGURES / "weight_sensitivity.png")

    # Does the disagreement concentrate where decisions are actually close?
    preds = {i: exps[0].prediction for i, exps in explanations_by_instance.items()}
    band = {i: abs(p - model.threshold) for i, p in preds.items()}
    cut = float(np.median(list(band.values())))
    div = div.copy()
    div["near_threshold"] = div["instance_id"].map(lambda i: band[i] <= cut)
    near_div = (
        div.groupby(["stakeholder", "near_threshold"])["diverges"]
        .mean().round(4).unstack()
        .rename(columns={True: "near_threshold", False: "far_from_threshold"})
        .to_dict(orient="index")
    )

    summary = {
        "config": {
            "n_patients": args.n_patients,
            "n_explained": len(instances),
            "seed": args.seed,
            "sensitivity_draws": args.sensitivity_draws,
        },
        "model": metrics,
        "explanation_space": list(METHODS),
        "criteria": list(CRITERIA),
        "selection_frequency": freq.round(4).to_dict(orient="index"),
        "mean_utility": util.round(4).to_dict(orient="index"),
        "divergence_rate": div.groupby("stakeholder")["diverges"].mean().round(4).to_dict(),
        "mean_regret": div.groupby("stakeholder")["regret"].mean().round(4).to_dict(),
        "max_regret": div.groupby("stakeholder")["regret"].max().round(4).to_dict(),
        "weight_stability": {k: float(v["matches_baseline"].mean()) for k, v in sens.items()},
        "mean_runtime_s": long.groupby("method")["runtime_s"].mean().round(4).to_dict(),
        "degenerate_fidelity_cases": int(n_degenerate),
        "divergence_near_threshold": near_div,
        "wall_clock_s": round(time.perf_counter() - t_start, 1),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    _print_report(summary, freq, util, prof, div, model, instances,
                  explanations_by_instance)


def _print_report(summary, freq, util, prof, div, model, instances, exps_by_inst):
    line = "=" * 74
    print(f"\n{line}\nRESULTS\n{line}")

    print("\nModel under explanation")
    m = summary["model"]
    print(f"  AUC {m['auc']:.3f} | Brier {m['brier']:.3f} | accuracy {m['accuracy']:.3f}")
    print(f"  decision threshold {m['decision_threshold']:.3f} "
          f"-> flags {m['flag_rate']:.1%} of patients | "
          f"precision {m['precision_at_threshold']:.2f} | "
          f"recall {m['recall_at_threshold']:.2f}")

    print("\nSelection frequency  (share of patients where the method wins)")
    print(freq.round(3).to_string())

    print("\nMean utility U_s per method")
    print(util.round(3).to_string())

    print("\nIntrinsic criterion profile per method")
    print(prof.round(3).to_string())

    print("\nDivergence from the model-centric choice")
    for k in summary["divergence_rate"]:
        print(f"  {k:<10} diverges on {summary['divergence_rate'][k]:.1%} of patients | "
              f"mean regret {summary['mean_regret'][k]:.4f} | "
              f"max regret {summary['max_regret'][k]:.4f}")

    print("\nDivergence split by distance from the decision threshold")
    for k, v in summary["divergence_near_threshold"].items():
        near = v.get("near_threshold", float("nan"))
        far = v.get("far_from_threshold", float("nan"))
        print(f"  {k:<10} near the threshold {near:.1%} | far from it {far:.1%}")

    print("\nStability of E*_s under +/-15% weight jitter")
    for k, v in summary["weight_stability"].items():
        print(f"  {k:<10} {v:.1%} of selections unchanged")

    # Worked example: the patient sitting closest to the decision threshold.
    # That is where the explanation actually has to carry a decision -- deep in
    # either tail every method agrees and nothing is at stake.
    inst_id = min(
        instances, key=lambda i: abs(exps_by_inst[i][0].prediction - model.threshold)
    )
    pred = exps_by_inst[inst_id][0].prediction
    print(f"\n{line}\nWORKED EXAMPLE -- patient nearest the decision threshold "
          f"(id {inst_id}, risk {pred:.1%} vs threshold {model.threshold:.1%})\n{line}")
    for e in exps_by_inst[inst_id]:
        print(f"\n  [{e.method}]  predicted 5-year risk {e.prediction:.1%}")
        print(f"    {e.describe()}")

    print(f"\n  Selected for each stakeholder:")
    row = div[div.instance_id == inst_id]
    for _, r in row.iterrows():
        mark = "differs from model-centric" if r["diverges"] else "same as model-centric"
        print(f"    {r['stakeholder']:<10} -> {r['stakeholder_choice']:<15} ({mark})")

    print(f"\nWall clock: {summary['wall_clock_s']}s")
    print(f"Tables in results/  |  Figures in figures/\n")


if __name__ == "__main__":
    main()
