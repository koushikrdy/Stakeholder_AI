
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .explanations import METHODS
from .stakeholders import CRITERIA

# Categorical slots 1-4, fixed order, assigned to methods once.
METHOD_COLOR = {
    "SHAP": "#2a78d6",
    "LIME": "#eb6834",
    "Counterfactual": "#1baf7a",
    "Rules": "#eda100",
}

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

STAKEHOLDER_LABEL = {
    "doctor": "Clinician",
    "patient": "Patient",
    "auditor": "Auditor",
    "model_centric": "Model-centric",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def selection_frequency(long: pd.DataFrame) -> pd.DataFrame:
    """Share of instances on which each method is the argmax, per stakeholder."""
    picks = long.loc[long.groupby(["instance_id", "stakeholder"])["utility"].idxmax()]
    tab = (
        picks.groupby(["stakeholder", "method"]).size().unstack(fill_value=0).reindex(
            columns=list(METHODS), fill_value=0
        )
    )
    return tab.div(tab.sum(axis=1), axis=0)


def mean_utility_table(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["stakeholder", "method"])["utility"]
        .mean()
        .unstack()
        .reindex(columns=list(METHODS))
    )


def property_profile(long: pd.DataFrame) -> pd.DataFrame:
    """Mean criterion values per method (stakeholder-invariant ones only)."""
    cols = [f"phi_{k}" for k in CRITERIA if k != "actionability"]
    prof = long.groupby("method")[cols].mean().reindex(list(METHODS))
    prof.columns = [c.replace("phi_", "") for c in prof.columns]
    return prof


def actionability_by_stakeholder(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long[long.stakeholder != "auditor"]
        .groupby(["stakeholder", "method"])["phi_actionability"]
        .mean()
        .unstack()
        .reindex(columns=list(METHODS))
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_selection_frequency(long: pd.DataFrame, path: Path):
    tab = selection_frequency(long)
    order = [s for s in ("doctor", "patient", "auditor") if s in tab.index]
    tab = tab.loc[order]

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=170)
    _style(ax)

    n_m = len(METHODS)
    width = 0.78 / n_m
    xs = np.arange(len(tab))

    for i, m in enumerate(METHODS):
        vals = tab[m].to_numpy()
        pos = xs - 0.39 + width * (i + 0.5)
        bars = ax.bar(
            pos, vals, width * 0.92, color=METHOD_COLOR[m], label=m,
            edgecolor=SURFACE, linewidth=2.0,
        )
        for b, v in zip(bars, vals):
            if v > 0.02:
                ax.text(
                    b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=8.5, color=INK,
                )

    ax.set_xticks(xs)
    ax.set_xticklabels([STAKEHOLDER_LABEL[s] for s in tab.index], fontsize=10, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of patients where the method is selected", fontsize=9.5, color=INK_SOFT)
    ax.set_title(
        "The same model, the same four explanations, three different winners",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    leg = ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center",
                    bbox_to_anchor=(0.5, -0.09))
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_utility_heatmap(long: pd.DataFrame, path: Path):
    tab = mean_utility_table(long)
    order = [s for s in ("doctor", "patient", "auditor") if s in tab.index]
    tab = tab.loc[order]

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=170)
    ax.set_facecolor(SURFACE)
    fig.set_facecolor(SURFACE)

    data = tab.to_numpy()
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))

    ax.set_xticks(range(len(tab.columns)))
    ax.set_xticklabels(tab.columns, fontsize=9.5, color=INK)
    ax.set_yticks(range(len(tab.index)))
    ax.set_yticklabels([STAKEHOLDER_LABEL[s] for s in tab.index], fontsize=10, color=INK)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    span = float(np.nanmax(data) - np.nanmin(data)) or 1.0
    for i in range(data.shape[0]):
        best = int(np.nanargmax(data[i]))
        for j in range(data.shape[1]):
            rel = (data[i, j] - np.nanmin(data)) / span
            colour = "#ffffff" if rel > 0.68 else INK
            label = f"{data[i, j]:.3f}"
            if j == best:
                label += "  *"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=colour)

    ax.set_title(
        "Mean stakeholder utility per method  (* = argmax)",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_property_profile(long: pd.DataFrame, path: Path):
    prof = property_profile(long)

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=170)
    _style(ax)

    crits = list(prof.columns)
    xs = np.arange(len(crits))
    width = 0.78 / len(METHODS)

    for i, m in enumerate(METHODS):
        vals = prof.loc[m].to_numpy()
        pos = xs - 0.39 + width * (i + 0.5)
        bars = ax.bar(pos, vals, width * 0.92, color=METHOD_COLOR[m], label=m,
                      edgecolor=SURFACE, linewidth=2.0)
        # Label the near-zero bars explicitly: a bar of zero height is
        # indistinguishable from a missing series otherwise.
        for b, v in zip(bars, vals):
            if v < 0.06:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.016, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK_SOFT)

    ax.set_xticks(xs)
    labels = [
        "Cost\n(lower is better)" if c == "cost" else c.capitalize()
        for c in crits
    ]
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean score (0-1)", fontsize=9.5, color=INK_SOFT)
    ax.set_title(
        "No method dominates: each criterion has a different best explainer",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    leg = ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center",
                    bbox_to_anchor=(0.5, -0.09))
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_regret(div: pd.DataFrame, path: Path):
    order = [s for s in ("doctor", "patient", "auditor") if s in set(div.stakeholder)]
    means = div.groupby("stakeholder")["regret"].mean().reindex(order)
    rates = div.groupby("stakeholder")["diverges"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=170)
    _style(ax)

    xs = np.arange(len(order))
    bars = ax.bar(xs, means.to_numpy(), 0.52, color="#2a78d6",
                  edgecolor=SURFACE, linewidth=2.0)
    for b, v, r in zip(bars, means.to_numpy(), rates.to_numpy()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                f"{v:.3f}\ndiverges on {r:.0%}", ha="center", va="bottom",
                fontsize=8.5, color=INK, linespacing=1.4)

    ax.set_xticks(xs)
    ax.set_xticklabels([STAKEHOLDER_LABEL[s] for s in order], fontsize=10, color=INK)
    ax.set_ylim(0, max(means.max() * 1.45, 0.02))
    ax.set_ylabel("Mean utility lost (stakeholder's own units)", fontsize=9.5, color=INK_SOFT)
    ax.set_title(
        "Cost of handing each stakeholder the model-centric explanation",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_sensitivity(sens: dict, path: Path):
    order = [s for s in ("doctor", "patient", "auditor") if s in sens]
    stability = [sens[s]["matches_baseline"].mean() for s in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=170)
    _style(ax)

    xs = np.arange(len(order))
    bars = ax.bar(xs, stability, 0.52, color="#2a78d6", edgecolor=SURFACE, linewidth=2.0)
    for b, v in zip(bars, stability):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.1%}",
                ha="center", va="bottom", fontsize=9, color=INK)

    ax.axhline(0.9, color="#52514e", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(-0.46, 0.912, "90% stability", ha="left", va="bottom",
            fontsize=8.5, color=INK_SOFT)

    ax.set_xticks(xs)
    ax.set_xticklabels([STAKEHOLDER_LABEL[s] for s in order], fontsize=10, color=INK)
    ax.set_ylim(0, 1.09)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Selections unchanged under reweighting", fontsize=9.5, color=INK_SOFT)
    ax.set_title(
        "Selections survive +/-15% log-normal jitter on the weight vector",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
