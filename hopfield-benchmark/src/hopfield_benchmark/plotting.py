"""Component f: plots that consume only standardized result tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runner import validate_paired_results


MODEL_LABELS = {
    "classical_hebb": "Classical Hopfield",
    "polynomial_dam_d3": "Polynomial DAM (d=3)",
}


def _label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


def _binary_curve(
    results: pd.DataFrame,
    x: str,
    fixed: dict[str, int | float | str],
) -> pd.DataFrame:
    subset = results.copy()
    for column, value in fixed.items():
        subset = subset[subset[column] == value]
    validate_paired_results(subset)
    grouped = (
        subset.groupby(["model_id", x], as_index=False)["exact_recall"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["se"] = np.sqrt(
        grouped["mean"] * (1.0 - grouped["mean"]) / grouped["count"]
    )
    grouped["lower"] = np.clip(grouped["mean"] - 1.96 * grouped["se"], 0.0, 1.0)
    grouped["upper"] = np.clip(grouped["mean"] + 1.96 * grouped["se"], 0.0, 1.0)
    return grouped


def f1_plot_recall_vs_corruption(
    results: pd.DataFrame,
    *,
    N: int,
    P: int,
) -> tuple[plt.Figure, plt.Axes]:
    curve = _binary_curve(results, "corruption_level", {"N": N, "P": P})
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for model_id, group in curve.groupby("model_id", sort=False):
        ordered = group.sort_values("corruption_level")
        axis.plot(
            ordered["corruption_level"],
            ordered["mean"],
            marker="o",
            label=_label(model_id),
        )
        axis.fill_between(
            ordered["corruption_level"],
            ordered["lower"],
            ordered["upper"],
            alpha=0.18,
        )
    axis.set(
        xlabel="Hamming corruption ratio",
        ylabel="Exact recall rate",
        ylim=(-0.03, 1.03),
    )
    axis.set_title(f"U2 noise recovery | N={N}, P={P} | native budget")
    axis.legend()
    fig.tight_layout()
    return fig, axis


def f2_plot_recall_vs_load(
    results: pd.DataFrame,
    *,
    N: int,
    corruption_level: float,
) -> tuple[plt.Figure, plt.Axes]:
    curve = _binary_curve(
        results,
        "P",
        {"N": N, "corruption_level": corruption_level},
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for model_id, group in curve.groupby("model_id", sort=False):
        ordered = group.sort_values("P")
        axis.plot(
            ordered["P"], ordered["mean"], marker="o", label=_label(model_id)
        )
        axis.fill_between(
            ordered["P"], ordered["lower"], ordered["upper"], alpha=0.18
        )
    axis.set(
        xlabel="Stored patterns P",
        ylabel="Exact recall rate",
        ylim=(-0.03, 1.03),
    )
    axis.set_xscale("log", base=2)
    axis.set_title(
        f"U3 finite-load curve | N={N}, corruption={corruption_level:.2f} | native budget"
    )
    axis.legend()
    fig.tight_layout()
    return fig, axis


def f3_plot_capacity_scaling(
    summary: pd.DataFrame,
) -> tuple[plt.Figure, plt.Axes]:
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for model_id, group in summary.groupby("model_id", sort=False):
        ordered = group.sort_values("N")
        axis.plot(ordered["N"], ordered["P_c"], marker="o", label=_label(model_id))
        censored = ordered[ordered["right_censored"]]
        axis.scatter(
            censored["N"],
            censored["P_c"],
            marker="^",
            s=90,
            facecolors="none",
            edgecolors="black",
            label=f"{_label(model_id)} right-censored" if not censored.empty else None,
        )
        left_censored = ordered[ordered["left_censored"]]
        axis.scatter(
            left_censored["N"],
            left_censored["P_c"],
            marker="v",
            s=90,
            facecolors="none",
            edgecolors="black",
            label=(
                f"{_label(model_id)} left-censored"
                if not left_censored.empty
                else None
            ),
        )
    axis.set(xlabel="State dimension N", ylabel="Empirical critical capacity P_c")
    axis.set_yscale("log", base=2)
    axis.set_title("U3 capacity scaling | identical success criterion")
    axis.legend()
    fig.tight_layout()
    return fig, axis


def f4_plot_quality_cost(
    results: pd.DataFrame,
    *,
    N: int,
    corruption_level: float,
) -> tuple[plt.Figure, plt.Axes]:
    subset = results[
        (results["N"] == N)
        & (results["corruption_level"] == corruption_level)
        & results["retrieval_flops"].notna()
    ]
    validate_paired_results(subset)
    grouped = subset.groupby("model_id", as_index=False).agg(
        exact_recall=("exact_recall", "mean"),
        retrieval_flops=("retrieval_flops", "mean"),
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for row in grouped.itertuples(index=False):
        axis.scatter(row.retrieval_flops, row.exact_recall, s=80)
        axis.annotate(
            _label(row.model_id),
            (row.retrieval_flops, row.exact_recall),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set(
        xlabel="Approximate retrieval FLOPs",
        ylabel="Exact recall rate",
        ylim=(-0.03, 1.03),
    )
    axis.set_xscale("log")
    axis.set_title(f"U7 quality-cost view | N={N}, corruption={corruption_level:.2f}")
    fig.tight_layout()
    return fig, axis


def f5_plot_error_dynamics(
    results: pd.DataFrame,
    *,
    run_id: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay a paired trial's common error metric, not model-specific energy."""

    subset = results[results["run_id"] == run_id]
    validate_paired_results(subset)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for row in subset.itertuples(index=False):
        trace = list(row.error_trace)
        axis.plot(
            range(len(trace)),
            trace,
            marker="o",
            label=_label(row.model_id),
        )
    axis.set(
        xlabel="Completed asynchronous sweeps",
        ylabel="Bit error ratio",
        ylim=(-0.03, 1.03),
    )
    axis.set_title(f"U5 paired dynamics | run_id={run_id}")
    axis.legend()
    fig.tight_layout()
    return fig, axis
