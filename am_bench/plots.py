"""f：读取结果作图，不运行模型。"""
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .runner import validate_paired_results

# [展示] 颜色只绑定 model_id，预算与动力学差异写进标题/表格
MODEL_LABELS = {
    "classical_hebb": "Classical Hopfield",
    "polynomial_dam_d3": "Polynomial DAM (d=3)",
    "exponential_dam": "Exponential DAM",
    "simplicial_r12_t50": "Simplicial R12 (50% triangles)",
    "pshn_k8": "PSHN (k=8)",
    "continuous_modern": "Continuous Modern Hopfield",
}


# [汇总] 先在 pattern set 内平均 target，再把 pattern set 当独立重复计算区间
def curve_table(
    frame: pd.DataFrame,
    x: str,
    fixed: dict[str, Any],
    metric: str = "top1_correct",
) -> pd.DataFrame:
    subset = frame.copy()
    for column, value in fixed.items():
        subset = subset[subset[column] == value]
    # [判断] 聚合之前再次验证配对，防止筛选条件误删某条模型线
    validate_paired_results(subset)
    # [中介变量] replicate_rates 每行对应一个独立 pattern-set 重复
    replicate_rates = (
        subset.groupby(["model_id", x, "pattern_set_id"], as_index=False)[metric]
        .mean()
    )
    curve = (
        replicate_rates.groupby(["model_id", x])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    curve["metric"] = metric
    # [观测·不确定性] SE 来自 pattern-set 间波动，不把同一记忆库的多个 target 当独立样本
    curve["se"] = curve["std"].fillna(0.0) / np.sqrt(curve["count"])
    curve["lower"] = np.clip(curve["mean"] - 1.96 * curve["se"], 0, 1)
    curve["upper"] = np.clip(curve["mean"] + 1.96 * curve["se"], 0, 1)
    return curve


# [展示] f1/f2 共用同一绘图函数；它不重新调用模型，也不改成功判据
def plot_success_curve(
    curve: pd.DataFrame,
    x: str,
    xlabel: str,
    title: str,
    *,
    log_x: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for model_id, group in curve.groupby("model_id", sort=False):
        ordered = group.sort_values(x)
        axis.plot(ordered[x], ordered["mean"], marker="o", label=MODEL_LABELS[model_id])
        axis.fill_between(ordered[x], ordered["lower"], ordered["upper"], alpha=0.14)
    metric = curve["metric"].iloc[0]
    ylabel = "Top-1 memory identification rate" if metric == "top1_correct" else metric
    axis.set(xlabel=xlabel, ylabel=ylabel, ylim=(-0.03, 1.03))
    if log_x:
        axis.set_xscale("log", base=2)
    axis.set_title(title)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()


# [展示] 空心三角形编码删失方向，提醒读者扫描边界不是实测临界点
def plot_capacity(summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for model_id, group in summary.groupby("model_id", sort=False):
        ordered = group.sort_values("N")
        axis.plot(ordered["N"], ordered["P_c"], marker="o", label=MODEL_LABELS[model_id])
        for marker, column in [("^", "right_censored"), ("v", "left_censored")]:
            censored = ordered[ordered[column]]
            axis.scatter(
                censored["N"], censored["P_c"], marker=marker, s=90,
                facecolors="none", edgecolors="black",
            )
    axis.set(xlabel="State dimension N", ylabel="Empirical critical capacity P_c")
    axis.set_yscale("log", base=2)
    axis.set_title("U1/U3 clean fixed-point capacity | unchanged ≥ 90%")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()


# [展示] 不同模型能量不可比，因此这里只叠加统一 signed bit-error 轨迹
def plot_dynamics(frame: pd.DataFrame, run_id: str) -> None:
    subset = frame[frame["run_id"] == run_id]
    validate_paired_results(subset)
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for row in subset.itertuples(index=False):
        axis.plot(
            range(len(row.error_trace)), row.error_trace, marker="o",
            label=MODEL_LABELS[row.model_id],
        )
    axis.set(xlabel="Registered retrieval step", ylabel="Signed bit error ratio", ylim=(-0.03, 1.03))
    axis.set_title(f"U5 paired dynamics | {run_id}")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()


# [展示] U7 把质量和登记 FLOPs 同时画出，避免只看召回率排冠军
def plot_quality_cost(frame: pd.DataFrame, N: int, P: int, level: float) -> None:
    subset = frame[
        (frame["N"] == N)
        & (frame["P"] == P)
        & (frame["corruption_level"] == level)
        & frame["retrieval_flops"].notna()
    ]
    validate_paired_results(subset)
    grouped = subset.groupby("model_id", as_index=False).agg(
        top1_correct=("top1_correct", "mean"),
        retrieval_flops=("retrieval_flops", "mean"),
    )
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for row in grouped.itertuples(index=False):
        axis.scatter(row.retrieval_flops, row.top1_correct, s=80)
        axis.annotate(
            MODEL_LABELS[row.model_id], (row.retrieval_flops, row.top1_correct),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )
    axis.set(
        xlabel="Approximate retrieval FLOPs", ylabel="Top-1 identification rate",
        ylim=(-0.03, 1.03),
    )
    axis.set_xscale("log")
    axis.set_title(f"U7 native quality-cost | N={N}, P={P}, corruption={level:.2f}")
    fig.tight_layout()
