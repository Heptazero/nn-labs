"""实验 2：用不等权双记忆把一维吸引域边界推离 0.5。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_01_boundary_localization import (
    BoundaryExperimentConfig,
    compiled_dynamics,
    estimate_metric_peak,
    locate_boundary,
    make_binary_pair,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MovingBoundaryConfig(BoundaryExperimentConfig):
    """除权重偏置轴外，沿用实验 1 的查询、动力学和度量设置。"""

    requested_overlaps: tuple[float, ...] = (0.0,)
    deltas: tuple[float, ...] = (-0.30, -0.15, 0.0, 0.15, 0.30)
    topology_points: int = 101
    estimator_error_tolerance: float = 0.01
    representative_delta: float = 0.30

    def validate(self) -> None:
        super().validate()
        if len(self.requested_overlaps) != 1:
            raise ValueError("experiment 2 isolates delta at one fixed A/B overlap")
        if not self.deltas or len(set(self.deltas)) != len(self.deltas):
            raise ValueError("deltas must not be empty or duplicated")
        if not all(-1 < delta < 1 for delta in self.deltas):
            raise ValueError("deltas must keep both memory coefficients positive")
        if self.topology_points < 11 or self.topology_points % 2 == 0:
            raise ValueError("topology_points must be an odd integer >= 11")
        if self.estimator_error_tolerance <= 0:
            raise ValueError("estimator_error_tolerance must be positive")
        if self.representative_delta not in self.deltas:
            raise ValueError("representative_delta must be one of deltas")


def biased_two_memory_weights(
    A: jax.Array,
    B: jax.Array,
    delta: float,
) -> jax.Array:
    """W_delta=((1+delta)AA^T+(1-delta)BB^T)/N，去除自连接。"""
    if A.ndim != 1 or B.shape != A.shape:
        raise ValueError("A and B must be aligned one-dimensional patterns")
    if not -1 < delta < 1:
        raise ValueError("delta must lie in (-1, 1)")
    N = A.shape[0]
    W = (
        (1.0 + delta) * jnp.outer(A, A)
        + (1.0 - delta) * jnp.outer(B, B)
    ) / N
    return W - jnp.diag(jnp.diag(W))


def estimate_zero_crossing(
    kappas: np.ndarray,
    values: np.ndarray,
    *,
    zero_tolerance: float = 1e-12,
) -> dict:
    """估计唯一零点；出现零个或多个符号切换时不伪造答案。"""
    kappas = np.asarray(kappas, dtype=float)
    values = np.asarray(values, dtype=float)
    exact = np.flatnonzero(np.abs(values) <= zero_tolerance)
    crossings = np.flatnonzero(values[:-1] * values[1:] < 0)
    candidates = []
    candidates.extend(float(kappas[index]) for index in exact)
    for index in crossings:
        left, right = values[index:index + 2]
        fraction = -left / (right - left)
        candidates.append(float(
            kappas[index] + fraction * (kappas[index + 1] - kappas[index])
        ))
    candidates = sorted(candidates)
    unique = []
    for candidate in candidates:
        if not unique or abs(candidate - unique[-1]) > zero_tolerance:
            unique.append(candidate)
    return {
        "zero_kappa": unique[0] if len(unique) == 1 else float("nan"),
        "zero_count": len(unique),
    }


def estimate_energy_ridge(kappas: np.ndarray, values: np.ndarray) -> float:
    """用能量最大值附近三点抛物线估计当前查询曲线的 ridge。"""
    kappas = np.asarray(kappas, dtype=float)
    values = np.asarray(values, dtype=float)
    index = int(np.argmax(values))
    peak = float(kappas[index])
    if 0 < index < len(kappas) - 1:
        left, middle, right = values[index - 1:index + 2]
        denominator = left - 2 * middle + right
        if denominator < -np.finfo(float).eps:
            offset = 0.5 * (left - right) / denominator
            if abs(offset) <= 1:
                peak += float(offset * (kappas[index + 1] - kappas[index]))
    return peak


def continuous_hopfield_energy(
    states: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    delta: float,
    gain: float,
) -> np.ndarray:
    """与 dx/dt=-x+W tanh(gx) 对应的连续 Hopfield Lyapunov 能量。"""
    states = np.asarray(states, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    N = A.shape[0]
    activations = np.tanh(gain * states)
    overlap_A = np.einsum("...i,i->...", activations, A)
    overlap_B = np.einsum("...i,i->...", activations, B)
    # 去除 W 的对角线后，补回的对角项恰为 sum(y_i^2)/N。
    interaction = -0.5 * (
        (1.0 + delta) * np.square(overlap_A)
        + (1.0 - delta) * np.square(overlap_B)
    ) / N + np.sum(np.square(activations), axis=-1) / N
    clipped = np.clip(
        activations,
        -1.0 + 10 * np.finfo(float).eps,
        1.0 - 10 * np.finfo(float).eps,
    )
    potential = np.sum(
        (
            clipped * np.arctanh(clipped)
            + 0.5 * np.log1p(-np.square(clipped))
        ) / gain,
        axis=-1,
    )
    return interaction + potential


def count_sign_transitions(values: np.ndarray, tolerance: float) -> int:
    """忽略接近零的格点后，统计长时标签沿查询线翻转几次。"""
    values = np.asarray(values, dtype=float)
    signs = np.sign(values[np.abs(values) > tolerance])
    if len(signs) < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _sustained_time(errors: np.ndarray, times: np.ndarray, tolerance: float) -> float:
    """返回从该时刻起一直不超过阈值的最早时间。"""
    valid = np.isfinite(errors) & (errors <= tolerance)
    sustained = np.logical_and.accumulate(valid[::-1])[::-1]
    indices = np.flatnonzero(sustained)
    return float(times[indices[0]]) if len(indices) else float("nan")


def run_experiment(config: MovingBoundaryConfig):
    """运行 delta 扫描，返回条件表、逐时刻估计和三种观测量网格。"""
    config.validate()
    maps = compiled_dynamics(config)
    kappas = np.linspace(config.kappa_min, config.kappa_max, config.kappa_points)
    topology_kappas = np.linspace(
        config.kappa_min, config.kappa_max, config.topology_points
    )
    times = np.arange(config.metric_steps + 1) * config.dt
    final_margin_grid = jax.jit(jax.vmap(
        maps["final_margin"], in_axes=(0, None, None, None)
    ))

    condition_rows = []
    time_rows = []
    metric_grids = []
    overlap_grids = []
    energy_grids = []
    topology_grids = []
    requested_overlap = config.requested_overlaps[0]
    condition_index = 0

    for delta in config.deltas:
        for seed in config.seeds:
            A_np, B_np, achieved_overlap = make_binary_pair(
                config.N, seed, requested_overlap
            )
            A = jnp.asarray(A_np)
            B = jnp.asarray(B_np)
            W = biased_two_memory_weights(A, B, delta)
            endpoint = {
                "left_endpoint_margin": float(
                    maps["final_margin"](config.kappa_min, A, B, W)
                ),
                "right_endpoint_margin": float(
                    maps["final_margin"](config.kappa_max, A, B, W)
                ),
                "left_endpoint_speed": float(
                    maps["final_speed"](config.kappa_min, A, B, W)
                ),
                "right_endpoint_speed": float(
                    maps["final_speed"](config.kappa_max, A, B, W)
                ),
            }
            topology_margin = np.asarray(
                final_margin_grid(jnp.asarray(topology_kappas), A, B, W)
            )
            transitions = count_sign_transitions(
                topology_margin, config.boundary_margin_tolerance
            )
            endpoints_valid = bool(
                endpoint["left_endpoint_margin"] >= config.endpoint_margin_minimum
                and endpoint["right_endpoint_margin"] <= -config.endpoint_margin_minimum
                and endpoint["left_endpoint_speed"] <= config.endpoint_speed_tolerance
                and endpoint["right_endpoint_speed"] <= config.endpoint_speed_tolerance
            )
            topology_valid = endpoints_valid and transitions == 1
            if topology_valid:
                boundary = locate_boundary(
                    maps["final_margin"], A, B, W, config
                )
            else:
                boundary = {
                    "boundary_kappa": float("nan"),
                    "boundary_bracket_width": float("nan"),
                    "boundary_iterations": 0,
                    "boundary_margin": float("nan"),
                }

            G = np.asarray(maps["metric_grid"](jnp.asarray(kappas), A, B, W))
            trajectories = np.asarray(
                maps["trajectory_grid"](jnp.asarray(kappas), A, B, W)
            )
            overlap_margin = np.einsum(
                "ktn,n->kt", trajectories, A_np - B_np
            ) / config.N
            energy = continuous_hopfield_energy(
                trajectories, A_np, B_np, delta, config.gain
            )
            condition_id = f"delta{delta:+.2f}_seed{seed:03d}"
            condition_rows.append({
                "condition_index": condition_index,
                "condition_id": condition_id,
                "seed": seed,
                "delta": delta,
                "requested_overlap": requested_overlap,
                "achieved_overlap": achieved_overlap,
                "topology_status": "valid" if topology_valid else "topology_changed",
                "sign_transitions": transitions,
                **endpoint,
                **boundary,
            })
            for time_index, time in enumerate(times):
                metric = estimate_metric_peak(
                    kappas,
                    G[:, time_index],
                    flat_relative_tolerance=config.flat_relative_tolerance,
                )
                overlap = estimate_zero_crossing(
                    kappas,
                    overlap_margin[:, time_index],
                    zero_tolerance=config.boundary_margin_tolerance,
                )
                energy_ridge = estimate_energy_ridge(
                    kappas, energy[:, time_index]
                )
                true_boundary = boundary["boundary_kappa"]
                metric_identifiable = bool(
                    np.isfinite(metric["metric_peak_kappa"])
                    and metric["metric_peak_contrast"] >= config.identifiable_contrast
                )
                time_rows.append({
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "seed": seed,
                    "delta": delta,
                    "time_index": time_index,
                    "time": float(time),
                    "true_boundary_kappa": true_boundary,
                    "metric_peak_kappa": metric["metric_peak_kappa"],
                    "metric_peak_contrast": metric["metric_peak_contrast"],
                    "metric_peak_identifiable": metric_identifiable,
                    "metric_error": (
                        abs(metric["metric_peak_kappa"] - true_boundary)
                        if metric_identifiable and topology_valid else float("nan")
                    ),
                    "overlap_zero_kappa": overlap["zero_kappa"],
                    "overlap_zero_count": overlap["zero_count"],
                    "overlap_error": (
                        abs(overlap["zero_kappa"] - true_boundary)
                        if np.isfinite(overlap["zero_kappa"]) and topology_valid
                        else float("nan")
                    ),
                    "energy_ridge_kappa": energy_ridge,
                    "energy_error": (
                        abs(energy_ridge - true_boundary)
                        if topology_valid else float("nan")
                    ),
                })
            metric_grids.append(G)
            overlap_grids.append(overlap_margin)
            energy_grids.append(energy)
            topology_grids.append(topology_margin)
            condition_index += 1

    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(time_rows),
        {
            "kappas": kappas,
            "topology_kappas": topology_kappas,
            "times": times,
            "G": np.stack(metric_grids),
            "overlap_margin": np.stack(overlap_grids),
            "energy": np.stack(energy_grids),
            "topology_margin": np.stack(topology_grids),
        },
    )


def summarize_estimators(
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    config: MovingBoundaryConfig,
) -> pd.DataFrame:
    """逐 condition 汇总三种估计器的持续达标时间和晚期误差。"""
    records = []
    columns = {
        "pullback_G": "metric_error",
        "overlap_zero": "overlap_error",
        "energy_ridge": "energy_error",
    }
    for condition in conditions.itertuples():
        rows = time_results[
            time_results.condition_index == condition.condition_index
        ].sort_values("time")
        for estimator, error_column in columns.items():
            errors = rows[error_column].to_numpy()
            late = rows[rows.time >= config.evaluation_start_time][error_column]
            records.append({
                "condition_index": condition.condition_index,
                "condition_id": condition.condition_id,
                "seed": condition.seed,
                "delta": condition.delta,
                "estimator": estimator,
                "first_sustained_time": _sustained_time(
                    errors, rows.time.to_numpy(), config.estimator_error_tolerance
                ),
                "median_error_after_start": float(late.median()),
                "p90_error_after_start": float(late.quantile(0.9)),
                "final_error": float(rows.iloc[-1][error_column]),
                "coverage": float(rows[error_column].notna().mean()),
            })
    return pd.DataFrame(records)


def experiment_summary(
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    estimator_summary: pd.DataFrame,
    config: MovingBoundaryConfig,
) -> dict:
    """冻结实验的两层判读：边界追踪，以及相对简单基线的增量价值。"""
    valid = conditions[conditions.topology_status == "valid"]
    late = time_results[
        (time_results.time >= config.evaluation_start_time)
        & time_results.condition_index.isin(valid.condition_index)
    ]
    identified = late[late.metric_peak_identifiable]
    moving = estimator_summary[
        estimator_summary.delta.abs() > np.finfo(float).eps
    ]
    accuracy = moving.groupby("estimator").median_error_after_start.median()
    timing = moving.groupby("estimator").first_sustained_time.median()
    direction_ok = bool(np.all(
        np.sign(valid[valid.delta != 0].boundary_kappa - 0.5)
        == np.sign(valid[valid.delta != 0].delta)
    ))
    g_tracks = bool(
        len(valid) == len(conditions)
        and direction_ok
        and late.metric_peak_identifiable.mean()
        >= config.required_identifiable_fraction
        and identified.metric_error.median() <= config.median_error_tolerance
        and identified.metric_error.quantile(0.9) <= config.p90_error_tolerance
    )
    g_more_accurate = bool(
        accuracy["pullback_G"]
        < min(accuracy["overlap_zero"], accuracy["energy_ridge"])
    )
    g_earlier = bool(
        timing["pullback_G"]
        < min(timing["overlap_zero"], timing["energy_ridge"])
    )
    boundary_by_delta = (
        valid.groupby("delta").boundary_kappa.median().to_dict()
    )
    return {
        "passed_boundary_tracking": g_tracks,
        "supports_independent_G_value": g_more_accurate or g_earlier,
        "valid_topology_fraction": float(len(valid) / len(conditions)),
        "boundary_moves_in_expected_direction": direction_ok,
        "boundary_kappa_median_by_delta": {
            f"{delta:+.2f}": float(value)
            for delta, value in boundary_by_delta.items()
        },
        "metric_identifiable_fraction_after_start": float(
            late.metric_peak_identifiable.mean()
        ),
        "metric_median_error_after_start": float(identified.metric_error.median()),
        "metric_p90_error_after_start": float(identified.metric_error.quantile(0.9)),
        "moving_boundary_median_error_after_start": {
            estimator: float(value) for estimator, value in accuracy.items()
        },
        "moving_boundary_median_first_sustained_time": {
            estimator: float(value) for estimator, value in timing.items()
        },
        "G_more_accurate_than_both_baselines": g_more_accurate,
        "G_earlier_than_both_baselines": g_earlier,
        "max_endpoint_speed": float(max(
            conditions.left_endpoint_speed.max(),
            conditions.right_endpoint_speed.max(),
        )),
        "min_endpoint_margin_magnitude": float(min(
            conditions.left_endpoint_margin.abs().min(),
            conditions.right_endpoint_margin.abs().min(),
        )),
    }


def plot_main_figure(
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    grids: dict,
    config: MovingBoundaryConfig,
):
    """显示移动边界、三估计器轨迹、误差和代表性 G 场。"""
    colors = {
        "pullback_G": "#D946EF",
        "overlap_zero": "#0F766E",
        "energy_ridge": "#D97706",
        "truth": "#111827",
    }
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    ax_delta, ax_track, ax_error, ax_heat = axes.ravel()

    boundaries = conditions.groupby("delta", as_index=False).boundary_kappa.median()
    ax_delta.plot(
        boundaries.delta, boundaries.boundary_kappa,
        color=colors["truth"], marker="o", lw=2.2, label=r"true $\kappa^*$",
    )
    readout_time = config.evaluation_start_time
    readout = time_results[np.isclose(time_results.time, readout_time)]
    for label, column in (
        ("pullback_G", "metric_peak_kappa"),
        ("overlap_zero", "overlap_zero_kappa"),
        ("energy_ridge", "energy_ridge_kappa"),
    ):
        values = readout.groupby("delta", as_index=False)[column].median()
        ax_delta.plot(
            values.delta, values[column], marker="o", ls="--", lw=1.5,
            color=colors[label], label=f"{label} at t={readout_time:g}",
        )
    ax_delta.axhline(0.5, color="#9CA3AF", ls=":", lw=1)
    ax_delta.set(
        title="A  Boundary leaves the symmetric midpoint",
        xlabel=r"memory-strength bias $\delta$",
        ylabel=r"boundary coordinate $\kappa$",
    )
    ax_delta.legend(frameon=False, fontsize=7.5)

    representative = conditions[
        (conditions.seed == config.representative_seed)
        & np.isclose(conditions.delta, config.representative_delta)
    ].iloc[0]
    trajectory = time_results[
        time_results.condition_index == representative.condition_index
    ]
    ax_track.axhline(
        representative.boundary_kappa, color=colors["truth"], ls="--", lw=1.5,
        label=rf"true $\kappa^*={representative.boundary_kappa:.3f}$",
    )
    for label, column in (
        ("pullback_G", "metric_peak_kappa"),
        ("overlap_zero", "overlap_zero_kappa"),
        ("energy_ridge", "energy_ridge_kappa"),
    ):
        ax_track.plot(
            trajectory.time, trajectory[column], color=colors[label], lw=2,
            label=label,
        )
    ax_track.set(
        title=f"B  Estimator trajectories (delta={config.representative_delta:+.2f})",
        xlabel="time",
        ylabel=r"estimated $\kappa$",
    )
    ax_track.legend(frameon=False, fontsize=8)

    moving = time_results[time_results.delta.abs() > np.finfo(float).eps]
    for label, column in (
        ("pullback_G", "metric_error"),
        ("overlap_zero", "overlap_error"),
        ("energy_ridge", "energy_error"),
    ):
        aggregate = moving.groupby("time")[column].median()
        ax_error.semilogy(
            aggregate.index,
            np.maximum(aggregate.values, 1e-7),
            color=colors[label], lw=2, label=label,
        )
    ax_error.axhline(
        config.estimator_error_tolerance, color=colors["truth"], ls=":", lw=1.2
    )
    ax_error.set(
        title="C  Median error on moving boundaries",
        xlabel="time",
        ylabel=r"median $|\hat\kappa-\kappa^*|$",
    )
    ax_error.legend(frameon=False, fontsize=8)

    G = grids["G"][int(representative.condition_index)]
    log_G = np.log10(G + 1e-300)
    low, high = np.percentile(log_G, [1, 99])
    image = ax_heat.pcolormesh(
        grids["kappas"], grids["times"], log_G.T,
        shading="auto", cmap="magma", vmin=low, vmax=high,
    )
    ax_heat.axvline(
        representative.boundary_kappa, color="white", ls="--", lw=1.5
    )
    ax_heat.plot(
        trajectory.metric_peak_kappa, trajectory.time,
        color="#67E8F9", lw=1.2, label=r"$\kappa_G(t)$",
    )
    ax_heat.set(
        title=f"D  Pullback field (delta={config.representative_delta:+.2f})",
        xlabel=r"query coordinate $\kappa$",
        ylabel="time",
    )
    ax_heat.legend(frameon=False, fontsize=8, loc="upper right")
    fig.colorbar(image, ax=ax_heat, label=r"$\log_{10}G(t,\kappa)$")
    fig.suptitle(
        "Can pullback sensitivity track a displaced Hopfield basin boundary?",
        fontsize=14,
        fontweight="bold",
    )
    return fig



def write_artifacts(
    directory: str | Path,
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    grids: dict,
    config: MovingBoundaryConfig,
) -> tuple[dict, Path]:
    """保存原始观测量、派生估计、配置、主图和结论草稿。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    estimator_summary = summarize_estimators(conditions, time_results, config)
    summary = experiment_summary(
        conditions, time_results, estimator_summary, config
    )
    conditions.to_csv(directory / "conditions.csv", index=False)
    time_results.to_csv(directory / "time_results.csv", index=False)
    estimator_summary.to_csv(directory / "estimator_summary.csv", index=False)
    np.savez_compressed(
        directory / "observable_grids.npz",
        # 估计器先用 float64 计算；只把审阅用的密集网格压成 float32 存档。
        kappas=grids["kappas"].astype(np.float32),
        topology_kappas=grids["topology_kappas"].astype(np.float32),
        times=grids["times"].astype(np.float32),
        G=grids["G"].astype(np.float32),
        overlap_margin=grids["overlap_margin"].astype(np.float32),
        energy=grids["energy"].astype(np.float32),
        topology_margin=grids["topology_margin"].astype(np.float32),
        condition_ids=conditions.condition_id.to_numpy(dtype=str),
        computed_with_x64=np.array(True),
        storage_dtype=np.array("float32"),
    )
    (directory / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure = plot_main_figure(conditions, time_results, grids, config)
    figure.savefig(directory / "main_figure.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / "main_figure.pdf", bbox_inches="tight")
    errors = summary["moving_boundary_median_error_after_start"]
    times = summary["moving_boundary_median_first_sustained_time"]
    conclusion = (
        "# 实验 2 结论草稿\n\n"
        f"- 边界追踪校准：{'通过' if summary['passed_boundary_tracking'] else '未通过'}。\n"
        f"- 五个 delta 的拓扑有效比例为 {summary['valid_topology_fraction']:.3f}；"
        f"边界移动方向{'符合' if summary['boundary_moves_in_expected_direction'] else '不符合'}预期。\n"
        f"- t >= {config.evaluation_start_time:g} 后，G 的中位误差为 "
        f"{summary['metric_median_error_after_start']:.4g}，90% 分位误差为 "
        f"{summary['metric_p90_error_after_start']:.4g}。\n"
        f"- 移动边界的晚期中位误差：G={errors['pullback_G']:.4g}，"
        f"overlap 零点={errors['overlap_zero']:.4g}，"
        f"能量 ridge={errors['energy_ridge']:.4g}。\n"
        f"- 持续进入 0.01 误差带的中位时刻：G={times['pullback_G']:.3g}，"
        f"overlap 零点={times['overlap_zero']:.3g}，"
        f"能量 ridge={times['energy_ridge']:.3g}。\n"
        f"- G 的独立分析价值："
        f"{'得到支持' if summary['supports_independent_G_value'] else '暂未得到支持'}。\n\n"
        "判读边界：通过第一层只说明 G 的峰能追踪一个由权重不对称推离 0.5 的"
        "未知边界。第二层必须与同轨迹上的 overlap 零点和能量 ridge 比较；若 G "
        "没有更早或更准，它仍是有用的几何可视化，但本实验不支持独立分析价值。\n"
    )
    conclusion_path = directory / "conclusion_draft.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    return summary, conclusion_path
