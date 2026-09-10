"""实验 1：用时间切片拉回度量定位两记忆连续 Hopfield 的吸引域边界。"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class BoundaryExperimentConfig:
    N: int = 60
    seeds: tuple[int, ...] = tuple(range(8))
    requested_overlaps: tuple[float, ...] = (-0.5, 0.0, 0.5, 0.7)
    gain: float = 4.0
    dt: float = 0.02
    metric_steps: int = 300
    boundary_steps: int = 4800
    kappa_min: float = 0.0
    kappa_max: float = 1.0
    kappa_points: int = 201
    boundary_tolerance: float = 1e-10
    boundary_margin_tolerance: float = 1e-10
    boundary_max_iterations: int = 60
    endpoint_speed_tolerance: float = 1e-6
    endpoint_margin_minimum: float = 1e-2
    flat_relative_tolerance: float = 1e-10
    identifiable_contrast: float = 1.10
    evaluation_start_time: float = 2.0
    median_error_tolerance: float = 0.01
    p90_error_tolerance: float = 0.025
    required_identifiable_fraction: float = 0.90
    representative_seed: int = 0
    representative_overlap: float = 0.5

    def validate(self) -> None:
        if self.N < 4 or self.kappa_points < 5:
            raise ValueError("N and kappa_points are too small")
        if self.kappa_points % 2 == 0:
            raise ValueError("kappa_points must be odd so the symmetric point is sampled")
        if not self.seeds or not self.requested_overlaps:
            raise ValueError("seeds and requested_overlaps must not be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if not all(-1 < overlap < 1 for overlap in self.requested_overlaps):
            raise ValueError("requested overlaps must lie strictly inside (-1, 1)")
        if self.dt <= 0 or self.metric_steps <= 0 or self.boundary_steps < self.metric_steps:
            raise ValueError("integration budgets are invalid")
        if not self.kappa_min < 0.5 < self.kappa_max:
            raise ValueError("the kappa bracket must contain 0.5")
        if self.gain <= 0 or self.boundary_tolerance <= 0:
            raise ValueError("gain and tolerances must be positive")
        if self.endpoint_speed_tolerance <= 0 or self.endpoint_margin_minimum <= 0:
            raise ValueError("endpoint validation thresholds must be positive")
        if not 0 < self.required_identifiable_fraction <= 1:
            raise ValueError("required_identifiable_fraction must lie in (0, 1]")


def make_binary_pair(
    N: int,
    seed: int,
    requested_overlap: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """固定翻转数生成 A、B；返回实际可达到的离散 overlap。"""
    if N <= 0 or not -1 < requested_overlap < 1:
        raise ValueError("invalid N or requested overlap")
    generator = np.random.default_rng(seed)
    A = generator.choice(np.array([-1.0, 1.0]), size=N)
    flip_count = int(round(N * (1 - requested_overlap) / 2))
    flip_count = min(max(flip_count, 1), N - 1)
    flipped = generator.choice(N, size=flip_count, replace=False)
    B = A.copy()
    B[flipped] *= -1
    achieved = float(np.dot(A, B) / N)
    return A, B, achieved


def hebbian_two_memory_weights(A: jax.Array, B: jax.Array) -> jax.Array:
    """实验 0 的等权 Hebbian 矩阵，去除自连接。"""
    if A.ndim != 1 or B.shape != A.shape:
        raise ValueError("A and B must be aligned one-dimensional patterns")
    N = A.shape[0]
    W = (jnp.outer(A, A) + jnp.outer(B, B)) / N
    return W - jnp.diag(jnp.diag(W))


def _trajectory(
    kappa: jax.Array,
    A: jax.Array,
    B: jax.Array,
    W: jax.Array,
    *,
    gain: float,
    dt: float,
    steps: int,
) -> jax.Array:
    x0 = (1.0 - kappa) * A + kappa * B

    def vector_field(x):
        return -x + W @ jnp.tanh(gain * x)

    def rk4_step(x, _):
        k1 = vector_field(x)
        k2 = vector_field(x + 0.5 * dt * k1)
        k3 = vector_field(x + 0.5 * dt * k2)
        k4 = vector_field(x + dt * k3)
        following = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return following, following

    _, states = jax.lax.scan(rk4_step, x0, None, length=steps)
    return jnp.concatenate([x0[None, :], states], axis=0)


def compiled_dynamics(config: BoundaryExperimentConfig):
    """同一形状的条件共享 JAX 编译结果，模式和权重作为显式输入。"""
    config.validate()

    def metric_trajectory(kappa, A, B, W):
        return _trajectory(
            kappa, A, B, W,
            gain=config.gain, dt=config.dt, steps=config.metric_steps,
        )

    def boundary_trajectory(kappa, A, B, W):
        return _trajectory(
            kappa, A, B, W,
            gain=config.gain, dt=config.dt, steps=config.boundary_steps,
        )

    tangent = jax.jacfwd(metric_trajectory, argnums=0)

    def metric(kappa, A, B, W):
        derivative = tangent(kappa, A, B, W)
        return jnp.sum(jnp.square(derivative), axis=-1)

    def final_margin(kappa, A, B, W):
        final = boundary_trajectory(kappa, A, B, W)[-1]
        return (final @ A - final @ B) / config.N

    def final_speed(kappa, A, B, W):
        final = boundary_trajectory(kappa, A, B, W)[-1]
        field = -final + W @ jnp.tanh(config.gain * final)
        return jnp.linalg.norm(field)

    return {
        "trajectory_grid": jax.jit(
            jax.vmap(metric_trajectory, in_axes=(0, None, None, None))
        ),
        "metric_grid": jax.jit(
            jax.vmap(metric, in_axes=(0, None, None, None))
        ),
        "final_margin": jax.jit(final_margin),
        "final_speed": jax.jit(final_speed),
    }


def locate_boundary(
    margin_fn,
    A: jax.Array,
    B: jax.Array,
    W: jax.Array,
    config: BoundaryExperimentConfig,
) -> dict:
    """只用长时终点 overlap margin 做二分，不读取 G。"""
    left = config.kappa_min
    right = config.kappa_max
    margin_left = float(margin_fn(left, A, B, W))
    margin_right = float(margin_fn(right, A, B, W))
    if not margin_left > 0 > margin_right:
        raise RuntimeError(
            f"endpoints do not retrieve opposite memories: {margin_left=}, {margin_right=}"
        )
    iterations = 0
    margin_mid = float("nan")
    while iterations < config.boundary_max_iterations:
        middle = 0.5 * (left + right)
        margin_mid = float(margin_fn(middle, A, B, W))
        iterations += 1
        if abs(margin_mid) <= config.boundary_margin_tolerance:
            left = right = middle
            break
        if margin_mid > 0:
            left = middle
        else:
            right = middle
        if right - left <= config.boundary_tolerance:
            break
    return {
        "boundary_kappa": 0.5 * (left + right),
        "boundary_bracket_width": right - left,
        "boundary_iterations": iterations,
        "boundary_margin": margin_mid,
        "left_endpoint_margin": margin_left,
        "right_endpoint_margin": margin_right,
    }


def estimate_metric_peak(
    kappas: np.ndarray,
    values: np.ndarray,
    *,
    flat_relative_tolerance: float,
) -> dict:
    """用 log G 的三点抛物线给出亚网格峰；平坦切片不伪造峰。"""
    values = np.asarray(values, dtype=float)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    relative_span = float(np.ptp(values) / scale)
    contrast = float(np.max(values) / max(np.median(values), np.finfo(float).tiny))
    if relative_span <= flat_relative_tolerance:
        return {
            "metric_peak_kappa": float("nan"),
            "metric_peak_contrast": contrast,
            "metric_relative_span": relative_span,
        }
    index = int(np.argmax(values))
    peak = float(kappas[index])
    if 0 < index < len(kappas) - 1:
        y_left, y_mid, y_right = np.log(
            np.maximum(values[index - 1:index + 2], np.finfo(float).tiny)
        )
        denominator = y_left - 2 * y_mid + y_right
        if denominator < -1e-14:
            offset = 0.5 * (y_left - y_right) / denominator
            if abs(offset) <= 1:
                peak += float(offset * (kappas[index + 1] - kappas[index]))
    return {
        "metric_peak_kappa": peak,
        "metric_peak_contrast": contrast,
        "metric_relative_span": relative_span,
    }


def run_experiment(config: BoundaryExperimentConfig):
    """运行冻结条件，返回 condition 表、逐时刻表和完整 G 网格。"""
    config.validate()
    maps = compiled_dynamics(config)
    kappas = np.linspace(config.kappa_min, config.kappa_max, config.kappa_points)
    times = np.arange(config.metric_steps + 1) * config.dt
    condition_rows = []
    time_rows = []
    metric_grids = []
    condition_index = 0

    for requested_overlap in config.requested_overlaps:
        for seed in config.seeds:
            A_np, B_np, achieved_overlap = make_binary_pair(
                config.N, seed, requested_overlap
            )
            A = jnp.asarray(A_np)
            B = jnp.asarray(B_np)
            W = hebbian_two_memory_weights(A, B)
            boundary = locate_boundary(
                maps["final_margin"], A, B, W, config
            )
            G = np.asarray(
                maps["metric_grid"](jnp.asarray(kappas), A, B, W)
            )
            if not np.isfinite(G).all():
                raise FloatingPointError(
                    f"non-finite metric for overlap={achieved_overlap}, seed={seed}"
                )
            condition_id = f"rho{achieved_overlap:+.3f}_seed{seed:03d}"
            condition_rows.append({
                "condition_index": condition_index,
                "condition_id": condition_id,
                "seed": seed,
                "requested_overlap": requested_overlap,
                "achieved_overlap": achieved_overlap,
                **boundary,
                "left_endpoint_speed": float(
                    maps["final_speed"](config.kappa_min, A, B, W)
                ),
                "right_endpoint_speed": float(
                    maps["final_speed"](config.kappa_max, A, B, W)
                ),
                "expected_G0": float(np.sum((B_np - A_np) ** 2)),
            })
            for time_index, time in enumerate(times):
                peak = estimate_metric_peak(
                    kappas,
                    G[:, time_index],
                    flat_relative_tolerance=config.flat_relative_tolerance,
                )
                peak_kappa = peak["metric_peak_kappa"]
                identifiable = (
                    np.isfinite(peak_kappa)
                    and peak["metric_peak_contrast"] >= config.identifiable_contrast
                )
                time_rows.append({
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "seed": seed,
                    "requested_overlap": requested_overlap,
                    "achieved_overlap": achieved_overlap,
                    "time_index": time_index,
                    "time": float(time),
                    **peak,
                    "peak_identifiable": bool(identifiable),
                    "localization_error": (
                        abs(peak_kappa - boundary["boundary_kappa"])
                        if np.isfinite(peak_kappa) else float("nan")
                    ),
                    "metric_max": float(G[:, time_index].max()),
                    "metric_median": float(np.median(G[:, time_index])),
                })
            metric_grids.append(G)
            condition_index += 1

    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(time_rows),
        {
            "kappas": kappas,
            "times": times,
            "G": np.stack(metric_grids),
        },
    )


def gate_summary(
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    config: BoundaryExperimentConfig,
) -> dict:
    """按预注册阈值汇总，不改变或筛选实验条件。"""
    evaluation = time_results[time_results.time >= config.evaluation_start_time]
    identifiable_fraction = float(evaluation.peak_identifiable.mean())
    identified = evaluation[evaluation.peak_identifiable]
    errors = identified.localization_error.dropna().to_numpy()
    median_error = float(np.median(errors)) if len(errors) else float("inf")
    p90_error = float(np.quantile(errors, 0.9)) if len(errors) else float("inf")
    endpoint_opposition_valid = bool(
        (conditions.left_endpoint_margin > 0).all()
        and (conditions.right_endpoint_margin < 0).all()
    )
    endpoint_converged = bool(
        (conditions.left_endpoint_speed <= config.endpoint_speed_tolerance).all()
        and (conditions.right_endpoint_speed <= config.endpoint_speed_tolerance).all()
    )
    endpoint_separated = bool(
        (conditions.left_endpoint_margin.abs() >= config.endpoint_margin_minimum).all()
        and (conditions.right_endpoint_margin.abs() >= config.endpoint_margin_minimum).all()
    )
    endpoint_valid = endpoint_opposition_valid and endpoint_converged and endpoint_separated
    passed = bool(
        endpoint_valid
        and identifiable_fraction >= config.required_identifiable_fraction
        and median_error <= config.median_error_tolerance
        and p90_error <= config.p90_error_tolerance
    )
    return {
        "passed": passed,
        "conditions": int(len(conditions)),
        "endpoint_valid": endpoint_valid,
        "endpoint_opposition_valid": endpoint_opposition_valid,
        "endpoint_converged": endpoint_converged,
        "endpoint_separated": endpoint_separated,
        "max_endpoint_speed": float(max(
            conditions.left_endpoint_speed.max(),
            conditions.right_endpoint_speed.max(),
        )),
        "min_endpoint_margin_magnitude": float(min(
            conditions.left_endpoint_margin.abs().min(),
            conditions.right_endpoint_margin.abs().min(),
        )),
        "identifiable_fraction_after_start": identifiable_fraction,
        "median_error_after_start": median_error,
        "p90_error_after_start": p90_error,
        "max_boundary_deviation_from_half": float(
            np.max(np.abs(conditions.boundary_kappa - 0.5))
        ),
        "max_boundary_bracket_width": float(
            conditions.boundary_bracket_width.max()
        ),
    }


def plot_main_figure(
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    grids: dict,
    config: BoundaryExperimentConfig,
):
    """四幅图对应峰的出现、位置、误差和可识别度。"""
    colors = dict(zip(
        sorted(conditions.achieved_overlap.unique()),
        ["#2563EB", "#0F766E", "#D97706", "#DC2626"],
    ))
    representative = conditions[
        (conditions.seed == config.representative_seed)
        & np.isclose(conditions.requested_overlap, config.representative_overlap)
    ]
    if len(representative) != 1:
        raise ValueError("representative condition is missing or ambiguous")
    row = representative.iloc[0]
    G = grids["G"][int(row.condition_index)]
    kappas = grids["kappas"]
    times = grids["times"]

    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    ax_heat, ax_peak, ax_error, ax_contrast = axes.ravel()

    log_G = np.log10(G + 1e-300)
    low, high = np.percentile(log_G, [1, 99])
    image = ax_heat.pcolormesh(
        kappas, times, log_G.T,
        shading="auto", cmap="magma", vmin=low, vmax=high,
    )
    ax_heat.axvline(row.boundary_kappa, color="white", ls="--", lw=1.5)
    ax_heat.set(
        title=f"A  Representative metric field (overlap={row.achieved_overlap:.2f})",
        xlabel=r"query coordinate $\kappa$",
        ylabel="time",
    )
    fig.colorbar(image, ax=ax_heat, label=r"$\log_{10}G(t,\kappa)$")

    grouped = time_results.groupby(["achieved_overlap", "time"], sort=True)
    aggregate = grouped.agg(
        peak_median=("metric_peak_kappa", "median"),
        error_median=("localization_error", "median"),
        error_p10=("localization_error", lambda x: x.quantile(0.1)),
        error_p90=("localization_error", lambda x: x.quantile(0.9)),
        contrast_median=("metric_peak_contrast", "median"),
    ).reset_index()
    for overlap, group in aggregate.groupby("achieved_overlap"):
        color = colors[overlap]
        ax_peak.plot(group.time, group.peak_median, color=color, lw=2, label=f"overlap={overlap:.2f}")
        ax_error.plot(group.time, group.error_median, color=color, lw=2)
        ax_error.fill_between(
            group.time,
            group.error_p10,
            group.error_p90,
            color=color,
            alpha=0.14,
        )
        ax_contrast.semilogy(group.time, group.contrast_median, color=color, lw=2)

    ax_peak.axhline(0.5, color="#111827", ls="--", lw=1.3, label="overlap boundary")
    ax_peak.set(
        title="B  Metric peak position",
        xlabel="time",
        ylabel=r"$\hat\kappa_G(t)$",
        ylim=(config.kappa_min - 0.03, config.kappa_max + 0.03),
    )
    ax_peak.legend(frameon=False, fontsize=8, ncol=2)

    ax_error.axhline(config.median_error_tolerance, color="#111827", ls=":", lw=1.2)
    ax_error.set(
        title="C  Boundary localization error",
        xlabel="time",
        ylabel=r"$|\hat\kappa_G(t)-\kappa_*|$",
        ylim=(-0.0005, config.median_error_tolerance * 1.15),
    )

    ax_contrast.axhline(config.identifiable_contrast, color="#111827", ls=":", lw=1.2)
    ax_contrast.set(
        title="D  Peak contrast",
        xlabel="time",
        ylabel="max G / median G",
    )

    fig.suptitle(
        "When does pullback sensitivity identify the Hopfield basin boundary?",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_artifacts(
    directory: str | Path,
    conditions: pd.DataFrame,
    time_results: pd.DataFrame,
    grids: dict,
    config: BoundaryExperimentConfig,
) -> tuple[dict, Path]:
    """保存配置、原始 G、派生表、主图和结论草稿。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(directory / "conditions.csv", index=False)
    time_results.to_csv(directory / "time_results.csv", index=False)
    first_identifiable = (
        time_results[time_results.peak_identifiable]
        .groupby(["achieved_overlap", "seed"], as_index=False)
        .time.min()
        .rename(columns={"time": "first_identifiable_time"})
    )
    overlap_summary = (
        first_identifiable.groupby("achieved_overlap")["first_identifiable_time"]
        .agg(["median", "min", "max"])
        .reset_index()
        .rename(columns={
            "median": "first_identifiable_time_median",
            "min": "first_identifiable_time_min",
            "max": "first_identifiable_time_max",
        })
    )
    final_time = float(time_results.time.max())
    final_contrast = (
        time_results[np.isclose(time_results.time, final_time)]
        .groupby("achieved_overlap", as_index=False)
        .metric_peak_contrast.median()
        .rename(columns={"metric_peak_contrast": "final_peak_contrast_median"})
    )
    overlap_summary = overlap_summary.merge(
        final_contrast, on="achieved_overlap", validate="one_to_one"
    )
    overlap_summary.to_csv(directory / "overlap_summary.csv", index=False)
    np.savez_compressed(
        directory / "metric_grid.npz",
        kappas=grids["kappas"],
        times=grids["times"],
        G=grids["G"],
        condition_ids=conditions.condition_id.to_numpy(),
    )
    (directory / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = gate_summary(conditions, time_results, config)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    figure = plot_main_figure(conditions, time_results, grids, config)
    figure.savefig(directory / "main_figure.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / "main_figure.pdf", bbox_inches="tight")
    conclusion = (
        "# 实验 1 结论草稿\n\n"
        f"- 预注册门槛：{'通过' if summary['passed'] else '未通过'}。\n"
        f"- t >= {config.evaluation_start_time:g} 后，峰可识别比例为 "
        f"{summary['identifiable_fraction_after_start']:.3f}。\n"
        f"- 可识别切片的中位定位误差为 {summary['median_error_after_start']:.4g}，"
        f"90% 分位误差为 {summary['p90_error_after_start']:.4g}。\n"
        f"- 独立 overlap 二分边界相对 0.5 的最大偏差为 "
        f"{summary['max_boundary_deviation_from_half']:.3e}。\n\n"
        f"- 最慢条件的长时终点速度为 {summary['max_endpoint_speed']:.3e}；"
        f"最小终点 overlap margin 绝对值为 "
        f"{summary['min_endpoint_margin_magnitude']:.3e}。\n\n"
        "结果判读：这是数值校准通过，不是未知边界发现。等权两记忆模型具有 "
        "A/B 交换对称性，理论上已经把边界钉在 kappa=0.5；固定 overlap 时，"
        "不同 seed 主要对应坐标置换和符号变换。相关性扫描仍能检验峰何时可识别、"
        "峰的对比度怎样衰减，但不能把跨 seed 一致性当成一般高维 Hopfield 的"
        "独立泛化证据。overlap=0.8 的预检查显示两个端点最终合并到同一吸引子，"
        "所以冻结实验上限改为 0.7，没有把不存在的两吸引域边界计为成功。\n"
    )
    conclusion_path = directory / "conclusion_draft.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    return summary, conclusion_path
