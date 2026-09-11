"""实验 3：二维查询流形上的拉回度量主方向与吸引域边界法向。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from experiment_01_boundary_localization import (
    BoundaryExperimentConfig,
    make_binary_pair,
)
from experiment_02_moving_boundary import (
    biased_two_memory_weights,
    continuous_hopfield_energy,
    count_sign_transitions,
    estimate_energy_ridge,
    estimate_zero_crossing,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class DirectionalGeometryConfig(BoundaryExperimentConfig):
    """二维方向实验的冻结条件。"""

    requested_overlaps: tuple[float, ...] = (0.0,)
    seeds: tuple[int, ...] = tuple(range(8))
    deltas: tuple[float, ...] = (-0.30, -0.15, 0.0, 0.15, 0.30)
    boundary_steps: int = 1200
    u_points: int = 81
    v_min: float = -0.30
    v_max: float = 0.30
    v_points: int = 13
    topology_u_points: int = 41
    boundary_iterations: int = 35
    anisotropy_threshold: float = 1.05
    direction_evaluation_start_time: float = 1.0
    median_angle_tolerance_deg: float = 5.0
    p90_angle_tolerance_deg: float = 15.0
    independent_angle_margin_deg: float = 1.0
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_911
    representative_seed: int = 0
    representative_delta: float = 0.30
    representative_time: float = 1.0

    def validate(self) -> None:
        super().validate()
        if len(self.requested_overlaps) != 1:
            raise ValueError("experiment 3 isolates direction at one A/B overlap")
        if not self.deltas or len(set(self.deltas)) != len(self.deltas):
            raise ValueError("deltas must not be empty or duplicated")
        if not all(-1 < delta < 1 for delta in self.deltas):
            raise ValueError("deltas must keep both memory coefficients positive")
        if self.u_points < 11 or self.topology_u_points < 11:
            raise ValueError("u grids are too small")
        if self.v_points < 5 or not self.v_min < 0 < self.v_max:
            raise ValueError("the v grid must straddle zero with at least five points")
        if self.v_points % 2 == 0:
            raise ValueError("v_points must be odd so v=0 is sampled")
        if self.boundary_iterations < 10:
            raise ValueError("boundary_iterations is too small")
        if self.anisotropy_threshold <= 1:
            raise ValueError("anisotropy_threshold must exceed one")
        if not 0 <= self.direction_evaluation_start_time <= self.metric_steps * self.dt:
            raise ValueError("direction evaluation time is outside the trajectory")
        if self.median_angle_tolerance_deg <= 0 or self.p90_angle_tolerance_deg <= 0:
            raise ValueError("angle tolerances must be positive")
        if self.independent_angle_margin_deg < 0:
            raise ValueError("independent angle margin must be non-negative")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("bootstrap_resamples must be at least 1000")
        if self.representative_delta not in self.deltas:
            raise ValueError("representative_delta must be one of deltas")


def make_perpendicular_direction(
    A: np.ndarray,
    B: np.ndarray,
    seed: int,
) -> np.ndarray:
    """生成垂直于 span(A,B) 的 eta，并令 ||eta||=||B-A||。"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    generator = np.random.default_rng(1_000_003 + seed)
    raw = generator.normal(size=A.shape[0])
    basis = np.column_stack([A, B])
    coefficients, *_ = np.linalg.lstsq(basis, raw, rcond=None)
    eta = raw - basis @ coefficients
    norm = np.linalg.norm(eta)
    if norm <= 100 * np.finfo(float).eps:
        raise RuntimeError("failed to construct a perpendicular direction")
    return eta * (np.linalg.norm(B - A) / norm)


def _trajectory_2d(
    coordinates: jax.Array,
    A: jax.Array,
    B: jax.Array,
    eta: jax.Array,
    W: jax.Array,
    *,
    gain: float,
    dt: float,
    steps: int,
) -> jax.Array:
    u, v = coordinates
    initial = (1.0 - u) * A + u * B + v * eta

    def vector_field(state):
        return -state + W @ jnp.tanh(gain * state)

    def rk4_step(state, _):
        k1 = vector_field(state)
        k2 = vector_field(state + 0.5 * dt * k1)
        k3 = vector_field(state + 0.5 * dt * k2)
        k4 = vector_field(state + dt * k3)
        following = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return following, following

    _, states = jax.lax.scan(rk4_step, initial, None, length=steps)
    return jnp.concatenate([initial[None, :], states], axis=0)


def compiled_2d_dynamics(config: DirectionalGeometryConfig):
    """编译二维轨迹、2x2 拉回度量和长时终点观测。"""
    config.validate()

    def metric_trajectory(coordinates, A, B, eta, W):
        return _trajectory_2d(
            coordinates, A, B, eta, W,
            gain=config.gain, dt=config.dt, steps=config.metric_steps,
        )

    def boundary_trajectory(coordinates, A, B, eta, W):
        return _trajectory_2d(
            coordinates, A, B, eta, W,
            gain=config.gain, dt=config.dt, steps=config.boundary_steps,
        )

    tangent = jax.jacfwd(metric_trajectory, argnums=0)

    def metric(coordinates, A, B, eta, W):
        jacobian = tangent(coordinates, A, B, eta, W)
        return jnp.einsum("tni,tnj->tij", jacobian, jacobian)

    def final_margin(coordinates, A, B, eta, W):
        final = boundary_trajectory(coordinates, A, B, eta, W)[-1]
        return (final @ A - final @ B) / config.N

    def final_speed(coordinates, A, B, eta, W):
        final = boundary_trajectory(coordinates, A, B, eta, W)[-1]
        field = -final + W @ jnp.tanh(config.gain * final)
        return jnp.linalg.norm(field)

    return {
        "trajectory_grid": jax.jit(jax.vmap(
            metric_trajectory, in_axes=(0, None, None, None, None)
        )),
        "metric_grid": jax.jit(jax.vmap(
            metric, in_axes=(0, None, None, None, None)
        )),
        "final_margin_grid": jax.jit(jax.vmap(
            final_margin, in_axes=(0, None, None, None, None)
        )),
        "final_speed_grid": jax.jit(jax.vmap(
            final_speed, in_axes=(0, None, None, None, None)
        )),
    }


def parameter_grid(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """第二坐标在外层，返回 shape [len(second)*len(first),2]。"""
    return np.array([[x, y] for y in second for x in first], dtype=float)


def locate_boundary_curve(
    maps: dict,
    v_values: np.ndarray,
    A: jax.Array,
    B: jax.Array,
    eta: jax.Array,
    W: jax.Array,
    config: DirectionalGeometryConfig,
) -> dict:
    """对每个 v 并行二分 u；真实边界计算不读取 G。"""
    v_values = np.asarray(v_values, dtype=float)
    left = np.full(len(v_values), config.kappa_min)
    right = np.full(len(v_values), config.kappa_max)

    def coordinates_for(u_values):
        return np.column_stack([u_values, v_values])

    def margins_for(u_values):
        coordinates = coordinates_for(u_values)
        return np.asarray(maps["final_margin_grid"](
            jnp.asarray(coordinates), A, B, eta, W
        ))

    def endpoint_observations(u_values):
        coordinates = np.column_stack([u_values, v_values])
        margins = np.asarray(maps["final_margin_grid"](
            jnp.asarray(coordinates), A, B, eta, W
        ))
        speeds = np.asarray(maps["final_speed_grid"](
            jnp.asarray(coordinates), A, B, eta, W
        ))
        return margins, speeds

    margin_left, speed_left = endpoint_observations(left)
    margin_right, speed_right = endpoint_observations(right)
    endpoints_valid = bool(
        np.all(margin_left >= config.endpoint_margin_minimum)
        and np.all(margin_right <= -config.endpoint_margin_minimum)
        and np.all(speed_left <= config.endpoint_speed_tolerance)
        and np.all(speed_right <= config.endpoint_speed_tolerance)
    )

    topology_u = np.linspace(
        config.kappa_min, config.kappa_max, config.topology_u_points
    )
    topology_coordinates = parameter_grid(topology_u, v_values)
    topology_margin = np.asarray(maps["final_margin_grid"](
        jnp.asarray(topology_coordinates), A, B, eta, W
    )).reshape(len(v_values), len(topology_u))
    transitions = np.array([
        count_sign_transitions(row, config.boundary_margin_tolerance)
        for row in topology_margin
    ])
    topology_valid = endpoints_valid and bool(np.all(transitions == 1))

    if topology_valid:
        for _ in range(config.boundary_iterations):
            middle = 0.5 * (left + right)
            margins = margins_for(middle)
            left = np.where(margins > 0, middle, left)
            right = np.where(margins > 0, right, middle)
        boundary = 0.5 * (left + right)
    else:
        boundary = np.full(len(v_values), np.nan)

    return {
        "boundary_u": boundary,
        "topology_margin": topology_margin,
        "transitions": transitions,
        "topology_valid": topology_valid,
        "max_endpoint_speed": float(max(speed_left.max(), speed_right.max())),
        "min_endpoint_margin_magnitude": float(min(
            margin_left.min(), np.abs(margin_right).min()
        )),
    }


def curve_normals(curve_u: np.ndarray, v_values: np.ndarray) -> np.ndarray:
    """曲线 u=b(v) 的单位法向，与 u,v 的环境尺度已在 eta 构造中配平。"""
    curve_u = np.asarray(curve_u, dtype=float)
    derivative = np.gradient(curve_u, np.asarray(v_values, dtype=float))
    normals = np.stack([np.ones_like(derivative), -derivative], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def acute_angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """方向不区分正负，返回 0 到 90 度的夹角。"""
    dot = np.sum(np.asarray(first) * np.asarray(second), axis=-1)
    return np.degrees(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))


def metric_directions(
    metric: np.ndarray,
    anisotropy_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回最大特征向量、特征值比和方向是否可识别。"""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(metric, dtype=float))
    positive_minimum = np.maximum(eigenvalues[..., 0], np.finfo(float).tiny)
    positive_maximum = np.maximum(eigenvalues[..., 1], np.finfo(float).tiny)
    log_ratio = np.log(positive_maximum) - np.log(positive_minimum)
    ratio = np.exp(np.minimum(log_ratio, np.log(np.finfo(float).max)))
    direction = eigenvectors[..., :, 1]
    identifiable = ratio >= anisotropy_threshold
    return direction, ratio, identifiable


def estimate_observable_curves(
    trajectories: np.ndarray,
    u_values: np.ndarray,
    v_values: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    delta: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从同一轨迹网格提取逐时刻 overlap 零线和 energy ridge。"""
    margin = np.einsum("vutn,n->vut", trajectories, A - B) / len(A)
    energy = continuous_hopfield_energy(
        trajectories, A, B, delta, gain
    )
    time_count = trajectories.shape[2]
    overlap_curve = np.full((time_count, len(v_values)), np.nan)
    energy_curve = np.full_like(overlap_curve, np.nan)
    for time_index in range(time_count):
        for v_index in range(len(v_values)):
            overlap = estimate_zero_crossing(
                u_values, margin[v_index, :, time_index], zero_tolerance=1e-10
            )
            overlap_curve[time_index, v_index] = overlap["zero_kappa"]
            energy_curve[time_index, v_index] = estimate_energy_ridge(
                u_values, energy[v_index, :, time_index]
            )
    return overlap_curve, energy_curve


def run_experiment(config: DirectionalGeometryConfig):
    """运行二维方向实验，返回条件、边界点、逐时刻方向和紧凑原始数组。"""
    config.validate()
    maps = compiled_2d_dynamics(config)
    u_values = np.linspace(config.kappa_min, config.kappa_max, config.u_points)
    v_values = np.linspace(config.v_min, config.v_max, config.v_points)
    times = np.arange(config.metric_steps + 1) * config.dt
    query_coordinates = parameter_grid(u_values, v_values)

    condition_rows = []
    boundary_rows = []
    direction_rows = []
    raw_boundary = []
    raw_normals = []
    raw_metric = []
    raw_overlap_curves = []
    raw_energy_curves = []
    condition_index = 0

    for delta in config.deltas:
        for seed in config.seeds:
            A_np, B_np, achieved_overlap = make_binary_pair(
                config.N, seed, config.requested_overlaps[0]
            )
            eta_np = make_perpendicular_direction(A_np, B_np, seed)
            A = jnp.asarray(A_np)
            B = jnp.asarray(B_np)
            eta = jnp.asarray(eta_np)
            W = biased_two_memory_weights(A, B, delta)
            boundary = locate_boundary_curve(
                maps, v_values, A, B, eta, W, config
            )
            condition_id = f"delta{delta:+.2f}_seed{seed:03d}"
            condition_rows.append({
                "condition_index": condition_index,
                "condition_id": condition_id,
                "seed": seed,
                "delta": delta,
                "achieved_overlap": achieved_overlap,
                "topology_status": (
                    "valid" if boundary["topology_valid"] else "topology_changed"
                ),
                "max_sign_transitions": int(boundary["transitions"].max()),
                "max_endpoint_speed": boundary["max_endpoint_speed"],
                "min_endpoint_margin_magnitude": boundary[
                    "min_endpoint_margin_magnitude"
                ],
                "boundary_u_range": float(np.ptp(boundary["boundary_u"])),
            })
            if not boundary["topology_valid"]:
                condition_index += 1
                continue

            true_normals = curve_normals(boundary["boundary_u"], v_values)
            boundary_coordinates = np.column_stack([
                boundary["boundary_u"], v_values
            ])
            metric = np.asarray(maps["metric_grid"](
                jnp.asarray(boundary_coordinates), A, B, eta, W
            ))
            directions, ratios, identifiable = metric_directions(
                metric, config.anisotropy_threshold
            )
            trajectories = np.asarray(maps["trajectory_grid"](
                jnp.asarray(query_coordinates), A, B, eta, W
            )).reshape(
                len(v_values), len(u_values), config.metric_steps + 1, config.N
            )
            overlap_curve, energy_curve = estimate_observable_curves(
                trajectories, u_values, v_values,
                A_np, B_np, delta, config.gain,
            )
            overlap_normals = np.stack([
                curve_normals(curve, v_values) for curve in overlap_curve
            ])
            energy_normals = np.stack([
                curve_normals(curve, v_values) for curve in energy_curve
            ])

            for v_index, v_value in enumerate(v_values):
                boundary_rows.append({
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "seed": seed,
                    "delta": delta,
                    "v": v_value,
                    "true_boundary_u": boundary["boundary_u"][v_index],
                    "true_normal_u": true_normals[v_index, 0],
                    "true_normal_v": true_normals[v_index, 1],
                    "sign_transitions": boundary["transitions"][v_index],
                })
            for time_index, time in enumerate(times):
                for v_index, v_value in enumerate(v_values):
                    g_angle = acute_angle_deg(
                        directions[v_index, time_index], true_normals[v_index]
                    )
                    if not identifiable[v_index, time_index]:
                        g_angle = float("nan")
                    direction_rows.append({
                        "condition_index": condition_index,
                        "condition_id": condition_id,
                        "seed": seed,
                        "delta": delta,
                        "time_index": time_index,
                        "time": float(time),
                        "v_index": v_index,
                        "v": v_value,
                        "G_direction_u": directions[v_index, time_index, 0],
                        "G_direction_v": directions[v_index, time_index, 1],
                        "G_anisotropy_ratio": ratios[v_index, time_index],
                        "G_direction_identifiable": bool(
                            identifiable[v_index, time_index]
                        ),
                        "G_angle_deg": g_angle,
                        "overlap_curve_u": overlap_curve[time_index, v_index],
                        "overlap_angle_deg": acute_angle_deg(
                            overlap_normals[time_index, v_index],
                            true_normals[v_index],
                        ),
                        "overlap_location_error": abs(
                            overlap_curve[time_index, v_index]
                            - boundary["boundary_u"][v_index]
                        ),
                        "energy_curve_u": energy_curve[time_index, v_index],
                        "energy_angle_deg": acute_angle_deg(
                            energy_normals[time_index, v_index],
                            true_normals[v_index],
                        ),
                        "energy_location_error": abs(
                            energy_curve[time_index, v_index]
                            - boundary["boundary_u"][v_index]
                        ),
                    })

            raw_boundary.append(boundary["boundary_u"])
            raw_normals.append(true_normals)
            raw_metric.append(metric)
            raw_overlap_curves.append(overlap_curve)
            raw_energy_curves.append(energy_curve)
            condition_index += 1

    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(boundary_rows),
        pd.DataFrame(direction_rows),
        {
            "u_values": u_values,
            "v_values": v_values,
            "times": times,
            "boundary_u": np.stack(raw_boundary),
            "true_normals": np.stack(raw_normals),
            "G": np.stack(raw_metric),
            "overlap_curve": np.stack(raw_overlap_curves),
            "energy_curve": np.stack(raw_energy_curves),
        },
    )


def _first_sustained_time(
    values: np.ndarray,
    times: np.ndarray,
    tolerance: float,
) -> float:
    valid = np.isfinite(values) & (values <= tolerance)
    sustained = np.logical_and.accumulate(valid[::-1])[::-1]
    indices = np.flatnonzero(sustained)
    return float(times[indices[0]]) if len(indices) else float("nan")


def summarize_estimators(
    conditions: pd.DataFrame,
    directions: pd.DataFrame,
    config: DirectionalGeometryConfig,
) -> pd.DataFrame:
    """逐 condition 汇总三种方向估计器。"""
    records = []
    columns = {
        "pullback_G": "G_angle_deg",
        "overlap_normal": "overlap_angle_deg",
        "energy_ridge_normal": "energy_angle_deg",
    }
    valid_indices = conditions[
        conditions.topology_status == "valid"
    ].condition_index
    for condition_index in valid_indices:
        rows = directions[directions.condition_index == condition_index]
        condition = conditions[
            conditions.condition_index == condition_index
        ].iloc[0]
        for estimator, column in columns.items():
            by_time = rows.groupby("time")[column].median()
            late = rows[
                rows.time >= config.direction_evaluation_start_time
            ][column].dropna()
            records.append({
                "condition_index": condition_index,
                "condition_id": condition.condition_id,
                "seed": condition.seed,
                "delta": condition.delta,
                "estimator": estimator,
                "median_angle_after_start": float(late.median()),
                "p90_angle_after_start": float(late.quantile(0.9)),
                "first_sustained_time": _first_sustained_time(
                    by_time.to_numpy(), by_time.index.to_numpy(),
                    config.median_angle_tolerance_deg,
                ),
                "coverage": float(rows[column].notna().mean()),
            })
    return pd.DataFrame(records)


def experiment_summary(
    conditions: pd.DataFrame,
    directions: pd.DataFrame,
    estimator_summary: pd.DataFrame,
    config: DirectionalGeometryConfig,
) -> dict:
    """方向校准与相对基线增量价值的两层结论。"""
    valid_fraction = float((conditions.topology_status == "valid").mean())
    moving = estimator_summary[
        estimator_summary.delta.abs() > np.finfo(float).eps
    ]
    angles = moving.groupby("estimator").median_angle_after_start.median()
    p90 = moving.groupby("estimator").p90_angle_after_start.median()
    timing = moving.groupby("estimator").first_sustained_time.median()
    seed_angles = (
        moving.groupby(["seed", "estimator"])
        .median_angle_after_start.median()
        .unstack()
    )
    paired_gains = {}
    generator = np.random.default_rng(config.bootstrap_seed)
    for baseline in ("overlap_normal", "energy_ridge_normal"):
        gains = (seed_angles[baseline] - seed_angles["pullback_G"]).to_numpy()
        samples = generator.integers(
            0, len(gains), size=(config.bootstrap_resamples, len(gains))
        )
        bootstrap = np.median(gains[samples], axis=1)
        paired_gains[baseline] = {
            "median": float(np.median(gains)),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
        }
    late_g = directions[
        (directions.delta.abs() > np.finfo(float).eps)
        & (directions.time >= config.direction_evaluation_start_time)
    ]
    coverage = float(late_g.G_direction_identifiable.mean())
    directional_pass = bool(
        valid_fraction == 1.0
        and coverage >= config.required_identifiable_fraction
        and angles["pullback_G"] <= config.median_angle_tolerance_deg
        and p90["pullback_G"] <= config.p90_angle_tolerance_deg
    )
    g_more_accurate = bool(
        all(
            result["ci95_low"] > config.independent_angle_margin_deg
            for result in paired_gains.values()
        )
    )
    g_earlier = bool(
        timing["pullback_G"]
        < min(timing["overlap_normal"], timing["energy_ridge_normal"])
    )
    return {
        "passed_directional_geometry": directional_pass,
        "supports_independent_directional_G_value": (
            directional_pass and (g_more_accurate or g_earlier)
        ),
        "valid_topology_fraction": valid_fraction,
        "G_direction_coverage_after_start": coverage,
        "moving_boundary_median_angle_after_start_deg": {
            estimator: float(value) for estimator, value in angles.items()
        },
        "moving_boundary_p90_angle_after_start_deg": {
            estimator: float(value) for estimator, value in p90.items()
        },
        "moving_boundary_median_first_sustained_time": {
            estimator: float(value) for estimator, value in timing.items()
        },
        "paired_seed_angle_gain_over_G_deg": paired_gains,
        "G_more_accurate_by_practical_margin": g_more_accurate,
        "G_earlier_than_both_baselines": g_earlier,
        "max_endpoint_speed": float(conditions.max_endpoint_speed.max()),
        "min_endpoint_margin_magnitude": float(
            conditions.min_endpoint_margin_magnitude.min()
        ),
        "median_boundary_curve_range": float(
            conditions[conditions.delta.abs() > np.finfo(float).eps]
            .boundary_u_range.median()
        ),
    }


def plot_main_figure(
    conditions: pd.DataFrame,
    boundaries: pd.DataFrame,
    directions: pd.DataFrame,
    raw: dict,
    config: DirectionalGeometryConfig,
):
    """画二维边界、度量椭圆和三种法向误差。"""
    colors = {
        "G": "#D946EF",
        "overlap": "#0F766E",
        "energy": "#D97706",
        "truth": "#111827",
    }
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    ax_plane, ax_angle, ax_summary, ax_ratio = axes.ravel()

    condition = conditions[
        (conditions.seed == config.representative_seed)
        & np.isclose(conditions.delta, config.representative_delta)
    ].iloc[0]
    points = boundaries[
        boundaries.condition_index == condition.condition_index
    ].sort_values("v")
    rows = directions[
        directions.condition_index == condition.condition_index
    ]
    time_index = int(round(config.representative_time / config.dt))
    snapshot = rows[rows.time_index == time_index].sort_values("v_index")
    raw_index = int(np.flatnonzero(
        conditions[conditions.topology_status == "valid"].condition_index.to_numpy()
        == condition.condition_index
    )[0])
    metric = raw["G"][raw_index, :, time_index]
    ax_plane.plot(
        points.true_boundary_u, points.v,
        color=colors["truth"], lw=2.2, label="true boundary",
    )
    ax_plane.plot(
        snapshot.overlap_curve_u, snapshot.v,
        color=colors["overlap"], ls="--", lw=1.5, label="overlap zero",
    )
    ax_plane.plot(
        snapshot.energy_curve_u, snapshot.v,
        color=colors["energy"], ls=":", lw=1.8, label="energy ridge",
    )
    for index in range(0, len(points), 2):
        values, vectors = np.linalg.eigh(metric[index])
        scale = np.sqrt(values / values.max())
        major = vectors[:, 1]
        angle = np.degrees(np.arctan2(major[1], major[0]))
        ellipse = Ellipse(
            (points.true_boundary_u.iloc[index], points.v.iloc[index]),
            width=0.055 * scale[1], height=0.055 * scale[0], angle=angle,
            edgecolor=colors["G"], facecolor="none", lw=1.3,
        )
        ax_plane.add_patch(ellipse)
    ax_plane.set(
        title=("A  Local metric ellipses on a curved boundary "
               f"(delta={config.representative_delta:+.2f}, t={config.representative_time:g})"),
        xlabel=r"query coordinate $u$",
        ylabel=r"perpendicular coordinate $v$",
        xlim=(0.43, 0.68),
        ylim=(config.v_min - 0.02, config.v_max + 0.02),
    )
    ax_plane.legend(frameon=False, fontsize=8)

    moving = directions[directions.delta.abs() > np.finfo(float).eps]
    for label, column, color in (
        ("pullback G", "G_angle_deg", colors["G"]),
        ("overlap normal", "overlap_angle_deg", colors["overlap"]),
        ("energy ridge normal", "energy_angle_deg", colors["energy"]),
    ):
        aggregate = moving.groupby("time")[column].median()
        ax_angle.plot(aggregate.index, aggregate.values, color=color, lw=2, label=label)
    ax_angle.axhline(
        config.median_angle_tolerance_deg, color=colors["truth"], ls=":", lw=1.2
    )
    ax_angle.set(
        title="B  Median angle to the true boundary normal",
        xlabel="time",
        ylabel="acute angle (degrees)",
        ylim=(-0.3, 18),
    )
    ax_angle.legend(frameon=False, fontsize=8)

    estimator_summary = summarize_estimators(conditions, directions, config)
    moving_summary = estimator_summary[
        estimator_summary.delta.abs() > np.finfo(float).eps
    ]
    order = ["pullback_G", "overlap_normal", "energy_ridge_normal"]
    labels = ["pullback G", "overlap", "energy"]
    medians = [
        moving_summary[moving_summary.estimator == estimator]
        .median_angle_after_start.median()
        for estimator in order
    ]
    p90s = [
        moving_summary[moving_summary.estimator == estimator]
        .p90_angle_after_start.median()
        for estimator in order
    ]
    x = np.arange(3)
    ax_summary.bar(
        x, medians,
        color=[colors["G"], colors["overlap"], colors["energy"]], alpha=0.85,
    )
    ax_summary.scatter(x, p90s, color=colors["truth"], marker="_", s=250, zorder=3)
    ax_summary.set_xticks(x, labels)
    ax_summary.set(
        title=f"C  Direction error after t={config.direction_evaluation_start_time:g}",
        ylabel="degrees (bar=median, mark=p90)",
    )

    ratio = moving.groupby("time").G_anisotropy_ratio.median()
    ax_ratio.semilogy(ratio.index, ratio.values, color=colors["G"], lw=2)
    ax_ratio.axhline(
        config.anisotropy_threshold, color=colors["truth"], ls=":", lw=1.2
    )
    ax_ratio.set(
        title="D  Pullback anisotropy creates an identifiable direction",
        xlabel="time",
        ylabel=r"median $\lambda_{max}/\lambda_{min}$",
    )
    fig.suptitle(
        "Does the strongest pullback stretching point across the basin boundary?",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_artifacts(
    directory: str | Path,
    conditions: pd.DataFrame,
    boundaries: pd.DataFrame,
    directions: pd.DataFrame,
    raw: dict,
    config: DirectionalGeometryConfig,
) -> tuple[dict, Path]:
    """保存冻结配置、方向观测、紧凑原始数组、主图和结论。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    estimator_summary = summarize_estimators(conditions, directions, config)
    summary = experiment_summary(
        conditions, directions, estimator_summary, config
    )
    conditions.to_csv(directory / "conditions.csv", index=False)
    boundaries.to_csv(directory / "boundary_points.csv", index=False)
    directions.to_csv(
        directory / "direction_results.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    estimator_summary.to_csv(directory / "estimator_summary.csv", index=False)
    np.savez_compressed(
        directory / "directional_arrays.npz",
        **{
            key: value.astype(np.float32) if value.dtype == np.float64 else value
            for key, value in raw.items()
        },
        condition_ids=conditions[
            conditions.topology_status == "valid"
        ].condition_id.to_numpy(dtype=str),
        computed_with_x64=np.array(True),
        storage_dtype=np.array("float32"),
    )
    (directory / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    figure = plot_main_figure(
        conditions, boundaries, directions, raw, config
    )
    figure.savefig(directory / "main_figure.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / "main_figure.pdf", bbox_inches="tight")
    angles = summary["moving_boundary_median_angle_after_start_deg"]
    gains = summary["paired_seed_angle_gain_over_G_deg"]
    conclusion = (
        "# 实验 3 结论草稿\n\n"
        f"- 二维方向校准："
        f"{'通过' if summary['passed_directional_geometry'] else '未通过'}。\n"
        f"- 拓扑有效比例：{summary['valid_topology_fraction']:.3f}；"
        f"G 方向覆盖率：{summary['G_direction_coverage_after_start']:.3f}。\n"
        f"- t >= {config.direction_evaluation_start_time:g} 的移动边界中位夹角："
        f"G={angles['pullback_G']:.3g}°，"
        f"overlap={angles['overlap_normal']:.3g}°，"
        f"energy={angles['energy_ridge_normal']:.3g}°。\n"
        f"- 相对 G 的配对 seed 夹角劣势中位数及 95% bootstrap CI："
        f"overlap={gains['overlap_normal']['median']:.3g}° "
        f"[{gains['overlap_normal']['ci95_low']:.3g}, "
        f"{gains['overlap_normal']['ci95_high']:.3g}]；"
        f"energy={gains['energy_ridge_normal']['median']:.3g}° "
        f"[{gains['energy_ridge_normal']['ci95_low']:.3g}, "
        f"{gains['energy_ridge_normal']['ci95_high']:.3g}]。\n"
        f"- G 的独立方向价值："
        f"{'得到支持' if summary['supports_independent_directional_G_value'] else '暂未得到支持'}。\n\n"
        "判读边界：G 在真实边界上读取，因此这是一个有利于 G 的条件性方向测试，"
        "不代表它能独立找到边界位置。只有当其主方向稳定贴近真实法向，并以预先"
        "规定的角度裕量优于 overlap 与 energy 法向，才支持二维带来的增量信息。\n"
    )
    conclusion_path = directory / "conclusion_draft.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    return summary, conclusion_path
