"""实验 4：在同一真实边界点审计 pullback 方向的来源。"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_01_boundary_localization import make_binary_pair
from experiment_02_moving_boundary import biased_two_memory_weights
from experiment_03_2d_directional_geometry import (
    DirectionalGeometryConfig,
    _trajectory_2d,
    acute_angle_deg,
    compiled_2d_dynamics,
    curve_normals,
    locate_boundary_curve,
    make_perpendicular_direction,
    metric_directions,
)

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MechanismControlConfig(DirectionalGeometryConfig):
    """冻结实验 3，只增加同位置方向控制和判读门。"""

    direction_gap_relative_tolerance: float = 1e-8
    energy_negative_curvature_tolerance: float = 1e-10
    early_mechanism_time: float = 0.02

    def validate(self) -> None:
        super().validate()
        if self.direction_gap_relative_tolerance <= 0:
            raise ValueError("direction-gap tolerance must be positive")
        if self.energy_negative_curvature_tolerance <= 0:
            raise ValueError("energy curvature tolerance must be positive")
        if not 0 < self.early_mechanism_time <= self.metric_steps * self.dt:
            raise ValueError("early mechanism time is outside the trajectory")
        index = self.early_mechanism_time / self.dt
        if not np.isclose(index, round(index)):
            raise ValueError("early mechanism time must lie on the time grid")


def continuous_hopfield_energy_jax(
    state: jax.Array,
    W: jax.Array,
    gain: float,
) -> jax.Array:
    """与 dx/dt=-x+W tanh(gx) 对应的可微 Lyapunov 能量。"""
    activation = jnp.tanh(gain * state)
    clipped = jnp.clip(
        activation,
        -1.0 + 10 * jnp.finfo(state.dtype).eps,
        1.0 - 10 * jnp.finfo(state.dtype).eps,
    )
    interaction = -0.5 * activation @ W @ activation
    potential = jnp.sum(
        (
            clipped * jnp.arctanh(clipped)
            + 0.5 * jnp.log1p(-jnp.square(clipped))
        ) / gain
    )
    return interaction + potential


def vector_field_jacobian(
    state: jax.Array,
    W: jax.Array,
    gain: float,
) -> jax.Array:
    """解析计算 Df(x)=-I+W diag(gain*sech^2(gain*x))。"""
    derivative = gain * (1.0 - jnp.square(jnp.tanh(gain * state)))
    return -jnp.eye(state.shape[0], dtype=state.dtype) + W * derivative[None, :]


def compiled_mechanism_controls(config: MechanismControlConfig):
    """编译同点轨迹、切映射、energy Hessian 与局部应变。"""
    config.validate()

    def trajectory(coordinates, A, B, eta, W):
        return _trajectory_2d(
            coordinates, A, B, eta, W,
            gain=config.gain, dt=config.dt, steps=config.metric_steps,
        )

    tangent = jax.jacfwd(trajectory, argnums=0)

    def energy_trajectory(coordinates, A, B, eta, W):
        states = trajectory(coordinates, A, B, eta, W)
        return jax.vmap(
            lambda state: continuous_hopfield_energy_jax(
                state, W, config.gain
            )
        )(states)

    energy_hessian = jax.jacfwd(jax.jacrev(energy_trajectory, argnums=0), argnums=0)

    def controls(coordinates, A, B, eta, W):
        states = trajectory(coordinates, A, B, eta, W)
        jacobians = tangent(coordinates, A, B, eta, W)
        metric = jnp.einsum("tni,tnj->tij", jacobians, jacobians)
        overlap_gradient = jnp.einsum(
            "tni,n->ti", jacobians, (A - B) / config.N
        )
        hessian = energy_hessian(coordinates, A, B, eta, W)
        field_jacobians = jax.vmap(
            lambda state: vector_field_jacobian(state, W, config.gain)
        )(states)
        symmetric_rate = field_jacobians + jnp.swapaxes(field_jacobians, -1, -2)
        strain = jnp.einsum(
            "tni,tnm,tmj->tij", jacobians, symmetric_rate, jacobians
        )
        return states, jacobians, metric, overlap_gradient, hessian, strain

    return jax.jit(jax.vmap(
        controls, in_axes=(0, None, None, None, None)
    ))


def normalize_directions(
    vectors: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """单位化向量；近零梯度没有可识别方向。"""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=-1)
    valid = norms > tolerance
    directions = np.full_like(vectors, np.nan)
    directions[valid] = vectors[valid] / norms[valid, None]
    return directions, valid


def symmetric_matrix_directions(
    matrices: np.ndarray,
    *,
    eigen_index: int,
    gap_relative_tolerance: float,
    require_negative: bool = False,
    negative_tolerance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """取对称 2x2 矩阵的指定特征方向，并显式检查可识别性。"""
    matrices = np.asarray(matrices, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    scale = np.maximum(np.max(np.abs(eigenvalues), axis=-1), np.finfo(float).tiny)
    relative_gap = (eigenvalues[..., 1] - eigenvalues[..., 0]) / scale
    identifiable = relative_gap >= gap_relative_tolerance
    if require_negative:
        identifiable &= eigenvalues[..., eigen_index] < -negative_tolerance
    return (
        eigenvectors[..., :, eigen_index],
        eigenvalues,
        relative_gap,
        identifiable,
    )


def _angle_or_nan(direction, normal, identifiable: bool) -> float:
    if not identifiable or not np.all(np.isfinite(direction)):
        return float("nan")
    return float(acute_angle_deg(direction, normal))


def _relative_linearization_error(
    metric: np.ndarray,
    initial_metric: np.ndarray,
    initial_strain: np.ndarray,
    time: float,
) -> float:
    change = metric - initial_metric
    denominator = np.linalg.norm(change)
    if denominator <= np.finfo(float).tiny:
        return float("nan")
    approximation = initial_metric + time * initial_strain
    return float(np.linalg.norm(metric - approximation) / denominator)


def run_experiment(config: MechanismControlConfig):
    """在实验 3 的真实边界点运行四种同位置方向控制。"""
    config.validate()
    boundary_maps = compiled_2d_dynamics(config)
    controls_map = compiled_mechanism_controls(config)
    v_values = np.linspace(config.v_min, config.v_max, config.v_points)
    times = np.arange(config.metric_steps + 1) * config.dt

    condition_rows = []
    boundary_rows = []
    direction_rows = []
    raw_records = {
        "boundary_u": [],
        "true_normals": [],
        "G": [],
        "energy_hessian": [],
        "strain_St": [],
        "overlap_gradient": [],
    }
    condition_index = 0

    for delta in config.deltas:
        for seed in config.seeds:
            A_np, B_np, achieved_overlap = make_binary_pair(
                config.N, seed, config.requested_overlaps[0]
            )
            eta_np = make_perpendicular_direction(A_np, B_np, seed)
            A, B, eta = map(jnp.asarray, (A_np, B_np, eta_np))
            W = biased_two_memory_weights(A, B, delta)
            boundary = locate_boundary_curve(
                boundary_maps, v_values, A, B, eta, W, config
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
            (
                states,
                jacobians,
                metric,
                overlap_gradient,
                energy_hessian,
                strain,
            ) = [
                np.asarray(value) for value in controls_map(
                    jnp.asarray(boundary_coordinates), A, B, eta, W
                )
            ]
            g_direction, g_ratio, g_identifiable = metric_directions(
                metric, config.anisotropy_threshold
            )
            overlap_direction, overlap_identifiable = normalize_directions(
                overlap_gradient
            )
            (
                energy_direction,
                energy_eigenvalues,
                energy_gap,
                energy_identifiable,
            ) = symmetric_matrix_directions(
                energy_hessian,
                eigen_index=0,
                gap_relative_tolerance=config.direction_gap_relative_tolerance,
                require_negative=True,
                negative_tolerance=config.energy_negative_curvature_tolerance,
            )
            (
                strain_direction,
                strain_eigenvalues,
                strain_gap,
                strain_identifiable,
            ) = symmetric_matrix_directions(
                strain,
                eigen_index=1,
                gap_relative_tolerance=config.direction_gap_relative_tolerance,
            )

            for v_index, v_value in enumerate(v_values):
                boundary_rows.append({
                    "condition_index": condition_index,
                    "condition_id": condition_id,
                    "seed": seed,
                    "delta": delta,
                    "v_index": v_index,
                    "v": v_value,
                    "true_boundary_u": boundary["boundary_u"][v_index],
                    "true_normal_u": true_normals[v_index, 0],
                    "true_normal_v": true_normals[v_index, 1],
                    "sign_transitions": boundary["transitions"][v_index],
                })
                initial_strain_direction = strain_direction[v_index, 0]
                initial_strain_identifiable = bool(strain_identifiable[v_index, 0])
                initial_strain_angle = _angle_or_nan(
                    initial_strain_direction,
                    true_normals[v_index],
                    initial_strain_identifiable,
                )
                for time_index, time in enumerate(times):
                    g_ok = bool(g_identifiable[v_index, time_index])
                    overlap_ok = bool(overlap_identifiable[v_index, time_index])
                    energy_ok = bool(energy_identifiable[v_index, time_index])
                    strain_ok = bool(strain_identifiable[v_index, time_index])
                    g_s0_angle = (
                        float(acute_angle_deg(
                            g_direction[v_index, time_index],
                            initial_strain_direction,
                        ))
                        if g_ok and initial_strain_identifiable else float("nan")
                    )
                    direction_rows.append({
                        "condition_index": condition_index,
                        "condition_id": condition_id,
                        "seed": seed,
                        "delta": delta,
                        "time_index": time_index,
                        "time": float(time),
                        "v_index": v_index,
                        "v": v_value,
                        "G_direction_u": g_direction[v_index, time_index, 0],
                        "G_direction_v": g_direction[v_index, time_index, 1],
                        "G_anisotropy_ratio": g_ratio[v_index, time_index],
                        "G_identifiable": g_ok,
                        "G_angle_deg": _angle_or_nan(
                            g_direction[v_index, time_index],
                            true_normals[v_index], g_ok,
                        ),
                        "overlap_direction_u": overlap_direction[v_index, time_index, 0],
                        "overlap_direction_v": overlap_direction[v_index, time_index, 1],
                        "overlap_identifiable": overlap_ok,
                        "overlap_angle_deg": _angle_or_nan(
                            overlap_direction[v_index, time_index],
                            true_normals[v_index], overlap_ok,
                        ),
                        "energy_direction_u": energy_direction[v_index, time_index, 0],
                        "energy_direction_v": energy_direction[v_index, time_index, 1],
                        "energy_lambda_min": energy_eigenvalues[v_index, time_index, 0],
                        "energy_lambda_max": energy_eigenvalues[v_index, time_index, 1],
                        "energy_relative_gap": energy_gap[v_index, time_index],
                        "energy_identifiable": energy_ok,
                        "energy_angle_deg": _angle_or_nan(
                            energy_direction[v_index, time_index],
                            true_normals[v_index], energy_ok,
                        ),
                        "strain_St_direction_u": strain_direction[v_index, time_index, 0],
                        "strain_St_direction_v": strain_direction[v_index, time_index, 1],
                        "strain_St_lambda_min": strain_eigenvalues[v_index, time_index, 0],
                        "strain_St_lambda_max": strain_eigenvalues[v_index, time_index, 1],
                        "strain_St_relative_gap": strain_gap[v_index, time_index],
                        "strain_St_identifiable": strain_ok,
                        "strain_St_angle_deg": _angle_or_nan(
                            strain_direction[v_index, time_index],
                            true_normals[v_index], strain_ok,
                        ),
                        "strain_S0_direction_u": initial_strain_direction[0],
                        "strain_S0_direction_v": initial_strain_direction[1],
                        "strain_S0_identifiable": initial_strain_identifiable,
                        "strain_S0_angle_deg": initial_strain_angle,
                        "G_vs_S0_angle_deg": g_s0_angle,
                        "G_linearization_relative_error": _relative_linearization_error(
                            metric[v_index, time_index],
                            metric[v_index, 0],
                            strain[v_index, 0],
                            float(time),
                        ),
                    })

            raw_records["boundary_u"].append(boundary["boundary_u"])
            raw_records["true_normals"].append(true_normals)
            raw_records["G"].append(metric)
            raw_records["energy_hessian"].append(energy_hessian)
            raw_records["strain_St"].append(strain)
            raw_records["overlap_gradient"].append(overlap_gradient)
            condition_index += 1

    raw = {
        "v_values": v_values,
        "times": times,
        **{key: np.stack(value) for key, value in raw_records.items()},
    }
    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(boundary_rows),
        pd.DataFrame(direction_rows),
        raw,
    )


ESTIMATOR_COLUMNS = {
    "pullback_G": "G_angle_deg",
    "same_point_overlap_gradient": "overlap_angle_deg",
    "same_point_energy_hessian": "energy_angle_deg",
    "instantaneous_strain_S0": "strain_S0_angle_deg",
    "time_varying_strain_St": "strain_St_angle_deg",
}


def summarize_estimators(
    conditions: pd.DataFrame,
    directions: pd.DataFrame,
    config: MechanismControlConfig,
) -> pd.DataFrame:
    """逐 condition 汇总所有同位置方向，seed 仍是统计独立单位。"""
    records = []
    valid_indices = conditions[
        conditions.topology_status == "valid"
    ].condition_index
    for condition_index in valid_indices:
        condition = conditions[
            conditions.condition_index == condition_index
        ].iloc[0]
        rows = directions[
            (directions.condition_index == condition_index)
            & (directions.time >= config.direction_evaluation_start_time)
        ]
        for estimator, column in ESTIMATOR_COLUMNS.items():
            values = rows[column].dropna()
            records.append({
                "condition_index": condition_index,
                "condition_id": condition.condition_id,
                "seed": int(condition.seed),
                "delta": float(condition.delta),
                "estimator": estimator,
                "median_angle_after_start": float(values.median()),
                "p90_angle_after_start": float(values.quantile(0.9)),
                "coverage_after_start": float(rows[column].notna().mean()),
            })
    return pd.DataFrame(records)


def paired_seed_bootstrap(
    estimator_summary: pd.DataFrame,
    config: MechanismControlConfig,
) -> dict:
    """先在 seed 内合并 delta，再对 seed 配对重采样。"""
    moving = estimator_summary[
        estimator_summary.delta.abs() > np.finfo(float).eps
    ]
    seed_angles = (
        moving.groupby(["seed", "estimator"])
        .median_angle_after_start.median()
        .unstack()
    )
    generator = np.random.default_rng(config.bootstrap_seed)
    results = {}
    for baseline in ESTIMATOR_COLUMNS:
        if baseline == "pullback_G":
            continue
        gains = (seed_angles[baseline] - seed_angles["pullback_G"]).to_numpy()
        indices = generator.integers(
            0, len(gains), size=(config.bootstrap_resamples, len(gains))
        )
        bootstrap = np.median(gains[indices], axis=1)
        results[baseline] = {
            "median": float(np.median(gains)),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
        }
    return results


def experiment_summary(
    conditions: pd.DataFrame,
    directions: pd.DataFrame,
    estimator_summary: pd.DataFrame,
    config: MechanismControlConfig,
) -> dict:
    """按冻结的三分停止规则给出机制归类。"""
    moving = estimator_summary[
        estimator_summary.delta.abs() > np.finfo(float).eps
    ]
    angles = moving.groupby("estimator").median_angle_after_start.median()
    p90 = moving.groupby("estimator").p90_angle_after_start.median()
    coverage = moving.groupby("estimator").coverage_after_start.median()
    paired = paired_seed_bootstrap(estimator_summary, config)
    beats_overlap_energy = all(
        paired[name]["ci95_low"] > config.independent_angle_margin_deg
        for name in (
            "same_point_overlap_gradient",
            "same_point_energy_hessian",
        )
    )
    beats_S0 = (
        paired["instantaneous_strain_S0"]["ci95_low"]
        > config.independent_angle_margin_deg
    )
    if beats_overlap_energy and beats_S0:
        outcome = "full_dynamic_advantage"
    elif beats_overlap_energy:
        outcome = "local_stability_equivalent"
    else:
        outcome = "same_location_controls_remove_G_advantage"

    early = directions[np.isclose(
        directions.time, config.early_mechanism_time
    )]
    moving_early = early[early.delta.abs() > np.finfo(float).eps]
    return {
        "mechanism_outcome": outcome,
        "valid_topology_fraction": float(
            (conditions.topology_status == "valid").mean()
        ),
        "moving_boundary_median_angle_after_start_deg": {
            key: float(value) for key, value in angles.items()
        },
        "moving_boundary_p90_angle_after_start_deg": {
            key: float(value) for key, value in p90.items()
        },
        "moving_boundary_coverage_after_start": {
            key: float(value) for key, value in coverage.items()
        },
        "paired_seed_angle_gain_over_G_deg": paired,
        "G_beats_same_point_overlap_and_energy_by_margin": bool(
            beats_overlap_energy
        ),
        "G_beats_initial_strain_S0_by_margin": bool(beats_S0),
        "early_mechanism_time": config.early_mechanism_time,
        "median_G_vs_S0_angle_at_early_time_deg": float(
            moving_early.G_vs_S0_angle_deg.median()
        ),
        "median_G_linearization_relative_error_at_early_time": float(
            moving_early.G_linearization_relative_error.median()
        ),
        "max_endpoint_speed": float(conditions.max_endpoint_speed.max()),
        "min_endpoint_margin_magnitude": float(
            conditions.min_endpoint_margin_magnitude.min()
        ),
    }


def plot_main_figure(
    conditions: pd.DataFrame,
    boundaries: pd.DataFrame,
    directions: pd.DataFrame,
    estimator_summary: pd.DataFrame,
    summary: dict,
    config: MechanismControlConfig,
):
    """画同点方向、公平误差、短时机制和配对增量。"""
    colors = {
        "pullback_G": "#D946EF",
        "same_point_overlap_gradient": "#0F766E",
        "same_point_energy_hessian": "#D97706",
        "instantaneous_strain_S0": "#2563EB",
        "time_varying_strain_St": "#7C3AED",
        "truth": "#111827",
    }
    labels = {
        "pullback_G": "pullback G",
        "same_point_overlap_gradient": "overlap gradient",
        "same_point_energy_hessian": "energy Hessian",
        "instantaneous_strain_S0": r"initial strain $S_0$",
        "time_varying_strain_St": r"current strain $S_t$",
    }
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
    ax_plane, ax_time, ax_early, ax_gain = axes.ravel()

    condition = conditions[
        (conditions.seed == config.representative_seed)
        & np.isclose(conditions.delta, config.representative_delta)
    ].iloc[0]
    points = boundaries[
        boundaries.condition_index == condition.condition_index
    ].sort_values("v_index")
    snapshot = directions[
        (directions.condition_index == condition.condition_index)
        & np.isclose(directions.time, config.representative_time)
    ].sort_values("v_index")
    ax_plane.plot(
        points.true_boundary_u, points.v,
        color=colors["truth"], lw=2.2, label="true boundary",
    )
    arrow_specs = (
        ("G_direction_u", "G_direction_v", "pullback_G"),
        ("overlap_direction_u", "overlap_direction_v", "same_point_overlap_gradient"),
        ("energy_direction_u", "energy_direction_v", "same_point_energy_hessian"),
        ("strain_St_direction_u", "strain_St_direction_v", "time_varying_strain_St"),
    )
    offsets = np.linspace(-0.012, 0.012, len(arrow_specs))
    for (u_col, v_col, name), offset in zip(arrow_specs, offsets):
        subset = snapshot.iloc[::3]
        directions_u = subset[u_col].to_numpy().copy()
        directions_v = subset[v_col].to_numpy().copy()
        signs = np.where(directions_u < 0, -1.0, 1.0)
        ax_plane.quiver(
            subset.true_boundary_u.to_numpy() if "true_boundary_u" in subset else
            points.true_boundary_u.iloc[::3].to_numpy(),
            subset.v.to_numpy() + offset,
            directions_u * signs,
            directions_v * signs,
            color=colors[name], scale=20, width=0.006,
            label=labels[name], alpha=0.9,
        )
    ax_plane.set(
        title=("A  All directions read at the same true-boundary points "
               f"(t={config.representative_time:g})"),
        xlabel=r"query coordinate $u$",
        ylabel=r"perpendicular coordinate $v$",
        xlim=(0.43, 0.68),
        ylim=(config.v_min - 0.03, config.v_max + 0.03),
    )
    ax_plane.legend(frameon=False, fontsize=7, ncol=2)

    moving = directions[directions.delta.abs() > np.finfo(float).eps]
    for name, column in ESTIMATOR_COLUMNS.items():
        if name == "instantaneous_strain_S0":
            continue
        by_time = moving.groupby("time")[column].median()
        ax_time.plot(
            by_time.index, by_time.values,
            color=colors[name], lw=2, label=labels[name],
        )
    s0_level = moving.strain_S0_angle_deg.median()
    ax_time.axhline(
        s0_level, color=colors["instantaneous_strain_S0"],
        lw=1.8, ls="--", label=labels["instantaneous_strain_S0"],
    )
    ax_time.set(
        title="B  Fair same-location angle to the true normal",
        xlabel="time",
        ylabel="acute angle (degrees)",
        ylim=(-0.3, 18),
    )
    ax_time.legend(frameon=False, fontsize=7, ncol=2)

    early_limit = min(0.5, config.metric_steps * config.dt)
    early = moving[moving.time <= early_limit]
    alignment = early.groupby("time").G_vs_S0_angle_deg.median()
    linear_error = early.groupby("time").G_linearization_relative_error.median()
    ax_early.plot(
        alignment.index, alignment.values,
        color=colors["pullback_G"], lw=2.2,
        label=r"angle$(e_{max}(G),e_{max}(S_0))$",
    )
    ax_early.set(
        title="C  Does initial local strain generate the early G direction?",
        xlabel="time",
        ylabel="direction angle (degrees)",
    )
    twin = ax_early.twinx()
    twin.plot(
        linear_error.index, linear_error.values,
        color="#64748B", lw=1.8, ls=":",
        label=r"relative error of $G_0+tS_0$",
    )
    twin.set_ylabel("relative linearization error", color="#64748B")
    lines = ax_early.get_lines() + twin.get_lines()
    ax_early.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8)

    paired = summary["paired_seed_angle_gain_over_G_deg"]
    order = [
        "same_point_overlap_gradient",
        "same_point_energy_hessian",
        "instantaneous_strain_S0",
        "time_varying_strain_St",
    ]
    medians = np.array([paired[name]["median"] for name in order])
    lower = medians - np.array([paired[name]["ci95_low"] for name in order])
    upper = np.array([paired[name]["ci95_high"] for name in order]) - medians
    x = np.arange(len(order))
    ax_gain.bar(x, medians, color=[colors[name] for name in order], alpha=0.86)
    ax_gain.errorbar(
        x, medians, yerr=np.vstack([lower, upper]), fmt="none",
        ecolor=colors["truth"], capsize=4, lw=1.2,
    )
    ax_gain.axhline(
        config.independent_angle_margin_deg,
        color=colors["truth"], ls=":", lw=1.2, label="frozen 1° margin",
    )
    ax_gain.axhline(0, color="#94A3B8", lw=0.8)
    for index, value in enumerate(medians):
        if abs(value) < 0.05:
            label_y = 0.13 + 0.12 * (index % 2)
            label = f"{value:.1e}°"
        else:
            label_y = value + 0.18
            label = f"{value:.2f}°"
        ax_gain.text(
            index, label_y, label,
            ha="center", va="bottom", fontsize=8, color=colors["truth"],
        )
    ax_gain.set_xticks(x, ["overlap", "energy", r"$S_0$", r"$S_t$"])
    ax_gain.set(
        title="D  Paired seed disadvantage relative to G",
        ylabel="baseline angle - G angle (degrees)",
    )
    ax_gain.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Is finite-time pullback geometry more than a same-point local control?",
        fontsize=14, fontweight="bold",
    )
    return fig


def write_artifacts(
    directory: str | Path,
    conditions: pd.DataFrame,
    boundaries: pd.DataFrame,
    directions: pd.DataFrame,
    raw: dict,
    config: MechanismControlConfig,
) -> tuple[dict, Path]:
    """保存配置、原始观测、图和不掩盖负结果的结论草稿。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    estimator_summary = summarize_estimators(conditions, directions, config)
    summary = experiment_summary(
        conditions, directions, estimator_summary, config
    )
    conditions.to_csv(directory / "conditions.csv", index=False)
    boundaries.to_csv(directory / "boundary_points.csv", index=False)
    directions.to_csv(
        directory / "direction_controls.csv.gz", index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    estimator_summary.to_csv(directory / "estimator_summary.csv", index=False)
    np.savez_compressed(
        directory / "mechanism_arrays.npz",
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
        conditions, boundaries, directions, estimator_summary, summary, config
    )
    figure.savefig(directory / "main_figure.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / "main_figure.pdf", bbox_inches="tight")
    angles = summary["moving_boundary_median_angle_after_start_deg"]
    paired = summary["paired_seed_angle_gain_over_G_deg"]
    outcome_text = {
        "full_dynamic_advantage": (
            "有限时间 G 同时超过同点 overlap、energy 与初始应变 S0，"
            "支持进入全平面盲定位实验。"
        ),
        "local_stability_equivalent": (
            "G 超过同点 overlap 与 energy，但没有超过初始应变 S0；"
            "其机制应降级为局部线性稳定性的有限时间表达。"
        ),
        "same_location_controls_remove_G_advantage": (
            "同位置控制消除了 G 对 overlap 或 energy 的冻结裕量优势；"
            "实验 3 的基线差距主要与读取位置不匹配一致。"
        ),
    }[summary["mechanism_outcome"]]
    conclusion = (
        "# 实验 4 结论草稿\n\n"
        f"- 机制判定：`{summary['mechanism_outcome']}`。\n"
        f"- 拓扑有效比例：{summary['valid_topology_fraction']:.3f}。\n"
        f"- t >= {config.direction_evaluation_start_time:g} 的移动边界中位夹角："
        f"G={angles['pullback_G']:.3g}°，"
        f"同点 overlap={angles['same_point_overlap_gradient']:.3g}°，"
        f"同点 energy Hessian={angles['same_point_energy_hessian']:.3g}°，"
        f"S0={angles['instantaneous_strain_S0']:.3g}°，"
        f"St={angles['time_varying_strain_St']:.3g}°。\n"
        f"- t={config.early_mechanism_time:g} 时 G 与 S0 方向中位夹角："
        f"{summary['median_G_vs_S0_angle_at_early_time_deg']:.3g}°；"
        "短时线性近似相对误差："
        f"{summary['median_G_linearization_relative_error_at_early_time']:.3g}。\n"
        "- 相对 G 的配对 seed 夹角劣势及 95% bootstrap CI：\n"
        + "".join(
            f"  - {name}: {result['median']:.3g}° "
            f"[{result['ci95_low']:.3g}, {result['ci95_high']:.3g}]。\n"
            for name, result in paired.items()
        )
        + "\n判读规则：只有 G 对同点 overlap、同点 energy 和固定的初始局部应变 "
        "S0 都超过冻结的 1° 裕量，才进入全平面盲定位。St 是逐时刻诊断量，"
        "因为它使用了 Jt 和 xt，所以单独报告，不与 S0 混为一个基线。\n\n"
        f"{outcome_text}\n"
    )
    conclusion_path = directory / "conclusion_draft.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    return summary, conclusion_path


def main() -> None:
    config = MechanismControlConfig()
    conditions, boundaries, directions, raw = run_experiment(config)
    output = Path(__file__).resolve().parent / "artifacts" / "experiment_04"
    summary, conclusion = write_artifacts(
        output, conditions, boundaries, directions, raw, config
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"artifacts: {output}")
    print(f"conclusion: {conclusion}")


if __name__ == "__main__":
    main()
