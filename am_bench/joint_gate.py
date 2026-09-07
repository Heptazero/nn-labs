"""反 Hebb 联合动力学：比较始终开启、初始速度门与衰减率门。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from .curvature_gate import (
    fit_exponential_rates,
    inspect_bank,
    make_pattern_bank,
    speed_trajectories,
    vector_field,
    CurvatureGateConfig,
    _spearman,
)


@dataclass(frozen=True)
class JointGateConfig:
    N: int = 20
    P: int = 3
    beta: float = 8.0
    seed_start: int = 20_000
    bank_count: int = 256
    probe_dt: float = 0.01
    probe_horizon: float = 1.25
    rate_fit_window: tuple[float, float] = (0.25, 1.25)
    # 由上一批 seed=10000..10255 的合格库冻结；评估集不再改阈值。
    rate_threshold: float = 0.9165554262224024
    initial_speed_threshold: float = 0.05095881235895465
    coupled_dt: float = 0.02
    coupled_horizon: float = 30.0
    gamma: float = 0.5
    restore_mu: float = 0.2
    plasticity_budget: float = 1.0
    convergence_tol: float = 1e-4
    weight_recovery_tol: float = 0.01
    min_eligible_banks: int = 30
    min_escape_advantage: float = 0.10
    max_damage_disadvantage: float = 0.00

    def validate(self) -> None:
        if (self.N, self.P) != (20, 3):
            raise ValueError("联合实验冻结为 N=20, P=3")
        positives = (
            self.beta, self.probe_dt, self.probe_horizon, self.coupled_dt,
            self.coupled_horizon, self.gamma, self.restore_mu, self.plasticity_budget,
        )
        if any(not math.isfinite(x) or x <= 0 for x in positives):
            raise ValueError("动力学参数必须是有限正数")
        if self.probe_horizon < self.rate_fit_window[1]:
            raise ValueError("探测前缀没有覆盖速率拟合窗")
        for horizon, dt in ((self.probe_horizon, self.probe_dt), (self.coupled_horizon, self.coupled_dt)):
            if not math.isclose(round(horizon / dt) * dt, horizon, abs_tol=1e-12):
                raise ValueError("时间长度必须是步长的整数倍")
        if not math.isclose(round(self.plasticity_budget / self.coupled_dt) * self.coupled_dt,
                            self.plasticity_budget, abs_tol=1e-12):
            raise ValueError("削弱预算必须是 coupled_dt 的整数倍")

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_start + self.bank_count)


ARMS = ("frozen", "always_on", "initial_speed_gate", "decay_rate_gate")


def static_probe(starts: np.ndarray, weights: np.ndarray, config: JointGateConfig) -> dict:
    times, speeds = speed_trajectories(
        starts, weights, config.beta, dt=config.probe_dt, horizon=config.probe_horizon,
    )
    rates, r2 = fit_exponential_rates(times, speeds, config.rate_fit_window)
    states = starts.astype(np.float64, copy=True)
    # 与记录速度所用的 RK4 定义一致；这里只需要探测窗末状态。
    from .curvature_gate import rk4_step
    for _ in range(len(times) - 1):
        states = rk4_step(states, weights, config.beta, config.probe_dt)
    return {
        "end_states": states,
        "rates": rates,
        "r2": r2,
        "initial_speeds": speeds[0],
        "end_speeds": speeds[-1],
    }


def anti_hebbian_direction(states: np.ndarray) -> np.ndarray:
    """每条轨迹独立的 -xx^T/N，保持对称并固定零对角。"""
    N = states.shape[1]
    direction = -np.einsum("bi,bj->bij", states, states) / N
    diagonal = np.arange(N)
    direction[:, diagonal, diagonal] = 0.0
    return direction


def coupled_vector_field(
    states: np.ndarray,
    weights: np.ndarray,
    base_weights: np.ndarray,
    active: np.ndarray,
    config: JointGateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    state_velocity = -states + np.tanh(
        config.beta * np.einsum("bij,bj->bi", weights, states)
    )
    plasticity = config.gamma * active[:, None, None] * anti_hebbian_direction(states)
    weight_velocity = plasticity - config.restore_mu * (weights - base_weights)
    return state_velocity, weight_velocity


def coupled_rk4_step(
    states: np.ndarray,
    weights: np.ndarray,
    base_weights: np.ndarray,
    active: np.ndarray,
    config: JointGateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dt = config.coupled_dt
    x1, w1 = coupled_vector_field(states, weights, base_weights, active, config)
    x2, w2 = coupled_vector_field(states + 0.5 * dt * x1, weights + 0.5 * dt * w1,
                                  base_weights, active, config)
    x3, w3 = coupled_vector_field(states + 0.5 * dt * x2, weights + 0.5 * dt * w2,
                                  base_weights, active, config)
    x4, w4 = coupled_vector_field(states + dt * x3, weights + dt * w3,
                                  base_weights, active, config)
    next_states = states + dt * (x1 + 2 * x2 + 2 * x3 + x4) / 6
    next_weights = weights + dt * (w1 + 2 * w2 + 2 * w3 + w4) / 6
    # 浮点误差也不允许破坏理论约束。
    next_weights = 0.5 * (next_weights + np.swapaxes(next_weights, 1, 2))
    diagonal = np.arange(next_weights.shape[1])
    next_weights[:, diagonal, diagonal] = 0.0
    return next_states, next_weights


def run_coupled_trials(
    starts: np.ndarray,
    base_weight: np.ndarray,
    enabled: np.ndarray,
    config: JointGateConfig,
) -> dict:
    trial_count = starts.shape[0]
    states = starts.astype(np.float64, copy=True)
    if base_weight.ndim == 2:
        base_weights = np.repeat(base_weight[None, :, :], trial_count, axis=0)
    elif base_weight.shape == (trial_count, states.shape[1], states.shape[1]):
        base_weights = base_weight.astype(np.float64, copy=True)
    else:
        raise ValueError("base_weight 必须是单个矩阵或逐轨迹矩阵")
    weights = base_weights.copy()
    exposures = np.zeros(trial_count, dtype=np.float64)
    max_weight_deviation = np.zeros(trial_count, dtype=np.float64)
    tail_speeds = []
    total_steps = int(round(config.coupled_horizon / config.coupled_dt))
    tail_steps = int(round(1.0 / config.coupled_dt))
    for step in range(total_steps):
        active = enabled & (exposures < config.plasticity_budget - 0.5 * config.coupled_dt)
        states, weights = coupled_rk4_step(states, weights, base_weights, active.astype(float), config)
        exposures += active.astype(float) * config.coupled_dt
        deviations = np.linalg.norm(weights - base_weights, axis=(1, 2))
        max_weight_deviation = np.maximum(max_weight_deviation, deviations)
        if step >= total_steps - tail_steps:
            dx, _ = coupled_vector_field(states, weights, base_weights, np.zeros(trial_count), config)
            tail_speeds.append(np.linalg.norm(dx, axis=1))
    final_dx, final_dw = coupled_vector_field(
        states, weights, base_weights, np.zeros(trial_count), config
    )
    tail = np.asarray(tail_speeds)
    return {
        "states": states,
        "weights": weights,
        "exposures": exposures,
        "max_weight_deviation": max_weight_deviation,
        "final_weight_deviation": np.linalg.norm(weights - base_weights, axis=(1, 2)),
        "final_state_speed": np.linalg.norm(final_dx, axis=1),
        "final_weight_speed": np.linalg.norm(final_dw, axis=(1, 2)),
        "tail_speed_max": tail.max(axis=0),
    }


def classify_state(state: np.ndarray, patterns: np.ndarray, mixture: np.ndarray) -> tuple[str, int]:
    signed = np.where(state >= 0, 1.0, -1.0)
    matches = np.flatnonzero(np.all(patterns == signed[None, :], axis=1))
    if matches.size:
        return "pure", int(matches[0])
    if np.array_equal(signed, mixture):
        return "mixture", -1
    return "other", -1


def outcome_flags(
    start_kind: str, start_id: int, outcome: str, pure_id: int, converged: bool,
) -> tuple[bool, bool]:
    """不收敛轨迹即使瞬时落在纯模式符号上，也不得记为成功。"""
    target_preserved = bool(
        converged and start_kind == "pure" and outcome == "pure" and pure_id == start_id
    )
    mixture_escaped = bool(converged and start_kind == "mixture" and outcome == "pure")
    return target_preserved, mixture_escaped


def run_joint_bank(seed: int, config: JointGateConfig, eligibility: CurvatureGateConfig) -> tuple[dict, list[dict]]:
    patterns = make_pattern_bank(seed, config.N, config.P)
    bank = inspect_bank(patterns, config.beta, eligibility)
    status = {
        "pattern_seed": seed,
        "eligible": bank["eligible"],
        "reason": bank["reason"],
    }
    if not bank["eligible"]:
        return status, []

    probe = static_probe(bank["intended"], bank["weights"], config)
    speed_decisions = probe["initial_speeds"] > config.initial_speed_threshold
    rate_decisions = probe["rates"] < config.rate_threshold
    rows = []
    starts, enabled, metadata = [], [], []
    for state_index in range(config.P + 1):
        kind = "pure" if state_index < config.P else "mixture"
        for arm in ARMS:
            if arm == "frozen":
                activation = False
            elif arm == "always_on":
                activation = True
            elif arm == "initial_speed_gate":
                activation = bool(speed_decisions[state_index])
            else:
                activation = bool(rate_decisions[state_index])
            starts.append(probe["end_states"][state_index])
            enabled.append(activation)
            metadata.append((state_index, kind, arm, activation))

    result = run_coupled_trials(
        np.asarray(starts), bank["weights"], np.asarray(enabled, dtype=bool), config
    )
    mixture = bank["intended"][-1]
    for trial, (state_index, kind, arm, activation) in enumerate(metadata):
        outcome, pure_id = classify_state(result["states"][trial], patterns, mixture)
        converged = bool(
            result["tail_speed_max"][trial] <= config.convergence_tol
            and result["final_weight_deviation"][trial] <= config.weight_recovery_tol
        )
        target_preserved, mixture_escaped = outcome_flags(
            kind, state_index, outcome, pure_id, converged
        )
        rows.append({
            "pattern_seed": seed,
            "start_kind": kind,
            "start_id": state_index,
            "arm": arm,
            "gate_activated": activation,
            "probe_rate": float(probe["rates"][state_index]),
            "probe_r2": float(probe["r2"][state_index]),
            "probe_initial_speed": float(probe["initial_speeds"][state_index]),
            "outcome": outcome,
            "outcome_pure_id": pure_id,
            "target_preserved": target_preserved,
            "mixture_escaped": mixture_escaped,
            "converged": converged,
            "plasticity_exposure": float(result["exposures"][trial]),
            "max_weight_deviation": float(result["max_weight_deviation"][trial]),
            "final_weight_deviation": float(result["final_weight_deviation"][trial]),
            "final_state_speed": float(result["final_state_speed"][trial]),
            "tail_speed_max": float(result["tail_speed_max"][trial]),
        })
    return status, rows


def paired_bootstrap_difference(
    first: np.ndarray, second: np.ndarray, *, seed: int, repetitions: int = 10_000,
) -> tuple[float, list[float]]:
    if first.shape != second.shape or first.size == 0:
        raise ValueError("配对 bootstrap 需要非空等长输入")
    difference = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(repetitions, difference.size))
    samples = difference[indices].mean(axis=1)
    return float(difference.mean()), [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize_joint(rows: list[dict], statuses: list[dict], config: JointGateConfig) -> dict:
    eligible = sum(bool(x["eligible"]) for x in statuses)
    arm_summaries = {}
    seeds = sorted({int(row["pattern_seed"]) for row in rows})
    per_bank = {}
    for seed in seeds:
        per_bank[seed] = {}
        for arm in ARMS:
            subset = [r for r in rows if r["pattern_seed"] == seed and r["arm"] == arm]
            pure = [r for r in subset if r["start_kind"] == "pure"]
            mixture = [r for r in subset if r["start_kind"] == "mixture"]
            per_bank[seed][arm] = {
                "pure_damage": 1.0 - float(np.mean([r["target_preserved"] for r in pure])),
                "mixture_escape": float(mixture[0]["mixture_escaped"]),
            }
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        pure = [r for r in subset if r["start_kind"] == "pure"]
        mixture = [r for r in subset if r["start_kind"] == "mixture"]
        arm_summaries[arm] = {
            "gate_activation_pure": float(np.mean([r["gate_activated"] for r in pure])),
            "gate_activation_mixture": float(np.mean([r["gate_activated"] for r in mixture])),
            "pure_damage_rate": float(np.mean([not r["target_preserved"] for r in pure])),
            "mixture_escape_rate": float(np.mean([r["mixture_escaped"] for r in mixture])),
            "convergence_rate": float(np.mean([r["converged"] for r in subset])),
            "mean_exposure": float(np.mean([r["plasticity_exposure"] for r in subset])),
            "mean_max_weight_deviation": float(np.mean([r["max_weight_deviation"] for r in subset])),
        }

    rate_escape = np.array([per_bank[s]["decay_rate_gate"]["mixture_escape"] for s in seeds])
    speed_escape = np.array([per_bank[s]["initial_speed_gate"]["mixture_escape"] for s in seeds])
    rate_damage = np.array([per_bank[s]["decay_rate_gate"]["pure_damage"] for s in seeds])
    speed_damage = np.array([per_bank[s]["initial_speed_gate"]["pure_damage"] for s in seeds])
    escape_difference, escape_ci = paired_bootstrap_difference(
        rate_escape, speed_escape, seed=4_000_001,
    ) if seeds else (math.nan, [math.nan, math.nan])
    damage_difference, damage_ci = paired_bootstrap_difference(
        rate_damage, speed_damage, seed=4_000_002,
    ) if seeds else (math.nan, [math.nan, math.nan])
    always_escape = np.array([per_bank[s]["always_on"]["mixture_escape"] for s in seeds])
    always_damage = np.array([per_bank[s]["always_on"]["pure_damage"] for s in seeds])
    speed_vs_always_escape, speed_vs_always_escape_ci = paired_bootstrap_difference(
        speed_escape, always_escape, seed=4_000_003,
    ) if seeds else (math.nan, [math.nan, math.nan])
    speed_vs_always_damage, speed_vs_always_damage_ci = paired_bootstrap_difference(
        speed_damage, always_damage, seed=4_000_004,
    ) if seeds else (math.nan, [math.nan, math.nan])
    rate_adds_value = bool(
        eligible >= config.min_eligible_banks
        and escape_difference >= config.min_escape_advantage
        and escape_ci[0] > 0
        and damage_difference <= config.max_damage_disadvantage
        and damage_ci[1] <= config.max_damage_disadvantage
    )
    if eligible < config.min_eligible_banks:
        decision = "INCONCLUSIVE_TOO_FEW_ELIGIBLE_BANKS"
    elif rate_adds_value:
        decision = "DECAY_RATE_GATE_ADDS_VALUE"
    else:
        decision = "STOP_CURVATURE_NARRATIVE"
    probes = [r for r in rows if r["arm"] == "frozen"]
    speed_decisions = np.array([
        r["probe_initial_speed"] > config.initial_speed_threshold for r in probes
    ])
    rate_decisions = np.array([
        r["probe_rate"] < config.rate_threshold for r in probes
    ])
    probe_spearman = _spearman(
        np.array([r["probe_initial_speed"] for r in probes]),
        -np.array([r["probe_rate"] for r in probes]),
    ) if probes else math.nan
    return {
        "decision": decision,
        "eligible_banks": eligible,
        "total_banks": config.bank_count,
        "arms": arm_summaries,
        "decay_minus_initial_escape": escape_difference,
        "decay_minus_initial_escape_ci95": escape_ci,
        "decay_minus_initial_damage": damage_difference,
        "decay_minus_initial_damage_ci95": damage_ci,
        "initial_minus_always_escape": speed_vs_always_escape,
        "initial_minus_always_escape_ci95": speed_vs_always_escape_ci,
        "initial_minus_always_damage": speed_vs_always_damage,
        "initial_minus_always_damage_ci95": speed_vs_always_damage_ci,
        "gate_decision_agreement": float(np.mean(speed_decisions == rate_decisions)) if probes else math.nan,
        "gate_decision_disagreements": int(np.sum(speed_decisions != rate_decisions)) if probes else 0,
        "initial_speed_vs_negative_rate_spearman": probe_spearman,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("拒绝写入空结果")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_joint_experiment(output_dir: Path, config: JointGateConfig | None = None) -> dict:
    config = config or JointGateConfig()
    config.validate()
    eligibility = CurvatureGateConfig(
        N=config.N, P=config.P, beta_primary=config.beta, beta_robustness=(),
        seed_start=config.seed_start, bank_count=config.bank_count,
    )
    statuses, rows = [], []
    for seed in config.seeds:
        status, bank_rows = run_joint_bank(seed, config, eligibility)
        statuses.append(status)
        rows.extend(bank_rows)
    summary = summarize_joint(rows, statuses, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "bank_status.csv", statuses)
    _write_csv(output_dir / "joint_trials.csv", rows)
    summary["config"] = asdict(config)
    summary["config_sha256"] = hashlib.sha256(
        json.dumps(summary["config"], sort_keys=True).encode()
    ).hexdigest()
    summary["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["runtime"] = {
        "python": platform.python_version(), "numpy": np.__version__,
        "device": "cpu", "dtype": "float64",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
