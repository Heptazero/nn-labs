"""三模式 Hopfield 真假记忆衰减率门控实验。

主检验只使用轨迹前缀中的 ||dx/dt||。Jacobian 谱仅作为曲率解释的
oracle 对照，绝不进入门控分类器。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CurvatureGateConfig:
    N: int = 20
    P: int = 3
    beta_primary: float = 8.0
    beta_robustness: tuple[float, ...] = (6.0, 10.0)
    seed_start: int = 10_000
    bank_count: int = 256
    equilibrium_dt: float = 0.03
    equilibrium_max_steps: int = 4_000
    equilibrium_tol: float = 1e-9
    stability_margin_tol: float = 1e-5
    trajectory_dt: float = 0.01
    trajectory_horizon: float = 4.0
    seed_fit_window: tuple[float, float] = (0.25, 1.25)
    local_fit_window: tuple[float, float] = (2.0, 4.0)
    perturbation_norm: float = 0.05
    perturbations_per_state: int = 8
    r2_min: float = 0.95
    min_eligible_banks: int = 30
    min_valid_fraction: float = 0.90
    min_balanced_accuracy: float = 0.75
    min_auc: float = 0.80
    min_abs_paired_effect: float = 0.80
    min_oracle_spearman: float = 0.70
    bootstrap_repetitions: int = 10_000

    def validate(self) -> None:
        if self.P != 3:
            raise ValueError("本实验冻结为 P=3；奇数多数混合态才没有 sign(0) 歧义")
        if self.N < 4 or self.bank_count < 5:
            raise ValueError("N 和 bank_count 太小")
        if self.seed_start < 0 or self.perturbations_per_state < 1:
            raise ValueError("seed 与扰动数必须有效")
        if self.beta_primary <= 0 or any(x <= 0 for x in self.beta_robustness):
            raise ValueError("beta 必须为正")
        if self.equilibrium_dt <= 0 or self.trajectory_dt <= 0:
            raise ValueError("积分步长必须为正")
        if self.trajectory_horizon < max(*self.seed_fit_window, *self.local_fit_window):
            raise ValueError("轨迹时长没有覆盖固定拟合窗")
        if not 0 < self.perturbation_norm < 1:
            raise ValueError("局部扰动范数应在 (0,1) 内")

    @property
    def betas(self) -> tuple[float, ...]:
        return (self.beta_primary, *self.beta_robustness)

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_start + self.bank_count)


def make_pattern_bank(seed: int, N: int = 20, P: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=(P, N))


def majority_mixture(patterns: np.ndarray) -> np.ndarray:
    if patterns.ndim != 2 or patterns.shape[0] % 2 != 1:
        raise ValueError("多数混合态要求奇数个二维模式")
    summed = patterns.sum(axis=0)
    if np.any(summed == 0):
        raise AssertionError("奇数个 ±1 模式不应产生逐位平票")
    return np.where(summed > 0, 1.0, -1.0)


def hebbian_weights(patterns: np.ndarray) -> np.ndarray:
    N = patterns.shape[1]
    weights = patterns.T @ patterns / N
    np.fill_diagonal(weights, 0.0)
    return weights


def vector_field(states: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    """x_dot = -x + tanh(beta W x)，最后一维是神经元。"""
    return -states + np.tanh(beta * (states @ weights.T))


def rk4_step(states: np.ndarray, weights: np.ndarray, beta: float, dt: float) -> np.ndarray:
    k1 = vector_field(states, weights, beta)
    k2 = vector_field(states + 0.5 * dt * k1, weights, beta)
    k3 = vector_field(states + 0.5 * dt * k2, weights, beta)
    k4 = vector_field(states + dt * k3, weights, beta)
    return states + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def settle_states(
    starts: np.ndarray,
    weights: np.ndarray,
    beta: float,
    *,
    dt: float,
    max_steps: int,
    tolerance: float,
) -> tuple[np.ndarray, bool, int, float]:
    states = starts.astype(np.float64, copy=True)
    residual = math.inf
    for step in range(max_steps + 1):
        velocity = vector_field(states, weights, beta)
        residual = float(np.linalg.norm(velocity, axis=-1).max())
        if residual <= tolerance:
            return states, True, step, residual
        if step < max_steps:
            states = rk4_step(states, weights, beta, dt)
    return states, False, max_steps, residual


def contraction_margin(state: np.ndarray, weights: np.ndarray, beta: float) -> float:
    field = beta * (weights @ state)
    derivative = beta * (1.0 - np.tanh(field) ** 2)
    jacobian = -np.eye(state.size) + np.diag(derivative) @ weights
    return float(-np.linalg.eigvals(jacobian).real.max())


def speed_trajectories(
    starts: np.ndarray,
    weights: np.ndarray,
    beta: float,
    *,
    dt: float,
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(round(horizon / dt))
    if not math.isclose(steps * dt, horizon, rel_tol=0, abs_tol=1e-12):
        raise ValueError("horizon 必须是 dt 的整数倍")
    states = starts.astype(np.float64, copy=True)
    times = np.arange(steps + 1, dtype=np.float64) * dt
    speeds = np.empty((steps + 1, states.shape[0]), dtype=np.float64)
    for step in range(steps + 1):
        velocity = vector_field(states, weights, beta)
        speeds[step] = np.linalg.norm(velocity, axis=-1)
        if step < steps:
            states = rk4_step(states, weights, beta, dt)
    return times, speeds


def fit_exponential_rates(
    times: np.ndarray,
    speeds: np.ndarray,
    window: tuple[float, float],
    *,
    speed_floor: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """固定时间窗拟合 log ||x_dot|| = intercept - rate * t。"""
    mask = (times >= window[0]) & (times <= window[1])
    selected_times = times[mask]
    if selected_times.size < 3:
        raise ValueError("固定拟合窗内至少需要三个时间点")
    selected = np.maximum(speeds[mask], speed_floor)
    centered_t = selected_times - selected_times.mean()
    centered_y = np.log(selected) - np.log(selected).mean(axis=0, keepdims=True)
    slopes = (centered_t[:, None] * centered_y).sum(axis=0) / np.square(centered_t).sum()
    fitted_centered = centered_t[:, None] * slopes[None, :]
    ss_res = np.square(centered_y - fitted_centered).sum(axis=0)
    ss_tot = np.square(centered_y).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return -slopes, r2


def matched_directions(seed: int, count: int, N: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, N))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def inspect_bank(patterns: np.ndarray, beta: float, config: CurvatureGateConfig) -> dict:
    mixture = majority_mixture(patterns)
    intended = np.vstack([patterns, mixture])
    weights = hebbian_weights(patterns)
    equilibria, converged, steps, residual = settle_states(
        intended,
        weights,
        beta,
        dt=config.equilibrium_dt,
        max_steps=config.equilibrium_max_steps,
        tolerance=config.equilibrium_tol,
    )
    signed = np.where(equilibria >= 0, 1.0, -1.0)
    sign_match = bool(np.all(signed == intended))
    distinct = len({tuple(row.tolist()) for row in signed}) == intended.shape[0]
    margins = np.array([contraction_margin(x, weights, beta) for x in equilibria])
    stable = bool(np.all(margins > config.stability_margin_tol))
    reasons = []
    if not converged:
        reasons.append("equilibrium_not_converged")
    if not sign_match:
        reasons.append("wrong_attractor_identity")
    if not distinct:
        reasons.append("attractors_not_distinct")
    if not stable:
        reasons.append("nonstable_equilibrium")
    return {
        "eligible": not reasons,
        "reason": "eligible" if not reasons else ";".join(reasons),
        "weights": weights,
        "intended": intended,
        "equilibria": equilibria,
        "margins": margins,
        "settle_steps": steps,
        "residual": residual,
    }


def run_bank(seed: int, beta: float, config: CurvatureGateConfig) -> tuple[dict, list[dict]]:
    patterns = make_pattern_bank(seed, config.N, config.P)
    bank = inspect_bank(patterns, beta, config)
    status = {
        "beta": beta,
        "pattern_seed": seed,
        "eligible": bank["eligible"],
        "reason": bank["reason"],
        "settle_steps": bank["settle_steps"],
        "equilibrium_residual": bank["residual"],
        "min_contraction_margin": float(bank["margins"].min()),
    }
    if not bank["eligible"]:
        return status, []

    rows: list[dict] = []
    labels = [("pure", i) for i in range(config.P)] + [("mixture", 3)]
    times, speeds = speed_trajectories(
        bank["intended"], bank["weights"], beta,
        dt=config.trajectory_dt, horizon=config.trajectory_horizon,
    )
    rates, r2 = fit_exponential_rates(times, speeds, config.seed_fit_window)
    for index, (kind, memory_id) in enumerate(labels):
        rows.append({
            "beta": beta,
            "pattern_seed": seed,
            "probe": "seed_relaxation",
            "memory_kind": kind,
            "memory_id": memory_id,
            "direction_id": -1,
            "rate": float(rates[index]),
            "r2": float(r2[index]),
            "initial_speed": float(speeds[0, index]),
            "final_speed": float(speeds[-1, index]),
            "oracle_margin": float(bank["margins"][index]),
        })

    directions = matched_directions(2_000_000 + seed, config.perturbations_per_state, config.N)
    local_starts = (
        bank["equilibria"][:, None, :]
        + config.perturbation_norm * directions[None, :, :]
    ).reshape(-1, config.N)
    times, speeds = speed_trajectories(
        local_starts, bank["weights"], beta,
        dt=config.trajectory_dt, horizon=config.trajectory_horizon,
    )
    rates, r2 = fit_exponential_rates(times, speeds, config.local_fit_window)
    cursor = 0
    for state_index, (kind, memory_id) in enumerate(labels):
        for direction_id in range(config.perturbations_per_state):
            rows.append({
                "beta": beta,
                "pattern_seed": seed,
                "probe": "matched_local_probe",
                "memory_kind": kind,
                "memory_id": memory_id,
                "direction_id": direction_id,
                "rate": float(rates[cursor]),
                "r2": float(r2[cursor]),
                "initial_speed": float(speeds[0, cursor]),
                "final_speed": float(speeds[-1, cursor]),
                "oracle_margin": float(bank["margins"][state_index]),
            })
            cursor += 1
    return status, rows


def _mean_by_bank(rows: Iterable[dict], beta: float, probe: str, value: str) -> dict[int, dict[str, float]]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row["beta"] == beta and row["probe"] == probe and math.isfinite(row[value]):
            grouped.setdefault((int(row["pattern_seed"]), row["memory_kind"]), []).append(float(row[value]))
    banks: dict[int, dict[str, float]] = {}
    for (seed, kind), values in grouped.items():
        banks.setdefault(seed, {})[kind] = float(np.median(values))
    return {seed: values for seed, values in banks.items() if set(values) == {"pure", "mixture"}}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    ranks = _rankdata(scores)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])


def cross_validated_scalar_gate(bank_values: dict[int, dict[str, float]]) -> dict:
    seeds = np.array(sorted(bank_values))
    predictions, labels, scores = [], [], []
    fold_records = []
    for fold in range(5):
        test = seeds[seeds % 5 == fold]
        train = seeds[seeds % 5 != fold]
        if test.size == 0 or train.size == 0:
            continue
        train_pure = np.array([bank_values[int(s)]["pure"] for s in train])
        train_mix = np.array([bank_values[int(s)]["mixture"] for s in train])
        orientation = 1.0 if np.median(train_mix) >= np.median(train_pure) else -1.0
        threshold = 0.5 * (np.median(train_mix) + np.median(train_pure))
        fold_records.append({"fold": fold, "orientation": orientation, "threshold": float(threshold)})
        for seed in test:
            for label, kind in ((0, "pure"), (1, "mixture")):
                value = bank_values[int(seed)][kind]
                score = orientation * (value - threshold)
                labels.append(label)
                # AUC 只需要折内学得的方向；不同折的阈值不应改变跨折排序。
                scores.append(orientation * value)
                predictions.append(int(score >= 0))
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    sensitivity = float(np.mean(predictions_array[labels_array == 1] == 1))
    specificity = float(np.mean(predictions_array[labels_array == 0] == 0))
    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "auc": _auc(labels_array, scores_array),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "folds": fold_records,
    }


def summarize_probe(rows: list[dict], statuses: list[dict], beta: float, probe: str, config: CurvatureGateConfig) -> dict:
    rates = _mean_by_bank(rows, beta, probe, "rate")
    initial_speeds = _mean_by_bank(rows, beta, probe, "initial_speed")
    seeds = sorted(rates)
    pure = np.array([rates[s]["pure"] for s in seeds])
    mixture = np.array([rates[s]["mixture"] for s in seeds])
    raw_delta = mixture - pure
    orientation = 1.0 if np.median(raw_delta) >= 0 else -1.0
    oriented_delta = orientation * raw_delta
    std = float(np.std(oriented_delta, ddof=1)) if len(oriented_delta) > 1 else math.nan
    effect = float(np.mean(oriented_delta) / std) if std > 0 else math.inf
    rng = np.random.default_rng(3_000_000 + int(beta * 1000) + (0 if probe == "seed_relaxation" else 1))
    if len(oriented_delta):
        indices = rng.integers(0, len(oriented_delta), size=(config.bootstrap_repetitions, len(oriented_delta)))
        boot = oriented_delta[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    else:
        ci_low = ci_high = math.nan
    matching_rows = [r for r in rows if r["beta"] == beta and r["probe"] == probe]
    valid_fraction = float(np.mean([r["r2"] >= config.r2_min for r in matching_rows])) if matching_rows else 0.0
    gate = cross_validated_scalar_gate(rates) if rates else {
        "balanced_accuracy": math.nan, "auc": math.nan,
        "sensitivity": math.nan, "specificity": math.nan, "folds": [],
    }
    speed_gate = cross_validated_scalar_gate(initial_speeds) if initial_speeds else {
        "balanced_accuracy": math.nan, "auc": math.nan,
    }
    eligible = sum(s["beta"] == beta and s["eligible"] for s in statuses)
    passed = bool(
        eligible >= config.min_eligible_banks
        and valid_fraction >= config.min_valid_fraction
        and gate["balanced_accuracy"] >= config.min_balanced_accuracy
        and gate["auc"] >= config.min_auc
        and ci_low > 0
        and abs(effect) >= config.min_abs_paired_effect
    )
    return {
        "beta": beta,
        "probe": probe,
        "eligible_banks": eligible,
        "total_banks": config.bank_count,
        "valid_exponential_fraction": valid_fraction,
        "orientation": "mixture_faster" if orientation > 0 else "mixture_slower",
        "pure_rate_median": float(np.median(pure)) if len(pure) else math.nan,
        "mixture_rate_median": float(np.median(mixture)) if len(mixture) else math.nan,
        "oriented_paired_mean_gap": float(np.mean(oriented_delta)) if len(oriented_delta) else math.nan,
        "paired_gap_ci95": [float(ci_low), float(ci_high)],
        "paired_effect_dz": effect,
        "rate_gate": gate,
        "initial_speed_gate": speed_gate,
        "pass": passed,
    }


def oracle_agreement(rows: list[dict], beta: float) -> float:
    fitted = _mean_by_bank(rows, beta, "matched_local_probe", "rate")
    oracle = _mean_by_bank(rows, beta, "matched_local_probe", "oracle_margin")
    common = sorted(set(fitted) & set(oracle))
    x, y = [], []
    for seed in common:
        for kind in ("pure", "mixture"):
            x.append(fitted[seed][kind])
            y.append(oracle[seed][kind])
    return _spearman(np.asarray(x), np.asarray(y))


def analyze(rows: list[dict], statuses: list[dict], config: CurvatureGateConfig) -> dict:
    summaries = [
        summarize_probe(rows, statuses, beta, probe, config)
        for beta in config.betas
        for probe in ("seed_relaxation", "matched_local_probe")
    ]
    primary_seed = next(x for x in summaries if x["beta"] == config.beta_primary and x["probe"] == "seed_relaxation")
    primary_local = next(x for x in summaries if x["beta"] == config.beta_primary and x["probe"] == "matched_local_probe")
    oracle = oracle_agreement(rows, config.beta_primary)
    if not primary_seed["pass"]:
        decision = "KILL_RATE_GATE"
    elif not primary_local["pass"] or not math.isfinite(oracle) or oracle < config.min_oracle_spearman:
        decision = "RATE_SIGNAL_ONLY_CURVATURE_CLAIM_FAILS"
    else:
        decision = "PASS_RATE_AND_CURVATURE_GATE"
    return {
        "decision": decision,
        "primary_beta": config.beta_primary,
        "primary_oracle_spearman": oracle,
        "summaries": summaries,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("拒绝写入空结果表")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(output_dir: Path, config: CurvatureGateConfig | None = None) -> dict:
    config = config or CurvatureGateConfig()
    config.validate()
    statuses, rows = [], []
    for beta in config.betas:
        for seed in config.seeds:
            status, bank_rows = run_bank(seed, beta, config)
            statuses.append(status)
            rows.extend(bank_rows)
    result = analyze(rows, statuses, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "bank_status.csv", statuses)
    _write_csv(output_dir / "trajectory_rates.csv", rows)
    config_dict = asdict(config)
    protocol_bytes = json.dumps(config_dict, sort_keys=True, ensure_ascii=False).encode("utf-8")
    result["config"] = config_dict
    result["config_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float64",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
