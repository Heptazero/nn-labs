"""实验 05R：逐 cue 匹配第一步 IPR 的探索性方法预跑。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiment_05_multimemory_rank_collapse import (
    RankCollapseConfig,
    add_dimension_survival,
    attention_diagnostics,
    eigenspectrum,
    make_cues_for_targets,
    make_memories,
    memory_span_basis,
    run_numerical_self_checks,
    score_jacobian,
    select_formal_targets,
    spectrum_metrics,
    transform_scores,
)


torch.set_default_dtype(torch.float64)


LEVELS = ("low", "middle", "high")
METHODS = ("softmax", "sparsemax")


@dataclass(frozen=True)
class PerCuePilotConfig:
    """05R protocol 中冻结的输入、求根与停止门。"""

    N: int = 128
    K: int = 100
    memory_seeds: tuple[int, ...] = tuple(range(8))
    corruption_rates: tuple[float, ...] = (0.10, 0.25, 0.40)
    targets_per_seed: int = 32
    mask_count: int = 1
    target_iprs: tuple[float, ...] = (4.0, 16.0, 64.0)
    jacobian_steps: int = 12
    alpha_bracket_start: float = 1.0
    alpha_max: float = float(2**20)
    bisection_steps: int = 80
    target_ipr_relative_tolerance: float = 0.005
    pair_ipr_relative_tolerance: float = 0.005
    monotonicity_tolerance: float = 1e-10
    monotonicity_samples: int = 65
    tie_tolerance: float = 1e-12
    pair_attainability_threshold: float = 0.95
    interaction_signal_threshold: float = 0.05
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 50_500
    target_seed_offset: int = 8_000
    corruption_seed_offset: int = 20_000
    support_tolerance: float = 1e-12
    numeric_rank_relative_tolerance: float = 1e-10
    sensitivity_extinction_relative_tolerance: float = 1e-14
    collapsed_at_entry_tolerance: float = 1e-8
    dtype: str = "float64"

    def validate(self) -> None:
        if self.N <= self.K or self.K < 3:
            raise ValueError("05R requires N > K >= 3")
        if len(self.target_iprs) != 3:
            raise ValueError("05R requires low, middle, and high IPR targets")
        if tuple(sorted(self.target_iprs)) != self.target_iprs:
            raise ValueError("target IPRs must be strictly increasing")
        if any(not 1 < target < self.K for target in self.target_iprs):
            raise ValueError("target IPRs must lie strictly between 1 and K")
        if not self.memory_seeds or len(set(self.memory_seeds)) != len(self.memory_seeds):
            raise ValueError("memory seeds must be unique and non-empty")
        if not self.corruption_rates or any(not 0 < rho < 0.5 for rho in self.corruption_rates):
            raise ValueError("corruption rates must lie in (0, 0.5)")
        if not 1 <= self.targets_per_seed <= self.K:
            raise ValueError("invalid target count")
        if self.mask_count != 1:
            raise ValueError("the frozen 05R pilot uses mask 0 only")
        if self.alpha_bracket_start <= 0 or self.alpha_max <= self.alpha_bracket_start:
            raise ValueError("invalid alpha bracket")
        if self.bisection_steps < 1 or self.monotonicity_samples < 3:
            raise ValueError("solver budgets are too small")
        if self.jacobian_steps < 1 or self.bootstrap_samples < 1_000:
            raise ValueError("trajectory/bootstrap budgets are too small")
        if self.dtype != "float64":
            raise ValueError("05R numerical checks require float64")

    def base_config(self) -> RankCollapseConfig:
        """构造与实验 05 数学核心兼容的配置。"""
        return RankCollapseConfig(
            N=self.N,
            K=self.K,
            memory_seeds=self.memory_seeds,
            corruption_rates=self.corruption_rates,
            jacobian_steps=self.jacobian_steps,
            formal_jacobian_targets=self.targets_per_seed,
            development_targets=min(max(3, self.targets_per_seed), self.K - 1),
            development_masks=1,
            formal_representation_masks=2,
            target_seed_offset=self.target_seed_offset,
            corruption_seed_offset=self.corruption_seed_offset,
            support_tolerance=self.support_tolerance,
            numeric_rank_relative_tolerance=self.numeric_rank_relative_tolerance,
            sensitivity_extinction_relative_tolerance=(
                self.sensitivity_extinction_relative_tolerance
            ),
            collapsed_at_entry_tolerance=self.collapsed_at_entry_tolerance,
        )


def _ipr_from_score(score: torch.Tensor, alpha: float, method: str) -> float:
    weights = transform_scores((float(alpha) * score).unsqueeze(0), method)[0]
    return float(1.0 / weights.square().sum())


def _relative_error(value: float, target: float) -> float:
    return abs(value - target) / target


def _pair_relative_error(first: float, second: float) -> float:
    return abs(first - second) / max(0.5 * (first + second), np.finfo(float).tiny)


def solve_alpha_for_ipr(
    score: torch.Tensor,
    target_ipr: float,
    method: str,
    config: PerCuePilotConfig,
) -> dict:
    """只由第一步 score 求 alpha；返回可审计的 bracket 与误差。"""
    if score.ndim != 1 or len(score) != config.K:
        raise ValueError(f"score must have shape ({config.K},)")
    maximum = float(score.max())
    top_tie_count = int(
        torch.isclose(
            score,
            torch.tensor(maximum, dtype=score.dtype),
            rtol=0.0,
            atol=config.tie_tolerance,
        ).sum()
    )
    base = {
        "method": method,
        "target_ipr": float(target_ipr),
        "top_tie_count": top_tie_count,
        "alpha": np.nan,
        "achieved_ipr": np.nan,
        "target_relative_error": np.nan,
        "bracket_lower": 0.0,
        "bracket_upper": np.nan,
        "bisection_iterations": 0,
        "maximum_monotonicity_increase": np.nan,
        "monotonicity_passed": False,
        "status": "",
    }
    if top_tie_count > target_ipr:
        return {**base, "status": "unattainable_by_tie"}

    lower = 0.0
    upper = config.alpha_bracket_start
    upper_ipr = _ipr_from_score(score, upper, method)
    while upper_ipr > target_ipr and upper < config.alpha_max:
        upper = min(2.0 * upper, config.alpha_max)
        upper_ipr = _ipr_from_score(score, upper, method)
    if upper_ipr > target_ipr:
        return {
            **base,
            "bracket_upper": upper,
            "achieved_ipr": upper_ipr,
            "target_relative_error": _relative_error(upper_ipr, target_ipr),
            "status": "unbracketed",
        }

    # The monotonicity audit is independent of the bisection stopping choice.
    sample_alphas = torch.linspace(
        lower, upper, config.monotonicity_samples, dtype=score.dtype
    )
    sample_iprs = np.asarray([
        _ipr_from_score(score, float(alpha), method) for alpha in sample_alphas
    ])
    max_increase = float(np.diff(sample_iprs).max(initial=-np.inf))
    monotonicity_passed = bool(max_increase <= config.monotonicity_tolerance)

    candidates = [
        (lower, _ipr_from_score(score, lower, method)),
        (upper, upper_ipr),
    ]
    iterations = 0
    for iterations in range(1, config.bisection_steps + 1):
        middle = 0.5 * (lower + upper)
        middle_ipr = _ipr_from_score(score, middle, method)
        candidates.append((middle, middle_ipr))
        if middle_ipr > target_ipr:
            lower = middle
        else:
            upper = middle
    alpha, achieved = min(
        candidates, key=lambda item: _relative_error(item[1], target_ipr)
    )
    error = _relative_error(achieved, target_ipr)
    if not monotonicity_passed:
        status = "monotonicity_failure"
    elif error > config.target_ipr_relative_tolerance:
        status = "match_error"
    else:
        status = "matched"
    return {
        **base,
        "alpha": float(alpha),
        "achieved_ipr": float(achieved),
        "target_relative_error": float(error),
        "bracket_lower": 0.0,
        "bracket_upper": float(sample_alphas[-1]),
        "bisection_iterations": iterations,
        "maximum_monotonicity_increase": max_increase,
        "monotonicity_passed": monotonicity_passed,
        "status": status,
    }


def solve_cue_pairs(
    X: torch.Tensor,
    cues: torch.Tensor,
    metadata: pd.DataFrame,
    config: PerCuePilotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐 cue、逐 IPR 档位独立求两个方法，并作对称删失。"""
    scores = (cues @ X) / config.N
    solver_rows = []
    pair_rows = []
    for cue_index, meta in metadata.iterrows():
        score = scores[cue_index]
        for level, target_ipr in zip(LEVELS, config.target_iprs):
            solutions = {}
            for method in METHODS:
                solution = solve_alpha_for_ipr(score, target_ipr, method, config)
                solution.update({
                    "cue_index": int(cue_index),
                    "memory_seed": int(meta.memory_seed),
                    "target": int(meta.target),
                    "rho": float(meta.rho),
                    "mask_index": int(meta.mask_index),
                    "level": level,
                })
                solver_rows.append(solution)
                solutions[method] = solution

            statuses = {solutions[method]["status"] for method in METHODS}
            if "unattainable_by_tie" in statuses:
                pair_status = "unattainable_by_tie"
            elif "unbracketed" in statuses:
                pair_status = "unbracketed"
            elif "monotonicity_failure" in statuses:
                pair_status = "monotonicity_failure"
            elif "match_error" in statuses:
                pair_status = "match_error"
            else:
                pair_status = "matched"
            pair_error = np.nan
            if pair_status == "matched":
                pair_error = _pair_relative_error(
                    solutions["softmax"]["achieved_ipr"],
                    solutions["sparsemax"]["achieved_ipr"],
                )
                if pair_error > config.pair_ipr_relative_tolerance:
                    pair_status = "pair_match_error"
            pair_rows.append({
                "cue_index": int(cue_index),
                "memory_seed": int(meta.memory_seed),
                "target": int(meta.target),
                "rho": float(meta.rho),
                "mask_index": int(meta.mask_index),
                "level": level,
                "target_ipr": float(target_ipr),
                "softmax_alpha": solutions["softmax"]["alpha"],
                "sparsemax_alpha": solutions["sparsemax"]["alpha"],
                "softmax_achieved_ipr": solutions["softmax"]["achieved_ipr"],
                "sparsemax_achieved_ipr": solutions["sparsemax"]["achieved_ipr"],
                "pair_relative_error": pair_error,
                "pair_status": pair_status,
                "included": pair_status == "matched",
            })
    return pd.DataFrame(solver_rows), pd.DataFrame(pair_rows)


def rank_trajectory_per_cue_alpha(
    cues: torch.Tensor,
    alphas: torch.Tensor,
    Q: torch.Tensor,
    C: torch.Tensor,
    method: str,
    config: PerCuePilotConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """每行 cue 使用自己的 alpha，精确记录 memory-span Jacobian 谱。"""
    batch = cues.shape[0]
    if alphas.shape != (batch,):
        raise ValueError("alphas must contain exactly one value per cue")
    rank = C.shape[0]
    coordinates = cues @ Q
    cumulative = torch.eye(rank, dtype=cues.dtype).expand(batch, rank, rank).clone()
    rows = []
    local_spectra = []
    cumulative_spectra = []
    with torch.no_grad():
        for step in range(config.jacobian_steps + 1):
            scores = (coordinates @ C) * (alphas / config.N).unsqueeze(-1)
            weights = transform_scores(scores, method)
            derivative = score_jacobian(weights, method, config.support_tolerance)
            local = (alphas / config.N)[:, None, None] * torch.einsum(
                "rk,bkl,sl->brs", C, derivative, C
            )
            local_eigenvalues = eigenspectrum(local)
            cumulative_eigenvalues = eigenspectrum(cumulative)
            local_metrics = spectrum_metrics(
                local_eigenvalues, config.numeric_rank_relative_tolerance
            )
            cumulative_metrics = spectrum_metrics(
                cumulative_eigenvalues, config.numeric_rank_relative_tolerance
            )
            diagnostics = attention_diagnostics(
                weights, scores, config.support_tolerance
            )
            following = weights @ C.T
            if step == 0:
                full_following = following @ Q.T
                step_norm = torch.linalg.vector_norm(
                    full_following - cues, dim=-1
                ) / math.sqrt(config.N)
            else:
                step_norm = torch.linalg.vector_norm(
                    following - coordinates, dim=-1
                ) / math.sqrt(config.N)

            for index in range(batch):
                rows.append({
                    "batch_index": index,
                    "method": method,
                    "alpha": float(alphas[index]),
                    "time": step,
                    "local_effective_rank": float(local_metrics["effective_rank"][index]),
                    "local_stable_rank": float(local_metrics["stable_rank"][index]),
                    "local_numeric_rank": int(local_metrics["numeric_rank"][index]),
                    "local_trace": float(local_metrics["trace"][index]),
                    "local_spectral": float(local_metrics["spectral"][index]),
                    "cumulative_effective_rank": float(
                        cumulative_metrics["effective_rank"][index]
                    ),
                    "cumulative_stable_rank": float(cumulative_metrics["stable_rank"][index]),
                    "cumulative_numeric_rank": int(cumulative_metrics["numeric_rank"][index]),
                    "trace_G": float(cumulative_metrics["trace"][index]),
                    "spectral_G": float(cumulative_metrics["spectral"][index]),
                    "attention_entropy": float(diagnostics["attention_entropy"][index]),
                    "ipr": float(diagnostics["ipr"][index]),
                    "support_size": int(diagnostics["support_size"][index]),
                    "score_gap": float(diagnostics["score_gap"][index]),
                    "max_weight": float(diagnostics["max_weight"][index]),
                    "top_memory": int(diagnostics["top_memory"][index]),
                    "step_norm": float(step_norm[index]),
                })
            local_spectra.append(local_eigenvalues.cpu().numpy())
            cumulative_spectra.append(cumulative_eigenvalues.cpu().numpy())
            if step < config.jacobian_steps:
                cumulative = local @ cumulative
                coordinates = following

    frame = add_dimension_survival(frame=pd.DataFrame(rows), config=config.base_config())
    return frame, {
        "local_eigenvalues": np.stack(local_spectra, axis=1),
        "cumulative_eigenvalues": np.stack(cumulative_spectra, axis=1),
    }


def run_seed(
    memory_seed: int,
    config: PerCuePilotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    """完成一个旧 seed 的求根、自检与所有保留对的秩轨迹。"""
    base = config.base_config()
    X = make_memories(base, memory_seed)
    Q, C = memory_span_basis(X)
    targets = select_formal_targets(base, memory_seed)
    cues, metadata = make_cues_for_targets(
        X, base, memory_seed, targets, mask_count=config.mask_count
    )
    checks = run_numerical_self_checks(X, Q, C, cues[0], base)
    checks.insert(0, "memory_seed", memory_seed)
    solver, pairs = solve_cue_pairs(X, cues, metadata, config)

    rank_frames = []
    spectra = {}
    timing_rows = []
    for level in LEVELS:
        level_pairs = pairs[(pairs.level == level) & pairs.included].copy()
        if level_pairs.empty:
            continue
        indices = torch.as_tensor(
            level_pairs.cue_index.to_numpy(dtype=np.int64, copy=True)
        )
        selected_cues = cues[indices]
        selected_meta = metadata.iloc[indices.numpy()].reset_index(drop=True)
        for method in METHODS:
            alphas = torch.as_tensor(
                level_pairs[f"{method}_alpha"].to_numpy(copy=True),
                dtype=torch.float64,
            )
            started = time.perf_counter()
            frame, arrays = rank_trajectory_per_cue_alpha(
                selected_cues, alphas, Q, C, method, config
            )
            elapsed = time.perf_counter() - started
            frame = frame.merge(
                selected_meta.reset_index(names="batch_index"),
                on="batch_index", how="left",
            )
            frame["level"] = level
            frame["target_ipr"] = float(
                config.target_iprs[LEVELS.index(level)]
            )
            rank_frames.append(frame)
            prefix = f"seed{memory_seed}_{level}_{method}"
            spectra[f"{prefix}_local"] = arrays["local_eigenvalues"].astype(np.float32)
            spectra[f"{prefix}_cumulative"] = arrays[
                "cumulative_eigenvalues"
            ].astype(np.float32)
            timing_rows.append({
                "memory_seed": memory_seed,
                "level": level,
                "method": method,
                "cue_count": len(selected_cues),
                "seconds": elapsed,
            })
    rank_results = (
        pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    )
    return solver, pairs, rank_results, spectra, checks, pd.DataFrame(timing_rows)


def summarize_interaction(
    rank_results: pd.DataFrame,
    config: PerCuePilotConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """先在 seed×档位×方法内平均所有 rho/target，再计算 Delta 与 C。"""
    entry = rank_results[rank_results.time == 1].copy()
    seed_level = (
        entry.groupby(["memory_seed", "level", "method"], observed=True)
        .dimension_auc.mean().rename("mean_dimension_auc").reset_index()
    )
    pivot = seed_level.pivot(
        index=["memory_seed", "level"],
        columns="method", values="mean_dimension_auc",
    ).reset_index()
    pivot["delta_sparsemax_minus_softmax"] = pivot.sparsemax - pivot.softmax
    delta_wide = pivot.pivot(
        index="memory_seed", columns="level",
        values="delta_sparsemax_minus_softmax",
    ).reset_index()
    missing = set(LEVELS) - set(delta_wide.columns)
    if missing:
        raise RuntimeError(f"missing interaction levels: {sorted(missing)}")
    delta_wide["C"] = delta_wide.middle - 0.5 * (
        delta_wide.low + delta_wide.high
    )
    return pivot, delta_wide


def bootstrap_interaction(
    values: np.ndarray,
    config: PerCuePilotConfig,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    generator = np.random.default_rng(config.bootstrap_seed)
    indices = generator.integers(
        0, len(values), size=(config.bootstrap_samples, len(values))
    )
    samples = values[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return pd.DataFrame({
        "sample": np.arange(config.bootstrap_samples),
        "mean_C": samples,
    }), (float(lower), float(upper))


def audit_pilot(
    solver: pd.DataFrame,
    pairs: pd.DataFrame,
    rank_results: pd.DataFrame,
    self_checks: pd.DataFrame,
    seed_interaction: pd.DataFrame,
    config: PerCuePilotConfig,
) -> dict:
    expected_pairs = (
        len(config.memory_seeds) * config.targets_per_seed
        * len(config.corruption_rates) * len(config.target_iprs)
    )
    included = pairs[pairs.included]
    attainability = len(included) / expected_pairs
    maximum_target_error = (
        float(solver.loc[solver.status == "matched", "target_relative_error"].max())
        if bool((solver.status == "matched").any()) else None
    )
    maximum_pair_error = (
        float(included.pair_relative_error.max()) if len(included) else None
    )
    expected_rows = len(included) * len(METHODS) * (config.jacobian_steps + 1)
    after_entry = rank_results[rank_results.time >= 1]
    critical = [
        "dimension_survival", "dimension_auc", "trace_G", "spectral_G",
        "cumulative_effective_rank", "local_effective_rank",
    ]
    nonfinite = (
        int((~np.isfinite(after_entry[critical].to_numpy(dtype=float))).sum())
        if len(after_entry) else 0
    )
    checks_passed = bool(
        len(self_checks) == len(config.memory_seeds) * 7
        and self_checks.passed.all()
    )
    mean_c = (
        float(seed_interaction.C.mean())
        if len(seed_interaction) == len(config.memory_seeds) else None
    )
    reachability_gate = attainability >= config.pair_attainability_threshold
    match_gate = bool(
        maximum_target_error is not None
        and maximum_pair_error is not None
        and maximum_target_error <= config.target_ipr_relative_tolerance
        and maximum_pair_error <= config.pair_ipr_relative_tolerance
    )
    numeric_gate = bool(
        checks_passed and len(rank_results) == expected_rows and nonfinite == 0
    )
    interaction_gate = bool(
        mean_c is not None
        and abs(mean_c) >= config.interaction_signal_threshold
    )
    if not reachability_gate:
        outcome = "stop_pair_attainability_below_95_percent"
    elif not match_gate:
        outcome = "stop_ipr_match_error_above_0.5_percent"
    elif not numeric_gate:
        outcome = "stop_numerical_or_completeness_failure"
    elif not interaction_gate:
        outcome = "stop_interaction_signal_below_0.05"
    else:
        outcome = "pass_method_pilot_design_experiment_06"
    return {
        "status": "exploratory_05R_method_pilot",
        "confirmatory_claim_allowed": False,
        "expected_pair_count": int(expected_pairs),
        "included_pair_count": int(len(included)),
        "pair_attainability_fraction": float(attainability),
        "pair_status_counts": {
            str(key): int(value) for key, value in pairs.pair_status.value_counts().items()
        },
        "maximum_target_ipr_relative_error": maximum_target_error,
        "maximum_method_pair_ipr_relative_error": maximum_pair_error,
        "all_seed_self_checks_passed": checks_passed,
        "self_check_count": int(len(self_checks)),
        "expected_rank_rows_for_included_pairs": int(expected_rows),
        "actual_rank_rows": int(len(rank_results)),
        "unexpected_nonfinite_after_entry": nonfinite,
        "mean_C": mean_c,
        "pair_attainability_gate_passed": bool(reachability_gate),
        "ipr_match_gate_passed": bool(match_gate),
        "numerical_gate_passed": bool(numeric_gate),
        "interaction_signal_gate_passed": bool(interaction_gate),
        "interaction_signal_absolute_threshold": config.interaction_signal_threshold,
        "stopping_outcome": outcome,
        "experiment_06_allowed": outcome == "pass_method_pilot_design_experiment_06",
        "05B_allowed": False,
        "atlas_allowed": False,
    }


def plot_pilot(
    solver: pd.DataFrame,
    rank_results: pd.DataFrame,
    seed_level: pd.DataFrame,
    seed_interaction: pd.DataFrame,
    summary: dict,
):
    colors = {"softmax": "#2563EB", "sparsemax": "#E11D48"}
    level_colors = {"low": "#0EA5E9", "middle": "#8B5CF6", "high": "#F59E0B"}
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9.3,
    })
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax_alpha, ax_error, ax_survival, ax_delta, ax_c, ax_mechanism = axes.ravel()

    matched_solver = solver[solver.status == "matched"]
    positions = np.arange(len(LEVELS))
    for offset, method in zip((-0.16, 0.16), METHODS):
        groups = [
            matched_solver[(matched_solver.level == level) & (matched_solver.method == method)].alpha
            for level in LEVELS
        ]
        parts = ax_alpha.violinplot(
            groups, positions=positions + offset, widths=0.27,
            showextrema=False, showmedians=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor(colors[method]); body.set_alpha(0.55)
        parts["cmedians"].set_color(colors[method])
    ax_alpha.set_yscale("log")
    ax_alpha.set_xticks(positions, LEVELS)
    ax_alpha.set(title="A  Per-cue sharpness solutions", ylabel=r"solved $\alpha$ (log)")
    ax_alpha.text(0.02, 0.04, "blue softmax   red sparsemax", transform=ax_alpha.transAxes)

    for method in METHODS:
        rows = matched_solver[matched_solver.method == method]
        ax_error.scatter(
            rows.target_ipr, 100 * rows.target_relative_error,
            s=7, alpha=0.20, color=colors[method], label=method,
        )
    ax_error.axhline(0.5, color="#111827", ls="--", lw=1)
    ax_error.set_yscale("symlog", linthresh=1e-10)
    ax_error.set(
        title="B  First-step IPR matching audit",
        xlabel="target IPR", ylabel="target relative error (%)",
    )
    ax_error.legend(frameon=False)

    aggregate = (
        rank_results.groupby(["level", "method", "time"], observed=True)
        .dimension_survival.mean().reset_index()
    )
    for level in LEVELS:
        for method in METHODS:
            rows = aggregate[(aggregate.level == level) & (aggregate.method == method)]
            ax_survival.plot(
                rows.time, rows.dimension_survival,
                color=level_colors[level],
                ls="-" if method == "sparsemax" else "--",
                lw=2, alpha=0.95,
                label=f"{level}, {method}",
            )
    ax_survival.set(
        title="C  Direction survival (solid sparsemax)",
        xlabel="iteration", ylabel=r"mean $d_t$",
    )
    ax_survival.legend(frameon=False, fontsize=7, ncol=2)

    jitter = np.linspace(-0.08, 0.08, len(seed_interaction))
    for index, level in enumerate(LEVELS):
        values = seed_level[seed_level.level == level].sort_values("memory_seed")
        ax_delta.scatter(
            np.full(len(values), index) + jitter,
            values.delta_sparsemax_minus_softmax,
            color=level_colors[level], s=32, alpha=0.8,
        )
        ax_delta.hlines(
            values.delta_sparsemax_minus_softmax.mean(), index - 0.22, index + 0.22,
            color="#111827", lw=2,
        )
    ax_delta.axhline(0, color="#64748B", lw=1)
    ax_delta.set_xticks(range(3), LEVELS)
    ax_delta.set(
        title="D  Paired dimension-AUC difference",
        ylabel="sparsemax − softmax",
    )

    ordered = seed_interaction.sort_values("memory_seed")
    ax_c.bar(
        ordered.memory_seed.astype(str), ordered.C,
        color=np.where(ordered.C >= 0, "#10B981", "#F97316"), alpha=0.85,
    )
    threshold = summary["interaction_signal_absolute_threshold"]
    ax_c.axhline(threshold, color="#111827", ls="--", lw=1)
    ax_c.axhline(-threshold, color="#111827", ls="--", lw=1)
    ax_c.axhline(summary["mean_C"], color="#7C3AED", lw=2, label="mean C")
    ax_c.set(title="E  Exploratory interaction by old seed", xlabel="memory seed", ylabel="C")
    ax_c.legend(frameon=False)

    # time=0 stores T(alpha s(x0)), the attention that produces the first update.
    entry = rank_results[rank_results.time == 0]
    mechanism = (
        entry.groupby(["level", "method"], observed=True)
        .agg(support=("support_size", "mean"), entropy=("attention_entropy", "mean"))
        .reset_index()
    )
    for method in METHODS:
        rows = mechanism[mechanism.method == method]
        ax_mechanism.plot(
            rows.support, rows.entropy, "o-", color=colors[method],
            label=method,
        )
        for row in rows.itertuples():
            ax_mechanism.annotate(row.level, (row.support, row.entropy), fontsize=7)
    ax_mechanism.set(
        title="F  Descriptive support–entropy state",
        xlabel="mean support producing first update",
        ylabel="mean attention entropy producing first update",
    )
    ax_mechanism.legend(frameon=False)
    fig.suptitle(
        "Experiment 05R — exploratory per-cue IPR matching pilot",
        fontsize=14, fontweight="bold",
    )
    return fig


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def write_artifacts(
    directory: str | Path,
    config: PerCuePilotConfig,
    solver: pd.DataFrame,
    pairs: pd.DataFrame,
    rank_results: pd.DataFrame,
    spectra: dict[str, np.ndarray],
    self_checks: pd.DataFrame,
    timings: pd.DataFrame,
    total_seconds: float,
) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_level, seed_interaction = summarize_interaction(rank_results, config)
    bootstrap, bootstrap_ci = bootstrap_interaction(
        seed_interaction.C.to_numpy(dtype=float), config
    )
    summary = audit_pilot(
        solver, pairs, rank_results, self_checks, seed_interaction, config
    )
    summary["descriptive_bootstrap_95_ci"] = list(bootstrap_ci)
    summary["runtime_seconds"] = total_seconds
    summary["jacobian_runtime_seconds"] = float(timings.seconds.sum())
    summary["execution_environment"] = "local_cpu"
    summary["device"] = "cpu"
    summary["torch_version"] = torch.__version__

    protocol_path = Path(__file__).with_name(
        "experiment_05R_per_cue_ipr_matching_protocol.md"
    )
    summary["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    summary["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    gzip_options = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    solver.to_csv(directory / "solver_results.csv.gz", index=False, compression=gzip_options)
    pairs.to_csv(directory / "pair_status.csv", index=False)
    rank_results.to_csv(
        directory / "jacobian_results.csv.gz", index=False, compression=gzip_options
    )
    np.savez_compressed(directory / "jacobian_spectra.npz", **spectra)
    self_checks.to_csv(directory / "self_checks.csv", index=False)
    timings.to_csv(directory / "jacobian_timing.csv", index=False)
    seed_level.to_csv(directory / "seed_level_auc.csv", index=False)
    seed_interaction.to_csv(directory / "seed_interaction.csv", index=False)
    bootstrap.to_csv(
        directory / "descriptive_bootstrap.csv.gz", index=False,
        compression=gzip_options,
    )
    (directory / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "protocol.json").write_text(
        json.dumps({
            "config": asdict(config),
            "protocol_sha256": summary["protocol_sha256"],
            "old_seeds_exploratory_only": True,
            "confirmatory_claim_allowed": False,
        }, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    if summary["experiment_06_allowed"]:
        decision = (
            "逐 cue 匹配、数值完整性与探索性交互门均通过；允许按预注册的全新 "
            "seeds 100..107 设计实验 06。此结果仍不构成确认性证据。"
        )
    else:
        decision = {
            "stop_pair_attainability_below_95_percent": "可达成对 cue 少于 95%，路线停止。",
            "stop_ipr_match_error_above_0.5_percent": "逐 cue IPR 匹配误差超过 0.5%，路线停止。",
            "stop_numerical_or_completeness_failure": "数值自检或数据完整性失败，路线停止。",
            "stop_interaction_signal_below_0.05": (
                "逐 cue 匹配可行，但旧 seeds 上 |mean(C)| 小于 0.05；没有足够的"
                "探索性信号支持新 seeds，路线停止。"
            ),
        }[summary["stopping_outcome"]]
    conclusion = (
        "# 实验 05R 判定\n\n"
        f"{decision}\n\n"
        f"保留 {summary['included_pair_count']}/{summary['expected_pair_count']} 对 cue "
        f"（{100 * summary['pair_attainability_fraction']:.2f}%）；最大单方法目标误差 "
        f"{100 * summary['maximum_target_ipr_relative_error']:.6g}%，最大方法间误差 "
        f"{100 * summary['maximum_method_pair_ipr_relative_error']:.6g}%。"
        f"探索性 mean(C)={summary['mean_C']:.6f}，描述性 bootstrap 95% CI="
        f"[{bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}]。\n\n"
        "05R 使用已看过的旧 seeds，只能决定是否值得投入实验 06；不能改写原 05A，"
        "也不启动 05B 或二维图册。\n"
    )
    (directory / "conclusion.md").write_text(conclusion, encoding="utf-8")
    figure = plot_pilot(solver, rank_results, seed_level, seed_interaction, summary)
    figure.savefig(directory / "main_figure.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / "main_figure.pdf", bbox_inches="tight")
    plt.close(figure)
    return summary


def run_pilot(
    config: PerCuePilotConfig,
    output_directory: str | Path,
) -> dict:
    config.validate()
    started = time.perf_counter()
    solvers = []
    pairs = []
    rank_frames = []
    spectra = {}
    checks = []
    timings = []
    for seed in config.memory_seeds:
        result = run_seed(seed, config)
        seed_solver, seed_pairs, seed_rank, seed_spectra, seed_checks, seed_timings = result
        solvers.append(seed_solver)
        pairs.append(seed_pairs)
        rank_frames.append(seed_rank)
        spectra.update(seed_spectra)
        checks.append(seed_checks)
        timings.append(seed_timings)
    return write_artifacts(
        output_directory,
        config,
        pd.concat(solvers, ignore_index=True),
        pd.concat(pairs, ignore_index=True),
        pd.concat(rank_frames, ignore_index=True),
        spectra,
        pd.concat(checks, ignore_index=True),
        pd.concat(timings, ignore_index=True),
        time.perf_counter() - started,
    )


def main() -> None:
    output = Path(__file__).with_name("artifacts") / "experiment_05R" / "pilot"
    summary = run_pilot(PerCuePilotConfig(), output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
