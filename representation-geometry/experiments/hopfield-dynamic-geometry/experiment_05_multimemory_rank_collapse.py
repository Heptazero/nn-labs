"""实验 05 开发核心：完整记忆库现代 Hopfield 的 Jacobian 秩轨迹。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

from entmax import sparsemax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class RankCollapseConfig:
    """冻结 protocol 的正式参数，并单列开发预跑预算。"""

    N: int = 128
    K: int = 100
    memory_seeds: tuple[int, ...] = tuple(range(8))
    corruption_rates: tuple[float, ...] = (0.10, 0.25, 0.40)
    # 开发 seed 的初始 11 点网格没有共同覆盖 rho=0.40；这是正式 seed 前
    # 唯一一次、只依据第一步 IPR 作出的扩展。
    alpha_grid: tuple[float, ...] = (
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.60, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50,
        1.60, 1.70, 1.80, 1.90, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
        3.50, 3.75, 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50,
        8.00, 8.50, 9.00, 9.50, 10.00, 10.50, 11.00, 11.50, 12.00,
        13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00,
        21.00, 22.00, 23.00, 24.00, 26.00, 28.00, 30.00, 32.00,
        36.00, 40.00, 48.00,
    )
    endpoint_calibration_alphas: tuple[float, ...] = (
        0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    )
    jacobian_steps: int = 12
    endpoint_steps: int = 100
    convergence_tolerance: float = 1e-8
    convergence_sustain_steps: int = 3
    stability_margin: float = 1e-6
    memory_like_attention: float = 0.95
    support_tolerance: float = 1e-12
    numeric_rank_relative_tolerance: float = 1e-10
    sensitivity_extinction_relative_tolerance: float = 1e-14
    collapsed_at_entry_tolerance: float = 1e-8
    ipr_match_relative_tolerance: float = 0.05
    matched_ipr_levels: int = 3
    development_seed: int = 0
    development_targets: int = 12
    development_masks: int = 2
    target_seed_offset: int = 8_000
    corruption_seed_offset: int = 20_000
    atlas_seed_offset: int = 9_000
    dtype: str = "float64"

    def validate(self) -> None:
        if self.N <= self.K or self.K < 3:
            raise ValueError("development design requires N > K >= 3")
        if not self.memory_seeds or len(set(self.memory_seeds)) != len(self.memory_seeds):
            raise ValueError("memory seeds must be unique and non-empty")
        if not self.alpha_grid or any(alpha <= 0 for alpha in self.alpha_grid):
            raise ValueError("alpha grid must contain positive values")
        if tuple(sorted(set(self.alpha_grid))) != self.alpha_grid:
            raise ValueError("alpha grid must be strictly increasing")
        if not set(self.endpoint_calibration_alphas).issubset(self.alpha_grid):
            raise ValueError("endpoint calibration alphas must be in alpha_grid")
        if any(not 0 < rho < 0.5 for rho in self.corruption_rates):
            raise ValueError("corruption rates must lie in (0, 0.5)")
        if self.jacobian_steps < 1 or self.endpoint_steps <= self.jacobian_steps:
            raise ValueError("iteration budgets are invalid")
        if self.development_targets > self.K or self.development_targets < 3:
            raise ValueError("invalid development target count")
        if self.development_masks < 1 or self.matched_ipr_levels != 3:
            raise ValueError("development masks and three IPR levels are required")
        if not 0 < self.memory_like_attention < 1:
            raise ValueError("memory-like attention threshold must lie in (0,1)")
        if self.dtype != "float64":
            raise ValueError("experiment 05 numerical checks require float64")


def make_memories(config: RankCollapseConfig, seed: int) -> torch.Tensor:
    """返回 X:(N,K)，每列是一条 {-1,+1} 记忆。"""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    bits = torch.randint(
        0, 2, (config.N, config.K), generator=generator, dtype=torch.int8
    )
    return bits.mul(2).sub(1).to(torch.float64)


def memory_span_basis(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """X=QC；Q 是 memory span 的正交基，C=Q^T X。"""
    Q, R = torch.linalg.qr(X, mode="reduced")
    rank = int(torch.linalg.matrix_rank(R).item())
    if rank != X.shape[1]:
        raise RuntimeError(f"memory matrix is rank deficient: {rank} < {X.shape[1]}")
    return Q, R


def select_development_targets(config: RankCollapseConfig, seed: int) -> np.ndarray:
    generator = np.random.default_rng(config.target_seed_offset + seed)
    return np.sort(generator.choice(
        config.K, size=config.development_targets, replace=False
    ))


def make_corrupted_cue(
    X: torch.Tensor,
    target: int,
    rho: float,
    mask_index: int,
    config: RankCollapseConfig,
    memory_seed: int,
) -> torch.Tensor:
    """用固定翻转数生成可复算的 Hamming 噪声查询。"""
    flips = int(round(config.N * rho))
    seed = (
        config.corruption_seed_offset
        + 1_000_003 * memory_seed
        + 10_007 * target
        + 101 * int(round(100 * rho))
        + mask_index
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(config.N, generator=generator)[:flips]
    cue = X[:, target].clone()
    cue[indices] *= -1
    return cue


def make_development_cues(
    X: torch.Tensor,
    config: RankCollapseConfig,
    memory_seed: int,
) -> tuple[torch.Tensor, pd.DataFrame]:
    rows = []
    cues = []
    cue_index = 0
    targets = select_development_targets(config, memory_seed)
    for rho in config.corruption_rates:
        for target in targets:
            for mask_index in range(config.development_masks):
                cues.append(make_corrupted_cue(
                    X, int(target), rho, mask_index, config, memory_seed
                ))
                rows.append({
                    "cue_index": cue_index,
                    "memory_seed": memory_seed,
                    "target": int(target),
                    "rho": rho,
                    "mask_index": mask_index,
                })
                cue_index += 1
    return torch.stack(cues), pd.DataFrame(rows)


def transform_scores(scores: torch.Tensor, method: str) -> torch.Tensor:
    if method == "softmax":
        return torch.softmax(scores, dim=-1)
    if method == "sparsemax":
        return sparsemax(scores, dim=-1)
    raise ValueError(f"unknown retrieval method: {method}")


def hopfield_step(
    states: torch.Tensor,
    X: torch.Tensor,
    alpha: float,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """全部 K 条记忆参与的批量一步更新。"""
    scores = (float(alpha) / X.shape[0]) * (states @ X)
    weights = transform_scores(scores, method)
    following = weights @ X.T
    return following, weights, scores


def attention_diagnostics(
    weights: torch.Tensor,
    scores: torch.Tensor,
    support_tolerance: float,
) -> dict[str, torch.Tensor]:
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    top_two = torch.topk(scores, 2, dim=-1).values
    return {
        "attention_entropy": -(weights * safe.log()).sum(dim=-1),
        "ipr": 1.0 / weights.square().sum(dim=-1),
        "support_size": (weights > support_tolerance).sum(dim=-1),
        "score_gap": top_two[:, 0] - top_two[:, 1],
        "max_weight": weights.max(dim=-1).values,
        "top_memory": weights.argmax(dim=-1),
    }


def score_jacobian(weights: torch.Tensor, method: str, tolerance: float) -> torch.Tensor:
    """T(theta) 对 theta 的解析 Jacobian，shape=(batch,K,K)。"""
    if method == "softmax":
        return torch.diag_embed(weights) - weights.unsqueeze(-1) * weights.unsqueeze(-2)
    if method == "sparsemax":
        support = (weights > tolerance).to(weights.dtype)
        count = support.sum(dim=-1).clamp_min(1.0)
        return (
            torch.diag_embed(support)
            - support.unsqueeze(-1) * support.unsqueeze(-2) / count[:, None, None]
        )
    raise ValueError(f"unknown retrieval method: {method}")


def local_jacobian(
    weights: torch.Tensor,
    basis_coordinates: torch.Tensor,
    alpha: float,
    N: int,
    method: str,
    tolerance: float,
) -> torch.Tensor:
    derivative = score_jacobian(weights, method, tolerance)
    return (float(alpha) / N) * torch.einsum(
        "rk,bkl,sl->brs", basis_coordinates, derivative, basis_coordinates
    )


def full_local_jacobian(
    weights: torch.Tensor,
    X: torch.Tensor,
    alpha: float,
    method: str,
    tolerance: float,
) -> torch.Tensor:
    return local_jacobian(weights, X, alpha, X.shape[0], method, tolerance)


def eigenspectrum(matrix: torch.Tensor) -> torch.Tensor:
    """对 J^T J 取非负升序谱，保留微小负值供自检检测。"""
    gram = matrix.transpose(-1, -2) @ matrix
    return torch.linalg.eigvalsh(gram)


def spectrum_metrics(
    eigenvalues: torch.Tensor,
    numeric_rank_relative_tolerance: float,
) -> dict[str, torch.Tensor]:
    clipped = eigenvalues.clamp_min(0.0)
    trace = clipped.sum(dim=-1)
    maximum = clipped[..., -1]
    probabilities = clipped / trace.clamp_min(torch.finfo(clipped.dtype).tiny).unsqueeze(-1)
    positive = probabilities > 0
    entropy = -torch.where(
        positive, probabilities * probabilities.clamp_min(torch.finfo(clipped.dtype).tiny).log(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    effective = entropy.exp()
    stable = trace / maximum.clamp_min(torch.finfo(clipped.dtype).tiny)
    numeric_rank = (
        clipped > numeric_rank_relative_tolerance * maximum.unsqueeze(-1)
    ).sum(dim=-1)
    return {
        "trace": trace,
        "spectral": maximum,
        "effective_rank": effective,
        "stable_rank": stable,
        "numeric_rank": numeric_rank,
    }


def _reduced_weights(
    coordinates: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    N: int,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = (float(alpha) / N) * (coordinates @ C)
    return transform_scores(scores, method), scores


def rank_trajectory(
    cues: torch.Tensor,
    Q: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    method: str,
    config: RankCollapseConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """批量记录 memory-span 内的局部与累计 Jacobian 全谱。"""
    batch = cues.shape[0]
    rank = C.shape[0]
    coordinates = cues @ Q
    cumulative = torch.eye(rank, dtype=cues.dtype).expand(batch, rank, rank).clone()
    rows = []
    local_spectra = []
    cumulative_spectra = []

    with torch.no_grad():
        for step in range(config.jacobian_steps + 1):
            weights, scores = _reduced_weights(
                coordinates, C, alpha, config.N, method
            )
            local = local_jacobian(
                weights, C, alpha, config.N, method, config.support_tolerance
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
                    "alpha": float(alpha),
                    "time": step,
                    "local_effective_rank": float(
                        local_metrics["effective_rank"][index]
                    ),
                    "local_stable_rank": float(local_metrics["stable_rank"][index]),
                    "local_numeric_rank": int(local_metrics["numeric_rank"][index]),
                    "local_trace": float(local_metrics["trace"][index]),
                    "local_spectral": float(local_metrics["spectral"][index]),
                    "cumulative_effective_rank": float(
                        cumulative_metrics["effective_rank"][index]
                    ),
                    "cumulative_stable_rank": float(
                        cumulative_metrics["stable_rank"][index]
                    ),
                    "cumulative_numeric_rank": int(
                        cumulative_metrics["numeric_rank"][index]
                    ),
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

    frame = pd.DataFrame(rows)
    frame = add_dimension_survival(frame, config)
    return frame, {
        "local_eigenvalues": np.stack(local_spectra, axis=1),
        "cumulative_eigenvalues": np.stack(cumulative_spectra, axis=1),
    }


def _first_sustained_threshold(values: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(values) & (values <= threshold)
    sustained = np.logical_and.accumulate(valid[::-1])[::-1]
    indices = np.flatnonzero(sustained)
    return float(indices[0] + 1) if len(indices) else float("nan")


def add_dimension_survival(
    frame: pd.DataFrame,
    config: RankCollapseConfig,
) -> pd.DataFrame:
    """以 t=1 为入口计算 d_t、dimension_auc、t50 和 t10。"""
    output = frame.copy()
    output["dimension_survival"] = np.nan
    output["sensitivity_extinct"] = False
    output["collapsed_at_entry"] = False
    output["dimension_auc"] = np.nan
    output["t50"] = np.nan
    output["t10"] = np.nan
    for batch_index, indices in output.groupby("batch_index").groups.items():
        rows = output.loc[indices].sort_values("time")
        entry = rows[rows.time == 1].iloc[0]
        denominator = entry.cumulative_effective_rank - 1.0
        collapsed = denominator <= config.collapsed_at_entry_tolerance
        trace_threshold = (
            config.sensitivity_extinction_relative_tolerance * entry.trace_G
        )
        values = []
        extinct_flags = []
        for row in rows.itertuples():
            extinct = row.time >= 1 and row.trace_G <= trace_threshold
            extinct_flags.append(extinct)
            if row.time < 1:
                values.append(float("nan"))
            elif collapsed or extinct:
                values.append(0.0)
            else:
                values.append((row.cumulative_effective_rank - 1.0) / denominator)
        output.loc[rows.index, "dimension_survival"] = values
        output.loc[rows.index, "sensitivity_extinct"] = extinct_flags
        output.loc[rows.index, "collapsed_at_entry"] = collapsed
        after_entry = np.asarray(values[1:], dtype=float)
        auc = float(np.mean(after_entry))
        t50 = _first_sustained_threshold(after_entry, 0.5)
        t10 = _first_sustained_threshold(after_entry, 0.1)
        output.loc[rows.index, "dimension_auc"] = auc
        output.loc[rows.index, "t50"] = t50
        output.loc[rows.index, "t10"] = t10
    return output


def classify_endpoints(
    cues: torch.Tensor,
    cue_metadata: pd.DataFrame,
    Q: torch.Tensor,
    C: torch.Tensor,
    alpha: float,
    method: str,
    config: RankCollapseConfig,
) -> pd.DataFrame:
    """用同一收敛、稳定性和 attention 阈值分类所有开发 cue。"""
    coordinates = cues @ Q
    streak = torch.zeros(len(cues), dtype=torch.int64)
    convergence_step = torch.full((len(cues),), -1, dtype=torch.int64)
    previous_full = cues
    with torch.no_grad():
        for step in range(1, config.endpoint_steps + 1):
            weights, scores = _reduced_weights(
                coordinates, C, alpha, config.N, method
            )
            following = weights @ C.T
            following_full = following @ Q.T
            delta = torch.linalg.vector_norm(
                following_full - previous_full, dim=-1
            ) / math.sqrt(config.N)
            streak = torch.where(
                delta < config.convergence_tolerance,
                streak + 1,
                torch.zeros_like(streak),
            )
            newly = (convergence_step < 0) & (
                streak >= config.convergence_sustain_steps
            )
            convergence_step[newly] = step
            coordinates = following
            previous_full = following_full

        weights, scores = _reduced_weights(
            coordinates, C, alpha, config.N, method
        )
        diagnostics = attention_diagnostics(
            weights, scores, config.support_tolerance
        )
        local = local_jacobian(
            weights, C, alpha, config.N, method, config.support_tolerance
        )
        stability_radius = torch.linalg.eigvalsh(local).abs().amax(dim=-1)
        stable = stability_radius < 1.0 - config.stability_margin
        converged = convergence_step >= 0

    rows = []
    for index, metadata in cue_metadata.iterrows():
        top = int(diagnostics["top_memory"][index])
        pmax = float(diagnostics["max_weight"][index])
        if not bool(converged[index]) or not bool(stable[index]):
            category = "unconverged_or_unstable"
        elif pmax >= config.memory_like_attention:
            category = (
                "correct_memory_like" if top == int(metadata.target)
                else "other_stored_memory_like"
            )
        else:
            category = "stable_mixture"
        rows.append({
            **metadata.to_dict(),
            "method": method,
            "alpha": float(alpha),
            "category": category,
            "converged": bool(converged[index]),
            "convergence_step": int(convergence_step[index]),
            "stable": bool(stable[index]),
            "stability_radius": float(stability_radius[index]),
            "final_top_memory": top,
            "final_max_weight": pmax,
            "final_ipr": float(diagnostics["ipr"][index]),
            "final_support_size": int(diagnostics["support_size"][index]),
        })
    return pd.DataFrame(rows)


def calibration_scan(
    X: torch.Tensor,
    cues: torch.Tensor,
    cue_metadata: pd.DataFrame,
    Q: torch.Tensor,
    C: torch.Tensor,
    config: RankCollapseConfig,
) -> pd.DataFrame:
    """廉价扫描第一步 IPR，并用统一分类器检查终态类别覆盖。"""
    records = []
    for method in ("softmax", "sparsemax"):
        for alpha in config.alpha_grid:
            _, weights, scores = hopfield_step(cues, X, alpha, method)
            diagnostics = attention_diagnostics(
                weights, scores, config.support_tolerance
            )
            evaluate_endpoints = alpha in config.endpoint_calibration_alphas
            endpoints = (
                classify_endpoints(
                    cues, cue_metadata, Q, C, alpha, method, config
                ).set_index("cue_index")
                if evaluate_endpoints else None
            )
            for index, metadata in cue_metadata.iterrows():
                endpoint = (
                    endpoints.loc[int(metadata.cue_index)]
                    if endpoints is not None else None
                )
                records.append({
                    **metadata.to_dict(),
                    "method": method,
                    "alpha": float(alpha),
                    "first_ipr": float(diagnostics["ipr"][index]),
                    "first_entropy": float(diagnostics["attention_entropy"][index]),
                    "first_support_size": int(diagnostics["support_size"][index]),
                    "first_score_gap": float(diagnostics["score_gap"][index]),
                    "first_max_weight": float(diagnostics["max_weight"][index]),
                    "category": (
                        endpoint.category if endpoint is not None else "not_evaluated"
                    ),
                    "converged": (
                        endpoint.converged if endpoint is not None else pd.NA
                    ),
                    "convergence_step": (
                        endpoint.convergence_step if endpoint is not None else pd.NA
                    ),
                    "stable": endpoint.stable if endpoint is not None else pd.NA,
                    "stability_radius": (
                        endpoint.stability_radius if endpoint is not None else np.nan
                    ),
                    "final_ipr": endpoint.final_ipr if endpoint is not None else np.nan,
                    "final_support_size": (
                        endpoint.final_support_size if endpoint is not None else pd.NA
                    ),
                })
    return pd.DataFrame(records)


def summarize_ipr(calibration: pd.DataFrame) -> pd.DataFrame:
    def correct_rate(values):
        evaluated = values[values != "not_evaluated"]
        if len(evaluated) == 0:
            return float("nan")
        return float((evaluated == "correct_memory_like").mean())

    return (
        calibration.groupby(["rho", "method", "alpha"])
        .agg(
            median_first_ipr=("first_ipr", "median"),
            p10_first_ipr=("first_ipr", lambda x: x.quantile(0.1)),
            p90_first_ipr=("first_ipr", lambda x: x.quantile(0.9)),
            correct_rate=("category", correct_rate),
        )
        .reset_index()
    )


def choose_matched_ipr_levels(
    ipr_summary: pd.DataFrame,
    config: RankCollapseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个 rho 在离散 alpha 网格上冻结三个可达的共同 IPR 水平。"""
    level_rows = []
    range_rows = []
    for rho in config.corruption_rates:
        subset = ipr_summary[np.isclose(ipr_summary.rho, rho)]
        curves = {
            method: subset[subset.method == method].sort_values("alpha")
            for method in ("softmax", "sparsemax")
        }
        lower = max(curve.median_first_ipr.min() for curve in curves.values())
        upper = min(curve.median_first_ipr.max() for curve in curves.values())
        signatures: dict[tuple[float, float], list[dict]] = {}
        if upper > lower:
            candidates = np.geomspace(lower, upper, 4_000)
            for target in candidates:
                match = {}
                valid = True
                for method, curve in curves.items():
                    index = (curve.median_first_ipr - target).abs().idxmin()
                    row = curve.loc[index]
                    error = abs(row.median_first_ipr - target) / target
                    if error > config.ipr_match_relative_tolerance:
                        valid = False
                        break
                    match[method] = row
                if valid:
                    signature = (
                        float(match["softmax"].alpha),
                        float(match["sparsemax"].alpha),
                    )
                    signatures.setdefault(signature, []).append({
                        "target": float(target), "match": match
                    })
        representatives = []
        for signature, candidates in signatures.items():
            representatives.append(candidates[len(candidates) // 2])
        representatives.sort(key=lambda item: item["target"])
        enough = len(representatives) >= config.matched_ipr_levels
        range_rows.append({
            "rho": rho,
            "common_ipr_min": float(lower),
            "common_ipr_max": float(upper),
            "distinct_matched_pairs": len(representatives),
            "three_levels_available": enough,
        })
        if not enough:
            continue
        requested_targets = np.geomspace(
            lower, upper, config.matched_ipr_levels
        )
        available = list(representatives)
        selected_items = []
        for requested in requested_targets:
            selected = min(
                range(len(available)),
                key=lambda index: abs(
                    math.log(available[index]["target"]) - math.log(requested)
                ),
            )
            selected_items.append((float(requested), available.pop(selected)))
        selected_items.sort(key=lambda pair: pair[0])
        for label, (requested, item) in zip(
            ("low", "middle", "high"), selected_items
        ):
            for method in ("softmax", "sparsemax"):
                row = item["match"][method]
                level_rows.append({
                    "rho": rho,
                    "level": label,
                    "requested_log_ipr": requested,
                    "target_ipr": item["target"],
                    "method": method,
                    "alpha": float(row.alpha),
                    "achieved_median_ipr": float(row.median_first_ipr),
                    "relative_match_error": float(
                        abs(row.median_first_ipr - item["target"]) / item["target"]
                    ),
                })
    return pd.DataFrame(level_rows), pd.DataFrame(range_rows)


def iterate_single(
    state: torch.Tensor,
    X: torch.Tensor,
    alpha: float,
    method: str,
    steps: int,
) -> torch.Tensor:
    for _ in range(steps):
        scores = (float(alpha) / X.shape[0]) * (X.T @ state)
        state = X @ transform_scores(scores.unsqueeze(0), method).squeeze(0)
    return state


def make_whitened_atlas_basis(
    X: torch.Tensor,
    anchors: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first, second, third = anchors
    B = torch.stack([X[:, second] - X[:, first], X[:, third] - X[:, first]], dim=1)
    h = B.T @ B
    values, vectors = torch.linalg.eigh(h)
    if float(values[0]) <= 100 * torch.finfo(X.dtype).eps:
        raise RuntimeError("atlas anchor differences are linearly dependent")
    inverse_sqrt = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    return B, h, B @ inverse_sqrt


def preregister_near_atlas(X: torch.Tensor, anchor: int = 0) -> tuple[int, int, int]:
    normalized = X / torch.linalg.vector_norm(X, dim=0, keepdim=True)
    cosine = normalized.T @ normalized
    values = cosine[anchor].clone()
    values[anchor] = -torch.inf
    neighbors = torch.topk(values, 2).indices.tolist()
    return anchor, int(neighbors[0]), int(neighbors[1])


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected).clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    return float(torch.linalg.vector_norm(actual - expected) / denominator)


def _autodiff_local(
    state: torch.Tensor,
    X: torch.Tensor,
    alpha: float,
    method: str,
) -> torch.Tensor:
    def update(query):
        scores = (float(alpha) / X.shape[0]) * (X.T @ query)
        return X @ transform_scores(scores.unsqueeze(0), method).squeeze(0)

    return torch.autograd.functional.jacobian(update, state)


def run_numerical_self_checks(
    X: torch.Tensor,
    Q: torch.Tensor,
    C: torch.Tensor,
    cue: torch.Tensor,
    config: RankCollapseConfig,
) -> pd.DataFrame:
    """运行 protocol 冻结的七项数值自检。"""
    alpha = 1.5
    anchors = preregister_near_atlas(X)
    _, _, B_white = make_whitened_atlas_basis(X, anchors)
    identity_error = _relative_error(
        B_white.T @ B_white, torch.eye(2, dtype=X.dtype)
    )

    def full_two_step(query):
        return iterate_single(query, X, alpha, "softmax", 2)

    full_jacobian = torch.autograd.functional.jacobian(full_two_step, cue)

    def atlas_two_step(z):
        return full_two_step(cue + B_white @ z)

    atlas_jacobian = torch.autograd.functional.jacobian(
        atlas_two_step, torch.zeros(2, dtype=X.dtype)
    )
    projected_metric = B_white.T @ full_jacobian.T @ full_jacobian @ B_white
    atlas_metric = atlas_jacobian.T @ atlas_jacobian
    atlas_projection_error = _relative_error(atlas_metric, projected_metric)
    ritz_violation = max(
        0.0,
        float(torch.linalg.eigvalsh(atlas_metric)[-1]
              - torch.linalg.eigvalsh(full_jacobian.T @ full_jacobian)[-1]),
    )

    complete_Q, _ = torch.linalg.qr(X, mode="complete")
    complement = complete_Q[:, X.shape[1]:]
    first_scores = (alpha / config.N) * (X.T @ cue)
    first_weights = transform_scores(first_scores.unsqueeze(0), "softmax")
    first_local = full_local_jacobian(
        first_weights, X, alpha, "softmax", config.support_tolerance
    )[0]
    complement_ratio = float(
        torch.linalg.matrix_norm(first_local @ complement)
        / torch.linalg.matrix_norm(first_local).clamp_min(torch.finfo(X.dtype).tiny)
    )
    first_rank = int(torch.linalg.matrix_rank(
        first_local, rtol=config.numeric_rank_relative_tolerance
    ).item())

    transform_errors = []
    for method in ("softmax", "sparsemax"):
        scores = (alpha / config.N) * (X.T @ cue)
        weights = transform_scores(scores.unsqueeze(0), method)
        analytic = full_local_jacobian(
            weights, X, alpha, method, config.support_tolerance
        )[0]
        autodiff = _autodiff_local(cue, X, alpha, method)
        transform_errors.append(_relative_error(analytic, autodiff))
    analytic_error = max(transform_errors)

    local_eigenvalues = eigenspectrum(first_local.unsqueeze(0))[0]
    psd_violation = max(0.0, -float(local_eigenvalues.min()))
    finite_failure = int(not bool(torch.isfinite(local_eigenvalues).all()))

    state = cue.clone()
    cumulative = torch.eye(config.N, dtype=X.dtype)
    for _ in range(3):
        scores = (alpha / config.N) * (X.T @ state)
        weights = transform_scores(scores.unsqueeze(0), "softmax")
        local = full_local_jacobian(
            weights, X, alpha, "softmax", config.support_tolerance
        )[0]
        cumulative = local @ cumulative
        state = X @ weights[0]

    def direct_three_step(query):
        return iterate_single(query, X, alpha, "softmax", 3)

    direct = torch.autograd.functional.jacobian(direct_three_step, cue)
    recurrence_error = _relative_error(cumulative, direct)

    reduced_first = local_jacobian(
        first_weights, C, alpha, config.N, "softmax", config.support_tolerance
    )[0]
    reduced_full_error = _relative_error(reduced_first, Q.T @ first_local @ Q)

    rows = [
        ("atlas_t0_whitened_identity", identity_error, 1e-10, identity_error <= 1e-10),
        ("atlas_AD_equals_full_projection", atlas_projection_error, 1e-8,
         atlas_projection_error <= 1e-8),
        ("ritz_upper_bound", ritz_violation, 1e-10, ritz_violation <= 1e-10),
        ("orthogonal_complement_annihilated", complement_ratio, 1e-10,
         complement_ratio <= 1e-10 and first_rank <= config.K - 1),
        ("analytic_transform_jacobians", analytic_error, 1e-8,
         analytic_error <= 1e-8),
        ("gram_spectrum_finite_and_psd", psd_violation + finite_failure, 1e-10,
         finite_failure == 0 and psd_violation <= 1e-10),
        ("cumulative_recurrence_and_reduced_basis",
         max(recurrence_error, reduced_full_error), 1e-8,
         recurrence_error <= 1e-8 and reduced_full_error <= 1e-8),
    ]
    return pd.DataFrame(rows, columns=["check", "error", "tolerance", "passed"])


def run_matched_rank_preflight(
    cues: torch.Tensor,
    cue_metadata: pd.DataFrame,
    Q: torch.Tensor,
    C: torch.Tensor,
    matched_levels: pd.DataFrame,
    config: RankCollapseConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    all_frames = []
    spectra = {}
    timing_rows = []
    for condition in matched_levels.itertuples():
        mask = np.isclose(cue_metadata.rho, condition.rho)
        metadata = cue_metadata[mask].reset_index(drop=True).copy()
        selected_cues = cues[torch.as_tensor(mask)]
        started = time.perf_counter()
        frame, arrays = rank_trajectory(
            selected_cues, Q, C, condition.alpha, condition.method, config
        )
        elapsed = time.perf_counter() - started
        frame = frame.merge(
            metadata.reset_index(names="batch_index"), on="batch_index", how="left"
        )
        frame["level"] = condition.level
        frame["target_ipr"] = condition.target_ipr
        frame["achieved_median_ipr"] = condition.achieved_median_ipr
        frame["relative_match_error"] = condition.relative_match_error
        all_frames.append(frame)
        key = f"rho{condition.rho:.2f}_{condition.level}_{condition.method}"
        spectra[f"{key}_local"] = arrays["local_eigenvalues"]
        spectra[f"{key}_cumulative"] = arrays["cumulative_eigenvalues"]
        timing_rows.append({
            "rho": condition.rho,
            "level": condition.level,
            "method": condition.method,
            "alpha": condition.alpha,
            "cue_count": len(selected_cues),
            "seconds": elapsed,
        })
    if not all_frames:
        return pd.DataFrame(), spectra, pd.DataFrame(timing_rows)
    return pd.concat(all_frames, ignore_index=True), spectra, pd.DataFrame(timing_rows)


def plot_development_preflight(
    ipr_summary: pd.DataFrame,
    matched_levels: pd.DataFrame,
    rank_results: pd.DataFrame,
    calibration: pd.DataFrame,
):
    colors = {"softmax": "#2563EB", "sparsemax": "#E11D48"}
    plt.rcParams.update({
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_ipr, ax_survival, ax_local, ax_category = axes.ravel()

    for rho, marker in zip(sorted(ipr_summary.rho.unique()), ("o", "s", "^")):
        for method in ("softmax", "sparsemax"):
            rows = ipr_summary[
                np.isclose(ipr_summary.rho, rho) & (ipr_summary.method == method)
            ].sort_values("alpha")
            ax_ipr.plot(
                rows.alpha, rows.median_first_ipr,
                color=colors[method], marker=marker, ms=4,
                label=f"{method}, rho={rho:.2f}",
            )
    ax_ipr.set_xscale("log")
    ax_ipr.set_yscale("log")
    ax_ipr.set(
        title="A  First-step competition calibration",
        xlabel=r"dimensionless sharpness $\alpha$",
        ylabel="median attention IPR",
    )
    ax_ipr.legend(frameon=False, fontsize=7, ncol=2)

    if not rank_results.empty:
        aggregate = (
            rank_results.groupby(["level", "method", "time"])
            .dimension_survival.median().reset_index()
        )
        styles = {"low": ":", "middle": "-", "high": "--"}
        for level in ("low", "middle", "high"):
            for method in ("softmax", "sparsemax"):
                rows = aggregate[
                    (aggregate.level == level) & (aggregate.method == method)
                ]
                ax_survival.plot(
                    rows.time, rows.dimension_survival,
                    color=colors[method], ls=styles[level], lw=2,
                    label=f"{method}, {level} IPR",
                )
        ax_survival.axhline(0.5, color="#64748B", lw=1, ls=":")
        ax_survival.set(
            title="B  Development-only cumulative dimension survival",
            xlabel="iteration",
            ylabel=r"$d_t$ (relative to t=1)",
        )
        ax_survival.legend(frameon=False, fontsize=7, ncol=2)

        middle = rank_results[rank_results.level == "middle"]
        for method in ("softmax", "sparsemax"):
            by_time = middle[middle.method == method].groupby("time").agg(
                local=("local_effective_rank", "median"),
                cumulative=("cumulative_effective_rank", "median"),
            )
            ax_local.plot(
                by_time.index, by_time.local,
                color=colors[method], ls="--", lw=1.7,
                label=f"{method} local A",
            )
            ax_local.plot(
                by_time.index, by_time.cumulative,
                color=colors[method], lw=2.2,
                label=f"{method} cumulative J",
            )
        ax_local.set_yscale("log")
        ax_local.set(
            title="C  Local versus cumulative effective rank",
            xlabel="iteration",
            ylabel="effective rank (log scale)",
        )
        ax_local.legend(frameon=False, fontsize=7)
    else:
        ax_survival.text(0.5, 0.5, "No three matched IPR levels", ha="center")
        ax_local.text(0.5, 0.5, "Jacobian preflight not run", ha="center")

    evaluated_calibration = calibration[
        calibration.category != "not_evaluated"
    ]
    counts = (
        evaluated_calibration.groupby(["method", "category"]).size()
        .rename("count").reset_index()
    )
    categories = sorted(counts.category.unique())
    x = np.arange(len(categories))
    width = 0.36
    for offset, method in zip((-width / 2, width / 2), ("softmax", "sparsemax")):
        values = [
            int(counts[(counts.method == method) & (counts.category == category)]["count"].sum())
            for category in categories
        ]
        ax_category.bar(
            x + offset, values, width=width,
            color=colors[method], label=method, alpha=0.85,
        )
    ax_category.set_xticks(x, [category.replace("_", "\n") for category in categories])
    ax_category.set(
        title="D  Endpoint category coverage across alpha scan",
        ylabel="development observations",
    )
    ax_category.legend(frameon=False)
    fig.suptitle(
        "Experiment 05 development preflight — not a confirmatory result",
        fontsize=14, fontweight="bold",
    )
    return fig


def _jsonable_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def audit_rank_preflight(
    rank_results: pd.DataFrame,
    config: RankCollapseConfig,
) -> tuple[pd.DataFrame, dict]:
    """区分设计内的删失/饱和与真正的数据失败。"""
    if rank_results.empty:
        return pd.DataFrame(), {
            "rank_results_present": False,
            "expected_rank_rows": 0,
            "actual_rank_rows": 0,
            "unexpected_nonfinite_count": 0,
            "unexpected_dimension_survival_nan_count": 0,
            "t50_censored_trajectories": 0,
            "t10_censored_trajectories": 0,
            "collapsed_at_entry_trajectories": 0,
            "sensitivity_extinct_at_final_trajectories": 0,
        }

    trajectory_keys = ["rho", "level", "method", "target", "mask_index"]
    entry = rank_results[rank_results.time == 1].copy()
    final = rank_results[rank_results.time == config.jacobian_steps].copy()
    audit = (
        entry.groupby(["rho", "level", "method"])
        .agg(
            trajectories=("batch_index", "size"),
            collapsed_at_entry=("collapsed_at_entry", "sum"),
            t50_censored=("t50", lambda values: int(values.isna().sum())),
            t10_censored=("t10", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )
    final_extinct = (
        final.groupby(["rho", "level", "method"])
        .sensitivity_extinct.sum().rename("extinct_at_final").reset_index()
    )
    audit = audit.merge(
        final_extinct, on=["rho", "level", "method"], how="left"
    )
    for count_column in (
        "collapsed_at_entry", "t50_censored", "t10_censored", "extinct_at_final"
    ):
        audit[f"{count_column}_fraction"] = (
            audit[count_column] / audit.trajectories
        )

    critical_columns = [
        "local_effective_rank", "local_stable_rank", "local_numeric_rank",
        "local_trace", "local_spectral", "cumulative_effective_rank",
        "cumulative_stable_rank", "cumulative_numeric_rank", "trace_G",
        "spectral_G", "attention_entropy", "ipr", "support_size",
        "score_gap", "max_weight", "step_norm", "dimension_auc",
    ]
    unexpected_nonfinite = int(
        (~np.isfinite(rank_results[critical_columns].to_numpy(dtype=float))).sum()
    )
    expected_rows = (
        config.development_targets
        * config.development_masks
        * len(config.corruption_rates)
        * config.matched_ipr_levels
        * 2
        * (config.jacobian_steps + 1)
    )
    trajectories = entry.drop_duplicates(trajectory_keys)
    summary = {
        "rank_results_present": True,
        "expected_rank_rows": int(expected_rows),
        "actual_rank_rows": int(len(rank_results)),
        "unexpected_nonfinite_count": unexpected_nonfinite,
        "expected_t0_dimension_survival_nan_count": int(
            rank_results[rank_results.time == 0].dimension_survival.isna().sum()
        ),
        "unexpected_dimension_survival_nan_count": int(
            rank_results[rank_results.time > 0].dimension_survival.isna().sum()
        ),
        "t50_censored_trajectories": int(trajectories.t50.isna().sum()),
        "t10_censored_trajectories": int(trajectories.t10.isna().sum()),
        "collapsed_at_entry_trajectories": int(
            trajectories.collapsed_at_entry.sum()
        ),
        "sensitivity_extinct_at_final_trajectories": int(
            final.sensitivity_extinct.sum()
        ),
    }
    return audit, summary


def write_development_artifacts(
    directory: str | Path,
    config: RankCollapseConfig,
    calibration: pd.DataFrame,
    ipr_summary: pd.DataFrame,
    matched_levels: pd.DataFrame,
    ipr_ranges: pd.DataFrame,
    self_checks: pd.DataFrame,
    rank_results: pd.DataFrame,
    spectra: dict[str, np.ndarray],
    timings: pd.DataFrame,
    total_seconds: float,
) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(__file__).with_name(
        "experiment_05_multimemory_rank_collapse_protocol.md"
    )
    import hashlib

    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    calibration.to_csv(directory / "calibration.csv.gz", index=False, compression={
        "method": "gzip", "compresslevel": 6, "mtime": 0,
    })
    ipr_summary.to_csv(directory / "ipr_summary.csv", index=False)
    matched_levels.to_csv(directory / "matched_ipr_levels.csv", index=False)
    ipr_ranges.to_csv(directory / "ipr_common_ranges.csv", index=False)
    self_checks.to_csv(directory / "self_checks.csv", index=False)
    timings.to_csv(directory / "jacobian_timing.csv", index=False)
    if not rank_results.empty:
        rank_results.to_csv(
            directory / "jacobian_results.csv.gz", index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        np.savez_compressed(directory / "jacobian_spectra.npz", **spectra)

    rank_audit, rank_audit_summary = audit_rank_preflight(rank_results, config)
    rank_audit.to_csv(directory / "rank_data_audit.csv", index=False)

    evaluated_calibration = calibration[
        calibration.category != "not_evaluated"
    ]
    category_counts = (
        evaluated_calibration.groupby(["method", "category"]).size()
        .rename("count").reset_index()
    )
    pmax_quantiles = (
        evaluated_calibration.groupby(["method", "category"])["final_ipr"]
        .quantile([0.1, 0.5, 0.9]).unstack().reset_index()
    )
    pmax_quantiles.columns = ["method", "category", "q10", "median", "q90"]
    pmax_quantiles.to_csv(directory / "endpoint_ipr_quantiles.csv", index=False)

    development_rank_trajectories = (
        config.development_targets
        * config.development_masks
        * len(config.corruption_rates)
        * config.matched_ipr_levels
        * 2
    )
    formal_rank_trajectories = (
        32 * len(config.corruption_rates) * config.matched_ipr_levels * 2
        * len(config.memory_seeds)
    )
    jacobian_seconds = float(timings.seconds.sum()) if len(timings) else None
    estimated_formal_seconds = (
        jacobian_seconds * formal_rank_trajectories / development_rank_trajectories
        if development_rank_trajectories and jacobian_seconds is not None
        else None
    )
    checks_passed = bool(self_checks.passed.all())
    levels_available = bool(
        len(ipr_ranges) == len(config.corruption_rates)
        and ipr_ranges.three_levels_available.all()
    )
    maximum_match_error = (
        float(matched_levels.relative_match_error.max())
        if len(matched_levels) else None
    )
    formal_ready = bool(
        checks_passed
        and levels_available
        and maximum_match_error is not None
        and maximum_match_error <= config.ipr_match_relative_tolerance
        and rank_audit_summary["rank_results_present"]
        and rank_audit_summary["actual_rank_rows"]
        == rank_audit_summary["expected_rank_rows"]
        and rank_audit_summary["unexpected_nonfinite_count"] == 0
        and rank_audit_summary["unexpected_dimension_survival_nan_count"] == 0
    )
    summary = {
        "status": "development_preflight_only",
        "confirmatory_claim_allowed": False,
        "formal_05A_ready": formal_ready,
        "protocol_sha256": protocol_sha256,
        "all_seven_self_checks_passed": checks_passed,
        "self_check_count": int(len(self_checks)),
        "all_rhos_have_three_matched_ipr_levels": levels_available,
        "maximum_ipr_match_relative_error": maximum_match_error,
        "ipr_common_ranges": _jsonable_records(ipr_ranges),
        "matched_ipr_levels": _jsonable_records(matched_levels),
        "rank_data_audit": rank_audit_summary,
        "endpoint_category_counts": _jsonable_records(category_counts),
        "provisional_endpoint_thresholds": {
            "convergence_tolerance_per_sqrt_N": config.convergence_tolerance,
            "convergence_sustain_steps": config.convergence_sustain_steps,
            "stability_radius_upper": 1.0 - config.stability_margin,
            "memory_like_attention_minimum": config.memory_like_attention,
        },
        "development_runtime_seconds": total_seconds,
        "jacobian_runtime_seconds": jacobian_seconds,
        "estimated_formal_jacobian_seconds_same_device": estimated_formal_seconds,
        "device": "cpu",
        "torch_version": torch.__version__,
    }
    (directory / "development_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "protocol.json").write_text(
        json.dumps({
            "config": asdict(config),
            "protocol_sha256": protocol_sha256,
            "matched_ipr_levels": _jsonable_records(matched_levels),
            "development_only": True,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    figure = plot_development_preflight(
        ipr_summary, matched_levels, rank_results, calibration
    )
    figure.savefig(directory / "development_figure.png", dpi=220, bbox_inches="tight")
    figure.savefig(directory / "development_figure.pdf", bbox_inches="tight")
    return summary


def run_development_preflight(
    config: RankCollapseConfig,
    output_directory: str | Path,
) -> dict:
    """只执行冻结的开发预算，不形成 05A 确认性结论。"""
    config.validate()
    started = time.perf_counter()
    X = make_memories(config, config.development_seed)
    Q, C = memory_span_basis(X)
    cues, cue_metadata = make_development_cues(
        X, config, config.development_seed
    )
    self_checks = run_numerical_self_checks(
        X, Q, C, cues[0], config
    )
    calibration = calibration_scan(
        X, cues, cue_metadata, Q, C, config
    )
    ipr_summary = summarize_ipr(calibration)
    matched_levels, ipr_ranges = choose_matched_ipr_levels(
        ipr_summary, config
    )
    can_run_rank = bool(
        self_checks.passed.all()
        and len(ipr_ranges) == len(config.corruption_rates)
        and ipr_ranges.three_levels_available.all()
        and len(matched_levels)
        == len(config.corruption_rates) * config.matched_ipr_levels * 2
    )
    if can_run_rank:
        rank_results, spectra, timings = run_matched_rank_preflight(
            cues, cue_metadata, Q, C, matched_levels, config
        )
    else:
        rank_results, spectra, timings = pd.DataFrame(), {}, pd.DataFrame()
    total_seconds = time.perf_counter() - started
    return write_development_artifacts(
        output_directory,
        config,
        calibration,
        ipr_summary,
        matched_levels,
        ipr_ranges,
        self_checks,
        rank_results,
        spectra,
        timings,
        total_seconds,
    )


def main() -> None:
    config = RankCollapseConfig()
    output = Path(__file__).resolve().parent / "artifacts" / "experiment_05" / "development"
    summary = run_development_preflight(config, output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"development artifacts: {output}")


if __name__ == "__main__":
    main()
