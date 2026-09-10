"""a/b/c/d/e：固定现代 Hopfield 的查询探测；目标标签只进入数据生成和评估。"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import torch

from .models.modern import ContinuousModernHopfield
from .provenance import source_info
from .state_spaces import project_to_tangent, spherical_geodesic
from .storage import BatchStore, atomic_json, fingerprint


@dataclass(frozen=True)
class ProbeConfig:
    dimensions: tuple = (32, 128)
    memory_count: int = 64
    memory_kinds: tuple = ("independent", "close_pairs")
    seeds: tuple = tuple(range(8))
    targets: int = 16
    cue_kinds: tuple = ("clean", "gaussian_1.0", "gaussian_1.6", "block")
    steps: int = 12
    angles: tuple = (0.10, 0.25)
    directions: int = 4
    methods: tuple = ("guided", "shuffled", "random")
    residual_tolerance: float = 1e-8


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def a1_make_memories(N, P, kind, seed):
    """[输入] 维度、条数、结构、种子；[输出] 独立或成对相关的 (P,N) 记忆。"""
    generator = _generator(seed)
    patterns = (2 * torch.randint(2, (P, N), generator=generator) - 1).double()
    if kind == "close_pairs":
        if P % 2 or N < 8:
            raise ValueError("成对记忆需要偶数 P 且 N >= 8")
        for index in range(0, P, 2):
            patterns[index + 1] = patterns[index]
            flipped = torch.randperm(N, generator=generator)[:N // 8]
            patterns[index + 1, flipped] *= -1
    elif kind != "independent":
        raise ValueError(kind)
    # [约束] ID 必须对应不同内容，否则所谓检索错误可能只是重复标签。
    if len(torch.unique(patterns, dim=0)) != P:
        raise ValueError("本次记忆库含重复模式；停止并记录，不能静默重抽")
    return patterns


def b1_store_memories(patterns):
    """[存储] 直接保存模式，固定 beta；不训练参数。"""
    return ContinuousModernHopfield(beta=1 / math.sqrt(patterns.shape[1])).fit(patterns)


def c1_make_queries(patterns, count, kinds, seed):
    """[输出] cues:(B,N)、targets:(B,)、条件名；标签只用于合成与事后评分。"""
    generator = _generator(seed)
    ids = torch.randperm(len(patterns), generator=generator)[:count]
    batches, labels, names = [], [], []
    N = patterns.shape[1]
    for kind in kinds:
        cues = patterns[ids].clone()
        if kind.startswith("gaussian_"):
            sigma = float(kind.split("_")[1])
            cues += sigma * torch.randn(cues.shape, generator=generator, dtype=cues.dtype)
        elif kind == "block":
            for row in cues:
                start = int(torch.randint(N - N // 4 + 1, (), generator=generator))
                row[start:start + N // 4] = 3 * torch.randn(N // 4, generator=generator).double()
        elif kind != "clean":
            raise ValueError(kind)
        # [实验控制] 初始范数相同，避免噪声条件同时改变 softmax 的有效温度。
        cues = math.sqrt(N) * torch.nn.functional.normalize(cues, dim=-1)
        batches.append(cues)
        labels.append(ids)
        names.extend([kind] * len(ids))
    return torch.cat(batches), torch.cat(labels), names


@torch.no_grad()
def d1_retrieve(model, queries, steps):
    """[动力学] 对 (B,N) 查询执行固定预算；vmap 直接复用原有单查询 readout。"""
    if steps <= 0:
        raise ValueError("steps must be positive")
    states = queries.clone().double()
    for _ in range(steps):
        states, _ = torch.vmap(model.readout)(states)
    ids = (states @ model.patterns.T).argmax(dim=-1)
    # [观测·机制] 再读出一次只测残差，不改变上面已经确定的预测 ID。
    next_states, _ = torch.vmap(model.readout)(states)
    residual = torch.linalg.vector_norm(next_states - states, dim=-1)
    return ids, states, residual


def c2_make_directions(model, cues, base_ids, method, count, seed):
    """[输入] 记忆、查询、已有预测；[输出] (B,K,N) 单位切向量及退化计数。"""
    if not 1 <= count <= len(model.patterns):
        raise ValueError("方向数需介于 1 和记忆条数之间")
    generator = _generator(seed)
    B, N = cues.shape
    random_axes = torch.randn((B, count, N), generator=generator, dtype=cues.dtype)
    if method in ("guided", "shuffled"):
        scores = cues @ model.patterns.T
        if method == "shuffled":
            scores = torch.rand(scores.shape, generator=generator, dtype=cues.dtype)
        scores[torch.arange(B), base_ids] = -torch.inf
        others = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :count - 1]
        candidates = torch.cat([base_ids[:, None], others], dim=1)
        axes = model.patterns[candidates]
    elif method == "random":
        axes = random_axes
    else:
        raise ValueError(method)
    tangent = project_to_tangent(cues, axes)
    degenerate = torch.linalg.vector_norm(tangent, dim=-1) < 1e-12
    fallback = project_to_tangent(cues, random_axes)
    tangent = torch.where(degenerate[..., None], fallback, tangent)
    return torch.nn.functional.normalize(tangent, dim=-1), degenerate.sum(dim=-1)


def c3_spherical_probes(cues, axes, angle):
    """[干预] 正负各走 angle 弧度；输出 (B,2K,N)，范数与原查询一致。"""
    return torch.cat([
        spherical_geodesic(cues, axes, angle),
        spherical_geodesic(cues, axes, -angle),
    ], dim=1)


def d2_majority_repair(base_ids, probe_ids, P):
    """[判断] 严格多数才改答；此函数没有真实目标这一输入。"""
    counts = torch.nn.functional.one_hot(probe_ids, num_classes=P).sum(dim=1)
    votes, winners = counts.max(dim=-1)
    return torch.where(votes > probe_ids.shape[1] / 2, winners, base_ids)


def e1_selective_risk(errors, risk_scores, coverage=0.8):
    """保留低风险查询；边界并列用等权抽取的期望，避免按标签打破并列。"""
    errors, scores = np.asarray(errors, dtype=float), np.asarray(risk_scores, dtype=float)
    if len(errors) == 0 or len(errors) != len(scores) or not 0 < coverage <= 1:
        raise ValueError("invalid risk inputs")
    if not np.isfinite(scores).all():
        raise ValueError("risk scores must be finite")
    quota = coverage * len(errors)
    remaining, total = quota, 0.0
    for score in np.unique(scores):
        group = errors[scores == score]
        kept = min(remaining, len(group))
        total += kept * group.mean()
        remaining -= kept
        if remaining <= 1e-10:
            break
    return float(total / quota)


def e2_seed_metrics(rows):
    """[观测·数据] 先在记忆库内计分；推断单位是独立种子，而不是探测票数。"""
    frame = pd.DataFrame(rows)
    records = []
    keys = ["N", "memory_kind", "cue_kind", "seed", "angle", "method"]
    for values, group in frame.groupby(keys, sort=True):
        result = dict(zip(keys, values))
        error = group.base_id.to_numpy() != group.target_id.to_numpy()
        repaired_error = group.repaired_id.to_numpy() != group.target_id.to_numpy()
        result.update(
            queries=len(group), baseline_accuracy=float(1 - error.mean()),
            repaired_accuracy=float(1 - repaired_error.mean()),
            rescued=int((error & ~repaired_error).sum()),
            harmed=int((~error & repaired_error).sum()),
            net_gain=float(error.mean() - repaired_error.mean()),
            nearest_accuracy=float((group.nearest_id == group.target_id).mean()),
            extended_accuracy=float((group.extended_id == group.target_id).mean()),
            risk_probe=e1_selective_risk(error, group.instability),
            risk_gap=e1_selective_risk(error, -group.gap),
            risk_entropy=e1_selective_risk(error, group.entropy),
            baseline_converged=float(group.base_converged.mean()),
            probe_converged=float(group.probe_converged.mean()),
        )
        records.append(result)
    return pd.DataFrame(records)


def e3_paired_intervals(seed_metrics, angle=0.1):
    """[判断] guided 相对各对照的配对风险差；负数有利，8 种子区间仅探索性。"""
    records = []
    subset = seed_metrics[np.isclose(seed_metrics.angle, angle)]
    for conditions, group in subset.groupby(["N", "memory_kind", "cue_kind"]):
        guided = group[group.method == "guided"].set_index("seed").sort_index()
        alternatives = {"gap": guided.risk_gap, "entropy": guided.risk_entropy}
        for method in ("random", "shuffled"):
            alternatives[method] = group[group.method == method].set_index("seed").risk_probe
        for name, other in alternatives.items():
            delta = (guided.risk_probe - other.reindex(guided.index)).to_numpy()
            rng = np.random.default_rng(20260906)
            draws = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            records.append(dict(zip(["N", "memory_kind", "cue_kind"], conditions)) |
                           {"control": name, "risk_difference": float(delta.mean()),
                            "ci_low": float(low), "ci_high": float(high)})
    return pd.DataFrame(records)


def run_probe_grid(config, output_root, progress=print):
    """[编排] a → b → c → d → e；不可覆盖批次，配置或源码变化自动用新目录。"""
    started = datetime.now(timezone.utc).isoformat()
    manifest = {"protocol": "retrieval-reliability-v1", "config": asdict(config),
                "source": source_info(), "torch": torch.__version__,
                "device": "cpu", "threads": torch.get_num_threads()}
    directory = Path(output_root) / fingerprint(manifest)[:16]
    store = BatchStore(directory, manifest)
    all_rows = []
    for N in config.dimensions:
        for kind_index, kind in enumerate(config.memory_kinds):
            for seed in config.seeds:
                key = f"N{N}_{kind}_seed{seed}"
                saved = store.load(key)
                if saved is not None:
                    all_rows.extend(saved)
                    progress(f"复用 {key}：{len(saved)} 条")
                    continue
                tick = time.perf_counter()
                data_seed = N * 100000 + kind_index * 10000 + seed * 100
                patterns = a1_make_memories(N, config.memory_count, kind, data_seed)
                model = b1_store_memories(patterns)
                cues, targets, names = c1_make_queries(patterns, config.targets, config.cue_kinds, data_seed + 1)
                base_ids, _, base_residual = d1_retrieve(model, cues, config.steps)
                extended_ids, _, _ = d1_retrieve(model, cues, (1 + 2 * config.directions) * config.steps)
                logits = model.beta * (cues @ patterns.T)
                nearest = logits.argmax(dim=-1)
                top = logits.topk(2, dim=-1).values
                weights = torch.softmax(logits, dim=-1)
                entropy = -(weights * weights.clamp_min(1e-300).log()).sum(dim=-1) / math.log(config.memory_count)
                rows = []
                for method_index, method in enumerate(config.methods):
                    axes, degenerate = c2_make_directions(model, cues, base_ids, method, config.directions,
                                                         data_seed + 10 + method_index)
                    for angle in config.angles:
                        probes = c3_spherical_probes(cues, axes, angle)
                        ids, _, residual = d1_retrieve(model, probes.flatten(0, 1), config.steps)
                        ids = ids.reshape(len(cues), -1)
                        residual = residual.reshape(len(cues), -1)
                        repaired = d2_majority_repair(base_ids, ids, config.memory_count)
                        for row_index in range(len(cues)):
                            # [评估边界] 到此才把目标 ID 与所有预测拼成记录；它没参与选方向或投票。
                            rows.append({"N": N, "memory_kind": kind, "seed": seed,
                                "query_index": row_index, "cue_kind": names[row_index],
                                "method": method, "angle": angle,
                                "target_id": int(targets[row_index]), "base_id": int(base_ids[row_index]),
                                "nearest_id": int(nearest[row_index]), "extended_id": int(extended_ids[row_index]),
                                "repaired_id": int(repaired[row_index]), "probe_ids": ids[row_index].tolist(),
                                "instability": float((ids[row_index] != base_ids[row_index]).double().mean()),
                                "gap": float(top[row_index, 0] - top[row_index, 1]),
                                "entropy": float(entropy[row_index]),
                                "base_residual": float(base_residual[row_index]),
                                "base_converged": bool(base_residual[row_index] < config.residual_tolerance),
                                "probe_converged": float((residual[row_index] < config.residual_tolerance).double().mean()),
                                "degenerate_directions": int(degenerate[row_index]),
                                "dynamic_readouts": (1 + 2 * config.directions) * config.steps,
                                "observational_readouts": 1 + 2 * config.directions})
                store.save(key, rows)
                all_rows.extend(rows)
                progress(f"完成 {key}：{len(rows)} 条，{time.perf_counter() - tick:.2f}s")
    run_info = {"started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(all_rows), "stop_reason": "frozen_grid_completed"}
    # [溯源] 每次调用单独记时间；不改写数值批次或把复用耗时冒充首次运行耗时。
    atomic_json(directory / ("run_" + started.replace(":", "-") + ".json"), run_info)
    return all_rows, directory, run_info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/retrieval-reliability")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, directory, run_info = run_probe_grid(ProbeConfig(), args.output)
    metrics = e2_seed_metrics(rows)
    metrics.to_csv(directory / "seed_metrics.csv", index=False)
    e3_paired_intervals(metrics).to_csv(directory / "paired_intervals.csv", index=False)
    print(directory, run_info)
