"""配对运行器：保留 phase-one 协议，逐步任务见 dynamics.py。"""
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from time import perf_counter
import torch
import pandas as pd
from .tasks import a1_make_independent_binary, c1_make_hamming_cue
from .metrics import retrieve_measured, e1_exact_recall, e2_overlap, e3_top1_memory, e4_attractor_class

SOURCE_NOTEBOOK = (
    "associative-memory/benchmarks/am-bench/notebooks/"
    "hopfield_benchmark_phase1_colab.ipynb"
)

# [实验控制] Gate 在运行前冻结扫描网格、重复数、停止上限与基础种子
@dataclass(frozen=True)
class BenchmarkConfig:
    N_values: tuple[int, ...]
    P_values: tuple[int, ...]
    corruption_levels: tuple[float, ...]
    pattern_sets: int
    targets_per_set: int
    max_sweeps: int
    base_seed: int
    experiment_id: str
    source_commit: str

    # [约束] 在生成任何结果前一次性拒绝空扫描、非法噪声和零重复
    def validate(self) -> None:
        if not self.N_values or not self.P_values or not self.corruption_levels:
            raise ValueError("scan dimensions must not be empty")
        if min(self.N_values) <= 0 or min(self.P_values) <= 0:
            raise ValueError("N and P must be positive")
        if not all(0.0 <= level <= 1.0 for level in self.corruption_levels):
            raise ValueError("corruption levels must lie in [0, 1]")
        if self.pattern_sets <= 0 or self.targets_per_set <= 0 or self.max_sweeps <= 0:
            raise ValueError("replicate counts and max_sweeps must be positive")


# [实验控制] 用坐标确定性派生子种子；循环顺序改变时 trial 随机性仍可追踪
def stable_seed(base: int, *coordinates: int) -> int:
    value = int(base) & 0x7FFFFFFF
    for coordinate in coordinates:
        value = (1_103_515_245 * value + 12_345 + int(coordinate)) & 0x7FFFFFFF
    return value


# [数据契约] 这些字段共同定义一条配对 trial；颜色不能掩盖输入条件变化
PAIR_KEY = [
    "run_id", "experiment_id", "source_commit", "task_id",
    "pattern_set_id", "target_id", "dataset_id", "encoding",
    "N", "P", "corruption_kind", "corruption_level",
    "data_seed", "cue_seed", "update_seed", "max_sweeps",
    "success_criterion", "retrieval_budget",
    "resource_budget_type", "stopping_rule",
]


# [判断] 作图前检查每个 trial 是否恰好包含一次完整模型集合
def validate_paired_results(
    frame: pd.DataFrame,
    expected_models: Iterable[str] | None = None,
) -> None:
    missing = set(PAIR_KEY).difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"invalid result table; missing={sorted(missing)}")
    # [中介变量] expected 来自预注册工厂，不从成功记录反推，避免失败模型消失
    expected = set(expected_models or sorted(frame["model_id"].unique()))
    for key, group in frame.groupby(PAIR_KEY, dropna=False, sort=False):
        observed = list(group["model_id"])
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(f"unpaired trial {key}: observed={observed}")


# [失败边界] 异常也生成标准行并留在分母；NaN 只用于本来没有定义的量
def failure_row(
    common: dict[str, Any],
    model_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        **common,
        "model_id": model_id,
        "model_config": "unavailable",
        "status": "numerical_failure",
        "sweeps": 0,
        "state_updates": 0,
        "retrieval_mode": "unavailable",
        "one_sweep_unchanged": False,
        "output_kind": "unknown",
        # [边界] 当前唯一无离散固定点语义的是 continuous_modern
        "fixed_point_eligible": model_id != "continuous_modern",
        "exact_recall": float("nan"),
        "top1_correct": False,
        "overlap": float("nan"),
        "mse": float("nan"),
        "top1_id": float("nan"),
        "attractor_class": "nonconverged",
        "parameter_count": float("nan"),
        "storage_bytes": float("nan"),
        "retrieval_flops": float("nan"),
        "wall_time_ms": float("nan"),
        "right_censored": False,
        "failure_reason": reason,
        "paper_reported": False,
        "energy_trace": [],
        "error_trace": [],
    }


# [执行] 一次生成 memory/cue，再依次交给全部模型；模型循环内禁止重采样
def run_paired_benchmark(
    model_factories: dict[str, Callable[[], Any]],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    config.validate()
    # [约束] 少于两条模型线就不再是跨模型基准
    if len(model_factories) < 2:
        raise ValueError("at least two models are required")
    rows: list[dict[str, Any]] = []
    case_index = 0
    for N_index, N in enumerate(config.N_values):
        for P_index, P in enumerate(config.P_values):
            for set_index in range(config.pattern_sets):
                # [实验控制] data_seed 只由 N、P、pattern-set 坐标决定
                data_seed = stable_seed(config.base_seed, N_index, P_index, set_index)
                # [输入] 同一个 MemorySet 会被本条件下全部模型读取
                memories = a1_make_independent_binary(N, P, data_seed)
                pattern_set_id = f"N{N}-P{P}-set{set_index}-seed{data_seed}"
                # [中介变量] 每个模型每个 pattern set 只 fit 一次，多个 cue 复用存储结果
                fitted: dict[str, Any] = {}
                fit_failures: dict[str, str] = {}
                for registered_id, factory in model_factories.items():
                    try:
                        model = factory()
                        if model.model_id != registered_id:
                            raise ValueError("registry key and model_id differ")
                        fitted[registered_id] = model.fit(memories.patterns)
                    # [失败边界] fit 失败先登记原因；随后每条配对 trial 都保留失败占位行
                    except Exception as error:
                        fit_failures[registered_id] = f"{type(error).__name__}: {error}"
                for target_id in range(min(P, config.targets_per_set)):
                    target = memories.patterns[target_id]
                    for level_index, level in enumerate(config.corruption_levels):
                        # [实验控制] cue_seed 与 update_seed 分离，方便独立复现实验噪声和动力学
                        cue_seed = stable_seed(
                            config.base_seed, N_index, P_index, set_index,
                            target_id, level_index,
                        )
                        update_seed = stable_seed(cue_seed, 91)
                        # [输入] cue 在 model_id 循环外构造，保证逐元素一致
                        cue = c1_make_hamming_cue(target, level, cue_seed)
                        # [数据契约] common 字段复制进每个模型结果，供配对检查逐项核对
                        common = {
                            "run_id": f"{config.experiment_id}-{case_index:07d}",
                            "experiment_id": config.experiment_id,
                            "source_commit": config.source_commit,
                            "source_notebook": SOURCE_NOTEBOOK,
                            "task_id": (
                                "U1_clean_fixed_point"
                                if level == 0.0
                                else "U2_noise_recovery"
                            ),
                            "pattern_set_id": pattern_set_id,
                            "target_id": target_id,
                            "dataset_id": memories.dataset_id,
                            "encoding": memories.encoding,
                            "N": N,
                            "P": P,
                            "corruption_kind": "hamming_flip",
                            "corruption_level": float(level),
                            "data_seed": data_seed,
                            "structure_seed": None,
                            "cue_seed": cue_seed,
                            "update_seed": update_seed,
                            "max_sweeps": config.max_sweeps,
                            "success_criterion": "top1_memory_identification",
                            "retrieval_budget": "native_registered_rule",
                            "resource_budget_type": "native",
                            "stopping_rule": "model_native",
                        }
                        case_index += 1
                        for model_id in model_factories:
                            if model_id in fit_failures:
                                rows.append(failure_row(common, model_id, fit_failures[model_id]))
                                continue
                            model = fitted[model_id]
                            try:
                                # [观测·资源] wall-clock 只做描述；公平图主要使用登记 FLOPs
                                started = perf_counter()
                                result = retrieve_measured(
                                    model, cue,
                                    target=target,
                                    update_seed=update_seed,
                                    max_sweeps=config.max_sweeps,
                                )
                                elapsed_ms = (perf_counter() - started) * 1_000.0
                                final_f = result.final_state.to(torch.float64)
                                target_f = target.to(torch.float64)
                                # [边界] 输出类型来自适配器，不根据 model_id 在指标函数里偷换
                                output_kind = result.diagnostics.get("output_kind", "binary")
                                fixed_point_eligible = bool(
                                    result.diagnostics.get(
                                        "fixed_point_eligible", output_kind == "binary"
                                    )
                                )
                                # [观测·数据] 所有模型共享的主成功判据：Top-1 是否等于 target_id
                                top1_id = e3_top1_memory(
                                    result.final_state, memories.patterns
                                )
                                rows.append({
                                    **common,
                                    "model_id": model_id,
                                    **model.resource_summary(),
                                    "status": result.status,
                                    "sweeps": result.sweeps,
                                    "state_updates": result.state_updates,
                                    "retrieval_mode": (
                                        "one_step"
                                        if result.status == "one_step"
                                        else "iterative_async"
                                    ),
                                    "output_kind": output_kind,
                                    "fixed_point_eligible": fixed_point_eligible,
                                    "one_sweep_unchanged": bool(
                                        result.diagnostics.get("first_sweep_unchanged", False)
                                    ),
                                    "exact_recall": (
                                        e1_exact_recall(result.final_state, target)
                                        if output_kind == "binary"
                                        else float("nan")
                                    ),
                                    "top1_correct": top1_id == target_id,
                                    "overlap": e2_overlap(result.final_state, target),
                                    "mse": float((final_f - target_f).pow(2).mean().item()),
                                    "top1_id": top1_id,
                                    "attractor_class": e4_attractor_class(
                                        result.final_state, target_id, memories.patterns,
                                        result.status, output_kind,
                                    ),
                                    "retrieval_flops": result.retrieval_flops,
                                    "wall_time_ms": elapsed_ms,
                                    "right_censored": False,
                                    "failure_reason": "",
                                    "paper_reported": False,
                                    "energy_trace": result.energy_trace,
                                    "error_trace": result.error_trace,
                                    "diagnostics": result.diagnostics,
                                })
                            # [失败边界] 单次检索失败不终止整批，也不从统计分母删除
                            except Exception as error:
                                reason = f"{type(error).__name__}: {error}"
                                rows.append(failure_row(common, model_id, reason))
    frame = pd.DataFrame(rows)
    # [判断] 只有完整配对表才能离开运行器进入 f 作图组件
    validate_paired_results(frame, tuple(model_factories))
    return frame


# [汇总] 从 clean cue 的一步不变率定义有限扫描经验容量，不做渐近外推
def capacity_summary(frame: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    # [边界] 连续模型不具备这里的离散固定点语义，因此显式排除
    clean = frame[
        (frame["corruption_level"] == 0.0)
        & frame["fixed_point_eligible"]
    ]
    rates = (
        clean.groupby(["model_id", "N", "P"], as_index=False)["one_sweep_unchanged"]
        .mean()
        .rename(columns={"one_sweep_unchanged": "success_rate"})
    )
    rows = []
    for (model_id, N), group in rates.groupby(["model_id", "N"], sort=False):
        ordered = group.sort_values("P")
        # [判断] P_c 是扫描网格中最后一个达到预注册阈值的点
        passing = ordered[ordered["success_rate"] >= threshold]
        # [删失] 最小 P 都失败只得到左删失上界，不伪造临界点
        if passing.empty:
            critical = int(ordered["P"].min())
            left_censored, right_censored = True, False
        else:
            critical = int(passing["P"].max())
            left_censored = False
            # [删失] 最大 P 仍成功只得到下界，不能拿去拟合容量阶数
            right_censored = critical == int(ordered["P"].max())
        rows.append({
            "model_id": model_id, "N": int(N), "P_c": critical,
            "success_threshold": threshold,
            "left_censored": left_censored,
            "right_censored": right_censored,
            "capacity_kind": "clean_fixed_point_capacity",
        })
    return pd.DataFrame(rows)
