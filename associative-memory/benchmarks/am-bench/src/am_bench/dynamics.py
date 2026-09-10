"""Q04：共同查询的逐轮纠错；答案仅在观察器中使用。"""
from dataclasses import asdict, dataclass
from time import perf_counter
import math
import pandas as pd
import torch

from .models import ClassicalHopfield, PolynomialDAM, ExponentialDAM, SimplicialR12
from .models.modern import IterativeModernHopfield
from .runner import stable_seed
from .storage import BatchStore
from .tasks import a1_make_independent_binary, c1_make_hamming_cue


@dataclass(frozen=True)
class DynamicsConfig:
    N_values: tuple[int, ...] = (32, 64)
    P_values: tuple[int, ...] = (4, 8)
    corruption_levels: tuple[float, ...] = (0.0, 0.1, 0.3)
    seeds: tuple[int, ...] = (0, 1, 2)
    targets_per_set: int = 4
    max_sweeps: int = 8
    models: tuple[str, ...] = ("classical_hebb", "polynomial_dam_d3", "continuous_modern_iterative")
    modern_beta: float = 0.1

    def validate(self):
        integers = (*self.N_values, *self.P_values, *self.seeds, self.targets_per_set, self.max_sweeps)
        if any(type(x) is not int for x in integers):
            raise ValueError("N/P/seeds/targets/sweeps must be integers")
        for values in (self.N_values, self.P_values, self.corruption_levels, self.seeds, self.models):
            if not values or len(values) != len(set(values)):
                raise ValueError("配置不得为空或含重复扫描点")
        if min(self.N_values) < 2 or min(self.P_values) < 2:
            raise ValueError("N/P must be at least 2")
        if self.targets_per_set < 1 or self.max_sweeps < 1:
            raise ValueError("targets and sweep budget must be positive")
        if len(self.models) < 2 or set(self.models) - set(model_factories(self)):
            raise ValueError("Q04 至少选择两种已登记的迭代模型")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in self.corruption_levels):
            raise ValueError("invalid corruption level")
        if not math.isfinite(self.modern_beta) or self.modern_beta <= 0:
            raise ValueError("modern_beta must be positive and finite")

    @property
    def batch_count(self):
        return len(self.N_values) * len(self.P_values) * len(self.seeds)

    @property
    def retrieval_count(self):
        return (len(self.N_values) * sum(min(p, self.targets_per_set) for p in self.P_values)
                * len(self.corruption_levels) * len(self.seeds) * len(self.models))


def model_factories(config):
    return {
        "classical_hebb": ClassicalHopfield,
        "polynomial_dam_d3": lambda: PolynomialDAM(3),
        "exponential_dam": ExponentialDAM,
        "simplicial_r12_t50": lambda: SimplicialR12(0.5, structure_seed=31415),
        "continuous_modern_iterative": lambda: IterativeModernHopfield(config.modern_beta),
    }


def _trial(model, cue, target_id, patterns, update_seed, max_sweeps, fit_error=None):
    snapshots = {}
    target = patterns[target_id].double()

    def record(step, state, energy, diagnostics):
        state = state.double()
        scores = patterns.double() @ state
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("non-finite state or scores")
        predicted = int(scores.argmax().item())
        snapshots[step] = {
            "step": step, "predicted_id": predicted, "top1_correct": predicted == target_id,
            "top1_tie_count": int((scores == scores.max()).sum().item()),
            "mse": float((state - target).square().mean().item()),
            "bit_error": float((torch.where(state >= 0, 1.0, -1.0) != target).double().mean().item()),
            "energy": energy, "update_weight_entropy": diagnostics.get("attention_entropy"),
            "update_max_weight": diagnostics.get("max_weight"),
            "observed": True, "held_after_fixed": False,
        }

    # 共同 t=0 基线不依赖模型是否构建成功。
    record(0, cue, None, {})
    result = None
    error = fit_error
    started = perf_counter()
    if error is None:
        try:
            result = model.retrieve(cue, update_seed=update_seed, max_sweeps=max_sweeps, observer=record)
            if result.status not in {"fixed", "max_steps"}:
                raise ValueError("Q04 requires a registered iterative configuration")
            if set(snapshots) != set(range(result.sweeps + 1)):
                raise ValueError("model omitted a trajectory observation")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = None
    elapsed = perf_counter() - started
    rows = []
    for step in range(max_sweeps + 1):
        if step in snapshots:
            row = dict(snapshots[step])
        elif result is not None and result.status == "fixed":
            row = {**snapshots[result.sweeps], "step": step,
                   "observed": False, "held_after_fixed": True}
        else:
            row = {"step": step, "predicted_id": None, "top1_correct": False,
                   "top1_tie_count": None, "mse": None, "bit_error": None, "energy": None,
                   "update_weight_entropy": None, "update_max_weight": None,
                   "observed": False, "held_after_fixed": False}
        row["available"] = row["observed"] or row["held_after_fixed"]
        row["status"] = "numerical_failure" if error else result.status
        row["failure_reason"] = error
        row["elapsed_seconds_with_observation"] = elapsed
        # 现有适配器的登记近似成本按完整轮均摊；不是硬件测得的 FLOPs。
        row["estimated_cumulative_flops"] = (
            result.retrieval_flops * min(step, result.sweeps) / result.sweeps
            if result is not None else (0 if step == 0 else None)
        )
        rows.append(row)
    return rows


def validate_batch(rows, config, N, P, seed):
    expected = {(target, float(level), model, step)
                for target in range(min(P, config.targets_per_set))
                for level in config.corruption_levels for model in config.models
                for step in range(config.max_sweeps + 1)}
    observed = [(r["target_id"], r["corruption_level"], r["model_id"], r["step"]) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("incomplete or duplicate paired trajectory batch")
    if any((r["N"], r["P"], r["seed"]) != (N, P, seed) for r in rows):
        raise ValueError("batch identity mismatch")


def run_dynamics(config, directory, provenance, *, max_new_batches=None):
    """一次存一个完整记忆库批次；重跑会核对配置并跳过已有批次。"""
    config.validate()
    if max_new_batches is not None and max_new_batches < 1:
        raise ValueError("max_new_batches must be positive")
    factories = model_factories(config)
    manifest = {"task": "Q04", "protocol_version": 1, "config": asdict(config),
                "provenance": provenance, "torch_version": str(torch.__version__),
                "device": "cpu", "dtype": "float64", "budget": "native_equal_max_sweeps",
                "criterion": "dot_product_top1_first_index_tie_break",
                "model_configs": {name: ("beta=" + str(config.modern_beta)
                                         if name == "continuous_modern_iterative" else "registered_default")
                                  for name in config.models}}
    store = BatchStore(directory, manifest)
    all_rows = []
    new_count = 0
    completed = 0
    for N in config.N_values:
        for P in config.P_values:
            for seed in config.seeds:
                key = f"N{N}_P{P}_seed{seed}"
                rows = store.load(key)
                if rows is None:
                    if max_new_batches is not None and new_count >= max_new_batches:
                        continue
                    started = perf_counter()
                    data_seed = stable_seed(20260905, N, P, seed)
                    memories = a1_make_independent_binary(N, P, data_seed)
                    fitted, failures = {}, {}
                    for name in config.models:
                        try:
                            fitted[name] = factories[name]().fit(memories.patterns)
                        except Exception as exc:
                            failures[name] = f"{type(exc).__name__}: {exc}"
                    rows = []
                    for target in range(min(P, config.targets_per_set)):
                        for level_index, level in enumerate(config.corruption_levels):
                            cue_seed = stable_seed(data_seed, target, level_index)
                            update_seed = stable_seed(cue_seed, 91)
                            cue = c1_make_hamming_cue(memories.patterns[target], level, cue_seed)
                            for name in config.models:
                                trial = _trial(fitted.get(name), cue, target, memories.patterns,
                                               update_seed, config.max_sweeps, failures.get(name))
                                for row in trial:
                                    rows.append({"N": N, "P": P, "seed": seed,
                                                 "pattern_set_id": key, "target_id": target,
                                                 "corruption_level": float(level), "model_id": name,
                                                 "data_seed": data_seed, "cue_seed": cue_seed,
                                                 "update_seed": update_seed, **row})
                    validate_batch(rows, config, N, P, seed)
                    store.save(key, rows)
                    new_count += 1
                    print(f"Saved {key}: {perf_counter() - started:.1f}s, {len(rows)} observations")
                validate_batch(rows, config, N, P, seed)
                all_rows.extend(rows)
                completed += 1
    frame = pd.DataFrame(all_rows)
    return frame, {"completed_batches": completed, "expected_batches": config.batch_count,
                   "complete": completed == config.batch_count, "manifest_hash": store.signature}


def transition_table(frame, *, relative_to_first=False):
    """条件率只用于描述；同时返回分母，失效后的缺测不伪装成错误状态。"""
    keys = ["N", "P", "seed", "pattern_set_id", "target_id", "corruption_level", "model_id"]
    ordered = frame.sort_values(keys + ["step"]).copy()
    if relative_to_first:
        first = ordered[ordered.step == 1][keys + ["top1_correct", "available"]]
        first = first.rename(columns={"top1_correct": "previous_correct", "available": "previous_available"})
        ordered = ordered.merge(first, on=keys, validate="many_to_one")
        ordered = ordered[ordered.step > 1].copy()
    else:
        ordered["previous_correct"] = ordered.groupby(keys)["top1_correct"].shift(1)
        ordered["previous_available"] = ordered.groupby(keys)["available"].shift(1)
        ordered = ordered[ordered.step > 0].copy()
    valid = ordered["available"] & ordered["previous_available"].fillna(False).astype(bool)
    previous = ordered["previous_correct"].fillna(False).astype(bool)
    ordered["valid_pair"] = valid
    ordered["previous_wrong"] = valid & ~previous
    ordered["previous_right"] = valid & previous
    ordered["wrong_to_right"] = valid & ~previous & ordered.top1_correct
    ordered["right_to_wrong"] = valid & previous & ~ordered.top1_correct
    counts = ordered.groupby(["N", "P", "corruption_level", "model_id", "step"], as_index=False)[
        ["valid_pair", "previous_wrong", "previous_right", "wrong_to_right", "right_to_wrong"]].sum()
    counts["correction_rate"] = counts.wrong_to_right / counts.previous_wrong.replace(0, float("nan"))
    counts["damage_rate"] = counts.right_to_wrong / counts.previous_right.replace(0, float("nan"))
    return counts


def plot_dynamics_curves(frame, *, N, P, corruption_level):
    """整组样本的识别率；阴影是记忆库间 ±1 SE，不是显著性结论。"""
    import matplotlib.pyplot as plt
    subset = frame[(frame.N == N) & (frame.P == P) & (frame.corruption_level == corruption_level)]
    if subset.empty:
        raise ValueError("selected condition has no results")
    rates = subset.groupby(["model_id", "pattern_set_id", "step"], as_index=False).agg(
        success=("top1_correct", "mean"), coverage=("available", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, group in rates.groupby("model_id"):
        curve = group.groupby("step").success.agg(["mean", "std", "count"])
        se = curve["std"] / curve["count"].pow(0.5)
        axes[0].plot(curve.index, curve["mean"], marker="o", label=name)
        axes[0].fill_between(curve.index, (curve["mean"] - se).clip(0, 1),
                             (curve["mean"] + se).clip(0, 1), alpha=0.15)
        coverage = group.groupby("step").coverage.mean()
        axes[1].plot(coverage.index, coverage, marker="o", label=name)
    axes[0].set(title=f"Q04 | N={N}, P={P}, noise={corruption_level}",
                xlabel="Registered update round", ylabel="Top-1 target success", ylim=(-0.03, 1.03))
    axes[1].set(title="Observed or explicitly held after fixed point",
                xlabel="Registered update round", ylabel="Available fraction", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.legend(fontsize=8)
    fig.tight_layout()
    return fig
