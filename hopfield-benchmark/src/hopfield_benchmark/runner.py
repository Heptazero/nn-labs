"""Paired benchmark runner and capacity summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Iterable

import pandas as pd
import torch

from .components import (
    TrialSpec,
    a1_make_independent_binary,
    c1_make_hamming_cue,
    e1_exact_recall,
    e2_overlap,
    e3_attractor_class,
    e4_top1_memory,
)
from .models import ModelAdapter


ModelFactory = Callable[[], ModelAdapter]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Finite, explicit Phase 1 scan configuration."""

    N_values: tuple[int, ...] = (64, 128)
    P_values: tuple[int, ...] = (4, 8, 16, 32)
    corruption_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    pattern_sets: int = 3
    targets_per_set: int = 4
    max_sweeps: int = 20
    base_seed: int = 20260905
    experiment_id: str = "phase1-unspecified"
    git_commit: str = "unknown"

    def validate(self) -> None:
        if not self.N_values or not self.P_values or not self.corruption_levels:
            raise ValueError("scan dimensions must not be empty")
        if min(self.N_values) <= 0 or min(self.P_values) <= 0:
            raise ValueError("N and P values must be positive")
        if not all(0.0 <= level <= 1.0 for level in self.corruption_levels):
            raise ValueError("corruption levels must lie in [0, 1]")
        if self.pattern_sets <= 0 or self.targets_per_set <= 0:
            raise ValueError("replicate counts must be positive")
        if self.max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if not self.git_commit.strip():
            raise ValueError("git_commit must not be empty")


def _seed(base: int, *coordinates: int) -> int:
    """Stable integer mixer; unlike Python hash(), it is reproducible."""

    value = int(base) & 0x7FFFFFFF
    for coordinate in coordinates:
        value = (1_103_515_245 * value + 12_345 + int(coordinate)) & 0x7FFFFFFF
    return value


def _failure_row(
    spec: TrialSpec,
    *,
    model_id: str,
    N: int,
    P: int,
    dataset_id: str,
    encoding: str,
    data_seed: int,
    experiment_id: str,
    git_commit: str,
    failure_reason: str,
) -> dict[str, object]:
    return {
        **asdict(spec),
        "model_id": model_id,
        "experiment_id": experiment_id,
        "git_commit": git_commit,
        "model_config": "unavailable",
        "dataset_id": dataset_id,
        "encoding": encoding,
        "data_seed": data_seed,
        "structure_seed": None,
        "N": N,
        "P": P,
        "status": "numerical_failure",
        "sweeps": 0,
        "state_updates": 0,
        "exact_recall": False,
        "one_sweep_unchanged": False,
        "overlap": float("nan"),
        "mse": float("nan"),
        "top1_id": float("nan"),
        "attractor_class": "nonconverged",
        "parameter_count": float("nan"),
        "storage_bytes": float("nan"),
        "retrieval_flops": float("nan"),
        "wall_time_ms": float("nan"),
        "right_censored": False,
        "failure_reason": failure_reason,
        "paper_reported": False,
        "energy_trace": [],
        "error_trace": [],
    }


def run_paired_benchmark(
    model_factories: dict[str, ModelFactory],
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Run every registered model on exactly the same generated cases."""

    config.validate()
    if len(model_factories) < 2:
        raise ValueError("a comparison requires at least two registered models")

    rows: list[dict[str, object]] = []
    run_counter = 0
    for N_index, N in enumerate(config.N_values):
        for P_index, P in enumerate(config.P_values):
            for set_index in range(config.pattern_sets):
                data_seed = _seed(config.base_seed, N_index, P_index, set_index)
                memory_set = a1_make_independent_binary(N, P, data_seed)
                pattern_set_id = f"N{N}-P{P}-set{set_index}-seed{data_seed}"
                fitted_models: dict[str, ModelAdapter] = {}
                fit_failures: dict[str, str] = {}
                for registered_id, factory in model_factories.items():
                    try:
                        adapter = factory()
                        if adapter.model_id != registered_id:
                            raise ValueError(
                                f"registry key {registered_id!r} does not match "
                                f"adapter model_id {adapter.model_id!r}"
                            )
                        fitted_models[registered_id] = adapter.fit(memory_set.patterns)
                    except Exception as error:  # failures remain in the raw table
                        fit_failures[registered_id] = f"{type(error).__name__}: {error}"

                target_count = min(P, config.targets_per_set)
                for target_id in range(target_count):
                    target = memory_set.patterns[target_id]
                    for level_index, corruption_level in enumerate(
                        config.corruption_levels
                    ):
                        cue_seed = _seed(
                            config.base_seed,
                            N_index,
                            P_index,
                            set_index,
                            target_id,
                            level_index,
                        )
                        update_seed = _seed(cue_seed, 91)
                        cue = c1_make_hamming_cue(target, corruption_level, cue_seed)
                        task_id = "U1_clean_fixed_point" if corruption_level == 0 else "U2_noise_recovery"
                        spec = TrialSpec(
                            run_id=f"{config.experiment_id}-{run_counter:07d}",
                            task_id=task_id,
                            pattern_set_id=pattern_set_id,
                            target_id=target_id,
                            corruption_kind="hamming_flip",
                            corruption_level=float(corruption_level),
                            cue_seed=cue_seed,
                            update_seed=update_seed,
                            max_sweeps=config.max_sweeps,
                        )
                        run_counter += 1

                        for model_id in model_factories:
                            if model_id in fit_failures:
                                rows.append(
                                    _failure_row(
                                        spec,
                                        model_id=model_id,
                                        N=N,
                                        P=P,
                                        dataset_id=memory_set.dataset_id,
                                        encoding=memory_set.encoding,
                                        data_seed=data_seed,
                                        experiment_id=config.experiment_id,
                                        git_commit=config.git_commit,
                                        failure_reason=fit_failures[model_id],
                                    )
                                )
                                continue

                            model = fitted_models[model_id]
                            try:
                                started_at = perf_counter()
                                result = model.retrieve(
                                    cue,
                                    target=target,
                                    update_seed=update_seed,
                                    max_sweeps=config.max_sweeps,
                                )
                                wall_time_ms = (perf_counter() - started_at) * 1_000.0
                                resources = model.resource_summary()
                                rows.append(
                                    {
                                        **asdict(spec),
                                        "model_id": model_id,
                                        "experiment_id": config.experiment_id,
                                        "git_commit": config.git_commit,
                                        **resources,
                                        "dataset_id": memory_set.dataset_id,
                                        "encoding": memory_set.encoding,
                                        "data_seed": data_seed,
                                        "structure_seed": None,
                                        "N": N,
                                        "P": P,
                                        "status": result.status,
                                        "sweeps": result.sweeps,
                                        "state_updates": result.state_updates,
                                        "exact_recall": e1_exact_recall(
                                            result.final_state, target
                                        ),
                                        "one_sweep_unchanged": bool(
                                            result.diagnostics.get(
                                                "first_sweep_unchanged", False
                                            )
                                        ),
                                        "overlap": e2_overlap(
                                            result.final_state, target
                                        ),
                                        "mse": float(
                                            (
                                                result.final_state.to(torch.float64)
                                                - target.to(torch.float64)
                                            )
                                            .pow(2)
                                            .mean()
                                            .item()
                                        ),
                                        "top1_id": e4_top1_memory(
                                            result.final_state,
                                            memory_set.patterns,
                                        ),
                                        "attractor_class": e3_attractor_class(
                                            result.final_state,
                                            target_id,
                                            memory_set.patterns,
                                            result.status,
                                        ),
                                        "retrieval_flops": result.retrieval_flops,
                                        "wall_time_ms": wall_time_ms,
                                        "right_censored": False,
                                        "failure_reason": "",
                                        "paper_reported": False,
                                        "energy_trace": result.energy_trace,
                                        "error_trace": result.error_trace,
                                    }
                                )
                            except Exception as error:
                                rows.append(
                                    _failure_row(
                                        spec,
                                        model_id=model_id,
                                        N=N,
                                        P=P,
                                        dataset_id=memory_set.dataset_id,
                                        encoding=memory_set.encoding,
                                        data_seed=data_seed,
                                        experiment_id=config.experiment_id,
                                        git_commit=config.git_commit,
                                        failure_reason=f"{type(error).__name__}: {error}",
                                    )
                                )

    results = pd.DataFrame(rows)
    validate_paired_results(results, expected_models=tuple(model_factories))
    return results


PAIR_KEY = (
    "run_id",
    "experiment_id",
    "git_commit",
    "task_id",
    "pattern_set_id",
    "target_id",
    "dataset_id",
    "encoding",
    "N",
    "P",
    "corruption_kind",
    "corruption_level",
    "data_seed",
    "cue_seed",
    "update_seed",
    "max_sweeps",
    "success_criterion",
    "retrieval_budget",
    "resource_budget_type",
    "stopping_rule",
)


def validate_paired_results(
    results: pd.DataFrame,
    *,
    expected_models: Iterable[str] | None = None,
) -> None:
    """Refuse comparison when a trial is missing or duplicates a model."""

    missing_columns = set(PAIR_KEY).difference(results.columns)
    if missing_columns:
        raise ValueError(f"missing pairing columns: {sorted(missing_columns)}")
    if results.empty:
        raise ValueError("results must not be empty")

    models = tuple(expected_models or sorted(results["model_id"].unique()))
    expected = set(models)
    for key, group in results.groupby(list(PAIR_KEY), dropna=False, sort=False):
        observed = list(group["model_id"])
        if len(observed) != len(set(observed)):
            raise ValueError(f"duplicate model in paired trial {key}")
        if set(observed) != expected:
            raise ValueError(
                f"unpaired trial {key}: expected {sorted(expected)}, "
                f"observed {sorted(observed)}"
            )


def capacity_summary(
    results: pd.DataFrame,
    *,
    success_threshold: float = 0.9,
) -> pd.DataFrame:
    """Estimate finite-scan Pc and mark scans that never cross the threshold."""

    if not 0.0 < success_threshold < 1.0:
        raise ValueError("success_threshold must lie in (0, 1)")

    clean = results[results["corruption_level"] == 0.0]
    rates = (
        clean.groupby(["model_id", "N", "P"], as_index=False)[
            "one_sweep_unchanged"
        ]
        .mean()
        .rename(columns={"one_sweep_unchanged": "success_rate"})
    )
    summaries: list[dict[str, object]] = []
    for (model_id, N), group in rates.groupby(["model_id", "N"], sort=False):
        ordered = group.sort_values("P")
        passing = ordered[ordered["success_rate"] >= success_threshold]
        if passing.empty:
            Pc = int(ordered["P"].min())
            left_censored = True
            right_censored = False
        else:
            Pc = int(passing["P"].max())
            left_censored = False
            right_censored = bool(Pc == int(ordered["P"].max()))
        summaries.append(
            {
                "model_id": model_id,
                "N": int(N),
                "P_c": Pc,
                "success_threshold": success_threshold,
                "left_censored": left_censored,
                "right_censored": right_censored,
                "capacity_kind": "clean_fixed_point_capacity",
            }
        )
    return pd.DataFrame(summaries)
