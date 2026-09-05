"""Shared a/c/e components and result contracts.

The functions in this module do not inspect model names. A memory set and cue
are generated once, then reused by every model adapter in the same trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class MemorySet:
    """Output of component a: a model-independent set of binary memories."""

    patterns: torch.Tensor
    pattern_ids: tuple[int, ...]
    dataset_id: str
    data_seed: int
    encoding: str = "binary_pm1"

    @property
    def P(self) -> int:
        return int(self.patterns.shape[0])

    @property
    def N(self) -> int:
        return int(self.patterns.shape[1])


@dataclass(frozen=True)
class TrialSpec:
    """Fields that must stay fixed when models share one comparison curve."""

    run_id: str
    task_id: str
    pattern_set_id: str
    target_id: int
    corruption_kind: str
    corruption_level: float
    cue_seed: int
    update_seed: int
    max_sweeps: int
    success_criterion: str = "exact_recall"
    retrieval_budget: str = "equal_max_sweeps"
    resource_budget_type: str = "native"
    stopping_rule: str = "fixed_or_max_sweeps"


@dataclass
class RetrievalResult:
    """Standard output of component d."""

    final_state: torch.Tensor
    status: str
    sweeps: int
    state_updates: int
    retrieval_flops: int
    energy_trace: list[float] = field(default_factory=list)
    error_trace: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def a1_make_independent_binary(
    N: int,
    P: int,
    data_seed: int,
    *,
    device: torch.device | str = "cpu",
) -> MemorySet:
    """Generate P independent {-1, +1} memories of dimension N."""

    if N <= 0 or P <= 0:
        raise ValueError("N and P must be positive")

    bits = torch.randint(
        0,
        2,
        (P, N),
        generator=_generator(data_seed, device),
        device=device,
        dtype=torch.int8,
    )
    patterns = bits.mul(2).sub(1)
    return MemorySet(
        patterns=patterns,
        pattern_ids=tuple(range(P)),
        dataset_id="independent_binary",
        data_seed=int(data_seed),
    )


def c1_make_hamming_cue(
    target: torch.Tensor,
    corruption_level: float,
    cue_seed: int,
) -> torch.Tensor:
    """Flip exactly round(rho*N) coordinates using a predeclared cue seed."""

    if target.ndim != 1:
        raise ValueError("target must be a one-dimensional state")
    if not 0.0 <= corruption_level <= 1.0:
        raise ValueError("corruption_level must lie in [0, 1]")

    cue = target.detach().clone().to(torch.int8)
    flip_count = int(round(float(corruption_level) * cue.numel()))
    if flip_count == 0:
        return cue

    indices = torch.randperm(
        cue.numel(),
        generator=_generator(cue_seed, cue.device),
        device=cue.device,
    )[:flip_count]
    cue[indices] *= -1
    return cue


def e1_exact_recall(final_state: torch.Tensor, target: torch.Tensor) -> bool:
    """Return True only when every coordinate equals the target."""

    return bool(torch.equal(final_state.to(torch.int8), target.to(torch.int8)))


def e2_overlap(final_state: torch.Tensor, target: torch.Tensor) -> float:
    """Mean binary overlap in [-1, 1]."""

    if final_state.shape != target.shape:
        raise ValueError("final_state and target must have identical shapes")
    return float(
        torch.mean(final_state.to(torch.float64) * target.to(torch.float64)).item()
    )


def e3_attractor_class(
    final_state: torch.Tensor,
    target_id: int,
    memories: torch.Tensor,
    status: str,
) -> str:
    """Classify a terminal state without using the model identity."""

    if status not in {"fixed", "one_step"}:
        return "nonconverged"

    final = final_state.to(torch.int8)
    binary_memories = memories.to(torch.int8)
    matches = torch.all(binary_memories == final.unsqueeze(0), dim=1)
    matched_ids = torch.nonzero(matches, as_tuple=False).flatten().tolist()
    if target_id in matched_ids:
        return "target"
    if matched_ids:
        return "wrong_memory"
    if torch.equal(final, -binary_memories[target_id]):
        return "inverse_target"
    return "spurious_fixed"


def e4_top1_memory(final_state: torch.Tensor, memories: torch.Tensor) -> int:
    """Return the stored pattern with the largest binary overlap."""

    scores = memories.to(torch.float64) @ final_state.to(torch.float64)
    return int(torch.argmax(scores).item())
