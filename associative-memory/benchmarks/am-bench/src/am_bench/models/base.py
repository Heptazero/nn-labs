"""模型输出与只读观察接口。正确答案由模型外的观察器持有。"""
from dataclasses import dataclass, field
from typing import Any, Callable
import torch

@dataclass
class RetrievalResult:
    # [观测·数据] 最终状态可为二值或连续，具体类型写进 diagnostics
    final_state: torch.Tensor
    status: str
    sweeps: int
    state_updates: int
    # [观测·资源] 实现登记的近似操作数，不等同于硬件实测时间
    retrieval_flops: int
    # [观测·数据] 各模型自己的能量标尺；只检查单模型内是否下降
    energy_trace: list[float] = field(default_factory=list)
    error_trace: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

def binary_state(state: torch.Tensor) -> torch.Tensor:
    values = set(torch.unique(state).tolist())
    if state.ndim != 1 or not values.issubset({-1, 1}):
        raise ValueError("state must be one-dimensional and use {-1, +1}")
    return state.detach().clone().to(torch.float64)

Observer = Callable[[int, torch.Tensor, float | None, dict[str, Any]], None]


def observe(observer: Observer | None, step: int, state: torch.Tensor,
            energy: float | None = None, diagnostics: dict[str, Any] | None = None) -> None:
    if not bool(torch.isfinite(state).all()):
        raise FloatingPointError("non-finite retrieval state")
    if observer is not None:
        # Pass a detached copy: an observer cannot mutate the evolving model state.
        observer(step, state.detach().clone(), energy, dict(diagnostics or {}))
