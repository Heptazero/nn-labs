"""固定预算的可微轨迹与带收敛判据的检索轨迹。"""
from dataclasses import dataclass
import math
from typing import Callable

import torch


StepMap = Callable[[torch.Tensor], torch.Tensor]


def _validate_step(previous: torch.Tensor, following: torch.Tensor) -> None:
    if not isinstance(following, torch.Tensor) or following.shape != previous.shape:
        raise ValueError("step_fn must return a tensor with the same shape as its input")


def fixed_horizon_rollout(
    step_fn: StepMap,
    initial_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """返回含 t=0 的固定长度轨迹，并保留 PyTorch 自动微分图。"""
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    states = [initial_state]
    current = initial_state
    for _ in range(steps):
        following = step_fn(current)
        _validate_step(current, following)
        states.append(following)
        current = following
    return torch.stack(states)


@dataclass(frozen=True)
class ConvergedTrajectory:
    states: torch.Tensor
    residuals: torch.Tensor
    status: str
    steps: int


def converged_rollout(
    step_fn: StepMap,
    initial_state: torch.Tensor,
    *,
    max_steps: int,
    tolerance: float,
) -> ConvergedTrajectory:
    """按无穷范数残差停止单条轨迹；未收敛记录为 max_steps。"""
    if initial_state.ndim != 1:
        raise ValueError("converged_rollout currently accepts one state at a time")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    states = [initial_state]
    residuals = []
    current = initial_state
    status = "max_steps"
    for _ in range(max_steps):
        following = step_fn(current)
        _validate_step(current, following)
        if not bool(torch.isfinite(following).all()):
            raise FloatingPointError("step_fn produced a non-finite state")
        residual = torch.linalg.vector_norm(following - current, ord=float("inf"))
        states.append(following)
        residuals.append(residual)
        current = following
        if float(residual.detach()) < tolerance:
            status = "converged"
            break
    return ConvergedTrajectory(
        states=torch.stack(states),
        residuals=torch.stack(residuals),
        status=status,
        steps=len(residuals),
    )
