"""联想记忆实验共享的状态空间几何操作。"""
import math

import torch


def _require_floating_last_dimension(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise ValueError(f"{name} must be a tensor with a state dimension")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if value.shape[-1] < 1:
        raise ValueError(f"{name} must have a non-empty state dimension")


def project_to_sphere(states: torch.Tensor, radius: float) -> torch.Tensor:
    """沿径向把 (..., N) 状态投影到给定半径的球面。"""
    _require_floating_last_dimension(states, "states")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    norms = torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    return states * (radius / norms)


def project_to_tangent(points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """把方向正交投影到球面在 points 处的切空间。"""
    _require_floating_last_dimension(points, "points")
    _require_floating_last_dimension(directions, "directions")
    if points.shape[-1] != directions.shape[-1]:
        raise ValueError("points and directions must share the state dimension")
    expanded = points
    while expanded.ndim < directions.ndim:
        expanded = expanded.unsqueeze(-2)
    squared_radius = torch.sum(expanded.square(), dim=-1, keepdim=True)
    radial_coefficient = torch.sum(directions * expanded, dim=-1, keepdim=True) / squared_radius
    return directions - radial_coefficient * expanded


def normalize_tangent(points: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """投影并归一化切向量；退化方向由调用方决定如何替换。"""
    tangent = project_to_tangent(points, directions)
    norms = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    return tangent / norms


def sample_uniform_sphere(
    count: int,
    dimension: int,
    *,
    radius: float | None = None,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """用归一化高斯向量在球面上均匀采样。"""
    if count <= 0 or dimension <= 0:
        raise ValueError("count and dimension must be positive")
    target_radius = math.sqrt(dimension) if radius is None else float(radius)
    samples = torch.randn(
        (count, dimension), generator=generator, dtype=dtype, device=device
    )
    return project_to_sphere(samples, target_radius)


def sample_tangent_directions(
    points: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """为每个 (..., N) 点生成 count 个随机单位切向量。"""
    _require_floating_last_dimension(points, "points")
    if count <= 0:
        raise ValueError("count must be positive")
    shape = (*points.shape[:-1], count, points.shape[-1])
    axes = torch.randn(shape, generator=generator, dtype=points.dtype, device=points.device)
    return normalize_tangent(points, axes)


def spherical_geodesic(
    points: torch.Tensor,
    directions: torch.Tensor,
    angle: float,
) -> torch.Tensor:
    """沿单位切向量走 angle 弧度，返回同一球面上的点。"""
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    unit_tangent = normalize_tangent(points, directions)
    expanded = points
    while expanded.ndim < unit_tangent.ndim:
        expanded = expanded.unsqueeze(-2)
    radius = torch.linalg.vector_norm(expanded, dim=-1, keepdim=True)
    return math.cos(angle) * expanded + math.sin(angle) * radius * unit_tangent


def spherical_retraction(points: torch.Tensor, tangent_offsets: torch.Tensor) -> torch.Tensor:
    """把 points + tangent_offsets 径向拉回原球面。"""
    tangent = project_to_tangent(points, tangent_offsets)
    expanded = points
    while expanded.ndim < tangent.ndim:
        expanded = expanded.unsqueeze(-2)
    radius = torch.linalg.vector_norm(expanded, dim=-1, keepdim=True)
    candidate = expanded + tangent
    norms = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
    return candidate * (radius / norms)
