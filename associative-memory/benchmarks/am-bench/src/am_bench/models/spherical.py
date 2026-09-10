"""球面连续多项式 Dense Associative Memory。"""
import math

import torch

from ..state_spaces import project_to_sphere
from ..trajectory import converged_rollout
from .base import Observer, RetrievalResult, observe


class SphericalPolynomialDAM:
    """在半径 sqrt(N) 的球面上执行同步投影检索。"""

    def __init__(self, degree: int = 3, convergence_tolerance: float = 1e-7) -> None:
        if type(degree) is not int or degree < 2:
            raise ValueError("degree must be an integer of at least 2")
        if not math.isfinite(convergence_tolerance) or convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive and finite")
        self.degree = degree
        self.convergence_tolerance = float(convergence_tolerance)
        self.model_id = f"spherical_polynomial_dam_d{degree}"

    def fit(self, patterns: torch.Tensor) -> "SphericalPolynomialDAM":
        if patterns.ndim != 2 or not patterns.is_floating_point():
            raise ValueError("patterns must be a floating tensor with shape [P, N]")
        if patterns.shape[0] < 1 or patterns.shape[1] < 1:
            raise ValueError("patterns must be non-empty")
        if not bool(torch.isfinite(patterns).all()):
            raise ValueError("patterns must be finite")
        self.P, self.N = map(int, patterns.shape)
        self.radius = math.sqrt(self.N)
        expected = torch.full(
            (self.P,), self.radius, dtype=patterns.dtype, device=patterns.device
        )
        if not torch.allclose(
            torch.linalg.vector_norm(patterns, dim=-1), expected, rtol=1e-9, atol=1e-10
        ):
            raise ValueError("every pattern must lie on the sphere of radius sqrt(N)")
        self.patterns = patterns.detach().clone()
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "patterns"):
            raise RuntimeError("fit must be called before energy, step, or retrieve")

    def overlaps(self, states: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        if states.shape[-1] != self.N:
            raise ValueError("state dimension does not match the stored patterns")
        return states.to(self.patterns.dtype) @ self.patterns.T / self.N

    def energy(self, states: torch.Tensor) -> torch.Tensor:
        """返回 H(x)=-(1/d) sum_mu overlap_mu**d，并保留梯度。"""
        values = self.overlaps(states)
        return -torch.sum(values.pow(self.degree), dim=-1) / self.degree

    def step(self, states: torch.Tensor) -> torch.Tensor:
        """执行 Proj_S(Xi F'(Xi^T x/N)) 的一次同步更新。"""
        values = self.overlaps(states)
        field = values.pow(self.degree - 1) @ self.patterns
        return project_to_sphere(field, self.radius)

    @torch.no_grad()
    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        observer: Observer | None = None,
        update_seed: int,
        max_sweeps: int,
    ) -> RetrievalResult:
        del update_seed
        self._require_fitted()
        initial = project_to_sphere(cue.to(self.patterns.dtype), self.radius)
        observe(observer, 0, initial, float(self.energy(initial).detach()))
        trajectory = converged_rollout(
            self.step,
            initial,
            max_steps=max_sweeps,
            tolerance=self.convergence_tolerance,
        )
        energies = [float(self.energy(state).detach()) for state in trajectory.states]
        for step in range(1, trajectory.steps + 1):
            observe(
                observer,
                step,
                trajectory.states[step],
                energies[step],
                {"residual_inf": float(trajectory.residuals[step - 1].detach())},
            )
        status = "fixed" if trajectory.status == "converged" else "max_steps"
        return RetrievalResult(
            final_state=trajectory.states[-1],
            status=status,
            sweeps=trajectory.steps,
            state_updates=trajectory.steps * self.N,
            retrieval_flops=trajectory.steps * (4 * self.P * self.N + self.P),
            energy_trace=energies,
            diagnostics={
                "degree": self.degree,
                "radius": self.radius,
                "output_kind": "continuous",
                "fixed_point_eligible": False,
                "retrieval_mode": "synchronous_spherical_projection",
                "stopping_rule": "infinity_norm_residual",
                "convergence_tolerance": self.convergence_tolerance,
                "final_residual_inf": float(trajectory.residuals[-1].detach()),
            },
        )

    def resource_summary(self) -> dict[str, int | str]:
        self._require_fitted()
        return {
            "model_config": f"spherical_power_degree_{self.degree}_synchronous",
            "parameter_count": self.P * self.N,
            "storage_bytes": self.patterns.numel() * self.patterns.element_size(),
        }
