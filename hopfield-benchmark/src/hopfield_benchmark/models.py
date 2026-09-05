"""Component b/d adapters for the Phase 1 binary benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .components import RetrievalResult


def _binary_state(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 1:
        raise ValueError("state must be one-dimensional")
    values = torch.unique(state.to(torch.int8)).tolist()
    if not set(values).issubset({-1, 1}):
        raise ValueError("state must use {-1, +1} encoding")
    return state.detach().clone().to(torch.float64)


class ModelAdapter(ABC):
    """The only model-specific boundary in the shared binary pipeline."""

    model_id: str

    @abstractmethod
    def fit(self, patterns: torch.Tensor) -> "ModelAdapter":
        """Component b: store a shared pattern set."""

    @abstractmethod
    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        target: torch.Tensor,
        update_seed: int,
        max_sweeps: int,
    ) -> RetrievalResult:
        """Component d: retrieve from a shared cue."""

    @abstractmethod
    def resource_summary(self) -> dict[str, int | float | str]:
        """Return model storage and configuration metadata."""


class ClassicalHopfield(ModelAdapter):
    """Hebbian pairwise Hopfield network with asynchronous updates."""

    model_id = "classical_hebb"

    def __init__(self) -> None:
        self.weights: torch.Tensor | None = None
        self.N = 0

    def fit(self, patterns: torch.Tensor) -> "ClassicalHopfield":
        binary = patterns.to(torch.float64)
        if binary.ndim != 2:
            raise ValueError("patterns must have shape [P, N]")
        self.N = int(binary.shape[1])
        self.weights = binary.T @ binary / self.N
        self.weights.fill_diagonal_(0.0)
        return self

    def _energy(self, state: torch.Tensor) -> float:
        if self.weights is None:
            raise RuntimeError("fit must be called before retrieve")
        return float((-0.5 * state @ self.weights @ state).item())

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        target: torch.Tensor,
        update_seed: int,
        max_sweeps: int,
    ) -> RetrievalResult:
        if self.weights is None:
            raise RuntimeError("fit must be called before retrieve")
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")

        state = _binary_state(cue)
        target_f = target.to(torch.float64)
        generator = torch.Generator(device=state.device)
        generator.manual_seed(int(update_seed))
        energy_trace = [self._energy(state)]
        error_trace = [float(torch.mean((state != target_f).to(torch.float64)).item())]
        state_updates = 0
        first_sweep_unchanged = False

        for sweep in range(1, max_sweeps + 1):
            changed = False
            order = torch.randperm(self.N, generator=generator, device=state.device)
            for index in order.tolist():
                field = float(torch.dot(self.weights[index], state).item())
                new_value = 1.0 if field > 0.0 else -1.0 if field < 0.0 else state[index].item()
                if new_value != state[index].item():
                    state[index] = new_value
                    changed = True
                state_updates += 1

            energy_trace.append(self._energy(state))
            error_trace.append(
                float(torch.mean((state != target_f).to(torch.float64)).item())
            )
            if sweep == 1:
                first_sweep_unchanged = not changed
            if not changed:
                return RetrievalResult(
                    final_state=state.to(torch.int8),
                    status="fixed",
                    sweeps=sweep,
                    state_updates=state_updates,
                    retrieval_flops=state_updates * (2 * self.N - 1),
                    energy_trace=energy_trace,
                    error_trace=error_trace,
                    diagnostics={"first_sweep_unchanged": first_sweep_unchanged},
                )

        return RetrievalResult(
            final_state=state.to(torch.int8),
            status="max_steps",
            sweeps=max_sweeps,
            state_updates=state_updates,
            retrieval_flops=state_updates * (2 * self.N - 1),
            energy_trace=energy_trace,
            error_trace=error_trace,
            diagnostics={"first_sweep_unchanged": first_sweep_unchanged},
        )

    def resource_summary(self) -> dict[str, int | float | str]:
        if self.weights is None:
            raise RuntimeError("fit must be called before resource_summary")
        return {
            "model_config": "hebbian_zero_diagonal_async",
            "parameter_count": self.N * (self.N - 1) // 2,
            "storage_bytes": self.weights.numel() * self.weights.element_size(),
        }


class PolynomialDAM(ModelAdapter):
    """Power-energy dense associative memory with exact coordinate descent."""

    def __init__(self, degree: int = 3) -> None:
        if degree < 2:
            raise ValueError("degree must be at least 2")
        self.degree = int(degree)
        self.model_id = f"polynomial_dam_d{self.degree}"
        self.patterns: torch.Tensor | None = None
        self.P = 0
        self.N = 0

    def fit(self, patterns: torch.Tensor) -> "PolynomialDAM":
        binary = patterns.to(torch.float64)
        if binary.ndim != 2:
            raise ValueError("patterns must have shape [P, N]")
        self.patterns = binary.detach().clone()
        self.P, self.N = map(int, binary.shape)
        return self

    def _energy_from_overlaps(self, overlaps: torch.Tensor) -> float:
        return float(-torch.sum(overlaps.pow(self.degree)).item())

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        target: torch.Tensor,
        update_seed: int,
        max_sweeps: int,
    ) -> RetrievalResult:
        if self.patterns is None:
            raise RuntimeError("fit must be called before retrieve")
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")

        state = _binary_state(cue)
        target_f = target.to(torch.float64)
        overlaps = self.patterns @ state / self.N
        generator = torch.Generator(device=state.device)
        generator.manual_seed(int(update_seed))
        energy_trace = [self._energy_from_overlaps(overlaps)]
        error_trace = [float(torch.mean((state != target_f).to(torch.float64)).item())]
        state_updates = 0
        first_sweep_unchanged = False

        for sweep in range(1, max_sweeps + 1):
            changed = False
            order = torch.randperm(self.N, generator=generator, device=state.device)
            for index in order.tolist():
                coordinate = self.patterns[:, index] / self.N
                base = overlaps - coordinate * state[index]
                score_plus = torch.sum((base + coordinate).pow(self.degree))
                score_minus = torch.sum((base - coordinate).pow(self.degree))
                old_value = state[index].item()
                if score_plus > score_minus:
                    new_value = 1.0
                elif score_minus > score_plus:
                    new_value = -1.0
                else:
                    new_value = old_value
                if new_value != old_value:
                    state[index] = new_value
                    overlaps = overlaps + coordinate * (new_value - old_value)
                    changed = True
                state_updates += 1

            energy_trace.append(self._energy_from_overlaps(overlaps))
            error_trace.append(
                float(torch.mean((state != target_f).to(torch.float64)).item())
            )
            if sweep == 1:
                first_sweep_unchanged = not changed
            if not changed:
                return RetrievalResult(
                    final_state=state.to(torch.int8),
                    status="fixed",
                    sweeps=sweep,
                    state_updates=state_updates,
                    retrieval_flops=state_updates * (8 * self.P + 2),
                    energy_trace=energy_trace,
                    error_trace=error_trace,
                    diagnostics={
                        "degree": self.degree,
                        "first_sweep_unchanged": first_sweep_unchanged,
                    },
                )

        return RetrievalResult(
            final_state=state.to(torch.int8),
            status="max_steps",
            sweeps=max_sweeps,
            state_updates=state_updates,
            retrieval_flops=state_updates * (8 * self.P + 2),
            energy_trace=energy_trace,
            error_trace=error_trace,
            diagnostics={
                "degree": self.degree,
                "first_sweep_unchanged": first_sweep_unchanged,
            },
        )

    def resource_summary(self) -> dict[str, int | float | str]:
        if self.patterns is None:
            raise RuntimeError("fit must be called before resource_summary")
        return {
            "model_config": f"power_energy_degree_{self.degree}_async",
            "parameter_count": self.P * self.N,
            "storage_bytes": self.patterns.numel() * self.patterns.element_size(),
        }
