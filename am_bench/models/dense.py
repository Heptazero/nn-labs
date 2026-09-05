"""b/d：PolynomialDAM, ExponentialDAM；提取自 phase-one Notebook。"""
from typing import Any
import torch
from ..tasks import make_generator
from .base import RetrievalResult, Observer, observe, binary_state

class PolynomialDAM:
    def __init__(self, degree: int = 3) -> None:
        # [约束] d<2 不再是这里要比较的高阶 DAM 条件
        if degree < 2:
            raise ValueError("degree must be at least 2")
        self.degree = int(degree)
        self.model_id = f"polynomial_dam_d{self.degree}"

    def fit(self, patterns: torch.Tensor) -> "PolynomialDAM":
        binary = patterns.to(torch.float64)
        if binary.ndim != 2:
            raise ValueError("patterns must have shape [P, N]")
        # [存储] 不展开高阶权重张量，直接保存 X:(P,N)
        self.patterns = binary.detach().clone()
        self.P, self.N = map(int, binary.shape)
        return self

    def energy(self, overlaps: torch.Tensor) -> float:
        # [观测·数据] E=-Σ_mu m_mu^d；这里只比较同一模型轨迹内的差值
        return float(-torch.sum(overlaps.pow(self.degree)).item())

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        observer: Observer | None = None,
        update_seed: int,
        max_sweeps: int,
    ) -> RetrievalResult:
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        state = binary_state(cue)
        # [中介变量] overlaps:(P,)，一次计算后靠单坐标增量更新
        # [中介变量] Polynomial 使用按 N 归一化的 overlap；指数模型保留原始内积
        overlaps = self.patterns @ state / self.N
        generator = make_generator(update_seed, state.device)
        energies = [self.energy(overlaps)]
        errors: list[float] = []
        observe(observer, 0, state, energies[-1])
        updates = 0
        first_sweep_unchanged = False
        for sweep in range(1, max_sweeps + 1):
            changed = False
            order = torch.randperm(self.N, generator=generator, device=state.device)
            for index in order.tolist():
                # [中介变量] coordinate:(P,) 是翻动 s_i 对所有 overlap 的贡献
                coordinate = self.patterns[:, index] / self.N
                # [中介变量] 先扣掉 s_i 旧贡献，再分别试算 +1 与 -1
                base = overlaps - coordinate * state[index]
                # [判断] 最大化 Σm^d 等价于最小化前面定义的负能量
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
                    # [更新] 只更新受 s_i 改变的 overlap，避免每次重算 X@state
                    overlaps += coordinate * (new_value - old_value)
                    changed = True
                updates += 1
            energies.append(self.energy(overlaps))
            observe(observer, sweep, state, energies[-1])
            if sweep == 1:
                first_sweep_unchanged = not changed
            diagnostics = {
                "degree": self.degree,
                "first_sweep_unchanged": first_sweep_unchanged,
            }
            if not changed:
                return RetrievalResult(
                    state.to(torch.int8),
                    "fixed",
                    sweep,
                    updates,
                    updates * (8 * self.P + 2),
                    energies,
                    errors,
                    diagnostics,
                )
        return RetrievalResult(
            state.to(torch.int8),
            "max_steps",
            max_sweeps,
            updates,
            updates * (8 * self.P + 2),
            energies,
            errors,
            diagnostics,
        )

    def resource_summary(self) -> dict[str, int | str]:
        return {
            "model_config": f"power_energy_degree_{self.degree}_async",
            "parameter_count": self.P * self.N,
            "storage_bytes": self.patterns.numel() * self.patterns.element_size(),
        }


class ExponentialDAM:
    model_id = "exponential_dam"

    def fit(self, patterns: torch.Tensor) -> "ExponentialDAM":
        self.patterns = patterns.to(torch.float64).detach().clone()
        self.P, self.N = map(int, self.patterns.shape)
        return self

    # [数值稳定] 用 -logsumexp(m) 代替直接计算 -Σexp(m)，排序与下降方向不变
    def log_energy(self, overlaps: torch.Tensor) -> float:
        return float(-torch.logsumexp(overlaps, dim=0).item())

    def retrieve(
        self, cue: torch.Tensor, *, observer: Observer | None = None,
        update_seed: int, max_sweeps: int,
    ) -> RetrievalResult:
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        state = binary_state(cue)
        overlaps = self.patterns @ state
        generator = make_generator(update_seed, state.device)
        energies = [self.log_energy(overlaps)]
        errors: list[float] = []
        observe(observer, 0, state, energies[-1])
        updates = 0
        first_sweep_unchanged = False
        for sweep in range(1, max_sweeps + 1):
            changed = False
            order = torch.randperm(self.N, generator=generator, device=state.device)
            for index in order.tolist():
                coordinate = self.patterns[:, index]
                base = overlaps - coordinate * state[index]
                # [判断] 分别计算 s_i=+1/-1 的 log-sum-exp，选择能量更低者
                plus = torch.logsumexp(base + coordinate, dim=0)
                minus = torch.logsumexp(base - coordinate, dim=0)
                old_value = state[index].item()
                new_value = 1.0 if plus > minus else -1.0 if minus > plus else old_value
                if new_value != old_value:
                    state[index] = new_value
                    overlaps += coordinate * (new_value - old_value)
                    changed = True
                updates += 1
            energies.append(self.log_energy(overlaps))
            observe(observer, sweep, state, energies[-1])
            if sweep == 1:
                first_sweep_unchanged = not changed
            diagnostics = {
                "first_sweep_unchanged": first_sweep_unchanged,
                # [溯源] 明记轨迹保存的是数值稳定代理，避免读图时当成原始指数和
                "energy_note": "negative log-sum-exp; monotone proxy",
            }
            if not changed:
                return RetrievalResult(
                    state.to(torch.int8), "fixed", sweep, updates,
                    updates * (8 * self.P + 2), energies, errors, diagnostics,
                )
        return RetrievalResult(
            state.to(torch.int8), "max_steps", max_sweeps, updates,
            updates * (8 * self.P + 2), energies, errors, diagnostics,
        )

    def resource_summary(self) -> dict[str, int | str]:
        return {
            "model_config": "exp_energy_async",
            "parameter_count": self.P * self.N,
            "storage_bytes": self.patterns.numel() * self.patterns.element_size(),
        }
