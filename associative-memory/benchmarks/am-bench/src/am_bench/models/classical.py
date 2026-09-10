"""b/d：ClassicalHopfield；提取自 phase-one Notebook。"""
from typing import Any
import torch
from ..tasks import make_generator
from .base import RetrievalResult, Observer, observe, binary_state

class ClassicalHopfield:
    model_id = "classical_hebb"

    def fit(self, patterns: torch.Tensor) -> "ClassicalHopfield":
        # [输入] patterns:(P,N)；转 float64 后再做 P 项累加
        binary = patterns.to(torch.float64)
        if binary.ndim != 2:
            raise ValueError("patterns must have shape [P, N]")
        self.N = int(binary.shape[1])
        # [存储] W=(1/N)XᵀX，(N,P)@(P,N)→(N,N)
        self.weights = binary.T @ binary / self.N
        # [约束] W_ii=0，禁止神经元把自身旧值当作局部场证据
        self.weights.fill_diagonal_(0.0)
        return self

    def energy(self, state: torch.Tensor) -> float:
        # [观测·数据] E=-1/2 sᵀWs；只作为异步下降检查，不参与判停
        return float((-0.5 * state @ self.weights @ state).item())

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
        # [输入] 复制 cue；后续原地更新不会污染共享线索
        state = binary_state(cue)
        # [实验控制] 同一 trial 的迭代模型共享 update_seed，消除更新顺序差异
        generator = make_generator(update_seed, state.device)
        # [观测·数据] 先记录 t=0，图中第一点就是损坏线索
        energies = [self.energy(state)]
        errors: list[float] = []
        observe(observer, 0, state, energies[-1])
        updates = 0
        first_sweep_unchanged = False
        for sweep in range(1, max_sweeps + 1):
            changed = False
            # [实验控制] 每轮随机排列全部 N 个坐标，避免固定编号顺序偏置
            order = torch.randperm(self.N, generator=generator, device=state.device)
            for index in order.tolist():
                # [中介变量] weights[index]:(N,)·state:(N,)→该神经元净局部场
                local_field = float(torch.dot(self.weights[index], state).item())
                # [判断] 正场取 +1、负场取 -1、恰好为零保持旧值
                new_value = (
                    1.0
                    if local_field > 0.0
                    else -1.0
                    if local_field < 0.0
                    else state[index].item()
                )
                if new_value != state[index].item():
                    # [更新] 立即写回；后续坐标会看到最新状态，这就是异步更新
                    state[index] = new_value
                    changed = True
                updates += 1
            energies.append(self.energy(state))
            observe(observer, sweep, state, energies[-1])
            # [观测·固定点] 干净记忆第一整轮完全不变，才通过 U1 判据
            if sweep == 1:
                first_sweep_unchanged = not changed
            # [判断] 一整轮零翻转说明到达离散不动点，可以提前停止
            if not changed:
                return RetrievalResult(
                    state.to(torch.int8),
                    "fixed",
                    sweep,
                    updates,
                    updates * (2 * self.N - 1),
                    energies,
                    errors,
                    {"first_sweep_unchanged": first_sweep_unchanged},
                )
        return RetrievalResult(
            state.to(torch.int8),
            "max_steps",
            max_sweeps,
            updates,
            updates * (2 * self.N - 1),
            energies,
            errors,
            {"first_sweep_unchanged": first_sweep_unchanged},
        )

    def resource_summary(self) -> dict[str, int | str]:
        return {
            "model_config": "hebbian_zero_diagonal_async",
            # [观测·资源] 对称矩阵只计独立非对角权重；storage_bytes 仍按真实密集张量计
            "parameter_count": self.N * (self.N - 1) // 2,
            "storage_bytes": self.weights.numel() * self.weights.element_size(),
        }
