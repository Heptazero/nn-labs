"""b/d：ContinuousModernHopfield；提取自 phase-one Notebook。"""
from typing import Any
import torch
from ..tasks import make_generator
from .base import RetrievalResult, Observer, observe, binary_state

class ContinuousModernHopfield:
    model_id = "continuous_modern"

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = float(beta)

    def fit(self, patterns: torch.Tensor) -> "ContinuousModernHopfield":
        self.patterns = patterns.to(torch.float64).detach().clone()
        self.P, self.N = map(int, self.patterns.shape)
        return self

    def retrieve(
        self, cue: torch.Tensor, *, observer: Observer | None = None,
        update_seed: int, max_sweeps: int,
    ) -> RetrievalResult:
        # [输入] query:(N,)；连续读出允许终态落在 [-1,+1]^N 内部
        query = cue.to(torch.float64)
        # [中介变量] attention:(P,)，beta 控制记忆竞争的集中程度
        output, attention = self.readout(query)
        errors: list[float] = []
        # [观测·机制] 注意力熵量化读出集中度，clamp 只防 log(0)
        entropy = -torch.sum(attention * torch.log(attention.clamp_min(1e-300)))
        observe(observer, 0, query)
        observe(observer, 1, output, diagnostics={"attention_entropy": float(entropy.item())})
        return RetrievalResult(
            output, "one_step", 1, self.N,
            4 * self.P * self.N, [], errors,
            {
                "first_sweep_unchanged": False,
                "fixed_point_eligible": False,
                "output_kind": "continuous",
                "attention_entropy": float(entropy.item()),
                "beta": self.beta,
            },
        )

    def readout(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """单步和迭代版共用同一更新，避免复制模型公式。"""
        attention = torch.softmax(self.beta * (self.patterns @ query), dim=0)
        return attention @ self.patterns, attention

    def resource_summary(self) -> dict[str, int | str]:
        return {
            "model_config": f"softmax_beta_{self.beta:g}_one_step",
            "parameter_count": self.P * self.N,
            "storage_bytes": self.patterns.numel() * self.patterns.element_size(),
        }


class IterativeModernHopfield(ContinuousModernHopfield):
    """固定记忆/温度，每轮把连续输出反馈为查询；不重新二值化。"""

    model_id = "continuous_modern_iterative"

    def retrieve(self, cue: torch.Tensor, *, update_seed: int, max_sweeps: int,
                 observer: Observer | None = None) -> RetrievalResult:
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        state = cue.detach().clone().to(torch.float64)
        observe(observer, 0, state)
        status = "max_steps"
        for step in range(1, max_sweeps + 1):
            output, weights = self.readout(state)
            entropy = float(-(weights * weights.clamp_min(1e-300).log()).sum().item())
            # 权重是从上一轮状态算出的，用于产生本轮输出，不是正确率。
            observe(observer, step, output, diagnostics={
                "attention_entropy": entropy,
                "max_weight": float(weights.max().item()),
            })
            unchanged = torch.equal(output, state)
            state = output
            if unchanged:
                status = "fixed"
                break
        return RetrievalResult(
            state, status, step, step * self.N, 4 * step * self.P * self.N,
            diagnostics={"output_kind": "continuous", "beta": self.beta,
                         "retrieval_mode": "iterative_continuous",
                         "stopping_rule": "exact_unchanged_or_budget"},
        )

    def resource_summary(self) -> dict[str, int | str]:
        return {**super().resource_summary(),
                "model_config": f"softmax_beta_{self.beta:g}_iterative_continuous"}
