"""b/d：PSHN；提取自 phase-one Notebook。"""
from typing import Any
import torch
from ..tasks import make_generator
from .base import RetrievalResult, Observer, observe, binary_state

class PSHN:
    def __init__(self, groups: int = 8) -> None:
        self.groups = int(groups)
        self.model_id = f"pshn_k{self.groups}"

    def fit(self, patterns: torch.Tensor) -> "PSHN":
        self.patterns = patterns.to(torch.float64).detach().clone()
        self.P, self.N = map(int, self.patterns.shape)
        # [约束] 每组必须等长，因此 k 必须整除 N
        if self.N % self.groups:
            raise ValueError("N must be divisible by the number of PSHN groups")
        self.group_size = self.N // self.groups
        # [存储] X:(P,N)→(P,k,N/k)，只改变视图，不复制出高阶张量
        self.grouped = self.patterns.reshape(self.P, self.groups, self.group_size)
        return self

    def retrieve(
        self, cue: torch.Tensor, *, observer: Observer | None = None,
        update_seed: int, max_sweeps: int,
    ) -> RetrievalResult:
        state = binary_state(cue)
        query = state.reshape(self.groups, self.group_size)
        # [中介变量] correlations:(P,k)，每条记忆在每一组上的局部相关度
        correlations = torch.einsum("kg,Pkg->Pk", query, self.grouped)
        # [中介变量] 更新第 g 组时，只乘其余 k-1 组，避免把待更新组自我代入
        products_without_group = []
        for group in range(self.groups):
            keep = torch.arange(self.groups) != group
            products_without_group.append(correlations[:, keep].prod(dim=1))
        coefficients = torch.stack(products_without_group, dim=1)
        # [更新] 记忆系数 C:(P,k) 与 X:(P,k,N/k) 汇总成各组局部场
        fields = torch.einsum("Pk,Pkg->kg", coefficients, self.grouped)
        # [判断] 一次性对所有坐标取符号；这是 one-step，不伪装成收敛轮数
        output = torch.sign(fields).reshape(self.N)
        # [边界约定] 零场保持 cue 原值，避免 torch.sign 产生不合法的 0 自旋
        ties = output == 0
        output[ties] = state[ties]
        errors: list[float] = []
        observe(observer, 0, state)
        observe(observer, 1, output)
        return RetrievalResult(
            output.to(torch.int8), "one_step", 1, self.N,
            2 * self.P * self.N + self.P * self.groups,
            [], errors,
            {
                "first_sweep_unchanged": bool(torch.equal(output, state)),
                "groups": self.groups,
                "output_kind": "binary",
            },
        )

    def resource_summary(self) -> dict[str, int | str]:
        return {
            "model_config": f"product_of_sums_k_{self.groups}_one_step",
            "parameter_count": self.P * self.N,
            "storage_bytes": self.patterns.numel() * self.patterns.element_size(),
        }
