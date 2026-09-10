"""b/d：SimplicialR12；提取自 phase-one Notebook。"""
from typing import Any
import torch
from ..tasks import make_generator
from .base import RetrievalResult, Observer, observe, binary_state

class SimplicialR12:
    def __init__(
        self, triangle_fraction: float = 0.5, structure_seed: int = 31415,
        budget_type: str = "parameter",
    ) -> None:
        if not 0.0 <= triangle_fraction <= 1.0:
            raise ValueError("triangle_fraction must lie in [0, 1]")
        # [约束] H3 只允许三种预注册资源反事实，不能运行后挑预算
        if budget_type not in {"parameter", "storage", "compute"}:
            raise ValueError("unknown simplicial budget_type")
        self.triangle_fraction = float(triangle_fraction)
        self.structure_seed = int(structure_seed)
        self.budget_type = budget_type
        self.model_id = f"simplicial_r12_t{int(100 * triangle_fraction):02d}"

    def fit(self, patterns: torch.Tensor) -> "SimplicialR12":
        binary = patterns.to(torch.float64)
        self.P, self.N = map(int, binary.shape)
        # [实验控制] pairwise R12 的 C(N,2) 个连接作为三种预算基准
        pairwise_budget = self.N * (self.N - 1) // 2
        # [实验控制] matched-parameter：各阶连接权重的总个数相同
        if self.budget_type == "parameter":
            connection_count = pairwise_budget
        # [实验控制] matched-storage：同时计算 int64 顶点索引与 float64 权重
        elif self.budget_type == "storage":
            bytes_per_connection = 24.0 + 8.0 * self.triangle_fraction
            connection_count = int((24 * pairwise_budget) // bytes_per_connection)
        # [实验控制] matched-compute：用每轮需访问的顶点 incidence 数作上限
        else:
            incidences_per_connection = 2.0 + self.triangle_fraction
            connection_count = int((2 * pairwise_budget) // incidences_per_connection)
        # [中介变量] 先按比例分配，再用下面循环修正整数取整越界
        triangle_count = int(round(self.triangle_fraction * connection_count))
        edge_count = connection_count - triangle_count
        if self.budget_type == "storage":
            while 24 * edge_count + 32 * triangle_count > 24 * pairwise_budget:
                if triangle_count:
                    triangle_count -= 1
                else:
                    edge_count -= 1
        if self.budget_type == "compute":
            while 2 * edge_count + 3 * triangle_count > 2 * pairwise_budget:
                if triangle_count:
                    triangle_count -= 1
                else:
                    edge_count -= 1
        # [实验控制] structure_seed 只控制拓扑；数据种子和线索种子保持不变
        generator = make_generator(self.structure_seed)
        # [中介变量] 枚举候选边后无放回采样，避免重复连接
        all_edges = torch.combinations(torch.arange(self.N), r=2)
        edge_order = torch.randperm(len(all_edges), generator=generator)[:edge_count]
        self.edges = all_edges[edge_order]
        # [中介变量] triangles:(T,3)，每行是一条三元相互作用
        all_triangles = torch.combinations(torch.arange(self.N), r=3)
        triangle_order = torch.randperm(len(all_triangles), generator=generator)[:triangle_count]
        self.triangles = all_triangles[triangle_order]
        # [存储] w_ij=(1/N)Σ_mu ξ_i^mu ξ_j^mu，一列对应一条采样边
        self.edge_weights = (
            binary[:, self.edges[:, 0]] * binary[:, self.edges[:, 1]]
        ).sum(dim=0) / self.N
        # [存储] w_ijk=(1/N)Σ_mu ξ_i^mu ξ_j^mu ξ_k^mu
        self.triangle_weights = (
            binary[:, self.triangles[:, 0]]
            * binary[:, self.triangles[:, 1]]
            * binary[:, self.triangles[:, 2]]
        ).sum(dim=0) / self.N
        # [中介变量] 预建每个神经元的 incident 列表，检索时不再全表扫描
        self.edge_incident = []
        self.triangle_incident = []
        for neuron in range(self.N):
            # [中介变量] 找出包含当前 neuron 的全部边及其另一个端点
            edge_mask = torch.any(self.edges == neuron, dim=1)
            incident_edges = self.edges[edge_mask]
            neighbors = torch.where(
                incident_edges[:, 0] == neuron,
                incident_edges[:, 1], incident_edges[:, 0],
            )
            self.edge_incident.append((neighbors, self.edge_weights[edge_mask]))
            # [中介变量] 对三元单形保存另外两个端点，供局部场计算乘积
            triangle_mask = torch.any(self.triangles == neuron, dim=1)
            incident_triangles = self.triangles[triangle_mask]
            others = (
                torch.stack([row[row != neuron] for row in incident_triangles])
                if len(incident_triangles)
                # [失败边界] 某神经元没有三元邻居时保留合法 (0,2) 张量
                else torch.empty((0, 2), dtype=torch.long)
            )
            self.triangle_incident.append((others, self.triangle_weights[triangle_mask]))
        return self

    def energy(self, state: torch.Tensor) -> float:
        # [观测·数据] 每条边贡献 w_ij s_i s_j
        edge_term = self.edge_weights * state[self.edges].prod(dim=1)
        # [观测·数据] 每个三元单形贡献 w_ijk s_i s_j s_k
        triangle_term = self.triangle_weights * state[self.triangles].prod(dim=1)
        return float(-(edge_term.sum() + triangle_term.sum()).item())

    def retrieve(
        self, cue: torch.Tensor, *, observer: Observer | None = None,
        update_seed: int, max_sweeps: int,
    ) -> RetrievalResult:
        if max_sweeps <= 0:
            raise ValueError("max_sweeps must be positive")
        state = binary_state(cue)
        generator = make_generator(update_seed)
        energies = [self.energy(state)]
        errors: list[float] = []
        observe(observer, 0, state, energies[-1])
        updates = 0
        first_sweep_unchanged = False
        for sweep in range(1, max_sweeps + 1):
            changed = False
            for neuron in torch.randperm(self.N, generator=generator).tolist():
                # [中介变量] 当前神经元的一阶邻接与二阶邻接已在 fit 阶段缓存
                neighbors, edge_weights = self.edge_incident[neuron]
                others, triangle_weights = self.triangle_incident[neuron]
                # [更新] 边给线性票数，三元单形给另外两点乘积后的票数
                local_field = torch.dot(edge_weights, state[neighbors])
                local_field += torch.sum(
                    triangle_weights * state[others].prod(dim=1)
                )
                old_value = state[neuron].item()
                # [判断] 论文 Θ 约定在零场取 +1；与 Classical 的零场保持不同并明确披露
                new_value = 1.0 if local_field >= 0.0 else -1.0
                if new_value != old_value:
                    state[neuron] = new_value
                    changed = True
                updates += 1
            energies.append(self.energy(state))
            observe(observer, sweep, state, energies[-1])
            if sweep == 1:
                first_sweep_unchanged = not changed
            diagnostics = {
                "first_sweep_unchanged": first_sweep_unchanged,
                "structure_seed": self.structure_seed,
                "triangle_fraction": self.triangle_fraction,
                # [证据边界] 这里是可检查能量下降的异步扩展，不冒充论文同步实验
                "paper_native_update": False,
            }
            if not changed:
                break
        status = "fixed" if not changed else "max_steps"
        mean_incidence = (2 * len(self.edges) + 3 * len(self.triangles)) / self.N
        return RetrievalResult(
            state.to(torch.int8), status, sweep, updates,
            int(updates * (3 * mean_incidence + 1)),
            energies, errors, diagnostics,
        )

    def resource_summary(self) -> dict[str, int | str]:
        # [观测·资源] 稀疏单形除权重外还必须存顶点索引，不能只报权重数
        index_bytes = (self.edges.numel() + self.triangles.numel()) * 8
        weight_bytes = (self.edge_weights.numel() + self.triangle_weights.numel()) * 8
        return {
            "model_config": (
                f"R12_triangle_fraction_{self.triangle_fraction:.2f}_"
                f"{self.budget_type}_budget_seed_{self.structure_seed}_"
                "async_extension"
            ),
            "parameter_count": len(self.edges) + len(self.triangles),
            "storage_bytes": index_bytes + weight_bytes,
        }
