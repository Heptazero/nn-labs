"""a/c：共享数据和查询，不包含模型更新。"""
from dataclasses import dataclass
import torch

# [数据契约] a 组件的输出；模式张量与身份、种子、编码一起传给所有模型
@dataclass(frozen=True)
class MemorySet:
    patterns: torch.Tensor
    pattern_ids: tuple[int, ...]
    dataset_id: str
    data_seed: int
    encoding: str = "binary_pm1"

    @property
    # [观测·数据] patterns:(P,N)，第 0 维就是实际存储模式数
    def P(self) -> int:
        return int(self.patterns.shape[0])

    @property
    # [观测·数据] patterns:(P,N)，第 1 维就是状态空间维数
    def N(self) -> int:
        return int(self.patterns.shape[1])


# [实验控制] 随机数生成器显式局部化；函数不会改写 PyTorch 全局种子
def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def a1_make_independent_binary(
    N: int,
    P: int,
    data_seed: int,
    *,
    device: torch.device | str = "cpu",
) -> MemorySet:
    # [约束] 空模式集没有可定义的存储或 Top-1 检索任务
    if N <= 0 or P <= 0:
        raise ValueError("N and P must be positive")
    # [输入] bits:(P,N) 独立 Bernoulli(1/2)，先用 int8 节省内存
    bits = torch.randint(
        0,
        2,
        (P, N),
        generator=make_generator(data_seed, device),
        device=device,
        dtype=torch.int8,
    )
    return MemorySet(
        # [更新] {0,1} → {-1,+1}，得到 Hopfield 二值自旋编码
        patterns=bits.mul(2).sub(1),
        pattern_ids=tuple(range(P)),
        dataset_id="independent_binary",
        data_seed=int(data_seed),
    )


def c1_make_hamming_cue(
    target: torch.Tensor,
    corruption_level: float,
    cue_seed: int,
) -> torch.Tensor:
    # [约束] 一条 cue 只能对应一个一维目标状态
    if target.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if not 0.0 <= corruption_level <= 1.0:
        raise ValueError("corruption_level must lie in [0, 1]")
    # [输入] 复制目标，噪声构造不能污染记忆库里的原样本
    cue = target.detach().clone().to(torch.int8)
    # [实验控制] 精确翻转 round(rho*N) 位，不使用期望翻转率近似
    flip_count = int(round(corruption_level * cue.numel()))
    if flip_count:
        # [中介变量] indices:(flip_count,) 无放回抽样，避免同一位翻两次
        indices = torch.randperm(
            cue.numel(),
            generator=make_generator(cue_seed, cue.device),
            device=cue.device,
        )[:flip_count]
        # [更新] 原地取反后，cue 与 target 的汉明距离恰好等于 flip_count
        cue[indices] *= -1
    return cue


# [输入] U6 随机初态；与任何已存记忆独立，用来探测吸引子体积
def c2_make_random_state(N: int, cue_seed: int) -> torch.Tensor:
    if N <= 0:
        raise ValueError("N must be positive")
    bits = torch.randint(
        0, 2, (N,), generator=make_generator(cue_seed), dtype=torch.int8
    )
    return bits.mul(2).sub(1)