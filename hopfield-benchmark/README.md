# Hopfield Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/hopfield-benchmark/hopfield_benchmark_phase1_colab.ipynb)

这个目录按“任务”而不是按“论文”组织 Hopfield 模型比较。

论文复现 notebook 继续保留，负责证明某个模型实现忠于原文；这里的统一基准负责让不同模型在相同数据、线索、随机种子、停止规则和资源预算下比较。

统一实验继续使用现有 notebook 的组件式管线：

`a 记忆输入 → b 模型存储 → c 检索线索 → d 检索动力学 → e 测量 → f 同图比较`

只要 `a/c/e/f` 保持不变，替换 `b/d` 就是在同一任务中替换模型。因此，同一张图上的每条线只代表一个模型，不会混入不同数据或不同噪声协议。

当前状态：协议和 Phase 1 代码骨架已经完成，但尚未在 Colab 运行实验。完整规则见 [PROTOCOL.md](./PROTOCOL.md)，模型资格与证据边界见 [MODEL_REGISTRY.md](./MODEL_REGISTRY.md)。

## Phase 1 组件

- [`components.py`](./src/hopfield_benchmark/components.py)：共享的记忆、线索和测量组件。
- [`models.py`](./src/hopfield_benchmark/models.py)：Classical Hopfield 与 Polynomial DAM adapter。
- [`runner.py`](./src/hopfield_benchmark/runner.py)：生成一次案例，再配对交给所有模型。
- [`plotting.py`](./src/hopfield_benchmark/plotting.py)：同图资格检查后的统一曲线。
- [`hopfield_benchmark_phase1_colab.ipynb`](./hopfield_benchmark_phase1_colab.ipynb)：Colab 入口；实验图之后紧跟读图标准和证据边界。

本地没有执行数值实验。Notebook 会在 Colab 中读取这个目录的代码并生成原始结果表、噪声恢复曲线、负载曲线、容量缩放图和质量—计算量图。

## 计划中的结果图

1. 固定存储量下：横轴为噪声强度，纵轴为召回率或 overlap，每条线一个模型。
2. 固定网络规模下：横轴为存储模式数 `P`，纵轴为召回率，每条线一个模型。
3. 容量缩放：横轴为 `N`，纵轴为经验临界容量 `P_c(N)`；`P_c` 使用对数纵轴。
4. 动力学：相同初态下比较能量、误差和收敛状态；没有对应能量的模型显示 `N/A`，不套用经典二次能量。
5. 资源 Pareto 图：横轴为参数、显存或 FLOPs，纵轴为容量或鲁棒召回率。

任何跨模型主图都必须满足 [PROTOCOL.md](./PROTOCOL.md) 中的“同图资格检查”。
