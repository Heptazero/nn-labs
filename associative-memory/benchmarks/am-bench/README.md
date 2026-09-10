# Associative Memory Benchmark

这里按共同任务比较联想记忆模型，不代表某一篇论文。

```text
am-bench/
├── notebooks/      # Colab 入口
├── src/am_bench/   # 可导入的共享 Python 包
├── tests/          # 静态与数值测试
├── PROTOCOL.md     # 比较协议
└── QUESTIONS.md    # 问题清单
```

- [Q04 retrieval dynamics（Colab）](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/benchmarks/am-bench/notebooks/retrieval_dynamics.ipynb)
- [Query-probe reliability（Colab）](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/benchmarks/am-bench/notebooks/retrieval_reliability.ipynb)
- [Phase-one benchmark（Colab）](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/benchmarks/am-bench/notebooks/hopfield_benchmark_phase1_colab.ipynb)

Notebook 会从 GitHub 获取固定版本的 `am_bench`。结果默认写入 Colab 临时目录 `/content/nn-labs-results/`，不要求 Google Drive。

## 公共几何组件

- `am_bench.state_spaces`：球面投影、切空间投影、球面采样、测地移动与回缩。
- `am_bench.trajectory`：固定长度的可微轨迹，以及带收敛判据的单条轨迹。
- `am_bench.models.spherical.SphericalPolynomialDAM`：球面连续多项式 DAM；它与二值异步 `PolynomialDAM` 是不同模型。
