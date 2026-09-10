# nn-labs

神经网络论文复现、统一基准和机制实验。所有可执行入口优先提供 Colab；课程作业不放在本仓库。

```text
nn-labs/
├── associative-memory/
│   ├── reproductions/   # 单篇论文复现
│   ├── benchmarks/      # 跨论文、同协议比较
│   └── experiments/     # 论文之外的机制实验
└── rnn/
    └── reproductions/
```

## Colab 入口

### Associative memory：论文复现

- [1982 Hopfield](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/reproductions/1982-hopfield-emergent-computation/hopfield_1982.ipynb)
- [1985 Hopfield spin glass](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/reproductions/1985-hopfield-spin-glass/hopfield_1985.ipynb)
- [2020 Hopfield image retrieval](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/reproductions/2020-hopfield-networks-is-all-you-need/hopfield_image_retrieval_colab.ipynb)
- [2020 HopfieldPooling / MNIST Bags](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/reproductions/2020-hopfield-networks-is-all-you-need/mnist_bags_hopfield_pooling_colab.ipynb)
- [2025 Hopfield-Fenchel-Young](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/reproductions/2025-hopfield-fenchel-young/hopfield_fenchel_young_capacity_colab.ipynb)

### Associative memory：统一基准

- [Q04 retrieval dynamics](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/benchmarks/am-bench/notebooks/retrieval_dynamics.ipynb)
- [Query-probe reliability](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/benchmarks/am-bench/notebooks/retrieval_reliability.ipynb)
- [Phase-one benchmark](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/benchmarks/am-bench/notebooks/hopfield_benchmark_phase1_colab.ipynb)

### Associative memory：机制实验

- [LAP associative memory](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/experiments/lap-associative-memory/lap_associative_memory_colab.ipynb)
- [Modern Hopfield playground](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/associative-memory/experiments/modern-hopfield-playground.ipynb)

### RNN：论文复现

- [Static network](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn/reproductions/2025-rnns-dynamic-warping/reproduction/static_network_colab.ipynb)
- [Evidence integration](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn/reproductions/2025-rnns-dynamic-warping/reproduction/evidence_integration_colab.ipynb)
- [Working memory](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn/reproductions/2025-rnns-dynamic-warping/reproduction/wm_colab.ipynb)

打开任一链接后选择 **Runtime → Restart session and run all**。Notebook 会使用 Colab 预装环境；只有 Colab 缺少的依赖才会在首个环境单元安装。
