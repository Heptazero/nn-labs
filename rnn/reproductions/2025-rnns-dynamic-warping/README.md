# RNNs Perform Task Computations by Dynamically Warping Neural Representations

```text
2025-rnns-dynamic-warping/
├── upstream/                  # 论文作者源码快照，不修改
├── reproduction/             # 可直接运行的 Colab 副本
└── prepare_colab_notebooks.py
```

- [Static network](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn/reproductions/2025-rnns-dynamic-warping/reproduction/static_network_colab.ipynb)
- [Evidence integration](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn/reproductions/2025-rnns-dynamic-warping/reproduction/evidence_integration_colab.ipynb)
- [Working memory](https://colab.research.google.com/github/Heptazero/nn-labs/blob/main/rnn/reproductions/2025-rnns-dynamic-warping/reproduction/wm_colab.ipynb)

不需要 Google Drive。首个代码单元会克隆本仓库，并从 `upstream/setup.py` 安装 `equinox`、`optax`、`diffrax` 等作者声明的依赖。
