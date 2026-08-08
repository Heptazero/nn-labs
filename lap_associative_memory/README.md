# LAP × 联想记忆：容量与干预鲁棒性实验

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Heptazero/nn-labs/blob/agent/add-lap-associative-memory-colab/notebooks/lap_associative_memory_colab.ipynb)

这个仓库实现了一个可复现实验：在 Dense Associative Memory / Modern Hopfield 对照下，测量 Locality–Autonomy Principle（LAP）正则化强度与干预泄漏、95% 存储容量、吸引域宽度之间的权衡。

## 最快使用方式

点击上面的 Colab 按钮，按顺序运行全部单元格。默认是小规模烟雾实验，用来验证数据、二阶导、干预 clamp、训练和绘图整条链路。确认无误后，把 notebook 中的 `FULL_EXPERIMENT = False` 改成 `True`，建议选择 GPU runtime 并把输出目录挂载到 Google Drive。

运行中会持续覆盖保存：

- `metrics.csv`：所有模型、λ、N、seed、噪声水平和指标；
- `training_history.csv`：重建损失和实际 LAP 二阶导惩罚；
- `config.json`：本次参数；
- `figure1_tradeoff.png` 至 `figure4_lap_training.png`。

## 一个必要的建模修正

原实验草案同时要求“局部 MLP 只能输入自身和父节点”以及“用 LAP 惩罚它对非父节点的混合二阶偏导”。这两条若严格同时满足，混合偏导会由计算图直接变成恒等于零，λ 不可能产生任何实验效应。

这里把每个机制写成：

```text
E_i(z) = E_i^legal(z_i, z_PA(i)) + α E_i^residual(z_all)
```

第一支保留硬编码的 DAG 归纳偏置；第二支提供可测量的非法软耦合。LAP 只会对后者产生压力，因此 λ=0 与 λ>0 的差异是可辨识的。非法混合 Hessian 的 Frobenius 范数用 Hutchinson probe 估计，每一项确实经过两次 `torch.autograd.grad(..., create_graph=True)`。

## 本地运行

```bash
python -m pip install -r ../requirements.txt
python ../run_experiment.py --quick --output ../results/quick
pytest
```

完整扫描：

```bash
python ../run_experiment.py --full --output ../results/full
```

完整配置为 `d=10`、`N∈{5,10,20,50,100}`、`λ∈{0,0.01,0.1,1,10}`、3 个 seed。二阶导训练开销较高，CSV 会在每个 `(seed,N,λ)` 完成后 checkpoint。

## 代码位置

- `experiment.py`：SCM、Modern Hopfield、E-SCM、LAP、三步干预、指标、扫描和绘图；
- `../notebooks/lap_associative_memory_colab.ipynb`：Colab 入口；
- `tests/`：数据、二阶导和 clamp 的最小自动测试。

> 当前实现是验证理论方向的研究原型，而不是预注册实验。正式报告结果前，应增加 seed 数、记录置信区间，并检查 λ 对 residual 权重尺度与检索优化收敛率的影响。
