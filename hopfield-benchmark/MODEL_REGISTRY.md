# Phase 1 模型注册表

状态：`implementation-only / not-run`

本表只登记第一阶段已经接入统一接口的模型。`not-run` 表示尚未在 Colab 执行，因此不能从这里得出容量或优劣结论。

|`model_id`|组件 b：存储|组件 d：检索|当前资格|
|---|---|---|---|
|`classical_hebb`|归一化 Hebb 权重，主对角线为零|随机顺序异步坐标更新，零场保持原状态|Gate 1 候选基线|
|`polynomial_dam_d3`|直接保存 `P×N` 二值记忆表|三次 power energy 的精确单坐标比较|Gate 1 候选基线|

## 目前允许的结论

- 两个 adapter 已采用相同的 `fit → retrieve → resource_summary` 接口。
- 统一运行器可以把同一组模式、目标、Hamming 线索和随机种子交给两个模型。
- 原始表会保留失败、达到最大步数和右删失信息。
- 作图前会检查 trial 是否完整配对。

## 目前不允许的结论

- 不宣称任何模型容量更高、恢复更强或计算更快。
- `retrieval_flops` 是显式操作计数的近似量，不是硬件实测延迟。
- `polynomial_dam_d3` 尚未完成对原论文图表或参考实现的数值验收。
- 当前只实现 `native / equal_max_sweeps` 视图，还没有完成 `matched_storage` 和严格的 `matched_compute` 视图。

## Gate 1 验收顺序

1. Classical Hopfield：干净固定点、异步能量不增、有限规模容量和吸引域。
2. Polynomial DAM：单坐标能量不增、degree=2 退化检查、degree=3 的论文级容量切片。
3. 两条基线分别通过后，才解释同图差异。
