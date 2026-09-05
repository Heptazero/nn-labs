# Hopfield 统一任务基准协议

状态：`v0.1 / comparison-contract`；实现状态核对：2026-09-05。

本文件规定问题、接口、指标、预算与停止条件，本身不产生实验结论。六个适配器与配对运行器已提取到 `am_bench/`，第一阶段 Notebook 通过固定代码版本加载。新增 [Q04 Notebook](./retrieval_dynamics.ipynb) 实现三模型逐轮观测与批次恢复；尚未满足本协议的全部目标，仓库未保存本次执行输出。数值验收待 Colab 运行，问题覆盖、实现边界和下一步见 [QUESTIONS.md](./QUESTIONS.md)。

Q04 当前实例化的是独立二值记忆、精确位翻转、共同 Top-1 判据和原生更新轮数预算，不是等计算预算实验。模型层不接收 target；外部观察器记录 t=0、逐轮身份/误差/诊断，达到固定点后显式保持，失败后的缺测不外推。完整批次按配置与代码摘要冻结，原始记录校验后恢复。Q05 的诊断字段已经保存，但其正式分析尚未实现。

## 1. 核心问题

统一基准首先回答：

> 在相同数据、查询和资源预算下，不同 Hopfield 模型的容量、吸引域、错误吸引子和动力学有何差异？高阶结构的收益是否仍能在同参数量和同计算量下成立？

这个问题与逐篇复现互补：

- 论文复现验证“实现是否忠于论文”。
- 统一基准验证“模型之间是否可以公平比较”。
- 机制专属实验验证“优势来自什么，而不只是哪个数字更高”。

## 2. 组件式实验管线

所有共享任务使用同一条可替换管线：

`a 记忆输入 → b 模型存储 → c 检索线索 → d 检索动力学 → e 测量 → f 展示`

### a：记忆输入

`a` 只负责生成或读取要存储的模式，不包含模型特有的权重或更新规则。

- `a1_independent_binary`：独立随机 `{-1,+1}` 模式。
- `a2_correlated_binary`：相关度可控的二值模式。
- `a3_mnist`：MNIST 灰度与二值版本。
- `a4_cifar`：CIFAR 灰度/彩色向量与论文需要的二值版本。

输出至少包含：

- `patterns`
- `pattern_ids`
- `dataset_id`
- `data_seed`
- `encoding`
- `correlation_summary`

### b：模型存储

`b` 把同一组模式转换成模型内部表示。

- 经典模型可以输出 Hebb 权重。
- Dense/Modern Hopfield 可以保留记忆表。
- Simplicial/AHN/PSHN 可以输出单形、骨架或分组结构。
- 理论模型若没有可执行检索器，只注册理论曲线，不伪装成经验模型。

输出至少包含：

- `model_state`
- `model_id`
- `model_config`
- `parameter_count`
- `storage_bytes`
- `construction_cost`

### c：检索线索

`c` 从目标模式制造线索。所有模型必须接收同一批预先生成的线索，不能在模型内部重新采样噪声。

- `c1_clean`：无损模式。
- `c2_hamming_flip`：随机翻转指定比例的 bit。
- `c3_gaussian_noise`：添加指定方差的 Gaussian noise。
- `c4_block_occlusion`：遮挡固定方向和比例的连续区域。
- `c5_random_start`：与存储模式独立的随机状态。

输出至少包含：

- `cue`
- `target_id`
- `corruption_kind`
- `corruption_level`
- `cue_seed`

### d：检索动力学

`d` 是模型适配器的主要差异点。

- 迭代模型返回完整的终止状态和轻量轨迹摘要。
- 单步模型明确返回 `steps=1`，不伪造迭代过程。
- 默认不保存所有状态；只有动力学任务显式请求时才记录轨迹。
- 有理论能量的模型记录相应能量；没有同一能量定义时返回 `energy=None`。

输出至少包含：

- `final_state`
- `status`
- `steps`
- `state_updates`
- `energy_trace`（可空）
- `error_trace`（可空）
- `retrieval_cost`
- `diagnostics`

允许的统一终止状态：

- `fixed`
- `cycle`
- `one_step`
- `max_steps`
- `numerical_failure`
- `unsupported`

### e：测量

`e` 不读取模型名称，只根据目标、终态、轨迹和资源记录计算指标。

- `e1_exact_recall`
- `e2_overlap`
- `e3_mse`
- `e4_top1_memory`
- `e5_basin_success`
- `e6_spurious_class`
- `e7_convergence`
- `e8_resource_cost`

### f：展示

`f` 从标准化结果表绘图。绘图函数不得重新计算模型输出，也不得丢弃失败、超时或达到扫描上限的运行。

- `f1_recall_vs_corruption`
- `f2_recall_vs_memory_load`
- `f3_capacity_scaling`
- `f4_attractor_outcomes`
- `f5_dynamics_trace`
- `f6_quality_cost_pareto`
- `f7_mechanism_interaction`

## 3. 共享任务层

|ID|任务|主要问题|主要指标|适用边界|
|---|---|---|---|---|
|U1|干净固定点|记忆自身是否稳定？|一步不变率、最终不变率、exact recall|所有可执行模型|
|U2|噪声恢复|损坏输入能否找回目标？|exact recall、overlap、MSE、Top-1|按编码选择合法指标|
|U3|存储容量|随 `P`、`N` 增加何时失效？|成功率曲线、经验 `P_c(N)`、删失状态|容量判据必须预先固定|
|U4|吸引域|最大可恢复噪声是多少？|完整曲线、AUC、`rho@90%`|单步模型也可用鲁棒恢复曲线参加|
|U5|动力学|是否收敛、需要多少计算？|终止状态、步数、更新数、能量/误差轨迹|能量只在模型定义允许时比较|
|U6|错误吸引与干扰|是否落入错误或混合状态？|目标、错误记忆、混合态、随机态、未收敛比例|同时测随机初态和相关模式|
|U7|资源效率|优势是否只是更多资源？|参数、存储字节、FLOPs、延迟|所有跨模型结论都必须报告|

## 4. 高阶机制层

这些任务不进入所有模型的统一总表，而是由相关模型参加专属机制比较。

|ID|机制实验|最小对照|要排除的替代解释|
|---|---|---|---|
|H1|阶数消融|pairwise、higher-order、mixed|只是阶数更高|
|H2|结构消融|原结构、打乱结构、保持阶数/度数的随机结构|只是参数位置或数量变化|
|H3|预算消融|同参数、同存储、同 FLOPs|只是使用更多资源|
|H4|有效温度反证|固定温度、匹配温度日程、状态反馈温度|只是普通退火或尺度重参数化|
|H5|相变与迟滞|控制参数正向与反向扫描|只观察到单方向突变|
|H6|特征到原型|实例误差、类别原型误差、同类误检|高 exact recall 被原型化输出掩盖|
|H7|结构×相似度|完整 `2×2` 设计|收益其实完全来自相似度函数|

二因素交互实验必须包含：基线、只改 A、只改 B、同时改 A+B。主要指标的交互尺度必须在运行前声明；错误率、概率和容量指数不能不加说明地直接相减。

## 5. 论文专属轨道

以下实验保留在论文 notebook，不强迫其他模型参加：

- Hopfield 1982：随机非对称动力学、截断权重、单向连接、自然遗忘、新颖性、严重过载熟悉度、时间序列、同步二周期。
- Amit–Gutfreund–Sompolinsky 1985：RS 鞍点、FM/SG 能量、全局基态与亚稳态、有限温度相图、AT 线与 RSB。
- Continuous Modern Hopfield：attention 权重、分离 margin、`beta` 集中度、一次读出与 Transformer 对应。
- Simplicial Hopfield：单形阶数配比、稀释拓扑、同调诊断、结构与相似度交互。
- Curved/Explosive network：有效逆温度、爆发相变、迟滞、混合相和两模式多稳态。
- AHN/PSHN：骨架、分组数 `k`、单步检索、稀疏连接以及 feature-to-prototype 转变。

这些结果可以与共享任务并排展示，但不能混入同一主指标排名。

## 6. 容量不是一个数字

每个容量结果必须完整记录：

`capacity(model, N, data, corruption, criterion, retrieval_mode, budget)`

至少区分：

- `clean_fixed_point_capacity`
- `robust_capacity`
- `typical_pattern_capacity`
- `all_patterns_capacity`
- `local_attractor_capacity`
- `global_ground_state_capacity`

### 共同容量图

不同增长阶的模型可以直接画在同一张有限规模图上：

- 横轴：实际存储模式数 `P`，使用对数坐标。
- 纵轴：同一成功判据下的 recall、overlap 或 exact recall。
- 每条线：一个模型。
- 分面：`N`、噪声类型、噪声强度和预算类型。

容量缩放图使用：

- 横轴：`N`。
- 纵轴：经验临界容量 `P_c(N)`，使用对数坐标。

只有在解释模型内部理论时，才另行报告：

- 线性率：`alpha = P/N`
- 多项式系数：`P/N^(d-1)`
- 指数率：`rho = log(P)/N`

三者不得当作同一个归一化横轴。

## 7. 同图资格检查

两个模型只有在以下字段全部相同时，才能作为同一张主图上的两条线：

- `task_id`
- `dataset_id` 与实际样本 ID
- `encoding`
- `N`
- `P`
- `target_id`
- `corruption_kind`
- `corruption_level`
- `data_seed` 与 `cue_seed`
- `success_criterion`
- `retrieval_budget`
- `resource_budget_type`
- `stopping_rule`

允许不同的只有：

- `model_id`
- 该模型预先登记的超参数
- 模型内部必需且已披露的状态

如果字段不一致，应拆成分面或不同图，不能靠颜色掩盖协议变化。

### 视觉约定

- 颜色只编码 `model_id`。
- 线型只编码预算或检索模式，不重复编码模型。
- 浅色点显示单次试验，实线显示条件均值或中位数。
- 阴影显示预先选定的不确定性区间。
- 失败、超时和删失必须在图或伴随表中出现。
- 理论曲线与经验曲线使用不同线型，并在图例中写明证据性质。

## 8. 公平预算

每个共享任务至少输出三种视图：

1. `native`：模型在论文或标准实现中的自然配置。
2. `matched_storage`：参数量或存储字节匹配。
3. `matched_compute`：检索 FLOPs、状态更新次数或延迟匹配。

调参搜索次数和候选范围也是预算。某个模型不能独享更大的超参数搜索。

## 9. 标准结果记录

每一行只记录一条目标记忆的一次检索，不提前聚合：

|字段组|字段|
|---|---|
|身份|`run_id`, `git_commit`, `model_id`, `task_id`|
|数据|`dataset_id`, `pattern_set_id`, `target_id`, `N`, `P`, `encoding`|
|随机性|`data_seed`, `structure_seed`, `cue_seed`, `update_seed`|
|线索|`corruption_kind`, `corruption_level`|
|检索|`retrieval_mode`, `max_steps`, `status`, `steps`, `state_updates`|
|质量|`exact_recall`, `overlap`, `mse`, `top1_id`, `attractor_class`|
|资源|`parameter_count`, `storage_bytes`, `retrieval_flops`, `wall_time_ms`|
|边界|`right_censored`, `failure_reason`, `paper_reported`|

原始记录不可变。均值、置信区间、临界容量和图表必须能从原始记录重新生成。

## 10. 第一阶段最小实验

第一阶段只使用独立随机二值模式，检验共享接口和公平预算，不立即加入所有真实数据实验。

### 模型候选

- Classical Hopfield
- Polynomial Dense Associative Memory
- Exponential Dense Associative Memory
- Continuous Modern Hopfield（仅参加兼容二值输入的任务）
- Simplicial `K1/R12`
- Curved associative memory
- PSHN

### 扫描维度

- 多个网络规模 `N`。
- 存储数 `P` 从低负载逐步增加。
- Hamming corruption 从干净线索递增到随机水平。
- 所有模型共享模式、目标、线索和种子。

### 主要指标

- `exact_recall`
- `overlap`
- `rho_at_90_recall`
- `spurious_rate`
- `convergence_rate`
- `retrieval_flops`
- `storage_bytes`

### 停止条件

- 容量扫描只有在成功率连续两个负载点低于预设失败线后才停止。
- 若到达资源上限仍未失效，标记为右删失，只报告 `P_c` 至少达到当前上界。
- 单次运行达到 `max_steps`、显存上限或时间上限后停止并保留失败记录。
- Gate 1 的经典基线未复现前，不进入高阶组合比较。

## 11. 与现有 notebook 的关系

现有 notebook 不删除、不搬空：

- `hopfield-1982/`：经典模型验收与论文专属动力学。
- `hopfield-1985/`：容量定义、FM/SG 与零温理论基准。
- `hopfield is all you need/`：经典、Dense、Continuous Modern 的共同图像接口。
- `hopfield-fenchel-young/`：softmax/sparsemax 容量与 margin 机制。

后续实现先从这些 notebook 中提取经过验证的组件，再接入统一适配器。提取前后必须保持论文 notebook 的输出和证据边界不变。

## 12. 实验门

- Gate 0：固定任务、指标、预算、模型资格和停止条件。
- Gate 1：逐模型复现论文或参考实现的关键基线。
- Gate 2：只替换一个组件，确认共享接口没有改变任务。
- Gate 3：高阶机制运行完整 `2×2` 交互实验。
- Gate 4：匹配温度、参数、自由度与计算量，排除重参数化解释。
- Gate 5：扩展到相关模式、MNIST/CIFAR 和不同规模。

当前已有模型注册表、六个内联适配器和部分机制实验入口，但代码存在不等于 Gate 1–5 已通过。下一步先提取共享组件、核对行为，再完善逐步检索比较；具体顺序见 [QUESTIONS.md](./QUESTIONS.md)。
