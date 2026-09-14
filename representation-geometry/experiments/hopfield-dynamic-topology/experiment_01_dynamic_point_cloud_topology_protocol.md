# 实验 01：Hopfield 检索过程中的动态点云拓扑

## 0. 证据身份与唯一问题

这是新路线的首个方法实验，尚无结果。它不继承
`hopfield-dynamic-geometry` 的正负结论，也不宣称拓扑具有独立预测价值。

唯一问题是：

> 把不同初态在第 t 个完整异步 sweep 后的网络状态看作点时，点云的 H0
> 层次结构能否在收敛前分离最终进入不同真记忆或伪吸引子的轨迹？

第一实验只用离散经典 Hopfield。tanh、softmax、sparsemax 的模型比较另立
实验；不能在看过实验 01 后把它们补进同一确认性问题。

## 1. 代码基础与模型

使用仓库已有组件：

- 随机二值记忆：`am_bench/tasks.py:32-57`；
- 精确 Hamming 翻转 cue：`am_bench/tasks.py:60-83`；
- 全局随机初态：`am_bench/tasks.py:86-93`；
- Hebbian 零对角权重：`am_bench/models/classical.py:10-20`；
- 能量：`am_bench/models/classical.py:22-24`；
- 异步更新和固定点判停：`am_bench/models/classical.py:26-92`；
- 代码版本与模块哈希：`am_bench/provenance.py:8-20`。

冻结状态空间为 `{-1,+1}^N`，距离为归一化 Hamming 距离。一个时间步是
全部 N 个神经元完成一次异步 sweep，不是单个神经元更新。

同一 `memory_seed` 下，所有 cue 在同一 sweep 使用相同的坐标排列序列，避免
点云混入 cue 特有的更新顺序噪声。不同 memory seed 使用不同固定排列序列。
提前收敛的轨迹用最终状态填充到第 30 sweep，保证每个时间点包含同一批轨迹。

## 2. 冻结预算

正式预算：

- `N=64`；
- `P in {4,8,12}`；
- memory seeds `0..9`；
- 最大 30 个完整异步 sweeps；
- 每条记忆在噪声率 `rho in {0.10,0.20,0.30}` 下各生成 50 条 cue；
- 每个 `seed × P` 再生成 1000 条独立随机初态；
- 展示时刻 `t in {0,1,2,5,10,30}`，统计量保留全部 `t=0..30`。

所有随机数使用局部 generator，并把记忆、cue、更新排列和拓扑抽样的 seed
分别记录。不能依赖 PyTorch、NumPy 或 Python 的全局随机状态。

开发预跑只使用 `P=8, memory_seed=0`，每个 `target × rho` 取 10 条 cue，
另取 100 条随机初态。开发数据只验证实现和资源，不进入正式统计。

## 3. 固定拓扑样本量

按“每条记忆 50 条每噪声”的生成规则，P 不同时总点数不同；直接比较条数会
把样本量误当作拓扑差异。因此分开保存：

1. **全量轨迹集**：用于恢复率、终点类别、吸引盆大小和收敛时间；
2. **拓扑子集**：每个 `seed × P` 固定 1600 条轨迹，包括全部 1000 条随机
   初态，以及从记忆附近 cue 中按 `target × rho` 分层、确定性抽取的 600 条。

P=4 时 600 条附近 cue 全部进入；P=8、12 时按固定拓扑抽样 seed 抽取。
不得按终点类别补抽样，因为那会用未来结果改变输入点云。

## 4. 终点与伪吸引子标签

每条轨迹必须先通过固定点复查：对终点再做一个完整异步 sweep，状态不变才算
稳定固定点。达到 30 sweeps 仍未固定的轨迹单列为 `unconverged`，不能归入
伪吸引子。

经典 Hebbian 网络具有全局符号对称。对所有终点统一取商，而不是只合并真
记忆：`x` 与 `-x` 用确定性的 canonical sign 映射为同一个吸引子族。

- canonical 终点等于某条 canonical 存储模式：`stored_memory_family_mu`；
- 其他通过复查的 canonical 固定点：逐个编号
  `spurious_family_1, spurious_family_2, ...`；
- 未固定：`unconverged`。

若两条存储记忆恰好互为相反数或 canonical 后重复，整组 memory seed 标记为
`degenerate_memory_set`，保留原始数据但不进入组间比较。每条干净记忆是否为
固定点也必须报告；不能删除“不稳定的存储模式”来美化恢复率。

## 5. 点云、多重性与距离

固定 t 时，拓扑子集的状态组成 `A_t in {-1,+1}^{1600×64}`。检索后大量轨迹
会落在完全相同的状态。持久同调把输入当集合，而吸引盆统计把它当带多重性的
样本，因此必须保存两层：

- `unique_states_t`：唯一状态，供 H0/H1 计算；
- `state_multiplicity_t` 与终点标签直方图：供盆大小和 component purity 加权。

若直接把重复点全部送进条形码，会制造大量零长度 H0 条；禁止用这些零条解释
拓扑坍缩。

主距离是归一化 Hamming：

`d_H(x,y) = number(x_i != y_i) / N`。

同时计算：

- 原始 `d_H`；
- 每个 t 除以该时刻最大有限距离后的 `d_H_normalized`。

若时刻 t 只剩一个唯一状态，归一化距离定义为全零，并显式标记
`single_unique_state=True`，不能除以零。

## 6. H0、H1 与 Naitzat 距离

H0 是主要对象。对完整的唯一状态距离矩阵计算最小生成树/union-find
dendrogram；H0 的有限 death times 与最小生成树边权一致。保存完整 H0 diagram、
树边和 component merge tree。

一根 H0 bar 本身不提供唯一、稳定的“成员集合”。需要成员与纯度时，在预先
给定的 epsilon 网格上切 union-find dendrogram，再读取连通分量；不能事后把
某根长条直接命名为伪坑。

H1 只是探索性补充：从拓扑子集中按**初态来源**分层固定抽取最多 500 条，
再在唯一状态上计算。不能按终点标签抽样；H1 失败或为空不影响实验 01 的门。

近邻图距离是稳健性检查，不进入第一轮主判定。Naitzat–Zhitnikov–Lim 使用
欧氏距离只构造 kNN 图，此后把每条边长度视为 1，以最少边数定义图测地距离。
这里对 Hamming kNN 图测试 `k in {10,20,30}`，先对唯一状态建图并对称化。
若图不连通，记录分量，不得把无穷距离任意截断成一个大数。

## 7. 与终点标签的联系

终点标签只用于事后评估，不参与生成点云、选择拓扑子集、选择 epsilon 或选择
H1 子样本。

每个 epsilon 分量的纯度按轨迹多重性加权：

`purity = component 内最多终点标签的轨迹数 / component 内轨迹总数`。

同时报告 coverage，避免只有两个点的纯分量被当成早期伪坑：

`coverage = 该分量中某终点族轨迹数 / 该终点族全部轨迹数`。

“伪坑群提前形成”只作描述性判据：在收敛前，某个至少包含 20 条轨迹的
component 对同一 spurious family 达到 `purity >= 0.90`、`coverage >= 0.50`，
并连续维持两个 sweeps。未满足最小族大小的 seed 记为 `not_testable_spurious`，
不能算失败或成功。

## 8. 主要拓扑量与简单基线

主拓扑量不是“长条数量”单独一个数，而是 H0 merge tree 给出的 cophenetic
distance：两条轨迹首次落入同一 component 的 epsilon。固定抽取同数目的
“相同终点族”和“不同终点族”轨迹对，使用负 cophenetic distance 预测两条
轨迹是否进入同一最终吸引子，得到 `H0_pair_AUC(t)`。

同一批 pair 必须计算：

- 原始 Hamming pair distance；
- 当前两点的最近记忆 identity、最大 overlap 与 overlap margin；
- 当前能量及能量差；
- 当前 sweep 的状态改变量；
- 固定 PCA 坐标中的普通聚类。PCA 每个 `seed × P` 只拟合一次并跨时间共用，
  禁止每个 t 单独重拟合。

拓扑的独立价值只在归一化 H0 仍有效，并且早期 `H0_pair_AUC` 比最强简单
pair baseline 平均高至少 0.03、以 memory seed 为单位的配对 bootstrap 95% CI
下界大于 0 时成立。否则 H0 只保留为动力学可视化。

伪吸引子子分析只有至少 5 个 seeds 各自存在两个或以上、且每族至少 20 条轨迹
的 spurious families 时才运行。样本不足时写 `not_testable_spurious_coverage`。

## 9. 数值与数据自检

Notebook 正式运行前，核心公式与数据契约的快速测试必须全部通过；随后在
Colab development preflight 中复核相同自检并生成数值产物：

1. 记忆和 cue 只含 `{-1,+1}`，翻转数精确等于 `round(rho*N)`；
2. `W` 对称、对角为零，并与共享 ClassicalHopfield 一致；
3. 每个异步单神经元更新后能量不增加，容差 `1e-10`；
4. 同 seed 重跑得到逐位相同轨迹，改变更新 seed 至少一条非平凡轨迹改变；
5. 提前收敛后的填充状态恒定；
6. canonical sign 满足 `canonical(x)=canonical(-x)`；
7. 唯一状态的 multiplicity 总和等于 1600，标签直方图总和一致；
8. 小点云 H0 death times 与独立 MST 边权逐项一致；
9. 原始和归一化距离矩阵对称、对角为零、满足取值范围；
10. 固定 PCA 在所有展示时刻使用同一组 components；
11. 原始输入、终点、未收敛轨迹和零长度条均未静默删除；
12. 依赖、源码提交、文件哈希、运行时间和随机种子完整保存。

任一项失败，不运行 10-seed 正式预算。本地测试只证明代码准备就绪，不能作为
拓扑现象或运行时间的证据；development 和 formal 数值证据都必须来自 Colab。

## 10. 产物与图

路径固定为：

- 开发：`artifacts/experiment_01/development/`；
- 正式：`artifacts/experiment_01/formal/`。

至少保存：

- `config.json`、`provenance.json`、`self_checks.csv`；
- 全量初态元数据、更新 seed、终点标签和吸引盆大小；
- 逐时刻激活矩阵，优先压缩 `int8 npz`；
- 唯一状态、多重性、H0 diagrams、MST/merge tree；
- H1 固定子样本索引和 diagrams；
- 逐 seed 指标、删失原因、pair AUC 与 bootstrap；
- `conclusion.md`、PNG/PDF 主图。

不保存每个时刻完整 pairwise distance matrix 作为正式产物。它可由激活矩阵
确定性重建，全部保存会造成数百 MB 到 GB 的重复数据。只保存校验用的小型或
选定时刻距离矩阵。

主图：固定 PCA 的 `t={0,1,2,5,10,30}` 点云；原始/归一化 H0 条形码；H0
pair AUC 与基线；伪坑 family 的 purity/coverage 时间；普通恢复率、伪坑率、
盆大小和收敛时间。

## 11. 停止顺序

1. 实现共享输入、异步轨迹、终点 quotient 和 H0/MST，不做 H1。
2. 用 `tests/test_experiment_01.py` 做快速公式与数据契约检查；这不是实验运行。
3. 生成固定源码提交和 SHA-256 的 Colab，在 Colab 只跑一个 development seed，
   提交自检、运行时间、唯一状态数量和伪坑覆盖。
4. development 门通过后，在 Colab 加入 H1 与固定 PCA 图；用户审查后再运行
   正式 `10 seeds × 3 P`。
5. 若归一化 H0 不优于简单基线，停止独立拓扑预测路线；不通过改 k、epsilon、
   抽样或终点阈值救结果。
6. 只有实验 01 通过，才另立模型比较实验，分别明确连续 tanh、现代 softmax
   和 sparsemax 的状态空间、收敛与竞争程度匹配。
