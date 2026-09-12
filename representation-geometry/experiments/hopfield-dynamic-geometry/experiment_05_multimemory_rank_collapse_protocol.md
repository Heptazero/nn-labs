# 实验 05：完整记忆库中的 Jacobian 秩坍缩

## 1. 研究边界

既有工作已用有限时 Jacobian 谱和 Lyapunov 指数研究 recurrent
self-attention 的稳定性与临界性。本实验研究更窄的问题：在完整记忆库的
现代 Hopfield 检索中，输入敏感度的有效秩在何时坍缩，softmax 与
sparsemax 的坍缩轨迹是否不同，以及这些轨迹是否与正确、错误、混合和
伪吸引子结局相关。

这里不再研究双记忆边界，也不声称首次沿迭代观察 Jacobian 谱。全谱负责
回答“还剩多少敏感方向”，预注册的二维图册只负责展示“选定方向在哪里、
怎样压缩”。定量结果与图册必须来自同一个迭代映射。

现有 `ContinuousModernHopfield.readout` 使用全部模式计算一次 softmax
读出；`IterativeModernHopfield` 再把输出反馈为下一步查询，见
`associative-memory/benchmarks/am-bench/src/am_bench/models/modern.py:43`
和 `:56`。实验 05 复用这条更新公式，不复用 `:76` 的浮点完全相等停止
条件。

## 2. 冻结更新与时间分段

记忆按列写成

\[
X=[\xi^1,\ldots,\xi^K]\in\mathbb R^{N\times K},
\qquad \|\xi^\mu\|^2=N.
\]

为使不同维度下的锐度可解释，正式代码使用无量纲锐度 `alpha`：

\[
\theta_t=\frac{\alpha}{N}X^\top x_t,
\qquad x_{t+1}=X\,T(\theta_t),
\]

其中 \(T\) 分别为 softmax 和 sparsemax。网络在每一步始终使用全部
\(K\) 条记忆；二维图册选择的 anchor 不会删除其他记忆。

softmax 的一步 Jacobian 为

\[
D F(x)=\frac{\alpha}{N}
X[\operatorname{Diag}(p)-pp^\top]X^\top.
\]

因此必须分开报告：

- `t=0 -> 1`：\(\mathrm{span}(X)^\perp\) 被结构性投影掉，一步秩至多
  \(K-1\)；
- `t>=1`：只在 memory span 内研究检索造成的后续有效秩坍缩。

若以后加入 residual、normalization，或使 `rank(X)=N`，必须另立实验，
不能沿用这条结构投影结论。

## 3. 一个主问题与一个条件性问题

### 05A：变体差异

> 在匹配第一步竞争锐度后，softmax 与 sparsemax 的 memory-span Jacobian
> 有效秩轨迹是否仍有至少一步的稳定差异？

这是第一道生死门，也是实验 05 唯一的确认性主问题。一次运行只检验这个
问题，不使用终态类别挑选条件或图册。

### 05B：结局预测

只有 05A 通过且冻结数据中具有足够的正确与非正确结局，才检验：

> 秩坍缩时序能否在 entropy、IPR、support、score gap 和检索速度之外，
> 提前预测正确与非正确结局；若类别覆盖允许，能否进一步区分错误记忆与
> 稳定混合态？

05B 使用 05A 已冻结的轨迹，不为提高 AUC 修改锐度、seed、阈值或标签。
若类别覆盖不足，则记录为 `not_testable`，不临时增加相关记忆或挑选失败
cue；相关记忆与真实图片另立实验 06。

## 4. 数据与预算

### 开发预跑

- `N=128, K=100`；
- 1 个 memory seed；
- 12 个预注册目标；
- 每个目标 2 个 corruption mask；
- 只用于检查速度、数值恒等式和类别覆盖，不形成研究结论。

### 正式运行：先廉价校准，再计算 Jacobian

- `N=128, K=100`；独立 Rademacher 记忆；
- memory seeds `0..7`，seed 是统计独立单位；
- Hamming corruption `rho in {0.10, 0.25, 0.40}`；
- 第一阶段只做前向校准：所有 100 条记忆都轮流作为目标，在开发预跑冻结
  的稠密 `alpha` 网格上记录第一步 IPR、support、success 和结局；
- 稠密网格初始候选为
  `alpha in {0.25,0.5,0.75,1,1.5,2,3,4,6,8,12}`；若开发 seed 显示共同
  IPR 覆盖不足，只能在正式 seed 开始前一次性扩展并写入 `protocol.json`；
- 第二阶段才计算 Jacobian：每个 memory seed 用
  `jacobian_target_seed=8000+memory_seed` 预抽 32 个目标，每个
  `target × rho` 使用一个由三者唯一确定的 corruption mask；
- 表示秩仍覆盖全部 100 个目标，每个目标 8 个固定 mask，但只做前向迭代；
- 局部与累计 Jacobian 记录 `t=0..12`；结局判定最多迭代 100 步；
- float64；在 memory-span 坐标中分块计算，不保存每个 cue 的完整 Jacobian，
  只保存谱、迹和可复算所需的 seed/config。

原始同 `alpha` 扫描回答“相同 logits 尺度下变体怎样不同”，是描述性结果。
确认性比较回答“匹配初始竞争集中度后是否仍不同”：先用开发 seed 在两种
方法共同覆盖的第一步 median IPR 区间内冻结 3 个对数等距 IPR 水平；正式
seed 内只用其余目标选择距离每个水平最近的 alpha，再在留出目标评估。
匹配后 median IPR 相对误差超过 5% 的条件标为 `unmatched`，不外推。

检索质量匹配只作为敏感性分析：只在两种方法实际成功率重叠的区间内
比较，不通过插值制造不存在的工作点。

## 5. 完整谱与秩指标

必须同时区分一步局部 Jacobian 与从初态累计到当前步的 Jacobian。令

\[
A_t=D F(x_t),\qquad
J_t=\frac{\partial x_t}{\partial x_0}=A_{t-1}\cdots A_0,
\qquad G_t=J_t^\top J_t.
\]

`A_t` 回答“当前这一步保留多少方向”，`J_t`/`G_t` 回答“初态信息至今还
剩多少方向”。对 \(t\ge1\)，用 `X=QR` 的正交基 \(Q\) 把计算精确限制在
\(\mathrm{span}(X)\)；这只是去掉已知的零正交补，不做降维近似。每个
cue、每步保存：

- 局部 `A_t^T A_t` 与累计 \(G_t\) 的全部非负特征值；
- `trace_G = tr(G_t)`；
- `spectral_G = lambda_max(G_t)`；
- `effective_rank = exp(H(lambda / sum(lambda)))`；
- `stable_rank = tr(G_t) / lambda_max(G_t)`；
- 数值秩，阈值为 `lambda_i > 1e-10 * lambda_max`；
- attention entropy、IPR `1/sum(p^2)`、support size、top-1/top-2 score
  gap、状态步长和结局标签。

effective rank 与 stable rank 都以 \(G\) 的特征值定义，不混用
\(J\) 的奇异值归一化。若 `trace_G` 低于 `1e-14 * trace_G(t=1)`，方向秩
记为 `sensitivity_extinct`，不把浮点噪声解释为剩余方向。

对累计谱定义方向存活率

\[
d_t=\frac{r_{eff}(t)-1}{r_{eff}(1)-1}.
\]

若 `r_eff(t=1)-1 <= 1e-8`，该轨迹记为 `collapsed_at_entry` 并令后续
`d_t=0`，不执行除法。

灵敏度灭绝后令 `d_t=0`，同时单独报告 `trace_G`，避免把方向减少与幅度
消失混成同一结论。主要轨迹摘要为 `dimension_auc = mean(d_t, t=1..12)`。
`t50`、`t10` 是次要指标，定义为 `d_t` 首次降到 50%/10% 且在余下记录
时刻不反弹的最早步；未达到时删失。sparsemax 另记局部 support 首次持续
变为 1 的步数。

## 6. 表示秩

每个 `seed × method × alpha × rho × t` 同时形成：

- within-target 矩阵：同一目标的 8 条 cue 轨迹先减去该目标均值，再计算
  有效秩；下降表示目标内去噪；
- between-target 矩阵：100 个目标中心先减全局均值，再计算有效秩；下降
  表示不同记忆中心彼此合并。

两者必须分开报告。总表示秩下降本身不被解释为好或坏。

## 7. 预注册二维图册

每个 memory seed 只根据 \(X\) 选择图册，不读取 cue 结局：

- 3 个近邻三元组：固定 anchor `0, 33, 66`，各取余弦最近的两个记忆；
- 3 个远邻三元组：相同 anchor，各取余弦最远的两个记忆；
- 3 个随机三元组：使用独立 `atlas_seed = 9000 + memory_seed`。

对三元组 \((\xi_1,\xi_2,\xi_3)\)，保留 barycentric 语义坐标

\[
q(z)=\xi_1+Bz,
\qquad B=[\xi_2-\xi_1,\xi_3-\xi_1],
\qquad h=B^\top B.
\]

统计和椭圆使用白化切向 \(B_w=B h^{-1/2}\)：

\[
G_{atlas,w}=B_w^\top G_{full}B_w.
\]

anchor 在 barycentric 图上的位置保持 `(0,0),(1,0),(0,1)`。不能直接把
`B` 换成 QR 的 `Q` 后仍沿用原三角形坐标。

每个图册使用每边 41 点的三角网格，输出固定时间切片
`t in {0,1,2,4,8,12}`：变形网格、metric
ellipse、`sqrt(det G)`、有效秩、各向异性和终态颜色。PCA 只展示状态像的
外在形状；PCA 从 \(X\) 拟合一次并跨方法、锐度和时间共用。另报告

\[
\mathrm{fidelity}=\|U_3^\top B_w\|_F^2/2,
\]

其中 `U_3` 是用于三维状态图的三个固定 PC。主图预注册为
`memory_seed=0`、anchor 0 的近邻/远邻图册，不允许按结果换图；其余图册
只进入汇总与补充材料。图册只运行三个匹配 IPR 水平中的中间水平，避免
把确认性全谱扫描复制九遍。

图册网格直接对二维坐标求 `J_atlas in R^(N x 2)`，不在 861 个格点逐个
构造完整 `N x N` Jacobian。开发预跑只在每张图册预注册的 25 个稀疏格点
同时计算完整 Jacobian，用来验证
`G_atlas^AD = B_w^T G_full B_w`；恒等式通过后，正式密网格只保存二维
pullback 结果。

瞬时 `G_full(t)` 前两个特征向量只作为最强方向诊断，不称为跨时间图册。

## 8. 图册拓扑对象

softmax 与 sparsemax 都是连续映射，所以有限步的连通三角形之像仍然
连通。实验不得报告“像断成几段”。允许报告的对象是：

- 终态标签在查询三角形中的连通区域数，按三角网格邻接计算；
- `lambda_min(G) / lambda_max(G) <= 1e-8` 的退化集合；
- 非相邻 query 点映到同一数值邻域的多重原像候选；
- 指定尺度下的终态点云簇；
- sparsemax 支持集分区和相邻格点支持发生变化的墙。

支持切换墙是查询域的分区边界，不称为状态像的连通分量。普通光滑曲率
不跨 sparsemax 支持墙计算；`det G` 接近零的格点一律 censor 曲率。

## 9. 统一结局分类器

图册和全量 cue 调用同一个分类函数。收敛先要求

\[
\|x_{t+1}-x_t\|/\sqrt N < 10^{-8}
\]

连续三步成立；100 步未满足则为 `unconverged`。收敛点还要检查局部
Jacobian 谱半径小于 `1-1e-6`，否则不能称为稳定吸引子。

现代 Hopfield 的任何读出都位于记忆凸包中，因此没有额外机制时，“稳定
混合态”与“其他伪吸引子”不能靠命名强行分开。确认性类别收窄为：

1. `correct_memory_like`；
2. `other_stored_memory_like`；
3. `stable_mixture`；
4. `unconverged_or_unstable`。

stored-memory-like、mixture 和 endpoint clustering 的数值阈值只允许在开发
预跑中根据固定点误差与数值精度冻结，不能根据哪种定义更有利于秩指标来
选择。稳定但不能归入前三类的候选记为 `unresolved_stable_endpoint`，只做
探索性聚类，不在实验 05 中宣称发现新的伪吸引子。未通过稳定性检查的
收敛候选单列 `nonstable_fixed_candidate`。

## 10. 数值自检

正式运行前必须全部通过：

1. `t=0` 的白化图册度量等于单位阵，允许相对误差 `1e-10`；
2. 图册自动微分结果满足
   \(G_{atlas}^{AD}=B_w^\top G_{full}B_w\)，相对误差 `1e-8`；
3. Ritz 夹逼满足
   \(\lambda_{max}(G_{atlas})\le\lambda_{max}(G_{full})+tol\)；
4. `t=0 -> 1` 的正交补 Jacobian 范数不超过 `1e-10` 倍完整范数；
5. 解析 softmax Jacobian、解析 sparsemax 固定支持 Jacobian分别与自动
   微分一致；支持墙上的不可微点不用于该恒等式；
6. 所有保存的 \(G\) 特征值在容差内非负；NaN、溢出或未匹配工作点必须
   显式保存状态。
7. 累计 Jacobian 递推满足 `J_(t+1)=A_t J_t`，并与直接对 `F^t` 自动微分
   一致；开发预跑相对误差阈值 `1e-8`。

## 11. 停止门

### 05A

- 主要效应：匹配第一步 IPR 后，softmax 与 sparsemax 的
  `dimension_auc` 配对 seed 差异绝对值至少 `0.10`，95% bootstrap CI
  不跨 0；`t50/t10` 与局部谱只负责解释差异发生在哪一步；
- 若曲线不分，停止“变体几何”路线；
- 若差异与 support/entropy/gap 的确定性变化完全同步，只保留为已知稀疏
  竞争机制的几何表达，不宣称新机制。

### 05B

- 以 memory seed 分组做 leave-one-seed-out；
- 首先预测 `correct` 与 `non-correct`；只有每个细分类至少出现在 2 个独立
  seed 中，才报告细分类宏平均 AUC；
- 基线只用早期 entropy、IPR、support、gap、状态步长和预测时刻前可获得的
  overlap margin；
- 加入早期 rank trajectory 后，宏平均 AUC 至少提高 `0.03`，配对 seed
  bootstrap CI 下界大于 0；
- 任何类别少于 2 个独立 seed 时，不报告该类别 AUC；
- 达不到则保留秩轨迹为描述图，不进入更多 Hopfield 类型或真实图片。

## 12. 冻结产物

正式 notebook 命名：

`experiment_05_multimemory_rank_collapse_colab.ipynb`

产物目录：

`artifacts/experiment_05/`

至少包含 `protocol.json`、`conditions.csv`、逐 cue 谱摘要、表示秩、图册
定义、图册网格观测、统一结局标签、数值自检、主图、负结果和删失条件。
开发预跑与正式结果写入不同子目录，正式运行不得覆盖预跑。
