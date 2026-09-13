# 实验 05R：逐 cue IPR 匹配的方法预跑

## 1. 证据身份

实验 05A 因 12/72 个组级 IPR 条件对未匹配而不可检验。已匹配子集的
总体 `sparsemax - softmax dimension_auc` 只有 `0.001619`，不值得为原总体
问题直接增加 alpha 网格。旧数据另出现事后线索：坍缩差异可能随竞争档位
反转。

05R 是探索性方法 pilot，不是 05A 的修复或续算。旧 memory seeds `0..7`
已经失去确认性资格，只用来检查逐 cue 求根是否可行、匹配是否精确，以及
反转交互是否达到继续投入新 seeds 的最低信号门。

## 2. 冻结输入

- `N=128, K=100`，Rademacher 记忆，memory seeds `0..7`；
- 每个 seed 沿用 `target_seed=8000+memory_seed` 预注册的 32 个目标；
- `rho in {0.10,0.25,0.40}`，每个 `target × rho` 沿用 mask 0；
- 共 `8 × 32 × 3 = 768` 条 cue；
- 每条 cue 检查三个内部竞争水平

\[
\mathrm{IPR}/K\in\{0.04,0.16,0.64\},
\qquad \mathrm{IPR}\in\{4,16,64\}.
\]

不使用旧 low IPR 约等于 1 的条件，因为 sparsemax 可有限步进入单支持，
而有限 alpha 的 softmax 只能渐近接近，比较会被变换定义支配。

## 3. 逐 cue 求根

对固定 cue 先计算一次

\[
s=X^\top x_0/N.
\]

对每个目标 IPR，softmax 与 sparsemax 分别求解

\[
\mathrm{IPR}(T(\alpha s))
=\frac{1}{\sum_\mu T_\mu(\alpha s)^2}
=q.
\]

求解器冻结为：

1. 下界 `alpha=0`，此时 IPR 为 K；上界从 1 开始倍增，直到 IPR 不高于
   目标，最多到 `2^20`；
2. 在有界区间内二分最多 80 次，同时保留相对误差最小的候选；
3. 每种方法相对目标 IPR 的误差及两方法之间的相对误差都必须低于
   `0.005`；
4. 同一 bracket 上抽样检查 IPR 随 alpha 非增，数值容差 `1e-10`；
5. 若最高 score 的并列数大于目标 IPR，该目标不可达；若任一方法不能
   bracket 或达到误差门，整对 cue 对称删失，不只删除一种方法；
6. 求根只读取第一步 score 与 IPR，不读取终态或 Jacobian，因此不做 target
   cross-fitting。

总工作点为 `768 cues × 3 levels = 2304` 个方法对。

## 4. 动力学与主要量

每条可达 cue 在各自求得的 alpha 上，继续沿用实验 05 的 memory-span 精确
Jacobian：

\[
A_t=DF(x_t),\qquad J_t=A_{t-1}\cdots A_0,
\qquad G_t=J_t^\top J_t,
\]

记录 `t=0..12` 的完整局部谱、累计谱、`trace_G`、effective/stable/numeric
rank、entropy、IPR、support、gap、步长、`dimension_auc`、`t50/t10`、
`collapsed_at_entry` 与 `sensitivity_extinct`。每条 cue 的 alpha 独立，不在
同一批次中用组级 alpha 替代。

对每个 seed，先在三个 rho 与 32 个目标上分别平均，得到

\[
\Delta_q=\overline{\mathrm{dimension\_auc}}_{sparsemax,q}
-\overline{\mathrm{dimension\_auc}}_{softmax,q}.
\]

pilot 的唯一主要量为

\[
C=\Delta_{middle}
-\frac{\Delta_{low}+\Delta_{high}}{2}.
\]

报告 8 个 seed 的 C 与均值；bootstrap 区间只作描述，不作为 pilot 门，也
不能写成确认性证据。

## 5. 停止门

05R 只有同时满足以下条件才允许设计实验 06：

1. 可达且成功匹配的方法对不少于全部 2304 对的 95%；
2. 所有保留对的最大 IPR 目标误差和方法间误差均不超过 0.5%；
3. seed 级 C 的平均值满足 `abs(mean(C)) >= 0.05`；
4. Jacobian 谱无意外 NaN/溢出，实验 05 的七项自检在每个 seed 都通过。

任一门失败即停止，不为提高 C 更改 IPR 水平、cue、rho、seed 或时间窗。
05B 始终不运行。二维图册也等到全新 seeds 的实验 06 通过后再做。

若 pilot 通过，实验 06 必须使用全新 memory seeds `100..107`，确认性门为

\[
C\ge0.10
\]

且 paired-seed bootstrap 95% CI 下界大于 0。实验 06 的问题是
`method × competition regime` 交互，不称为修复后的 05A。

## 6. 产物

写入 `artifacts/experiment_05R/pilot/`：冻结配置、逐 cue 求根表、对称删失
表、Jacobians 与谱、seed 级 Delta/C、描述性 bootstrap、主图、自动停止
判定和运行时间。原实验 05 的任何文件均不覆盖。
