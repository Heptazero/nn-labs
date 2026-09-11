# 实验 4 结论草稿

- 机制判定：`same_location_controls_remove_G_advantage`。
- 拓扑有效比例：1.000。
- t >= 1 的移动边界中位夹角：G=0.109°，同点 overlap=0.109°，同点 energy Hessian=0.109°，S0=4°，St=0.109°。
- t=0.02 时 G 与 S0 方向中位夹角：0.104°；短时线性近似相对误差：0.0331。
- 相对 G 的配对 seed 夹角劣势及 95% bootstrap CI：
  - same_point_overlap_gradient: -1.45e-06° [-3.19e-06, 5.62e-07]。
  - same_point_energy_hessian: -4.37e-06° [-1.93e-05, -2.32e-06]。
  - instantaneous_strain_S0: 4.04° [3.45, 4.36]。
  - time_varying_strain_St: 6.8e-07° [-6.92e-06, 1.27e-06]。

判读规则：只有 G 对同点 overlap、同点 energy 和固定的初始局部应变 S0 都超过冻结的 1° 裕量，才进入全平面盲定位。St 是逐时刻诊断量，因为它使用了 Jt 和 xt，所以单独报告，不与 S0 混为一个基线。

同位置控制消除了 G 对 overlap 或 energy 的冻结裕量优势；实验 3 的基线差距主要与读取位置不匹配一致。
