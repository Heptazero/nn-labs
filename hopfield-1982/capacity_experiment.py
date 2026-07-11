"""
复刻 Hopfield (1982) 存储容量曲线（对应论文图2）

思路：
  1. 随机生成 n 条长度为 N 的 ±1 记忆向量
  2. 用 Hebb 规则算出权重矩阵 T_ij = sum_s mu_i^s * mu_j^s, T_ii = 0
  3. 把每条记忆本身当作初始状态，跑异步更新动力学直到收敛
  4. 收敛后的状态与原始记忆比较，数错误位数（Hamming distance）
  5. 对不同的 n 重复多次 trial，统计平均错误率 -> 画出 n vs 错误率曲线
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def make_memories(n: int, N: int, rng: np.random.Generator) -> np.ndarray:
    # 随机生成 n 条 ±1 记忆，形状 (n, N)
    return rng.choice([-1, 1], size=(n, N))


def hebb_weights(memories: np.ndarray) -> np.ndarray:
    # Hebb 规则算连接强度矩阵, 对角线置0，不允许自连接
    T = memories.T @ memories
    np.fill_diagonal(T, 0)
    return T


def run_dynamics(
    T: np.ndarray, V0: np.ndarray, rng: np.random.Generator, max_sweeps: int = 50
) -> np.ndarray:
    # 异步更新直到一整轮 sweep 内没有任何神经元变化（收敛）
    N = V0.shape[0]
    V = V0.copy()
    for _ in range(max_sweeps):
        order = rng.permutation(N)
        changed = False
        for i in order:
            h = T[i] @ V
            new_v = 1 if h > 0 else -1  # h == 0 时保持不变
            if h != 0 and new_v != V[i]:
                V[i] = new_v
                changed = True
        if not changed:
            break
    return V


def experiment3(N: int, n_values, trials: int, seed: int = 0):
    """
    对每个 n：随机生成一批记忆 -> 算权重矩阵 -> 把每条记忆当初始状态跑收敛
    -> 记录错误位数，多次 trial 取平均
    返回：每个 n 对应的 (平均错误位数, 完全正确恢复的比例)
    """
    rng = np.random.default_rng(seed)
    mean_errors = []
    exact_recall_rate = []

    for n in n_values:
        errors_this_n = []
        exact_this_n = []
        for _ in range(trials):
            memories = make_memories(n, N, rng)
            T = hebb_weights(memories)
            # 随机挑一条记忆做初始状态（也可以全部n条都测,这里简化为每次trial测1条）
            s = rng.integers(n)
            V0 = memories[s].copy()
            V_final = run_dynamics(T, V0, rng)
            errors = np.sum(V_final != memories[s])
            errors_this_n.append(errors)
            exact_this_n.append(errors == 0)
        mean_errors.append(np.mean(errors_this_n))
        exact_recall_rate.append(np.mean(exact_this_n))

    return np.array(mean_errors), np.array(exact_recall_rate)


def theoretical_error_prob(n: int, N: int) -> float:
    """论文式(10)：单个bit出错的理论概率，用误差函数近似高斯积分"""
    from math import erfc, sqrt

    sigma = sqrt((n - 1) * N / 2) if n > 1 else 1e-9
    # P = 1/sqrt(2*pi*sigma^2) * integral_{N/2}^{inf} exp(-x^2/2sigma^2) dx
    #   = 0.5 * erfc( (N/2) / (sigma*sqrt(2)) )
    return 0.5 * erfc((N / 2) / (sigma * sqrt(2)))


if __name__ == "__main__":
    N = 100
    n_values = list(range(1, 21))
    trials = 100

    mean_errors, exact_rate = experiment3(N, n_values, trials)
    theory_p = [theoretical_error_prob(n, N) for n in n_values]

    print(
        f"{'n':>4} {'avg_errors':>12} {'exact_recall_rate':>18} {'theory_P(bit err)':>18}"
    )
    for n, e, r, p in zip(n_values, mean_errors, exact_rate, theory_p):
        print(f"{n:>4} {e:>12.2f} {r:>18.2f} {p:>18.4f}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(n_values, mean_errors, "o-", label="平均错误位数（仿真）")
    ax1.set_xlabel("n (存储记忆数)")
    ax1.set_ylabel("平均错误位数")

    ax2 = ax1.twinx()
    ax2.plot(n_values, exact_rate, "s--", color="tab:orange", label="完全正确恢复率")
    ax2.plot(
        n_values,
        [1 - p for p in theory_p],
        "x:",
        color="tab:green",
        label="理论单bit正确概率(近似基线)",
    )
    ax2.set_ylabel("比例")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(f"Hopfield 网络存储容量 (N={N})")
    plt.tight_layout()
    plt.savefig("capacity_curve.png", dpi=150)
    print("图已保存到 capacity_curve.png")
