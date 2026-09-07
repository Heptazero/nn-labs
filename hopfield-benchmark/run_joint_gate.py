"""运行冻结的反 Hebb 联合门控实验并生成图。"""
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from am_bench.joint_gate import ARMS, run_joint_experiment


def make_figure(summary: dict, results_dir: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    for candidate in (
        ROOT / "NotoSansCJKtc-Regular.otf",
        Path.home() / "Library/Fonts/SourceHanSerifSC-Regular.otf",
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ):
        if candidate.exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    with (results_dir / "joint_trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("probe_rate", "probe_initial_speed"):
            row[key] = float(row[key])
        for key in ("gate_activated", "target_preserved", "mixture_escaped", "converged"):
            row[key] = row[key] == "True"

    labels = {
        "frozen": "固定权重", "always_on": "始终反 Hebb",
        "initial_speed_gate": "初始速度门", "decay_rate_gate": "衰减率门",
    }
    colors = ["#8c8c8c", "#c44e52", "#4c72b0", "#55a868"]
    x = np.arange(len(ARMS))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    pure_activation = [summary["arms"][a]["gate_activation_pure"] for a in ARMS]
    mix_activation = [summary["arms"][a]["gate_activation_mixture"] for a in ARMS]
    width = 0.36
    axes[0, 0].bar(x - width / 2, pure_activation, width, label="纯模式", color="#4c72b0")
    axes[0, 0].bar(x + width / 2, mix_activation, width, label="混合态", color="#dd8452")
    axes[0, 0].set_xticks(x, [labels[a] for a in ARMS], rotation=12)
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("门激活率")
    axes[0, 0].set_title("两个学习门作出完全相同的决定")
    axes[0, 0].legend()

    for arm, color in zip(ARMS, colors):
        values = summary["arms"][arm]
        axes[0, 1].scatter(values["pure_damage_rate"], values["mixture_escape_rate"],
                           s=95, color=color)
        offset = (5, 5) if arm != "decay_rate_gate" else (5, -15)
        axes[0, 1].annotate(labels[arm],
                            (values["pure_damage_rate"], values["mixture_escape_rate"]),
                            xytext=offset, textcoords="offset points")
    axes[0, 1].set_xlabel("纯记忆损伤率（越低越好）")
    axes[0, 1].set_ylabel("混合态逃出率（越高越好）")
    axes[0, 1].set_xlim(-0.02, 0.25)
    axes[0, 1].set_ylim(-0.03, 0.40)
    axes[0, 1].set_title("门控改善损伤—逃出权衡，但速率没有增益")

    probes = [row for row in rows if row["arm"] == "frozen"]
    for kind, color, label in (("pure", "#4c72b0", "纯模式"), ("mixture", "#dd8452", "混合态")):
        subset = [row for row in probes if row["start_kind"] == kind]
        axes[1, 0].scatter([row["probe_initial_speed"] for row in subset],
                           [row["probe_rate"] for row in subset], s=22, alpha=0.65,
                           color=color, label=label)
    axes[1, 0].axvline(summary["config"]["initial_speed_threshold"], linestyle="--", color="#4c72b0")
    axes[1, 0].axhline(summary["config"]["rate_threshold"], linestyle="--", color="#55a868")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("初始 ||dx/dt||（对数轴）")
    axes[1, 0].set_ylabel("轨迹衰减率")
    axes[1, 0].set_title("两阈值在 192 个起点上零分歧")
    axes[1, 0].legend()

    convergence = [summary["arms"][a]["convergence_rate"] for a in ARMS]
    exposure = [summary["arms"][a]["mean_exposure"] for a in ARMS]
    axes[1, 1].bar(x - width / 2, convergence, width, color="#8172b3", label="收敛率")
    axes[1, 1].bar(x + width / 2, exposure, width, color="#ccb974", label="平均反 Hebb 暴露")
    axes[1, 1].set_xticks(x, [labels[a] for a in ARMS], rotation=12)
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_title("所有激活轨迹使用相同最大预算")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle("反 Hebb 联合动力学：STOP_CURVATURE_NARRATIVE", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results_dir = ROOT / "hopfield-benchmark" / "joint-gate-results"
    result = run_joint_experiment(results_dir)
    make_figure(result, results_dir, ROOT / "hopfield-benchmark" / "joint_gate_results.png")
    print(result["decision"])

