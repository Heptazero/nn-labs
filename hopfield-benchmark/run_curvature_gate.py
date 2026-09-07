"""运行冻结的三模式曲率门实验并生成图。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from am_bench.curvature_gate import CurvatureGateConfig, run_experiment


def make_figure(result: dict, results_dir: Path, output_path: Path) -> None:
    import csv
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager

    font = None
    font_candidates = (
        ROOT / "NotoSansCJKtc-Regular.otf",
        Path.home() / "Library/Fonts/SourceHanSerifSC-Regular.otf",
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    )
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)
            font = font_manager.FontProperties(fname=font_path)
            plt.rcParams["font.family"] = font.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    rows = list(csv.DictReader((results_dir / "trajectory_rates.csv").open(encoding="utf-8")))
    for row in rows:
        for key in ("beta", "rate", "r2", "initial_speed", "oracle_margin"):
            row[key] = float(row[key])
        row["pattern_seed"] = int(row["pattern_seed"])

    primary = result["primary_beta"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {"pure": "#2f6f9f", "mixture": "#d95f45"}

    def bank_medians(probe, beta):
        grouped = {}
        for row in rows:
            if row["probe"] == probe and row["beta"] == beta:
                grouped.setdefault((row["pattern_seed"], row["memory_kind"]), []).append(row["rate"])
        return {key: float(np.median(values)) for key, values in grouped.items()}

    medians = bank_medians("seed_relaxation", primary)
    seeds = sorted({seed for seed, kind in medians if (seed, "pure") in medians and (seed, "mixture") in medians})
    pure = np.array([medians[(seed, "pure")] for seed in seeds])
    mixture = np.array([medians[(seed, "mixture")] for seed in seeds])
    axes[0, 0].scatter(pure, mixture, s=22, alpha=0.7, color="#6a51a3")
    limits = [min(pure.min(), mixture.min()), max(pure.max(), mixture.max())]
    axes[0, 0].plot(limits, limits, linestyle="--", color="0.4")
    axes[0, 0].set_xlabel("纯模式衰减率（库内中位数）", fontproperties=font)
    axes[0, 0].set_ylabel("混合态衰减率", fontproperties=font)
    axes[0, 0].set_title(f"主检验：β={primary:g}，每点一个模式库", fontproperties=font)

    for position, probe in enumerate(("seed_relaxation", "matched_local_probe")):
        values = []
        for kind in ("pure", "mixture"):
            values.append([row["rate"] for row in rows if row["beta"] == primary and row["probe"] == probe and row["memory_kind"] == kind])
        bp = axes[0, 1].boxplot(values, positions=[position * 3 + 1, position * 3 + 2], widths=0.7, patch_artist=True, showfliers=False)
        for patch, kind in zip(bp["boxes"], ("pure", "mixture")):
            patch.set_facecolor(colors[kind])
            patch.set_alpha(0.65)
    axes[0, 1].set_xticks([1.5, 4.5], ["原始状态\n自由松弛", "平衡点附近\n匹配扰动"], fontproperties=font)
    axes[0, 1].set_ylabel("拟合衰减率", fontproperties=font)
    axes[0, 1].set_title("红：混合态；蓝：纯模式", fontproperties=font)

    summaries = result["summaries"]
    betas = sorted({x["beta"] for x in summaries})
    eligible = [next(x for x in summaries if x["beta"] == beta)["eligible_banks"] for beta in betas]
    axes[1, 0].bar([str(x) for x in betas], eligible, color="#4c956c")
    axes[1, 0].axhline(result["config"]["min_eligible_banks"], linestyle="--", color="0.3")
    axes[1, 0].set_xlabel("β")
    axes[1, 0].set_ylabel("合格模式库 / 256", fontproperties=font)
    axes[1, 0].set_title("所有不合格库均保留在 bank_status.csv", fontproperties=font)

    for probe, marker, label in (("seed_relaxation", "o", "自由松弛"), ("matched_local_probe", "s", "局部扰动")):
        subset = sorted((x for x in summaries if x["probe"] == probe), key=lambda x: x["beta"])
        axes[1, 1].plot([x["beta"] for x in subset], [x["rate_gate"]["auc"] for x in subset], marker=marker, label=label)
    axes[1, 1].axhline(result["config"]["min_auc"], linestyle="--", color="0.3", label="预注册阈值")
    axes[1, 1].set_ylim(0.45, 1.02)
    axes[1, 1].set_xlabel("β")
    axes[1, 1].set_ylabel("五折跨库 AUC", fontproperties=font)
    axes[1, 1].set_title("分类器只看一个标量：衰减率", fontproperties=font)
    axes[1, 1].legend(prop=font)

    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle(f"Hopfield 三模式真假记忆门控：{result['decision']}", fontproperties=font, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    output_dir = ROOT / "hopfield-benchmark" / "curvature-gate-results"
    result = run_experiment(output_dir, CurvatureGateConfig())
    make_figure(result, output_dir, ROOT / "hopfield-benchmark" / "curvature_gate_results.png")
    print(result["decision"])
