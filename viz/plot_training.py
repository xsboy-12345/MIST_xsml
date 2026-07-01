"""
viz/plot_training.py

可视化训练过程：
  1. Tau 相关系数随 epoch 的变化（Llama vs Qwen）
  2. 推理结果的预测分 vs 人工分散点图
  3. 预测分数分布 vs 人工分布对比
  4. 各维度 tau 雷达图

用法:
    python viz/plot_training.py
    python viz/plot_training.py --tau-only   # 只画 tau 曲线
"""

from __future__ import annotations
import argparse, json, math, pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = pathlib.Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
PRED_DIR = ROOT / "eval/outputs"
FIG_DIR  = ROOT / "viz/figures/training"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS   = ["faithfulness", "coverage", "naturalness", "coherence"]
COLORS = {"llama": "#4C72B0", "qwen": "#DD8452"}


def load_tau_log(name: str) -> list[dict] | None:
    path = LOG_DIR / f"{name}_tau_log.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions(name: str) -> list[dict] | None:
    path = PRED_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── 1. Tau 曲线 ──────────────────────────────────────────────────────────────

def plot_tau_curves(logs: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：平均 tau
    ax = axes[0]
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        taus   = [e.get("tau_average", float("nan")) for e in log]
        color  = COLORS.get(name.lower().split("-")[0], None)
        ax.plot(epochs, taus, marker="o", label=name, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kendall's τ (average)")
    ax.set_title("Average Tau vs. Epoch", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # 右：各维度 tau（最后一个 epoch）
    ax = axes[1]
    x = np.arange(len(DIMS))
    width = 0.35
    for i, (name, log) in enumerate(logs.items()):
        last = log[-1]
        vals = [last.get(f"tau_{d}", last.get(d, float("nan"))) for d in DIMS]
        color = COLORS.get(name.lower().split("-")[0], None)
        offset = (i - (len(logs) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIMS])
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Per-Dimension Tau (Final Epoch)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Training Progress: Tau Correlation with Human Ratings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "tau_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


# ── 2. 预测分 vs 人工分散点图 ─────────────────────────────────────────────────

def plot_scatter(preds_map: dict[str, list[dict]]) -> None:
    n_models = len(preds_map)
    fig, axes = plt.subplots(n_models, 4, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (name, preds) in enumerate(preds_map.items()):
        valid = [p for p in preds if p["answer"] not in ("-1", None)]
        for col, dim in enumerate(DIMS):
            ax = axes[row, col]
            dim_preds = [p for p in valid if p["_dimension"] == dim]
            xs = [p["_human_score"] for p in dim_preds]
            ys = [int(p["answer"])  for p in dim_preds]

            # 抖动避免重叠
            rng = np.random.default_rng(42)
            xs_j = np.array(xs) + rng.uniform(-0.2, 0.2, len(xs))
            ys_j = np.array(ys) + rng.uniform(-0.2, 0.2, len(ys))

            color = COLORS.get(name.lower().split("-")[0], "#555")
            ax.scatter(xs_j, ys_j, alpha=0.3, s=15, color=color)
            ax.plot([1, 7], [1, 7], "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0.5, 7.5)
            ax.set_xlabel("Human Score")
            ax.set_ylabel("Predicted Score")
            ax.set_title(f"{name} — {dim.capitalize()}", fontsize=10)

            if xs:
                tau = _tau(xs, ys)
                ax.text(0.05, 0.92, f"τ={tau:.3f}", transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", color="darkred")

    fig.suptitle("Predicted vs. Human Scores", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "scatter_pred_vs_human.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


def _tau(xs, ys) -> float:
    from itertools import combinations
    pairs = list(zip(xs, ys))
    c = d = 0
    for (x1, y1), (x2, y2) in combinations(pairs, 2):
        dx, dy = x1 - x2, y1 - y2
        if dx * dy > 0: c += 1
        elif dx * dy < 0: d += 1
    n = len(pairs)
    denom = n * (n - 1) / 2
    return (c - d) / denom if denom else float("nan")


# ── 3. 分数分布对比 ───────────────────────────────────────────────────────────

def plot_score_dist_comparison(preds_map: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    x = np.arange(1, 8)

    n_bars = 1 + len(preds_map)
    bar_width = 0.7 / n_bars
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_width

    for ax, dim in zip(axes, DIMS):
        first_preds = list(preds_map.values())[0]
        human_scores = [p["_human_score"] for p in first_preds if p["_dimension"] == dim]
        human_counts = np.array([human_scores.count(i) for i in range(1, 8)], dtype=float)
        if human_counts.sum() > 0:
            human_counts /= human_counts.sum()
        ax.bar(x + offsets[0], human_counts, width=bar_width, label="Human", color="gray", alpha=0.7, edgecolor="white")

        for i, (name, preds) in enumerate(preds_map.items(), 1):
            valid = [p for p in preds if p["answer"] not in ("-1",) and p["_dimension"] == dim]
            pred_scores = [int(p["answer"]) for p in valid]
            pred_counts = np.array([pred_scores.count(i) for i in range(1, 8)], dtype=float)
            if pred_counts.sum() > 0:
                pred_counts /= pred_counts.sum()
            color = COLORS.get(name.lower().split("-")[0], "#555")
            ax.bar(x + offsets[i], pred_counts, width=bar_width, label=name, color=color, alpha=0.7, edgecolor="white")

        ax.set_title(dim.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Proportion")
        ax.set_xticks(range(1, 8))
        ax.legend(fontsize=8)

    fig.suptitle("Score Distribution: Predicted vs. Human", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "score_distribution_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


# ── 4. 雷达图 ─────────────────────────────────────────────────────────────────

def plot_radar(tau_reports: list[dict]) -> None:
    categories = [d.capitalize() for d in DIMS]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})

    for report in tau_reports:
        name   = report.get("model", "?")
        values = [report["tau_by_dim"].get(d, 0) for d in DIMS]
        values += values[:1]
        color  = COLORS.get(name.lower().split("-")[0], None)
        ax.plot(angles, values, linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("Per-Dimension Tau — Radar", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "tau_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-only", action="store_true")
    args = parser.parse_args()

    # tau 曲线
    tau_logs = {}
    for name in ("llama", "qwen"):
        log = load_tau_log(name)
        if log:
            tau_logs[name] = log

    if tau_logs:
        plot_tau_curves(tau_logs)
    else:
        print("未找到 tau log（训练完成后再运行）")

    if args.tau_only:
        return

    # 推理结果可视化
    preds_map: dict[str, list[dict]] = {}
    for name in ("llama-judge", "qwen-judge"):
        preds = load_predictions(name)
        if preds:
            preds_map[name] = preds

    if preds_map:
        plot_scatter(preds_map)
        plot_score_dist_comparison(preds_map)

    # tau 雷达图
    tau_report_path = PRED_DIR / "tau_report.json"
    if tau_report_path.exists():
        with open(tau_report_path) as f:
            reports = json.load(f)
        plot_radar(reports)

    if not tau_logs and not preds_map:
        print("没有可用数据。请先完成训练和推理。")
    else:
        print(f"\n所有图表已保存至 {FIG_DIR}")


if __name__ == "__main__":
    main()
