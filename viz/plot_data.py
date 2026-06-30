"""
viz/plot_data.py

对训练数据做 EDA 可视化，输出到 viz/figures/data/。

用法:
    python viz/plot_data.py
"""

from __future__ import annotations
import json, pathlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data/processed"
FIG_DIR  = ROOT / "viz/figures/data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["faithfulness", "coverage", "naturalness", "coherence"]
LANG_LABELS = {
    "ar": "Arabic", "cs": "Czech",  "de": "German",  "es": "Spanish",
    "fr": "French", "hi": "Hindi",  "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "sv": "Swedish",
    "tr": "Turkish", "zh": "Chinese",
}


def load_all_records() -> list[dict]:
    records = []
    for split in ("train", "dev", "test"):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def plot_score_distribution(records: list[dict]) -> None:
    """每个维度的分数分布直方图（1–7）。"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, dim, color in zip(axes, DIMS, colors):
        scores = [r["meta"]["human_score"] for r in records if r["meta"]["dimension"] == dim]
        counts = [scores.count(i) for i in range(1, 8)]
        ax.bar(range(1, 8), counts, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(dim.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Score (1–7)")
        ax.set_xticks(range(1, 8))
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color="black", linestyle="--", linewidth=1.2, label=f"μ={mean_score:.2f}")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Count")
    fig.suptitle("Human Score Distribution by Dimension", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "score_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


def plot_language_distribution(records: list[dict]) -> None:
    """语言分布条形图。"""
    lang_counts: dict[str, int] = {}
    for r in records:
        if r["meta"]["dimension"] != "faithfulness":
            continue
        lang = r["meta"]["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    langs  = sorted(lang_counts, key=lang_counts.get, reverse=True)
    counts = [lang_counts[l] for l in langs]
    labels = [LANG_LABELS.get(l, l) for l in langs]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels[::-1], counts[::-1], color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Number of Samples")
    ax.set_title("Training Samples per Language", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = FIG_DIR / "language_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


def plot_score_heatmap(records: list[dict]) -> None:
    """语言 × 维度 的平均分热力图。"""
    langs = sorted({r["meta"]["language"] for r in records})
    matrix = np.zeros((len(langs), len(DIMS)))

    for i, lang in enumerate(langs):
        for j, dim in enumerate(DIMS):
            scores = [
                r["meta"]["human_score"]
                for r in records
                if r["meta"]["language"] == lang and r["meta"]["dimension"] == dim
            ]
            matrix[i, j] = np.mean(scores) if scores else np.nan

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=1, vmax=7, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Human Score")

    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([d.capitalize() for d in DIMS], fontsize=11)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([LANG_LABELS.get(l, l) for l in langs], fontsize=10)

    for i in range(len(langs)):
        for j in range(len(DIMS)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color="black" if val < 5 else "white")

    ax.set_title("Mean Human Score: Language × Dimension", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "score_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


def plot_model_scores(records: list[dict]) -> None:
    """各生成模型的平均人工分（按维度分组）。"""
    gen_models = sorted({r["meta"]["gen_model"] for r in records})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    colors = plt.cm.tab20.colors  # type: ignore

    for ax, dim in zip(axes.flat, DIMS):
        avgs = []
        for gm in gen_models:
            scores = [
                r["meta"]["human_score"]
                for r in records
                if r["meta"]["gen_model"] == gm and r["meta"]["dimension"] == dim
            ]
            avgs.append(np.mean(scores) if scores else 0)

        y_pos = np.arange(len(gen_models))
        bars  = ax.barh(y_pos, avgs, color=colors[:len(gen_models)], alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(gen_models, fontsize=8)
        ax.set_xlim(0, 7.5)
        ax.set_title(dim.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Mean Human Score per Generator Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "model_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {path}")


def main() -> None:
    records = load_all_records()
    if not records:
        print("找不到数据，先运行 python data/prepare.py")
        return

    print(f"加载 {len(records)} 条记录")
    plot_score_distribution(records)
    plot_language_distribution(records)
    plot_score_heatmap(records)
    plot_model_scores(records)
    print(f"\n所有图表已保存至 {FIG_DIR}")


if __name__ == "__main__":
    main()
