"""
viz/plot_training.py

Visualize training progress (supports v1/v2 version comparison):
  1. Kendall's tau vs. epoch curve
  2. Test-set v1 vs v2 tau bar chart
  3. Predicted score vs. human score scatter plot
  4. Predicted score distribution vs. human distribution
  5. Per-dimension tau radar chart

Usage:
    python viz/plot_training.py
    python viz/plot_training.py --tau-only
"""

from __future__ import annotations
import argparse, json, math, pathlib
import matplotlib.pyplot as plt
import numpy as np

ROOT     = pathlib.Path(__file__).parent.parent
LOG_DIR  = ROOT / "logs"
PRED_DIR = ROOT / "eval/outputs"
FIG_DIR  = ROOT / "viz/figures/training"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["faithfulness", "coverage", "naturalness", "coherence"]

COLORS = {
    "llama_v1":        "#A8C8E8",
    "llama_v2":        "#4C72B0",
    "llama_v3":        "#1A3A6B",
    "qwen_v1":         "#F5C28A",
    "qwen_v2":         "#DD8452",
    "qwen_v3":         "#8B3A0F",
    "llama-judge-v1":  "#A8C8E8",
    "llama-judge-v2":  "#4C72B0",
    "llama-judge-v3":  "#1A3A6B",
    "qwen-judge-v1":   "#F5C28A",
    "qwen-judge-v2":   "#DD8452",
    "qwen-judge-v3":   "#8B3A0F",
    "llama-zero-shot": "#B0B8C0",
    "qwen-zero-shot":  "#C8B8A0",
}

LABELS = {
    "llama_v1":        "Llama v1",
    "llama_v2":        "Llama v2",
    "llama_v3":        "Llama v3",
    "qwen_v1":         "Qwen v1",
    "qwen_v2":         "Qwen v2",
    "qwen_v3":         "Qwen v3",
    "llama-judge-v1":  "Llama v1",
    "llama-judge-v2":  "Llama v2",
    "llama-judge-v3":  "Llama v3",
    "qwen-judge-v1":   "Qwen v1",
    "qwen-judge-v2":   "Qwen v2",
    "qwen-judge-v3":   "Qwen v3",
    "llama-zero-shot": "Llama Zero-shot",
    "qwen-zero-shot":  "Qwen Zero-shot",
}

LINESTYLES = {"v1": "--", "v2": "-", "v3": "-.", "zero": ":"}

# ── Known-issue registry ───────────────────────────────────────────────────────
# Per-epoch dev-tau logs (logs/*_tau_log.json) affected by a known train/eval
# precision mismatch: TauCallback evaluates the 4-bit quantized training model,
# while eval/predict.py (used for tau_report.json / test-set numbers) evaluates
# an unquantized reload by default. Once generation is hard-restricted to 7
# candidate tokens, this precision gap is enough to flip the argmax, so these
# curves read as near-random even though the corresponding test-set results in
# eval/outputs/tau_report.json are normal. See README § Version History & Known
# Issues. Only affects the *_tau_log.json curves — tau_report.json entries for
# the same versions are NOT flagged here because they come from predict.py.
KNOWN_ISSUES = {
    "llama_v3": "dev-τ log unreliable (train/eval precision mismatch) — see README",
    "qwen_v3":  "dev-τ log unreliable (train/eval precision mismatch) — see README",
}


def get_color(name: str) -> str:
    return COLORS.get(name, "#555555")


def get_label(name: str) -> str:
    return LABELS.get(name, name)


def get_linestyle(name: str) -> str:
    if "zero" in name:  return LINESTYLES["zero"]
    if "v3"   in name:  return LINESTYLES["v3"]
    if "v1"   in name:  return LINESTYLES["v1"]
    return LINESTYLES["v2"]


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


# ── 1. Tau training curves ────────────────────────────────────────────────────

def plot_tau_curves(logs: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    flagged_present = [n for n in logs if n in KNOWN_ISSUES]

    ax = axes[0]
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        taus   = [e.get("tau_average", float("nan")) for e in log]
        flagged = name in KNOWN_ISSUES
        label = ("⚠ " if flagged else "") + get_label(name)
        ax.plot(epochs, taus, marker="x" if flagged else "o", label=label,
                color=get_color(name), linestyle=get_linestyle(name),
                linewidth=2, alpha=0.45 if flagged else 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kendall's τ (average)")
    ax.set_title("Average Tau vs. Epoch", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax = axes[1]
    x = np.arange(len(DIMS))
    n = len(logs)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    for i, (name, log) in enumerate(logs.items()):
        last = log[-1]
        vals = [last.get(d, float("nan")) for d in DIMS]
        flagged = name in KNOWN_ISSUES
        label = ("⚠ " if flagged else "") + get_label(name)
        ax.bar(x + offsets[i], vals, width, label=label,
               color=get_color(name), alpha=0.45 if flagged else 0.85,
               edgecolor="white", hatch="///" if flagged else None)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIMS])
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Per-Dimension Tau (Final Epoch)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Training Progress: Tau Correlation with Human Ratings", fontsize=13, fontweight="bold")
    if flagged_present:
        fig.text(0.5, -0.02,
                  "⚠ Faded/hatched series (v3): dev-τ log unreliable due to a known train/eval precision "
                  "mismatch — see README § Version History & Known Issues. Test-set results are unaffected.",
                  ha="center", fontsize=8.5, color="#8B0000", style="italic")
    plt.tight_layout()
    path = FIG_DIR / "tau_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 2. Test-set version comparison ───────────────────────────────────────────
# Built from eval/outputs/tau_report.json (predict.py output) — these numbers
# are NOT affected by the v3 dev-τ known issue (see KNOWN_ISSUES above), so
# nothing here needs to be flagged.

def plot_version_comparison(reports: list[dict]) -> None:
    metrics = DIMS + ["tau_average"]
    metric_labels = [d.capitalize() for d in DIMS] + ["Average"]
    x = np.arange(len(metrics))
    n = len(reports)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: tau per dimension
    ax = axes[0]
    for i, report in enumerate(reports):
        name = report["model"]
        vals = [report["tau_by_dim"].get(d, 0) for d in DIMS] + [report["tau_average"]]
        ax.bar(x + offsets[i], vals, width, label=get_label(name),
               color=get_color(name), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Test Tau by Version", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # right: ranking accuracy
    ax = axes[1]
    names  = [get_label(r["model"]) for r in reports]
    accs   = [r["ranking_accuracy"] for r in reports]
    colors = [get_color(r["model"]) for r in reports]
    bars = ax.bar(names, accs, color=colors, alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_ylabel("Ranking Accuracy")
    ax.set_title("Ranking Accuracy by Version", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

    fig.suptitle("Test Set Performance Across Versions (zero-shot / v1 / v2 / v3)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "version_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 3. Predicted vs. human score scatter plot ─────────────────────────────────

def plot_scatter(preds_map: dict[str, list[dict]], missing: list[str] | None = None) -> None:
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
            ys = [int(p["answer"]) for p in dim_preds]

            rng = np.random.default_rng(42)
            xs_j = np.array(xs) + rng.uniform(-0.2, 0.2, len(xs))
            ys_j = np.array(ys) + rng.uniform(-0.2, 0.2, len(ys))

            ax.scatter(xs_j, ys_j, alpha=0.3, s=15, color=get_color(name))
            ax.plot([1, 7], [1, 7], "k--", linewidth=0.8, alpha=0.5)
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0.5, 7.5)
            ax.set_xlabel("Human Score")
            ax.set_ylabel("Predicted Score")
            ax.set_title(f"{get_label(name)} — {dim.capitalize()}", fontsize=10)

            if xs:
                tau = _tau(xs, ys)
                ax.text(0.05, 0.92, f"τ={tau:.3f}", transform=ax.transAxes,
                        fontsize=9, verticalalignment="top", color="darkred")

    fig.suptitle("Predicted vs. Human Scores", fontsize=13, fontweight="bold")
    if missing:
        fig.text(0.5, -0.01,
                  f"⚠ Not shown (raw predictions unavailable locally): {', '.join(get_label(m) for m in missing)}",
                  ha="center", fontsize=8.5, color="#8B0000", style="italic")
    plt.tight_layout()
    path = FIG_DIR / "scatter_pred_vs_human.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


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


# ── 4. Score distribution comparison ─────────────────────────────────────────

def plot_score_dist_comparison(preds_map: dict[str, list[dict]], missing: list[str] | None = None) -> None:
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
        ax.bar(x + offsets[0], human_counts, width=bar_width,
               label="Human", color="gray", alpha=0.7, edgecolor="white")

        for idx, (name, preds) in enumerate(preds_map.items(), 1):
            valid = [p for p in preds if p["answer"] not in ("-1",) and p["_dimension"] == dim]
            pred_scores = [int(p["answer"]) for p in valid]
            pred_counts = np.array([pred_scores.count(i) for i in range(1, 8)], dtype=float)
            if pred_counts.sum() > 0:
                pred_counts /= pred_counts.sum()
            ax.bar(x + offsets[idx], pred_counts, width=bar_width,
                   label=get_label(name), color=get_color(name), alpha=0.7, edgecolor="white")

        ax.set_title(dim.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Proportion")
        ax.set_xticks(range(1, 8))
        ax.legend(fontsize=7)

    fig.suptitle("Score Distribution: Predicted vs. Human", fontsize=13, fontweight="bold")
    if missing:
        fig.text(0.5, -0.02,
                  f"⚠ Not shown (raw predictions unavailable locally): {', '.join(get_label(m) for m in missing)}",
                  ha="center", fontsize=8.5, color="#8B0000", style="italic")
    plt.tight_layout()
    path = FIG_DIR / "score_distribution_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 5. Radar chart ────────────────────────────────────────────────────────────

def plot_radar(tau_reports: list[dict]) -> None:
    categories = [d.capitalize() for d in DIMS]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for report in tau_reports:
        name   = report.get("model", "?")
        values = [report["tau_by_dim"].get(d, 0) for d in DIMS]
        values += values[:1]
        ls = get_linestyle(name)
        ax.plot(angles, values, linewidth=2, label=get_label(name),
                color=get_color(name), linestyle=ls)
        ax.fill(angles, values, alpha=0.08, color=get_color(name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(-0.1, 0.35)
    ax.set_title("Per-Dimension Tau — Radar", fontsize=12, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / "tau_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── 6. Score consistency: Pearson r + MAE ─────────────────────────────────────

def plot_score_consistency(reports: list[dict]) -> None:
    """Pearson r and MAE per dimension and model — score consistency metrics."""
    metrics = DIMS + ["average"]
    metric_labels = [d.capitalize() for d in DIMS] + ["Average"]
    x = np.arange(len(metrics))
    n = len(reports)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: Pearson r
    ax = axes[0]
    for i, report in enumerate(reports):
        name = report["model"]
        if "pearson_by_dim" not in report:
            continue
        vals = [report["pearson_by_dim"].get(d, float("nan")) for d in DIMS] + [report.get("pearson_average", float("nan"))]
        ax.bar(x + offsets[i], vals, width, label=get_label(name),
               color=get_color(name), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Pearson r")
    ax.set_title("Pearson Correlation (score consistency)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # right: MAE (lower is better)
    ax = axes[1]
    for i, report in enumerate(reports):
        name = report["model"]
        if "mae_by_dim" not in report:
            continue
        vals = [report["mae_by_dim"].get(d, float("nan")) for d in DIMS] + [report.get("mae_average", float("nan"))]
        ax.bar(x + offsets[i], vals, width, label=get_label(name),
               color=get_color(name), alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("MAE (Likert points) ↓")
    ax.set_title("Mean Absolute Error (lower = better)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Score Consistency Metrics: Pearson r and MAE vs. Human Scores",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "score_consistency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-only", action="store_true")
    args = parser.parse_args()

    # tau training curves
    tau_logs = {}
    for name in ("llama_v1", "llama_v2", "llama_v3", "qwen_v1", "qwen_v2", "qwen_v3"):
        log = load_tau_log(name)
        if log:
            tau_logs[name] = log

    if tau_logs:
        plot_tau_curves(tau_logs)
    else:
        print("No tau log found.")

    if args.tau_only:
        return

    # inference result visualizations
    all_pred_names = ("llama-zero-shot", "llama-judge-v1", "llama-judge-v2", "llama-judge-v3",
                       "qwen-zero-shot",  "qwen-judge-v1",  "qwen-judge-v2",  "qwen-judge-v3")
    preds_map: dict[str, list[dict]] = {}
    for name in all_pred_names:
        preds = load_predictions(name)
        if preds:
            preds_map[name] = preds
    missing_preds = [n for n in all_pred_names if n not in preds_map]
    if missing_preds:
        print(f"Note: raw predictions unavailable locally, skipping in scatter/distribution plots: {missing_preds}")

    if preds_map:
        plot_scatter(preds_map, missing=missing_preds)
        plot_score_dist_comparison(preds_map, missing=missing_preds)

    # tau report (version comparison + radar)
    tau_report_path = PRED_DIR / "tau_report.json"
    if tau_report_path.exists():
        with open(tau_report_path) as f:
            reports = json.load(f)
        if reports:
            plot_version_comparison(reports)
            plot_radar(reports)
            plot_score_consistency(reports)

    if not tau_logs and not preds_map:
        print("No data available. Run training and inference first.")
    else:
        print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
