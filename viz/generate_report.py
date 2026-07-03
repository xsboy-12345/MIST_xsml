"""
viz/generate_report.py

Generate a PDF report summarizing the training approach, hyperparameter tuning,
and results for LLM-as-judge fine-tuning on the XL-Sum task.

Usage:
    python viz/generate_report.py
    python viz/generate_report.py --output my_report.pdf
"""

from __future__ import annotations
import argparse, json, math, pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT     = pathlib.Path(__file__).parent.parent
LOG_DIR  = ROOT / "logs"
PRED_DIR = ROOT / "eval/outputs"
FIG_DIR  = ROOT / "viz/figures/training"

DIMS = ["faithfulness", "coverage", "naturalness", "coherence"]

COLORS = {
    "llama_v1": "#A8C8E8",
    "llama_v2": "#4C72B0",
    "qwen_v1":  "#F5C28A",
    "qwen_v2":  "#DD8452",
}
LABELS = {
    "llama_v1": "Llama v1",
    "llama_v2": "Llama v2",
    "qwen_v1":  "Qwen v1",
    "qwen_v2":  "Qwen v2",
}

V1_PARAMS = {
    "model (Llama)":      "meta-llama/Llama-3.1-8B-Instruct",
    "model (Qwen)":       "Qwen/Qwen2.5-7B-Instruct",
    "lora_r":             "16",
    "lora_alpha":         "32",
    "lora_dropout":       "0.05",
    "target_modules":     "q, k, v, o",
    "learning_rate":      "2e-4",
    "weight_decay":       "0.01",
    "num_epochs":         "5",
    "warmup_steps":       "50",
    "quantization":       "4-bit NF4 (QLoRA)",
}

V2_PARAMS = {
    "model (Llama)":      "meta-llama/Llama-3.1-8B-Instruct",
    "model (Qwen)":       "Qwen/Qwen2.5-7B-Instruct",
    "lora_r":             "32 (Llama) / 16 (Qwen)",
    "lora_alpha":         "64 (Llama) / 32 (Qwen)",
    "lora_dropout":       "0.05 (Llama) / 0.1 (Qwen)",
    "target_modules":     "q, k, v, o, gate, up, down",
    "learning_rate":      "1e-4",
    "weight_decay":       "0.01 (Llama) / 0.05 (Qwen)",
    "num_epochs":         "8",
    "warmup_steps":       "100",
    "quantization":       "4-bit NF4 (QLoRA)",
}

REASONS = {
    "lora_r":         "Increased for Llama to boost model capacity and break the plateau",
    "lora_dropout":   "Increased for Qwen to reduce overfitting (dev→test gap)",
    "target_modules": "Added FFN layers (gate/up/down) for better generalization",
    "learning_rate":  "Halved to allow slower, more stable convergence",
    "weight_decay":   "Increased for Qwen to add regularization",
    "num_epochs":     "Extended since Qwen v1 was still improving at epoch 5",
    "warmup_steps":   "Doubled to stabilize early training with lower LR",
}


def load_tau_log(name: str) -> list[dict] | None:
    path = LOG_DIR / f"{name}_tau_log.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_tau_report() -> list[dict]:
    path = PRED_DIR / "tau_report.json"
    return json.loads(path.read_text()) if path.exists() else []


def title_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.75, "LLM-as-Judge Fine-Tuning", ha="center", va="center",
            fontsize=28, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.65, "Training Report: Hyperparameter Tuning & Results",
            ha="center", va="center", fontsize=16, color="#555555", transform=ax.transAxes)

    ax.text(0.5, 0.50, "Models:  Llama-3.1-8B-Instruct  ·  Qwen2.5-7B-Instruct",
            ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.43, "Task:  XL-Sum multilingual summarisation quality judgement",
            ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.36, "Method:  QLoRA (4-bit NF4)  +  LoRA adapter",
            ha="center", va="center", fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.29, "Metric:  Kendall's τ  ·  Ranking Accuracy",
            ha="center", va="center", fontsize=13, transform=ax.transAxes)

    ax.axhline(0.22, xmin=0.1, xmax=0.9, color="#CCCCCC", linewidth=1)
    ax.text(0.5, 0.16, "v1 → v2 key improvements:\n"
            "  • Fixed parse failures with LogitsProcessor (91% → 0% for Qwen)\n"
            "  • Extended training to 8 epochs\n"
            "  • Expanded LoRA target modules to include FFN layers\n"
            "  • Qwen v2 dev τ: 0.301 → 0.317",
            ha="center", va="center", fontsize=11, color="#333333",
            transform=ax.transAxes, linespacing=1.8)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def approach_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.95, "Training Approach", ha="center", va="top",
            fontsize=20, fontweight="bold", transform=ax.transAxes)

    sections = [
        ("Task", (
            "Fine-tune an LLM to act as a judge for multilingual text summarisation.\n"
            "The model receives a source article + candidate summary and outputs\n"
            "a quality score from 1–7 on one of four dimensions:\n"
            "Faithfulness, Coverage, Naturalness, Coherence."
        )),
        ("Data", (
            "XL-Sum dataset with human evaluation annotations.\n"
            "Split: train / dev / test  (~13,700 / 1,713 / 1,715 samples).\n"
            "Languages: 44 languages including Hindi, Arabic, Chinese, Swedish, etc."
        )),
        ("Method: QLoRA", (
            "Base models loaded in 4-bit NF4 quantisation to reduce GPU memory.\n"
            "LoRA adapters injected into attention and FFN layers (v2).\n"
            "Only adapter weights (~0.5–1% of parameters) are trained.\n"
            "Loss computed only on assistant response tokens."
        )),
        ("Output Constraint", (
            "Problem: models sometimes output text in the input language instead of a digit.\n"
            "Fix: LogitsProcessor that restricts generation to tokens '1'–'7' only.\n"
            "This eliminated parse failures entirely (Qwen: 91% → 0%)."
        )),
        ("Evaluation", (
            "• Kendall's τ: rank correlation between predicted and human scores (per sample)\n"
            "• Ranking Accuracy: fraction of model pairs correctly ordered by average score\n"
            "• Computed on dev set after each epoch (TauCallback) and on test set after training"
        )),
    ]

    y = 0.87
    for title, body in sections:
        ax.text(0.05, y, title, fontsize=13, fontweight="bold",
                transform=ax.transAxes, color="#2C5F8A")
        y -= 0.04
        ax.text(0.07, y, body, fontsize=10, transform=ax.transAxes,
                color="#333333", linespacing=1.6,
                verticalalignment="top")
        y -= 0.13

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def hyperparam_page(pdf: PdfPages) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.97, "Hyperparameter Tuning: v1 → v2", ha="center", va="top",
            fontsize=20, fontweight="bold", transform=ax.transAxes)

    keys = list(V1_PARAMS.keys())
    col_x = [0.03, 0.25, 0.50, 0.75]
    headers = ["Parameter", "v1", "v2", "Reason for change"]

    y_start = 0.88
    row_h   = 0.065

    for i, (hdr, x) in enumerate(zip(headers, col_x)):
        ax.text(x, y_start, hdr, fontsize=11, fontweight="bold",
                transform=ax.transAxes, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2C5F8A", edgecolor="none"))

    ax.axhline(y_start - 0.015, xmin=0.02, xmax=0.98,
               color="#2C5F8A", linewidth=1.5, transform=ax.transAxes)

    for row_i, key in enumerate(keys):
        y = y_start - row_h * (row_i + 1)
        bg = "#F5F8FF" if row_i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0.02, y - 0.01), 0.96, row_h - 0.005,
                                   transform=ax.transAxes, color=bg, zorder=0))

        v1_val = V1_PARAMS[key]
        v2_val = V2_PARAMS[key]
        changed = v1_val != v2_val
        color = "#C0392B" if changed else "#333333"

        ax.text(col_x[0], y + 0.015, key,       fontsize=9.5, transform=ax.transAxes, color="#333333")
        ax.text(col_x[1], y + 0.015, v1_val,    fontsize=9.5, transform=ax.transAxes, color="#333333")
        ax.text(col_x[2], y + 0.015, v2_val,    fontsize=9.5, transform=ax.transAxes, color=color,
                fontweight="bold" if changed else "normal")
        reason = REASONS.get(key, "—")
        ax.text(col_x[3], y + 0.015, reason,    fontsize=8.5, transform=ax.transAxes,
                color="#555555", wrap=True)

    ax.text(0.5, 0.02,
            "Red = changed from v1.  Changes target overfitting (Qwen) and capacity/stability (Llama).",
            ha="center", fontsize=9, color="#C0392B", transform=ax.transAxes)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def training_curves_page(pdf: PdfPages) -> None:
    logs = {n: load_tau_log(n) for n in ("llama_v1", "llama_v2", "qwen_v1", "qwen_v2")}
    logs = {k: v for k, v in logs.items() if v}
    if not logs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        taus   = [e.get("tau_average", float("nan")) for e in log]
        ls = "--" if "v1" in name else "-"
        ax.plot(epochs, taus, marker="o", label=LABELS[name],
                color=COLORS[name], linestyle=ls, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kendall's τ (average, dev set)")
    ax.set_title("Training Progress: Average Tau per Epoch", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax = axes[1]
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        for dim in DIMS:
            pass
    x = np.arange(len(DIMS))
    n = len(logs)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width
    for i, (name, log) in enumerate(logs.items()):
        last = log[-1]
        vals = [last.get(d, float("nan")) for d in DIMS]
        ls = "--" if "v1" in name else "-"
        ax.bar(x + offsets[i], vals, width, label=LABELS[name],
               color=COLORS[name], alpha=0.85, edgecolor="white",
               hatch="//" if "v1" in name else "")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DIMS])
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Per-Dimension Tau at Final Epoch (dev set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Dev Set Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def test_results_page(pdf: PdfPages) -> None:
    reports = load_tau_report()
    if not reports:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    metrics = DIMS + ["tau_average"]
    metric_labels = [d.capitalize() for d in DIMS] + ["Average"]
    x = np.arange(len(metrics))
    n = len(reports)
    width = 0.7 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    ax = axes[0]
    for i, report in enumerate(reports):
        name = report["model"]
        vals = [report["tau_by_dim"].get(d, 0) for d in DIMS] + [report["tau_average"]]
        color = COLORS.get(name, "#555")
        label = LABELS.get(name, name)
        ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Kendall's τ")
    ax.set_title("Test Set Tau: v1 vs v2", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    ax = axes[1]
    names  = [LABELS.get(r["model"], r["model"]) for r in reports]
    accs   = [r["ranking_accuracy"] for r in reports]
    colors = [COLORS.get(r["model"], "#555") for r in reports]
    bars = ax.bar(names, accs, color=colors, alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random baseline (0.5)")
    ax.set_ylabel("Ranking Accuracy")
    ax.set_title("Test Set Ranking Accuracy: v1 vs v2", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Test Set Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def findings_page(pdf: PdfPages) -> None:
    reports = load_tau_report()
    report_by_model = {r["model"]: r for r in reports}

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.97, "Key Findings & Conclusions", ha="center", va="top",
            fontsize=20, fontweight="bold", transform=ax.transAxes)

    findings = []

    # Build dynamic findings from data if available
    if "llama_v2" in report_by_model and "llama_v1" in report_by_model:
        lv1 = report_by_model["llama_v1"]
        lv2 = report_by_model["llama_v2"]
        findings.append((
            "Llama: τ improved but ranking accuracy dropped then recovered",
            f"v1 τ={lv1['tau_average']:.3f}, rank_acc={lv1['ranking_accuracy']:.3f}  →  "
            f"v2 τ={lv2['tau_average']:.3f}, rank_acc={lv2['ranking_accuracy']:.3f}\n"
            "v2 shows clear τ improvement (+32%) and ranking accuracy went from below random (47.8%) "
            "to above (62.5%).\nHowever, Llama v2 overfit after epoch 2 (dev τ peaked at 0.195); "
            "the final checkpoint was not the best."
        ))

    if "qwen_v2" in report_by_model and "qwen_v1" in report_by_model:
        qv1 = report_by_model["qwen_v1"]
        qv2 = report_by_model["qwen_v2"]
        findings.append((
            "Qwen: τ improved with a trade-off in ranking accuracy",
            f"v1 τ={qv1['tau_average']:.3f}, rank_acc={qv1['ranking_accuracy']:.3f}  →  "
            f"v2 τ={qv2['tau_average']:.3f}, rank_acc={qv2['ranking_accuracy']:.3f}\n"
            "Per-sample τ improved (+38%). Ranking accuracy decreased from 80% to 65%,\n"
            "suggesting the two metrics capture different aspects of judge quality."
        ))

    findings += [
        (
            "Parse failure fix was critical",
            "Qwen had 91% parse failures on test set due to multilingual continuation.\n"
            "Adding LogitsProcessor to restrict output to digits 1–7 eliminated failures entirely.\n"
            "Without this fix, test τ was artificially inflated by sampling bias."
        ),
        (
            "Large dev→test gap indicates generalisation challenges",
            "Qwen v2 dev τ = 0.317  vs  test τ = 0.109 — a significant drop.\n"
            "Likely causes: (1) domain shift across 44 languages, (2) overfitting to dev distribution.\n"
            "Increasing dropout and weight decay partially addressed this."
        ),
        (
            "Recommended next steps",
            "• Use best checkpoint (epoch 2) for Llama inference instead of final epoch\n"
            "• Try smaller LoRA rank (r=8) for Qwen to further reduce overfitting\n"
            "• Investigate per-language performance to identify weak language groups"
        ),
    ]

    y = 0.88
    for title, body in findings:
        ax.text(0.04, y, f"▶  {title}", fontsize=12, fontweight="bold",
                transform=ax.transAxes, color="#2C5F8A")
        y -= 0.04
        ax.text(0.06, y, body, fontsize=10, transform=ax.transAxes,
                color="#333333", linespacing=1.6, verticalalignment="top")
        y -= body.count("\n") * 0.045 + 0.07

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="viz/training_report.pdf")
    args = parser.parse_args()

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        title_page(pdf)
        approach_page(pdf)
        hyperparam_page(pdf)
        training_curves_page(pdf)
        test_results_page(pdf)
        findings_page(pdf)

    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
