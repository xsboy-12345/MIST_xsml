"""
eval/compute_tau.py

Read predict.py output and compute multiple evaluation metrics in parallel:

  Ranking metrics (measure ordering agreement with human scores):
    - Kendall's τ      : fraction of concordant pairs minus discordant pairs
    - Ranking accuracy : pairwise model-level ranking agreement

  Score consistency metrics (measure how close predicted scores are to human scores):
    - Pearson r        : linear correlation between predicted and human scores
    - MAE              : mean absolute error (average score gap, in Likert points)

Usage:
    python eval/compute_tau.py --pred eval/outputs/llama-judge.json
    python eval/compute_tau.py --pred eval/outputs/llama-judge.json eval/outputs/qwen-judge.json
"""

from __future__ import annotations
import argparse, json, math, pathlib
from collections import defaultdict
from itertools import combinations

ROOT     = pathlib.Path(__file__).parent.parent
OUT_DIR  = ROOT / "eval/outputs"
DIMS     = ["faithfulness", "coverage", "naturalness", "coherence"]


# ── Ranking metrics ───────────────────────────────────────────────────────────

def kendall_tau(xs: list[float], ys: list[float]) -> float:
    pairs = list(zip(xs, ys))
    concordant = discordant = 0
    for (x1, y1), (x2, y2) in combinations(pairs, 2):
        dx, dy = x1 - x2, y1 - y2
        if dx * dy > 0:
            concordant += 1
        elif dx * dy < 0:
            discordant += 1
    n = len(pairs)
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else float("nan")


def ranking_accuracy(model_avg_scores: dict[str, float], human_avg_scores: dict[str, float]) -> float:
    """Fraction of model pairs where predicted ranking matches human ranking."""
    models = sorted(set(model_avg_scores) & set(human_avg_scores))
    correct = total = 0
    for m1, m2 in combinations(models, 2):
        total += 1
        pred_order  = model_avg_scores[m1] > model_avg_scores[m2]
        human_order = human_avg_scores[m1] > human_avg_scores[m2]
        if pred_order == human_order:
            correct += 1
    return correct / total if total else float("nan")


# ── Score consistency metrics ─────────────────────────────────────────────────

def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Linear correlation between predicted and human scores."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / denom if denom else float("nan")


def mean_absolute_error(xs: list[float], ys: list[float]) -> float:
    """Average absolute difference between predicted and human scores (in Likert points)."""
    if not xs:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


# ── Main report ───────────────────────────────────────────────────────────────

def load_predictions(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_report(preds: list[dict], model_name: str) -> dict:
    valid = [p for p in preds if p["answer"] not in ("-1", None)]

    preds_by_dim: dict[str, list[int]]   = defaultdict(list)
    refs_by_dim:  dict[str, list[int]]   = defaultdict(list)
    for p in valid:
        dim   = p["_dimension"]
        human = p["_human_score"]
        pred  = int(p["answer"])
        preds_by_dim[dim].append(pred)
        refs_by_dim[dim].append(human)

    # ── Kendall's τ ───────────────────────────────────────────────────────────
    tau_by_dim = {
        dim: round(kendall_tau(preds_by_dim[dim], refs_by_dim[dim]), 6)
        for dim in DIMS
    }
    valid_taus = [v for v in tau_by_dim.values() if not math.isnan(v)]
    tau_avg = round(sum(valid_taus) / len(valid_taus), 6) if valid_taus else float("nan")

    # ── Pearson r ─────────────────────────────────────────────────────────────
    pearson_by_dim = {
        dim: round(pearson_r(preds_by_dim[dim], refs_by_dim[dim]), 6)
        for dim in DIMS
    }
    valid_pearson = [v for v in pearson_by_dim.values() if not math.isnan(v)]
    pearson_avg = round(sum(valid_pearson) / len(valid_pearson), 6) if valid_pearson else float("nan")

    # ── MAE ───────────────────────────────────────────────────────────────────
    mae_by_dim = {
        dim: round(mean_absolute_error(preds_by_dim[dim], refs_by_dim[dim]), 6)
        for dim in DIMS
    }
    valid_mae = [v for v in mae_by_dim.values() if not math.isnan(v)]
    mae_avg = round(sum(valid_mae) / len(valid_mae), 6) if valid_mae else float("nan")

    # ── Ranking accuracy ──────────────────────────────────────────────────────
    pred_per_genmodel:  dict[str, list[int]] = defaultdict(list)
    human_per_genmodel: dict[str, list[int]] = defaultdict(list)
    for p in valid:
        gm = p["_gen_model"]
        pred_per_genmodel[gm].append(int(p["answer"]))
        human_per_genmodel[gm].append(p["_human_score"])

    pred_avg  = {gm: sum(v) / len(v) for gm, v in pred_per_genmodel.items()}
    human_avg = {gm: sum(v) / len(v) for gm, v in human_per_genmodel.items()}
    rank_acc  = round(ranking_accuracy(pred_avg, human_avg), 6)

    parse_failures = sum(1 for p in preds if p["answer"] == "-1")

    return {
        "model":            model_name,
        "total":            len(preds),
        "valid":            len(valid),
        "parse_failures":   parse_failures,
        # ranking metrics
        "tau_average":      tau_avg,
        "tau_by_dim":       tau_by_dim,
        "ranking_accuracy": rank_acc,
        # score consistency metrics
        "pearson_average":  pearson_avg,
        "pearson_by_dim":   pearson_by_dim,
        "mae_average":      mae_avg,
        "mae_by_dim":       mae_by_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True, help="JSON output files from predict.py")
    args = parser.parse_args()

    all_reports: list[dict] = []
    for pred_path in args.pred:
        p = pathlib.Path(pred_path)
        model_name = p.stem
        preds = load_predictions(p)
        report = compute_report(preds, model_name)
        all_reports.append(report)

        print(f"\n=== {model_name} ===")
        print(f"  valid:            {report['valid']} / {report['total']}")
        print(f"  parse_failures:   {report['parse_failures']}")
        print(f"  --- Ranking metrics ---")
        print(f"  tau_average:      {report['tau_average']:.4f}")
        for dim, v in report["tau_by_dim"].items():
            print(f"    {dim:15s}: {v:.4f}")
        print(f"  ranking_accuracy: {report['ranking_accuracy']:.4f}")
        print(f"  --- Score consistency metrics ---")
        print(f"  pearson_average:  {report['pearson_average']:.4f}")
        for dim, v in report["pearson_by_dim"].items():
            print(f"    {dim:15s}: {v:.4f}")
        print(f"  mae_average:      {report['mae_average']:.4f}")
        for dim, v in report["mae_by_dim"].items():
            print(f"    {dim:15s}: {v:.4f}")

    out_path = OUT_DIR / "tau_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
