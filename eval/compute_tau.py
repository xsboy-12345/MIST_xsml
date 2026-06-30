"""
eval/compute_tau.py

读取 predict.py 的输出，计算 Kendall tau 和 ranking_accuracy，
与 humeval_aggregated/judge_xlsum.json 格式对齐。

用法:
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
    """所有模型对之间，排名方向一致的比例。"""
    models = sorted(set(model_avg_scores) & set(human_avg_scores))
    correct = total = 0
    for m1, m2 in combinations(models, 2):
        total += 1
        pred_order  = model_avg_scores[m1] > model_avg_scores[m2]
        human_order = human_avg_scores[m1] > human_avg_scores[m2]
        if pred_order == human_order:
            correct += 1
    return correct / total if total else float("nan")


def load_predictions(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_report(preds: list[dict], model_name: str) -> dict:
    valid = [p for p in preds if p["answer"] not in ("-1", None)]

    # tau per dimension
    preds_by_dim:  dict[str, list[int]] = defaultdict(list)
    refs_by_dim:   dict[str, list[int]] = defaultdict(list)
    for p in valid:
        dim   = p["_dimension"]
        human = p["_human_score"]
        pred  = int(p["answer"])
        preds_by_dim[dim].append(pred)
        refs_by_dim[dim].append(human)

    tau_by_dim = {
        dim: round(kendall_tau(preds_by_dim[dim], refs_by_dim[dim]), 6)
        for dim in DIMS
    }
    valid_taus = [v for v in tau_by_dim.values() if not math.isnan(v)]
    tau_avg = round(sum(valid_taus) / len(valid_taus), 6) if valid_taus else float("nan")

    # ranking_accuracy: 按 gen_model 聚合平均分
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
        "tau_average":      tau_avg,
        "tau_by_dim":       tau_by_dim,
        "ranking_accuracy": rank_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True, help="predict.py 输出的 JSON 文件")
    args = parser.parse_args()

    all_reports: list[dict] = []
    for pred_path in args.pred:
        p = pathlib.Path(pred_path)
        model_name = p.stem
        preds = load_predictions(p)
        report = compute_report(preds, model_name)
        all_reports.append(report)

        print(f"\n=== {model_name} ===")
        print(f"  有效条数:       {report['valid']} / {report['total']}")
        print(f"  解析失败:       {report['parse_failures']}")
        print(f"  tau_average:    {report['tau_average']:.4f}")
        for dim, tau in report["tau_by_dim"].items():
            print(f"    {dim:15s}: {tau:.4f}")
        print(f"  ranking_accuracy: {report['ranking_accuracy']:.4f}")

    out_path = OUT_DIR / "tau_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"\n报告写入: {out_path}")


if __name__ == "__main__":
    main()
