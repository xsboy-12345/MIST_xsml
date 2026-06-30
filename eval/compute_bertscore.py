"""
eval/compute_bertscore.py

对 xlsum_humeval.json 里的摘要计算 BERTScore（F1）。
参考摘要 = 每条 task 里人工评分均值最高的模型输出（同 compute_rouge.py）。

BERTScore 优于 ROUGE 的地方：
  - 基于语义相似度，不依赖词表重叠
  - 使用多语言预训练模型，天然支持 14 种语言
  - 和人工判断的相关性更强

用法:
    pip install bert-score
    python eval/compute_bertscore.py
    python eval/compute_bertscore.py --model microsoft/mdeberta-v3-base
"""

from __future__ import annotations
import argparse, json, pathlib
from collections import defaultdict
from bert_score import score as bert_score

ROOT    = pathlib.Path(__file__).parent.parent
SRC     = ROOT / "data/raw/xlsum_humeval.json"
OUT_DIR = ROOT / "eval/outputs"

DIMS = ["faithfulness", "coverage", "naturalness", "coherence"]

LANG_NAMES = {
    "ar": "Arabic",    "cs": "Czech",      "de": "German",    "es": "Spanish",
    "fr": "French",    "hi": "Hindi",      "id": "Indonesian","it": "Italian",
    "ja": "Japanese",  "ko": "Korean",     "ru": "Russian",   "sv": "Swedish",
    "tr": "Turkish",   "zh": "Chinese",
}

# bert-score 接受的语言代码（部分语言需要映射）
BERT_LANG_MAP = {
    "zh": "zh",  "ja": "ja",  "ko": "ko",  "ar": "ar",
    "hi": "hi",  "de": "de",  "fr": "fr",  "es": "es",
    "it": "it",  "ru": "ru",  "tr": "tr",  "cs": "cs",
    "sv": "sv",  "id": "id",
}


def avg_human_score(rating: dict) -> float:
    scores = [rating[d] for d in DIMS if d in rating]
    return sum(scores) / len(scores) if scores else 0.0


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (
        sum((x - mx) ** 2 for x in xs) *
        sum((y - my) ** 2 for y in ys)
    ) ** 0.5
    return round(num / den, 4) if den else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   default=str(SRC))
    parser.add_argument("--out",   default=str(OUT_DIR / "bertscore_report.json"))
    parser.add_argument("--model", default=None,
                        help="BERTScore 模型，默认按语言自动选择。"
                             "推荐多语言模型: microsoft/mdeberta-v3-base")
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)

    # ── Step 1: 按 taskid 找最高分参考摘要 ───────────────────────────────────
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_task[item["taskid"]].append(item)

    references: dict[str, dict] = {
        tid: max(items, key=lambda x: avg_human_score(x["rating"]))
        for tid, items in by_task.items()
    }

    # ── Step 2: 剔除自身，按语言分批计算 BERTScore ────────────────────────────
    # bert-score 按语言批量计算更高效
    by_lang: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for item in data:
        ref = references[item["taskid"]]
        if item["model"] == ref["model"]:
            skipped += 1
            continue
        by_lang[item["language"]].append({
            "item": item,
            "ref":  ref,
        })

    print(f"共 {len(data)} 条，跳过参考自身 {skipped} 条")
    print(f"按 {len(by_lang)} 种语言分批计算 BERTScore...\n")

    all_results: list[dict] = []

    for lang, pairs in sorted(by_lang.items()):
        hypotheses = [p["item"]["answer"].strip() for p in pairs]
        refs_text  = [p["ref"]["answer"].strip()  for p in pairs]
        bert_lang  = BERT_LANG_MAP.get(lang, "en")

        print(f"  [{lang}] {LANG_NAMES.get(lang, lang):12s} {len(pairs):4d} 条 ...", end=" ", flush=True)

        try:
            _, _, F1 = bert_score(
                hypotheses,
                refs_text,
                lang=bert_lang if args.model is None else None,
                model_type=args.model,
                verbose=False,
                device="cpu",   # 本地 Mac 用 CPU；服务器上改为 "cuda"
            )
            f1_scores = F1.tolist()
        except Exception as e:
            print(f"失败: {e}")
            f1_scores = [None] * len(pairs)

        for p, f1 in zip(pairs, f1_scores):
            item = p["item"]
            all_results.append({
                "taskid":     item["taskid"],
                "language":   lang,
                "gen_model":  item["model"],
                "ref_model":  p["ref"]["model"],
                "human_avg":  round(avg_human_score(item["rating"]), 3),
                "bertscore_f1": round(f1, 4) if f1 is not None else None,
            })

        valid = [r["bertscore_f1"] for r in all_results
                 if r["language"] == lang and r["bertscore_f1"] is not None]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"avg F1 = {avg:.4f}")

    # ── Step 3: 整体统计 ──────────────────────────────────────────────────────
    valid_all = [r["bertscore_f1"] for r in all_results if r["bertscore_f1"] is not None]
    overall_avg = round(sum(valid_all) / len(valid_all), 4) if valid_all else float("nan")

    human_avgs  = [r["human_avg"]      for r in all_results if r["bertscore_f1"] is not None]
    bert_scores = [r["bertscore_f1"]   for r in all_results if r["bertscore_f1"] is not None]
    corr = pearson(bert_scores, human_avgs)

    print(f"\n=== 整体 BERTScore F1 均值: {overall_avg:.4f} ===")
    print(f"=== BERTScore 与人工评分 Pearson 相关: {corr:.4f} ===")

    # ── Step 4: 按语言 / 按模型汇总 ──────────────────────────────────────────
    by_lang_agg: dict[str, list[float]] = defaultdict(list)
    by_model_agg: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        if r["bertscore_f1"] is not None:
            by_lang_agg[r["language"]].append(r["bertscore_f1"])
            by_model_agg[r["gen_model"]].append(r["bertscore_f1"])

    lang_stats  = {l: round(sum(v)/len(v), 4) for l, v in by_lang_agg.items()}
    model_stats = {m: round(sum(v)/len(v), 4) for m, v in by_model_agg.items()}

    print("\n=== 按模型 BERTScore F1（排序）===")
    for m, s in sorted(model_stats.items(), key=lambda x: -x[1]):
        print(f"  {m:25s}: {s:.4f}")

    # ── 写报告 ────────────────────────────────────────────────────────────────
    report = {
        "total_evaluated": len(all_results),
        "skipped_as_reference": skipped,
        "model_used": args.model or "lang-specific default",
        "overall_bertscore_f1": overall_avg,
        "correlation_with_human": corr,
        "by_language": lang_stats,
        "by_model": model_stats,
        "records": all_results,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告写入: {args.out}")


if __name__ == "__main__":
    main()
