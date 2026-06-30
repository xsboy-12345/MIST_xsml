"""
eval/compute_rouge.py

对 xlsum_humeval.json 里的摘要计算 ROUGE 分数。
参考摘要（reference）= 每条 task 里人工评分均值最高的模型的输出。

用法:
    pip install rouge-score
    python eval/compute_rouge.py
    python eval/compute_rouge.py --out eval/outputs/rouge_report.json
"""

from __future__ import annotations
import argparse, json, math, pathlib
from collections import defaultdict
from rouge_score import rouge_scorer

ROOT     = pathlib.Path(__file__).parent.parent
SRC      = ROOT / "data/raw/xlsum_humeval.json"
OUT_DIR  = ROOT / "eval/outputs"

DIMS     = ["faithfulness", "coverage", "naturalness", "coherence"]
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

LANG_NAMES = {
    "ar": "Arabic",    "cs": "Czech",      "de": "German",    "es": "Spanish",
    "fr": "French",    "hi": "Hindi",      "id": "Indonesian","it": "Italian",
    "ja": "Japanese",  "ko": "Korean",     "ru": "Russian",   "sv": "Swedish",
    "tr": "Turkish",   "zh": "Chinese",
}

# CJK 和阿拉伯语需要按字符分词
CHAR_LEVEL_LANGS = {"zh", "ja", "ko", "ar", "hi"}


def avg_human_score(rating: dict) -> float:
    scores = [rating[d] for d in DIMS if d in rating]
    return sum(scores) / len(scores) if scores else 0.0


def get_scorer(language: str) -> rouge_scorer.RougeScorer:
    use_stemmer = language not in CHAR_LEVEL_LANGS
    tokenizer = None
    if language in CHAR_LEVEL_LANGS:
        # 字符级 tokenizer：按字符切分
        class CharTokenizer:
            def tokenize(self, text):
                return list(text.replace(" ", ""))
        tokenizer = CharTokenizer()
    return rouge_scorer.RougeScorer(
        ROUGE_TYPES,
        use_stemmer=use_stemmer,
        tokenizer=tokenizer,
    )


def compute_rouge_scores(
    hypothesis: str,
    reference: str,
    language: str,
) -> dict[str, float]:
    scorer = get_scorer(language)
    scores = scorer.score(reference, hypothesis)
    return {rt: round(scores[rt].fmeasure, 4) for rt in ROUGE_TYPES}


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
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--out", default=str(OUT_DIR / "rouge_report.json"))
    args = parser.parse_args()

    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)

    # ── Step 1: 按 taskid 分组，找最高分参考摘要 ──────────────────────────────
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_task[item["taskid"]].append(item)

    # taskid 格式里包含语言码，但我们直接从 item["language"] 取
    references: dict[str, dict] = {}   # taskid -> best item
    for taskid, items in by_task.items():
        best = max(items, key=lambda x: avg_human_score(x["rating"]))
        references[taskid] = best

    print(f"共 {len(by_task)} 个 task，{len(data)} 条摘要")
    print(f"每条 task 选评分最高的作为 reference\n")

    # ── Step 2: 计算 ROUGE ────────────────────────────────────────────────────
    results: list[dict] = []
    skipped = 0

    for item in data:
        ref_item = references[item["taskid"]]

        # 自身作 reference 时跳过（避免完美分数拉偏统计）
        if item["model"] == ref_item["model"]:
            skipped += 1
            continue

        rouge = compute_rouge_scores(
            hypothesis=item["answer"].strip(),
            reference=ref_item["answer"].strip(),
            language=item["language"],
        )
        results.append({
            "taskid":      item["taskid"],
            "language":    item["language"],
            "gen_model":   item["model"],
            "ref_model":   ref_item["model"],
            "human_avg":   round(avg_human_score(item["rating"]), 3),
            "human_scores": item["rating"],
            **rouge,
        })

    print(f"计算了 {len(results)} 条（跳过 {skipped} 条 reference 自身）\n")

    # ── Step 3: 整体统计 ──────────────────────────────────────────────────────
    def mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else float("nan")

    overall = {
        rt: mean([r[rt] for r in results]) for rt in ROUGE_TYPES
    }
    print("=== 整体平均 ROUGE ===")
    for rt, v in overall.items():
        print(f"  {rt:8s}: {v:.4f}")

    # ── Step 4: ROUGE 与人工评分相关性 ───────────────────────────────────────
    human_avgs = [r["human_avg"] for r in results]
    corr = {}
    for rt in ROUGE_TYPES:
        rouge_vals = [r[rt] for r in results]
        corr[rt] = pearson(rouge_vals, human_avgs)

    print("\n=== ROUGE 与人工评分均值 Pearson 相关 ===")
    for rt, r in corr.items():
        print(f"  {rt:8s}: {r:.4f}")

    # ── Step 5: 按语言分组 ───────────────────────────────────────────────────
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)

    lang_stats = {}
    print("\n=== 按语言 rougeL 均值 ===")
    for lang in sorted(by_lang):
        items = by_lang[lang]
        rl = mean([x["rougeL"] for x in items])
        lang_stats[lang] = {rt: mean([x[rt] for x in items]) for rt in ROUGE_TYPES}
        print(f"  {LANG_NAMES.get(lang, lang):12s} ({lang}): {rl:.4f}")

    # ── Step 6: 按模型分组 ───────────────────────────────────────────────────
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model[r["gen_model"]].append(r)

    model_stats = {}
    print("\n=== 按模型 rougeL 均值（排序）===")
    model_rl = {
        m: mean([x["rougeL"] for x in items])
        for m, items in by_model.items()
    }
    for m, rl in sorted(model_rl.items(), key=lambda x: -x[1]):
        model_stats[m] = {rt: mean([x[rt] for x in by_model[m]]) for rt in ROUGE_TYPES}
        print(f"  {m:25s}: {rl:.4f}")

    # ── 写报告 ────────────────────────────────────────────────────────────────
    report = {
        "total_evaluated": len(results),
        "skipped_as_reference": skipped,
        "overall": overall,
        "correlation_with_human": corr,
        "by_language": lang_stats,
        "by_model": model_stats,
        "records": results,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告写入: {out_path}")


if __name__ == "__main__":
    main()
