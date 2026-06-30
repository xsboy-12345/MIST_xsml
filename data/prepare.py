"""
data/prepare.py

将 xlsum_humeval.json 展开为 judge 训练格式，输出 train/dev/test.jsonl。
每条 humeval 记录 × 4 维度 = 4 条训练样本，assistant 只输出一个数字（1-7）。

用法:
    python data/prepare.py
    python data/prepare.py --min-score 0   # 不过滤，保留全部 17136 条
"""

from __future__ import annotations
import argparse, json, pathlib, random

ROOT      = pathlib.Path(__file__).parent.parent
SRC       = ROOT / "data/raw/xlsum_humeval.json"
OUT_DIR   = ROOT / "data/processed"

DIMENSIONS = ["faithfulness", "coverage", "naturalness", "coherence"]

LANG_NAMES = {
    "ar": "Egyptian Arabic", "cs": "Czech",    "de": "German",
    "es": "Spanish",         "fr": "French",   "hi": "Hindi",
    "id": "Indonesian",      "it": "Italian",  "ja": "Japanese",
    "ko": "Korean",          "ru": "Russian",  "sv": "Swedish",
    "tr": "Turkish",         "zh": "Simplified Chinese",
}

SYSTEM = (
    "You are an expert multilingual evaluation assistant. "
    "Given a set of reviews and a summary, score the summary on the specified criterion "
    "using a scale from 1 to 7. Output only the integer score."
)

DIM_PROMPTS: dict[str, str] = {
    "faithfulness": (
        "Score the following summary based on FAITHFULNESS (1-7).\n"
        "Evaluate whether all information in the summary can be traced back to the reviews.\n"
        "7 = fully supported  5 = mostly supported  3 = mostly unsupported  1 = entirely fabricated\n\n"
        "Language: {language}\n"
        "Reviews:\n{reviews}\n\n"
        "Summary: {summary}\n\n"
        "Faithfulness score (1-7):"
    ),
    "coverage": (
        "Score the following summary based on COVERAGE (1-7).\n"
        "Evaluate whether the most important points from the reviews are covered.\n"
        "7 = all key points covered  5 = ~2/3 covered  3 = ~1/3 covered  1 = none covered\n\n"
        "Language: {language}\n"
        "Reviews:\n{reviews}\n\n"
        "Summary: {summary}\n\n"
        "Coverage score (1-7):"
    ),
    "naturalness": (
        "Score the following summary based on NATURALNESS (1-7).\n"
        "Evaluate fluency and naturalness in the target language.\n"
        "7 = native-like  5 = minor disfluencies  3 = highly disfluent  1 = incomprehensible\n\n"
        "Language: {language}\n"
        "Reviews:\n{reviews}\n\n"
        "Summary: {summary}\n\n"
        "Naturalness score (1-7):"
    ),
    "coherence": (
        "Score the following summary based on COHERENCE (1-7).\n"
        "Evaluate logical soundness and internal consistency.\n"
        "7 = logically sound  5 = minor gaps  3 = poor flow  1 = incoherent\n\n"
        "Language: {language}\n"
        "Reviews:\n{reviews}\n\n"
        "Summary: {summary}\n\n"
        "Coherence score (1-7):"
    ),
}


def format_reviews(inputs: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {inp['translated'].strip()}"
        for i, inp in enumerate(inputs, 1)
    )


def to_training_records(item: dict) -> list[dict]:
    lang_name = LANG_NAMES.get(item["language"], item["language"])
    reviews   = format_reviews(item["inputs"])
    records   = []
    for dim in DIMENSIONS:
        score = item["rating"].get(dim)
        if score is None:
            continue
        user_msg = DIM_PROMPTS[dim].format(
            language=lang_name,
            reviews=reviews,
            summary=item["answer"].strip(),
        )
        records.append({
            "messages": [
                {"role": "system",    "content": SYSTEM},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": str(score)},
            ],
            "meta": {
                "taskid":      item["taskid"],
                "language":    item["language"],
                "gen_model":   item["model"],
                "dimension":   dim,
                "human_score": score,
            },
        })
    return records


def split(records: list, seed: int = 42) -> tuple[list, list, list]:
    rng = random.Random(seed)
    data = records[:]
    rng.shuffle(data)
    n  = len(data)
    i1 = int(n * 0.8)
    i2 = i1 + int(n * 0.1)
    return data[:i1], data[i1:i2], data[i2:]


def write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=1.0,
                        help="过滤人工评分均值低于此值的记录（默认不过滤）")
    args = parser.parse_args()

    with open(SRC, encoding="utf-8") as f:
        raw = json.load(f)

    if args.min_score > 1.0:
        def avg(item):
            vals = [item["rating"][d] for d in DIMENSIONS if d in item["rating"]]
            return sum(vals) / len(vals) if vals else 0
        raw = [x for x in raw if avg(x) >= args.min_score]

    all_records: list[dict] = []
    for item in raw:
        all_records.extend(to_training_records(item))

    train, dev, test = split(all_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "dev.jsonl",   dev)
    write_jsonl(OUT_DIR / "test.jsonl",  test)

    print(f"总计: {len(all_records)} 条  (来自 {len(raw)} 条 humeval × {len(DIMENSIONS)} 维度)")
    print(f"  train : {len(train)}")
    print(f"  dev   : {len(dev)}")
    print(f"  test  : {len(test)}")

    # 写统计
    lang_dist: dict[str, int] = {}
    dim_dist:  dict[str, int] = {}
    score_dist: dict[int, int] = {}
    for r in all_records:
        m = r["meta"]
        lang_dist[m["language"]]   = lang_dist.get(m["language"], 0) + 1
        dim_dist[m["dimension"]]   = dim_dist.get(m["dimension"], 0) + 1
        s = m["human_score"]
        score_dist[s] = score_dist.get(s, 0) + 1

    stats = {
        "total": len(all_records),
        "train": len(train), "dev": len(dev), "test": len(test),
        "by_language":  lang_dist,
        "by_dimension": dim_dist,
        "score_distribution": {str(k): v for k, v in sorted(score_dist.items())},
    }
    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计写入: {stats_path}")


if __name__ == "__main__":
    main()
