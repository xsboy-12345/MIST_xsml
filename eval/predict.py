"""
eval/predict.py

用训练好的 LoRA adapter 在 test.jsonl 上推理，输出预测分数。
同时以 submissions_judge_xlsum 兼容格式输出（可直接送 compute_tau）。

用法:
    python eval/predict.py --adapter outputs/llama/adapter_final --model-name llama-judge
    python eval/predict.py --adapter outputs/qwen/adapter_final  --model-name qwen-judge
"""

from __future__ import annotations
import argparse, json, re, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data/processed"
OUT_DIR  = ROOT / "eval/outputs"


def parse_score(text: str) -> str:
    """返回 '1'–'7' 字符串，解析不到则返回 '-1'。"""
    text = text.strip()
    if re.fullmatch(r"[1-7]", text):
        return text
    m = re.search(r"\b([1-7])\b", text)
    return m.group(1) if m else "-1"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",    required=True, help="adapter 目录，包含 adapter_config.json")
    parser.add_argument("--model-name", required=True, help="输出文件名（无扩展名）")
    parser.add_argument("--split",      default="test", choices=["dev", "test"])
    parser.add_argument("--limit",      type=int, default=None, help="只推理前 N 条（调试）")
    parser.add_argument("--use-4bit",   action="store_true")
    args = parser.parse_args()

    adapter_path = pathlib.Path(args.adapter)

    # ── 加载 tokenizer + base model + adapter ────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if args.use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    base_model_name = json.loads((adapter_path / "adapter_config.json").read_text())["base_model_name_or_path"]
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    # ── 推理 ─────────────────────────────────────────────────────────────────
    records = load_jsonl(DATA_DIR / f"{args.split}.jsonl")
    if args.limit:
        records = records[: args.limit]

    results: list[dict] = []
    for i, rec in enumerate(records, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(records)}]")

        prompt = tokenizer.apply_chat_template(
            rec["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_score = parse_score(generated)

        meta = rec["meta"]
        # submissions_judge_xlsum 兼容格式
        taskid = f"judge_{meta['dimension']}_{meta['taskid']}_{meta['gen_model']}"
        results.append({
            "taskid":      taskid,
            "answer":      pred_score,
            "raw_output":  generated,
            "tokens":      {"input_tokens": enc["input_ids"].shape[1], "output_tokens": 1},
            # 额外保留便于 compute_tau 直接使用
            "_dimension":  meta["dimension"],
            "_language":   meta["language"],
            "_gen_model":  meta["gen_model"],
            "_human_score": meta["human_score"],
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total   = len(results)
    invalid = sum(1 for r in results if r["answer"] == "-1")
    print(f"\n推理完成: {total} 条，解析失败: {invalid} 条")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
