"""
train/train.py

LoRA / QLoRA fine-tuning for LLM-as-judge on xlsum task.
在 GPU 服务器上运行。

用法:
    python train/train.py --config train/configs/llama_config.yaml
    python train/train.py --config train/configs/qwen_config.yaml
    python train/train.py --config train/configs/llama_config.yaml --dry-run
"""

from __future__ import annotations
import argparse, json, math, pathlib, time
import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT      = pathlib.Path(__file__).parent.parent
DATA_DIR  = ROOT / "data/processed"
LOG_DIR   = ROOT / "logs"


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def tokenize_fn(example: dict, tokenizer, max_len: int) -> dict:
    """只对 assistant 部分计算 loss（labels 中 user/system 部分置为 -100）。"""
    full = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    input_ids = full["input_ids"]

    # 找 assistant 开始的位置：用 chat_template 的 generation_prompt 做切割
    prompt_only = tokenizer.apply_chat_template(
        example["messages"][:-1],       # 去掉 assistant 那条
        tokenize=True,
        add_generation_prompt=True,
    )
    n_prompt = len(prompt_only)

    labels = [-100] * n_prompt + input_ids[n_prompt:]
    labels = labels[:max_len]

    return {
        "input_ids":      input_ids,
        "attention_mask": full["attention_mask"],
        "labels":         labels,
    }


# ── 评估：在 dev 集上算 tau ──────────────────────────────────────────────────

def compute_tau(predictions: list[int | None], references: list[int]) -> float:
    """Kendall's tau-b，忽略预测为 None 的条目。"""
    pairs = [(p, r) for p, r in zip(predictions, references) if p is not None]
    if len(pairs) < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    n = len(pairs)
    denom = math.sqrt(n * (n - 1) / 2)
    return (concordant - discordant) / denom if denom else float("nan")


def parse_score(text: str) -> int | None:
    import re
    text = text.strip()
    if re.fullmatch(r"[1-7]", text):
        return int(text)
    m = re.search(r"\b([1-7])\b", text)
    return int(m.group(1)) if m else None


def evaluate_on_dev(model, tokenizer, dev_records: list[dict], max_len: int, device) -> dict:
    model.eval()
    preds_by_dim: dict[str, list] = {d: [] for d in ["faithfulness", "coverage", "naturalness", "coherence"]}
    refs_by_dim:  dict[str, list] = {d: [] for d in ["faithfulness", "coverage", "naturalness", "coherence"]}

    with torch.no_grad():
        for rec in dev_records:
            prompt = tokenizer.apply_chat_template(
                rec["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred  = parse_score(generated)
            dim   = rec["meta"]["dimension"]
            human = rec["meta"]["human_score"]
            preds_by_dim[dim].append(pred)
            refs_by_dim[dim].append(human)

    taus = {dim: compute_tau(preds_by_dim[dim], refs_by_dim[dim]) for dim in preds_by_dim}
    valid = [v for v in taus.values() if not math.isnan(v)]
    taus["tau_average"] = sum(valid) / len(valid) if valid else float("nan")
    return taus


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只跑 1 step 验证流程")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    out_dir    = ROOT / cfg["output_dir"]
    max_len    = cfg.get("max_seq_length", 1024)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model (QLoRA) ────────────────────────────────────────────────────────
    bnb_cfg = None
    if cfg.get("use_4bit", False):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── dataset ──────────────────────────────────────────────────────────────
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    dev_records   = load_jsonl(DATA_DIR / "dev.jsonl")

    if args.dry_run:
        train_records = train_records[:32]
        dev_records   = dev_records[:16]

    def prepare(records):
        ds = Dataset.from_list(records)
        ds = ds.map(lambda ex: apply_chat_template(ex, tokenizer))
        ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer, max_len))
        ds = ds.remove_columns([c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")])
        return ds

    train_ds = prepare(train_records)
    dev_ds   = prepare(dev_records)

    # ── training args ────────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1 if args.dry_run else cfg.get("num_epochs", 5),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_steps=int(cfg.get("warmup_steps", 50)),
        weight_decay=cfg.get("weight_decay", 0.01),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        logging_steps=cfg.get("logging_steps", 20),
        fp16=False,
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        seed=cfg.get("seed", 42),
        report_to="none",
        max_steps=2 if args.dry_run else -1,
    )

    # ── trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
    )

    # ── 训练 + 每 epoch 算 tau ───────────────────────────────────────────────
    tau_log: list[dict] = []
    device = next(model.parameters()).device

    for epoch in range(1, (2 if args.dry_run else cfg.get("num_epochs", 5)) + 1):
        trainer.train(resume_from_checkpoint=None if epoch == 1 else True)

        print(f"\n[Epoch {epoch}] 计算 dev tau...")
        taus = evaluate_on_dev(model, tokenizer, dev_records[:200], max_len, device)
        entry = {"epoch": epoch, **taus}
        tau_log.append(entry)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in entry.items()))

        log_path = LOG_DIR / f"{pathlib.Path(cfg['output_dir']).name}_tau_log.json"
        with open(log_path, "w") as f:
            json.dump(tau_log, f, indent=2)

    # 保存最终 adapter
    model.save_pretrained(str(out_dir / "adapter_final"))
    tokenizer.save_pretrained(str(out_dir / "adapter_final"))
    print(f"\n模型已保存: {out_dir}/adapter_final")


if __name__ == "__main__":
    main()
