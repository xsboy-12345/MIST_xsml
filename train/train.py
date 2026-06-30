"""
train/train.py

LoRA / QLoRA fine-tuning for LLM-as-judge on xlsum task.
在 GPU 服务器上运行。

用法:
    python train/train.py --config train/configs/qwen_config.yaml
    python train/train.py --config train/configs/llama_config.yaml
    python train/train.py --config train/configs/qwen_config.yaml --dry-run
"""

from __future__ import annotations
import argparse, json, math, pathlib, re
import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data/processed"
LOG_DIR  = ROOT / "logs"


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def apply_chat_template(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def tokenize_fn(example: dict, tokenizer, max_len: int) -> dict:
    """只对 assistant 部分计算 loss。"""
    full = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_len,
        padding=False,
    )
    input_ids = full["input_ids"]

    prompt_ids = tokenizer.apply_chat_template(
        example["messages"][:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    n_prompt = len(prompt_ids)
    labels = [-100] * n_prompt + input_ids[n_prompt:]
    labels = labels[:max_len]

    return {
        "input_ids":      input_ids,
        "attention_mask": full["attention_mask"],
        "labels":         labels,
    }


# ── Tau 评估 ──────────────────────────────────────────────────────────────────

def parse_score(text: str) -> int | None:
    text = text.strip()
    if re.fullmatch(r"[1-7]", text):
        return int(text)
    m = re.search(r"\b([1-7])\b", text)
    return int(m.group(1)) if m else None


def kendall_tau(xs: list, ys: list) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None]
    if len(pairs) < 2:
        return float("nan")
    c = d = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dx = pairs[i][0] - pairs[j][0]
            dy = pairs[i][1] - pairs[j][1]
            if dx * dy > 0: c += 1
            elif dx * dy < 0: d += 1
    denom = len(pairs) * (len(pairs) - 1) / 2
    return (c - d) / denom if denom else float("nan")


def evaluate_tau(model, tokenizer, records: list[dict], max_len: int) -> dict:
    model.eval()
    DIMS = ["faithfulness", "coverage", "naturalness", "coherence"]
    preds: dict[str, list] = {d: [] for d in DIMS}
    refs:  dict[str, list] = {d: [] for d in DIMS}

    with torch.no_grad():
        for rec in records:
            prompt = tokenizer.apply_chat_template(
                rec["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=max_len)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc, max_new_tokens=4, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen = tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            dim   = rec["meta"]["dimension"]
            preds[dim].append(parse_score(gen))
            refs[dim].append(rec["meta"]["human_score"])

    taus = {d: kendall_tau(preds[d], refs[d]) for d in DIMS}
    valid = [v for v in taus.values() if not math.isnan(v)]
    taus["tau_average"] = sum(valid) / len(valid) if valid else float("nan")
    model.train()
    return taus


# ── Callback：每个 epoch 结束后算 tau ────────────────────────────────────────

class TauCallback(TrainerCallback):
    def __init__(self, tokenizer, dev_records: list[dict], max_len: int,
                 log_path: pathlib.Path):
        self._tokenizer   = tokenizer
        self._dev_records = dev_records
        self._max_len     = max_len
        self._log_path    = log_path
        self._history: list[dict] = []

    def on_epoch_end(self, args, state: TrainerState,
                     control: TrainerControl, model=None, **kwargs):
        epoch = round(state.epoch or 0)
        print(f"\n[Epoch {epoch}] 计算 dev tau...")
        taus = evaluate_tau(model, self._tokenizer,
                            self._dev_records, self._max_len)
        entry = {"epoch": epoch, **taus}
        self._history.append(entry)
        print("  " + "  ".join(
            f"{k}={v:.4f}" if not math.isnan(v) else f"{k}=nan"
            for k, v in entry.items()
        ))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "w") as f:
            json.dump(self._history, f, indent=2)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑 2 steps 验证流程")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    out_dir    = ROOT / cfg["output_dir"]
    max_len    = cfg.get("max_seq_length", 512)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model (QLoRA) ─────────────────────────────────────────────────────────
    bnb_cfg = None
    if cfg.get("use_4bit", False):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
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

    # ── dataset ───────────────────────────────────────────────────────────────
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    dev_records   = load_jsonl(DATA_DIR / "dev.jsonl")

    if args.dry_run:
        train_records = train_records[:16]
        dev_records   = dev_records[:8]

    def prepare(records: list[dict]) -> Dataset:
        ds = Dataset.from_list(records)
        ds = ds.map(lambda ex: apply_chat_template(ex, tokenizer))
        ds = ds.map(lambda ex: tokenize_fn(ex, tokenizer, max_len))
        keep = ["input_ids", "attention_mask", "labels"]
        ds = ds.remove_columns(
            [c for c in ds.column_names if c not in keep]
        )
        return ds

    train_ds = prepare(train_records)
    dev_ds   = prepare(dev_records)

    # ── training args ─────────────────────────────────────────────────────────
    model_short = pathlib.Path(cfg["output_dir"]).name
    log_path    = LOG_DIR / f"{model_short}_tau_log.json"

    train_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=1 if args.dry_run else cfg.get("num_epochs", 5),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_steps=int(cfg.get("warmup_steps", 50)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        eval_strategy=cfg.get("eval_strategy", "epoch"),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_total_limit=2,
        logging_steps=cfg.get("logging_steps", 20),
        fp16=False,
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        seed=cfg.get("seed", 42),
        report_to="none",
        max_steps=2 if args.dry_run else -1,
    )

    tau_cb = TauCallback(
        tokenizer=tokenizer,
        dev_records=dev_records,
        max_len=max_len,
        log_path=log_path,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100
        ),
        callbacks=[tau_cb],
    )

    # ── 单次调用，Trainer 内部处理所有 epoch ──────────────────────────────────
    trainer.train()

    # 保存最终 adapter
    model.save_pretrained(str(out_dir / "adapter_final"))
    tokenizer.save_pretrained(str(out_dir / "adapter_final"))
    print(f"\n模型已保存: {out_dir}/adapter_final")
    print(f"Tau 日志:   {log_path}")


if __name__ == "__main__":
    main()
