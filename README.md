# XLSum Judge Fine-Tuning

Fine-tune **Llama-3.1-8B-Instruct** and **Qwen2.5-7B-Instruct** as multilingual summary quality judges using human evaluation data from the WMT25-MIST shared task.

## Task

Given a set of reviews and a generated summary, the model scores the summary on four dimensions (1–7 Likert scale):

| Dimension | Description |
|-----------|-------------|
| Faithfulness | Is all information in the summary supported by the reviews? |
| Coverage | Are the most important review points covered? |
| Naturalness | Is the summary fluent and natural in the target language? |
| Coherence | Is the summary logically sound and internally consistent? |

Training signal comes from human annotations. The quality of fine-tuning is measured by **Kendall's τ** correlation and **ranking accuracy** against human scores.

## Data

Source: `data/humeval/xlsum.json` from WMT25-MIST (4,284 records, 14 languages, 17 generator models).

Each record is expanded into 4 training examples (one per dimension), giving **17,136 samples** total split 80/10/10.

| Split | Size |
|-------|------|
| train | 13,708 |
| dev   | 1,713  |
| test  | 1,715  |

Languages: Arabic, Czech, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Russian, Swedish, Turkish, Chinese

## Project Structure

```
xlsum/
├── data/
│   ├── prepare.py              # Build train/dev/test from raw humeval data
│   └── processed/              # train.jsonl / dev.jsonl / test.jsonl
├── train/
│   ├── train.py                # QLoRA fine-tuning (runs on GPU server)
│   └── configs/
│       ├── llama_config.yaml   # Llama-3.1-8B-Instruct config
│       └── qwen_config.yaml    # Qwen2.5-7B-Instruct config
├── eval/
│   ├── predict.py              # Run fine-tuned model on test set
│   └── compute_tau.py          # Compute τ and ranking accuracy vs human ratings
├── viz/
│   ├── plot_data.py            # EDA: score distributions, language heatmap
│   └── plot_training.py        # Training curves, scatter plots, radar chart
└── requirements.txt
```

## Quickstart

### 1. Prepare data

```bash
python data/prepare.py
```

### 2. Visualize training data

```bash
pip install matplotlib numpy seaborn
python viz/plot_data.py
# → viz/figures/data/
```

### 3. Fine-tune on GPU server

```bash
# Clone on the server
git clone <repo-url> && cd xlsum
pip install -r requirements.txt

# Fine-tune Llama
python train/train.py --config train/configs/llama_config.yaml

# Fine-tune Qwen
python train/train.py --config train/configs/qwen_config.yaml

# Dry run (smoke test, 2 steps only)
python train/train.py --config train/configs/llama_config.yaml --dry-run
```

Training logs τ per epoch to `logs/{model}_tau_log.json` automatically.

### 4. Evaluate

```bash
python eval/predict.py --adapter outputs/llama/adapter_final --model-name llama-judge
python eval/predict.py --adapter outputs/qwen/adapter_final  --model-name qwen-judge
python eval/compute_tau.py --pred eval/outputs/llama-judge.json eval/outputs/qwen-judge.json
```

### 5. Visualize results (local)

```bash
# Pull results from server first
rsync -av cqa1:~/xlsum/logs/      logs/
rsync -av cqa1:~/xlsum/eval/outputs/ eval/outputs/

python viz/plot_training.py
# → viz/figures/training/
```

## Training Format

Each training sample uses chat format:

```
System: You are an expert multilingual evaluation assistant. ...
User:   Score the following summary based on FAITHFULNESS (1-7).
        ...
        Language: Turkish
        Reviews:
        [1] Delicious food, great atmosphere...
        [2] ...
        Summary: Bu restoranın atmosferi temiz ve sıcak...
        Faithfulness score (1-7):
Assistant: 5
```

## Model Config

Both models use QLoRA (4-bit) + LoRA adapters:

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o_proj |
| Quantization | 4-bit NF4 |
| Learning rate | 2e-4 |
| Epochs | 5 |
| Batch size (effective) | 16 |

## Evaluation Metrics

- **Kendall's τ** per dimension and average — correlation with human scores
- **Ranking accuracy** — pairwise ranking agreement across generator models

Reference (GPT-4.1 zero-shot, from WMT25-MIST):

| Metric | GPT-4.1 |
|--------|---------|
| τ average | 0.020 |
| Ranking accuracy | 1.000 |

## Requirements

```
torch>=2.1.0
transformers>=4.45.0
peft>=0.12.0
datasets>=2.20.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
pyyaml>=6.0
matplotlib>=3.8.0  # local visualization only
```
