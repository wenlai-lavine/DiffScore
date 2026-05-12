# DiffScore

DiffScore is a text evaluation framework built on Masked Diffusion LLMs (MDLLM). It leverages the ELBO framework of models like LLaDA to provide multi-granularity quality assessment with bidirectional context, eliminating the positional bias inherent in autoregressive evaluation methods.

DiffScore supports both **zero-shot** (DiffScore-Zero) and **fine-tuned** (DiffScore-FT) evaluation. The fine-tuning follows a single-step SFT approach analogous to BARTScore: fine-tune on task-specific corpora (e.g., CNN/DailyMail) so the model assigns higher reconstruction probability to higher-quality text.

## Key Features

- **Four Scoring Configurations**: Marginal (fluency), Conditional (faithfulness), Reverse (coverage), Bidirectional (comprehensive)
- **Multi-Timestep Quality Profile**: Decomposes evaluation into fine-grained temporal quality signals
- **PMI Decomposition**: Orthogonal fluency-relevance separation
- **Token-Level Scoring**: Per-token quality heatmaps for interpretability
- **Structured Masking**: Entity/function/content-word masking for diagnostic analysis
- **16 Benchmarks (BartScore-aligned)**: WMT (7 lang pairs), SUM (6 datasets), D2T (3 datasets)

## Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Download spaCy model (for structured masking)
python -m spacy download en_core_web_sm

# Prepare datasets
download dataset from bartscore repository
```

**Requirements**: Python 3.8+, PyTorch 2.1+, CUDA-capable GPU (A100 recommended for 8B models).

## Project Structure

```
DiffScore/
├── configs/default.yaml            # Hyperparameters & benchmark config
├── datasets/
│   ├── WMT/{de-en,...,zh-en}/      # 7 WMT lang pairs (data.pkl each)
│   ├── SUM/{SummEval,...}/         # 6 SUM datasets (data.pkl each)
│   └── D2T/{BAGEL,SFHOT,SFRES}/   # 3 D2T datasets (data.pkl each)
├── sft/                             # Single-step SFT pipeline
│   ├── prepare_data.py             # Load HF corpora → chat-template SFT format
│   ├── train_sft.py                # MDLM masked reconstruction fine-tuning
│   └── score_finetuned.py          # Score benchmarks with DiffScore-FT
├── src/
│   ├── model/                      # MDLLM wrappers (LLaDA, Dream)
│   ├── scoring/                    # DiffScore core (4 configs, PMI, profile)
│   ├── baselines/                  # BARTScore, BERTScore, BLEU, ROUGE, etc.
│   ├── data/
│   │   ├── bartscore_benchmark.py  # Unified WMT/SUM/D2T loader (primary)
│   │   ├── summeval.py             # Legacy loader
│   │   └── adversarial.py          # Adversarial test construction
│   ├── evaluation/
│   │   ├── bartscore_eval.py       # BartScore-aligned eval metrics
│   │   ├── correlation.py          # Generic correlation utilities
│   │   ├── bootstrap.py            # Bootstrap CI
│   │   └── williams_test.py        # Williams significance test
│   ├── analysis/                   # Position bias, direction consistency
│   └── visualization/              # Heatmaps, profile plots
├── dllm/                            # DLLM toolkit (MDLM training library)
├── experiments/
│   ├── exp0_sanity_check.py        # Pre-validation
│   ├── exp1_meta_evaluation.py     # Main meta-evaluation (BartScore-aligned)
│   ├── exp2-8                      # Profile, PMI, ablation, efficiency, etc.
├── main.py
└── requirements.txt
```

## Quick Start

### Score a single text pair

```python
from src.model import MDLMWrapper
from src.scoring import DiffScorer

model = MDLMWrapper("GSAI-ML/LLaDA-8B-Instruct", device="cuda")
scorer = DiffScorer(model, K=20, T=10, scoring_mode="mean_lp")

template = {"prefix": "Document: ", "mid": "\n\nSummary: ", "suffix": ""}
result = scorer.score_conditional(
    source="The United Nations held an emergency session...",
    candidate="The UN met to discuss the ongoing crisis.",
    prompt_template=template,
)
print(f"Score: {result.scalar:.4f}")
print(f"Profile: {result.profile}")
```

### Run a specific experiment

```bash
# Experiment 0: Sanity Check (run first!)
python experiments/exp0_sanity_check.py \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --device cuda --K 20 --T 10 --max_samples 100

# Experiment 1: Meta-evaluation on all BartScore-aligned benchmarks
python experiments/exp1_meta_evaluation.py --K 20 --T 10

# Experiment 1: Run on specific task category
python experiments/exp1_meta_evaluation.py --tasks WMT --K 20 --T 10
python experiments/exp1_meta_evaluation.py --tasks SUM --K 20 --T 10
python experiments/exp1_meta_evaluation.py --tasks D2T --K 20 --T 10

# Experiment 1: Run on specific benchmarks
python experiments/exp1_meta_evaluation.py \
    --benchmarks sum_SummEval,d2t_BAGEL,wmt_de-en --K 20 --T 10

# Experiment 2: Quality profile analysis
python experiments/exp2_quality_profile.py --K 20 --T 10

# Experiment 4: Direction consistency (now fixed)
python experiments/exp4_direction_consistency.py --K 20 --T 10 --n_pairs 200

# Experiment 5: Ablation study
python experiments/exp5_ablation.py --ablations K T prompt masking scoring_mode
```

### Run all experiments

```bash
# Full pipeline (adjust DEVICE and MAX_SAMPLES as needed)
DEVICE=cuda MAX_SAMPLES=200 bash scripts/run_all.sh
```

## Fine-Tuning (DiffScore-FT)

DiffScore-FT uses single-step SFT on task-specific corpora, directly analogous to BARTScore using BART-large-CNN.

```bash
# Step 1: Prepare CNN/DailyMail data
python sft/prepare_data.py --corpus cnn_dailymail

# Step 2: Fine-tune (single GPU with LoRA)
accelerate launch --num_processes 1 \
    sft/train_sft.py \
    --dataset_args sft/data/sft_cnn_dailymail \
    --load_preprocessed_data True \
    --load_in_4bit True --lora True

# Step 3: Score benchmarks with fine-tuned model
python sft/score_finetuned.py \
    --adapter_path .models/DiffScore-FT/checkpoint-final \
    --benchmarks sum_SummEval wmt_zh-en d2t_BAGEL
```

See [finetuning.md](finetuning.md) for full details.

## Key Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `K` | 20 | Total Monte Carlo samples |
| `T` | 10 | Number of discrete timesteps |
| `bi_alpha` | 0.7 | Weight for bidirectional configuration (cond vs rev) |
| `scoring_mode` | `mean_lp` | Scoring mode: `elbo`, `mean_lp`, or `weighted` |
| `exclude_t_1` | `true` | Exclude t=1.0 (full masking) timestep |
| `batch_size` | 4-8 | GPU batch size for MC samples |

**Scoring modes**:
- `mean_lp` (recommended): Average log-prob at masked positions, more stable than ELBO
- `elbo`: Faithful ELBO with 1/t weighting, can introduce variance at small t
- `weighted`: Learned per-timestep weights from annotated data

**Efficiency presets**:
- Fast: `K=5, T=5` (5 forward passes/sample)
- Standard: `K=20, T=10` (20 forward passes/sample)
- Full: `K=50, T=10` (50 forward passes/sample)

## Supported Benchmarks (BartScore-aligned)

Evaluation is organized into three task categories, fully aligned with the BartScore paper (Yuan et al., 2021) for fair comparison.

### WMT — Machine Translation (Preference-based Kendall τ)

| Lang Pair | # Segments | Evaluation |
|:----------|:-----------|:-----------|
| de-en | 85,365 | Concordance Kendall τ |
| fi-en | — | Concordance Kendall τ |
| gu-en | — | Concordance Kendall τ |
| kk-en | — | Concordance Kendall τ |
| lt-en | — | Concordance Kendall τ |
| ru-en | — | Concordance Kendall τ |
| zh-en | — | Concordance Kendall τ |

WMT uses WMT-19 DARR (Direct Assessment with Relative Ranking). Each data point is a segment with two system outputs where human annotators chose which is better. **No numeric human score is provided** — evaluation measures how often the metric agrees with human preference via concordance-based Kendall τ = (conc - disc) / (conc + disc).

### SUM — Text Summarization (Per-document Correlation)

| Dataset | # Docs | Human Metrics | Evaluation | Ref |
|:--------|:-------|:--------------|:-----------|:----|
| SummEval | 100 (×16 sys) | coherence, consistency, fluency, relevance | Spearman/Kendall (per-doc avg) | multi |
| Newsroom | 60 (×7 sys) | coherence, fluency, informativeness, relevance | Spearman/Kendall (per-doc avg) | single |
| REALSumm | 100 | litepyramid_recall | Spearman/Kendall (per-doc avg) | single |
| Rank19 | 373 | fact | Accuracy (correct vs incorrect) | single |
| QAGS_CNN | 235 | fact | Pearson | single |
| QAGS_XSUM | — | fact | Pearson | single |

### D2T — Data-to-Text (Document-level Correlation)

| Dataset | # Docs | Human Metrics | # Refs |
|:--------|:-------|:--------------|:-------|
| BAGEL | 404 | informativeness, naturalness, quality | 18 |
| SFHOT | — | informativeness, naturalness, quality | — |
| SFRES | — | informativeness, naturalness, quality | — |

### Data Format

All benchmark data is stored as pickle files in `datasets/{task}/{name}/data.pkl`, following the BartScore data format:

- **WMT**: `{doc_id: {src, ref, better: {sys_name, sys, scores}, worse: {sys_name, sys, scores}}}`
- **SUM**: `{doc_id: {src, ref_summ/ref_summs, sys_summs: {sys_name: {sys_summ, scores}}}}`
- **D2T**: `{doc_id: {src, sys_summ, ref_summs: [...], scores: {human_metrics...}}}`

## Outputs

All experiment outputs are saved to `outputs/`:
- `outputs/exp*/` — per-experiment results (JSON), figures (PDF), logs
- Results include correlation tables, bootstrap CIs, and significance tests
- Figures include heatmaps, profile curves, Pareto plots, and bias distributions

## Supported Models

| Model | Size | Type | Role |
|:------|:-----|:-----|:-----|
| LLaDA-8B-Instruct | 8B | MDLLM | Primary model |
| LLaDA-8B-Base | 8B | MDLLM | Ablation (instruction tuning effect) |
| Dream-7B | 7B | MDLLM | Ablation (model-agnostic validation) |
| BART-large-CNN | 400M | AR seq2seq | BARTScore baseline |
