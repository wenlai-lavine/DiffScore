#!/bin/bash
# DiffScore: Run all experiments sequentially
# Usage: bash scripts/run_all.sh [--device cuda:0] [--max_samples N]

set -e

DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
K="${K:-50}"
T="${T:-10}"
SEEDS="${SEEDS:-42,123,456}"

COMMON_ARGS="--model_name $MODEL --device $DEVICE --dtype $DTYPE"
if [ -n "$MAX_SAMPLES" ]; then
    COMMON_ARGS="$COMMON_ARGS --max_samples $MAX_SAMPLES"
fi

echo "============================================================"
echo "DiffScore: Full Experiment Pipeline"
echo "Model: $MODEL | Device: $DEVICE | K=$K T=$T"
echo "Max samples: ${MAX_SAMPLES:-all} | Seeds: $SEEDS"
echo "============================================================"

# Experiment 0: Sanity Check
echo -e "\n>>> Experiment 0: Sanity Check"
python experiments/exp0_sanity_check.py $COMMON_ARGS \
    --output_dir outputs/exp0_sanity_check \
    --K $K --T $T \
    --max_samples 300

# Experiment 1: Meta-Evaluation (with all baselines)
echo -e "\n>>> Experiment 1: Meta-Evaluation (SummEval)"
python experiments/exp1_meta_evaluation.py $COMMON_ARGS \
    --output_dir outputs/exp1_meta_eval \
    --benchmarks summeval \
    --K $K --T $T \
    --seeds $SEEDS \
    --scoring_mode elbo \
    --enable_heavy_baselines

echo -e "\n>>> Experiment 1: Meta-Evaluation (WMT22 zh-en)"
python experiments/exp1_meta_evaluation.py $COMMON_ARGS \
    --output_dir outputs/exp1_meta_eval_wmt_zhen \
    --benchmarks wmt22_zhen \
    --K $K --T $T \
    --seeds $SEEDS \
    --scoring_mode elbo \
    --max_samples 2000 \
    --enable_heavy_baselines

echo -e "\n>>> Experiment 1: Meta-Evaluation (WebNLG)"
python experiments/exp1_meta_evaluation.py $COMMON_ARGS \
    --output_dir outputs/exp1_meta_eval_webnlg \
    --benchmarks webnlg \
    --K $K --T $T \
    --seeds $SEEDS \
    --scoring_mode elbo \
    --enable_heavy_baselines \
    --webnlg_path /home/ubuntu-1/shenyingli/DiffScore/datasets/webnlg_2020.json

echo -e "\n>>> Experiment 1: Meta-Evaluation (TopicalChat)"
python experiments/exp1_meta_evaluation.py $COMMON_ARGS \
    --output_dir outputs/exp1_meta_eval_topicalchat \
    --benchmarks topicalchat \
    --K $K --T $T \
    --seeds $SEEDS \
    --scoring_mode elbo \
    --enable_heavy_baselines \
    --topicalchat_path /home/ubuntu-1/shenyingli/DiffScore/datasets/tc_usr_data.json

# Experiment 2: Quality Profile
echo -e "\n>>> Experiment 2: Quality Profile Analysis"
python experiments/exp2_quality_profile.py $COMMON_ARGS \
    --output_dir outputs/exp2_quality_profile \
    --K $K --T $T

# Experiment 3: PMI Decoupling
echo -e "\n>>> Experiment 3: PMI Decoupling"
python experiments/exp3_pmi_decoupling.py $COMMON_ARGS \
    --output_dir outputs/exp3_pmi \
    --K $K --T $T

# Experiment 4: Direction Consistency
echo -e "\n>>> Experiment 4: Bidirectional Reasoning Consistency"
python experiments/exp4_direction_consistency.py $COMMON_ARGS \
    --output_dir outputs/exp4_direction \
    --n_pairs 200

# Experiment 5: Ablation
echo -e "\n>>> Experiment 5: Ablation Study"
python experiments/exp5_ablation.py $COMMON_ARGS \
    --output_dir outputs/exp5_ablation \
    --ablations K T prompt masking scoring_mode

# Experiment 6: Efficiency
echo -e "\n>>> Experiment 6: Efficiency Analysis"
python experiments/exp6_efficiency.py $COMMON_ARGS \
    --output_dir outputs/exp6_efficiency \
    --max_samples 200

# Experiment 7: Case Study
echo -e "\n>>> Experiment 7: Interpretability Case Study"
python experiments/exp7_case_study.py $COMMON_ARGS \
    --output_dir outputs/exp7_case_study

# Experiment 8: Position Bias
echo -e "\n>>> Experiment 8: Position Bias Analysis"
python experiments/exp8_position_bias.py $COMMON_ARGS \
    --output_dir outputs/exp8_position_bias \
    --max_samples 200

echo -e "\n============================================================"
echo "All experiments completed!"
echo "Results are in outputs/ directory."
echo "============================================================"