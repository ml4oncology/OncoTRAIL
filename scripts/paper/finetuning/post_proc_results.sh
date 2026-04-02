#!/bin/bash
set -e

export PATH=$PATH:$(pwd)

# -------------------------
# Usage check
# -------------------------
DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 {EPR|EPIC} [root_prefix]"
    exit 1
fi

MODE="$1"
ROOT_PREFIX="${2:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

if [[ "$MODE" != "EPR" && "$MODE" != "EPIC" ]]; then
    echo "Error: argument must be 'EPR' or 'EPIC'"
    exit 1
fi

# -------------------------
# Common SLURM settings
# -------------------------
userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-01:00:00'

model_name="ModernBERT-base"

# -------------------------
# Mode-specific settings
# -------------------------
if [[ "$MODE" == "EPR" ]]; then
    base_dir="${PROJECT_ROOT}/paper/pmh_method/methods/finetuning/train_test"
    save_dir="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/finetuning"

    cmd="../../../src/finetune/post_proc_results.py \
        $base_dir \
        $save_dir \
        $model_name \
        --mode train"

else  # inference
    base_dir="${PROJECT_ROOT}/paper/pmh_method/methods/finetuning/inference"
    save_dir="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/finetuning"
    path_to_best_train="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/finetuning/best_finetune_results_for_comparison.csv"

    cmd="../../../src/finetune/post_proc_results.py \
        $base_dir \
        $save_dir \
        $model_name \
        --mode inference \
        --path_to_best_train $path_to_best_train"
fi

# -------------------------
# Submit job
# -------------------------
../../pySLURMargs.py \
    "$userName" \
    "$memory" \
    "$condaEnv" \
    "$nGPU" \
    "$runTime" \
    "$cmd"
