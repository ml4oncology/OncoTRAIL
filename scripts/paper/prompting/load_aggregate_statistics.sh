#!/bin/bash
set -e
export PATH=$PATH:$(pwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${PROJECT_ROOT_DIR}/env.sh"

# -------------------------
# Usage
# -------------------------
DEFAULT_ROOT_PREFIX="${CLUSTER_ROOT_PREFIX}"

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage:"
    echo "  $0 aggregate <stage> [root_prefix]"
    echo "  $0 concatenate [root_prefix]"
    exit 1
fi

mode="$1"

# -------------------------
# SLURM config
# -------------------------
userName="${CLUSTER_USERNAME}"
memory=4
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-02:00:00'

# ============================================================
# MODE: AGGREGATE
# ============================================================
if [[ "$mode" == "aggregate" ]]; then

    if [[ $# -lt 2 || $# -gt 3 ]]; then
        echo "Usage: $0 aggregate <stage> [root_prefix]"
        exit 1
    fi

    stage="$2"
    ROOT_PREFIX="${3:-$DEFAULT_ROOT_PREFIX}"
    PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

    if [[ "$stage" == "stage1" ]]; then
        results_dir=${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/stage1
        save_dir=$results_dir
        save_string="$stage"
    elif [[ "$stage" == "stage2" ]]; then
        results_dir=${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/stage2
        save_dir=$results_dir
        save_string="$stage"
    elif [[ "$stage" == "stage3" ]]; then
        results_dir=${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/stage3
        save_dir=$results_dir
        save_string="$stage"
    elif [[ "$stage" == "EPR_train" ]]; then
        results_dir=${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/train
        save_dir=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/prompting
        save_string="train"
        stage="train"
    elif [[ "$stage" == "EPR_test" ]]; then
        results_dir=${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/test
        save_dir=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/prompting
        save_string="test"
        stage="test"
    elif [[ "$stage" == "EPIC" ]]; then
        results_dir=${PROJECT_ROOT}/paper/pmh_method/methods/prompting/inference
        stage="test"
        save_string="inference"
        save_dir=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting
    else
        echo "Error: unknown stage '$stage'"
        echo "Valid stages: stage1, stage2, stage3, EPR_train, EPR_test, EPIC"
        exit 1
    fi

    target_list="[
    'target_hemoglobin_grade2plus',
    'target_neutrophil_grade2plus',
    'target_platelet_grade2plus',
    'target_AKI_grade2plus',
    'target_ALT_grade2plus',
    'target_AST_grade2plus',
    'target_bilirubin_grade2plus',
    'target_esas_pain_3pt_change',
    'target_esas_tiredness_3pt_change',
    'target_esas_nausea_3pt_change',
    'target_esas_depression_3pt_change',
    'target_esas_anxiety_3pt_change',
    'target_esas_drowsiness_3pt_change',
    'target_esas_appetite_3pt_change',
    'target_esas_well_being_3pt_change',
    'target_esas_shortness_of_breath_3pt_change',
    'target_death_in_30d',
    'target_death_in_365d',
    'target_ED_visit'
    ]"

    ../../pySLURMargs.py \
        $userName $memory $condaEnv $nGPU $runTime \
        "../../../src/prompt/utils.py $mode $results_dir \"$target_list\" $stage $save_string $save_dir"

# ============================================================
# MODE: CONCATENATE
# ============================================================
elif [[ "$mode" == "concatenate" ]]; then
    if [[ $# -gt 2 ]]; then
        echo "Usage: $0 concatenate [root_prefix]"
        exit 1
    fi

    ROOT_PREFIX="${2:-$DEFAULT_ROOT_PREFIX}"
    PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

    # ---- MANUALLY SPECIFY PATHS HERE ----
    results_dir_train=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/prompting
    results_dir_test=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/prompting
    results_dir_inference=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting
    save_dir=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting

    ../../pySLURMargs.py \
        $userName $memory $condaEnv $nGPU $runTime \
        "../../../src/prompt/utils.py $mode \
        $results_dir_train \
        $results_dir_test \
        $results_dir_inference \
        $save_dir"

else
    echo "Error: unknown mode '$mode'"
    echo "Valid modes are: aggregate, concatenate"
    exit 1
fi
