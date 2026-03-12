#!/bin/bash
set -e
export PATH=$PATH:$(pwd)

# -------------------------
# Usage
# -------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  $0 aggregate <stage>"
    echo "  $0 concatenate"
    exit 1
fi

mode="$1"

# -------------------------
# SLURM config
# -------------------------
userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-02:00:00'

# ============================================================
# MODE: AGGREGATE
# ============================================================
if [[ "$mode" == "aggregate" ]]; then

    if [[ $# -ne 2 ]]; then
        echo "Usage: $0 aggregate <stage>"
        exit 1
    fi

    stage="$2"

    if [[ "$stage" == "stage1" ]]; then
        results_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage1
        save_dir=$results_dir
        save_string="$stage"
    elif [[ "$stage" == "stage2" ]]; then
        results_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage2
        save_dir=$results_dir
        save_string="$stage"
    elif [[ "$stage" == "stage3" ]]; then
        results_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage3
        save_dir=$results_dir
        save_string="$stage"
    elif [[ "$stage" == "train" ]]; then
        results_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/train
        save_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/prompting
        save_string="$stage"
    elif [[ "$stage" == "test" ]]; then
        results_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/test
        save_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/prompting
        save_string="$stage"
    elif [[ "$stage" == "inference" ]]; then
        results_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/inference
        stage="test"
        save_string="inference"
        save_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/inference/prompting
    else
        echo "Error: unknown stage '$stage'"
        echo "Valid stages: stage1, stage2, stage3, train, test, inference"
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

    # ---- MANUALLY SPECIFY PATHS HERE ----
    results_dir_train=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/prompting
    results_dir_test=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/prompting
    results_dir_inference=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/inference/prompting
    save_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/inference/prompting

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
