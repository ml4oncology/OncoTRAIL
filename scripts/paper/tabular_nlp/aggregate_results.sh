#!/bin/bash
set -e

export PATH=$PATH:$(pwd)

# -------------------------
# Usage check
# -------------------------
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {train|inference}"
    exit 1
fi

MODE="$1"

if [[ "$MODE" != "train" && "$MODE" != "inference" ]]; then
    echo "Error: argument must be 'train' or 'inference'"
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

split_list="['Temporal']"
note_list="['firstTreatmentOnly-medOnc-ConsultLetterClinic_deid']"

# -------------------------
# Mode-specific settings
# -------------------------
if [[ "$MODE" == "train" ]]; then
    pred_directory='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/tabular_nlp/train_test/results/'
    model_directory='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/tabular_nlp/train_test/models/'
    save_dir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/tabular_nlp'
    mode="train"

else  # inference
    pred_directory='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/tabular_nlp/inference/results/'
    model_directory='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/tabular_nlp/train_test/models/'
    save_dir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/inference/tabular_nlp'
    mode="inference"
    
fi

# -------------------------
# Submit job
# -------------------------

for data_type in 'tabular' 'nlp-tfidf' 'nlp-count'
do
    for model_restriction_list in "[]" # "['LR']"
    do
        if [[ "$MODE" == "train" ]]; then
            cmd="../../../src/ML/aggregate.py \
                $pred_directory \
                $model_directory \
                \"$target_list\" \
                \"$split_list\" \
                \"$note_list\" \
                $data_type \
                \"$model_restriction_list\" \
                $save_dir \
                $mode"
        else
            path_to_best_train=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_${data_type}_all_Temporal.csv

            cmd="../../../src/ML/aggregate.py \
                $pred_directory \
                $model_directory \
                \"$target_list\" \
                \"$split_list\" \
                \"$note_list\" \
                $data_type \
                \"$model_restriction_list\" \
                $save_dir \
                $mode \
                --path_to_best_train $path_to_best_train"
        fi

        ../../pySLURMargs.py \
            "$userName" \
            "$memory" \
            "$condaEnv" \
            "$nGPU" \
            "$runTime" \
            "$cmd"
    done
done
