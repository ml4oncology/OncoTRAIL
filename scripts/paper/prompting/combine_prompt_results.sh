#!/bin/bash
export PATH=$PATH:$(pwd)

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <stage>"
    exit 1
fi

stage="$1"

userName="t127556uhn"
memory=8
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-00:30:00'

# Note: need to get a node before you can run this script!

if [[ "$stage" == "stage1" ]]; then
    root_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage1
elif [[ "$stage" == "stage2" ]]; then
    root_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage2
elif [[ "$stage" == "stage3" ]]; then
    root_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/stage3
elif [[ "$stage" == "train" ]]; then
    root_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/train
elif [[ "$stage" == "test" ]]; then
    root_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/test
elif [[ "$stage" == "inference" ]]; then
    root_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/inference
else
    echo "Error: unknown stage '$stage'"
    echo "Valid stages are: stage1, stage2, stage3, train, test, inference"
    exit 1
fi

target_list=(
    "target_hemoglobin_grade2plus"
    "target_neutrophil_grade2plus"
    "target_platelet_grade2plus"
    "target_AKI_grade2plus"
    "target_ALT_grade2plus"
    "target_AST_grade2plus"
    "target_bilirubin_grade2plus"
    "target_esas_pain_3pt_change"
    "target_esas_tiredness_3pt_change"
    "target_esas_nausea_3pt_change"
    "target_esas_depression_3pt_change"
    "target_esas_anxiety_3pt_change"
    "target_esas_drowsiness_3pt_change"
    "target_esas_appetite_3pt_change"
    "target_esas_well_being_3pt_change"
    "target_esas_shortness_of_breath_3pt_change"
    "target_death_in_30d"
    "target_death_in_365d"
    "target_ED_visit"
)

for target in "${target_list[@]}"; do
    TARGET_DIR="$root_dir/$target"
    # Loop through all subdirectories in the target directory
    for subdir in "$TARGET_DIR"/*/; do
        # Check if the subdirectory name starts with "note_anchored" or "note_tabular_anchored"
        if [[ -d "$subdir" ]] && ([[ $(basename "$subdir") == note_anchored* ]] || [[ $(basename "$subdir") == note_tabular_anchored* ]]); then
            if compgen -G "$subdir/summary_*.csv" > /dev/null; then
            # summary exists → do nothing
            :
            else
                ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prompt/combine_prompt_results.py $subdir $target"
            fi
        fi
    done
done