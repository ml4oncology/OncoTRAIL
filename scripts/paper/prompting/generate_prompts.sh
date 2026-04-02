#!/bin/bash
export PATH=$PATH:$(pwd)

DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [root_prefix]"
    exit 1
fi

ROOT_PREFIX="${1:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

userName="t127556uhn"
memory=1
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-01:00:00'

rootDirProj=${PROJECT_ROOT}
saveDir=${rootDirProj}/paper/pmh_method/methods/prompting/prompts

target_list=(
    "target_hemoglobin_grade2plus"
    "target_hemoglobin_grade3plus"
    "target_neutrophil_grade2plus"
    "target_neutrophil_grade3plus"
    "target_platelet_grade2plus"
    "target_platelet_grade3plus"
    "target_AKI_grade2plus"
    "target_ALT_grade2plus"
    "target_ALT_grade3plus"
    "target_AST_grade2plus"
    "target_AST_grade3plus"
    "target_bilirubin_grade2plus"
    "target_bilirubin_grade3plus"
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

target_array=$(IFS=','; echo "${target_list[*]}")

numeric_proba=1

repeated_sampling=0

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prompt/generate_prompts.py $target_array $numeric_proba $saveDir $repeated_sampling"
