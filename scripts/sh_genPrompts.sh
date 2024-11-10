#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-01:00:00'

rootDirProj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
saveDir=${rootDirProj}/data/first_visit_super_simplified_prompt

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

numeric_proba=1

cmd="pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime"

# Loop through the array and append each genPrompts.py call to the command
for target_name in "${target_list[@]}"; do
    cmd+=" \"../src/genPrompts.py $target_name $numeric_proba $saveDir\""
done

# Output the final command
eval "$cmd"