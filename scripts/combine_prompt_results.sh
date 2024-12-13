#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python"
nGPU=0
runTime='0-01:00:00'

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

target_names=$(IFS=','; echo "${target_list[*]}")

file_name=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
start_date='2008-01-01'
end_date='2015-12-31'
random_sampling=1
n_few_shot=0
numeric_proba=1
quant_level=NA

for prompt_num in 1
do

for LLM_name in Llama3-8B # Mistral-7B Gemma2-9B
do

for top_k in 10 40 100
do

for min_p in 0.01 0.05 0.1
do

for top_p in 1.0 0.9 0.8
do

for temperature in 0.5 0.7 1.0 1.5
do

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
results_dir_parent=${root_dir_proj}/data/prompt_engineering

results_dir="${results_dir_parent}/"\
"${file_name}_${LLM_name}_${quant_level}"\
"${start_date}_${end_date}_${random_sampling}_"\
"${n_few_shot}_${numeric_proba}_${prompt_num}_"\
"${top_k}_${min_p}_${top_p}_${temperature}"

pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prompt/combine_prompt_results.py $results_dir $target_names"

done
done
done
done
done
done



