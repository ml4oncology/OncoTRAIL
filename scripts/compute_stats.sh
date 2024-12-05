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

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
data_dir=${root_dir_proj}/data/note_anchored_deid
original_data=${data_dir}/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv

file_name=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
quant_level=4
start_date='2008-01-01'
end_date='2015-12-31'
random_sampling=1
n_few_shot=0
numeric_proba=1

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

results_dir_parent=${root_dir_proj}/data/prompt_engineering

prompt_results="${results_dir_parent}/"\
"${file_name}_${LLM_name}_${quant_level}_"\
"${start_date}_${end_date}_${random_sampling}_"\
"${n_few_shot}_${numeric_proba}_${prompt_num}_"\
"${top_k}_${min_p}_${top_p}_${temperature}"

pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prompt/compute_stats.py $prompt_results $target_names $original_data"

done
done
done
done
done
done



