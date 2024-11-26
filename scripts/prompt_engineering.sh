#!/bin/bash
export PATH=$PATH:$(pwd)

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification

data_dir=${root_dir_proj}/data/note_anchored_deid
file_name=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
save_dir=${root_dir_proj}/data/prompt_engineering
start_date='2008-01-01'
end_date='2015-12-31'
random_sampling=1
few_shot_file_path="None"
n_few_shot=0
LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Meta-Llama-3-8B-Instruct
LLM_name=Llama3-8B
quant_level=4
num_samples=1
numeric_proba=1
prompt_file_dir=${root_dir_proj}/data/prompts
n_partitions=2
n_hours=8
memory=16

# LOOP
prompt_num=1
top_k=5
min_p=0.01
top_p=0.3
temperature=0.6

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

target_names=$(IFS=','; echo "${target_list[*]}")

python3 ../src/prompt/prompt_engineering.py \
  $data_dir $file_name $save_dir \
  $start_date $end_date $random_sampling \
  $few_shot_file_path $n_few_shot \
  $LLM_path $LLM_name $quant_level \
  $num_samples $numeric_proba \
  $prompt_file_dir $prompt_num \
  $top_k $min_p $top_p $temperature \
  $target_names $n_partitions \
  $n_hours $memory
