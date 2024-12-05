#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python"
nGPU=1
runTime='0-01:00:00'

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification

data_dir=${root_dir_proj}/data/note_anchored_deid
file_name=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
save_dir=${root_dir_proj}/data/prompt_engineering
start_date='2008-01-01'
end_date='2015-12-31'
random_sampling=1
n_few_shot=0
quant_level=4
num_samples=1
numeric_proba=1
prompt_file_dir=${root_dir_proj}/data/prompts
n_partitions=10
n_hours=8
memory=16

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

for prompt_num in 1 # change this to a different persona
do

for LLM_name in Gemma2-9B # Llama3-8B Mistral-7B 
do

    if [ "$LLM_name" == "Llama3-8B" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Meta-Llama-3-8B-Instruct
    elif [ "$LLM_name" == "Mistral-7B" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Mistral-7B-Instruct-v0.3
    elif [ "$LLM_name" == "Gemma2-9B" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/gemma-2-9b-it
    fi

for top_k in -1 # 10 # 40 100
do

for min_p in -1 # 0.01 # 0.05 0.1
do

for top_p in -1 # 1.0 # 0.9 0.8
do

for temperature in -1 # 0.5 # 0.7 1.0 1.5
do

pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prompt/prompt_engineering.py $data_dir $file_name $save_dir $start_date $end_date $random_sampling $n_few_shot $LLM_path $LLM_name $quant_level $num_samples $numeric_proba $prompt_file_dir $prompt_num $top_k $min_p $top_p $temperature $target_names $n_partitions $n_hours $memory"

# python3 ../src/prompt/prompt_engineering.py \
#   $data_dir $file_name $save_dir \
#   $start_date $end_date $random_sampling \
#   $n_few_shot \
#   $LLM_path $LLM_name $quant_level \
#   $num_samples $numeric_proba \
#   $prompt_file_dir $prompt_num \
#   $top_k $min_p $top_p $temperature \
#   $target_names $n_partitions \
#   $n_hours $memory &

done
done
done
done
done
done

# wait