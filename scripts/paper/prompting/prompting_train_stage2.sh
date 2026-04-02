#!/bin/bash
export PATH=$PATH:$(pwd)

DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [root_prefix]"
    exit 1
fi

ROOT_PREFIX="${1:-$DEFAULT_ROOT_PREFIX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../../src" && pwd)"

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-00:15:00'

root_dir_proj="${ROOT_PREFIX}/OncoTRAIL"

eval "$(
export SRC_DIR
python - <<'EOF'
import sys
import os
sys.path.insert(0, os.environ["SRC_DIR"])
import config
print(f'start_test_date="{config.start_test_date}"')
print(f'end_devt_date="{config.end_devt_date}"')
print(f'date_lower_limit="{config.date_lower_limit}"')
print(f'date_upper_limit="{config.date_upper_limit}"')
print(f'date_few_shot="{config.few_shot_date}"')
EOF
)"

start_date=$date_lower_limit
end_date=$end_devt_date
random_sampling=1
quant_level=NA
num_samples=1
numeric_proba=1
prompt_file_dir=${root_dir_proj}/paper/pmh_method/methods/prompting/prompts
n_partitions=10
n_hours=5
memory_submitit=16
gpu_constraint=0

use_vllm=0

LLM_list=(
    "Qwen2.5-14B-IQ2-M"
    "Qwen2.5-14B-Q4-K-M"
    "Qwen2.5-7B-IQ2-M"
    "Qwen2.5-7B-IQ3-M"
    "Qwen2.5-7B-IQ4-XS"
    "Qwen2.5-7B-Q4-K-M"
    "Qwen2.5-7B-Q6-K"
    "Qwen2.5-7B-Q8-0"
    "Qwen2.5-3B-IQ3-M"
    "Qwen2.5-3B-Q4-K-M"
    "Qwen2.5-3B-Q6-K"
    "Qwen2.5-3B-Q8-0"
)

target_list_master=(
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

top_k=-1
min_p=-1
top_p=-1
temperature=-1
add_tabularML_prediction=0

few_shot_date=$date_few_shot

format_val () {
    if [[ "$1" == "-1" ]]; then
        echo "-1.0"
    else
        echo "$1"
    fi
}

stage1_csv_file=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/stage1/qwen_best_results_stage1.csv

for target in "${target_list_master[@]}"
do

    save_dir=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/stage2/${target}

    target_list=(
        "${target}"
    )

    target_names=$(IFS=','; echo "${target_list[*]}")

    # Replace underscores with hyphens
    target_modified="${target//_/-}"

    # Use awk to find the corresponding 'prompt_num' and 'tabular' values
    values=$(awk -F, -v target="$target_modified" '
        BEGIN {OFS=","}
        NR==1 {for (i=1; i<=NF; i++) if ($i == "Target") target_col=i}
        NR==1 {for (i=1; i<=NF; i++) if ($i == "prompt_num") prompt_col=i}
        NR==1 {for (i=1; i<=NF; i++) if ($i == "tabular") tabular_col=i}
        NR==1 {for (i=1; i<=NF; i++) if ($i == "n_few_shot") few_shot_col=i}
        NR>1 && $target_col == target {print $prompt_col, $tabular_col, $few_shot_col; exit}
    ' "$stage1_csv_file")

    # Extract prompt_num and tabular values
    IFS=',' read -r prompt_num tabular n_few_shot <<< "$values"

    if [ "$tabular" == "note" ]; then
        fname=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
        data_dir=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
    elif [ "$tabular" == "note-tabular" ]; then
        fname=note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
        data_dir=${root_dir_proj}/paper/pmh_method/data/train_test/note_tabular_anchored
    fi

    file_name=${fname}.csv

for LLM_name in "${LLM_list[@]}"
do

    if [ "$LLM_name" == "Qwen2.5-14B-IQ2-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-IQ2_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-14B-Q4-K-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-Q4_K_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-14B-Q6-K" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-Q6_K.gguf
    elif [ "$LLM_name" == "Qwen2.5-7B-IQ2-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-7B-Instruct-IQ2_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-7B-IQ3-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-7B-Instruct-IQ3_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-7B-IQ4-XS" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-7B-Instruct-IQ4_XS.gguf
    elif [ "$LLM_name" == "Qwen2.5-7B-Q4-K-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-7B-Instruct-Q4_K_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-7B-Q6-K" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-7B-Instruct-Q6_K.gguf
    elif [ "$LLM_name" == "Qwen2.5-7B-Q8-0" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-7B-Instruct-Q8_0.gguf
    elif [ "$LLM_name" == "Qwen2.5-3B-IQ3-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-3B-Instruct-IQ3_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-3B-Q4-K-M" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-3B-Instruct-Q4_K_M.gguf
    elif [ "$LLM_name" == "Qwen2.5-3B-Q6-K" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-3B-Instruct-Q6_K.gguf
    elif [ "$LLM_name" == "Qwen2.5-3B-Q8-0" ]; then
        LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-3B-Instruct-Q8_0.gguf
    fi

    tokenizer_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct

    top_k_fmt=$(format_val "$top_k")
    min_p_fmt=$(format_val "$min_p")
    top_p_fmt=$(format_val "$top_p")
    temperature_fmt=$(format_val "$temperature")

    results_dir_name="${save_dir}/${fname}_${LLM_name}_${quant_level}_${start_date}_${end_date}_${random_sampling}_${n_few_shot}_${numeric_proba}_${prompt_num}_${top_k_fmt}_${min_p_fmt}_${top_p_fmt}_${temperature_fmt}"

    # Count CSV files starting with "mrn" in the directory (if it exists)
    count_mrn_csv=$(find "$results_dir_name" -maxdepth 1 -type f -name "mrn*.csv" 2>/dev/null | wc -l)

    if [[ "$count_mrn_csv" -eq 600 ]]; then
        echo "Skipping: $results_dir_name (600 mrn*.csv files already exist)"
        continue    # skip this iteration of the for loop
    else
        echo "Running Python script for: $results_dir_name"
        ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prompt/main.py $data_dir $file_name $save_dir $start_date $end_date $few_shot_date $random_sampling $n_few_shot $LLM_path $tokenizer_path $LLM_name $quant_level $num_samples $numeric_proba $prompt_file_dir $prompt_num $top_k $min_p $top_p $temperature $target_names $n_partitions $n_hours $memory_submitit $gpu_constraint $use_vllm $add_tabularML_prediction"
    fi

done
done

