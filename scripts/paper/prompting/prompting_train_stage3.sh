#!/bin/bash
export PATH=$PATH:$(pwd)

# ── Mode selection ────────────────────────────────────────────────────────────
# Usage: ./prompting_train_stage3_consolidated.sh <mode>
# Modes: llama_cpp_sequential | vllm_offline
mode="${1:?Usage: $0 <llama_cpp_sequential|vllm_offline>}"

case "$mode" in
    llama_cpp_sequential|vllm_offline) ;;
    *) echo "Error: unknown mode '$mode'"; exit 1 ;;
esac

# ── Resource settings (differ by mode) ───────────────────────────────────────
if [ "$mode" == "vllm_offline" ]; then
    nGPU=1
    runTime='0-02:00:00'
else
    nGPU=0
    runTime='0-00:15:00'
fi

# ── Common settings ───────────────────────────────────────────────────────────
userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL

eval "$(
python - <<'EOF'
import sys
sys.path.insert(0, "/cluster/home/t127556uhn/gitrepo/2024/OncoTRAIL/src") 
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

# ── Backend flags (differ by mode) ────────────────────────────────────────────
if [ "$mode" == "vllm_offline" ]; then
    use_vllm=1
    vllm_mode="offline"
else
    use_vllm=0
fi

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

add_tabularML_prediction=0

few_shot_date=$date_few_shot

LLM_name=Qwen2.5-14B-IQ4-XS
LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-IQ4_XS.gguf
tokenizer_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct

stage1_csv_file=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/stage1/qwen_best_results_stage1.csv

fmt() {
    if [[ "$1" =~ ^-?[0-9]+$ ]]; then
        echo "${1}.0"
    else
        echo "$1"
    fi
}

for target in "${target_list_master[@]}"
do

    save_dir=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/stage3/${target}

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

for top_k in 10 40 100
do

for min_p in 0.01 0.05 0.1
do

for top_p in 1.0 0.9 0.8
do

for temperature in 0.5 0.7 1.0 1.5
do

    top_k_fmt=$(fmt "$top_k")
    min_p_fmt=$(fmt "$min_p")
    top_p_fmt=$(fmt "$top_p")
    temperature_fmt=$(fmt "$temperature")

    results_dir_name="${save_dir}/${fname}_${LLM_name}_${quant_level}_${start_date}_${end_date}_${random_sampling}_${n_few_shot}_${numeric_proba}_${prompt_num}_${top_k_fmt}_${min_p_fmt}_${top_p_fmt}_${temperature_fmt}"

    # Count CSV files starting with "mrn" in the directory (if it exists)
    count_mrn_csv=$(find "$results_dir_name" -maxdepth 1 -type f -name "mrn*.csv" 2>/dev/null | wc -l)

    if [[ "$count_mrn_csv" -eq 600 ]]; then
        echo "Skipping: $results_dir_name (600 mrn*.csv files already exist)"
        continue    # skip this iteration of the for loop
    else
        echo "Running Python script for: $results_dir_name"

        # ── Build the mode-specific trailing argument ─────────────────────────
        if [ "$mode" == "vllm_offline" ]; then
            mode_arg="--vllm_mode $vllm_mode"
        else
            mode_arg=""
        fi

        # ── Invoke pySLURMargs.py (vllm_offline pins to node159) ─────────────
        common_py_args="../../../src/prompt/main.py $data_dir $file_name $save_dir $start_date $end_date $few_shot_date $random_sampling $n_few_shot $LLM_path $tokenizer_path $LLM_name $quant_level $num_samples $numeric_proba $prompt_file_dir $prompt_num $top_k $min_p $top_p $temperature $target_names $n_partitions $n_hours $memory_submitit $gpu_constraint $use_vllm $add_tabularML_prediction $mode_arg"

        if [ "$mode" == "vllm_offline" ]; then
            ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime node159 "$common_py_args"
        else
            ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "$common_py_args"
        fi
    fi
    
done

done
done
done
done
done