#!/bin/bash
export PATH=$PATH:$(pwd)

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <dataset_type>"
    exit 1
fi

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-00:15:00'

dataset_type="$1"

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
print(f'inference_start_date="{config.inference_start_date}"')
print(f'inference_end_date="{config.inference_end_date}"')
EOF
)"

if [[ "$dataset_type" == "train" ]]; then
    start_date=$date_lower_limit
    end_date=$end_devt_date
    save_dir_base=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/train
    data_dir_note=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/train_test/note_tabular_anchored
    extra_args=""
elif [[ "$dataset_type" == "test" ]]; then
    start_date=$start_test_date
    end_date=$date_upper_limit
    save_dir_base=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/test
    data_dir_note=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/train_test/note_tabular_anchored
    extra_args=""
elif [[ "$dataset_type" == "inference" ]]; then
    start_date=$inference_start_date
    end_date=$inference_end_date
    save_dir_base=${root_dir_proj}/paper/pmh_method/methods/prompting/inference
    data_dir_note=${root_dir_proj}/paper/pmh_method/data/inference/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/inference/note_tabular_anchored
    mode="inference"
    few_shot_train_dir_base=${root_dir_proj}/paper/pmh_method/data/train_test
fi

random_sampling=0
quant_level=NA
num_samples=1
numeric_proba=1
prompt_file_dir=${root_dir_proj}/paper/pmh_method/methods/prompting/prompts
n_partitions=10
n_hours=8
memory=16
gpu_constraint=0

use_vllm=0

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

# make sure not to include CI in column!
stage1_csv_file=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/stage1/qwen_best_results_stage1.csv
stage3_csv_file=${root_dir_proj}/paper/pmh_method/methods/prompting/train_test/stage3/llmhyperparameters_best_results_stage3.csv

few_shot_date=$date_few_shot

LLM_name=Qwen2.5-14B-IQ4-XS
LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-IQ4_XS.gguf
tokenizer_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct

add_tabularML_prediction=0

for target in "${target_list_master[@]}"
do

    save_dir=${save_dir_base}/${target}
    
    target_list=(
        "${target}"
    )

    target_names=$(IFS=','; echo "${target_list[*]}")

    # Replace underscores with hyphens
    target_modified="${target//_/-}"

    # Extract prompt_num and tabular from stage1_csv_file
    values_stage1=$(awk -F, -v target="$target_modified" '
        BEGIN {OFS=","}
        NR==1 {
            for (i=1; i<=NF; i++) {
                if ($i == "Target") target_col=i;
                else if ($i == "prompt_num") prompt_col=i;
                else if ($i == "tabular") tabular_col=i;
                else if ($i == "n_few_shot") few_shot_col=i;
            }
        }
        NR>1 && $target_col == target {print $(prompt_col), $(tabular_col), $(few_shot_col); exit}
    ' "$stage1_csv_file")

    # Extract temperature, min_p, top_p, top_k from stage3_csv_file
    values_stage3=$(awk -F, -v target="$target_modified" '
        BEGIN {OFS=","}
        NR==1 {
            for (i=1; i<=NF; i++) {
                if ($i == "Target") target_col=i;
                else if ($i == "temp") temp_col=i;
                else if ($i == "min_p") min_p_col=i;
                else if ($i == "top_p") top_p_col=i;
                else if ($i == "top_k") top_k_col=i;
            }
        }
        NR>1 && $target_col == target {print $(temp_col), $(min_p_col), $(top_p_col), $(top_k_col); exit}
    ' "$stage3_csv_file")

    # Extract values
    IFS=',' read -r prompt_num tabular n_few_shot <<< "$values_stage1"
    IFS=',' read -r temperature min_p top_p top_k <<< "$values_stage3"

    if [ "$tabular" == "note" ]; then
        fname=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
        data_dir=$data_dir_note
    elif [ "$tabular" == "note-tabular" ]; then
        fname=note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
        data_dir=$data_dir_note_tabular
    fi

    file_name=${fname}.csv

    # Build extra_args for inference mode
    if [[ "$dataset_type" == "inference" ]]; then
        if [ "$tabular" == "note" ]; then
            few_shot_train_dir=${few_shot_train_dir_base}/note_anchored/data_partitions/${fname}
        elif [ "$tabular" == "note-tabular" ]; then
            few_shot_train_dir=${few_shot_train_dir_base}/note_tabular_anchored/data_partitions/${fname}
        fi
        extra_args="--mode $mode --few_shot_train_dir $few_shot_train_dir"
    fi

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prompt/main.py $data_dir $file_name $save_dir $start_date $end_date $few_shot_date $random_sampling $n_few_shot $LLM_path $tokenizer_path $LLM_name $quant_level $num_samples $numeric_proba $prompt_file_dir $prompt_num $top_k $min_p $top_p $temperature $target_names $n_partitions $n_hours $memory $gpu_constraint $use_vllm $add_tabularML_prediction $extra_args"

done
