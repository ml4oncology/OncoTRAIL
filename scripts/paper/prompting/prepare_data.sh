#!/bin/bash
export PATH=$PATH:$(pwd)

DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <dataset_type> [root_prefix]"
    exit 1
fi

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-00:30:00'

dataset_type="$1"
ROOT_PREFIX="${2:-$DEFAULT_ROOT_PREFIX}"
root_dir_proj="${ROOT_PREFIX}/OncoTRAIL"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../../src" && pwd)"

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
print(f'inference_start_date="{config.inference_start_date}"')
print(f'inference_end_date="{config.inference_end_date}"')
EOF
)"

if [[ "$dataset_type" == "devt" ]]; then
    start_date=$date_lower_limit
    end_date=$end_devt_date
    numeric_proba=1
    n_partitions=10
    random_sampling=1
    mode="train"
    data_dir_note_anchored=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/train_test/note_tabular_anchored
    few_shot_loop=(0 4 8 16)
elif [[ "$dataset_type" == "EPR_train" ]]; then
    start_date=$date_lower_limit
    end_date=$end_devt_date
    numeric_proba=1
    n_partitions=10
    random_sampling=0
    mode="train"
    data_dir_note_anchored=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/train_test/note_tabular_anchored
    few_shot_loop=(0 4 8 16)
elif [[ "$dataset_type" == "EPR_test" ]]; then
    start_date=$start_test_date
    end_date=$date_upper_limit
    numeric_proba=1
    n_partitions=10
    random_sampling=0
    mode="train"
    data_dir_note_anchored=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/train_test/note_tabular_anchored
    few_shot_loop=(0 4 8 16)
elif [[ "$dataset_type" == "EPIC" ]]; then
    start_date=$inference_start_date
    end_date=$inference_end_date
    numeric_proba=1
    n_partitions=10
    random_sampling=0
    mode="inference"
    data_dir_note_anchored=${root_dir_proj}/paper/pmh_method/data/inference/note_anchored
    data_dir_note_tabular=${root_dir_proj}/paper/pmh_method/data/inference/note_tabular_anchored
    few_shot_loop=(0)
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

target_names=$(IFS=','; echo "${target_list[*]}")

few_shot_date=$date_few_shot

for fname in note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
do

for n_few_shot in "${few_shot_loop[@]}"
do

    file_name=${fname}.csv

    if [ "$fname" == "note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" ]; then
        data_dir=$data_dir_note_anchored
    elif [ "$fname" == "note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" ]; then
        data_dir=$data_dir_note_tabular
    fi


../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prompt/prepare_data.py $mode $data_dir $file_name $start_date $end_date $few_shot_date $random_sampling $n_few_shot $numeric_proba $target_names $n_partitions"

done
done
