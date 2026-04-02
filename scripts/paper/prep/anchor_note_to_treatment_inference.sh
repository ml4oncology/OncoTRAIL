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
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-02:00:00'

data_path="/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/data_pull_2025-01-08/splits/deid_merged_processed_cleaned_clinical_notes_medonc_only_epic_records_only.parquet.gzip"
treatment_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data_2025-03-29/processed/treatment_centered_data.parquet"
train_test_anchored_df_path="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
opis_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data_2025-03-29/interim/chemo.parquet"
test_end_date=None
lookback_window=30

mode="inference"

add_tabular_to_note=0
save_dir="${PROJECT_ROOT}/paper/pmh_method/data/inference/note_anchored"
treatment_dates_path="/cluster/projects/gliugroup/2BLAST/data/final/data_2025-03-29/processed/treatment_centered_dates.parquet"
for config_name in "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid"; do
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/anchor_note_to_treatment.py $mode $data_path $treatment_data_path $opis_data_path $save_dir $config_name $test_end_date $lookback_window $add_tabular_to_note --treatment_dates_path $treatment_dates_path --train_test_anchored_df_path $train_test_anchored_df_path" 
done

add_tabular_to_note=1
save_dir="${PROJECT_ROOT}/paper/pmh_method/data/inference/note_tabular_anchored"
for config_name in "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid"; do
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/anchor_note_to_treatment.py $mode $data_path $treatment_data_path $opis_data_path $save_dir $config_name $test_end_date $lookback_window $add_tabular_to_note --treatment_dates_path $treatment_dates_path --train_test_anchored_df_path $train_test_anchored_df_path" 
done
