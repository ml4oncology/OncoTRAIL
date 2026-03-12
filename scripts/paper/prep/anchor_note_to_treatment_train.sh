#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-02:00:00'

data_path="/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/data_pull_2024-06-04/splits/deid_merged_processed_cleaned_clinical_notes_medonc_only.parquet.gzip"
treatment_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data_2023-02-21/processed/treatment_centered_dataset.parquet"
opis_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data_2023-02-21/raw/opis.parquet.gzip"
test_end_date="2019-12-31"
lookback_window=30

mode="train"

add_tabular_to_note=0
save_dir="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/data/train_test/note_anchored"
for config_name in "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid"; do
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/anchor_note_to_treatment.py $mode $data_path $treatment_data_path $opis_data_path $save_dir $config_name $test_end_date $lookback_window $add_tabular_to_note" 
done

add_tabular_to_note=1
save_dir="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/data/train_test/note_tabular_anchored"
for config_name in "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid"; do
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/anchor_note_to_treatment.py $mode $data_path $treatment_data_path $opis_data_path $save_dir $config_name $test_end_date $lookback_window $add_tabular_to_note" 
done