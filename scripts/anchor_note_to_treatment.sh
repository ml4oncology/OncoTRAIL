#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python"
nGPU=0
runTime='0-01:00:00'

data_path="/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/deid_data_pull_2024-06-04/deid_merged_processed_cleaned_clinical_notes_medonc_only.parquet.gzip"
treatment_data_path="/cluster/projects/gliugroup/2BLAST/data/final/treatment_centered_clinical_dataset.parquet.gzip"
ed_visit_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data/interim/emergency_room_visit.parquet.gzip"
symptom_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data/interim/symptom.parquet.gzip" 
last_seen_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data/processed/last_seen_dates.parquet.gzip"
lab_values_data_path="/cluster/projects/gliugroup/2BLAST/data/final/data/interim/lab.parquet.gzip"
save_dir="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification/data/note_anchored_deid"
test_end_date="2019-12-31"
lookback_window=30

for config_name in "firstTreatmentOnly-medOnc-ConsultLetterClinic" "firstVisitOnly-medOnc-ConsultLetterClinic_deid" "mostRecentVisit-medOnc-ConsultLetterClinic_deid"; do
    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prep/anchor_note_to_treatment.py $data_path $treatment_data_path $ed_visit_data_path $symptom_data_path $last_seen_data_path $lab_values_data_path $save_dir $config_name $test_end_date $lookback_window" 
done