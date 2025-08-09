#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python"
nGPU=0
runTime='0-00:30:00'

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification

# start_date='2008-01-01'
# end_date='2015-12-31'
# numeric_proba=1
# n_partitions=10
# random_sampling=1

start_date='2008-01-01'
end_date='2015-12-31'
numeric_proba=1
n_partitions=10
random_sampling=0

# start_date='2016-01-01'
# end_date='2019-12-31'
# numeric_proba=1
# n_partitions=10
# random_sampling=0

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

# few_shot_date=NA
few_shot_date='2016-01-01'

for fname in note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid
do

for n_few_shot in 0 4 10 20
do

    file_name=${fname}.csv

    if [ "$fname" == "note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" ]; then
        data_dir=${root_dir_proj}/data/note_anchored_deid
    elif [ "$fname" == "note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" ]; then
        data_dir=${root_dir_proj}/data/note_tabular_anchored_deid
    fi


pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prompt/prepare_data.py $data_dir $file_name $start_date $end_date $few_shot_date $random_sampling $n_few_shot $numeric_proba $target_names $n_partitions"

done
done

