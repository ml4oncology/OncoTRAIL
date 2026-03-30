#!/bin/bash
set -e

export PATH=$PATH:$(pwd)

# -------------------------
# Usage check
# -------------------------
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {EPR|EPIC}"
    exit 1
fi

EMR_system="$1"

if [[ "$EMR_system" != "EPR" && "$EMR_system" != "EPIC" ]]; then
    echo "Error: argument must be 'EPR' or 'EPIC'"
    exit 1
fi

# -------------------------
# Common SLURM settings
# -------------------------
userName="t127556uhn"
memory=8
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-03:00:00'

target_list="[
'target_hemoglobin_grade2plus',
'target_neutrophil_grade2plus',
'target_platelet_grade2plus',
'target_AKI_grade2plus',
'target_ALT_grade2plus',
'target_AST_grade2plus',
'target_bilirubin_grade2plus',
'target_esas_pain_3pt_change',
'target_esas_tiredness_3pt_change',
'target_esas_nausea_3pt_change',
'target_esas_depression_3pt_change',
'target_esas_anxiety_3pt_change',
'target_esas_drowsiness_3pt_change',
'target_esas_appetite_3pt_change',
'target_esas_well_being_3pt_change',
'target_esas_shortness_of_breath_3pt_change',
'target_death_in_30d',
'target_death_in_365d',
'target_ED_visit'
]"

# -------------------------
# EMR-specific settings
# -------------------------
if [[ "$EMR_system" == "EPR" ]]; then
    held_out_set="test"
    anchored_notes_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
    prompting_data_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/train_test/test
    tabular_results=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv
    raw_treatment_path='/cluster/projects/gliugroup/2BLAST/data/final/data_2023-02-21/processed/treatment_centered_dataset.parquet'
else  # EPIC
    held_out_set="inference"
    anchored_notes_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/data/inference/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
    prompting_data_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/prompting/inference
    tabular_results=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv
    raw_treatment_path='/cluster/projects/gliugroup/2BLAST/data/final/data_2025-03-29/processed/treatment_centered_data.parquet'
fi

output_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/physician_variability/regression

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/EDA/physician_effect.py $held_out_set \"$target_list\" $anchored_notes_path $output_dir $prompting_data_dir $tabular_results $raw_treatment_path"