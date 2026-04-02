#!/bin/bash
set -e

export PATH=$PATH:$(pwd)

# -------------------------
# Usage check
# -------------------------
DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 {EPR|EPIC} [root_prefix]"
    exit 1
fi

EMR_system="$1"
ROOT_PREFIX="${2:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

if [[ "$EMR_system" != "EPR" && "$EMR_system" != "EPIC" ]]; then
    echo "Error: argument must be 'EPR' or 'EPIC'"
    exit 1
fi

# -------------------------
# Common SLURM settings
# -------------------------
userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-01:00:00'

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
    anchored_notes_path="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
    prompting_data_dir="${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/test"
    tabular_results="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv"
else  # EPIC
    anchored_notes_path="${PROJECT_ROOT}/paper/pmh_method/data/inference/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
    prompting_data_dir="${PROJECT_ROOT}/paper/pmh_method/methods/prompting/inference"
    tabular_results="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv"
fi

output_dir="${PROJECT_ROOT}/paper/pmh_method/results/plots/physician_variability"

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/postproc/plot_physician_characteristics.py \"$target_list\" $EMR_system $anchored_notes_path $prompting_data_dir $tabular_results $output_dir"
