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
memory=4
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
    held_out_set="test"
    ICC_results_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/physician_variability/regression/ICC_results_test.csv
    regression_coefficients_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/physician_variability/regression/characteristics_regression_coefficients_test.csv
else  # EPIC
    held_out_set="inference"
    ICC_results_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/physician_variability/regression/ICC_results_inference.csv
    regression_coefficients_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/physician_variability/regression/characteristics_regression_coefficients_inference.csv
fi

prompting_results_path=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/inference/prompting/prompting_results_train_test_inference.csv
output_dir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/physician_variability/regression

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/EDA/plot_regressed_physician_effect.py $held_out_set $prompting_results_path $ICC_results_path $regression_coefficients_path $output_dir"