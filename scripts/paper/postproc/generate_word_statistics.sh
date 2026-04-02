#!/bin/bash
# Initialize conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate OncoTRAIL

# Configure R
export R_HOME=$(R RHOME)
export PATH=$R_HOME/bin:$PATH

export PATH=$PATH:$(pwd)

DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 {EPR|EPIC} [root_prefix]"
    exit 1
fi

# note: need to export R_HOME=$(R RHOME) after activating environment

MODE="$1"
ROOT_PREFIX="${2:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-04:00:00'

target_list=(
    target_hemoglobin_grade2plus
    target_neutrophil_grade2plus
    target_platelet_grade2plus
    target_AKI_grade2plus
    target_ALT_grade2plus
    target_AST_grade2plus
    target_bilirubin_grade2plus
    target_esas_pain_3pt_change
    target_esas_tiredness_3pt_change
    target_esas_nausea_3pt_change
    target_esas_depression_3pt_change
    target_esas_anxiety_3pt_change
    target_esas_drowsiness_3pt_change
    target_esas_appetite_3pt_change
    target_esas_well_being_3pt_change
    target_esas_shortness_of_breath_3pt_change
    target_death_in_30d
    target_death_in_365d
    target_ED_visit
)

# n_targets=${#target_list[@]}

if [[ "$MODE" == "EPR" ]]; then
    data_dir="${PROJECT_ROOT}/paper/pmh_method/methods/prompting/train_test/test"
    save_dir="${PROJECT_ROOT}/paper/pmh_method/results/plots/word_analysis/epr"
    path_to_anchored_notes="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
else
    data_dir="${PROJECT_ROOT}/paper/pmh_method/methods/prompting/inference"
    save_dir="${PROJECT_ROOT}/paper/pmh_method/results/plots/word_analysis/epic"
    path_to_anchored_notes="${PROJECT_ROOT}/paper/pmh_method/data/inference/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
fi

for target in "${target_list[@]}"; do

    note_type='Reason'
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime \
        "../../../src/postproc/generate_word_statistics.py \
        $data_dir $target $note_type $save_dir"

    note_type='note'
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime \
        "../../../src/postproc/generate_word_statistics.py \
        $data_dir $target $note_type $save_dir \
        --path_to_anchored_notes $path_to_anchored_notes"

done
