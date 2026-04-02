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
runTime='0-01:00:00'

results_dir="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_anchored/note_summary"
notes_df_path="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/combine_note_summary.py $results_dir $notes_df_path"

notes_df_path="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_tabular_anchored/note_tabular_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/combine_note_summary.py $results_dir $notes_df_path"
