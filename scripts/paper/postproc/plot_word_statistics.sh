#!/bin/bash
# Initialize conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate OncoTRAIL

# Configure R
export R_HOME=$(R RHOME)
export PATH=$R_HOME/bin:$PATH

export PATH=$PATH:$(pwd)

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {epr|epic}"
    exit 1
fi

MODE="$1"

userName="t127556uhn"
memory=16
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

if [[ "$MODE" == "epr" ]]; then
    save_dir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/word_analysis/epr'
    data_dir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/word_analysis/epr'
else
    save_dir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/word_analysis/epic'
    data_dir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/plots/word_analysis/epic'
fi

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime \
            "../../../src/postproc/plot_word_statistics.py $data_dir $save_dir \"$target_list\""
