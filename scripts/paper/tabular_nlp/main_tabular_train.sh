#!/bin/bash
export PATH=$PATH:$(pwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${PROJECT_ROOT_DIR}/env.sh"

DEFAULT_ROOT_PREFIX="${CLUSTER_ROOT_PREFIX}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [root_prefix]"
    exit 1
fi

ROOT_PREFIX="${1:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"
SRC_DIR="$(cd "${SCRIPT_DIR}/../../../src" && pwd)"

userName="${CLUSTER_USERNAME}"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-05:00:00'

eval "$(
export SRC_DIR
python - <<'EOF'
import sys
import os
sys.path.insert(0, os.environ["SRC_DIR"])
import config
print(f'start_test_date="{config.start_test_date}"')
print(f'end_devt_date="{config.end_devt_date}"')
print(f'date_lower_limit="{config.date_lower_limit}"')
print(f'date_upper_limit="{config.date_upper_limit}"')
print(f'date_few_shot="{config.few_shot_date}"')
EOF
)"

end_devt_date=$end_devt_date
date_lower_limit=$date_lower_limit
date_upper_limit=$date_upper_limit
rootDir=${PROJECT_ROOT}/paper/pmh_method
resultsRootDir=${rootDir}/methods/tabular_nlp/train_test
modelDir=${resultsRootDir}/models
resultsDir=${resultsRootDir}/results

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

mode="train"
embeddingPath=None
anchorType="firstTreatmentOnly-medOnc-ConsultLetterClinic_deid"
setupStr="$anchorType"
notesPath=${rootDir}/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv

for dataType in "tabular" "nlp-tfidf" "nlp-count"
do 

for targetName in "${target_list[@]}"
do

for splitConfig in 'Temporal'
do

for hyperParamEval in 'AUROC' 'logloss'
do

for modelName in 'MLP' 'LR' 'LGBM' 'XGB'
do

if [[ $modelName == "MLP" ]]; then
    nGPU=1
    runTime='0-06:00:00'
else
    nGPU=0
    runTime='0-05:00:00'
fi

    targetNameFile="${targetName//_/-}"

    resultsFile="${resultsDir}/${modelName}_${anchorType}_${splitConfig}_${hyperParamEval}_${dataType}_${targetNameFile}.csv"

    if [[ -f "$resultsFile" ]]; then
        echo "Skipping (exists): $resultsFile"
        continue
    fi

    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime group=gliugroup_gpu gpu_type=v100 "../../../src/ML/main.py $notesPath $embeddingPath $resultsDir $date_lower_limit $date_upper_limit $dataType $targetName --mode $mode --model_dir $modelDir --split_config $splitConfig --hyperparam_eval $hyperParamEval --model_name $modelName --setup_str $setupStr --end_devt_date $end_devt_date"
done

done

done

done

done
