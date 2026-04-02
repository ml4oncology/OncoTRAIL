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
nGPU=1
runTime='0-10:00:00'

eval "$(
python - <<'EOF'
import sys
sys.path.insert(0, "/cluster/home/t127556uhn/gitrepo/2024/OncoTRAIL/src") 
import config
print(f'start_test_date="{config.start_test_date}"')
print(f'end_devt_date="{config.end_devt_date}"')
print(f'date_lower_limit="{config.date_lower_limit}"')
print(f'date_upper_limit="{config.date_upper_limit}"')
print(f'date_few_shot="{config.few_shot_date}"')
EOF
)"

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

LLMpath=/cluster/projects/gliugroup/2BLAST/LLMs/ModernBERT-base
baseDir=${PROJECT_ROOT}/paper/pmh_method
notesPath=${baseDir}/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
resultsDir=${baseDir}/methods/finetuning/train_test
developmentSetDate=$end_devt_date
batchSizeTest=16
batchSizeTrain=8
modelType="encoder"

LLMname=$(basename "$LLMpath")

for targetName in "${target_list[@]}"; do
for learningRate in 0.00001 0.00005 0.0001; do 
for nEpochs in 8; do
for gradientSteps in 4 8; do 

    # Convert learning rate to scientific notation if needed
    if [[ "$learningRate" == "0.00001" ]]; then
        lr_str="1e-05"
    elif [[ "$learningRate" == "0.00005" ]]; then
        lr_str="5e-05"
    else
        lr_str="$learningRate"
    fi

    # Construct output file path
    outputFile="post_finetune_${targetName}_metrics_LLM-${LLMname}_lr-${lr_str}_epochs-${nEpochs}_batchsizetrain-${batchSizeTrain}_gradientsteps-${gradientSteps}.csv"
    outputDir="${resultsDir}/${targetName}"
    filepath="${outputDir}/${outputFile}"

    if [[ -f "$filepath" ]]; then
        echo "Skipping: $filepath already exists."
        continue
    fi

    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime node159 "../../../src/finetune/main_train.py $modelType $LLMpath $notesPath $targetName $developmentSetDate $resultsDir $learningRate $nEpochs $batchSizeTrain $batchSizeTest $gradientSteps"

done
done
done 
done

