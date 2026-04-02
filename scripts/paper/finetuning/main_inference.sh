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

modelType="encoder"
batchSizeTest=4
resultsDir=${PROJECT_ROOT}/paper/pmh_method/methods/finetuning/inference
notesPath=${PROJECT_ROOT}/paper/pmh_method/data/inference/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv

# Path to the CSV file containing model information
csvFilePath="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/finetuning/best_finetune_results_no_CI.csv"

# List of targets to process
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

# Function to get value from CSV file based on target
get_csv_value() {
    local csv_file="$1"
    local target_value="$2"
    local column_name="$3"
    
    # Use awk to find the row with matching target and extract the specified column
    # Assumes CSV has headers and target is in the "target" column
    awk -F',' -v target="$target_value" -v col="$column_name" '
    NR==1 {
        for(i=1; i<=NF; i++) {
            if($i == "target") target_col = i
            if($i == col) value_col = i
        }
    }
    NR>1 && $target_col == target {
        print $value_col
        exit
    }' "$csv_file"
}

# Loop through each target
for targetName in "${target_list[@]}"
do
    echo "Processing target: $targetName"
    
    # Get saved model path from CSV
    savedModelPath=$(get_csv_value "$csvFilePath" "$targetName" "saved_model_path")
    
    echo "Using saved model path: $savedModelPath"

    # Check if model path was found
    if [[ -z "$savedModelPath" ]]; then
        echo "Error: Could not find saved model path for target: $targetName"
        continue
    fi
    
    # Submit inference job
    ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime node159 "../../../src/finetune/main_inference.py $modelType $savedModelPath $notesPath $targetName $resultsDir $batchSizeTest"
    
done
