#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-05:00:00'

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
print(f'inference_start_date="{config.inference_start_date}"')
print(f'inference_end_date="{config.inference_end_date}"')
EOF
)"

date_lower_limit=$inference_start_date
date_upper_limit=$inference_end_date

rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL
resultsDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/methods/tabular_nlp/inference/results

# Path to the CSV file containing model information
csvRoot=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL/paper/pmh_method/results/aggregate/train_test/tabular_nlp

embeddingPath=None

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

mode="inference"

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

for dataType in "tabular" "nlp-tfidf" "nlp-count"
do 
    for anchorType in "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid"
    do

        csvFilePath="${csvRoot}/best_result_summary_${anchorType}_${dataType}_all_Temporal_noCI.csv"

        # change this too
        notesPath=${rootDir}/paper/pmh_method/data/inference/note_anchored/note_anchored_${anchorType}.csv

        for targetName in "${target_list[@]}"
        do
            # Convert target name format (replace _ with -)
            targetFormatted=$(echo "$targetName" | sed 's/_/-/g')
            
            # Get model file path and preprocessing file path from CSV
            modelFile=$(get_csv_value "$csvFilePath" "$targetFormatted" "saved_model_path")
            preprocessingFile=$(get_csv_value "$csvFilePath" "$targetFormatted" "preprocessing_file_name")
            trainingSetup=$(get_csv_value "$csvFilePath" "$targetFormatted" "training-setup")
            
            # Check if paths were found
            if [[ -z "$modelFile" || -z "$preprocessingFile" ]]; then
                echo "Error: Could not find model or preprocessing file paths for target: $targetFormatted"
                continue
            fi
            
            echo "Found model file: $modelFile"
            echo "Found preprocessing file: $preprocessingFile"
            echo "Training setup: $trainingSetup"

            # Adjust memory and GPU requirements based on training setup
            if [[ "$trainingSetup" == *"MLP"* ]]; then
                nGPU=1
                memory=30
                runTime='0-06:00:00'
            else
                nGPU=0
                memory=16
                runTime='0-02:00:00'  # Inference is typically faster than training
            fi
            
            # Submit inference job
            ../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime group=gliugroup_gpu gpu_type=v100 "../../../src/ML/main.py $notesPath $embeddingPath $resultsDir $date_lower_limit $date_upper_limit $dataType $targetName --mode $mode --model_file $modelFile --preprocessing_file $preprocessingFile"

        done

    done
done
