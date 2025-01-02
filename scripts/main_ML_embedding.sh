#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-05:00:00'

rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
resultsRootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
modelDir=${resultsRootDir}/models_tabular
resultsDir=${resultsRootDir}/results_tabular
embeddingDir=/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/embedding_2024-06-04

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

for dataType in "tabular" # "notes-tabular" "notes" 
do 
    for anchorType in "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" # "firstVisitOnly-medOnc-ConsultLetterClinic_deid" "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic_deid" "mostRecentVisit-medOnc-ConsultLetterClinic_deid" 
    do

        notesPath=${rootDir}/data/note_anchored_deid/note_anchored_${anchorType}.csv

        for targetName in "${target_list[@]}"
        do

        for splitConfig in 'Temporal' 'Random'
        do

        for LLMName in 'ClinicalLongformer' 'Llama3-8B' 'Mistral' 'BioMistral'
        do

        for hyperParamEval in 'logloss' 'AUROC'
        do

        embeddingPath=${embeddingDir}/note_anchored_${anchorType}_${LLMName}_embedding.npz
        
        if [ "$dataType" == "tabular" ]; then
            setupStr="$anchorType"
        else
            setupStr="${LLMName}_${anchorType}"
        fi

        for modelName in 'LR' 'LGBM' 'XGB' 'MLP' 
        do

        if [[ $modelName == "MLP" ]]; then
            nGPU=1
        else
            nGPU=0
        fi

        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/ML/main.py $notesPath $embeddingPath $splitConfig $hyperParamEval $modelName $setupStr $dataType $targetName $modelDir $resultsDir"

        done

        done

        done

        done

        done

    done
done