#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-05:00:00'

rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
resultsRootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
modelDir=${resultsRootDir}/models_refactor
resultsDir=${resultsRootDir}/results_refactor
embeddingDir=/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/embedding_2024-06-04

for dataType in "notes-tabular" "notes" "tabular"
do 
    for anchorType in "firstVisitOnly-medOnc-ConsultLetterClinic_deid" # "firstTreatmentOnly-medOnc-ConsultLetterClinic_deid" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic_deid" "mostRecentVisit-medOnc-ConsultLetterClinic_deid" 
    do

        notesPath=${rootDir}/data/note_anchored_deid/note_anchored_${anchorType}.csv

        for targetName in target_esas_pain_3pt_change #target_death_in_365d target_esas_nausea_3pt_change target_ED_visit target_death_in_30d 
        do

        for splitConfig in 'Temporal' #'Random'
        do

        for LLMName in  'ClinicalLongformer' # 'Llama3-8B' 'Mistral' 'BioMistral'
        do

        for hyperParamEval in 'logloss' #'AUROC'
        do

        embeddingPath=${embeddingDir}/note_anchored_${anchorType}_${LLMName}_embedding.npz
        
        if [ "$dataType" == "tabular" ]; then
            setupStr="$anchorType"
        else
            setupStr="${LLMName}_${anchorType}"
        fi

        for modelName in 'LR' #'MLP' 'LGBM' 'XGB'  
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