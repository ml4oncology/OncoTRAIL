#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='2-00:00:00'

#rootDir=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification
resultsRootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
modelDir=${resultsRootDir}/models_deid
resultsDir=${resultsRootDir}/results_deid

for tabular in 0 # 1
do 
    for anchorType in "mostRecentVisit-medOnc-ConsultLetterClinic" # "firstVisitOnly-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" 
    do

        notesPath=${resultsRootDir}/data/notes_deid/noteAnchored_${anchorType}_deid.csv

        for targetName in target_sex # target_esas_pain_3pt_change target_death_in_365d target_esas_nausea_3pt_change target_ED_visit target_death_in_30d 
        do

        for splitConfig in 'Temporal' 'Random'
        do

        for LLMName in 'Llama3-8B' 'ClinicalLongformer' 'Mistral' 'BioMistral'
        do

        for hyperParamEval in 'logloss' 'AUROC'
        do

        embeddingPath=${resultsRootDir}/data/embedding_deid/embedding_${LLMName}_noteAnchored_${anchorType}_deid.npz
        setupStr=${LLMName}_${anchorType}

        for modelName in 'MLP' 'LR' 'LGBM' 'XGB'  
        do

        if [[ $modelName == "MLP" ]]; then
            nGPU=1
        else
            nGPU=0
        fi

        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/main.py $notesPath $embeddingPath $splitConfig $hyperParamEval $modelName $setupStr $tabular $targetName $modelDir $resultsDir"

        done

        done

        done

        done

        done

    done
done