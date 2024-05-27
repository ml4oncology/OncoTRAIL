#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

rootDir=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification
modelDir=${rootDir}/models
resultsDir=${rootDir}/results
tabular=1

for anchorType in "firstVisitOnly-medOnc-ConsultLetterClinic" "mostRecentVisit-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" 
do

    notesPath=${rootDir}/data/notes/noteAnchored_${anchorType}.csv

    for targetName in target_esas_nausea_3pt_change target_ED_visit target_death_in_365d target_death_in_30d 
    do

    for splitConfig in 'Temporal' 'Random'
    do

    for LLMName in ClinicalLongformer 'Mistral' 'BioMistral'
    do

    for hyperParamEval in 'logloss' 'AUROC'
    do

    embeddingPath=${rootDir}/data/embedding/embedding_${LLMName}_noteAnchored_${anchorType}.npz
    setupStr=${LLMName}_${anchorType}

    for modelName in 'LR' 'LGBM' 'XGB' 'MLP' 
    do

    if [[ $modelName == "MLP" ]]; then
        nGPU=1
    else
        nGPU=0
    fi

    pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/main.py $notesPath $embeddingPath $splitConfig $hyperParamEval $modelName $setupStr $tabular $targetName $modelDir $resultsDir"

    done

    done

    done

    done

    done

done