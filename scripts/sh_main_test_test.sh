#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/embedding'
#LLMName=BioMistral
#LLMName=Mistral

#modelName='LGBM'

modelDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/models_test_test'
resultsDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/results_test_test'

anchorType='mostRecentVisit-medOnc-ConsultLetterClinic'
notesPath=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/noteAnchored_${anchorType}.csv

for targetName in target_ED_visit target_death_in_30d target_esas_nausea_3pt_change
do

for splitConfig in 'Temporal'
do

for LLMName in 'ClinicalLongformer' 'Mistral' 'BioMistral'
do

for hyperParamEval in 'logloss'
do

embeddingPath=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/embedding/embedding_${LLMName}_noteAnchored_${anchorType}.npz
setupStr=${LLMName}_${anchorType}

for modelName in 'MLP'
do

if [[ $modelName == "MLP" ]]; then
    nGPU=1
else
    nGPU=0
fi

pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/main.py $notesPath $embeddingPath $splitConfig $hyperParamEval $modelName $setupStr $targetName $modelDir $resultsDir"

done

done

done

done

done