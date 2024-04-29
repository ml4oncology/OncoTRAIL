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

modelDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/models'
resultsDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/results'

# for LLMName in 'Mistral' 'BioMistral'
# do

# trainDataPath=${dataDir}/embedding_${LLMName}_train_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.npz
# validDataPath=${dataDir}/embedding_${LLMName}_valid_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.npz
# testDataPath=${dataDir}/embedding_${LLMName}_test_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.npz

# for modelName in 'LGBM' 'LR' 'XGB'
# do

# setupStr=${LLMName}_mostRecentVisit_medOnc_ConsultLetterClinic

# pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/main.py $trainDataPath $validDataPath $testDataPath $modelName $setupStr $modelDir $resultsDir"

# done

# done

notesPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.csv'

for splitConfig in 'Temporal' 'Random'
do

for LLMName in 'Mistral' 'BioMistral'
do

for hyperParamEval in 'logloss' 'auroc'
do

embeddingPath=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/embedding/embedding_${LLMName}_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.npz
setupStr=${LLMName}_mostRecentVisit_medOnc_ConsultLetterClinic

for modelName in 'LGBM' 'LR' 'XGB'
do

pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/main.py $notesPath $embeddingPath $splitConfig $hyperParamEval $modelName $setupStr $modelDir $resultsDir"

done

done

done

done