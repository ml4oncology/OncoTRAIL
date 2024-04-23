#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/embedding'
trainDataPath=${dataDir}/embedding_train_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic.npz
validDataPath=${dataDir}/embedding_valid_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic.npz
testDataPath=${dataDir}/embedding_test_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic.npz

modelName='XGB'
setupStr='bioMistral_firstVisit_medOnc_ConsultLetterClinic'
modelDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/models'
resultsDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/results'

pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/main.py $trainDataPath $validDataPath $testDataPath $modelName $setupStr $modelDir $resultsDir"
