#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'
saveDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/embedding'

dataFileName='embedding_train_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic'
numFiles=16

# dataFileName='embedding_valid_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic'
# numFiles=4

# dataFileName='embedding_test_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic'
# numFiles=11

pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/mergeEmbedding.py $dataDir $dataFileName $saveDir $numFiles"