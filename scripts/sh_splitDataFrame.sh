#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

#dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/train_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.csv'
#dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/valid_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.csv'
#dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/test_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.csv'
#dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic.csv'
#dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/noteAnchored_mostRecentVisit-medOnc-ConsultLetterClinic.csv'
saveDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'
numRowsPerPart=1500

dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/noteAnchored_mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic.csv'
pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/splitDataFrame.py $dataPath $saveDir $numRowsPerPart"

dataPath='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes/noteAnchored_firstVisitOnly-medOnc-ConsultLetterClinic.csv'
pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/splitDataFrame.py $dataPath $saveDir $numRowsPerPart"