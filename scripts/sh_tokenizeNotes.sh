#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=24
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/notes'
modelPath="/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs"

#modelName="Mistral-7B-v0.1"
#modelName="BioMistral-7B"
modelName="Clinical-Longformer"

for notesFileName in "noteAnchored_mostRecentVisit-medOnc-ConsultLetterClinic.csv" "noteAnchored_mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic.csv" "noteAnchored_firstVisitOnly-medOnc-ConsultLetterClinic.csv"
do
    pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/tokenizeNotes.py $dataDir $modelName $modelPath $notesFileName"
done
