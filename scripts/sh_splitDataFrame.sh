#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
rootDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification'
# saveDir=${rootDir}/data/interim
# numRowsPerPart=1500
saveDir=${rootDir}/data/interim_deid
numRowsPerPart=200

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic'   
do

    dataPath=${rootDir}/data/notes/noteAnchored_${noteConfig}.csv
    pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/splitDataFrame.py $dataPath $saveDir $numRowsPerPart"

done 