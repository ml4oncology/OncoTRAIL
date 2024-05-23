#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

rootDir=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification
dataDir=${rootDir}/data/interim
saveDir=${rootDir}/data/embedding

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic'
do 

    if [ $noteConfig == 'mostRecentVisit-medOnc-ConsultLetterClinic' ] || [ $noteConfig == 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' ]; then
        numFiles=31
    elif [ $noteConfig == 'firstVisitOnly-medOnc-ConsultLetterClinic' ]; then
        numFiles=6
    fi

    for LLMName in 'ClinicalLongformer' 'Mistral' 'BioMistral'
    do

        dataFileName=embedding_${LLMName}_noteAnchored_${noteConfig}
        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/mergeEmbedding.py $dataDir $dataFileName $saveDir $numFiles"

    done
done