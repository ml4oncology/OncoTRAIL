#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-01:00:00'

rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDir}/data/interim_strip
saveDir=${rootDir}/data/embedding_strip

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic' 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic'
do 

    if [ $noteConfig == 'mostRecentVisit-medOnc-ConsultLetterClinic' ] || [ $noteConfig == 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' ]; then
        numFiles=26
    elif [ $noteConfig == 'firstVisitOnly-medOnc-ConsultLetterClinic' ]; then
        numFiles=4
    fi

    for LLMName in 'Llama3-8B-Instruct' # 'Llama3-8B' # 'ClinicalLongformer' 'Mistral' 'BioMistral'
    do

        dataFileName=embedding_${LLMName}_noteAnchored_${noteConfig}
        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/mergeEmbedding.py $dataDir $dataFileName $saveDir $numFiles"

    done
done
