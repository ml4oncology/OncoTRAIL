#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-01:00:00'

#rootDir=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification
rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDir}/data/embedding_deid
notesPath=${rootDir}/data/notes_deid

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' # 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic'
do 

    for LLMName in 'BioMistral' 'Llama3-8B' 'ClinicalLongformer' 'Mistral' 
    do

        dataFileName=embedding_${LLMName}_noteAnchored_${noteConfig}_deid
        notesFname=noteAnchored_${noteConfig}_deid
        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/addNewTargetToEmbedding.py $dataDir $dataFileName $notesPath $notesFname"

    done
done