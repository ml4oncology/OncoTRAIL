#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1

rootDirProj='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification'
dataDir=${rootDirProj}/data/interim
saveDir=${rootDirProj}/data/interim
rootDirLLM='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs'

for note_config in "mostRecentVisit-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" "firstVisitOnly-medOnc-ConsultLetterClinic" ; do

    if [[ $note_config == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=5
    else
        upper_limit=30
    fi

    LLMpath=${rootDirLLM}/Clinical-Longformer
    LLMName='ClinicalLongformer'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir" 

    done

    LLMpath=${rootDirLLM}/Mistral-7B-v0.1
    LLMName='Mistral'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir" 

    done

    LLMpath=${rootDirLLM}/BioMistral-7B
    LLMName='BioMistral'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir" 

    done

done
