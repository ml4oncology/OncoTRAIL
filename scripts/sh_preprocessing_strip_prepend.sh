#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1
runTime='0-16:00:00'

# rootDirProj='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification'
rootDirProj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDirProj}/data/interim_strip_prepend
saveDir=${rootDirProj}/data/interim_strip_prepend
rootDirLLM='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs'

# Note to self: Why is the number of data frames for strip not the same as the number of data frames for deid? 
# The strip notes dataframe were processed from the split dataframe with original notes

for note_config in "mostRecentVisit-medOnc-ConsultLetterClinic"; do # "firstVisitOnly-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" 

    if [[ $note_config == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=3
    else
        upper_limit=25
    fi

    LLMpath=${rootDirLLM}/Meta-Llama-3-8B-Instruct
    LLMName='Llama3-8B-Instruct'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir -p 1" 

    done

    LLMpath=${rootDirLLM}/Meta-Llama-3-8B
    LLMName='Llama3-8B'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir -p 1" 

    done

done
