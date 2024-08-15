#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='2-00:00:00'

#rootDirProj='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification'
rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDir}/data/interim
saveDir=${rootDir}/data/interim_strip

for note_config in "firstVisitOnly-medOnc-ConsultLetterClinic" "mostRecentVisit-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" ; 
do

    if [[ $note_config == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=3
    else
        upper_limit=25
    fi

    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/stripNewLine.py $dataPath $saveDir"  

    done

done
