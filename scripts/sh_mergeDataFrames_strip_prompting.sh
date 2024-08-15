#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='2-00:00:00'

rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDir}/data/interim_strip_prompting
saveDir=${rootDir}/data/interim_strip_prompting

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' # 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic'
do 
    if [[ $noteConfig == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=27
    else
        upper_limit=191
    fi

    note_config=noteAnchored_${noteConfig}

    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/mergeDataFrames.py $dataDir $saveDir $note_config 0 $upper_limit"
done
