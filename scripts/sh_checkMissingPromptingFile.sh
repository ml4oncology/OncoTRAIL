#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='1-00:00:00'

rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDir}/data/interim_deid_prompting
#dataDir=${rootDir}/data/interim_strip_prompting
#dataDir=${rootDir}/data/interim_prompting
llm_name='LLama-3-8B'
#target_name='target_death_in_365d'
target_name='target_sex'
suffix='deid'

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' #  'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic'
do 
    if [[ $noteConfig == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=27
    else
        upper_limit=191
    fi

    note_config=noteAnchored_${noteConfig}

    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/checkMissingPromptingFile.py $dataDir $note_config 0 $upper_limit $llm_name $target_name -s $suffix"
done
