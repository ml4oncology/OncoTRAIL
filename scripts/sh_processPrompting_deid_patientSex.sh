#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1
runTime='0-16:00:00'

rootDirProj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDirProj}/data/interim_deid_nonmedical_prompting
saveDir=${rootDirProj}/data/interim_deid_nonmedical_prompting
rootDirLLM='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs'

LLMpath=${rootDirLLM}/Meta-Llama-3-8B-Instruct
LLMName='LLama-3-8B'

for target_name in 'target_male' 'target_female'; do 
    for note_config in "mostRecentVisit-medOnc-ConsultLetterClinic"; do # "firstVisitOnly-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" ; do

        if [[ $note_config == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
            upper_limit=27
        else
            upper_limit=191
        fi

        for (( i = 0; i <= upper_limit; i++ )); do

            dataPath="${dataDir}/noteAnchored_${note_config}_part${i}_deid.csv"

            pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/processPromptingForPatientSex.py $dataPath $LLMpath $LLMName $saveDir $target_name" 

        done

    done
done
