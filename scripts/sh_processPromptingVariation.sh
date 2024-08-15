#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1
runTime='0-16:00:00'

#rootDirProj='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification'
rootDirProj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDirProj}/data/prompting_variation_deid
saveDir=${rootDirProj}/data/prompting_variation_deid
rootDirLLM='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs'

LLMpath=${rootDirLLM}/Meta-Llama-3-8B-Instruct
LLMName='LLama-3-8B'

target_name='target_death_in_365d'

for note_config in "mostRecentVisit-medOnc-ConsultLetterClinic"; do # "firstVisitOnly-medOnc-ConsultLetterClinic" "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" ; do

    upper_limit=249

    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_deid_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/processPromptingVariation.py $dataPath $LLMpath $LLMName $saveDir $target_name" 

    done

done



