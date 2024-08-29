#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1
runTime='0-16:00:00'

rootDirProj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
dataDir=${rootDirProj}/data/prompting_variation_deid

rootDirLLM='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs'

LLMpath=${rootDirLLM}/Meta-Llama-3-8B-Instruct
LLMName='LLama-3-8B'
target_name='target_death_in_365d'
promptPath=${rootDirProj}/data/prompting_variation_deid/promptList_${target_name}

file_name=noteAnchored_mostRecentVisit-medOnc-ConsultLetterClinic_deid_template

num_samples=100
lower_limit_patient=0
upper_limit_patient=100 #1000
lower_limit_prompt=0
upper_limit_prompt=24

for probaType in numeric-proba0 numeric-proba1 ; do
    promptFile=${promptPath}_${probaType}.json
    saveDir=${rootDirProj}/data/prompting_variation_deid/${probaType}

    for (( i_patient = lower_limit_patient; i_patient < upper_limit_patient; i_patient++ )); do
        for (( i_prompt = lower_limit_prompt; i_prompt < upper_limit_prompt; i_prompt++ )); do
            pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/processPromptEngineering.py $dataDir $LLMpath $LLMName $saveDir $target_name $file_name $i_patient $promptFile $i_prompt $num_samples"
        done
    done
    
done