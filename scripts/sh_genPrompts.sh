#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-01:00:00'

rootDirProj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
saveDir=${rootDirProj}/data/prompting_variation_deid
target_name='target_death_in_365d'

for numeric_proba in 0 1; do

    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/genPrompts.py $target_name $numeric_proba $saveDir"

done



