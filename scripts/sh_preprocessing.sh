#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'
LLMpath='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/BioMistral-7B'
saveDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'
targetCol='target_ED_visit'

# upper_limit=15
# dataType='train'

# upper_limit=3
# dataType='valid'

upper_limit=10
dataType='test'

for (( i = 0; i <= upper_limit; i++ )); do
    
    dataPath="${dataDir}/${dataType}_noteAnchored_ED_visit_firstVisit_medOnc_ConsultLetterClinic_part${i}.csv"

    pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $saveDir $targetCol" 

done

