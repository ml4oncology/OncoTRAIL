#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=1

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'
# LLMpath='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/BioMistral-7B'
# LLMName='BioMistral'

LLMpath='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/Mistral-7B-v0.1'
LLMName='Mistral'

saveDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'

# upper_limit=15
# dataType='train'

# upper_limit=3
# dataType='valid'

# upper_limit=10
# dataType='test'

# for (( i = 0; i <= upper_limit; i++ )); do
    
#     dataPath="${dataDir}/${dataType}_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic_part${i}.csv"

#     pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir $targetCol" 

# done

upper_limit=30
note_config=mostRecentVisit-medOnc-ConsultLetterClinic

for note_config in "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" "firstVisitOnly-medOnc-ConsultLetterClinic" ; do

    if [[ $note_config == "mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=30
    elif [[ $note_config == "firstVisitOnly-medOnc-ConsultLetterClinic" ]]; then
        upper_limit=5
    fi

    LLMpath='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/Clinical-Longformer'
    LLMName='ClinicalLongformer'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir" 

    done

    LLMpath='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/Mistral-7B-v0.1'
    LLMName='Mistral'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir" 

    done

    LLMpath='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs/BioMistral-7B'
    LLMName='BioMistral'
    for (( i = 0; i <= upper_limit; i++ )); do
        
        dataPath="${dataDir}/noteAnchored_${note_config}_part${i}.csv"

        pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/preprocessing.py $dataPath $LLMpath $LLMName $saveDir" 

    done

done
