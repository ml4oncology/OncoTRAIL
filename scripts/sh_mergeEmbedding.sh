#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0

dataDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/interim'
saveDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification/data/embedding'

#LLMName='BioMistral'
#LLMName='Mistral'

# dataFileName=embedding_${LLMName}_train_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic
# numFiles=16

# dataFileName=embedding_${LLMName}_valid_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic
# numFiles=4

# dataFileName=embedding_${LLMName}_test_noteAnchored_ED_visit_mostRecentVisit_medOnc_ConsultLetterClinic
# numFiles=11

# numFiles=31
# note_config=mostRecentVisit-medOnc-ConsultLetterClinic

# for LLMName in 'ClinicalLongformer' 'Mistral' 'BioMistral'
# do

# dataFileName=embedding_${LLMName}_noteAnchored_${note_config}
# pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/mergeEmbedding.py $dataDir $dataFileName $saveDir $numFiles"

# done

numFiles=31
note_config=mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic

for LLMName in 'ClinicalLongformer' 'Mistral' 'BioMistral'
do

dataFileName=embedding_${LLMName}_noteAnchored_${note_config}
pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/mergeEmbedding.py $dataDir $dataFileName $saveDir $numFiles"

done

numFiles=6
note_config=firstVisitOnly-medOnc-ConsultLetterClinic

for LLMName in 'ClinicalLongformer' 'Mistral' 'BioMistral'
do

dataFileName=embedding_${LLMName}_noteAnchored_${note_config}
pySLURMargs.py $userName $memory $condaEnv $nGPU "../src/mergeEmbedding.py $dataDir $dataFileName $saveDir $numFiles"

done