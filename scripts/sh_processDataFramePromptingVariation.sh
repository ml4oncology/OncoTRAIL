#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='0-12:00:00'

#rootDir='/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification'
rootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
saveDir=${rootDir}/data/prompting_variation_deid
num_mrns=1000
num_replicates=1

for noteConfig in 'mostRecentVisit-medOnc-ConsultLetterClinic' # 'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic' 'firstVisitOnly-medOnc-ConsultLetterClinic'   
do

    dataPath=${rootDir}/data/notes_deid/noteAnchored_${noteConfig}_deid.csv
    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/processDataFramePromptingVariation.py $dataPath $saveDir $num_mrns $num_replicates"

done 