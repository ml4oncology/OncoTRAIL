#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='2-00:00:00'

rootDir=/cluster/home/t127556uhn/gitrepo/2024/LLM-notes-classification
projectRootDir=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification
modelDir=${rootDir}/new_line_models
resultsDir=${rootDir}/new_line_results
rootDirLLM='/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs'

for anchorType in "mostRecentVisit-medOnc-ConsultLetterClinic" 
do

    notesPath=${projectRootDir}/data/notes/noteAnchored_${anchorType}.csv

    for targetName in target_death_in_365d 
    do

        for splitConfig in 'Temporal' 'Random'
        do

            for LLMpath in ${rootDirLLM}/Mistral-7B-v0.1
            do

                for modelName in 'LR' 'LGBM' 'XGB'
                do

                    for hyperParamEval in 'auroc' 'logloss'
                    do
                    
                        pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/test_newline_model.py $notesPath $splitConfig $hyperParamEval $modelName $modelDir $resultsDir $LLMpath $targetName"

                    done

                done

            done

        done
    
    done

done