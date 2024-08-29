#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=24
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python3"
nGPU=0
runTime='1-00:00:00'

#dataDir='/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification/data/notes'
dataDir="/cluster/projects/gliugroup/2BLAST/clinical_notes/HealthReportRecords/results_status_dates/processed/dataframes"
modelPath="/cluster/projects/gliugroup/2BLAST/HuggingFace_LLMs"

modelName="Meta-Llama-3.1-8B-Instruct"

for notesFileName in "merged_processed_cleaned_clinicalNotes_2008-01-01_2017-12-31"
do
    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/tokenizePatientNoteHistory.py $dataDir $modelName $modelPath $notesFileName"
done
