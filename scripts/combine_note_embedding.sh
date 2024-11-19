#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python"
nGPU=0
runTime='0-01:00:00'

LLM_names_array=("ClinicalLongformer") # "Llama3-8B" "Mistral" "BioMistral")

data_dir="/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/embedding_2024-06-04"
root_file_name="note_anchored_firstVisitOnly-medOnc-ConsultLetterClinic_deid"
num_files=10

n_LLMs=${#LLM_names_array[@]}

for (( index=0; index<n_LLMs; index++ )); do
    LLM_name=${LLM_names_array[$index]}
    file_name=${root_file_name}_${LLM_name}_embedding

    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prep/combine_note_embedding.py $data_dir $file_name $num_files"
done