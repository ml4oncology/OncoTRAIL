#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="~/miniforge3/envs/LLMfinetune/bin/python"
nGPU=0
runTime='0-01:00:00'

LLM_names_array=("Llama3-8B") # "ClinicalLongformer" "Mistral" "BioMistral")

data_dir=?
root_file_name=?
num_files=?

n_LLMs=${#LLM_names_array[@]}

for (( index=0; index<n_LLMs; index++ )); do
    LLM_name=${LLM_names_array[$index]}
    file_name=${root_file_name}_${LLM_name}_embedding

    pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../src/prep/combine_note_embedding.py $data_dir $file_name $num_files"
done