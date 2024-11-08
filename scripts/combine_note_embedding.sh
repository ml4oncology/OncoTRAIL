#!/bin/bash
export PATH=$PATH:$(pwd)

LLM_names_array=("Llama3-8B" "ClinicalLongformer" "Mistral" "BioMistral")

data_dir=?
root_file_name=?
num_files=?

n_LLMs=${#LLM_names_array[@]}

for (( index=0; index<n_LLMs; index++ )); do
    LLM_name=${LLM_names_array[$index]}
    file_name=${root_file_name}_${LLM_name}_embedding

    python3 ../src/combine_note_embedding.py $data_dir $file_name $num_files
done