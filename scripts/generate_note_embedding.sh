#!/bin/bash
export PATH=$PATH:$(pwd)

LLM_names_array=("Llama3-8B" "ClinicalLongformer" "Mistral" "BioMistral")
LLM_path_array=("Meta-Llama-3-8B" "Clinical-Longformer" "Mistral-7B-v0.1" "BioMistral-7B")
data_dir=?
file_name=?
save_dir=?
notes_col_name="clinical_notes"
n_partitions=?
n_hours=5
memory=16
root_dir_llm='/cluster/projects/gliugroup/2BLAST/LLMs'

n_LLMs=${#LLM_names_array[@]}

for (( index=0; index<n_LLMs; index++ )); do
    LLM_name=${LLM_names_array[$index]}
    LLM_path=${root_dir_llm}/${LLM_path_array[$index]}

    python3 ../src/generate_note_embedding.py $data_dir $file_name $LLM_path $LLM_name $save_dir $notes_col_name $n_partitions $n_hours $memory
done