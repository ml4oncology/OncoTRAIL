#!/bin/bash
export PATH=$PATH:$(pwd)

# LLM_names_array=("ClinicalLongformer") # "Llama3-8B" "Mistral" "BioMistral")
# LLM_path_array=("Clinical-Longformer") # "Meta-Llama-3-8B" "Mistral-7B-v0.1" "BioMistral-7B")
LLM_names_array=("Qwen2.5-14B-IQ4-XS")
LLM_path_array=("/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-IQ4_XS.gguf")

data_dir="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification/data/note_anchored_deid"
save_dir="/cluster/projects/gliugroup/2BLAST/data/processed/clinical_notes/embedding_2024-06-04"
n_hours=10
memory=16
root_dir_llm='/cluster/projects/gliugroup/2BLAST/LLMs'
llama_cpp=1

file_name="note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
n_partitions=10

n_LLMs=${#LLM_names_array[@]}

for (( index=0; index<n_LLMs; index++ )); do
    # LLM_name=${LLM_names_array[$index]}
    # LLM_path=${root_dir_llm}/${LLM_path_array[$index]}
    LLM_name=${LLM_names_array[$index]}
    LLM_path=${LLM_path_array[$index]}

    python3 ../src/prep/generate_note_embedding.py $data_dir $file_name $LLM_path $LLM_name $save_dir $n_partitions $n_hours $memory $llama_cpp
done
