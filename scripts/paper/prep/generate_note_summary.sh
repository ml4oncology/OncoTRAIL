#!/bin/bash
export PATH=$PATH:$(pwd)

userName="t127556uhn"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-01:00:00'

root_dir_proj=/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/OncoTRAIL
data_dir=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
file_name=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
LLM_name=Qwen2.5-14B-IQ4-XS
LLM_path=/cluster/projects/gliugroup/2BLAST/LLMs/Qwen2.5-14B-Instruct-IQ4_XS.gguf
save_dir=${data_dir}/note_summary
n_partitions=10
n_hours=8
memory=16

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/generate_note_summary.py $data_dir $file_name $LLM_path $LLM_name $save_dir $n_partitions $n_hours $memory"