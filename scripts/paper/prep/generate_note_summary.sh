#!/bin/bash
export PATH=$PATH:$(pwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${PROJECT_ROOT_DIR}/env.sh"

DEFAULT_ROOT_PREFIX="${CLUSTER_ROOT_PREFIX}"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [root_prefix]"
    exit 1
fi

ROOT_PREFIX="${1:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

userName="${CLUSTER_USERNAME}"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-01:00:00'

root_dir_proj=${PROJECT_ROOT}
data_dir=${root_dir_proj}/paper/pmh_method/data/train_test/note_anchored
file_name=note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv
LLM_name=Qwen2.5-14B-IQ4-XS
LLM_path=${LLM_BASE_DIR}/Qwen2.5-14B-Instruct-IQ4_XS.gguf
save_dir=${data_dir}/note_summary
n_partitions=10
n_hours=8
memory=16

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/prep/generate_note_summary.py $data_dir $file_name $LLM_path $LLM_name $save_dir $n_partitions $n_hours $memory"
