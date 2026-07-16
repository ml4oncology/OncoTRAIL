#!/bin/bash
export PATH=$PATH:$(pwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${PROJECT_ROOT_DIR}/env.sh"

DEFAULT_ROOT_PREFIX="${CLUSTER_ROOT_PREFIX}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 {EPR|EPIC} [root_prefix]"
    exit 1
fi

MODE="$1"
ROOT_PREFIX="${2:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

userName="${CLUSTER_USERNAME}"
memory=16
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-01:00:00'

if [[ "$MODE" == "EPR" ]]; then
    save_dir="${PROJECT_ROOT}/paper/pmh_method/results/plots"
    methods="prompting=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting/prompting_results_train_test_inference.csv,"
    methods+="tabular=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv,"
    methods+="nlp-tfidf=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-tfidf_all_Temporal.csv,"
    methods+="nlp-count=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-count_all_Temporal.csv,"
    methods+="finetune=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/finetuning/best_finetune_results_for_comparison.csv"
    mode="train"

else  # inference
    save_dir="${PROJECT_ROOT}/paper/pmh_method/results/plots"
    methods="prompting=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting/prompting_results_train_test_inference.csv,"
    methods+="tabular=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv,"
    methods+="nlp-tfidf=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-tfidf_all_Temporal.csv,"
    methods+="nlp-count=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-count_all_Temporal.csv,"
    methods+="finetune=${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/finetuning/best_finetune_results_for_comparison.csv"
    mode="inference"
fi

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/postproc/compare_methods.py --methods $methods --save_dir $save_dir --mode $mode"
