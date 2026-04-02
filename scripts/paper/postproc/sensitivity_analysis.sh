#!/bin/bash
export PATH=$PATH:$(pwd)

DEFAULT_ROOT_PREFIX="/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [root_prefix]"
    exit 1
fi

ROOT_PREFIX="${1:-$DEFAULT_ROOT_PREFIX}"
PROJECT_ROOT="${ROOT_PREFIX}/OncoTRAIL"

userName="t127556uhn"
memory=1
condaEnv="$(conda run -n OncoTRAIL which python)"
nGPU=0
runTime='0-02:00:00'

prompting_tt="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting/prompting_results_train_test_inference.csv"
tabular_tt="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv"
nlptfidf_tt="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-tfidf_all_Temporal.csv"
nlpcount_tt="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-count_all_Temporal.csv"
finetune_tt="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/train_test/finetuning/best_finetune_results_for_comparison.csv"

prompting_inf="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/prompting/prompting_results_train_test_inference.csv"
tabular_inf="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv"
nlptfidf_inf="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-tfidf_all_Temporal.csv"
nlpcount_inf="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/tabular_nlp/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_nlp-count_all_Temporal.csv"
finetune_inf="${PROJECT_ROOT}/paper/pmh_method/results/aggregate/inference/finetuning/best_finetune_results_for_comparison.csv"

save_dir="${PROJECT_ROOT}/paper/pmh_method/results/aggregate"

tt_anchored_notes_path="${PROJECT_ROOT}/paper/pmh_method/data/train_test/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"
inf_anchored_notes_path="${PROJECT_ROOT}/paper/pmh_method/data/inference/note_anchored/note_anchored_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid.csv"

../../pySLURMargs.py $userName $memory $condaEnv $nGPU $runTime "../../../src/postproc/sensitivity_analysis.py $prompting_tt $prompting_inf $tabular_tt $tabular_inf $nlptfidf_tt $nlptfidf_inf $nlpcount_tt $nlpcount_inf $finetune_tt $finetune_inf $save_dir $tt_anchored_notes_path $inf_anchored_notes_path"
