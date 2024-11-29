import pandas as pd
import argparse
import logging
import glob
import os
from sklearn.metrics import roc_auc_score
logger = logging.getLogger(__name__)

def compute_stats(prompt_results, target_names, original_data):

    # list of all target names
    target_list = target_names.split(",")

    # replace each _ in target_list with -
    target_list_dash = [target.replace("_", "-") for target in target_list]

    # find all files that start with summary
    matching_files = glob.glob(os.path.join(prompt_results, f"summary_*.csv"))
    
    auc_results = []
    target_results = []

    # loop through all files and compute stats
    for file in matching_files:
        df_summary = pd.read_csv(file, index_col=0)

        # make sure that there is only 1 unique row per mrn, treatment_data
        assert df_summary.groupby(["mrn", "treatment_date"]).size().max() == 1,\
              "There should be only 1 unique row per mrn, treatment_date"
        
        # find which target in target_list this file is for
        target_name = next(target for target in target_list_dash if target in file)

        # compute AUC
        df_summary = df_summary[df_summary['Probability'].notna()]

        # check if there is a column in df_summary that contains the substring "target"
        target_col_name = [col for col in df_summary.columns if 'target' in col]
        if len(target_col_name) == 1:            
            auc_value = roc_auc_score(df_summary[target_col_name[0]], df_summary['Probability'])
        
        else:
            df_original = pd.read_csv(original_data, index_col=0)
            df_original = df_original[['mrn', 'treatment_date', target_name.replace("-", "_")]]
            df_summary = pd.merge(df_summary, df_original, on=['mrn', 'treatment_date'], how='left')
            auc_value = roc_auc_score(df_summary[target_name.replace("-", "_")], df_summary['Probability'])

        auc_results.append(auc_value)
        target_results.append(target_name)

    # save AUC and target to csv
    df_auc = pd.DataFrame({"Target": target_results, "AUC": auc_results})
    df_auc.to_csv(os.path.join(prompt_results, "statistics.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt_results', type=str, help='path to prompt results') # path to prompt results
    parser.add_argument('target_names', type=str, help='Comma-separated list of targets') # targets
    parser.add_argument('original_data', type=str, help='file path to original data') # original data
    args = parser.parse_args()
    compute_stats(args.prompt_results, args.target_names, args.original_data)