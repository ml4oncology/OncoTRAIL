import pandas as pd
import argparse
import logging
import glob
import os
import numpy as np
from sklearn.metrics import roc_auc_score
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)

def compute_stats(prompt_results, target_names, original_data):

    # bootstrapping parameters
    n_bootstraps = 1000
    quantile_left = 0.025
    quantile_right = 0.975

    # list of all target names
    target_list = target_names.split(",")

    # replace each _ in target_list with -
    target_list_dash = [target.replace("_", "-") for target in target_list]

    # find all files that start with summary
    matching_files = glob.glob(os.path.join(prompt_results, f"summary_*.csv"))
    
    auc_results = []
    target_results = []
    n_samples = []
    mean_proba = []
    auc_left = []
    auc_right = []

    # loop through all files and compute stats
    for file in matching_files:

        if 'grade3plus' in file:
            continue

        logger.info(f"file: {file}")
    
        df_summary = pd.read_csv(file, index_col=0)

        # make sure that there is only 1 unique row per mrn, treatment_data
        assert df_summary.groupby(["mrn", "treatment_date"]).size().max() == 1,\
              "There should be only 1 unique row per mrn, treatment_date"

        # find which target in target_list this file is for
        target_name = next(target for target in target_list_dash if target in file)

        # compute AUC
        df_summary = df_summary[df_summary['Probability'].notna()]

        # compute number of samples
        n_samples.append(df_summary.shape[0])

        # compute the mean of the column 'Probability'
        mean_proba.append(df_summary['Probability'].mean())

        # check if there is a column in df_summary that contains the substring "target"
        target_col_name = [col for col in df_summary.columns if 'target' in col]
        if len(target_col_name) == 1:            
            auc_value = roc_auc_score(df_summary[target_col_name[0]], df_summary['Probability'])
        
        else:
            df_original = pd.read_csv(original_data, index_col=0)
            df_original = df_original[['mrn', 'treatment_date', target_name.replace("-", "_")]]
            df_summary = pd.merge(df_summary, df_original, on=['mrn', 'treatment_date'], how='left')
            auc_value = roc_auc_score(df_summary[target_name.replace("-", "_")], df_summary['Probability'])

        # perform bootstrapping here
        bootstrapped_auc = []
        for seed in range(n_bootstraps):
            bootstrap_sample = df_summary.sample(frac=1, replace=True, random_state=seed)
            if len(target_col_name) == 1:
                bootstrapped_auc.append(roc_auc_score(bootstrap_sample[target_col_name[0]], bootstrap_sample['Probability']))
            else:
                bootstrapped_auc.append(roc_auc_score(bootstrap_sample[target_name.replace("-", "_")], bootstrap_sample['Probability']))
        
        auc_results.append(auc_value)
        target_results.append(target_name)
        auc_left.append(np.quantile(bootstrapped_auc, quantile_left))
        auc_right.append(np.quantile(bootstrapped_auc, quantile_right))

    # save AUC and target to csv
    df_auc = pd.DataFrame({"Target": target_results, "AUC": auc_results, "n_samples": n_samples, "mean_proba": mean_proba,
                           "AUC_left": auc_left, "AUC_right": auc_right})
    # create a string of confidence interval for the AUC
    df_auc['CI'] = df_auc.apply(lambda x: f"[{x['AUC_left']:.3f}, {x['AUC_right']:.3f}]", axis=1)
    # remove AUC_left and AUC_right
    df_auc = df_auc[['Target', 'AUC', 'n_samples', 'mean_proba', 'CI']]
    df_auc.to_csv(os.path.join(prompt_results, "statistics.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt_results', type=str, help='path to prompt results') # path to prompt results
    parser.add_argument('target_names', type=str, help='Comma-separated list of targets') # targets
    parser.add_argument('original_data', type=str, help='file path to original data') # original data
    args = parser.parse_args()
    compute_stats(args.prompt_results, args.target_names, args.original_data)