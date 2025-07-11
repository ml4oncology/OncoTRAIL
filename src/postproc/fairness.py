import pandas as pd
import argparse
import logging
import glob
import os
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)
import ast
import sys
sys.path.insert(1, "/cluster/projects/gliugroup/2BLAST/data/info")
from phys_names import aliasDictionary, fellow_alias 

def compute_fairness_metrics(notes_df_path, prompting_results_dir,
                             target_name, category, threshold):
    
    # load notes dataframe with demographic info
    df_notes = pd.read_csv(notes_df_path, index_col=0)
    df_notes['treatment_date'] = pd.to_datetime(df_notes['treatment_date']).dt.date

    # load summary dataframe
    target_name = target_name.replace("_", "-")
    matching_files = glob.glob(os.path.join(prompting_results_dir, f"summary_{target_name}_*.csv"))
    df_summary = pd.read_csv(matching_files[0], index_col=0)
    df_summary['treatment_date'] = pd.to_datetime(df_summary['treatment_date']).dt.date

    # apply thresholding
    df_summary['binary_pred_outcome'] = (df_summary['Probability'] >= threshold).astype(int)

    # find column name in df_notes corresponding to category
    if category == 'sex':
        col_name = 'female'
        subgroups = [f'{col_name} == 1', f'{col_name} == 0']
        id_names = ['female', 'male']
    elif category == 'age':
        col_name = 'age'
        subgroups = [f"{col_name} <= 56", f"56 < {col_name} < 71", f"{col_name} >= 71"]
        id_names = ['<= 56','(56,71)','>=71']
    elif category == 'physician':
        # process df_notes
        def extract_names(s):
            try:
                # Extract the line that looks like a list of names
                list_line = [line for line in s.split('\n') if line.startswith('[')][0]
                names = ast.literal_eval(list_line)

                # Apply alias mapping to each name
                return [
                    fellow_alias.get(aliasDictionary.get(name, name), aliasDictionary.get(name, name))
                    for name in names
                ]

            except Exception:
                return []
            
        df_notes['dictating_physician'] = df_notes['stats_dictated_by'].apply(extract_names)
        col_name = 'dictating_physician'
        subgroups = ['Hamzeh AdelAmin Albaba', 'Natasha Basant Leighl', 'Raymond Jang', 'Xueyu Eric Chen', 'Anna Spreafico']
        id_names = subgroups
    else:
        raise NotImplementedError("To be added later.")
    
    # extract columns from df_notes and merge with df_summary
    df_merged = df_summary.merge(df_notes[['mrn','treatment_date', col_name]], how='left', on=['mrn','treatment_date'])
    
    col_names = ['demographic_parity', 'equal_opportunity', 'false_positive_rate', 'predictive_parity']
    metrics = [[] for _ in range(len(col_names))]

    for grp in subgroups:
        # query df_merged
        if category != 'physician':
            df_sub = df_merged.query(grp).copy()
        else:
            df_sub = df_merged[df_merged[col_name].apply(lambda x: grp in x)]

        # compute metrics
        true_target = df_sub[f'{target_name.replace("-","_")}'].to_numpy()
        predicted_val = df_sub['binary_pred_outcome'].to_numpy()

        # compute demographic parity # P(Y=1|subgroup)
        metrics[0].append(df_sub['binary_pred_outcome'].mean())

        # compute equal opportunity (recall)
        metrics[1].append(recall_score(true_target, predicted_val))

        # compute false positive rate
        tn, fp, fn, tp = confusion_matrix(true_target, predicted_val, labels=[0,1]).ravel()
        fpr = fp / (fp + tn)
        metrics[2].append(fpr)

        # compute predictive parity (precision)
        metrics[3].append(precision_score(true_target, predicted_val))
    
    transposed_metrics = list(zip(*metrics))
    df_fairness = pd.DataFrame(transposed_metrics, columns=col_names, index=id_names)

    # save df_fairness
    df_fairness.to_csv(f'{prompting_results_dir}/fairness_metrics_{target_name}_{category}_{threshold}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('notes_df_path', type=str, help='path to notes dataframe') # path to notes dataframe
    parser.add_argument('prompting_results_dir', type=str, help='directory of prompting results') # directory of prompting results
    parser.add_argument('target_name', type=str, help='target name') # target name
    parser.add_argument('category', type=str, help='category') # category for stratifying patients
    parser.add_argument('threshold', type=float, help='threshold for probability classification') # threshold for probability classification 
    args = parser.parse_args()
    compute_fairness_metrics(args.notes_df_path, args.prompting_results_dir, args.target_name, args.category, args.threshold)