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

def compute_fairness_metrics(notes_df_path, method_dfs, target_name,
                             category, threshold, save_dir):
    
    print(target_name)

    # load notes dataframe with demographic info
    df_notes = pd.read_csv(notes_df_path, header=0)
    if isinstance(df_notes.columns, pd.MultiIndex):
        df_notes.columns = ["_".join(map(str, col)).strip() for col in df_notes.columns.values]

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

    for method, method_df in method_dfs.items():

        print(method)

        if method == 'prompting':
            # load summary dataframe from prompting
            target_name = target_name.replace("_", "-")
            # find the value under the column "path_to_predictions" for "target" value equal to target_name
            prompting_results_dir = method_df[method_df['target'] == target_name]['path_to_predictions'].values[0]
            df_predictions = pd.read_csv(prompting_results_dir, header=0)
            # apply thresholding
            df_predictions['binary_pred_outcome'] = (df_predictions['Probability'] >= threshold).astype(int)
            
        elif method == 'finetune':
            # load summary dataframe from finetuning
            target_name = target_name.replace("-", "_")
            finetuning_results_dir = method_df[method_df['target'] == target_name]['path_to_predictions_test'].values[0]
            df_predictions = pd.read_csv(finetuning_results_dir, header=0)
            df_predictions['Probability'] = np.stack(df_predictions["prob"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).values)[:,1]
            # apply thresholding
            df_predictions['binary_pred_outcome'] = (df_predictions['Probability'] >= threshold).astype(int)

        else:
            target_name = target_name.replace("_", "-")
            tabular_nlp_results_npz = method_df[method_df['target'] == target_name]['pred_file_name'].values[0]
            tabular_nlp_files = np.load(tabular_nlp_results_npz)
            df_predictions = pd.DataFrame({'mrn': tabular_nlp_files['mrn_test'], 'prob': np.ravel(tabular_nlp_files['test_pred'])})
            # apply thresholding
            df_predictions['binary_pred_outcome'] = (df_predictions['prob'] >= threshold).astype(int)

        # extract columns from df_notes and merge with df_predictions
        target_name = target_name.replace("-", "_")
        df_merged = df_predictions[['mrn', 'binary_pred_outcome']].merge(df_notes[['mrn', col_name, target_name]], how='left', on=['mrn'])
        
        col_names = ['demographic_parity', 'equal_opportunity', 'false_positive_rate', 'predictive_parity']
        metrics = [[] for _ in range(len(col_names))]

        for grp in subgroups:
            # query df_merged
            if category != 'physician':
                df_sub = df_merged.query(grp).copy()
            else:
                df_sub = df_merged[df_merged[col_name].apply(lambda x: grp in x)]

            # compute metrics
            true_target = df_sub[target_name].to_numpy()
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
        df_fairness.to_csv(f'{save_dir}/fairness_metrics_{target_name.replace("_", "-")}_{method}_{category}_{threshold}.csv')

def parse_method_csv_args(arg: str) -> dict[str, str]:
    """
    Convert a string like "prompting=path1.csv,tabular=path2.csv"
    into a dictionary { "prompting": "path1.csv", ... }
    """
    result = {}
    for item in arg.split(','):
        if '=' not in item:
            raise ValueError(f"Invalid format for method-csv pair: {item}")
        method, path = item.split('=', 1)
        result[method.strip()] = path.strip()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('notes_df_path', type=str, help='path to notes dataframe') # path to notes dataframe
    parser.add_argument('methods', type=str, help="Comma-separated method=csv_path pairs")
    parser.add_argument('target_name', type=str, help='target name') # target name
    parser.add_argument('category', type=str, help='category') # category for stratifying patients
    parser.add_argument('threshold', type=float, help='threshold for probability classification') # threshold for probability classification
    parser.add_argument('save_dir', type=str, help='save directory')
    args = parser.parse_args()

    method_csvs = parse_method_csv_args(args.methods)
    method_dfs = {
        method: pd.read_csv(csv_path)
        for method, csv_path in method_csvs.items()
    }

    compute_fairness_metrics(args.notes_df_path, method_dfs, args.target_name, args.category, args.threshold, args.save_dir)