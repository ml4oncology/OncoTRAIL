import os
import re
import glob
import random
import warnings
import numpy as np
import pandas as pd
import argparse
import ast
from pathlib import Path
from sklearn.metrics import roc_auc_score

warnings.simplefilter(action='ignore', category=FutureWarning)


def tabulate_results(directory, pattern, exclude=None):
    """
    Collects logloss and AUROC results across multiple CSV files.

    Args:
        directory (str): Directory to search in.
        pattern (str): Glob pattern to match files.
        exclude (str, optional): Pattern to exclude files.

    Returns:
        pd.DataFrame: Multi-indexed DataFrame of results.
    """
    files = glob.glob(os.path.join(directory, pattern))

    results = {
        ('logloss', 'train'): [],
        ('logloss', 'valid'): [],
        ('logloss', 'test'): [],
        ('auc', 'train'): [],
        ('auc', 'valid'): [],
        ('auc', 'test'): [],
    }
    configs = []

    for f in files:
        if exclude and exclude in f:
            continue

        df = pd.read_csv(f, index_col=0)
        file_name = Path(f).stem
        model_name = file_name.split("_")[0]

        metric_type = 'logloss' if 'logloss' in file_name else 'AUROC'
        config_label = f'{model_name}-{metric_type}'
        configs.append(config_label)

        results[('logloss', 'train')].append(df.loc['train', 'log_loss_value'])
        results[('logloss', 'valid')].append(df.loc['valid', 'log_loss_value'])
        results[('logloss', 'test')].append(df.loc['test', 'log_loss_value'])

        results[('auc', 'train')].append(df.loc['train', 'AUROC'])
        results[('auc', 'valid')].append(df.loc['valid', 'AUROC'])
        results[('auc', 'test')].append(df.loc['test', 'AUROC'])

    df_result = pd.DataFrame(results, index=configs)
    df_result.index.name = "training-setup"
    return df_result


def compute_AUC_CI(data_dict, n_bootstraps=1000, alpha=0.05):
    """
    Computes bootstrapped confidence intervals for AUROC.

    Args:
        data_dict (dict): Dictionary with 'test_pred' and 'Y_test'.
        n_bootstraps (int): Number of bootstrap samples.
        alpha (float): Significance level.

    Returns:
        Tuple[float, float]: (lower, upper) bounds of CI.
    """
    preds_test = data_dict['test_pred']
    labels_test = data_dict['Y_test']
    n_test = preds_test.shape[0]

    preds_train = data_dict['train_pred']
    labels_train = data_dict['Y_train']
    n_train = preds_train.shape[0]

    bootstrapped_scores_test = []
    bootstrapped_scores_train = []
    for seed in range(n_bootstraps):
        random.seed(seed)
        idx = random.choices(range(n_test), k=n_test)
        bootstrapped_scores_test.append(roc_auc_score(labels_test[idx], preds_test[idx]))

        random.seed(seed)
        idx = random.choices(range(n_train), k=n_train)
        bootstrapped_scores_train.append(roc_auc_score(labels_train[idx], preds_train[idx]))

    lower_test = np.quantile(bootstrapped_scores_test, alpha / 2)
    upper_test = np.quantile(bootstrapped_scores_test, 1 - alpha / 2)

    lower_train = np.quantile(bootstrapped_scores_train, alpha / 2)
    upper_train = np.quantile(bootstrapped_scores_train, 1 - alpha / 2)

    return lower_test, upper_test, lower_train, upper_train


def summarize_best_result(pred_directory, model_directory, target_list, split_list, note_list,
                          data_type, model_restriction_list, save_dir):
    """
    Summarizes best performing models across targets, splits, and notes.

    Args:
        pred_directory (str): Input directory with model result CSVs and npz predictions.
        model_directory (str): Input directory with model weights.
        target_list (list[str]): List of targets.
        split_list (list[str]): List of data splits (e.g., Random, Temporal).
        note_list (list[str]): List of note configurations.
        data_type (str): 'tabular', 'nlp', etc.
        model_restriction_list (list[str]): If not empty, filters to specific models.
        save_dir (str): Directory to save the CSV output.
    """
    target_list = [t.replace('_', '-') for t in target_list]
    split_str = '-'.join(split_list)
    model_str = 'all' if not model_restriction_list else '-'.join(model_restriction_list)

    for note_config in note_list:
        all_best_results = []

        for target in target_list:
            for split in split_list:
                pattern = f'*_{note_config}_{split}_*_{data_type}_{target}.csv'
                df = tabulate_results(pred_directory, pattern)

                if df.empty:
                    continue

                df.sort_values(by=('auc', 'valid'), ascending=False, inplace=True)

                if model_restriction_list:
                    regex = '|'.join(map(re.escape, model_restriction_list))
                    df = df[df.index.str.contains(regex, na=False)]

                if df.empty:
                    continue

                top = df.head(1).copy()
                top.reset_index(inplace=True)
                top['target'] = target
                top['split_type'] = split
                top['note_config'] = note_config
                all_best_results.append(top)

        if not all_best_results:
            continue

        summary = pd.concat(all_best_results, ignore_index=True)

        # Flatten MultiIndex columns and append data_type
        summary.columns = [
            col if isinstance(col, str) else '_'.join(col).strip()
            for col in summary.columns
        ]
        summary.columns = [col.rstrip('_') for col in summary.columns]

        # Compute CI
        CI_vals_test, CI_vals_train, pred_file_names, model_file_names = [], [], [], []
        for _, row in summary.iterrows():
            setup = row['training-setup']
            target = row['target']
            split = row['split_type']
            model_name, metric = setup.split('-')

            # load prediction file names

            npz_name = f'predictdata_{model_name}_{note_config}_{split}_{metric}_{data_type}_{target}.npz'
            npz_path = os.path.join(pred_directory, npz_name)

            if not os.path.exists(npz_path):
                CI_vals_test.append('[NA,NA]')
                CI_vals_train.append('[NA,NA]')
                pred_file_names.append('missing')
                model_file_names.append('missing')
                continue

            data = np.load(npz_path)
            low_test, high_test, low_train, high_train = compute_AUC_CI(data)
            CI_vals_test.append(f'[{low_test:.3f},{high_test:.3f}]')
            CI_vals_train.append(f'[{low_train:.3f},{high_train:.3f}]')
            pred_file_names.append(npz_path)

            # load model file names
            if 'LR' in model_name:
                model_coef_name = f'model_{model_name}_{note_config}_{split}_{metric}_{data_type}_{target}_coefficients.npz'
                model_coef_path = os.path.join(model_directory, model_coef_name)    
                model_file_names.append(model_coef_path)
            else:
                model_file_names.append('not available')

        summary['test_CI'] = CI_vals_test
        summary['train_CI'] = CI_vals_train
        summary['pred_file_name'] = pred_file_names
        summary['model_file_name'] = model_file_names

        out_csv = f'best_result_summary_{note_config}_{data_type}_{model_str}_{split_str}.csv'
        summary.to_csv(os.path.join(save_dir, out_csv), index=False)

        # remove all columns from summary with CI string
        summary = summary.loc[:, ~summary.columns.str.contains('CI')]
        out_csv = f'best_result_summary_{note_config}_{data_type}_{model_str}_{split_str}_noCI.csv'
        summary.to_csv(os.path.join(save_dir, out_csv), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Summarize best model results with confidence intervals.")

    parser.add_argument("pred_directory", type=str, help="Directory containing result CSVs and prediction .npz files.")
    parser.add_argument("model_directory", type=str, help="Directory containing model files.")
    parser.add_argument("target_list", type=str, help="List of target names. E.g., \"['target1', 'target2']\"")
    parser.add_argument("split_list", type=str, help="List of split types. E.g., \"['Random', 'Temporal']\"")
    parser.add_argument("note_list", type=str, help="List of note configs.")
    parser.add_argument("data_type", type=str, help="Data type: 'tabular', 'nlp', etc.")
    parser.add_argument("model_restriction_list", type=str, help="List of model names to include, or '[]' for all.")
    parser.add_argument("save_dir", type=str, help="Directory to save the summary CSV.")
    args = parser.parse_args()

        # Convert string lists to actual lists
    target_list = ast.literal_eval(args.target_list)
    split_list = ast.literal_eval(args.split_list)
    note_list = ast.literal_eval(args.note_list)
    model_restriction_list = ast.literal_eval(args.model_restriction_list)

    summarize_best_result(
        pred_directory=args.pred_directory,
        model_directory=args.model_directory,
        target_list=target_list,
        split_list=split_list,
        note_list=note_list,
        data_type=args.data_type,
        model_restriction_list=model_restriction_list,
        save_dir=args.save_dir
    )