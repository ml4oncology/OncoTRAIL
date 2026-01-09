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


def tabulate_results(directory, pattern, exclude=None, mode="train"):
    """
    Collects logloss and AUROC results across multiple CSV files.

    Args:
        directory (str): Directory to search in.
        pattern (str): Glob pattern to match files.
        exclude (str, optional): Pattern to exclude files.
        mode (str): "train" or "inference" mode.

    Returns:
        pd.DataFrame: Multi-indexed DataFrame of results.
    """
    files = glob.glob(os.path.join(directory, pattern))

    if mode == "train":
        results = {
            ('logloss', 'train'): [],
            ('logloss', 'valid'): [],
            ('logloss', 'test'): [],
            ('auc', 'train'): [],
            ('auc', 'valid'): [],
            ('auc', 'test'): [],
        }
    else:  # mode == "inference"
        results = {
            ('logloss', 'inference'): [],
            ('auc', 'inference'): [],
        }
    
    configs = []

    for f in files:
        if exclude and exclude in f:
            continue

        df = pd.read_csv(f, index_col=0)
        file_name = Path(f).stem
        if file_name.startswith("model_"):
            model_name = file_name.split("_")[1]
        else:
            model_name = file_name.split("_")[0]

        metric_type = 'logloss' if 'logloss' in file_name else 'AUROC'
        config_label = f'{model_name}-{metric_type}'
        configs.append(config_label)

        if mode == "train":
            results[('logloss', 'train')].append(df.loc['train', 'log_loss_value'])
            results[('logloss', 'valid')].append(df.loc['valid', 'log_loss_value'])
            results[('logloss', 'test')].append(df.loc['test', 'log_loss_value'])

            results[('auc', 'train')].append(df.loc['train', 'AUROC'])
            results[('auc', 'valid')].append(df.loc['valid', 'AUROC'])
            results[('auc', 'test')].append(df.loc['test', 'AUROC'])
        else:  # mode == "inference"
            results[('logloss', 'inference')].append(df.loc['test', 'log_loss_value'])
            results[('auc', 'inference')].append(df.loc['test', 'AUROC'])

    df_result = pd.DataFrame(results, index=configs)
    df_result.index.name = "training-setup"
    return df_result


def compute_AUC_CI(data_dict, n_bootstraps=1000, alpha=0.05, mode="train"):
    """
    Computes bootstrapped confidence intervals for AUROC.

    Args:
        data_dict (dict): Dictionary with 'test_pred' and 'Y_test'.
        n_bootstraps (int): Number of bootstrap samples.
        alpha (float): Significance level.
        mode (str): "train" or "inference" mode.

    Returns:
        Tuple: (lower_test, upper_test, lower_train, upper_train) for train mode,
               (lower_test, upper_test, None, None) for inference mode.
    """
    preds_test = data_dict['test_pred']
    labels_test = data_dict['Y_test']
    n_test = preds_test.shape[0]

    if mode == "train":
        preds_train = data_dict['train_pred']
        labels_train = data_dict['Y_train']
        n_train = preds_train.shape[0]

    bootstrapped_scores_test = []
    if mode == "train":
        bootstrapped_scores_train = []
    
    for seed in range(n_bootstraps):
        random.seed(seed)
        idx = random.choices(range(n_test), k=n_test)
        bootstrapped_scores_test.append(roc_auc_score(labels_test[idx], preds_test[idx]))

        if mode == "train":
            random.seed(seed)
            idx = random.choices(range(n_train), k=n_train)
            bootstrapped_scores_train.append(roc_auc_score(labels_train[idx], preds_train[idx]))

    lower_test = np.quantile(bootstrapped_scores_test, alpha / 2)
    upper_test = np.quantile(bootstrapped_scores_test, 1 - alpha / 2)

    if mode == "train":
        lower_train = np.quantile(bootstrapped_scores_train, alpha / 2)
        upper_train = np.quantile(bootstrapped_scores_train, 1 - alpha / 2)
        return lower_test, upper_test, lower_train, upper_train
    else:
        return lower_test, upper_test, None, None


def summarize_best_result(pred_directory, model_directory, target_list, split_list, note_list,
                          data_type, model_restriction_list, save_dir, mode="train", path_to_best_train=None):
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
        mode (str): "train" or "inference" mode.
        path_to_best_train (str, optional): Path to best training results CSV for inference mode.
    """
    target_list = [t.replace('_', '-') for t in target_list]
    split_str = '-'.join(split_list)
    model_str = 'all' if not model_restriction_list else '-'.join(model_restriction_list)

    # Load training results if in inference mode
    train_results_df = None
    if mode == "inference" and path_to_best_train:
        train_results_df = pd.read_csv(path_to_best_train)
        # Keep only relevant columns
        train_results_df = train_results_df[['target', 'auc_train', 'train_CI', 'auc_test', 'test_CI']].copy()

    for note_config in note_list:
        all_best_results = []

        for target in target_list:
            for split in split_list:
                pattern = f'*_{note_config}_{split}_*_{data_type}_{target}.csv'
                df = tabulate_results(pred_directory, pattern, mode=mode)

                if df.empty:
                    continue

                # Only sort by validation AUC in train mode
                if mode == "train":
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
        if mode == "train":
            (CI_vals_test, 
            CI_vals_train, 
            pred_file_names, 
            model_file_names, 
            preprocessing_file_names,
            saved_model_file_names) = [], [], [], [], [], []
        else:  # mode == "inference"
            CI_vals_test = []
            pred_file_names = []

        for _, row in summary.iterrows():
            setup = row['training-setup']
            target = row['target']
            split = row['split_type']
            model_name, metric = setup.split('-')

            # load prediction file names
            if mode == "train":
                npz_name = f'predictdata_{model_name}_{note_config}_{split}_{metric}_{data_type}_{target}.npz'
            elif mode == "inference":
                npz_name = f'inference_model_{model_name}_{note_config}_{split}_{metric}_{data_type}_{target}.npz'
            npz_path = os.path.join(pred_directory, npz_name)

            if not os.path.exists(npz_path):
                CI_vals_test.append('[NA,NA]')
                if mode == "train":
                    CI_vals_train.append('[NA,NA]')
                    pred_file_names.append('missing')
                    model_file_names.append('missing')
                continue

            data = np.load(npz_path)
            if mode == "train":
                low_test, high_test, low_train, high_train = compute_AUC_CI(data, mode=mode)
                CI_vals_train.append(f'[{low_train:.3f},{high_train:.3f}]')
            else:
                low_test, high_test, _, _ = compute_AUC_CI(data, mode=mode)
            
            CI_vals_test.append(f'[{low_test:.3f},{high_test:.3f}]')

            pred_file_names.append(npz_path)

            if mode == "train":
                # load model file names
                if 'LR' in model_name:
                    model_coef_name = f'model_{model_name}_{note_config}_{split}_{metric}_{data_type}_{target}_coefficients.npz'
                    model_coef_path = os.path.join(model_directory, model_coef_name)    
                    model_file_names.append(model_coef_path)
                    model_type = "LR"
                else:
                    model_file_names.append('not available')
                    model_type = "NonLR"
                
                preprocessing_fname = f'{note_config}_{target}_{split}_{data_type}_{model_type}_preprocessing_artifacts.pkl'
                preprocessing_file_names.append(os.path.join(model_directory, preprocessing_fname))

                saved_model_path = f'model_{model_name}_{note_config}_{split}_{metric}_{data_type}_{target}'
                saved_model_file_names.append(os.path.join(model_directory, saved_model_path))

        if mode == "train":
            summary['test_CI'] = CI_vals_test
            summary['train_CI'] = CI_vals_train
            summary['pred_file_name'] = pred_file_names
            summary['model_file_name'] = model_file_names
            summary['preprocessing_file_name'] = preprocessing_file_names
            summary['saved_model_path'] = saved_model_file_names
        else:  # mode == "inference"
            summary['inference_CI'] = CI_vals_test
            summary['pred_file_name'] = pred_file_names

            # Merge with training results if available
            if train_results_df is not None:
                summary = summary.merge(train_results_df, on='target', how='left')

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
    parser.add_argument("mode", type=str, choices=["train", "inference"], help="Mode: 'train' or 'inference'")
    parser.add_argument("--path_to_best_train", type=str, default=None, 
                       help="Path to CSV file with best training results (required for inference mode)")
    
    args = parser.parse_args()

    # Convert string lists to actual lists
    target_list = ast.literal_eval(args.target_list)
    split_list = ast.literal_eval(args.split_list)
    note_list = ast.literal_eval(args.note_list)
    model_restriction_list = ast.literal_eval(args.model_restriction_list)

    # Validate arguments for inference mode
    if args.mode == "inference" and not args.path_to_best_train:
        parser.error("--path_to_best_train is required when mode is 'inference'")

    summarize_best_result(
        pred_directory=args.pred_directory,
        model_directory=args.model_directory,
        target_list=target_list,
        split_list=split_list,
        note_list=note_list,
        data_type=args.data_type,
        model_restriction_list=model_restriction_list,
        save_dir=args.save_dir,
        mode=args.mode,
        path_to_best_train=args.path_to_best_train
    )