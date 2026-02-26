import os
import re
import sys
import glob
import argparse
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from oncotrail.constants import target_dict_mapping
from sklearn.metrics import roc_auc_score

# Setup logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def compute_AUC_CI(df, n_bootstraps=1000, alpha=0.05):

    bootstrapped_scores = []
    for seed in range(n_bootstraps):
        sampled_df = df.sample(frac=1, replace=True, random_state=seed) 
        probs_np = np.stack(sampled_df["prob"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).values)
        true_labels_np = np.array(sampled_df["label"].values)
        auc = roc_auc_score(true_labels_np, probs_np[:, 1])
        bootstrapped_scores.append(auc)

    lower = np.quantile(bootstrapped_scores, alpha / 2)
    upper = np.quantile(bootstrapped_scores, 1 - alpha / 2)

    return lower, upper

def format_lr(lr):
    return f"{lr:.6f}".rstrip('0').rstrip('.') if lr >= 0.0001 else f"{lr:.0e}"

def plot_loss_vs_step(df, save_path):
    train_loss_df = df.groupby('step', as_index=False).first()
    eval_loss_df = df[df['eval_loss'].notna()]
    train_loss_df['epoch_floor'] = np.floor(train_loss_df['epoch']).astype(int)

    epoch_avg_df = train_loss_df.groupby('epoch_floor').agg({
        'loss': 'mean',
        'step': 'max'
    }).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_df['step'], train_loss_df['loss'], label='Train Loss (raw)', color='blue', alpha=0.3)
    plt.plot(eval_loss_df['step'], eval_loss_df['eval_loss'], label='Eval Loss', color='orange')
    plt.plot(epoch_avg_df['step'], epoch_avg_df['loss'], label='Train Loss (per-epoch avg)', color='blue', linestyle='--')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss vs. Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {save_path}.png")

def collect_finetune_metrics(base_dir, mode="train"):
    all_results = []

    for subdir in os.listdir(base_dir):
        if not subdir.startswith("target_"):
            continue

        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        target_name = subdir
        grouped_files = {}

        for filename in os.listdir(subdir_path):
            if "metrics" not in filename or not filename.endswith(".csv") or filename.startswith("._"):
                continue

            if mode == "train":
                match = re.search(
                    r"(pre_finetune|post_finetune).*lr-(?P<lr>[\de.-]+)_epochs-(?P<epochs>\d+)_batchsizetrain-\d+_gradientsteps-(?P<gradientsteps>\d+)",
                    filename
                )
                if not match:
                    continue

                phase = "pre" if "pre_finetune" in filename else "post"
            
            elif mode == "inference":
                match = re.search(
                    r"inference.*lr-(?P<lr>[\de.-]+)_epochs-(?P<epochs>\d+)_batchsizetrain-\d+_gradientsteps-(?P<gradientsteps>\d+)",
                    filename
                )
                if not match:
                    continue

                phase = "inference"
            
            lr = float(match.group("lr"))
            epochs = int(match.group("epochs"))
            gradientsteps = int(match.group("gradientsteps"))

            key = (lr, epochs, gradientsteps)
            grouped_files.setdefault(key, {})[phase] = os.path.join(subdir_path, filename)

        for (lr, epochs, gradientsteps), files in grouped_files.items():
            if mode == "train":
                if "pre" not in files or "post" not in files:
                    continue
                try:
                    df_pre = pd.read_csv(files["pre"])
                    df_post = pd.read_csv(files["post"])
                    if df_pre.shape[0] != 1 or df_post.shape[0] != 1:
                        continue

                    df_pre = df_pre.add_prefix("pre_")
                    df_post = df_post.add_prefix("post_")

                    combined = pd.concat([df_pre, df_post], axis=1)
                    combined["target"] = target_name
                    combined["lr"] = lr
                    combined["epochs"] = epochs
                    combined["gradientsteps"] = gradientsteps

                    all_results.append(combined)
                except Exception as e:
                    logger.warning(f"Failed to process {files}: {e}")
            
            elif mode == "inference":
                if "inference" not in files:
                    continue
                try:
                    df_inference = pd.read_csv(files["inference"])
                    if df_inference.shape[0] != 1:
                        continue

                    df_inference = df_inference.add_prefix("inference_")
                    df_inference["target"] = target_name
                    df_inference["lr"] = lr
                    df_inference["epochs"] = epochs
                    df_inference["gradientsteps"] = gradientsteps

                    all_results.append(df_inference)
                except Exception as e:
                    logger.warning(f"Failed to process {files}: {e}")

    if not all_results:
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)

    # Reorder columns (reinserted logic)
    fixed_cols = ["target", "lr", "epochs", "gradientsteps"]
    metric_cols = [
        "train_auc", "valid_auc", "test_auc",
        "train_loss", "valid_loss", "test_loss"
    ]
    
    if mode == "train":
        reordered = fixed_cols + [f"pre_{m}" for m in metric_cols] + [f"post_{m}" for m in metric_cols]
    else:  # inference
        reordered = fixed_cols + [f"inference_{m}" for m in metric_cols]
    
    final_df = final_df[[col for col in reordered if col in final_df.columns]]

    return final_df

def get_best_configs(df, base_dir, model_name, mode="train"):
    if df.empty:
        logger.warning("Empty DataFrame passed to get_best_configs.")
        return pd.DataFrame()
    
    target_counts = df["target"].value_counts()
    if target_counts.nunique() > 1:
        logger.warning("Not all targets have the same number of rows.")
        logger.warning(target_counts)

    if mode == "train":
        best_results = df.groupby("target").apply(lambda x: x.loc[x['post_valid_auc'].idxmax()]).reset_index(drop=True)
        best_results.sort_values(by="post_test_auc", ascending=False, inplace=True)
    else:  # inference
        best_results = df.copy()
        best_results.sort_values(by="inference_test_auc", ascending=False, inplace=True)

    def process_ci_file(target_path, data_type, formatted_lr, epochs, gradientsteps, mode):
        """Process a single CI file and return the CI string."""
        if mode == "train":
            pattern = f"*post_finetune*_{data_type}_*lr-{formatted_lr}_epochs-{int(epochs)}_batchsizetrain-*_gradientsteps-{int(gradientsteps)}*.csv"
        else:  # inference
            pattern = f"*inference*_{data_type}_*lr-{formatted_lr}_epochs-{int(epochs)}_batchsizetrain-*_gradientsteps-{int(gradientsteps)}*.csv"
        
        search_path = os.path.join(target_path, pattern)
        matching_files = glob.glob(search_path)

        if len(matching_files) == 1:
            try:
                df = pd.read_csv(matching_files[0])
                lower, upper = compute_AUC_CI(df)
                return f'[{lower:.3f},{upper:.3f}]', matching_files[0]
            except Exception as e:
                logger.error(f"Failed to process {matching_files[0]}: {e}")
                return None, None
        else:
            logger.warning(f"Expected 1 file for {data_type}, found {len(matching_files)}: {matching_files}")
            return None, None

    # Compute CI and model paths
    if mode == "train":
        train_CI = []
        saved_model_path = []
    test_CI = []
    path_to_predictions_train = []
    path_to_predictions_test = []

    for _, row in best_results.iterrows():
        target = row["target"]
        lr = row["lr"]
        formatted_lr = format_lr(lr)
        epochs = row["epochs"]
        gradientsteps = row["gradientsteps"]

        target_path = os.path.join(base_dir, target)
        if not os.path.isdir(target_path):
            logger.warning(f"Target directory not found: {target_path}")
            if mode == "train":
                train_CI.append(None)
                test_CI.append(None)
                saved_model_path.append(None)
            else:
                test_CI.append(None)
            continue

        if mode == "train":
            # Get saved model path
            saved_fname = f"LLM-{model_name}_lr-{formatted_lr}_epochs-{int(epochs)}_batchsizetrain-*_gradientsteps-{int(gradientsteps)}"
            matching_dirs = glob.glob(os.path.join(target_path, saved_fname))
            saved_model_path.append(matching_dirs[0] if matching_dirs else None)

            # Process train and test CI files
            train_ci, pattern_train = process_ci_file(target_path, "train", formatted_lr, epochs, gradientsteps, mode)
            test_ci, pattern_test = process_ci_file(target_path, "test", formatted_lr, epochs, gradientsteps, mode)
            
            train_CI.append(train_ci)
            test_CI.append(test_ci)
        else:  # inference
            # Process only test CI files
            test_ci, pattern_test = process_ci_file(target_path, "test", formatted_lr, epochs, gradientsteps, mode)
            test_CI.append(test_ci)
            pattern_train = None
        path_to_predictions_train.append(pattern_train)
        path_to_predictions_test.append(pattern_test)

    # Add results to dataframe
    if mode == "train":
        best_results["train_CI"] = train_CI
        best_results["test_CI"] = test_CI
        best_results["saved_model_path"] = saved_model_path
    else:  # inference
        best_results["inference_CI"] = test_CI
    
    best_results['path_to_predictions_train'] = path_to_predictions_train
    best_results['path_to_predictions_test'] = path_to_predictions_test

    return best_results

def plot_delta_auc(best_df, save_path):
    best_df['delta_train_auc'] = best_df['post_train_auc'] - best_df['pre_train_auc']
    best_df['delta_valid_auc'] = best_df['post_valid_auc'] - best_df['pre_valid_auc']
    best_df['delta_test_auc'] = best_df['post_test_auc'] - best_df['pre_test_auc']

    df_delta = best_df.groupby("target")[
        ["delta_train_auc", "delta_valid_auc", "delta_test_auc"]
    ].mean().reset_index()

    df_long = df_delta.melt(id_vars='target', var_name='set', value_name='delta_auc')
    df_long['set'] = df_long['set'].str.replace('delta_', '').str.replace('_auc', '')
    df_long['target'] = df_long['target'].map(lambda x: target_dict_mapping[x.replace('_', '-')])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_long, x='target', y='delta_auc', hue='set', palette='colorblind')
    plt.axhline(0, color='gray', linestyle='--')
    plt.ylabel('Δ AUC (Post - Pre)')
    plt.xlabel('Target')
    plt.title('Change in AUC After Finetuning')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Set')
    plt.tight_layout()
    plt.savefig(f"{save_path}/change_in_auc_after_finetuning.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved AUC delta plot to {save_path}/change_in_auc_after_finetuning.png")

def plot_all_loss_curves(best_df, base_dir, save_dir):
    for _, row in best_df.iterrows():
        target = row["target"]
        lr = row["lr"]
        epochs = row["epochs"]
        gradientsteps = row["gradientsteps"]

        target_path = os.path.join(base_dir, target)
        if not os.path.isdir(target_path):
            logger.warning(f"Target directory not found: {target_path}")
            continue

        formatted_lr = format_lr(lr)
        pattern = f"*loss_log*lr-{formatted_lr}_epochs-{int(epochs)}_batchsizetrain-*_gradientsteps-{int(gradientsteps)}*.csv"
        search_path = os.path.join(target_path, pattern)
        matching_files = glob.glob(search_path)

        if len(matching_files) == 1:
            try:
                df = pd.read_csv(matching_files[0])
                logger.info(f"Plotting: {matching_files[0]}")
                filename = f"{target}_lr-{formatted_lr}_epochs-{epochs}_gradientsteps-{gradientsteps}"
                plot_loss_vs_step(df, os.path.join(save_dir, filename))
            except Exception as e:
                logger.error(f"Failed to plot {matching_files[0]}: {e}")
        else:
            logger.warning(f"Expected 1 file, found {len(matching_files)}: {matching_files}")

def main(base_dir, save_dir, model_name, mode="train", path_to_best_train=None):
    os.makedirs(save_dir, exist_ok=True)
    df = collect_finetune_metrics(base_dir, mode)

    if df.empty:
        logger.error("No valid result pairs found.")
        return

    best_df = get_best_configs(df, base_dir, model_name, mode)
    
    if mode == "train":
        best_df.to_csv(os.path.join(save_dir, "best_finetune_results.csv"), index=False)
        logger.info(f"Saved best configurations to {save_dir}/best_finetune_results.csv")
        
        plot_delta_auc(best_df, save_dir)
        plot_all_loss_curves(best_df, base_dir, save_dir)
        
        # make adjustments to saving best_df for comparison with other methods
        # rename post_train_auc to auc_train and post_test_auc to auc_test
        best_df = best_df.rename(columns={
            "post_train_auc": "auc_train",
            "post_test_auc": "auc_test"
        })
        
    else:  # inference
        # rename inference_test_auc to auc_inference
        best_df = best_df.rename(columns={
            "inference_test_auc": "auc_inference"
        })
        
        # Load and merge with training results if path provided
        if path_to_best_train:
            try:
                train_df = pd.read_csv(path_to_best_train)
                # Select only the columns we need
                train_subset = train_df[["target", "auc_train", "train_CI", "auc_test", "test_CI"]].copy()
                # Merge with inference results
                best_df = best_df.merge(train_subset, on="target", how="left")
                logger.info(f"Successfully merged with training results from {path_to_best_train}")
            except Exception as e:
                logger.error(f"Failed to load or merge training results from {path_to_best_train}: {e}")
    
    best_df.to_csv(os.path.join(save_dir, "best_finetune_results_for_comparison.csv"), index=False)

    # drop any columns with _CI in the name
    best_df = best_df[[col for col in best_df.columns if "_CI" not in col]]
    best_df.to_csv(os.path.join(save_dir, "best_finetune_results_no_CI.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot figures for finetuning results")
    parser.add_argument("base_dir", type=str, help="Base directory where results are saved")
    parser.add_argument("save_dir", type=str, help="Directory where to save plots and CSVs")
    parser.add_argument("model_name", type=str, help="LLM name")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="Mode: train or inference")
    parser.add_argument("--path_to_best_train", type=str, default=None, help="Path to CSV file containing best training results (for inference mode)")
    args = parser.parse_args()
    main(args.base_dir, args.save_dir, args.model_name, args.mode, args.path_to_best_train)