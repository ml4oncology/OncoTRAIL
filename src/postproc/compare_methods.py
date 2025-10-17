import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import os
from llm_notes_classification.constants import target_dict_mapping

import logging
import sys
# Setup logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def parse_ci_column(df: pd.DataFrame, ci_col: str, prefix: str) -> pd.DataFrame:
    ci_bounds = df[ci_col].str.strip('[]').str.split(',', expand=True).astype(float)
    df[f"{prefix}_CI_low"] = ci_bounds[0]
    df[f"{prefix}_CI_high"] = ci_bounds[1]
    return df


def get_color(target: str) -> str:
    if 'grade2plus' in target:
        return '#1b9e77'  # Lab
    elif 'change' in target:
        return '#7570b3'  # Symptom
    else:
        return '#d95f02'  # Clinic


def prepare_dataframe(df: pd.DataFrame, method_name: str, mode: str) -> pd.DataFrame:
    df = df.copy()
    df['target'] = df['target'].str.replace('_', '-')
    df.columns = df.columns.str.lower()
    df['method'] = method_name
    
    # Always parse train_ci, since we may need it even in inference mode
    if 'train_ci' in df.columns:
        df = parse_ci_column(df, 'train_ci', 'train')
    
    if mode == "train":
        if 'test_ci' in df.columns:
            df = parse_ci_column(df, 'test_ci', 'test')
    elif mode == "inference":
        if 'test_ci' in df.columns:
            df = parse_ci_column(df, 'test_ci', 'test')
        if 'inference_ci' in df.columns:
            df = parse_ci_column(df, 'inference_ci', 'inference')
    
    df['color'] = df['target'].apply(get_color)
    return df

def plot_train_vs_test(df: pd.DataFrame, ax: plt.Axes, label_map: Dict[str, str], min_val: float, max_val: float, mode: str) -> None:
    texts = []
    diffs = []
    
    if mode == "train":
        x_col, y_col = 'auc_train', 'auc_test'
        x_ci_low, x_ci_high = 'train_CI_low', 'train_CI_high'
        y_ci_low, y_ci_high = 'test_CI_low', 'test_CI_high'
        x_label, y_label = 'Train AUC', 'Test AUC'
    elif mode == "inference":
        x_col, y_col = 'auc_test', 'auc_inference'
        x_ci_low, x_ci_high = 'test_CI_low', 'test_CI_high'
        y_ci_low, y_ci_high = 'inference_CI_low', 'inference_CI_high'
        x_label, y_label = 'Test AUC', 'Inference AUC'
    elif mode == "train_inference":
        x_col, y_col = 'auc_train', 'auc_inference'
        x_ci_low, x_ci_high = 'train_CI_low', 'train_CI_high'
        y_ci_low, y_ci_high = 'inference_CI_low', 'inference_CI_high'
        x_label, y_label = 'Train AUC', 'Inference AUC'
    
    for _, row in df.iterrows():
        # Collect absolute difference
        diffs.append(abs(row[x_col] - row[y_col]))

        ax.errorbar(
            row[x_col], row[y_col],
            xerr=[[row[x_col] - row[x_ci_low]], [row[x_ci_high] - row[x_col]]],
            yerr=[[row[y_col] - row[y_ci_low]], [row[y_ci_high] - row[y_col]]],
            fmt='o', color=row['color'], alpha=0.3
        )
        ax.scatter(row[x_col], row[y_col], color=row['color'], edgecolors='black')
        texts.append(ax.text(row[x_col], row[y_col], label_map.get(row['target'], row['target']), fontsize=9))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)

    # Compute and show generalization error
    mean_diff = np.mean(diffs)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    adjust_text(texts, ax=ax)

    return mean_diff

def plot_test_vs_test(df1: pd.DataFrame, df2: pd.DataFrame, ax: plt.Axes, label_map: Dict[str, str], min_val: float, max_val: float, mode: str) -> None:
    texts = []
    diffs = []
    merged = pd.merge(df1, df2, on='target', suffixes=('_1', '_2'))
    
    if mode == "train":
        x_col, y_col = 'auc_test_1', 'auc_test_2'
        x_ci_low, x_ci_high = 'test_CI_low_1', 'test_CI_high_1'
        y_ci_low, y_ci_high = 'test_CI_low_2', 'test_CI_high_2'
        x_label, y_label = "Test AUC", "Test AUC"
    elif mode == "inference":
        x_col, y_col = 'auc_inference_1', 'auc_inference_2'
        x_ci_low, x_ci_high = 'inference_CI_low_1', 'inference_CI_high_1'
        y_ci_low, y_ci_high = 'inference_CI_low_2', 'inference_CI_high_2'
        x_label, y_label = "Inference AUC", "Inference AUC"
    
    for _, row in merged.iterrows():
        # Collect absolute difference
        diffs.append(abs(row[x_col] - row[y_col]))

        ax.errorbar(
            row[x_col], row[y_col],
            xerr=[[row[x_col] - row[x_ci_low]], [row[x_ci_high] - row[x_col]]],
            yerr=[[row[y_col] - row[y_ci_low]], [row[y_ci_high] - row[y_col]]],
            fmt='o', color=row['color_1'], alpha=0.3
        )
        ax.scatter(row[x_col], row[y_col], color=row['color_1'], edgecolors='black')
        texts.append(ax.text(row[x_col], row[y_col], label_map.get(row['target'], row['target']), fontsize=9))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)

    # Compute and show generalization error
    mean_diff = np.mean(diffs)

    ax.set_xlabel(f"{row['method_1']} {x_label}", fontsize=12)
    ax.set_ylabel(f"{row['method_2']} {y_label}", fontsize=12)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    adjust_text(texts, ax=ax)

    return mean_diff

def get_axis_limits(dfs: List[pd.DataFrame], mode: str) -> Tuple[float, float]:
    all_lows = []
    all_highs = []
    
    for df in dfs:
        if mode == "train":
            all_lows.extend(df[['train_CI_low', 'test_CI_low']].min().values)
            all_highs.extend(df[['train_CI_high', 'test_CI_high']].max().values)
        elif mode == "inference":
            all_lows.extend(df[['inference_CI_low', 'test_CI_low']].min().values)
            all_highs.extend(df[['inference_CI_high', 'test_CI_high']].max().values)
        elif mode == "train_inference":
            all_lows.extend(df[['train_CI_low', 'inference_CI_low']].min().values)
            all_highs.extend(df[['train_CI_high', 'inference_CI_high']].max().values)
    
    min_val = max(0, min(all_lows) - 0.05)
    max_val = min(1, max(all_highs) + 0.05)
    return min_val, max_val

def plot_comparison_grid(method_dfs: Dict[str, pd.DataFrame], label_map: Dict[str, str], save_dir: str, mode: str, diag_mode: str = None) -> None:
    """
    diag_mode controls what to plot along the diagonal when mode == 'inference':
    - None (default): plot auc_test vs auc_inference
    - 'train_inference': plot auc_train vs auc_inference
    """
    method_names = list(method_dfs.keys())
    n = len(method_names)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n), squeeze=False)
    min_val, max_val = get_axis_limits(list(method_dfs.values()), diag_mode if diag_mode else mode)

    logger.info(f"Axis limits: {min_val} to {max_val}")

    for i in range(n):
        logger.info(f"Plotting {method_names[i]}")
        for j in range(n):
            logger.info(f"Plotting {method_names[j]}")
            ax = axes[i, j]
            if i == j:
                if mode == "train":
                    mean_diff = plot_train_vs_test(method_dfs[method_names[i]], ax, label_map, min_val, max_val, mode)
                    ax.set_title(f"{method_names[i]}: Mean Abs. Diff = {mean_diff:.3f}", fontsize=12)
                elif mode == "inference":
                    # If diag_mode == 'train_inference', use auc_train vs auc_inference
                    if diag_mode == "train_inference":
                        mean_diff = plot_train_vs_test(method_dfs[method_names[i]], ax, label_map, min_val, max_val, "train_inference")
                        ax.set_title(f"{method_names[i]} (Train vs Inference): Mean Abs. Diff = {mean_diff:.3f}", fontsize=12)
                    else:
                        mean_diff = plot_train_vs_test(method_dfs[method_names[i]], ax, label_map, min_val, max_val, mode)
                        ax.set_title(f"{method_names[i]} (Test vs Inference): Mean Abs. Diff = {mean_diff:.3f}", fontsize=12)
            else:
                mean_diff = plot_test_vs_test(method_dfs[method_names[j]], method_dfs[method_names[i]], ax, label_map, min_val, max_val, mode)
                ax.set_title(f"{method_names[j]} vs {method_names[i]}: Mean Abs. Diff = {mean_diff:.3f}", fontsize=12)

    legend_elements = [
        mpatches.Patch(color='#1b9e77', label='Lab'),
        mpatches.Patch(color='#7570b3', label='Symptom'),
        mpatches.Patch(color='#d95f02', label='Clinic')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12, frameon=True)

    # Compose filename from method names
    # Filename logic

    filename = "_".join(method_names)
    if mode == "inference" and diag_mode == "train_inference":
        filename += "_train_inference.png"
    else:
        filename += f"_{mode}.png"
    output_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def parse_method_csv_args(arg: str) -> Dict[str, str]:
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


def main():
    parser = argparse.ArgumentParser(description="Compare AUCs for multiple methods")
    parser.add_argument('--methods', required=True, help="Comma-separated method=csv_path pairs")
    parser.add_argument('--save_dir', required=True, help="Directory to save the plot")
    parser.add_argument('--mode', required=True, choices=['train', 'inference'], help="Mode: 'train' or 'inference'")
    args = parser.parse_args()

    method_csvs = parse_method_csv_args(args.methods)

    # Load and prepare each method's dataframe
    method_dfs = {
        method: prepare_dataframe(pd.read_csv(csv_path), method, args.mode)
        for method, csv_path in method_csvs.items()
    }

    # Generate the plot
    plot_comparison_grid(method_dfs, target_dict_mapping, args.save_dir, args.mode)
    if args.mode == "inference":
        plot_comparison_grid(method_dfs, target_dict_mapping, args.save_dir, args.mode, diag_mode="train_inference")

    # need to edit for saving the plot

if __name__ == "__main__":
    main()