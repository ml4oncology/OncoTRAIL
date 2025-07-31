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


def prepare_dataframe(df: pd.DataFrame, method_name: str) -> pd.DataFrame:
    df = df.copy()
    df['target'] = df['target'].str.replace('_', '-')
    df.columns = df.columns.str.lower()
    df['method'] = method_name
    df = parse_ci_column(df, 'test_ci', 'test')
    df = parse_ci_column(df, 'train_ci', 'train')
    df['color'] = df['target'].apply(get_color)
    return df


def plot_train_vs_test(df: pd.DataFrame, ax: plt.Axes, label_map: Dict[str, str], min_val: float, max_val: float) -> None:
    texts = []
    diffs = []
    for _, row in df.iterrows():
        # Collect absolute difference
        diffs.append(abs(row['auc_train'] - row['auc_test']))

        ax.errorbar(
            row['auc_train'], row['auc_test'],
            xerr=[[row['auc_train'] - row['train_CI_low']], [row['train_CI_high'] - row['auc_train']]],
            yerr=[[row['auc_test'] - row['test_CI_low']], [row['test_CI_high'] - row['auc_test']]],
            fmt='o', color=row['color'], alpha=0.3
        )
        ax.scatter(row['auc_train'], row['auc_test'], color=row['color'], edgecolors='black')
        texts.append(ax.text(row['auc_train'], row['auc_test'], label_map.get(row['target'], row['target']), fontsize=9))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)

    # Compute and show generalization error
    mean_diff = np.mean(diffs)

    ax.set_xlabel('Train AUC', fontsize=12)
    ax.set_ylabel('Test AUC', fontsize=12)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    adjust_text(texts, ax=ax)

    return mean_diff

def plot_test_vs_test(df1: pd.DataFrame, df2: pd.DataFrame, ax: plt.Axes, label_map: Dict[str, str], min_val: float, max_val: float) -> None:
    texts = []
    diffs = []
    merged = pd.merge(df1, df2, on='target', suffixes=('_1', '_2'))
    for _, row in merged.iterrows():
        # Collect absolute difference
        diffs.append(abs(row['auc_test_1'] - row['auc_test_2']))

        ax.errorbar(
            row['auc_test_1'], row['auc_test_2'],
            xerr=[[row['auc_test_1'] - row['test_CI_low_1']], [row['test_CI_high_1'] - row['auc_test_1']]],
            yerr=[[row['auc_test_2'] - row['test_CI_low_2']], [row['test_CI_high_2'] - row['auc_test_2']]],
            fmt='o', color=row['color_1'], alpha=0.3
        )
        ax.scatter(row['auc_test_1'], row['auc_test_2'], color=row['color_1'], edgecolors='black')
        texts.append(ax.text(row['auc_test_1'], row['auc_test_2'], label_map.get(row['target'], row['target']), fontsize=9))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.6)

    # Compute and show generalization error
    mean_diff = np.mean(diffs)

    ax.set_xlabel(f"{row['method_1']} Test AUC", fontsize=12)
    ax.set_ylabel(f"{row['method_2']} Test AUC", fontsize=12)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    adjust_text(texts, ax=ax)

    return mean_diff

def get_axis_limits(dfs: List[pd.DataFrame]) -> Tuple[float, float]:
    all_lows = []
    all_highs = []
    for df in dfs:
        all_lows.extend(df[['train_CI_low', 'test_CI_low']].min().values)
        all_highs.extend(df[['train_CI_high', 'test_CI_high']].max().values)
    min_val = max(0, min(all_lows) - 0.05)
    max_val = min(1, max(all_highs) + 0.05)
    return min_val, max_val


def plot_comparison_grid(method_dfs: Dict[str, pd.DataFrame], label_map: Dict[str, str], save_dir) -> None:
    method_names = list(method_dfs.keys())
    n = len(method_names)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))
    min_val, max_val = get_axis_limits(list(method_dfs.values()))

    for i in range(n):
        logger.info(f"Plotting {method_names[i]}")
        for j in range(n):
            logger.info(f"Plotting {method_names[j]}")
            ax = axes[i, j]
            if i == j:
                mean_diff = plot_train_vs_test(method_dfs[method_names[i]], ax, label_map, min_val, max_val)
                ax.set_title(f"{method_names[i]}: Mean Abs. Diff = {mean_diff:.3f}", fontsize=12)
            else:
                mean_diff =plot_test_vs_test(method_dfs[method_names[j]], method_dfs[method_names[i]], ax, label_map, min_val, max_val)
                ax.set_title(f"{method_names[j]} vs {method_names[i]}: Mean Abs. Diff = {mean_diff:.3f}", fontsize=12)

            # ax.set_xlim(min_val, max_val)
            # ax.set_ylim(min_val, max_val)

    legend_elements = [
        mpatches.Patch(color='#1b9e77', label='Lab'),
        mpatches.Patch(color='#7570b3', label='Symptom'),
        mpatches.Patch(color='#d95f02', label='Clinic')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=12, frameon=True)

    # Compose filename from method names
    filename = "_".join(method_names) + ".png"
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.tight_layout()
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
    args = parser.parse_args()

    method_csvs = parse_method_csv_args(args.methods)

    # Load and prepare each method's dataframe
    method_dfs = {
        method: prepare_dataframe(pd.read_csv(csv_path), method)
        for method, csv_path in method_csvs.items()
    }

    # Generate the plot
    plot_comparison_grid(method_dfs, target_dict_mapping, args.save_dir)
    # need to edit for saving the plot

if __name__ == "__main__":
    main()
