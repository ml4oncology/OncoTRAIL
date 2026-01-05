import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import itertools
from scipy.stats import wilcoxon, ranksums
from scipy.stats import gaussian_kde
import os
import pandas as pd
import glob
import argparse
import ast
import logging

logger = logging.getLogger(__name__)
def generate_mixed_violin_plot(aggregate_statistics_df, col_name, jitter, 
                               xtick_labels=None, plot_title="Violin Plot",
                               use_paired_test=True, base_cols=None,
                               print_all_pvalues=False):
    """
    Generate a mixed violin plot with statistical significance testing.
    
    Parameters:
    -----------
    aggregate_statistics_df : pd.DataFrame
        DataFrame containing the data to plot
    col_name : str
        Column name to group by (x-axis categories)
    jitter : int
        If 0, show KDE plots on right side; if non-zero, show jittered scatter points
    xtick_labels : list, optional
        Custom labels for x-axis ticks
    plot_title : str
        Title for the plot
    use_paired_test : bool
        If True, use Wilcoxon signed-rank test (paired); if False, use Mann-Whitney U (unpaired)
    base_cols : list, required when use_paired_test=True
        All columns that could be used for grouping. The function will exclude col_name from 
        this list to create the final grouping columns for the pivot table.
        Must be provided when use_paired_test=True.
    print_all_pvalues : bool
        If True, print all p-values; if False, only print significant ones
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot figure
    """


    # Parameters
    base_color = '#666666'
    target_type_colors = {
        'clinic': '#d95f02',
        'lab': '#1b9e77',
        'symptom': '#7570b3',
    }
    violin_width = 0.4
    all_patch = mpatches.Patch(color=base_color, label='all')

    # Unique values
    ys = sorted(aggregate_statistics_df[col_name].unique())
    target_types = aggregate_statistics_df['target_type'].unique()

    # Statistical testing
    pairwise_results = []
    
    if use_paired_test:
        # Use Wilcoxon signed-rank test (paired)
        if base_cols is None:
            raise ValueError("base_cols must be provided when use_paired_test=True")
        
        # Create grouping columns by excluding col_name from base_cols
        grouping_cols = [col for col in base_cols if col != col_name]
        
        # Create pivot table for paired testing
        pivot_df = aggregate_statistics_df.pivot_table(
            index=grouping_cols, 
            columns=col_name, 
            values='AUC', 
            aggfunc='first'
        )
        pivot_df = pivot_df.dropna()
        
        # Perform pairwise Wilcoxon tests
        for a, b in itertools.combinations(ys, 2):
            try:
                stat, p = wilcoxon(pivot_df[a], pivot_df[b], alternative='two-sided')
            except ValueError:
                p = 1.0
            pairwise_results.append(((a, b), p))
    else:
        # Use Mann-Whitney U test (unpaired)
        for a, b in itertools.combinations(ys, 2):
            try:
                auc_a = aggregate_statistics_df[aggregate_statistics_df[col_name] == a]['AUC']
                auc_b = aggregate_statistics_df[aggregate_statistics_df[col_name] == b]['AUC']
                stat, p = ranksums(auc_a, auc_b, alternative='two-sided')
            except ValueError:
                p = 1.0
            pairwise_results.append(((a, b), p))

    # Print p-values and mean differences
    if print_all_pvalues:
        for (a, b), p in pairwise_results:
            if use_paired_test:
                mean_a = pivot_df[a].mean()
                mean_b = pivot_df[b].mean()
            else:
                mean_a = aggregate_statistics_df[aggregate_statistics_df[col_name] == a]['AUC'].mean()
                mean_b = aggregate_statistics_df[aggregate_statistics_df[col_name] == b]['AUC'].mean()
            print(f"Mean difference between {a} and {b}: {mean_a - mean_b:.3f}")
            print(f"p-value between {a} and {b}: {p:.3g}")

    # Bonferroni correction
    n_tests = len(pairwise_results)
    corrected_results = [((a, b), p, p * n_tests) for (a, b), p in pairwise_results]
    alpha = 0.05
    significant_pairs = [((a, b), p_corrected) for (a, b), _, p_corrected in corrected_results if p_corrected < alpha]

    # Print significant results
    for (a, b), p_corrected in significant_pairs:
        if use_paired_test:
            mean_a = pivot_df[a].mean()
            mean_b = pivot_df[b].mean()
        else:
            mean_a = aggregate_statistics_df[aggregate_statistics_df[col_name] == a]['AUC'].mean()
            mean_b = aggregate_statistics_df[aggregate_statistics_df[col_name] == b]['AUC'].mean()
        print(f"Significant: Mean difference between {a} and {b}: {mean_a - mean_b:.3f}")

    # Create plot
    fig, ax = plt.subplots(figsize=(len(ys)*2.5, 6))
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    for i, y_val in enumerate(ys):
        x_center = i
        
        # Left half: KDE of AUC for all target_types at Y=y_val
        auc_all = aggregate_statistics_df[aggregate_statistics_df[col_name] == y_val]['AUC']
        kde = gaussian_kde(auc_all)
        auc_vals = np.linspace(0, 1, 200)
        densities = kde(auc_vals)
        densities = densities / densities.max() * violin_width
        
        ax.fill_betweenx(
            auc_vals, 
            x_center - densities, 
            x_center, 
            facecolor=base_color, 
            alpha=0.6
        )

        # Add box plot elements
        q1, q2, q3 = np.percentile(auc_all, [25, 50, 75])
        iqr = q3 - q1
        lw = max(q1 - 1.5 * iqr, auc_all.min())
        uw = min(q3 + 1.5 * iqr, auc_all.max())
        
        # Whiskers
        ax.vlines(x_center - 0.1, lw, uw, color='gray', linestyle='dashed', linewidth=0.5)
        # IQR box
        ax.vlines(x_center - 0.1, q1, q3, color='black', linewidth=1)
        # Median line
        ax.hlines(q2, x_center - 0.12, x_center - 0.08, color='black', linewidth=1.5)

        # Right half: either KDE or jittered points by target type
        if jitter == 0:
            # KDE plots for each target type
            for tt in target_types:
                auc_tt = aggregate_statistics_df[
                    (aggregate_statistics_df[col_name] == y_val) & 
                    (aggregate_statistics_df['target_type'] == tt)
                ]['AUC']
                
                if len(auc_tt) < 2:
                    continue
                
                kde_tt = gaussian_kde(auc_tt)
                densities_tt = kde_tt(auc_vals)
                densities_tt = densities_tt / densities_tt.max() * violin_width / len(target_types)
                
                ax.fill_betweenx(
                    auc_vals,
                    x_center,
                    x_center + densities_tt,
                    facecolor=target_type_colors[tt],
                    alpha=0.8,
                    label=tt if i == 0 else None
                )
        else:
            # Jittered scatter points
            jitter_scale = 0.04
            for tt in target_types:
                auc_tt = aggregate_statistics_df[
                    (aggregate_statistics_df[col_name] == y_val) & 
                    (aggregate_statistics_df['target_type'] == tt)
                ]['AUC']
                
                if auc_tt.empty:
                    continue
                
                x_vals = np.random.normal(loc=x_center + 0.1, scale=jitter_scale, size=len(auc_tt))
                ax.scatter(
                    x_vals, auc_tt,
                    alpha=0.7, color=target_type_colors[tt],
                    edgecolor='black', linewidth=0.3, s=20,
                    label=tt if i == 0 else None
                )

    # Add significance annotations
    x_positions = {x_val: i for i, x_val in enumerate(ys)}
    above_positions, below_positions = [], []
    above_base, below_base, step = 0.85, 0.35, 0.04

    for idx, ((a, b), p_val) in enumerate(significant_pairs):
        x1, x2 = x_positions[a], x_positions[b]
        x_center = (x1 + x2) / 2
        
        if idx % 2 == 0:
            height = above_base - len(above_positions) * step
            text_va, text_offset = 'bottom', 0.01
            above_positions.append(height)
        else:
            height = below_base + len(below_positions) * step
            text_va, text_offset = 'top', -0.01
            below_positions.append(height)

        ax.plot([x1, x1, x2, x2], [height, height + text_offset, height + text_offset, height],
                lw=1.2, c='black')
        ax.text(x_center, height + (1.5 * text_offset), f"p = {p_val:.3g}",
                ha='center', va=text_va, fontsize=14)

    # Format axes
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels(xtick_labels if xtick_labels else ys, rotation=15, ha='right', fontsize=14)
    ax.set_ylabel('AUC', fontsize=18)
    ax.set_ylim(0.2, 1)
    ax.set_title(plot_title, fontsize=18)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if 'all' not in labels:
        handles = [all_patch] + handles
        labels = ['all'] + labels
    labels = [label.capitalize() for label in labels]

    ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=12
    )

    plt.tight_layout()
    plt.show()

    return fig

# Utility functions and mappings
def target_category(target):
    if 'grade2plus' in target:
        return 'lab'
    elif 'change' in target:
        return 'symptom'
    else:
        return 'clinic'

def prompt_meaning(stats_df, prompt_num):
    if prompt_num < 24:
        stats_df['task_phrase'] = 'non-simplified'
    else:
        stats_df['task_phrase'] = 'simplified'

    if prompt_num % 2 == 0:
        stats_df['cot'] = 'no cot'
    else:
        stats_df['cot'] = 'cot'

    if prompt_num in [8, 9, 32, 33]:
        stats_df['persona'] = 'med-onc'
    elif prompt_num in [16, 17, 40, 41]:
        stats_df['persona'] = 'ai-model'

    return stats_df

def few_shot_mapping(val):
    if val > 8:
        return 16
    elif val > 4:
        return 8
    elif val > 0:
        return 4
    else:
        return 0

quantization_map = {"IQ2": 1, "IQ3": 2, "IQ4": 3, "Q4": 4, "Q6": 5, "Q8": 6}

# Main function
def load_aggregate_statistics(results_dir_parent, target_list_master, stage, save_string, save_dir):
    stage_columns = {
        "stage1": ['Target', 'AUC', 'n_samples', 'mean_proba', 'CI', 'LLM_name', 'prompt_num', 'tabular', 'n_few_shot', 'target_type', 'task_phrase', 'cot', 'persona'],
        "stage2": ['Target', 'AUC', 'n_samples', 'mean_proba', 'CI', 'n_few_shot_added_mean', 'LLM_name', 'n_params', 'quantization_level', 'quant_ranking', 'n_few_shot', 'target_type'],
        "stage3": ['Target', 'AUC', 'n_samples', 'mean_proba', 'CI', 'n_few_shot_added_mean', 'LLM_name', 'temp', 'top_p', 'min_p', 'top_k', 'n_few_shot', 'target_type'],
        "train": ['Target', 'AUC', 'n_samples', 'mean_proba', 'CI', 'path_to_predictions'],
        "test": ['Target', 'AUC', 'n_samples', 'mean_proba', 'CI', 'path_to_predictions'],
    }

    assert stage in stage_columns, f"Invalid stage: {stage}. Must be one of {list(stage_columns.keys())}"
    cols_to_keep = stage_columns[stage]

    aggregate_statistics = []

    for target in target_list_master:
        print(f"Processing target: {target}")
        target_results_dir = os.path.join(results_dir_parent, target)
        if not os.path.exists(target_results_dir):
            continue
        for subdir in os.listdir(target_results_dir):
            if not ('note_anchored' in subdir or 'note_tabular_anchored' in subdir):
                continue

            full_path = os.path.join(target_results_dir, subdir)
            stats_file = os.path.join(full_path, "statistics.csv")
            if not os.path.exists(stats_file):
                print(f"Skipping {stats_file} because statistics.csv does not exist.")
                continue

            stats_df = pd.read_csv(stats_file, index_col=0)
            split_params = subdir.split('_')

            try:
                LLM_name = split_params[-12]
                prompt_num = int(split_params[-5])
                stats_df['LLM_name'] = LLM_name
                stats_df['prompt_num'] = prompt_num
                stats_df['temp'] = split_params[-1]
                stats_df['top_p'] = split_params[-2]
                stats_df['min_p'] = split_params[-3]
                stats_df['top_k'] = split_params[-4]
                stats_df['n_few_shot'] = split_params[-7]
            except (IndexError, ValueError):
                print(f"Skipping {stats_file} because it is malformed.")
                continue  # skip malformed directory names

            stats_df['tabular'] = 'note-tabular' if 'tabular' in subdir else 'note'
            stats_df = prompt_meaning(stats_df, prompt_num)

            if 'Qwen' in LLM_name:
                try:
                    LLM_parts = LLM_name.split('-')
                    stats_df["n_params"] = int(LLM_parts[1].replace("B", ""))
                    stats_df["quantization_level"] = LLM_parts[2]
                    stats_df["quant_ranking"] = quantization_map[LLM_parts[2]]
                except (IndexError, KeyError, ValueError):
                    stats_df["n_params"] = np.nan
                    stats_df["quantization_level"] = np.nan
                    stats_df["quant_ranking"] = np.nan
            else:
                stats_df["n_params"] = np.nan
                stats_df["quantization_level"] = np.nan
                stats_df["quant_ranking"] = np.nan
            
            # get summary file path
            target_name = target.replace("_", "-")
            matching_files = glob.glob(os.path.join(full_path, f"summary_{target_name}_*.csv"))
            stats_df['path_to_predictions'] = matching_files[0]

            aggregate_statistics.append(stats_df)

    if not aggregate_statistics:
        return pd.DataFrame(columns=cols_to_keep)

    aggregate_statistics_df = pd.concat(aggregate_statistics).reset_index(drop=True)

    if 'n_few_shot_added_mean' in aggregate_statistics_df.columns:
        aggregate_statistics_df.loc[aggregate_statistics_df['n_few_shot_added_mean'].isna(), 'n_few_shot_added_mean'] = 0
        aggregate_statistics_df['n_few_shot'] = aggregate_statistics_df['n_few_shot_added_mean'].apply(few_shot_mapping)

    if 'Target' in aggregate_statistics_df.columns:
        aggregate_statistics_df['target_type'] = aggregate_statistics_df['Target'].apply(target_category)

    # return aggregate_statistics_df[cols_to_keep]
    aggregate_statistics_df[cols_to_keep].to_csv(f"{save_dir}/aggregate_statistics_{save_string}.csv", index=False)

def concatenate_train_test_inference(
    results_dir_train,
    results_dir_test,
    results_dir_inference,
    save_dir
):
    train_file = os.path.join(results_dir_train, "aggregate_statistics_train.csv")
    test_file = os.path.join(results_dir_test, "aggregate_statistics_test.csv")
    inference_file = os.path.join(
        results_dir_inference, "aggregate_statistics_inference.csv"
    )

    # -------------------------
    # Existence check
    # -------------------------
    missing_files = [
        f for f in [train_file, test_file, inference_file] if not os.path.exists(f)
    ]
    if missing_files:
        logger.error(f"Missing aggregate statistics files: {missing_files}")
        return

    # -------------------------
    # Read CSVs
    # -------------------------
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    inference_df = pd.read_csv(inference_file)

    # -------------------------
    # Merge train + test
    # -------------------------
    merged_df = pd.merge(
        train_df,
        test_df,
        on="Target",
        suffixes=("_train", "_test")
    )

    # -------------------------
    # Merge inference
    # -------------------------
    merged_df = pd.merge(
        merged_df,
        inference_df,
        on="Target",
        suffixes=("", "_inference")
    )

    # -------------------------
    # Rename columns
    # -------------------------
    merged_df = merged_df.rename(columns={
        "Target": "target",
        "AUC_train": "auc_train",
        "CI_train": "train_ci",
        "AUC_test": "auc_test",
        "CI_test": "test_ci",
        "AUC": "auc_inference",
        "CI": "inference_ci",
    })

    # -------------------------
    # Save
    # -------------------------
    os.makedirs(save_dir, exist_ok=True)
    merged_df.to_csv(
        os.path.join(save_dir, "prompting_results_train_test_inference.csv"),
        index=False
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Utilities CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- Mode 1: aggregate ---
    agg = subparsers.add_parser("aggregate", help="Aggregate statistics")
    agg.add_argument("results_dir", type=str)
    agg.add_argument("target_list", type=str)
    agg.add_argument("stage", type=str)
    agg.add_argument("save_string", type=str)
    agg.add_argument("save_dir", type=str)

    # --- Mode 2: concatenate train and test results ---
    concat = subparsers.add_parser("concatenate", help="Train vs Test vs Inference")
    concat.add_argument("results_dir_train", type=str)
    concat.add_argument("results_dir_test", type=str)
    concat.add_argument("results_dir_inference", type=str)
    concat.add_argument("save_dir", type=str)

    args = parser.parse_args()

    if args.mode == "aggregate":
        target_list = ast.literal_eval(args.target_list)
        load_aggregate_statistics(
            args.results_dir,
            target_list,
            args.stage,
            args.save_string,
            args.save_dir
        )

    elif args.mode == "concatenate":
        concatenate_train_test_inference(
            args.results_dir_train, 
            args.results_dir_test, 
            args.results_dir_inference,
            args.save_dir
        )