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
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"   # Keep text as text (not paths)
mpl.rcParams["pdf.fonttype"] = 42       # Editable text in PDF
mpl.rcParams["ps.fonttype"] = 42

logger = logging.getLogger(__name__)
def generate_mixed_violin_plot(aggregate_statistics_df, col_name, jitter, 
                               xtick_labels=None, plot_title="Violin Plot",
                               use_paired_test=True, base_cols=None,
                               print_all_pvalues=False,
                               fig_size_mm=None):
    
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
        Title for the plot (unused, kept for API compatibility)
    use_paired_test : bool
        If True, use Wilcoxon signed-rank test (paired); if False, use Mann-Whitney U (unpaired)
    base_cols : list, required when use_paired_test=True
        All columns that could be used for grouping. The function will exclude col_name from 
        this list to create the final grouping columns for the pivot table.
        Must be provided when use_paired_test=True.
    print_all_pvalues : bool
        If True, print all p-values; if False, only print significant ones
    fig_size_mm : tuple, optional
        Figure size as (width_mm, height_mm) for the first plot. If None, defaults to
        (len(ys)*2.5 inches, 6 inches) as before.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot figure
    """
        
    # Parameters
    base_color = '#666666'
    target_type_colors = {
        'clinic': '#88bb99',
        'lab': '#e6b3bb',
        'symptom': '#9a91c4',
    }
    violin_width = 0.4
    fontsize_axes = 4       # font size for x/y-axis labels and tick labels
    fontsize_legend = 4     # font size for legend and p-values
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

        # FEATURE 1: Print mean differences stratified by target type
        for tt in target_types:
            if use_paired_test:
                pivot_df_tt = aggregate_statistics_df[
                    aggregate_statistics_df['target_type'] == tt
                ].pivot_table(
                    index=grouping_cols,
                    columns=col_name,
                    values='AUC',
                    aggfunc='first'
                ).dropna()
                
                if a in pivot_df_tt.columns and b in pivot_df_tt.columns:
                    mean_a_tt = pivot_df_tt[a].mean()
                    mean_b_tt = pivot_df_tt[b].mean()
                    print(f"  {tt.capitalize()}: Mean difference = {mean_a_tt - mean_b_tt:.3f}")
            else:
                auc_a_tt = aggregate_statistics_df[
                    (aggregate_statistics_df[col_name] == a) & 
                    (aggregate_statistics_df['target_type'] == tt)
                ]['AUC']
                auc_b_tt = aggregate_statistics_df[
                    (aggregate_statistics_df[col_name] == b) & 
                    (aggregate_statistics_df['target_type'] == tt)
                ]['AUC']
                
                if len(auc_a_tt) > 0 and len(auc_b_tt) > 0:
                    mean_a_tt = auc_a_tt.mean()
                    mean_b_tt = auc_b_tt.mean()
                    print(f"  {tt.capitalize()}: Mean difference = {mean_a_tt - mean_b_tt:.3f}")

    # Create first plot
    if fig_size_mm is not None:
        figsize = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)
    else:
        figsize = (len(ys) * 2.5, 6)
    fig, ax = plt.subplots(figsize=figsize)

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
        )
        
        # Add box plot elements
        q1, q2, q3 = np.percentile(auc_all, [25, 50, 75])
        iqr = q3 - q1
        lw = max(q1 - 1.5 * iqr, auc_all.min())
        uw = min(q3 + 1.5 * iqr, auc_all.max())
        
        # Whiskers
        ax.vlines(x_center - 0.1, lw, uw, color='gray', linestyle='dashed', linewidth=0.8)
        # IQR box
        ax.vlines(x_center - 0.1, q1, q3, color='black', linewidth=1)
        # Median line
        ax.hlines(q2, x_center - 0.12, x_center - 0.08, color='black', linewidth=1.5)

        # Right half: either KDE or jittered points by target type
        if jitter == 0:
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
                    label=tt if i == 0 else None
                )
        else:
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
                    color=target_type_colors[tt],
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
                ha='center', va=text_va, fontsize=fontsize_legend)

    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels(xtick_labels if xtick_labels else ys, rotation=15, ha='right', fontsize=fontsize_axes)
    ax.set_ylabel('AUC', fontsize=fontsize_axes)
    ax.tick_params(axis='y', labelsize=fontsize_axes)
    ax.set_ylim(0.2, 1)

    # Remove all spines except left (y-axis) and bottom (x-axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    handles, labels = ax.get_legend_handles_labels()
    if 'all' not in labels:
        handles = [all_patch] + handles
        labels = ['all'] + labels
    labels = [label.lower() for label in labels]

    ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=fontsize_legend
    )
    plt.tight_layout()

    # FEATURE 2: Create delta AUC plot for paired test with exactly 2 conditions
    if use_paired_test and len(ys) == 2:
        print("\n" + "="*50)
        print("Creating Delta AUC Plot")
        print("="*50)
        
        baseline, comparison = sorted(ys)
        delta_col_name = f"Delta_AUC_{comparison}_vs_{baseline}"
        
        delta_df_list = []
        for idx, row in pivot_df.iterrows():
            delta_auc = row[comparison] - row[baseline]
            
            filter_dict = {k: v for k, v in zip(grouping_cols, idx if isinstance(idx, tuple) else [idx])}
            matching_rows = aggregate_statistics_df.copy()
            for key, val in filter_dict.items():
                matching_rows = matching_rows[matching_rows[key] == val]
            
            if not matching_rows.empty:
                target_type = matching_rows.iloc[0]['target_type']
                delta_df_list.append({
                    'delta_AUC': delta_auc,
                    'target_type': target_type,
                    'comparison': f"{comparison} vs {baseline}"
                })
        
        delta_df = pd.DataFrame(delta_df_list)
        
        # Create delta plot
        fig_delta, ax_delta = plt.subplots(figsize=(4, 6))
        ax_delta.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        x_center = 0
        
        # Left half: KDE of delta AUC for all target_types        
        delta_all = delta_df['delta_AUC']
        kde_delta = gaussian_kde(delta_all)
        delta_vals = np.linspace(delta_all.min() - 0.05, delta_all.max() + 0.05, 200)
        densities_delta = kde_delta(delta_vals)
        densities_delta = densities_delta / densities_delta.max() * violin_width
        
        ax_delta.fill_betweenx(
            delta_vals,
            x_center - densities_delta,
            x_center,
            facecolor=base_color,
        )

        # Add box plot elements
        q1_delta, q2_delta, q3_delta = np.percentile(delta_all, [25, 50, 75])
        iqr_delta = q3_delta - q1_delta
        lw_delta = max(q1_delta - 1.5 * iqr_delta, delta_all.min())
        uw_delta = min(q3_delta + 1.5 * iqr_delta, delta_all.max())
        
        # Whiskers
        ax_delta.vlines(x_center - 0.1, lw_delta, uw_delta, color='gray', linestyle='dashed', linewidth=0.8)
        # IQR box
        ax_delta.vlines(x_center - 0.1, q1_delta, q3_delta, color='black', linewidth=1)
        # Median line
        ax_delta.hlines(q2_delta, x_center - 0.12, x_center - 0.08, color='black', linewidth=1.5)
        
        # Right half: either KDE or jittered points by target type
        if jitter == 0:
            for tt in target_types:
                delta_tt = delta_df[delta_df['target_type'] == tt]['delta_AUC']
                
                if len(delta_tt) < 2:
                    continue
                
                kde_tt_delta = gaussian_kde(delta_tt)
                densities_tt_delta = kde_tt_delta(delta_vals)
                densities_tt_delta = densities_tt_delta / densities_tt_delta.max() * violin_width / len(target_types)
                
                ax_delta.fill_betweenx(
                    delta_vals,
                    x_center,
                    x_center + densities_tt_delta,
                    facecolor=target_type_colors[tt],
                    label=tt
                )
        else:
            jitter_scale = 0.04
            for tt in target_types:
                delta_tt = delta_df[delta_df['target_type'] == tt]['delta_AUC']
                
                if delta_tt.empty:
                    continue
                
                x_vals_delta = np.random.normal(loc=x_center + 0.1, scale=jitter_scale, size=len(delta_tt))
                ax_delta.scatter(
                    x_vals_delta, delta_tt,
                    color=target_type_colors[tt],
                    edgecolor='black', linewidth=0.3, s=20,
                    label=tt
                )
        
        ax_delta.set_xticks([0])
        delta_xlabel = xtick_labels[1] if xtick_labels and len(xtick_labels) >= 2 else comparison
        ax_delta.set_xticklabels([f"{delta_xlabel}\nvs {xtick_labels[0] if xtick_labels else baseline}"], fontsize=fontsize_axes)
        ax_delta.set_ylabel('Delta AUC', fontsize=fontsize_axes)
        ax_delta.tick_params(axis='y', labelsize=fontsize_axes)

        # Remove all spines except left (y-axis) and bottom (x-axis)
        ax_delta.spines['top'].set_visible(False)
        ax_delta.spines['right'].set_visible(False)
        ax_delta.spines['left'].set_linewidth(1.0)
        ax_delta.spines['bottom'].set_linewidth(1.0)
        
        # Add legend
        handles_delta, labels_delta = ax_delta.get_legend_handles_labels()
        if 'all' not in labels_delta:
            handles_delta = [all_patch] + handles_delta
            labels_delta = ['all'] + labels_delta
        labels_delta = [label.lower() for label in labels_delta]
        
        ax_delta.legend(
            handles_delta, labels_delta,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=fontsize_legend
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print delta statistics
        print(f"\nDelta AUC Statistics ({comparison} - {baseline}):")
        print(f"Overall mean delta: {delta_all.mean():.3f}")
        print(f"Overall median delta: {delta_all.median():.3f}")
        for tt in target_types:
            delta_tt = delta_df[delta_df['target_type'] == tt]['delta_AUC']
            if len(delta_tt) > 0:
                print(f"{tt.capitalize()} mean delta: {delta_tt.mean():.3f}")

    
    return fig
    
def generate_multi_delta_violin_plot(aggregate_statistics_df, comparisons, jitter,
                                     plot_title="Delta AUC Comparison",
                                     base_cols=None,
                                     fig_size_mm=(100, 150)):
    """
    Generate a multi-panel delta violin plot comparing multiple paired design configurations.
    
    Parameters:
    -----------
    aggregate_statistics_df : pd.DataFrame
        DataFrame containing the data to plot
    comparisons : list of dict
        List of comparison specifications. Each dict should contain:
        - 'col_name': str - column name to group by
        - 'labels': list - [baseline_label, comparison_label] for x-axis
        - 'title': str - title for this comparison column
    jitter : int
        If 0, show KDE plots on right side; if non-zero, show jittered scatter points
    plot_title : str
        Overall title for the plot (unused, kept for API compatibility)
    base_cols : list
        All columns that could be used for grouping
    fig_size_mm : tuple
        Figure size as (width_mm, height_mm). Default is (100, 150).
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot figure
    
    Example:
    --------
    comparisons = [
        {'col_name': 'persona', 'labels': ['ai-model', 'med-onc'], 'title': 'Persona'},
        {'col_name': 'cot', 'labels': ['cot', 'no-cot'], 'title': 'CoT'},
        {'col_name': 'task_phrase', 'labels': ['non-simplified', 'simplified'], 'title': 'Task Phrase'},
        {'col_name': 'tabular', 'labels': ['note', 'note-tabular'], 'title': 'Note Type'}
    ]
    fig = generate_multi_delta_violin_plot(agg_df_train, comparisons, 0,
                                           base_cols=base_cols,
                                           fig_size_mm=(100, 150))
    """
    
    # Parameters
    base_color = '#666666'
    target_type_colors = {
        'clinic': '#88bb99',
        'lab': '#e6b3bb',
        'symptom': '#9a91c4',
    }
    violin_width = 0.4
    all_patch = mpatches.Patch(color=base_color, label='all')
    fontsize_axes = 4
    fontsize_legend = 4
    target_types = aggregate_statistics_df['target_type'].unique()
    
    # Collect all delta AUC data for each comparison
    all_delta_data = []
    
    for comp_idx, comp in enumerate(comparisons):
        col_name = comp['col_name']
        baseline, comparison = comp['labels']
        comp_title = comp['title']
        
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
        
        # Calculate delta AUC and collect data
        for idx, row in pivot_df.iterrows():
            delta_auc = row[comparison] - row[baseline]
            
            # Get the original row to extract target_type
            filter_dict = {k: v for k, v in zip(grouping_cols, idx if isinstance(idx, tuple) else [idx])}
            matching_rows = aggregate_statistics_df.copy()
            for key, val in filter_dict.items():
                matching_rows = matching_rows[matching_rows[key] == val]
            
            # Get target_type from matching rows
            if not matching_rows.empty:
                target_type = matching_rows.iloc[0]['target_type']
                all_delta_data.append({
                    'delta_AUC': delta_auc,
                    'target_type': target_type,
                    'comparison_idx': comp_idx,
                    'comparison_name': comp_title,
                    'comparison_label': f"{comparison}\nvs\n{baseline}"
                })
        
        # Perform Wilcoxon test and print statistics
        try:
            stat, p = wilcoxon(pivot_df[comparison], pivot_df[baseline], alternative='two-sided')
        except ValueError:
            p = 1.0
        
        delta_vals = pivot_df[comparison] - pivot_df[baseline]
        print(f"\n{comp_title} ({comparison} vs {baseline}):")
        print(f"  Mean delta AUC: {delta_vals.mean():.3f}")
        print(f"  Median delta AUC: {delta_vals.median():.3f}")
        print(f"  Wilcoxon p-value: {p:.3g}")
        
        # Print stratified statistics
        for tt in target_types:
            pivot_df_tt = aggregate_statistics_df[
                aggregate_statistics_df['target_type'] == tt
            ].pivot_table(
                index=grouping_cols,
                columns=col_name,
                values='AUC',
                aggfunc='first'
            ).dropna()
            
            if comparison in pivot_df_tt.columns and baseline in pivot_df_tt.columns:
                delta_tt = pivot_df_tt[comparison] - pivot_df_tt[baseline]
                print(f"  {tt.capitalize()} mean delta: {delta_tt.mean():.3f}")
    
    delta_df = pd.DataFrame(all_delta_data)
    
    # Convert mm to inches for matplotlib
    fig_width_in = fig_size_mm[0] / 25.4
    fig_height_in = fig_size_mm[1] / 25.4

    # Create the plot
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    # Plot each comparison
    n_comparisons = len(comparisons)
    for i in range(n_comparisons):
        x_center = i
        
        # Get delta data for this comparison
        delta_comp = delta_df[delta_df['comparison_idx'] == i]['delta_AUC']
        
        # Determine y-axis range for KDE
        all_deltas = delta_df['delta_AUC']
        y_min = all_deltas.min() - 0.05
        y_max = all_deltas.max() + 0.05
        delta_vals = np.linspace(y_min, y_max, 200)
        
        # Left half: KDE of delta AUC for all target_types
        kde_delta = gaussian_kde(delta_comp)
        densities_delta = kde_delta(delta_vals)
        densities_delta = densities_delta / densities_delta.max() * violin_width
        
        ax.fill_betweenx(
            delta_vals,
            x_center - densities_delta,
            x_center,
            facecolor=base_color,
            # alpha=0.6,
            zorder=2
        )
        
        # Add box plot elements
        q1_delta, q2_delta, q3_delta = np.percentile(delta_comp, [25, 50, 75])
        iqr_delta = q3_delta - q1_delta
        lw_delta = max(q1_delta - 1.5 * iqr_delta, delta_comp.min())
        uw_delta = min(q3_delta + 1.5 * iqr_delta, delta_comp.max())
        
        # Whiskers
        ax.vlines(x_center - 0.1, lw_delta, uw_delta, color='gray', linestyle='dashed', linewidth=0.8, zorder=3)
        # IQR box
        ax.vlines(x_center - 0.1, q1_delta, q3_delta, color='black', linewidth=1, zorder=3)
        # Median line
        ax.hlines(q2_delta, x_center - 0.12, x_center - 0.08, color='black', linewidth=1.5, zorder=3)
        
        # Right half: either KDE or jittered points by target type
        if jitter == 0:
            # KDE plots for each target type
            for tt in target_types:
                delta_tt = delta_df[
                    (delta_df['comparison_idx'] == i) & 
                    (delta_df['target_type'] == tt)
                ]['delta_AUC']
                
                if len(delta_tt) < 2:
                    continue
                
                kde_tt_delta = gaussian_kde(delta_tt)
                densities_tt_delta = kde_tt_delta(delta_vals)
                densities_tt_delta = densities_tt_delta / densities_tt_delta.max() * violin_width / len(target_types)
                
                ax.fill_betweenx(
                    delta_vals,
                    x_center,
                    x_center + densities_tt_delta,
                    facecolor=target_type_colors[tt],
                    # alpha=0.8,
                    label=tt if i == 0 else None,
                    zorder=2
                )
        else:
            # Jittered scatter points
            jitter_scale = 0.04
            for tt in target_types:
                delta_tt = delta_df[
                    (delta_df['comparison_idx'] == i) & 
                    (delta_df['target_type'] == tt)
                ]['delta_AUC']
                
                if delta_tt.empty:
                    continue
                
                x_vals_delta = np.random.normal(loc=x_center + 0.1, scale=jitter_scale, size=len(delta_tt))
                ax.scatter(
                    x_vals_delta, delta_tt,
                    # alpha=0.7, 
                    color=target_type_colors[tt],
                    edgecolor='black', linewidth=0.3, s=20,
                    label=tt if i == 0 else None,
                    zorder=3
                )
    
    # Format axes
    ax.set_xticks(range(n_comparisons))
    ax.set_xticklabels([comp['title'] for comp in comparisons], fontsize=fontsize_axes)
    ax.set_ylabel('Δ AUC', fontsize=fontsize_axes)
    ax.tick_params(axis='y', labelsize=fontsize_axes)

    # Remove all spines except left (y-axis) and bottom (x-axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Add legend
    handles_delta, labels_delta = ax.get_legend_handles_labels()
    if 'all' not in labels_delta:
        handles_delta = [all_patch] + handles_delta
        labels_delta = ['all'] + labels_delta
    labels_delta = [label.lower() for label in labels_delta]
    
    ax.legend(
        handles_delta, labels_delta,
        loc='upper left',
        fontsize=fontsize_legend,
        framealpha=0.9
    )
    
    plt.tight_layout()
    # plt.show()

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
        "path_to_predictions": 'path_to_predictions_inference'
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