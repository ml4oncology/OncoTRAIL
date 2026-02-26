import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu
from matplotlib.patches import Patch
import pandas as pd
import ast
import sys
from collections import Counter
import numpy as np
import os
import glob
sys.path.insert(1, "/cluster/projects/gliugroup/2BLAST/data/info/")
from phys_names import aliasDictionary, fellow_alias
from oncotrail.constants import df_physician_char_EPR, df_physician_char_EPIC, target_dict_mapping

sns.set(style="whitegrid")

def target_category(target: str) -> str:
    """
    Categorize target based on naming patterns.
    
    Parameters:
    -----------
    target : str
        Target name
        
    Returns:
    --------
    str
        Target category ('lab', 'symptom', or 'clinic')
    """
    if 'grade2plus' in target:
        return 'lab'
    elif 'change' in target:
        return 'symptom'
    else:
        return 'clinic'

def plot_physician_violin(
    df,
    title,
    save_path,
    top_n=10,
    figsize=(12, 6),
):
    """
    Create violin plots of (prob_prompting - prob_tabular)
    for the top N dictating physicians by volume.
    """

    # Drop missing values defensively
    df = df.dropna(
        subset=["dictating_physician", "prob_prompting", "prob_tabular"]
    ).copy()

    if df.empty:
        print(f"Skipping plot (empty dataframe): {title}")
        return

    # Identify top N physicians by volume
    top_physicians = (
        df["dictating_physician"]
        .value_counts()
        .head(top_n)
        .index
    )

    df = df[df["dictating_physician"].isin(top_physicians)].copy()

    if df.empty:
        print(f"Skipping plot (no data after filtering): {title}")
        return

    # Compute difference
    df["prob_diff"] = df["prob_prompting"] - df["prob_tabular"]

    # Order physicians by median difference (nice for readability)
    order = (
        df.groupby("dictating_physician")["prob_diff"]
        .median()
        .sort_values()
        .index
    )

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=df,
        x="dictating_physician",
        y="prob_diff",
        order=order,
        cut=0,
        inner="box"
    )

    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Dictating Physician")
    plt.ylabel("Prompting - Tabular Probability")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def prepare_physician_merged_df(df_all, df_physician_char):
    """
    Merge physician characteristics and compute YOE.
    """

    df = df_all.merge(
        df_physician_char,
        left_on="dictating_physician",
        right_on="med_onc",
        how="left"
    )

    # Compute years of experience
    df["YOE"] = df["treatment_year"] - df["YOG"]

    # Probability difference
    df["prob_diff"] = df["prob_prompting"] - df["prob_tabular"]

    # Defensive filtering
    df = df.dropna(
        subset=[
            "prob_diff",
            "YOE",
            "Canadian_Medical_Graduate",
            "Speaks_2nd_Language"
        ]
    )

    # create a new column 'target_type'
    df['target_type'] = df['target'].apply(target_category)

    # create a new column 'target_name_plotting'
    df['target_name_plotting'] = df['target'].astype(str).str.replace('_','-').map(target_dict_mapping)

    return df

def compute_physician_target_means(df):
    """
    Compute mean(prompting - tabular) per physician × target.
    """
    df = df.copy()
    df["diff_raw"] = df["prob_prompting"] - df["prob_tabular"]

    df_pt = (
        df.groupby(
            ["dictating_physician", "target", "target_type", "target_name_plotting"],
            observed=True
        )["diff_raw"]
        .mean()
        .reset_index(name="mean_diff")
    )

    return df_pt

def add_normalized_diff(df, diff_col="diff_norm"):
    """
    For each target, compute z-scored (prob_prompting - prob_tabular).
    """
    df = df.copy()
    df["diff_raw"] = df["prob_prompting"] - df["prob_tabular"]

    def _zscore(x):
        sd = x.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return np.zeros(len(x))
        return (x - x.mean()) / sd

    df[diff_col] = (
        df.groupby("target", observed=True)["diff_raw"]
          .transform(_zscore)
    )

    return df

def bootstrap_target_contrasts(
    df,
    x_col,
    diff_col="diff_norm",
    n_boot=1000,
    random_state=0
):
    """
    For each target, bootstrap the difference in means between X=True and X=False.
    Returns one row per (target, bootstrap draw).
    """
    rng = np.random.default_rng(random_state)
    records = []

    for target, dft in df.groupby("target", observed=True):
        y1 = dft.loc[dft[x_col] == True, diff_col].values
        y0 = dft.loc[dft[x_col] == False, diff_col].values

        if len(y1) == 0 or len(y0) == 0:
            continue

        for b in range(n_boot):
            m1 = rng.choice(y1, size=len(y1), replace=True).mean()
            m0 = rng.choice(y0, size=len(y0), replace=True).mean()
            records.append({
                "target": target,
                "boot_diff": m1 - m0
            })

    boot_df = pd.DataFrame(records)

    # Attach metadata for plotting
    meta_cols = ["target", "target_type", "target_name_plotting"]
    meta = df[meta_cols].drop_duplicates()

    return boot_df.merge(meta, on="target", how="left")

from sklearn.linear_model import LinearRegression

def bootstrap_target_slopes(
    df,
    x_col="YOE",
    y_col="diff_norm",
    n_boot=1000,
    random_state=0
):
    """
    For each target, compute slope of y ~ x and bootstrap 95% CI.
    Returns one row per target.
    """
    rng = np.random.default_rng(random_state)
    records = []

    meta_cols = ["target", "target_type", "target_name_plotting"]
    meta = df[meta_cols].drop_duplicates().set_index("target")

    for target, dft in df.groupby("target", observed=True):
        X = dft[[x_col]].values
        y = dft[y_col].values

        if len(np.unique(X)) < 2:
            continue  # cannot fit slope

        # Point estimate
        lr = LinearRegression().fit(X, y)
        slope_hat = lr.coef_[0]

        # Bootstrap
        boot_slopes = []
        n = len(dft)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            lr_b = LinearRegression().fit(X[idx], y[idx])
            boot_slopes.append(lr_b.coef_[0])

        boot_slopes = np.array(boot_slopes)

        records.append({
            "target": target,
            "slope": slope_hat,
            "lo": np.percentile(boot_slopes, 2.5),
            "hi": np.percentile(boot_slopes, 97.5),
            **meta.loc[target].to_dict()
        })

    return pd.DataFrame(records)

def plot_physician_target_dots(
    df_pt,
    target_type_colors,
    save_path,
    top_n=10,
    show_global_line=True,
    jitter_width=0.12
):

    fig, ax = plt.subplots(figsize=(12, 6))

    # -----------------------------
    # restrict to top-N physicians
    # -----------------------------
    top_physicians = (
        df_pt["dictating_physician"]
        .value_counts()
        .head(top_n)
        .index
    )

    df_pt = df_pt[df_pt["dictating_physician"].isin(top_physicians)].copy()

    # -----------------------------
    # order physicians by median
    # -----------------------------
    physician_order = (
        df_pt.groupby("dictating_physician", observed=True)["mean_diff"]
        .median()
        .sort_values()
        .index.tolist()
    )

    x_positions = {p: i for i, p in enumerate(physician_order)}

    # -----------------------------
    # plot dots with horizontal jitter
    # -----------------------------
    rng = np.random.default_rng(42)  # reproducible jitter

    for _, row in df_pt.iterrows():
        x_jittered = (
            x_positions[row["dictating_physician"]]
            + rng.uniform(-jitter_width, jitter_width)
        )

        ax.scatter(
            x_jittered,
            row["mean_diff"],
            color=target_type_colors[row["target_type"]],
            s=35,
            alpha=0.8,
        )

    # -----------------------------
    # physician-specific median lines
    # -----------------------------
    medians = (
        df_pt
        .groupby("dictating_physician", observed=True)["mean_diff"]
        .median()
    )

    for p, m in medians.items():
        x = x_positions[p]
        ax.plot(
            [x - 0.25, x + 0.25],
            [m, m],
            color="black",
            lw=1.5,
            zorder=5,
        )

    # -----------------------------
    # optional global reference line
    # -----------------------------
    if show_global_line:
        global_median = df_pt["mean_diff"].median()
        ax.axhline(
            global_median,
            color="gray",
            linestyle="--",
            lw=0.8,
            alpha=0.6,
        )

    # Zero reference line
    ax.axhline(0, color="gray", linestyle=":", lw=0.8)

    # -----------------------------
    # legend for target types
    # -----------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=c,
            markersize=6,
            label=tt,
        )
        for tt, c in target_type_colors.items()
    ]

    ax.legend(
        handles=legend_handles,
        title="Target type",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    # -----------------------------
    # axis formatting
    # -----------------------------
    ax.set_xticks(range(len(physician_order)))
    ax.set_xticklabels(physician_order, rotation=45, ha="right")
    ax.set_ylabel("Mean Δ (Prompting − Tabular)")
    ax.set_xlabel("Physician")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_target_slopes(
    df_slopes,
    target_type_colors,
    save_path
):
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order targets: by target_type, then slope
    df_plot = df_slopes.copy()
    df_plot["target_type"] = pd.Categorical(
        df_plot["target_type"],
        categories=["clinic", "lab", "symptom"],
        ordered=True
    )

    df_plot = df_plot.sort_values(
        ["target_type", "slope"]
    ).reset_index(drop=True)

    # X positions
    df_plot["x"] = np.arange(len(df_plot))
    x_positions = dict(zip(df_plot["target"], df_plot["x"]))

    # Plot CI + point
    for _, row in df_plot.iterrows():
        color = target_type_colors[row["target_type"]]
        ax.plot(
            [row["x"], row["x"]],
            [row["lo"], row["hi"]],
            color=color,
            lw=1.2
        )
        ax.scatter(
            row["x"],
            row["slope"],
            color=color,
            s=35,
            zorder=3
        )

    # Reference line at 0
    ax.axhline(0, color="gray", linestyle="--", lw=0.7, alpha=0.6)

    # Group separators by target_type
    group_boundaries = (
        df_plot.groupby("target_type", observed=True)["x"]
        .max()
        .sort_values()
        .values
    )
    for x_sep in group_boundaries[:-1]:
        ax.axvline(x=x_sep + 0.5, color="lightgray", linestyle=":", lw=0.8)

    # Axis formatting
    ax.set_xticks(df_plot["x"])
    ax.set_xticklabels(
        df_plot["target_name_plotting"],
        rotation=45,
        ha="right"
    )
    ax.set_ylabel("Slope of Δ (Prompting − Tabular)\nper year of experience")
    ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def plot_target_contrast_half_violin(
    boot_df,
    target_type_colors,
    save_path
):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Order targets by target_type, then median effect
    summary = (
        boot_df.groupby(
            ["target", "target_type", "target_name_plotting"],
            observed=True
        )["boot_diff"]
        .agg(
            median="median",
            lo=lambda x: np.percentile(x, 2.5),
            hi=lambda x: np.percentile(x, 97.5),
        )
        .reset_index()
    )

    summary["target_type"] = pd.Categorical(
        summary["target_type"],
        categories=["clinic", "lab", "symptom"],
        ordered=True
    )

    summary = summary.sort_values(
        ["target_type", "median"]
    ).reset_index(drop=True)

    x_positions = {t: i for i, t in enumerate(summary["target"])}
    summary["x"] = summary["target"].map(x_positions)

    # --- Half violins ---
    for _, row in summary.iterrows():
        vals = boot_df.loc[
            boot_df["target"] == row["target"], "boot_diff"
        ]
        sns.violinplot(
            y=vals,
            x=[row["x"]] * len(vals),
            orient="v",
            cut=0,
            inner=None,
            bw_adjust=0.8,
            color=target_type_colors[row["target_type"]],
            alpha=0.25,
            linewidth=0,
            ax=ax
        )

    # --- Median + interval ---
    for _, row in summary.iterrows():
        ax.plot(
            [row["x"], row["x"]],
            [row["lo"], row["hi"]],
            color=target_type_colors[row["target_type"]],
            lw=1.2
        )
        ax.scatter(
            row["x"],
            row["median"],
            color=target_type_colors[row["target_type"]],
            s=30,
            zorder=3
        )

    # Reference line at 0
    ax.axhline(0, color="gray", linestyle="--", lw=0.7, alpha=0.6)

    # Group separators by target_type
    group_boundaries = (
        summary.groupby("target_type", observed=True)["x"]
        .max()
        .sort_values()
        .values
    )
    for x_sep in group_boundaries[:-1]:
        ax.axvline(x=x_sep + 0.5, color="lightgray", linestyle=":", lw=0.8)

    # Axis formatting
    ax.set_xticks(summary["x"])
    ax.set_xticklabels(summary["target_name_plotting"], rotation=45, ha="right")
    ax.set_ylabel("Normalized Δ (Prompting − Tabular)")
    ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_physician_characteristics(
    df,
    title,
    save_path,
    figsize=(16, 5),
):
    """
    3-panel plot:
      1) prob_diff vs YOE
      2) prob_diff by Canadian_Medical_Graduate
      3) prob_diff by Speaks_2nd_Language
    """

    if df.empty:
        print(f"Skipping plot (empty dataframe): {title}")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    # -------------------------
    # Panel 1: YOE (continuous)
    # -------------------------
    sns.scatterplot(
        data=df,
        x="YOE",
        y="prob_diff",
        alpha=0.4,
        s=30,
        ax=axes[0]
    )

    sns.regplot(
        data=df,
        x="YOE",
        y="prob_diff",
        scatter=False,
        lowess=True,
        color="black",
        ax=axes[0]
    )

    axes[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_title(title)
    axes[0].set_xlabel("years of experience")
    axes[0].set_ylabel("Prompting - Tabular Probability")

    # --------------------------------------
    # Panel 2: Canadian Medical Graduate
    # --------------------------------------
    sns.violinplot(
        data=df,
        x="Canadian_Medical_Graduate",
        y="prob_diff",
        cut=0,
        inner="box",
        ax=axes[1]
    )

    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Canadian Medical Graduate")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    # --------------------------------------
    # Panel 3: Speaks Second Language
    # --------------------------------------
    sns.violinplot(
        data=df,
        x="Speaks_2nd_Language",
        y="prob_diff",
        cut=0,
        inner="box",
        ax=axes[2]
    )

    axes[2].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[2].set_title("Speaks Second Language")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the full 3-panel plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------------------------
    # Create standalone Panel 0 figure
    # ---------------------------------
    fig0, ax0 = plt.subplots(figsize=(figsize[0] / 3, figsize[1]))

    sns.scatterplot(
        data=df,
        x="YOE",
        y="prob_diff",
        alpha=0.4,
        s=30,
        ax=ax0
    )

    sns.regplot(
        data=df,
        x="YOE",
        y="prob_diff",
        scatter=False,
        lowess=True,
        color="black",
        ax=ax0
    )

    ax0.axhline(0, linestyle="--", color="black", linewidth=1)
    ax0.set_title(title)
    ax0.set_xlabel("years of experience")
    ax0.set_ylabel("Prompting - Tabular Probability")

    fig0.tight_layout()

    save_path_panel0 = save_path.replace('.png', '_YOE_scatter.png')
    fig0.savefig(save_path_panel0, dpi=300)
    plt.close(fig0)

def plot_binary_contrast_half_violin(
    df,
    binary_var,
    target_type_colors,
    save_path,
    return_stats=True
):
    """
    Plot paired half-violins comparing prob_diff distributions by binary variable.
    
    Parameters:
    -----------
    df : DataFrame
        Must contain columns: target, target_type, target_name_plotting, prob_diff, and binary_var
    binary_var : str
        Name of the binary (True/False) variable to split on
    target_type_colors : dict
        Mapping from target_type to color
    save_path : str
        Path to save the figure
    return_stats : bool
        Whether to return the statistics table
        
    Returns:
    --------
    stats_df : DataFrame (if return_stats=True)
        Contains target, target_type, target_name_plotting, p_value, median_true, median_false
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bonferroni correction threshold
    alpha_corrected = 0.05 / (19 * 4)
    
    # Order targets by target_type, then median effect (using True group)
    summary = (
        df[df[binary_var] == True]
        .groupby(
            ["target", "target_type", "target_name_plotting"],
            observed=True
        )["prob_diff"]
        .median()
        .reset_index(name="median")
    )
    
    summary["target_type"] = pd.Categorical(
        summary["target_type"],
        categories=["clinic", "lab", "symptom"],
        ordered=True
    )
    
    summary = summary.sort_values(
        ["target_type", "median"]
    ).reset_index(drop=True)
    
    x_positions = {t: i for i, t in enumerate(summary["target"])}
    summary["x"] = summary["target"].map(x_positions)
    
    # Store statistics
    stats_list = []
    
    # --- Process each target ---
    for _, row in summary.iterrows():
        target = row["target"]
        x_pos = row["x"]
        color = target_type_colors[row["target_type"]]
        
        # Convert color to RGB for shading
        rgb = mcolors.to_rgb(color)
        lighter_color = tuple(0.5 * c + 0.5 for c in rgb)  # Mix with white for lighter shade
        darker_color = rgb  # Keep original as darker shade
        
        # Get data for both groups
        vals_true = df.loc[
            (df["target"] == target) & (df[binary_var] == True), 
            "prob_diff"
        ]
        vals_false = df.loc[
            (df["target"] == target) & (df[binary_var] == False), 
            "prob_diff"
        ]
        
        # Mann-Whitney U test
        if len(vals_true) > 0 and len(vals_false) > 0:
            stat, p_value = mannwhitneyu(vals_true, vals_false, alternative='two-sided')
        else:
            p_value = np.nan
        
        median_true = vals_true.median() if len(vals_true) > 0 else np.nan
        median_false = vals_false.median() if len(vals_false) > 0 else np.nan
        
        stats_list.append({
            "target": target,
            "target_type": row["target_type"],
            "target_name_plotting": row["target_name_plotting"],
            "p_value": p_value,
            "median_true": median_true,
            "median_false": median_false,
            "n_true": len(vals_true),
            "n_false": len(vals_false)
        })
        
        # --- Left half-violin (True) - LIGHTER shade ---
        if len(vals_true) > 0:
            parts = ax.violinplot(
                [vals_true],
                positions=[x_pos],
                widths=0.8,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts['bodies']:
                pc.set_facecolor(lighter_color)
                pc.set_alpha(0.4)
                pc.set_edgecolor('none')
                
                # Get violin path and clip to left half
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                vertices = pc.get_paths()[0].vertices
                vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, m)
            
            # Median line for True
            ax.hlines(
                median_true,
                x_pos - 0.2,
                x_pos,
                color=darker_color,
                lw=2,
                zorder=4
            )
        
        # --- Right half-violin (False) - DARKER shade ---
        if len(vals_false) > 0:
            parts = ax.violinplot(
                [vals_false],
                positions=[x_pos],
                widths=0.8,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts['bodies']:
                pc.set_facecolor(darker_color)
                pc.set_alpha(0.4)
                pc.set_edgecolor('none')
                
                # Get violin path and clip to right half
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                vertices = pc.get_paths()[0].vertices
                vertices[:, 0] = np.clip(vertices[:, 0], m, np.inf)
            
            # Median line for False
            ax.hlines(
                median_false,
                x_pos,
                x_pos + 0.2,
                color=darker_color,
                lw=2,
                zorder=4
            )
        
        # --- Add asterisk if significant ---
        if p_value < alpha_corrected:
            y_max = max(
                vals_true.max() if len(vals_true) > 0 else -np.inf,
                vals_false.max() if len(vals_false) > 0 else -np.inf
            )
            ax.text(
                x_pos,
                y_max + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                '*',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold',
                color='black'
            )
    
    # Reference line at 0
    ax.axhline(0, color="gray", linestyle="--", lw=0.7, alpha=0.6)
    
    # Group separators by target_type
    group_boundaries = (
        summary.groupby("target_type", observed=True)["x"]
        .max()
        .sort_values()
        .values
    )
    for x_sep in group_boundaries[:-1]:
        ax.axvline(x=x_sep + 0.5, color="lightgray", linestyle=":", lw=0.8)
    
    # Axis formatting
    ax.set_xticks(summary["x"])
    ax.set_xticklabels(summary["target_name_plotting"], rotation=45, ha="right")
    ax.set_ylabel("prompting - tabular probability difference")
    ax.set_xlabel("")
    
    # Add legend
    # Use gray with lighter/darker shading pattern
    base_gray = 'gray'
    rgb = mcolors.to_rgb(base_gray)
    
    # Create lighter shade (mix with white) for True
    lighter = tuple(0.5 * c + 0.5 for c in rgb)
    # Create darker shade for False (keep original)
    darker = rgb
    
    legend_elements = [
        Patch(facecolor=lighter, alpha=0.4, label=f'{binary_var.replace("_", " ")}=True, left'),
        Patch(facecolor=darker, alpha=0.4, label=f'{binary_var.replace("_", " ")}=False, right')
    ]
    
    # loc options: 'upper left', 'upper right', 'lower left', 'lower right', 
    #              'upper center', 'lower center', 'center left', 'center right', 'center', 'best'
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Return statistics table
    if return_stats:
        stats_df = pd.DataFrame(stats_list)
        return stats_df
    else:
        return None

def plot_YOE_histogram(df, save_path):

    # Consider only unique combinations of mrn and dictating_physician
    df_unique = df.drop_duplicates(subset=["mrn", "dictating_physician"])

    # Create the figure and axis
    plt.figure(figsize=(10, 6))

    # Plot normalized histogram with colorblind-friendly color
    plt.hist(df_unique['YOE'], bins=30, density=True, alpha=0.7, color='#0173B2', edgecolor='black')

    # Add labels and title
    plt.xlabel('years of experience', fontsize=12)
    plt.ylabel('probability density', fontsize=12)
    # plt.title('Normalized Histogram of Years of Experience', fontsize=14, pad=20)

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

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

def load_prompting_result(target_name, data_dir):
    target_dir = os.path.join(data_dir, target_name)
    note_dirs = [d for d in glob.glob(os.path.join(target_dir, "note_*")) if os.path.isdir(d)]
    assert len(note_dirs) == 1, f"Expected one note_ directory in {target_dir}, found: {note_dirs}"
    summary_files = glob.glob(os.path.join(note_dirs[0], "summary_*.csv"))
    assert len(summary_files) == 1, f"Expected one summary_*.csv in {note_dirs[0]}, found: {summary_files}"
    prompting_summary_df = pd.read_csv(summary_files[0])
    
    return prompting_summary_df

def load_tabular_result(target_name, tabular_results):
    df_model = pd.read_csv(tabular_results)
    target_row = df_model[df_model["target"] == target_name.replace("_", "-")]
    assert not target_row.empty, f"Target {target_name.replace('_', '-')} not found in dataframe"
    data = np.load(target_row["pred_file_name"].values[0], allow_pickle=True)
    mrns = data['mrn_test']
    probs = data['test_pred'].ravel()

    return pd.DataFrame({'mrn': mrns, 'prob': probs})

def get_dataframe(target_name, df_treatment, prompting_data_dir, tabular_results): 

    prompting_summary_df = load_prompting_result(target_name, prompting_data_dir)
    tabular_summary_df = load_tabular_result(target_name, tabular_results)

    # rename 'Probability' column in prompting_summary_df to 'prob_prompting'
    prompting_summary_df = prompting_summary_df.rename(columns={'Probability': 'prob_prompting'})
    # rename 'prob' column in tabular_summary_df to 'prob_tabular'
    tabular_summary_df = tabular_summary_df.rename(columns={'prob': 'prob_tabular'})
    # merge prompting_summary_df['mrn', 'prob_prompting'] with df_treatment on 'mrn'
    df_merged = pd.merge(prompting_summary_df[['mrn', 'prob_prompting']], df_treatment, on='mrn', how='inner')
    # merge tabular_summary_df with df_merged on 'mrn'
    df_merged = pd.merge(df_merged, tabular_summary_df, on='mrn', how='inner')

    df_merged['dictating_physician'] = df_merged['stats_dictated_by'].apply(extract_names)
    df_merged['num_dictators'] = df_merged['dictating_physician'].apply(len)
    df_merged = df_merged.loc[df_merged['num_dictators']==1].copy()
    df_merged['dictating_physician'] = df_merged['dictating_physician'].apply(lambda x: x[0] if x else 'Unknown')
    # remove 'Unknown' dictating_physician
    df_merged = df_merged[df_merged['dictating_physician'] != 'Unknown']
    eps = 1e-6
    df_merged["logit_prompting"] = np.log((df_merged["prob_prompting"] + eps) / (1 - df_merged["prob_prompting"] + eps))
    df_merged["logit_tabular"]  = np.log((df_merged["prob_tabular"] + eps)  / (1 - df_merged["prob_tabular"]  + eps))
    df_merged['average_treatment_year'] = df_merged.groupby('dictating_physician')['treatment_year'].transform('mean')

    df_merged["logit_tabular_c"] = df_merged["logit_tabular"] - df_merged["logit_tabular"].mean()

    return df_merged

def prepare_all_dataframes(target_list, emr_source, anchored_notes_path, prompting_data_dir, tabular_results):

    df_treatment = pd.read_csv(anchored_notes_path)
    df_treatment['treatment_date'] = pd.to_datetime(df_treatment['treatment_date'])
    df_treatment['treatment_year'] = df_treatment['treatment_date'].dt.year
    df_treatment = df_treatment[['mrn', 'stats_dictated_by', 'treatment_year']].copy()  

    if emr_source == 'EPR':
        top_physician_names = df_physician_char_EPR['med_onc'].tolist()
    else:
        top_physician_names = df_physician_char_EPIC['med_onc'].tolist()

    concat_list_df = []
    for target in target_list:
        print(f"Preparing dataframe for target: {target}")
        df_target = get_dataframe(target, df_treatment, prompting_data_dir, tabular_results)
        df_target['target'] = target
        df_target = df_target[df_target['dictating_physician'].isin(top_physician_names)].copy()
        concat_list_df.append(df_target)
    
    df_all = pd.concat(concat_list_df, ignore_index=True)
    return df_all

def plot_physician_characteristics_main(
    target_list,
    emr_source,
    anchored_notes_path,
    prompting_data_dir,
    tabular_results,
    output_dir
):
    
    target_type_colors = {
        'clinic': '#d95f02',
        'lab': '#1b9e77',
        'symptom': '#7570b3',
    }
        
    df_all = prepare_all_dataframes(
        target_list,
        emr_source,
        anchored_notes_path,
        prompting_data_dir,
        tabular_results
    )

    if emr_source == 'EPR':
        df_physician_char = df_physician_char_EPR
    else:
        df_physician_char = df_physician_char_EPIC

    df_merged = prepare_physician_merged_df(df_all, df_physician_char)

    # Plot violin for top physicians
    violin_save_path = os.path.join(output_dir, f"physician_violin_{emr_source}.png")
    plot_physician_violin(
        df_all,
        title=f"Top Physicians Violin Plot ({emr_source})",
        save_path=violin_save_path,
        top_n=10,
        figsize=(12, 6)
    )

    df_physician_target_means = compute_physician_target_means(df_merged)
    plot_physician_target_dots(
        df_physician_target_means,
        target_type_colors,
        save_path = os.path.join(output_dir, f"physician_target_dots_{emr_source}.png")
    )

    # Plot physician characteristics
    char_save_path = os.path.join(output_dir, f"physician_characteristics_{emr_source}.png")
    plot_physician_characteristics(
        df_merged,
        title=f"Physician Characteristics ({emr_source})",
        save_path=char_save_path,
        figsize=(16, 5)
    )

    # Plot physician characteristics for each target
    for target in target_list:
        target_save_path = os.path.join(output_dir, f"physician_characteristics_{target}_{emr_source}.png")
        # get the value of target_name_plotting for this target
        target_name_plotting = df_merged.loc[df_merged['target'] == target, 'target_name_plotting'].iloc[0]
        plot_physician_characteristics(
            df_merged[df_merged['target'] == target],
            # title=f"Physician Characteristics ({target}, {emr_source})",
            title=f"{target_name_plotting} ({emr_source})",
            save_path=target_save_path,
            figsize=(16, 5)
        )
    
    # Plot histogram of YOE
    plot_YOE_histogram(df_merged, save_path = os.path.join(output_dir, f"YOE_histogram_{emr_source}.png"))

    df_norm = add_normalized_diff(df_merged)

    # Compute slopes
    df_slopes = bootstrap_target_slopes(
        df_norm,
        x_col="YOE",
        y_col="diff_raw",
        n_boot=1000
    )

    plot_target_slopes(
        df_slopes,
        target_type_colors,
        save_path = os.path.join(output_dir, f"target_slopes_YOE_{emr_source}.png")
    )

    col_name = 'Canadian_Medical_Graduate'
    boot_df_CMG = bootstrap_target_contrasts(
        df_norm,
        x_col=col_name,
        n_boot=1000
    )
    col_name = 'Speaks_2nd_Language'
    boot_df_S2L = bootstrap_target_contrasts(
        df_norm,
        x_col=col_name,
        n_boot=1000
    )

    plot_target_contrast_half_violin(
        boot_df_CMG,
        target_type_colors,
        save_path = os.path.join(output_dir, f"target_contrast_half_violin_CMG_{emr_source}.png")
    )

    plot_target_contrast_half_violin(
        boot_df_S2L,
        target_type_colors,
        save_path = os.path.join(output_dir, f"target_contrast_half_violin_S2L_{emr_source}.png")
    )

    # Plot binary contrasts
    binary_var = 'Canadian_Medical_Graduate'
    save_path = os.path.join(output_dir, f"binary_contrast_half_violin_CMG_{emr_source}.png")
    stats = plot_binary_contrast_half_violin(
        df_merged,
        binary_var,
        target_type_colors,
        save_path,
        return_stats=True
    )
    stats.to_csv(os.path.join(output_dir, f"binary_contrast_stats_CMG_{emr_source}.csv"), index=False)

    binary_var = 'Speaks_2nd_Language'
    save_path = os.path.join(output_dir, f"binary_contrast_half_violin_S2L_{emr_source}.png")
    stats = plot_binary_contrast_half_violin(
        df_merged,
        binary_var,
        target_type_colors,
        save_path,
        return_stats=True
    )
    stats.to_csv(os.path.join(output_dir, f"binary_contrast_stats_S2L_{emr_source}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_list", type=str, help="List of target names. E.g., \"['target1', 'target2']\"")
    parser.add_argument("emr_source", type=str, choices=['EPR', 'EPIC'], help="EMR source: EPR or EPIC")
    parser.add_argument("anchored_notes_path", type=str, help="Path to anchored notes CSV file")
    parser.add_argument("prompting_data_dir", type=str, help="Directory containing prompting results")
    parser.add_argument("tabular_results", type=str, help="Path to tabular results CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save output plots")
    args = parser.parse_args()

    target_list = ast.literal_eval(args.target_list)
    plot_physician_characteristics_main(
        target_list,
        args.emr_source,
        args.anchored_notes_path,
        args.prompting_data_dir,
        args.tabular_results,
        args.output_dir
    )