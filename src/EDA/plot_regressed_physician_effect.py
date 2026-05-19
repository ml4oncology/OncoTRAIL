import argparse
import pandas as pd
import matplotlib.pyplot as plt
from oncotrail.constants import target_dict_mapping

def plot_icc_forest(df, auc_col, title_suffix, fig_size_mm=(180,50)):
    # Sort by AUC descending so highest AUC is at top
    df = df.copy().sort_values(auc_col, ascending=False)

    fontsize_axes = 4
    fontsize_legend = 4

    if fig_size_mm is not None:
        figsize = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)
    else:
        figsize = (6, 8)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x_positions = range(len(df))

    # Normalize AUC for colormap
    norm = plt.Normalize(vmin=df[auc_col].min(), vmax=df[auc_col].max())
    cmap = plt.cm.RdYlGn  # Red (low AUC) → Yellow → Green (high AUC)

    for i, (_, row) in enumerate(df.iterrows()):
        color = cmap(norm(row[auc_col]))

        # Vertical error bar for ICC CI
        ax.errorbar(
            i, row["ICC"],
            yerr=[[row["ICC"] - row["ICC_lower"]], [row["ICC_upper"] - row["ICC"]]],
            fmt="none",
            ecolor="gray",
            elinewidth=0.8,
            capsize=3,
            zorder=1
        )

        # Marker colored by AUC
        ax.scatter(
            i, row["ICC"],
            marker="D",
            color=color,
            s=15,
            edgecolor="black",
            linewidth=0.6,
            zorder=2
        )

    # X-axis labels (target names), tilted to avoid overlap
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(df["target_short"].values, rotation=45, ha="right",
                       fontsize=fontsize_axes)

    # Reference line at ICC = 0
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)

    # Labels & aesthetics
    ax.set_ylabel("Intraclass Correlation Coefficient (ICC)", fontsize=fontsize_axes)
    ax.tick_params(labelsize=fontsize_axes, pad=1)

    # Remove all spines except left (y-axis) and bottom (x-axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('black')

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label(f"AUC ({title_suffix})", fontsize=fontsize_axes)
    cbar.ax.tick_params(labelsize=fontsize_axes)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout(pad=0.5)
    fig.set_size_inches(figsize[0], figsize[1])

    return fig


def plot_stage2_forest(df: pd.DataFrame, target_map: dict,
                       fig_size_mm=(180,90)) -> plt.Figure:
    """
    Plot a 4-panel forest plot of Stage 2 model coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Target, Feature, Median, 2.5%, 97.5%
    target_map : dict
        Mapping from raw target names (with dashes) to short display labels.
    fig_size_mm : tuple, optional
        Figure size as (width_mm, height_mm). If None, defaults to (14 inches, 8 inches).

    Returns
    -------
    fig : plt.Figure
    """

    fontsize_axes = 4
    fontsize_legend = 4

    if fig_size_mm is not None:
        figsize = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)
    else:
        figsize = (14, 8)

    # --- Preprocess ---
    df = df.copy()
    df["target_short"] = df["Target"].str.replace("_", "-").map(target_map)

    features = ["Intercept", "Canadian_Medical_Graduate", "Speaks_2nd_Language", "average_YOE"]

    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False, dpi=300)

    yticks = None
    yticklabels = None

    for i, feature in enumerate(features):
        ax = axes[i]

        sub = df[df["Feature"] == feature].copy()

        ordered_labels = [v for v in target_map.values() if v in sub["target_short"].values]
        sub["_order"] = sub["target_short"].map({label: idx for idx, label in enumerate(ordered_labels)})
        sub = sub.sort_values("_order")

        y_positions = list(range(len(sub)))

        lower = sub["2.5%"].values
        upper = sub["97.5%"].values
        median = sub["Median"].values

        ax.errorbar(
            median, y_positions,
            xerr=[median - lower, upper - median],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            markersize=5,
        )

        ax.set_title(feature.replace("_", " "), fontsize=fontsize_axes, fontweight="normal")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

        # Remove all spines except left (y-axis) and bottom (x-axis)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['bottom'].set_color('black')

        ax.tick_params(labelsize=fontsize_axes, pad=1)
        ax.tick_params(axis='x', labelrotation=45)

        # Store ticks from first panel
        if i == 0:
            yticks = y_positions
            yticklabels = sub["target_short"].values

        ax.invert_yaxis()

    # Apply to shared axis AFTER loop
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticklabels, fontsize=fontsize_axes)

    for ax in axes[1:]:
        ax.set_yticks(yticks)
        ax.set_yticklabels([])

    plt.tight_layout(pad=0.5)
    fig.set_size_inches(figsize[0], figsize[1])

    # Apply ha="right" to all x-tick labels after layout is finalized
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_ha("right")

    return fig

def plot_regressed_physician_effect(held_out_set, 
                                    prompting_results_path, ICC_results_path, 
                                    regression_coefficients_path,
                                    save_dir):

    # load prompting results
    prompting_df = pd.read_csv(prompting_results_path)
    prompting_df['target'] = prompting_df['target'].str.replace('-', '_')

    # load ICC results
    ICC_df = pd.read_csv(ICC_results_path)
    if held_out_set == "test":
        auc_col = 'auc_test'
    else:
        auc_col = 'auc_inference'
    # merge prompting_df[['target', auc_col]] into ICC_df on target
    ICC_df = ICC_df.merge(prompting_df[['target', auc_col]], on='target', how='left')
    ICC_df['target_short'] = ICC_df['target'].str.replace('_', '-').map(target_dict_mapping)
    fig = plot_icc_forest(ICC_df, auc_col=auc_col, title_suffix="Prompting")
    fig.savefig(f"{save_dir}/physician_effect_ICC_forest_{held_out_set}.png", dpi=300)
    fig.savefig(f"{save_dir}/physician_effect_ICC_forest_{held_out_set}.svg")
    
    # load regression coefficients
    reg_df = pd.read_csv(regression_coefficients_path)
    fig = plot_stage2_forest(reg_df, target_map=target_dict_mapping)
    fig.savefig(f"{save_dir}/physician_effect_regression_forest_{held_out_set}.png", dpi=300)
    fig.savefig(f"{save_dir}/physician_effect_regression_forest_{held_out_set}.svg")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce plots for hierarchical models regressed on physician characteristics."
    )
    parser.add_argument("held_out_set", choices=["test", "inference"],help="Which held-out set to analyze (e.g. 'test', 'inference').")
    parser.add_argument("prompting_results_path", help="CSV of prompting results.")
    parser.add_argument("ICC_results_path", help="CSV of ICC results from physician_effect.py.")
    parser.add_argument("regression_coefficients_path", help="CSV of regression coefficients from physician_effect.py.")
    parser.add_argument("save_dir", help="Directory to save plots.")
    args = parser.parse_args()

    plot_regressed_physician_effect(
        args.held_out_set,
        args.prompting_results_path,
        args.ICC_results_path,
        args.regression_coefficients_path,
        args.save_dir
    )