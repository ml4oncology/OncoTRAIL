import pandas as pd
import polars as pl
import argparse
from transformers import AutoTokenizer
import seaborn as sns
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os
import importlib.util
from oncotrail.utils.env_loader import load_env
load_env()
spec = importlib.util.spec_from_file_location("phys_names", os.path.join(os.environ.get("PHYS_NAMES_DIR", ""), "phys_names.py"))
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)
aliasDictionary = constants.aliasDictionary

def plot_note_length_over_time(df_notes_visit, first_year_EPIC, fig_size_mm=None):
    """
    Plot distribution of tokenized clinical note length over time,
    highlighting the EMR system change (EPIC_FLAG == 1).
    """

    fontsize_axes = 4
    fontsize_legend = 4

    if fig_size_mm is not None:
        figsize = (fig_size_mm[0] / 25.4, fig_size_mm[1] / 25.4)
    else:
        figsize = (12, 6)

    # ----------------------------
    # Compute yearly summary stats
    # ----------------------------
    summary = (
        df_notes_visit
        .groupby("year")["tokenized_length"]
        .agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .reset_index()
    )

    # ----------------------------
    # Compute IQR + whiskers
    # ----------------------------
    summary["IQR"] = summary["q3"] - summary["q1"]
    summary["lower_whisker_calc"] = summary["q1"] - 1.5 * summary["IQR"]
    summary["upper_whisker"] = summary["q3"] + 1.5 * summary["IQR"]

    # Lower whisker cannot be below observed minimum
    summary["lower_whisker"] = summary[["lower_whisker_calc", "min"]].max(axis=1)

    # Ensure clean numeric data
    summary = summary.apply(pd.to_numeric, errors="coerce").dropna()

    # ----------------------------
    # Colorblind-friendly palette
    # ----------------------------
    COLOR_IQR = "#E69F00"       # Orange
    COLOR_WHISKER = "#56B4E9"   # Sky Blue
    COLOR_MEDIAN = "#000000"    # Black
    COLOR_EMR = "#999999"       # Neutral gray

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Highlight EMR change period
    if pd.notna(first_year_EPIC):
        ax.axvspan(
            first_year_EPIC,
            summary["year"].max(),
            color=COLOR_EMR,
            alpha=0.2,
            label="EMR change",
        )

    # Whisker region
    ax.fill_between(
        summary["year"],
        summary["lower_whisker"],
        summary["upper_whisker"],
        alpha=0.2,
        color=COLOR_WHISKER,
        label="min to Q3 + 1.5×IQR",
    )

    # IQR region
    ax.fill_between(
        summary["year"],
        summary["q1"],
        summary["q3"],
        alpha=0.4,
        color=COLOR_IQR,
        label="IQR (Q1–Q3)",
    )

    # Median line
    ax.plot(
        summary["year"],
        summary["median"],
        marker="o",
        markersize=5,
        color=COLOR_MEDIAN,
        label="median",
    )

    # Labels & formatting
    ax.set_xlabel("year", fontsize=fontsize_axes)
    ax.set_ylabel("tokenized length of oncology clinical note", fontsize=fontsize_axes)
    ax.set_xticks(summary["year"])
    ax.set_xticklabels(summary["year"], rotation=45, fontsize=fontsize_axes)
    ax.tick_params(axis='both', labelsize=fontsize_axes, pad=1)
    ax.grid(False)

    # Remove all spines except left (y-axis) and bottom (x-axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('black')

    ax.legend(loc="upper left", fontsize=fontsize_legend)

    plt.tight_layout(pad=0.5)
    fig.set_size_inches(figsize[0], figsize[1])
    plt.show()

    return fig, ax

def main(notes_path, save_dir, model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    df = pl.scan_parquet(notes_path)  # lazy

    proc_name_epic0 = [
        "Clinic Note", "Letter", "History & Physical Note",
        "Consultation Note", "Clinic Note (Non-dictated)"
    ]
    proc_name_epic1 = [
        "PROGRESS", "CONSULT", "H&P", "TELEPHONE EN"
    ]
    unique_medoncs = list(set(aliasDictionary.values()))

    filtered = (
        df.filter(
            pl.col("processed_physician_name").is_in(unique_medoncs) &
            (
                ((pl.col("EPIC_FLAG") == 0) & pl.col("Observations.ProcName").is_in(proc_name_epic0))
                |
                ((pl.col("EPIC_FLAG") == 1) & pl.col("Observations.ProcName").is_in(proc_name_epic1))
            )
        )
    )

    # only loads into memory now:
    df_notes_visit = filtered.collect().to_pandas()
    df_notes_visit['processed_date'] = pd.to_datetime(df_notes_visit['processed_date'], errors='coerce')
    df_notes_visit['year'] = df_notes_visit['processed_date'].dt.year
    df_notes_visit = df_notes_visit.loc[df_notes_visit['year'] >= 2008].copy()
    df_notes_visit['tokenized_length'] = [len(tokenizer.tokenize(note)) for note in df_notes_visit['clinical_notes']]
    df_notes_visit = df_notes_visit.loc[df_notes_visit['tokenized_length'] >= 10]

    # find the first year where EPIC_FLAG==1
    first_year_EPIC = df_notes_visit[df_notes_visit['EPIC_FLAG'] == 1]['year'].min()

    fig, _ = plot_note_length_over_time(df_notes_visit, first_year_EPIC, fig_size_mm=(150, 45))

    fig.savefig(save_dir + '/clinical_note_tokenized_length_over_time.svg')

    # statistics

    # Split the groups
    group0 = df_notes_visit[df_notes_visit['EPIC_FLAG'] == 0]['tokenized_length'].dropna()
    group1 = df_notes_visit[df_notes_visit['EPIC_FLAG'] == 1]['tokenized_length'].dropna()

    # Compute medians
    median0 = group0.median()
    median1 = group1.median()

    # Mann–Whitney U test (two-sided)
    stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')

    # Print results
    print(f"Median (EPIC_FLAG=0): {median0:.1f}")
    print(f"Median (EPIC_FLAG=1): {median1:.1f}")
    print(f"P-value: {p_value}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot figure for tokenized note length")
    parser.add_argument("notes_path", type=str, help="Path to where clinical notes are stored")
    parser.add_argument("save_dir", type=str, help="Directory where to save plot")
    parser.add_argument("model_path", type=str, help="Path to LLM model for tokenization")
    args = parser.parse_args()
    main(args.notes_path, args.save_dir, args.model_path)

