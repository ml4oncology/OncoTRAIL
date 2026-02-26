import pandas as pd
import polars as pl
import argparse
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import importlib.util
spec = importlib.util.spec_from_file_location("phys_names", "/cluster/projects/gliugroup/2BLAST/data/info/phys_names.py")
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)
aliasDictionary = constants.aliasDictionary

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

    # Compute summary statistics
    summary = df_notes_visit.groupby("year")["tokenized_length"].agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        min="min",
        max="max"
    ).reset_index()

    # Custom colorblind-friendly palette
    COLOR_IQR = "#E69F00"       # Orange
    COLOR_WHISKER = "#56B4E9"   # Sky Blue
    COLOR_MEDIAN = "#000000"    # Black

    # Compute IQR and whisker limits
    summary["IQR"] = summary["q3"] - summary["q1"]
    summary["lower_whisker_calc"] = summary["q1"] - 1.5 * summary["IQR"]
    summary["upper_whisker"] = summary["q3"] + 1.5 * summary["IQR"]

    # Replace negative lower whisker with actual minimum
    summary["lower_whisker"] = summary[["lower_whisker_calc", "min"]].max(axis=1)

    # Ensure all values are numeric and clean
    summary = summary.apply(pd.to_numeric, errors="coerce").dropna()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Highlight the EMR change period
    ax.axvspan(first_year_EPIC, summary["year"].max(), 
            color="#999999", alpha=0.2, label="EMR change")  # gray, subtle

    # Shaded region: adjusted whiskers (min to upper limit)
    ax.fill_between(summary["year"], summary["lower_whisker"], summary["upper_whisker"],
                    alpha=0.2, label="min to Q3+1.5×IQR", color=COLOR_WHISKER)

    # Shaded region: interquartile range (Q1 to Q3)
    ax.fill_between(summary["year"], summary["q1"], summary["q3"],
                    alpha=0.4, label="IQR (Q1 to Q3)", color=COLOR_IQR)

    # Line for median
    ax.plot(summary["year"], summary["median"], marker="o", label="median", color=COLOR_MEDIAN)

    # Axis labels and title
    ax.set_title("Distribution of oncology clinical note length over time", fontsize=18)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Tokenized length", fontsize=18)
    ax.set_xticks(summary["year"])
    ax.set_xticklabels(summary["year"], rotation=45)
    ax.grid(True)

    # Legend
    ax.legend(loc="upper left", fontsize=14)
    plt.tight_layout()
    plt.show()

    fig.savefig(save_dir + '/tokenized_note_length.png', format='png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot figure for tokenized note length")
    parser.add_argument("notes_path", type=str, help="Path to where clinical notes are stored")
    parser.add_argument("save_dir", type=str, help="Directory where to save plot")
    parser.add_argument("model_path", type=str, help="Path to LLM model for tokenization")
    args = parser.parse_args()
    main(args.notes_path, args.save_dir, args.model_path)

