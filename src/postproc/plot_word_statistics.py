import pandas as pd
import argparse
import ast
import os
from llm_notes_classification.postproc.word_analyzer_utils import (
    generate_popularity_plot,
    plot_input_output_alignment,
)
from llm_notes_classification.constants import target_dict_mapping


def plot_word_statistics(data_dir, save_dir, target_list):
    os.makedirs(save_dir, exist_ok=True)

    for target in target_list:
        target_clean = target.replace("_", "-")
        print(f"\nProcessing target: {target_clean}")

        # -----------------------
        # Load CSVs safely
        # -----------------------
        try:
            df_reason = pd.read_csv(
                f"{data_dir}/word_stats_all_Reason_{target_clean}.csv"
            )
        except Exception as e:
            print(f"  ⚠️ Could not load Reason CSV for {target_clean}: {e}")
            df_reason = pd.DataFrame()

        try:
            df_note = pd.read_csv(
                f"{data_dir}/word_stats_all_note_{target_clean}.csv"
            )
        except Exception as e:
            print(f"  ⚠️ Could not load Note CSV for {target_clean}: {e}")
            df_note = pd.DataFrame()

        # -----------------------
        # Popularity plots
        # -----------------------
        if not df_reason.empty:
            try:
                fig = generate_popularity_plot(
                    df_reason,
                    target_dict_mapping[target_clean],
                    "Reason",
                )
                fig.savefig(
                    f"{save_dir}/word_popularity_reason_{target_clean}.png"
                )
            except Exception as e:
                print(f"  ⚠️ Failed Reason popularity plot: {e}")
        else:
            print("  ⚠️ df_reason is empty — skipping Reason popularity plot")

        if not df_note.empty:
            try:
                fig = generate_popularity_plot(
                    df_note,
                    target_dict_mapping[target_clean],
                    "note",
                )
                fig.savefig(
                    f"{save_dir}/word_popularity_note_{target_clean}.png"
                )
            except Exception as e:
                print(f"  ⚠️ Failed Note popularity plot: {e}")
        else:
            print("  ⚠️ df_note is empty — skipping Note popularity plot")

        # -----------------------
        # Alignment plot
        # -----------------------
        if not df_reason.empty and not df_note.empty:
            try:
                merged = pd.merge(
                    df_reason,
                    df_note,
                    on="word",
                    suffixes=("_input", "_output"),
                )

                if merged.empty:
                    print("  ⚠️ Merged dataframe is empty — skipping alignment plot")
                    continue

                merged["frequency_ratio"] = (
                    merged["frequency_output"] / merged["frequency_input"]
                )

                df_for_plotting = merged[
                    [
                        "word",
                        "change_in_metric_input",
                        "change_in_metric_output",
                        "frequency_ratio",
                    ]
                ]

                fig = plot_input_output_alignment(
                    df_for_plotting,
                    target_dict_mapping[target_clean],
                )
                fig.savefig(
                    f"{save_dir}/word_alignment_{target_clean}.png"
                )

            except Exception as e:
                print(f"  ⚠️ Failed alignment plot: {e}")
        else:
            print("  ⚠️ One or both dataframes empty — skipping alignment plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing word stats.")
    parser.add_argument("save_dir", type=str, help="Directory to save figures.")
    parser.add_argument(
        "target_list",
        type=str,
        help="List of target names. E.g., \"['target1', 'target2']\"",
    )
    args = parser.parse_args()

    target_list = ast.literal_eval(args.target_list)
    plot_word_statistics(args.data_dir, args.save_dir, target_list)
