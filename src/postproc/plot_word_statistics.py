import pandas as pd
import argparse
import ast
import os
from llm_notes_classification.postproc.word_analyzer_utils import (
    generate_popularity_plot,
    plot_input_output_alignment,
    adjust_p_values
)
from llm_notes_classification.constants import target_dict_mapping


def plot_word_statistics(data_dir, save_dir, target_list):

    # concatenate all data for p-value adjustment
    reason_dfs = []
    note_dfs = []
    for target in target_list:
        target_clean = target.replace("_", "-")
        reason_path = os.path.join(
            data_dir,
            f"word_stats_all_Reason_{target_clean}_before_pval_adjustment.csv"
        )
        note_path = os.path.join(
            data_dir,
            f"word_stats_all_note_{target_clean}_before_pval_adjustment.csv"
        )

        if os.path.exists(reason_path):
            df_reason = pd.read_csv(reason_path)
            df_reason['target'] = target_clean
            reason_dfs.append(df_reason)

        if os.path.exists(note_path):
            df_note = pd.read_csv(note_path)
            df_note['target'] = target_clean
            note_dfs.append(df_note)
    
    df_concat_reason = pd.concat(reason_dfs, ignore_index=True)
    df_concat_note = pd.concat(note_dfs, ignore_index=True)

    # Adjust p-values
    df_concat_reason = adjust_p_values(df_concat_reason, 'BY')
    df_concat_note = adjust_p_values(df_concat_note, 'BY')

    # only keep significant words
    df_concat_reason = df_concat_reason.loc[
        df_concat_reason['p_adj'] < 0.05
    ].copy()
    df_concat_note = df_concat_note.loc[
        df_concat_note['p_adj'] < 0.05
    ].copy()

    os.makedirs(save_dir, exist_ok=True)

    for target in target_list:
        target_clean = target.replace("_", "-")
        print(f"\nProcessing target: {target_clean}")

        # -----------------------
        # Load CSVs safely
        # -----------------------
        try:
            df_reason = df_concat_reason.loc[
                df_concat_reason['target'] == target_clean
            ].copy()
        except Exception as e:
            print(f"Could not load Reason CSV for {target_clean}: {e}")
            df_reason = pd.DataFrame()

        df_reason.to_csv(
                f"{data_dir}/word_stats_all_Reason_{target_clean}.csv"
            )

        try:
            df_note = df_concat_note.loc[
                df_concat_note['target'] == target_clean
            ].copy()
        except Exception as e:
            print(f"Could not load Note CSV for {target_clean}: {e}")
            df_note = pd.DataFrame()

        df_note.to_csv(
                f"{data_dir}/word_stats_all_note_{target_clean}.csv"
            )

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
                    f"{save_dir}/word_popularity_reason_{target_clean}.png", dpi=300, bbox_inches="tight"
                )
            except Exception as e:
                print(f"Failed Reason popularity plot: {e}")
        else:
            print("df_reason is empty — skipping Reason popularity plot")

        if not df_note.empty:
            try:
                fig = generate_popularity_plot(
                    df_note,
                    target_dict_mapping[target_clean],
                    "note",
                )
                fig.savefig(
                    f"{save_dir}/word_popularity_note_{target_clean}.png", dpi=300, bbox_inches="tight"
                )
            except Exception as e:
                print(f"Failed Note popularity plot: {e}")
        else:
            print("df_note is empty — skipping Note popularity plot")

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
                    print("Merged dataframe is empty — skipping alignment plot")
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
                    f"{save_dir}/word_alignment_{target_clean}.png", dpi=300, bbox_inches="tight"
                )

            except Exception as e:
                print(f"Failed alignment plot: {e}")
        else:
            print("One or both dataframes empty — skipping alignment plot")


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
