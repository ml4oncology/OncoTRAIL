import pandas as pd
import argparse
import logging
import glob
import os
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO
)

def combine_prompt_results(results_dir, notes_df_path):

    logger.info(f"results_dir: {results_dir}")

    concat_results_list = []

    # loop over files in directory with pattern mrn*_summary.csv
    matching_files = glob.glob(os.path.join(results_dir, f"mrn*_summary.csv"))

    for file in matching_files:

        df = pd.read_csv(file, index_col=0)
        concat_results_list.append(df)

    # concatenate all dataframes in list
    concat_df = pd.concat(concat_results_list).reset_index(drop=True)

    # save concatenated dataframe to file
    concat_df.to_csv(f"{results_dir}/combined_notes_summary.csv")

    concat_df["treatment_date"] = pd.to_datetime(concat_df["treatment_date"]).dt.date

    # load notes_df
    notes_df = pd.read_csv(notes_df_path, index_col=0)
    notes_df["treatment_date"] = pd.to_datetime(notes_df["treatment_date"]).dt.date

    # if 'note_summary' is a column in notes_df, drop it
    if 'note_summary' in notes_df.columns:
        notes_df = notes_df.drop('note_summary', axis=1)

    # rename 'Summary' column in concat_df to 'note_summary'
    concat_df = concat_df.rename(columns={'Summary': 'note_summary'})

    # merge concat_df and notes_df on 'mrn' and 'treatment_date'
    df = pd.merge(concat_df[['mrn', 'treatment_date', 'note_summary']], notes_df, on=['mrn', 'treatment_date'])

    # save df to notes_df_path
    df.to_csv(notes_df_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, help='results directory') # results directory
    parser.add_argument('notes_df_path', type=str, help='file path and file name to notes dataframe') # notes dataframe path
    args = parser.parse_args()
    combine_prompt_results(args.results_dir, args.notes_df_path)
