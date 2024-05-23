import pandas as pd
import os
import argparse
from pathlib import Path


def splitDataFrame(data_path, save_dir, num_rows_per_part):
    # extract file name
    file_name = os.path.basename(data_path)
    file_name = Path(file_name).stem

    # load the clinical notes file
    clinical_notes = pd.read_csv(data_path, index_col=False)

    # split into parts for parallel embedding processing
    for ctr, idx in enumerate(range(0, len(clinical_notes), num_rows_per_part)):
        clinical_notes.loc[idx : idx + num_rows_per_part - 1, :].to_csv(
            f"{save_dir}/{file_name}_part{ctr}.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="data file path", type=str)  # data file path
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument(
        "num_rows_per_part", help="number of rows per part", type=int
    )  # number of rows per dataframe
    args = parser.parse_args()

    splitDataFrame(args.data_path, args.save_dir, args.num_rows_per_part)
