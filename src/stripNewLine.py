import pandas as pd
import os
import argparse
from pathlib import Path
import re


def stripNewLine(data_path, save_dir):
    # extract file name
    file_name = os.path.basename(data_path)
    file_name = Path(file_name).stem

    # load the clinical notes file
    clinical_notes = pd.read_csv(data_path, index_col=False)
    clinical_notes["note"] = clinical_notes["note"].apply(
        lambda x: x.rstrip("\n").replace("\n\n", "\n")
    )
    clinical_notes["note"] = clinical_notes["note"].apply(
        lambda x: re.sub(r"[\n]+", "\n", x)
    )

    clinical_notes.to_csv(f"{save_dir}/{file_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="data file path", type=str)  # data file path
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    args = parser.parse_args()

    stripNewLine(args.data_path, args.save_dir)
