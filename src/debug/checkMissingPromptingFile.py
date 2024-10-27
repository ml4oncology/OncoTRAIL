import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import json
import math


def checkMissingPromptingFile(
    data_dir, file_name, file_part_min, file_part_max, llm_name, target_name, suffix=""
):
    """
    Check which files had OOM in prompting.

    data_dir: directory path where the processed csv files are saved
    file_name: file name of data frames to be merged
    suffix: suffix for the file name
    file_part_min: minimum file part number
    file_part_max: maximum file part number
    """

    if len(suffix) == 0:
        add_suffix = ""
    else:
        add_suffix = f"_{suffix}"

    col_to_search = f"probability_{llm_name}_{target_name}"

    for ctr in range(file_part_min, file_part_max + 1):
        # load dataframe
        dfTemp = pd.read_csv(
            f"{data_dir}/{file_name}_part{ctr}{add_suffix}.csv", index_col=0
        )

        if col_to_search not in dfTemp.columns:
            print("Missing index", ctr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument("file_name", help="file name", type=str)  # file name
    parser.add_argument(
        "file_part_min", help="file part number minimum", type=int
    )  # minimum file part number
    parser.add_argument(
        "file_part_max", help="file part number maximum", type=int
    )  # maximum file part number
    parser.add_argument("llm_name", help="name of LLM", type=str)  # name of llm
    parser.add_argument(
        "target_name", help="name of target", type=str
    )  # name of target
    parser.add_argument("-s", "--suffix", help="suffix", type=str, default="")  # suffix
    args = parser.parse_args()

    checkMissingPromptingFile(
        args.data_dir,
        args.file_name,
        args.file_part_min,
        args.file_part_max,
        args.llm_name,
        args.target_name,
        args.suffix,
    )
