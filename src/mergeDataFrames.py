import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import json
import math

def mergeDataFrames(data_dir, save_dir, file_name, file_part_min, file_part_max, suffix=''):
    """
    Merge processed data frames into a single data frame.

    data_dir: directory path where the processed csv files are saved
    save_dir: directory path where merged parquet file will be saved
    file_name: file name of data frames to be merged
    suffix: suffix for the file name
    file_part_min: minimum file part number of files to be merged
    file_part_max: maximum file part number of files to be merged
    """

    if len(suffix) == 0:
        add_suffix = ''
    else:
        add_suffix = f'_{suffix}'

    merged_df_list = []
    for ctr in range(file_part_min, file_part_max + 1):
        # load dataframe
        dfTemp = pd.read_csv( f"{data_dir}/{file_name}_part{ctr}{add_suffix}.csv", index_col = 0 )
        merged_df_list.append( dfTemp )

    merged_df = pd.concat( merged_df_list )

    # Find columns containing 'Unnamed:'
    unnamed_columns = [col for col in merged_df.columns if 'Unnamed:' in col]

    # remove unnamed columns
    merged_df.drop(columns=unnamed_columns, inplace=True)
    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv(f'{save_dir}/{file_name}{add_suffix}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help = "data directory", type = str) # data directory
    parser.add_argument("save_dir", help = "save directory", type = str) # save directory
    parser.add_argument("file_name", help = "file name", type = str) # file name
    parser.add_argument("file_part_min", help = "file part number minimum", type = int) # minimum file part number
    parser.add_argument("file_part_max", help = "file part number maximum", type = int) # maximum file part number
    parser.add_argument('-s',"--suffix", help = "suffix", type = str, default='') # suffix
    args = parser.parse_args()

    mergeDataFrames(args.data_dir, args.save_dir, args.file_name, args.file_part_min, args.file_part_max, args.suffix)