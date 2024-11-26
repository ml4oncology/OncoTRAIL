import pandas as pd
import argparse
import logging
import glob
import os
import re
logger = logging.getLogger(__name__)

def combine_prompt_results(results_dir, target_names):

    target_list = target_names.split(",")
    for target in target_list:

        logger.info(f"target: {target}")

        # replace _ in target with -
        target_dash = target.replace("_", "-")
        
        concat_results_list = []

        # find all csv files in results_dir with target in name
        matching_files = glob.glob(os.path.join(results_dir, f"*{target_dash}*.csv"))

        for file in matching_files:
            df = pd.read_csv(file, index_col=0)
            concat_results_list.append(df)

        # concatenate all dataframes in list
        concat_df = pd.concat(concat_results_list)

        match = re.search(r'target[^/]*(?=\.csv)', file)
        str_identifier = match.group(0)

        # save concatenated dataframe to csv
        concat_df.to_csv(f"{results_dir}/summary_{str_identifier}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, help='results directory') # results directory
    parser.add_argument('target_names', type=str, help='Comma-separated list of targets') # targets
    args = parser.parse_args()
    combine_prompt_results(args.results_dir,
                           args.target_names)
