import pandas as pd
import argparse
import logging
import glob
import os
import re
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO         # Log level (you can adjust it to INFO, DEBUG, etc.)
)

def extract_proba(raw_string):
    # extract probability value from the raw string
    match = re.search(r"'Probability'\s*:\s*['\"]?(\d+\.\d+)['\"]?", raw_string)
    probability_value = float(match.group(1))
    return probability_value

def combine_prompt_results(results_dir, target_names):

    logger.info(f"results_dir: {results_dir}")

    target_list = target_names.split(",")
    for target in target_list:

        logger.info(f"target: {target}")

        # replace _ in target with -
        target_dash = target.replace("_", "-")

        concat_results_list = []

        # find all csv files in results_dir with target in name
        matching_files = glob.glob(os.path.join(results_dir, f"*{target_dash}*.csv"))

        for file in matching_files:

            if 'summary' in file: continue
            try:
                df = pd.read_csv(file, index_col=0)
            except:
                continue

            # only retain rows where 'Probability' is a float
            if 'Probability' in df.columns:
                df = df[pd.to_numeric(df['Probability'], errors='coerce').notna()]

            if 'Prediction' in df.columns:
                df = df[pd.to_numeric(df['Prediction'], errors='coerce').notna()]

            # only retain rows where 'mrn' is not nan
            df = df[df['mrn'].notna()]

            # only retain rows where 'treatment_date' is not nan
            df = df[df['treatment_date'].notna()]

            # check the number of rows of df
            if df.shape[0] == 1:
                concat_results_list.append(df)
            else:
                try:
                    # find the average of the prediction column
                    if 'Prediction' in df.columns:
                        df['Probability'] = df['Prediction'].mean()
                        df['std_Prob'] = df['Prediction'].std()
                    else:
                        df['std_Prob'] = df['Probability'].std()
                        df['Probability'] = df['Probability'].mean()
                except:
                    df['Probability'] = pd.NA
                    df['std_Prob'] = pd.NA

                concat_results_list.append(df[['Reason', 'Probability', 'std_Prob', target, 'mrn', 'treatment_date']].head(1))

        # concatenate all dataframes in list
        concat_df = pd.concat(concat_results_list).reset_index(drop=True)

        # if the probability is nan, try to extract it from the raw column
        for i, proba in enumerate(concat_df['Probability']):
            if pd.isna(proba):
                try:
                    concat_df.loc[i,'Probability'] = extract_proba(concat_df.loc[i,'Raw'])
                except:
                    concat_df.loc[i,'Probability'] = pd.NA

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
