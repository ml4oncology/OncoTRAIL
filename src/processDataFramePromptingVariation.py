import pandas as pd
import os
import argparse
from pathlib import Path


def processDataFramePromptingVariation(data_path, save_dir, num_mrns, num_replicates):
    # extract file name
    file_name = os.path.basename(data_path)
    file_name = Path(file_name).stem

    # load the clinical notes file
    clinical_notes = pd.read_csv(data_path, index_col=False)
    clinical_notes_filter = (
        clinical_notes.sample(n=clinical_notes.shape[0], random_state=100)
        .groupby(["mrn"])
        .first()
        .reset_index(drop=True)
        .sample(n=num_mrns, random_state=100)
        .reset_index(drop=True)
    )

    print(clinical_notes_filter)

    # split into parts for parallel embedding processing
    for idx in range(len(clinical_notes_filter)):
        temp_df = clinical_notes_filter.loc[idx:idx, :].copy()
        # temp_df = clinical_notes_filter.loc[idx:idx,:].copy().to_csv(f"{save_dir}/{file_name}_template_patient{idx}.csv")
        temp_df.loc[temp_df.index.repeat(num_replicates)].reset_index(drop=True).to_csv(
            f"{save_dir}/{file_name}_template_patient{idx}.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="data file path", type=str)  # data file path
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument("num_mrns", help="number of mrns", type=int)  # number of mrns
    parser.add_argument(
        "num_replicates", help="number of replicates", type=int
    )  # number of replicates
    args = parser.parse_args()

    processDataFramePromptingVariation(
        args.data_path, args.save_dir, args.num_mrns, args.num_replicates
    )
