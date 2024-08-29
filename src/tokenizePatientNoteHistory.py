import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer


def tokenizeNotes(data_dir, model_name, model_path, notes_file_name):
    # notes = pd.read_csv(f"{data_dir}/{notes_file_name}.csv")
    notes = pd.read_parquet(
        f"{data_dir}/{notes_file_name}.parquet.gzip",
        engine="pyarrow",
        use_nullable_dtypes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/{model_name}")
    # Batched tokenization
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mrn_list = []
    concat_note_list = []
    tokenized_len_list = []

    notes_grouped = notes.groupby("MRN")
    for name, group in notes_grouped:
        mrn_list.append(name)
        cat_note = (
            group.groupby("MRN")
            .agg(processed_note=("clinical_notes", lambda x: "\n".join(x)))
            .reset_index()["processed_note"]
            .values[0]
        )
        concat_note_list.append(cat_note)
        tokenized_note = tokenizer.tokenize(cat_note)
        tokenized_len_list.append(len(tokenized_note))

    concatenate_notes_df = pd.DataFrame(
        {
            "mrn": mrn_list,
            "concatenated_note": concat_note_list,
            "tokenized_len": tokenized_len_list,
        }
    )

    concatenate_notes_df.to_csv(f"{data_dir}/{notes_file_name}_tokenized.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument("model_name", help="model name", type=str)  # model name
    parser.add_argument("model_path", help="path to model", type=str)  # path to model
    parser.add_argument(
        "notes_file_name", help="notes file name", type=str
    )  # path to notes file
    args = parser.parse_args()

    tokenizeNotes(args.data_dir, args.model_name, args.model_path, args.notes_file_name)
