import numpy as np
import pandas as pd
import argparse
from transformers import AutoTokenizer


def tokenizeNotes(data_dir, model_name, model_path, notes_file_name):
    notes = pd.read_csv(f"{data_dir}/{notes_file_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/{model_name}")
    # Batched tokenization
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # convert column to list of strings
    str_to_tokenize = notes["note"].tolist()

    nnz_list = []

    # do in batches of 1000
    slice_size = 1000
    for i in range(0, len(str_to_tokenize), slice_size):
        str_to_tokenize_slice = str_to_tokenize[i : i + slice_size]
        encoded_batch_slice = tokenizer.batch_encode_plus(
            str_to_tokenize_slice, padding=True, truncation=False, return_tensors="np"
        )
        input_ids_slice = encoded_batch_slice["input_ids"]
        nnz_slice = np.sum(input_ids_slice != tokenizer.pad_token_id, axis=1)
        nnz_slice = nnz_slice.ravel()
        nnz_list.append(nnz_slice)

    nnz = np.concatenate(nnz_list)

    notes[f"token_length_{model_name}"] = nnz
    notes.to_csv(f"{data_dir}/{notes_file_name}")


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
