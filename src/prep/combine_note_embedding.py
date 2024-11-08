import numpy as np
import argparse


def combine_note_embedding(data_dir, data_file_name, num_files):
    # initialize the dictionary here
    embedding_list = {}

    for idx in range(num_files):
        with np.load(f"{data_dir}/{idx}_{data_file_name}.npz") as data:
            # initialize all keys in the dictionary
            dict_keys = list(data.keys())
            if idx == 0:
                for elem in dict_keys:
                    embedding_list[elem] = []

            for elem in dict_keys:
                embedding_list[elem].append(data[elem])

    embedding_concat = {}
    for key in embedding_list:
        embedding_concat[key] = np.concatenate(embedding_list[key])

    np.savez(f"{data_dir}/{data_file_name}.npz", **embedding_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument(
        "data_file_name", help="data file name", type=str
    )  # data file name
    parser.add_argument(
        "num_files", help="number of files", type=int
    )  # number of files
    args = parser.parse_args()

    combine_note_embedding(
        args.data_dir, args.data_file_name, args.num_files
    )
