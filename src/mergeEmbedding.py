import numpy as np
import argparse


def mergeEmbedding(data_dir, data_file_name, save_dir, num_files, suffix=""):
    # initialize the dictionary here
    target_embedding_list = {}

    if len(suffix) == 0:
        add_suffix = ""
    else:
        add_suffix = f"_{suffix}"

    for idx in range(num_files):
        with np.load(f"{data_dir}/{data_file_name}_part{idx}{add_suffix}.npz") as data:
            # initialize all keys in the dictionary
            dict_keys = list(data.keys())
            if idx == 0:
                for elem in dict_keys:
                    target_embedding_list[elem] = []

            for elem in dict_keys:
                target_embedding_list[elem].append(data[elem])

    target_embedding_concat = {}
    for key in target_embedding_list:
        target_embedding_concat[key] = np.concatenate(target_embedding_list[key])

    np.savez(f"{save_dir}/{data_file_name}{add_suffix}.npz", **target_embedding_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="data directory", type=str)  # data directory
    parser.add_argument(
        "data_file_name", help="data file name", type=str
    )  # data file name
    parser.add_argument("save_dir", help="save directory", type=str)  # save directory
    parser.add_argument(
        "num_files", help="number of files", type=int
    )  # number of files
    parser.add_argument(
        "-s", "--suffix", help="file suffix", type=str, default=""
    )  # suffix to be added to file
    args = parser.parse_args()

    mergeEmbedding(
        args.data_dir, args.data_file_name, args.save_dir, args.num_files, args.suffix
    )
