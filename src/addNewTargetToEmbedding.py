import numpy as np
import argparse
import pandas as pd


def addTargetToEmbedding(embed_data_dir, data_file_name, notes_path, notes_fname):

    # load embedding
    embedding_data = dict(np.load(f'{embed_data_dir}/{data_file_name}.npz'))

    # load notes
    df = pd.read_csv(f'{notes_path}/{notes_fname}.csv', index_col=0)

    # extract target from notes
    df['female'] = df['female'].astype(int)

    # save target to embedding
    embedding_data['target_sex'] = df['female'].to_numpy()

    # save embedding
    np.savez(f"{embed_data_dir}/{data_file_name}.npz", **embedding_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embed_data_dir", help="data directory", type=str)  # data directory
    parser.add_argument(
        "data_file_name", help="data file name", type=str
    )  # data file name
    parser.add_argument("notes_path", help="path to notes", type=str)  # path to notes
    parser.add_argument("notes_fname", help="file name of notes", type=str)  # file name of notes
    args = parser.parse_args()

    addTargetToEmbedding(args.embed_data_dir, args.data_file_name, args.notes_path, args.notes_fname)
