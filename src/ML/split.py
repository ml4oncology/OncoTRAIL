import logging
import sys
import re
import numpy as np
import pandas as pd
from ml_common.prep import Splitter, PrepData
from sklearn.preprocessing import StandardScaler
from llm_notes_classification.ML.nlp import (process_df, 
                                             extract_top_ngrams,
                                             build_tfidf_matrix,
                                             build_count_matrix)


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def gen_data_split(
    df, test_start_date, split_config, embedding, target, data_type, model_name
):
    """
    data_type: notes, notes-tabular, tabular, nlp
    """

    if 'nlp' in data_type:
        # preprocess notes before splitting
        df = process_df(df, "note")

    splitter = Splitter()
    if split_config == "Temporal":
        train_eval_data, valid_data, test_data = splitter.split_data(
            df, test_start_date
        )

    elif split_config == "Random":
        devt_cohort, test_data = splitter.random_split(df, test_size=0.35)
        train_eval_data, valid_data = splitter.random_split(devt_cohort, test_size=0.2)

    train_data, eval_data = splitter.random_split(train_eval_data, test_size=0.15)

    index_sets = {
        "train": train_data.index.to_list(),
        "eval": eval_data.index.to_list(),
        "valid": valid_data.index.to_list(),
        "test": test_data.index.to_list(),
    }

    # Extract mrn for each split
    if model_name == "LR":
        mrn_sets = {
            "train": np.concatenate([train_data["mrn"].to_numpy(), eval_data["mrn"].to_numpy()]).astype(int),
            "eval": None,
            "valid": valid_data["mrn"].to_numpy().astype(int),
            "test": test_data["mrn"].to_numpy().astype(int),
        }
    else:
        mrn_sets = {
            "train": train_data["mrn"].to_numpy().astype(int),
            "eval": eval_data["mrn"].to_numpy().astype(int),
            "valid": valid_data["mrn"].to_numpy().astype(int),
            "test": test_data["mrn"].to_numpy().astype(int),
        }

    Y = {k: target[idx] for k, idx in index_sets.items()}

    if data_type in ["notes", "notes-tabular"]:
        # extract note data
        X = {k: embedding[idx, :] for k, idx in index_sets.items()}

    elif data_type == "tabular":
        X = {k: np.zeros((len(idx), 0)) for k, idx in index_sets.items()}

    elif 'nlp' in data_type:

        X = {}

        # generate the tf-idf vocabulary from the train set
        vocab = extract_top_ngrams(train_data, text_col="note_lemmatized_note", top_k=250)
        
        if model_name == "LR":
            train_data = pd.concat([train_data, eval_data])
            Y["train"] = np.concatenate([Y["train"], Y["eval"]])
            X["eval"], Y["eval"] = None, None

        # Build the feature matrix from training data
        if 'tfidf' in data_type:
            X["train"], word_vec = build_tfidf_matrix(train_data, text_col="note_lemmatized_note", vocabulary=vocab)
        else:
            X["train"], word_vec = build_count_matrix(train_data, text_col="note_lemmatized_note", vocabulary=vocab)
        X["train"] = X["train"].toarray().astype(float)

        # Transform eval, valid, test using same vectorizer and scaler
        for split in ["eval", "valid", "test"]:
            if split != "eval" or model_name != "LR":
                X[split] = word_vec.transform(eval(f"{split}_data")["note_lemmatized_note"]).toarray().astype(float)

        train_col_names = vocab

    else:
        raise ValueError(
            "data_type must be one of ['notes', 'notes-tabular', 'tabular', 'nlp']"
        )
    
    # Merge train + eval for LR if needed (already handled in nlp above)
    if model_name == "LR" and data_type in ["notes", "notes-tabular", "tabular"]:
        X["train"] = np.concatenate([X["train"], X["eval"]])
        Y["train"] = np.concatenate([Y["train"], Y["eval"]])
        X["eval"], Y["eval"] = None, None
    
    if "nlp" in data_type or data_type in ["notes", "notes-tabular"]:
        scaler = StandardScaler()
        X["train"] = scaler.fit_transform(X["train"])
        if X["eval"] is not None:
            X["eval"] = scaler.transform(X["eval"])
        X["valid"] = scaler.transform(X["valid"])
        X["test"] = scaler.transform(X["test"])

    if data_type in ["notes-tabular","tabular"]:
        if data_type == "notes-tabular":
            # convert physician name to tabular data and concatenate to embedding data
            physician_names_train = find_unique_phys(train_data)

            # convert physician name to tabular
            for split, df_split in zip(["train", "eval", "valid", "test"], [train_data, eval_data, valid_data, test_data]):
                if model_name == "LR" and split == "train":
                    df_concat = pd.concat([train_data, eval_data])
                    phys = convert_physician_name_tabular(df_concat, physician_names_train)
                elif model_name != "LR" or split != "eval":
                    phys = convert_physician_name_tabular(df_split, physician_names_train)
                else:
                    continue  # Skip eval for LR
                X[split] = np.concatenate([X[split], phys], axis=1)

        prep = PrepData()
        data_frames = {
            "train": prep.transform_data(train_data, data_name="training"),
            "eval": prep.transform_data(eval_data, data_name="evaluation"),
            "valid": prep.transform_data(valid_data, data_name="validation"),
            "test": prep.transform_data(test_data, data_name="test"),
        }

        # remove columns that are not needed
        drop_cols = ["mrn", "treatment_date", "stats_physician"]
        for k in data_frames:
            data_frames[k].drop(columns=drop_cols, inplace=True)

        if model_name == "LR":
            combined_df = pd.concat([data_frames["train"], data_frames["eval"]])
            X["train"] = np.concatenate([X["train"], combined_df.to_numpy()], axis=1)
        else:
            for split in ["train", "eval"]:
                X[split] = np.concatenate([X[split], data_frames[split].to_numpy()], axis=1)

        X["valid"] = np.concatenate([X["valid"], data_frames["valid"].to_numpy()], axis=1)
        X["test"] = np.concatenate([X["test"], data_frames["test"].to_numpy()], axis=1)

    if 'nlp' not in data_type:
        train_col_names = train_data.columns.to_list()

    return (X["train"], Y["train"], 
            X["eval"], Y["eval"],
            X["valid"], Y["valid"],
            X["test"], Y["test"],
            train_col_names,
            mrn_sets["train"], 
            mrn_sets["eval"], 
            mrn_sets["valid"], 
            mrn_sets["test"])


def convert_str_list(y):
    
    match = re.search(r'\[.*?\]', y)
    y = match.group(0)

    # Remove the brackets and split the string by single quotes
    words = y.strip("[]").split("'")

    # Filter out empty strings and spaces
    result = [word for word in words if word.strip()]

    return result


def find_unique_phys(df):
    physician_names = df["stats_physician"].unique()
    unique_physician_names = []
    for elem in physician_names:
        unique_physician_names = unique_physician_names + convert_str_list(elem)
    unique_physician_names = list(set(unique_physician_names))
    return np.array(unique_physician_names)


def convert_physician_name_tabular(df, unique_phys):
    physician_names_tabular = []
    physician_names_values = df["stats_physician"].values

    for elem in physician_names_values:
        phys_list = np.array(convert_str_list(elem))
        temp = [int(phys in phys_list) for phys in unique_phys]
        physician_names_tabular.append(temp)

    return np.array(physician_names_tabular)
