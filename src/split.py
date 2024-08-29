import logging
import sys
import numpy as np
import pandas as pd
from common.src.prep import Splitter
from common.src.prep import PrepData
from sklearn.preprocessing import StandardScaler

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def gen_data_split(
    df, test_start_date, split_config, embedding, target, tabular, model_name
):
    """
    tabular: 0 - notes only, 1 - notes+tabular, 2 - tabular only
    """
    splitter = Splitter()
    if split_config == "Temporal":
        train_eval_data, valid_data, test_data = splitter.split_data(
            df, test_start_date
        )

    elif split_config == "Random":
        devt_cohort, test_data = splitter.random_split(df, test_size=0.35)
        train_eval_data, valid_data = splitter.random_split(devt_cohort, test_size=0.2)

    train_data, eval_data = splitter.random_split(train_eval_data, test_size=0.15)

    train_idx = train_data.index.to_list()
    eval_idx = eval_data.index.to_list()
    valid_idx = valid_data.index.to_list()
    test_idx = test_data.index.to_list()

    # extract target
    Y_train = target[train_idx]
    Y_eval = target[eval_idx]
    Y_valid = target[valid_idx]
    Y_test = target[test_idx]

    if tabular < 2:
        # extract note data
        X_train = embedding[train_idx, :]
        X_eval = embedding[eval_idx, :]
        X_valid = embedding[valid_idx, :]
        X_test = embedding[test_idx, :]

        # if Logistic Regression, merge train and evaluation data sets
        if model_name == "LR":
            X_train = np.concatenate([X_train, X_eval])
            Y_train = np.concatenate([Y_train, Y_eval])
            X_eval = None
            Y_eval = None

        # preprocess the note data by scaling and centering
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_eval is not None:
            X_eval = scaler.transform(X_eval)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
    else:
        X_train = np.zeros((len(train_idx), 0))
        X_eval = np.zeros((len(eval_idx), 0))
        X_valid = np.zeros((len(valid_idx), 0))
        X_test = np.zeros((len(test_idx), 0))

        if model_name == "LR":
            X_train = np.concatenate([X_train, X_eval])
            Y_train = np.concatenate([Y_train, Y_eval])
            X_eval = None
            Y_eval = None

    if tabular >= 1:
        if tabular == 1:
            # convert physician name to tabular data and concatenate to embedding data
            physician_names_train = find_unique_phys(train_data)

            # convert physician name to tabular for
            # train data
            if model_name == "LR":
                train_physician = convert_physician_name_tabular(
                    pd.concat([train_data, eval_data], axis=0), physician_names_train
                )
            else:
                train_physician = convert_physician_name_tabular(
                    train_data, physician_names_train
                )
            X_train = np.concatenate([X_train, train_physician], axis=1)

            # eval data
            if model_name != "LR":
                eval_physician = convert_physician_name_tabular(
                    eval_data, physician_names_train
                )
                X_eval = np.concatenate([X_eval, eval_physician], axis=1)

            # valid data
            valid_physician = convert_physician_name_tabular(
                valid_data, physician_names_train
            )
            X_valid = np.concatenate([X_valid, valid_physician], axis=1)

            # test data
            test_physician = convert_physician_name_tabular(
                test_data, physician_names_train
            )
            X_test = np.concatenate([X_test, test_physician], axis=1)

        # process the other numerical features
        prep = PrepData()
        train_data = prep.transform_data(train_data, data_name="training")
        eval_data = prep.transform_data(eval_data, data_name="evaluation")
        valid_data = prep.transform_data(valid_data, data_name="validation")
        test_data = prep.transform_data(test_data, data_name="test")

        # remove columns that are not needed
        train_data.drop(
            columns=["mrn", "treatment_date", "stats_physician"], inplace=True
        )
        eval_data.drop(
            columns=["mrn", "treatment_date", "stats_physician"], inplace=True
        )
        valid_data.drop(
            columns=["mrn", "treatment_date", "stats_physician"], inplace=True
        )
        test_data.drop(
            columns=["mrn", "treatment_date", "stats_physician"], inplace=True
        )

        if model_name != "LR":
            X_train = np.concatenate([X_train, train_data.to_numpy()], axis=1)
            X_eval = np.concatenate([X_eval, eval_data.to_numpy()], axis=1)
        else:
            X_train = np.concatenate(
                [X_train, pd.concat([train_data, eval_data], axis=0).to_numpy()], axis=1
            )

        X_valid = np.concatenate([X_valid, valid_data.to_numpy()], axis=1)
        X_test = np.concatenate([X_test, test_data.to_numpy()], axis=1)

    return X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, Y_test


def convert_str_list(y):
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
