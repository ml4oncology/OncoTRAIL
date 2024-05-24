import logging
import sys
from common.src.prep import Splitter

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def gen_data_split(df, test_start_date, split_config, embedding, target):
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

    X_train = embedding[train_idx, :]
    Y_train = target[train_idx]
    X_eval = embedding[eval_idx, :]
    Y_eval = target[eval_idx]
    X_valid = embedding[valid_idx, :]
    Y_valid = target[valid_idx]
    X_test = embedding[test_idx, :]
    Y_test = target[test_idx]

    return X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, Y_test
