import numpy as np
import pandas as pd
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.as_posix()
sys.path.append(ROOT_DIR)
from src.config import start_test_date
from src.split import gen_data_split
from src.train import Trainer
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(
    notes_path,
    embedding_path,
    split_config,
    hyperparam_eval,
    model_name,
    setup_str,
    tabular,
    target_name,
    model_dir,
    results_dir,
):
    # save string for file
    file_save_str = (
        f"{model_name}_{setup_str}_{split_config}_{hyperparam_eval}_tabular{int(tabular)}_{target_name}"
    )
    logger.info(file_save_str)

    # load data frame
    df = pd.read_csv(f"{notes_path}", index_col=0)
    df.reset_index(drop=True, inplace=True)

    # get indices of target != -1
    mask = (df[target_name] != -1).to_numpy()

    # load embedding
    with np.load(f"{embedding_path}") as data:
        embedding = data["embeddings"]
        target = data[target_name]

    # only extract embedding and target where index != -1
    embedding = embedding[mask, :]
    target = target[mask]
    if tabular == 1:
        df = df.loc[mask, ["mrn", "treatment_date", "note", "stats_physician", target_name]]
    else:
        df = df.loc[mask, ["mrn", "treatment_date", "note", target_name]]
    df.reset_index(drop=True, inplace=True)

    # generate train-validation-test split
    X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, Y_test = gen_data_split(
        df, start_test_date, split_config, embedding, target, tabular, model_name
    )

    # call trainer on predictions
    trainer = Trainer(
        X_train,
        Y_train,
        X_eval,
        Y_eval,
        X_valid,
        Y_valid,
        X_test,
        hyperparam_eval,
        model_dir,
        model_name,
        file_save_str,
    )
    train_pred, val_pred, test_pred = trainer.run()

    # save data
    np.savez(
        f"{results_dir}/predData_{file_save_str}.npz",
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        Y_train=Y_train,
        Y_valid=Y_valid,
        Y_test=Y_test,
    )

    # evaluate errors
    def evaluate(Y, pred, split):
        auprc = average_precision_score(Y, pred)
        auroc = roc_auc_score(Y, pred)
        log_loss_value = log_loss(Y, pred)
        result = {"AUPRC": auprc, "AUROC": auroc, "log_loss_value": log_loss_value}

        return pd.DataFrame(result, index=[split])

    train_results = evaluate(Y_train, train_pred, split="train")
    valid_results = evaluate(Y_valid, val_pred, split="valid")
    test_results = evaluate(Y_test, test_pred, split="test")

    results = pd.concat([train_results, valid_results, test_results])
    results.to_csv(f"{results_dir}/{file_save_str}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notes_path", help="path of notes", type=str)  # path of notes
    parser.add_argument(
        "embedding_path", help="path of embedding", type=str
    )  # path of embedding
    parser.add_argument(
        "split_config", help="configuration of train-valid-test split", type=str
    )  # configuration of train-valid-test split
    parser.add_argument(
        "hyperparam_eval", help="function for hyperparameter evaluation", type=str
    )  # hyperparameter evaluation function
    parser.add_argument(
        "model_name", help="model name", type=str
    )  # name of machine learning model
    parser.add_argument(
        "setup_str", help="set up string", type=str
    )  # name of set up string
    parser.add_argument(
        "tabular", help="include tabular data", type=bool
    ) # include tabular data
    parser.add_argument("target_name", help="name of target", type=str)  # name of target
    parser.add_argument(
        "model_dir", help="model directory", type=str
    )  # directory to save model
    parser.add_argument(
        "results_dir", help="results directory", type=str
    )  # directory to save results

    args = parser.parse_args()

    main(
        args.notes_path,
        args.embedding_path,
        args.split_config,
        args.hyperparam_eval,
        args.model_name,
        args.setup_str,
        args.tabular,
        args.target_name,
        args.model_dir,
        args.results_dir,
    )
