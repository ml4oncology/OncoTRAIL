import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from pathlib import Path
import sys

from llm_notes_classification.config import start_test_date
from llm_notes_classification.ML.split import gen_data_split
from llm_notes_classification.ML.train import Trainer
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
    data_type,
    target_name,
    model_dir,
    results_dir,
):
    """
    Main code for predicting risk of undesired cancer events
    using clinical notes and tabular data.

    notes_path: file path to the notes and tabular data frame
    embedding_path: file path to the embedding of the clinical note. implies selection of LLM
    split_config: temporal or random split
    hyperparam_eval: select best hyperparameters based on AUC or logloss
    model_name: machine learning/deep learning model
    setup_str: combination of LLM and note configuration
    data_type: notes, notes-tabular, tabular
    target_name: name of target
    model_dir: directory where to save trained model parameters
    results_dir: directory where to save the results of the model runs
    """

    if model_name == "Midfusion":
        assert data_type == "notes-tabular", "Midfusion requires both tabular and note data."
    if target_name == "target_sex":
        assert data_type == "notes", "Implementation not yet available when sex is a target."

    # extract LLM_name from setup_str
    if data_type != 'tabular':
        LLM_name = setup_str.split("_")[0]
    else:
        LLM_name = None

    # save string for file
    target_name_nospace = target_name.replace("_", "-")
    file_save_str = f"{model_name}_{setup_str}_{split_config}_{hyperparam_eval}_{data_type}_{target_name_nospace}"
    logger.info(file_save_str)

    # load data frame
    if 'parquet.gzip' in notes_path:
        df = pd.read_parquet(notes_path)
    else:
        df = pd.read_csv(notes_path, index_col=0)    
    df.reset_index(drop=True, inplace=True)

    # get indices of target != -1
    mask = None
    if target_name != "target_sex":
        mask = (df[target_name] != -1).to_numpy()

        # only extract target where index != -1
        target = df.loc[mask, target_name].to_numpy()
    else:
        raise NotImplementedError
    
    embedding = None
    if data_type != 'tabular':
        # load embedding
        with np.load(embedding_path) as data:
            embedding_unique = data["embeddings"]

            # create the full embedding matrix according
            # to note_index
            note_index = df['note_index'].to_numpy()
            embedding = embedding_unique[note_index, :]

            # only extract embedding where index != -1
            embedding = embedding[mask, :]

    if data_type in ["notes-tabular","tabular"]:
        cols = df.columns
        targ_cols = cols[cols.str.contains("target")].tolist()
        # TO-DO: edit stats_noteType name in anchor code
        extra_cols = ["cohort", "split", "note", "note_index", "stats_noteType"] + cols[
            cols.str.contains("date")
        ].tolist()
        extra_cols.remove("treatment_date")
        keep_cols = [col for col in cols if col not in extra_cols + targ_cols]
        df = df.loc[mask, keep_cols]
    else:
        if mask is not None:
            df = df.loc[mask, ["mrn", "treatment_date"]]
        else:
            df = df.loc[:, ["mrn", "treatment_date"]]
    df.reset_index(drop=True, inplace=True)

    # generate train-validation-test split
    X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, Y_test = gen_data_split(
        df, start_test_date, split_config, embedding, target, data_type, model_name
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
        LLM_name,
    )
    train_pred, val_pred, test_pred = trainer.run()

    # save data
    np.savez(
        f"{results_dir}/predictdata_{file_save_str}.npz",
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
        "data_type", help="data type", type=str
    )  # data type: notes, notes-tabular, tabular
    parser.add_argument(
        "target_name", help="name of target", type=str
    )  # name of target
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
        args.data_type,
        args.target_name,
        args.model_dir,
        args.results_dir,
    )