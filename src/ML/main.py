import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
import sys
import os
import torch
from oncotrail.config import date_lower_limit, date_upper_limit
from oncotrail.ML.data_pipeline import DataPreprocessor
from oncotrail.ML.training_pipeline import ModelTrainer
from oncotrail.ML.inference_pipeline import ModelInference
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def evaluate(Y, pred, split):
        auprc = average_precision_score(Y, pred)
        auroc = roc_auc_score(Y, pred)
        log_loss_value = log_loss(Y, pred)
        result = {"AUPRC": auprc, "AUROC": auroc, "log_loss_value": log_loss_value}
        return pd.DataFrame(result, index=[split])

def main_train(
    notes_path,
    embedding_path,
    results_dir,
    date_lower_limit,
    date_upper_limit,
    data_type,
    target_name,
    model_dir,
    split_config,
    hyperparam_eval,
    model_name,
    setup_str,
    end_devt_date
):
    """
    Main training pipeline
    """
    if model_name == "Midfusion":
        assert data_type == "notes-tabular", "Midfusion requires both tabular and note data."
    if target_name == "target_sex":
        assert data_type == "notes", "Implementation not yet available when sex is a target."

    # Extract LLM_name from setup_str
    if data_type != 'tabular' and 'nlp' not in data_type:
        LLM_name = setup_str.split("_")[0]
    else:
        LLM_name = None

    # Create file save string
    target_name_nospace = target_name.replace("_", "-")
    file_save_str = f"{model_name}_{setup_str}_{split_config}_{hyperparam_eval}_{data_type}_{target_name_nospace}"
    logger.info(file_save_str)

    # Load data
    if 'parquet.gzip' in notes_path:
        df = pd.read_parquet(notes_path)
    else:
        # Robust CSV loading: prefer reading without an index; if the file contains
        # a saved index it will show up as "Unnamed: 0" so read again with index_col=0.
        try:
            df_noidx = pd.read_csv(notes_path, header=0)
        except Exception:
            # fallback: try reading with index_col=0
            df = pd.read_csv(notes_path, index_col=0)
        else:
            if 'Unnamed: 0' in df_noidx.columns:
                df = pd.read_csv(notes_path, index_col=0)
            else:
                df = df_noidx

        # If reading produced MultiIndex columns, flatten them to single-level names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

    # restrict date range
    df = df[(df['treatment_date'] >= date_lower_limit) & (df['treatment_date'] <= date_upper_limit)].copy()

    df.reset_index(drop=True, inplace=True)

    # Get target mask
    mask = None
    if target_name != "target_sex":
        mask = (df[target_name] != -1).to_numpy()
    else:
        raise NotImplementedError

    # Update mask for treatment_date
    date_mask = ((df['treatment_date'] >= date_lower_limit) & (df['treatment_date'] <= date_upper_limit)).to_numpy()
    mask = mask & date_mask

    # Extract target
    target = df.loc[mask, target_name].to_numpy()
    
    # Load embedding if needed
    embedding = None
    if data_type in ["notes", "notes-tabular"]:
        with np.load(embedding_path) as data:
            embedding_unique = data["embeddings"]
            note_index = df['note_index'].to_numpy()
            embedding = embedding_unique[note_index, :]
            embedding = embedding[mask, :]

    # Filter dataframe
    if data_type in ["notes-tabular", "tabular"]:
        cols = df.columns
        targ_cols = cols[cols.str.contains("target")].tolist()
        extra_cols = ["cohort", "split", "note", "note_index", 
                      "stats_note_type", "stats_dictated_by",
                      "note_summary", "sentencized_tabular_data", "original_note"] +\
                          cols[cols.str.contains("date")].tolist()
        extra_cols.remove("treatment_date")
        keep_cols = [col for col in cols if col not in extra_cols + targ_cols]
        df = df.loc[mask, keep_cols]
    else:
        if mask is not None:
            df = df.loc[mask, ["mrn", "note", "treatment_date"]]
        else:
            df = df.loc[:, ["mrn", "note", "treatment_date"]]
    df.reset_index(drop=True, inplace=True)

    # Data preprocessing and splitting
    preprocessor = DataPreprocessor()
    data_splits = preprocessor.prepare_data(
        df, embedding, target, end_devt_date, split_config, 
        data_type, model_name
    )
    
    # Save preprocessing artifacts
    preprocessor.save_preprocessing_artifacts(model_dir, target_name_nospace, split_config, data_type, setup_str, model_name)

    # Model training
    trainer = ModelTrainer(
        data_splits['X_train'],
        data_splits['Y_train'],
        data_splits['X_eval'],
        data_splits['Y_eval'],
        data_splits['X_valid'],
        data_splits['Y_valid'],
        data_splits['X_test'],
        hyperparam_eval,
        model_dir,
        model_name,
        file_save_str,
        LLM_name,
        data_type
    )
    
    (
        train_pred,
        eval_pred,
        val_pred,
        test_pred,
        shap_values_test,
        corr_coeff
    ) = trainer.train()

    # Save results
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    np.savez(
        f"{results_dir}/predictdata_{file_save_str}.npz",
        train_pred=train_pred,
        val_pred=val_pred,
        test_pred=test_pred,
        eval_pred=eval_pred,
        Y_train=data_splits['Y_train'],
        Y_valid=data_splits['Y_valid'],
        Y_test=data_splits['Y_test'],
        Y_eval=data_splits['Y_eval'],
        mrn_train=data_splits['mrn_train'],
        mrn_valid=data_splits['mrn_valid'],
        mrn_test=data_splits['mrn_test'],
        mrn_eval=data_splits['mrn_eval'],
        shap_values_test=shap_values_test,
        corr_coeff=corr_coeff,
        var_names=data_splits['var_names']
    )

    # Evaluate and save metrics
    train_results = evaluate(data_splits['Y_train'], train_pred, split="train")
    valid_results = evaluate(data_splits['Y_valid'], val_pred, split="valid")
    test_results = evaluate(data_splits['Y_test'], test_pred, split="test")

    results = pd.concat([train_results, valid_results, test_results])
    results.to_csv(f"{results_dir}/{file_save_str}.csv")


def main_inference(
    notes_path,
    embedding_path,
    results_dir,
    date_lower_limit,
    date_upper_limit,
    data_type,
    target_name,
    model_file,
    preprocessing_file,
):
    """
    Main inference pipeline
    """
    # Load new data
    if 'parquet.gzip' in notes_path:
        df = pd.read_parquet(notes_path)
    else:
        df = pd.read_csv(notes_path, header=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

    # restrict date range
    df = df[(df['treatment_date'] >= date_lower_limit) & (df['treatment_date'] <= date_upper_limit)].copy()

    df.reset_index(drop=True, inplace=True)

    # get indices of target != -1
    mask = (df[target_name] != -1).to_numpy()

    # only extract target where index != -1
    target = df.loc[mask, target_name].to_numpy()

    # Load embedding if needed
    embedding = None
    if data_type in ["notes", "notes-tabular"]:
        with np.load(embedding_path) as data:
            embedding_unique = data["embeddings"]
            note_index = df['note_index'].to_numpy()
            embedding = embedding_unique[note_index, :]
            embedding = embedding[mask, :]

    # Filter dataframe (similar logic to training)
    if data_type in ["notes-tabular", "tabular"]:
        cols = df.columns
        targ_cols = cols[cols.str.contains("target")].tolist()
        extra_cols = ["cohort", "split", "note", "note_index", 
                      "stats_note_type", "stats_dictated_by",
                      "note_summary"] + cols[cols.str.contains("date")].tolist()
        extra_cols.remove("treatment_date")
        keep_cols = [col for col in cols if col not in extra_cols + targ_cols]
        df = df.loc[mask, keep_cols]
    else:
        if mask is not None:
            df = df.loc[mask, ["mrn", "note", "treatment_date"]]
        else:
            df = df.loc[:, ["mrn", "note", "treatment_date"]]
    df.reset_index(drop=True, inplace=True)

    # Initialize inference pipeline
    inference = ModelInference(model_file, preprocessing_file)
    inference.load_model_and_preprocessor()

    # Make predictions
    predictions = inference.predict(df, embedding, data_type)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    base = os.path.basename(model_file)
    filename, _ = os.path.splitext(base)

    # Save predictions
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    np.savez(
        f"{results_dir}/inference_{filename}.npz",
        test_pred=predictions,
        Y_test=target,
        mrn_test=df['mrn'].to_numpy()
    )

    # compute metrics
    test_results = evaluate(target, predictions, split="test")
    test_results.to_csv(f"{results_dir}/{filename}.csv")

    logger.info(f"Inference completed. Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True,
                       help="Whether to train a new model or run inference")
    parser.add_argument("notes_path", help="path of notes", type=str)
    parser.add_argument("embedding_path", help="path of embedding", type=str)
    parser.add_argument("results_dir", help="results directory", type=str)
    parser.add_argument("date_lower_limit", help="date lower limit", type=str)
    parser.add_argument("date_upper_limit", help="date upper limit", type=str)
    parser.add_argument("data_type", help="data type", type=str)
    parser.add_argument("target_name", help="name of target", type=str)

    # Training-specific arguments
    parser.add_argument("--model_dir", help="model directory", type=str)
    parser.add_argument("--split_config", help="configuration of train-valid-test split", type=str)
    parser.add_argument("--hyperparam_eval", help="function for hyperparameter evaluation", type=str)
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--setup_str", help="set up string", type=str)
    parser.add_argument("--end_devt_date", help="end date of development", type=str)

    # Inference-specific arguments
    parser.add_argument("--model_file", help="filepath of machine learning model", type=str)
    parser.add_argument("--preprocessing_file", help="filepath of preprocessing artifacts", type=str)
    args = parser.parse_args()

    if args.mode == "train":
        main_train(
            args.notes_path,
            args.embedding_path,
            args.results_dir,
            args.date_lower_limit,
            args.date_upper_limit,
            args.data_type,
            args.target_name,
            args.model_dir,
            args.split_config,
            args.hyperparam_eval,
            args.model_name,
            args.setup_str,
            args.end_devt_date
        )
    elif args.mode == "inference":
        main_inference(
            args.notes_path,
            args.embedding_path,
            args.results_dir,
            args.date_lower_limit,
            args.date_upper_limit,
            args.data_type,
            args.target_name,
            args.model_file,
            args.preprocessing_file
        )