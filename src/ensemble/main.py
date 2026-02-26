import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Dict
import logging
import os
import glob
import argparse
import json
import sys
import itertools
from pathlib import Path
from autogluon.tabular import TabularPredictor
from llm_notes_classification.finetune.post_proc_results import format_lr

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_predictions(model_name, path, is_test=False):
    """
    Load predictions and MRNs from a given path depending on whether it's 'prompting' or a standard ML model.
    Returns a DataFrame with columns: mrn, prob
    """
    # print model_name
    logger.info(f"Loading predictions for model: {model_name} from path: {path}")
    if model_name == 'prompting':
        df = pd.read_csv(path)
        df = df[['mrn', 'Probability']].rename(columns={'Probability': 'prob'})
        return df
    elif model_name == 'finetune':
        if is_test:
            splits = ["test"]
        else:
            splits = ["train", "valid", "eval"]
        
        mrns_list = []
        probs_list = []
        
        for split in splits:
            split_path_pattern = path.format(split)
            matches = glob.glob(split_path_pattern)
            assert len(matches) == 1, f"Expected 1 file for split '{split}', found {matches}"
            
            df_split = pd.read_csv(matches[0])
            # probs_np = np.stack(df_split["prob"].values)  # shape (n_samples, 2)
            probs_np = np.stack(df_split["prob"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).values)
            probs_split = probs_np[:, 1]  # column for class 1
            
            mrns_list.append(df_split["mrn"].values)
            probs_list.append(probs_split)
        
        mrns = np.concatenate(mrns_list)
        probs = np.concatenate(probs_list)
        
        return pd.DataFrame({'mrn': mrns, 'prob': probs})
    else:
        data = np.load(path, allow_pickle=True)
        if is_test:
            mrns = data['mrn_test']
            probs = data['test_pred'].ravel()
        else:

            mrn_arrays = [data['mrn_train'], data['mrn_valid']]
            prob_arrays = [data['train_pred'].ravel(), data['val_pred'].ravel()]

            if data['mrn_eval'] is not None and data['mrn_eval'].shape != ():
                mrn_arrays.append(data['mrn_eval'])
                prob_arrays.append(data['eval_pred'].ravel())

            # Concatenate the arrays
            mrns = np.concatenate(mrn_arrays)
            probs = np.concatenate(prob_arrays)
        return pd.DataFrame({'mrn': mrns, 'prob': probs})

def ensemble_aggregation(train_dict: Dict[str, str], 
                         test_dict: Dict[str, str], 
                         train_set_df: pd.DataFrame,
                         test_set_df: pd.DataFrame,
                         target_name: str,
                         save_dir):
    # Step 1: Load all train predictions
    train_dfs = {}
    common_mrns = None

    for model_name, path in train_dict.items():
        df = load_predictions(model_name, path, is_test=False)
        df = df.drop_duplicates(subset='mrn')
        df = df.set_index('mrn')  # Set index BEFORE calculating common_mrns
        train_dfs[model_name] = df

        if common_mrns is None:
            common_mrns = set(df.index)
        else:
            common_mrns = common_mrns.intersection(set(df.index))
    
    common_mrns = sorted(list(common_mrns))
    logger.info(f"Number of common MRNs in train: {len(common_mrns)}")

    # Build matrix of train probabilities
    train_matrix = np.stack([train_dfs[model].loc[common_mrns, 'prob'].values for model in train_dict], axis=1)
    y_train = train_set_df.set_index('mrn').loc[common_mrns][target_name].values

    # Step 2: Load all test predictions
    test_dfs = {}
    test_mrns = set(test_set_df['mrn'])

    for model_name, path in test_dict.items():
        df = load_predictions(model_name, path, is_test=True)
        df = df.drop_duplicates(subset='mrn')
        df = df[df['mrn'].isin(test_mrns)]
        df = df.set_index('mrn')
        test_dfs[model_name] = df
    
    common_test_mrns = sorted(list(set.intersection(*[set(df.index) for df in test_dfs.values()])))
    logger.info(f"Number of common MRNs in test: {len(common_test_mrns)}")

    test_matrix = np.stack([test_dfs[model].loc[common_test_mrns, 'prob'].values for model in test_dict], axis=1)
    y_test = test_set_df.set_index('mrn').loc[common_test_mrns][target_name].values

    # Step 3: Aggregation and AUC computation
    aggregations = {
        'mean': np.mean,
        'min': np.min,
        'max': np.max
    }

    results = []

    for name, func in aggregations.items():
        y_train_pred = func(train_matrix, axis=1)
        y_test_pred = func(test_matrix, axis=1)
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        results.append({
            'aggregation': name,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'target': target_name
        })

    # append to results the individual model performances
    for i, model_name in enumerate(train_dict.keys()):
        y_train_pred = train_matrix[:, i]
        y_test_pred = test_matrix[:, i]
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        results.append({
            'aggregation': model_name,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'target': target_name
        })

    # Convert to DataFrame and save
    os.makedirs(save_dir, exist_ok=True)
    result_df = pd.DataFrame(results)
    output_path = os.path.join(save_dir, f"{target_name}_ensemble_auc.csv")
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved ensemble results to {output_path}")

    # Step 4: AutoGluon ML ensemble
    train_df_ag = pd.DataFrame(train_matrix, columns=list(train_dict.keys()))
    train_df_ag[target_name] = y_train

    predictor = TabularPredictor(
        label=target_name,
        eval_metric='roc_auc',
        path=os.path.join(save_dir, f"autogluon_{target_name}")
    )
    predictor.fit(train_df_ag, presets='medium_quality', verbosity=2)
    # fit_kwargs["tuning_data"]
    # fit_kwargs["refit_full"]
    # fit_kwargs["set_best_to_refit_full"]
    # **fit_kwargs

    # Predictions
    train_proba = predictor.predict_proba(train_df_ag)
    y_train_pred_ag = train_proba.iloc[:, 1].values  # Get second column (positive class)
    # y_train_pred_ag = predictor.predict_proba(train_df_ag)[:, 1]
    test_df_ag = pd.DataFrame(test_matrix, columns=list(test_dict.keys()))
    # y_test_pred_ag = predictor.predict_proba(test_df_ag)[:, 1]
    test_proba = predictor.predict_proba(test_df_ag)
    y_test_pred_ag = test_proba.iloc[:, 1].values

    meta_results_df = pd.DataFrame([{
        'aggregation': 'autogluon_meta',
        'train_auc': roc_auc_score(y_train, y_train_pred_ag),
        'test_auc': roc_auc_score(y_test, y_test_pred_ag),
        'target': target_name
    }])
    meta_results_df.to_csv(os.path.join(save_dir, f"{target_name}_autogluon_auc.csv"), index=False)

    # Step 5: Save feature importance
    fi_df = predictor.feature_importance(train_df_ag)
    fi_df.reset_index(inplace=True)
    fi_df.rename(columns={'index': 'feature', 'importance': 'importance_score'}, inplace=True)
    fi_df['target'] = target_name
    fi_path = os.path.join(save_dir, f"{target_name}_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    logger.info(f"Saved feature importance to {fi_path}")

    # ===== Step 6: Correlation analysis =====
    model_names = list(train_dict.keys())
    def compute_pairwise_corr(matrix, dataset_name):
        corr_results = []
        for m1, m2 in itertools.combinations(range(len(model_names)), 2):
            corr_val = np.corrcoef(matrix[:, m1], matrix[:, m2])[0, 1]
            corr_results.append({
                'model1': model_names[m1],
                'model2': model_names[m2],
                'correlation': corr_val,
                'dataset': dataset_name,
                'target': target_name
            })
        return corr_results

    corr_train = compute_pairwise_corr(train_matrix, 'train')
    corr_test = compute_pairwise_corr(test_matrix, 'test')

    corr_df = pd.DataFrame(corr_train + corr_test)
    corr_df.to_csv(os.path.join(save_dir, f"{target_name}_model_correlations.csv"), index=False)
    logger.info(f"Saved correlations")

def run_ensemble_aggregation(
    train_dict,
    test_dict,
    df_treat_path,
    train_start_date,
    train_end_date,
    test_start_date,
    test_end_date,
    target_name,
    save_dir,
):
    # Load treatment dataframe and filter relevant columns
    df = pd.read_csv(df_treat_path)
    df = df[["mrn", "treatment_date", target_name]]

    # Create train and test subsets
    train_set_df = df[(df["treatment_date"] >= train_start_date) & (df["treatment_date"] <= train_end_date)]
    test_set_df = df[(df["treatment_date"] >= test_start_date) & (df["treatment_date"] <= test_end_date)]

    # Convert dicts
    def build_prediction_paths_dict(input_dict, target_name):
        prediction_path_dict = {}
        for model_key, model_val in input_dict.items():
            if model_key == "prompting":
                # Find path: {model_val}/{target_name}/note_*/summary_*.csv
                target_dir = os.path.join(model_val, target_name)
                note_dirs = [d for d in glob.glob(os.path.join(target_dir, "note_*")) if os.path.isdir(d)]
                assert len(note_dirs) == 1, f"Expected one note_ directory in {target_dir}, found: {note_dirs}"
                summary_files = glob.glob(os.path.join(note_dirs[0], "summary_*.csv"))
                assert len(summary_files) == 1, f"Expected one summary_*.csv in {note_dirs[0]}, found: {summary_files}"
                prediction_path_dict[model_key] = summary_files[0]
            elif model_key == "finetune":
                assert isinstance(model_val, dict), f"For 'finetune', expected dict, got {type(model_val).__name__}"
                assert "results_dir" in model_val and "best_results_path" in model_val, \
                    f"'finetune' entry must have keys 'results_dir' and 'best_results_path'"
                
                results_dir = model_val["results_dir"]
                best_results_path = model_val["best_results_path"]
                target_dir = os.path.join(results_dir, target_name)
                
                # Load best results CSV
                df_best = pd.read_csv(best_results_path)
                target_row = df_best[df_best["target"] == target_name]
                assert not target_row.empty, f"Target {target_name} not found in {best_results_path}"
                
                lr_val = format_lr(target_row["lr"].iloc[0])
                epochs_val = target_row["epochs"].iloc[0]
                gradientsteps_val = target_row["gradientsteps"].iloc[0]
                
                # Template path with placeholder for dataset split
                template_path_to_csv = (
                    f"post_finetune_{target_name}_{{}}_predictions_*"
                    f"_lr-{lr_val}_epochs-{epochs_val}_*_gradientsteps-{gradientsteps_val}.csv"
                )
                
                prediction_path_dict[model_key] = os.path.join(target_dir, template_path_to_csv)
            else:
                df_model = pd.read_csv(model_val)
                target_row = df_model[df_model["target"] == target_name.replace("_", "-")]
                assert not target_row.empty, f"Target {target_name.replace('_', '-')} not found in {model_val}"
                prediction_path_dict[model_key] = target_row["pred_file_name"].values[0]
        return prediction_path_dict

    train_prediction_paths = build_prediction_paths_dict(train_dict, target_name)
    test_prediction_paths = build_prediction_paths_dict(test_dict, target_name)

    # Call the aggregation function
    ensemble_aggregation(
        train_dict=train_prediction_paths,
        test_dict=test_prediction_paths,
        train_set_df=train_set_df,
        test_set_df=test_set_df,
        target_name=target_name,
        save_dir=save_dir,
    )

# To do:
# send to Rob
# de-identify new notes
# set up inference pipeline for finetuning
# figure out prompting vllm parallelization with kevin
# fix tabular + nlp pipeline

def main():
    parser = argparse.ArgumentParser(description="Run ensemble aggregation on predictions.")
    parser.add_argument("--train_dict", type=str, required=True, help="JSON string or path to JSON file for training predictions.")
    parser.add_argument("--test_dict", type=str, required=True, help="JSON string or path to JSON file for test predictions.")
    parser.add_argument("--df_treat_path", type=str, required=True, help="Path to the treatment dataframe CSV.")
    parser.add_argument("--train_start_date", type=str, required=True, help="Start date for training set (YYYY-MM-DD).")
    parser.add_argument("--train_end_date", type=str, required=True, help="End date for training set (YYYY-MM-DD).")
    parser.add_argument("--test_start_date", type=str, required=True, help="Start date for test set (YYYY-MM-DD).")
    parser.add_argument("--test_end_date", type=str, required=True, help="End date for test set (YYYY-MM-DD).")
    parser.add_argument("--target_name", type=str, required=True, help="Target variable name.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save output.")

    args = parser.parse_args()

    # print target_name
    logger.info(f"Target name: {args.target_name}")

    def load_dict(arg_val: str):
        """Load dictionary from JSON string or JSON file path."""
        path = Path(arg_val)
        if path.is_file():
            with open(path, "r") as f:
                return json.load(f)
        return json.loads(arg_val)

    train_dict = load_dict(args.train_dict)
    test_dict = load_dict(args.test_dict)

    run_ensemble_aggregation(
        train_dict=train_dict,
        test_dict=test_dict,
        df_treat_path=args.df_treat_path,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        target_name=args.target_name,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()