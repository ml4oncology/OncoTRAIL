import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import argparse

# --------------------------------------------------------
# Helpers to load predictions AND labels for each method
# --------------------------------------------------------

def load_npz_predictions_with_labels(path):
    """
    Load .npz containing:
        mrn_train, mrn_test,
        train_pred, test_pred,
        Y_train, Y_test
    """
    data = np.load(path, allow_pickle=True)

    return {
        "mrn_train": data["mrn_train"].ravel(),
        "pred_train": data["train_pred"].ravel(),
        "y_train":   data["Y_train"].ravel(),

        "mrn_test":  data["mrn_test"].ravel(),
        "pred_test": data["test_pred"].ravel(),
        "y_test":    data["Y_test"].ravel(),
    }


def load_finetune_csv_with_labels(path):
    """
    Finetune CSV contains:
        prob (as "[p0 p1]"), mrn, label
    """
    df = pd.read_csv(path)

    # Convert "[p0 p1]" to numpy array
    probs_np = np.stack(df["prob"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).values)
    pos_prob = probs_np[:, 1]

    return {
        "mrn": df["mrn"].values,
        "pred": pos_prob,
        "y": df["label"].astype(int).values
    }


def load_prompting_csv_with_labels(path, target_name):
    """
    Prompting CSV:
        mrn, Probability, <target_name>  (label column)
    """
    df = pd.read_csv(path)

    return {
        "mrn": df["mrn"].values,
        "pred": df["Probability"].values,
        "y": df[target_name.replace("-", "_")].astype(int).values
    }


# --------------------------------------------------------
# Build a unified dictionary of raw predictions + labels
# --------------------------------------------------------

def load_all_raw_predictions(
    df_prompting,
    df_tabular,
    df_nlptfidf,
    df_nlpcount,
    df_finetune
):
    """
    Return:
        predictions[(method, target)] = dict(
            mrn_train, pred_train, y_train,
            mrn_test,  pred_test,  y_test
        )
    """
    predictions = {}

    # ---- prompting ----
    for _, row in df_prompting.iterrows():
        target = row["target"]

        # Train CSV
        tr = load_prompting_csv_with_labels(row["path_to_predictions_train"], target)
        # Test CSV
        te = load_prompting_csv_with_labels(row["path_to_predictions_test"], target)

        predictions[("prompting", target)] = {
            "mrn_train": tr["mrn"], "pred_train": tr["pred"], "y_train": tr["y"],
            "mrn_test":  te["mrn"], "pred_test": te["pred"], "y_test": te["y"],
        }

    # ---- tabular / nlp-tfidf / nlp-count ----
    method_dfs = [
        ("tabular", df_tabular),
        ("nlp-tfidf", df_nlptfidf),
        ("nlp-count", df_nlpcount),
    ]
    for method, df_m in method_dfs:
        for _, row in df_m.iterrows():
            target = row["target"]
            d = load_npz_predictions_with_labels(row["pred_file_name"])

            predictions[(method, target)] = {
                "mrn_train": d["mrn_train"], "pred_train": d["pred_train"], "y_train": d["y_train"],
                "mrn_test":  d["mrn_test"],  "pred_test": d["pred_test"],  "y_test":  d["y_test"],
            }

    # ---- finetune ----
    for _, row in df_finetune.iterrows():
        target = row["target"]

        tr = load_finetune_csv_with_labels(row["path_to_predictions_train"])
        te = load_finetune_csv_with_labels(row["path_to_predictions_test"])

        predictions[("finetune", target)] = {
            "mrn_train": tr["mrn"], "pred_train": tr["pred"], "y_train": tr["y"],
            "mrn_test":  te["mrn"], "pred_test": te["pred"], "y_test": te["y"],
        }

    return predictions


# --------------------------------------------------------
# Bootstrap
# --------------------------------------------------------

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def bootstrap_aucs(pred_dict_by_method, n_boot=1000, base_seed=12345):
    """
    pred_dict_by_method:
        method → {
            mrn_train, pred_train, y_train,
            mrn_test,  pred_test,  y_test
        }
    Ensures the SAME test bootstrap indices are used across all methods.
    Uses deterministic seeds for full reproducibility.
    """

    results = []

    # ----------------------------------------------------
    # Compute shared test MRNs across all methods
    # ----------------------------------------------------
    all_test_sets = [
        set(pred_dict_by_method[m]["mrn_test"])
        for m in pred_dict_by_method
    ]
    shared_test_mrns = sorted(set.intersection(*all_test_sets))
    shared_test_mrns = np.array(shared_test_mrns)

    # Build a unified test label vector (MRN → label)
    first_method = next(iter(pred_dict_by_method.keys()))
    ref = pred_dict_by_method[first_method]
    y_test_map = dict(zip(ref["mrn_test"], ref["y_test"]))
    y_test_aligned = np.array([y_test_map[mrn] for mrn in shared_test_mrns])

    # Build aligned test prediction vectors per method
    aligned_test_preds = {}
    for method, info in pred_dict_by_method.items():
        pred_map = dict(zip(info["mrn_test"], info["pred_test"]))
        aligned_test_preds[method] = np.array(
            [pred_map[mrn] for mrn in shared_test_mrns]
        )

    N_test = len(shared_test_mrns)

    # ----------------------------------------------------
    # BOOTSTRAP — outer loop over iterations
    # ----------------------------------------------------
    for b in range(n_boot):

        # --- Deterministic RNG for this iteration ---
        rng = np.random.default_rng(base_seed + b)

        # ------- SHARED TEST INDICES FOR THIS ITERATION -------
        idx_test = rng.choice(N_test, N_test, replace=True)

        # Each method gets its own train bootstrap,
        # but all share the SAME idx_test.
        for method, info in pred_dict_by_method.items():

            # Train bootstrap (method-specific, but deterministic)
            n_train = len(info["mrn_train"])
            idx_train = rng.choice(n_train, n_train, replace=True)

            auc_train = safe_auc(info["y_train"][idx_train],
                                 info["pred_train"][idx_train])

            # Test bootstrap (shared across methods)
            auc_test = safe_auc(y_test_aligned[idx_test],
                                aligned_test_preds[method][idx_test])

            results.append({
                "method": method,
                "split": "train",
                "boot_id": b,
                "auc": auc_train,
            })
            results.append({
                "method": method,
                "split": "test",
                "boot_id": b,
                "auc": auc_test,
            })
            results.append({
                "method": method,
                "split": "test_minus_train",
                "boot_id": b,
                "auc": auc_test - auc_train,
            })

    return pd.DataFrame(results)


# --------------------------------------------------------
# High-level wrapper for Option A tidy format
# --------------------------------------------------------

def build_tidy_bootstrap_dataframe(
    df_prompting,
    df_tabular,
    df_nlptfidf,
    df_nlpcount,
    df_finetune,
    save_dir,
    n_boot=1000
):
    predictions_all = load_all_raw_predictions(
        df_prompting,
        df_tabular,
        df_nlptfidf,
        df_nlpcount,
        df_finetune
    )

    all_rows = []
    targets = sorted({t for (_, t) in predictions_all.keys()})

    for target in tqdm(targets):
        print(f"Processing target: {target}")
        pred_dict = {
            method: predictions_all[(method, target)]
            for (method, t) in predictions_all
            if t == target
        }

        df_boot = bootstrap_aucs(pred_dict, n_boot=n_boot)
        df_boot["target"] = target
        all_rows.append(df_boot)

    pd.concat(all_rows, ignore_index=True).to_csv(
        os.path.join(save_dir, "aggregate_bootstrap_results.csv"), index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompting_path", type=str, help="Path to prompting predictions CSV")
    parser.add_argument("tabular_path", type=str, help="Path to tabular predictions NPZ")
    parser.add_argument("nlptfidf_path", type=str, help="Path to nlp-tfidf predictions NPZ")
    parser.add_argument("nlpcount_path", type=str, help="Path to nlp-count predictions NPZ")
    parser.add_argument("finetune_path", type=str, help="Path to finetune predictions CSV")
    parser.add_argument("save_dir", type=str, help="Directory to save the aggregated results")
    args = parser.parse_args()

    # Load dataframes
    df_prompting = pd.read_csv(args.prompting_path)
    df_tabular = pd.read_csv(args.tabular_path)
    df_nlptfidf = pd.read_csv(args.nlptfidf_path)
    df_nlpcount = pd.read_csv(args.nlpcount_path)
    df_finetune = pd.read_csv(args.finetune_path)
    df_finetune['target'] = df_finetune['target'].str.replace('_', '-')

    build_tidy_bootstrap_dataframe(
        df_prompting,
        df_tabular,
        df_nlptfidf,
        df_nlpcount,
        df_finetune,
        args.save_dir,
        n_boot=1000
    )