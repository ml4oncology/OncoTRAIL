import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import argparse
from oncotrail.constants import cancer_novelty_map

# --------------------------------------------------------
# Helpers to load predictions AND labels
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


def load_npz_inference_predictions_with_labels(path):
    """
    Load inference .npz containing:
        mrn_test, test_pred, Y_test
    """
    data = np.load(path, allow_pickle=True)

    return {
        "mrn_inference":  data["mrn_test"].ravel(),
        "pred_inference": data["test_pred"].ravel(),
        "y_inference":    data["Y_test"].ravel(),
    }


def load_finetune_csv_with_labels(path):
    df = pd.read_csv(path)
    probs_np = np.stack(df["prob"].apply(lambda x: np.fromstring(x.strip("[]"), sep=' ')).values)
    pos_prob = probs_np[:, 1]

    return {
        "mrn": df["mrn"].values,
        "pred": pos_prob,
        "y": df["label"].astype(int).values
    }


def load_prompting_csv_with_labels(path, target_name):
    df = pd.read_csv(path)
    return {
        "mrn": df["mrn"].values,
        "pred": df["Probability"].values,
        "y": df[target_name.replace("-", "_")].astype(int).values
    }


# --------------------------------------------------------
# Load all raw predictions
# --------------------------------------------------------

def load_all_raw_predictions(
    df_prompting_tt, df_prompting_inf,
    df_tabular_tt, df_tabular_inf,
    df_nlptfidf_tt, df_nlptfidf_inf,
    df_nlpcount_tt, df_nlpcount_inf,
    df_finetune_tt, df_finetune_inf
):

    predictions = {}

    # ---- prompting ----
    for _, row in df_prompting_tt.iterrows():
        target = row["target"]

        tr = load_prompting_csv_with_labels(row["path_to_predictions_train"], target)
        te = load_prompting_csv_with_labels(row["path_to_predictions_test"], target)

        inf_row = df_prompting_inf[df_prompting_inf["target"] == target].iloc[0]
        inf = load_prompting_csv_with_labels(
            inf_row["path_to_predictions_inference"], target
        )

        predictions[("prompting", str(target).replace("-", "_"))] = {
            "mrn_train": tr["mrn"], "pred_train": tr["pred"], "y_train": tr["y"],
            "mrn_test":  te["mrn"], "pred_test":  te["pred"], "y_test":  te["y"],
            "mrn_inference":  inf["mrn"], "pred_inference": inf["pred"], "y_inference": inf["y"],
        }

    # ---- tabular / nlp ----
    method_dfs = [
        ("tabular", df_tabular_tt, df_tabular_inf),
        ("nlp-tfidf", df_nlptfidf_tt, df_nlptfidf_inf),
        ("nlp-count", df_nlpcount_tt, df_nlpcount_inf),
    ]

    for method, df_tt, df_inf in method_dfs:
        for _, row in df_tt.iterrows():
            target = row["target"]

            d_tt = load_npz_predictions_with_labels(row["pred_file_name"])
            inf_row = df_inf[df_inf["target"] == target].iloc[0]
            d_inf = load_npz_inference_predictions_with_labels(inf_row["pred_file_name"])

            predictions[(method, str(target).replace("-", "_"))] = {**d_tt, **d_inf}

    # ---- finetune ----
    df_finetune_tt['target'] = df_finetune_tt['target'].astype(str).str.replace('_','-')
    df_finetune_inf['target'] = df_finetune_inf['target'].astype(str).str.replace('_','-')
    
    for _, row in df_finetune_tt.iterrows():
        target = row["target"]

        tr = load_finetune_csv_with_labels(row["path_to_predictions_train"])
        te = load_finetune_csv_with_labels(row["path_to_predictions_test"])

        inf_row = df_finetune_inf[df_finetune_inf["target"] == target].iloc[0]
        inf = load_finetune_csv_with_labels(
            inf_row["path_to_predictions_test"]
        ) # just the name, this is really inference

        predictions[("finetune", str(target).replace("-", "_"))] = {
            "mrn_train": tr["mrn"], "pred_train": tr["pred"], "y_train": tr["y"],
            "mrn_test":  te["mrn"], "pred_test":  te["pred"], "y_test":  te["y"],
            "mrn_inference":  inf["mrn"], "pred_inference": inf["pred"], "y_inference": inf["y"],
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


# def bootstrap_aucs(pred_dict_by_method, n_boot=1000, base_seed=12345):
def bootstrap_aucs(
    pred_dict_by_method,
    mrn_inference_aero,
    n_boot=1000,
    base_seed=12345
):
    results = []

    # ---- shared TEST ----
    shared_test_mrns = sorted(set.intersection(*[
        set(v["mrn_test"]) for v in pred_dict_by_method.values()
    ]))
    shared_test_mrns = np.array(shared_test_mrns)

    ref = next(iter(pred_dict_by_method.values()))
    y_test_map = dict(zip(ref["mrn_test"], ref["y_test"]))
    y_test = np.array([y_test_map[m] for m in shared_test_mrns])

    test_preds = {
        m: np.array([dict(zip(v["mrn_test"], v["pred_test"]))[mrn]
                     for mrn in shared_test_mrns])
        for m, v in pred_dict_by_method.items()
    }

    # ---- shared INFERENCE ----
    shared_inf_mrns = sorted(set.intersection(*[
        set(v["mrn_inference"]) for v in pred_dict_by_method.values()
    ]))
    shared_inf_mrns = np.array(shared_inf_mrns)

    y_inf_map = dict(zip(ref["mrn_inference"], ref["y_inference"]))
    y_inf = np.array([y_inf_map[m] for m in shared_inf_mrns])

    inf_preds = {
        m: np.array([dict(zip(v["mrn_inference"], v["pred_inference"]))[mrn]
                     for mrn in shared_inf_mrns])
        for m, v in pred_dict_by_method.items()
    }

    # ---- partition inference into aero / other ----
    mrn_inference_aero = set(mrn_inference_aero)

    aero_mask = np.array([mrn in mrn_inference_aero for mrn in shared_inf_mrns])
    other_mask = ~aero_mask

    y_inf_aero = y_inf[aero_mask]
    y_inf_other = y_inf[other_mask]

    inf_preds_aero = {
        m: p[aero_mask] for m, p in inf_preds.items()
    }
    inf_preds_other = {
        m: p[other_mask] for m, p in inf_preds.items()
    }


    for b in range(n_boot):
        rng = np.random.default_rng(base_seed + b)

        idx_test = rng.choice(len(y_test), len(y_test), replace=True)
        idx_inf = rng.choice(len(y_inf), len(y_inf), replace=True)

        if len(y_inf_aero) > 0:
            idx_aero = rng.choice(len(y_inf_aero), len(y_inf_aero), replace=True)
        if len(y_inf_other) > 0:
            idx_other = rng.choice(len(y_inf_other), len(y_inf_other), replace=True)

        for method, info in pred_dict_by_method.items():

            idx_train = rng.choice(len(info["y_train"]), len(info["y_train"]), replace=True)

            auc_train = safe_auc(info["y_train"][idx_train],
                                 info["pred_train"][idx_train])

            auc_test = safe_auc(y_test[idx_test],
                                test_preds[method][idx_test])

            auc_inf = safe_auc(y_inf[idx_inf],
                               inf_preds[method][idx_inf])

            # bootstrap aero / other separately
            if len(y_inf_aero) > 0:
                auc_inf_aero = safe_auc(
                    y_inf_aero[idx_aero],
                    inf_preds_aero[method][idx_aero]
                )
            else:
                auc_inf_aero = np.nan

            if len(y_inf_other) > 0:
                auc_inf_other = safe_auc(
                    y_inf_other[idx_other],
                    inf_preds_other[method][idx_other]
                )
            else:
                auc_inf_other = np.nan


            results.extend([
                {"method": method, "split": "train", "boot_id": b, "auc": auc_train},
                {"method": method, "split": "test", "boot_id": b, "auc": auc_test},
                {"method": method, "split": "inference", "boot_id": b, "auc": auc_inf},
                {"method": method, "split": "test_minus_train", "boot_id": b, "auc": auc_test - auc_train},
                {"method": method, "split": "inference_minus_train", "boot_id": b, "auc": auc_inf - auc_train},
                {"method": method, "split": "inference_minus_test", "boot_id": b, "auc": auc_inf - auc_test},

                {"method": method, "split": "inference_aero", "boot_id": b, "auc": auc_inf_aero},
                {"method": method, "split": "inference_other", "boot_id": b, "auc": auc_inf_other},
                {"method": method,
                    "split": "inference_aero_minus_inference_other",
                    "boot_id": b,
                    "auc": auc_inf_aero - auc_inf_other}
            ])

    return pd.DataFrame(results)

# --------------------------------------------------------
# Find aerodigestive mrns
# --------------------------------------------------------

def mrn_inference_aero_from_anchored_notes(path_to_anchored_notes):
    df_anchored_notes = pd.read_csv(path_to_anchored_notes)
    df_anchored_notes['cancer_novelty'] = df_anchored_notes['primary_site_desc'].map(cancer_novelty_map)

    mrn_inference_aero = df_anchored_notes[df_anchored_notes['cancer_novelty'] == 0]['mrn'].unique().tolist()

    return mrn_inference_aero


# --------------------------------------------------------
# High-level wrapper
# --------------------------------------------------------

def build_tidy_bootstrap_dataframe(
    df_prompting_tt, df_prompting_inf,
    df_tabular_tt, df_tabular_inf,
    df_nlptfidf_tt, df_nlptfidf_inf,
    df_nlpcount_tt, df_nlpcount_inf,
    df_finetune_tt, df_finetune_inf,
    save_dir,
    inf_anchored_notes_path,
    n_boot=1000
):
    
    mrn_inference_aero = mrn_inference_aero_from_anchored_notes(inf_anchored_notes_path)

    preds = load_all_raw_predictions(
        df_prompting_tt, df_prompting_inf,
        df_tabular_tt, df_tabular_inf,
        df_nlptfidf_tt, df_nlptfidf_inf,
        df_nlpcount_tt, df_nlpcount_inf,
        df_finetune_tt, df_finetune_inf
    )

    all_rows = []
    targets = sorted({t for (_, t) in preds.keys()})

    for target in tqdm(targets):
        pred_dict = {
            method: preds[(method, target)]
            for (method, t) in preds if t == target
        }

        df_boot = bootstrap_aucs(pred_dict, mrn_inference_aero, n_boot=n_boot)
        df_boot["target"] = target
        all_rows.append(df_boot)

    df_tidy = pd.concat(all_rows, ignore_index=True)
    df_tidy['target'] = df_tidy['target'].astype(str).str.replace('_','-')
    df_tidy.to_csv(
        os.path.join(save_dir, "aggregate_bootstrap_results.csv"),
        index=False
    )


# --------------------------------------------------------
# CLI
# --------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("prompting_tt")
    parser.add_argument("prompting_inf")
    parser.add_argument("tabular_tt")
    parser.add_argument("tabular_inf")
    parser.add_argument("nlptfidf_tt")
    parser.add_argument("nlptfidf_inf")
    parser.add_argument("nlpcount_tt")
    parser.add_argument("nlpcount_inf")
    parser.add_argument("finetune_tt")
    parser.add_argument("finetune_inf")
    parser.add_argument("save_dir")
    parser.add_argument("inf_anchored_notes_path")

    args = parser.parse_args()

    build_tidy_bootstrap_dataframe(
        pd.read_csv(args.prompting_tt),
        pd.read_csv(args.prompting_inf),
        pd.read_csv(args.tabular_tt),
        pd.read_csv(args.tabular_inf),
        pd.read_csv(args.nlptfidf_tt),
        pd.read_csv(args.nlptfidf_inf),
        pd.read_csv(args.nlpcount_tt),
        pd.read_csv(args.nlpcount_inf),
        pd.read_csv(args.finetune_tt),
        pd.read_csv(args.finetune_inf),
        args.save_dir,
        args.inf_anchored_notes_path,
        n_boot=1000
    )
