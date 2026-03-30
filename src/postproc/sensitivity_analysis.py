import pandas as pd
import argparse
from oncotrail.prep.constants import CTCAE_constants
from oncotrail.postproc.aggregate_methods_targets_results import (bootstrap_aucs, 
    load_all_raw_predictions)   
from oncotrail.config import start_test_date
from oncotrail.constants import cancer_novelty_map
import numpy as np

def filter_mrns_by_grade2plus(df):
    """
    For each grade2plus target column, filter MRNs based on baseline lab value criteria
    derived from CTCAE_constants (grade2plus thresholds).

    Parameters:
        df: pd.DataFrame with columns mrn, lab values, and target_*_grade2plus columns
        CTCAE_constants: dict with lab thresholds

    Returns:
        dict: {target_col_name: [list of eligible MRNs]}
    """

    # Map each grade2plus target column to its:
    #   - source lab column in df
    #   - direction of filtering ('gte' = keep if value >= threshold, 'lte' = keep if value <= threshold)
    #   - threshold computation
    target_config = {
        'target_hemoglobin_grade2plus': {
            'lab_col': 'hemoglobin',
            'direction': 'gte',
            'threshold': CTCAE_constants['hemoglobin']['grade2plus']           # 100
        },
        'target_neutrophil_grade2plus': {
            'lab_col': 'neutrophil',
            'direction': 'gte',
            'threshold': CTCAE_constants['neutrophil']['grade2plus']            # 1.5
        },
        'target_platelet_grade2plus': {
            'lab_col': 'platelet',
            'direction': 'gte',
            'threshold': CTCAE_constants['platelet']['grade2plus']              # 75
        },
        'target_AKI_grade2plus': {
            'lab_col': 'creatinine',
            'direction': 'lte',
            'threshold': CTCAE_constants['AKI']['grade2plus'] * CTCAE_constants['AKI']['ULN']   # 1.5 * 353.68
        },
        'target_ALT_grade2plus': {
            'lab_col': 'alanine_aminotransferase',
            'direction': 'lte',
            'threshold': CTCAE_constants['ALT']['grade2plus'] * CTCAE_constants['ALT']['ULN']   # 3.0 * 40.0
        },
        'target_AST_grade2plus': {
            'lab_col': 'aspartate_aminotransferase',
            'direction': 'lte',
            'threshold': CTCAE_constants['AST']['grade2plus'] * CTCAE_constants['AST']['ULN']   # 3.0 * 34.0
        },
        'target_bilirubin_grade2plus': {
            'lab_col': 'total_bilirubin',
            'direction': 'lte',
            'threshold': CTCAE_constants['bilirubin']['grade2plus'] * CTCAE_constants['bilirubin']['ULN']  # 1.5 * 22.0
        },
    }

    result = {}

    for target_col, config in target_config.items():
        lab_col    = config['lab_col']
        direction  = config['direction']
        threshold  = config['threshold']

        # Step 1: keep only rows where the target is not -1
        subset = df[df[target_col] != -1]
        n_before = len(subset)

        # Step 2: apply the lab-value filter
        if direction == 'gte':
            filtered = subset[subset[lab_col] >= threshold]
        else:  # 'lte'
            filtered = subset[subset[lab_col] <= threshold]

        n_after   = len(filtered)
        n_dropped = n_before - n_after

        print(
            f"[{target_col}]  "
            f"eligible before lab filter: {n_before}  |  "
            f"dropped: {n_dropped}  |  "
            f"remaining: {n_after}  "
            f"(criterion: {lab_col} {'≥' if direction == 'gte' else '≤'} {threshold:.4f})"
        )

        result[target_col] = filtered['mrn'].tolist()

    return result

def run_grade2plus_bootstrap(
    all_predictions,        # output of load_all_raw_predictions
    test_filtered_mrns,     # output of filter_mrns_by_grade2plus on test set
    inf_filtered_mrns,      # output of filter_mrns_by_grade2plus on inference set
    mrn_inference_aero,     # from mrn_inference_aero_from_anchored_notes (full set is fine)
    n_boot=1000,
    base_seed=12345
):
    """
    Runs bootstrap_aucs restricted to:
      - grade2plus targets only
      - MRNs passing lab-value eligibility filters (per target, per split)
    """

    def _filter_split(pred_dict, mrn_key, pred_key, y_key, allowed_mrns):
        """Keep only rows whose MRN is in allowed_mrns."""
        allowed = set(allowed_mrns)
        mask = np.array([m in allowed for m in pred_dict[mrn_key]])
        return (
            pred_dict[mrn_key][mask],
            pred_dict[pred_key][mask],
            pred_dict[y_key][mask],
        )

    results_all = []

    # Group by target (second element of the key tuple)
    grade2plus_targets = sorted({
        target for (_, target) in all_predictions.keys()
        if "grade2plus" in target
    })

    for target in grade2plus_targets:

        # Collect all methods for this target
        pred_dict_by_method = {}

        for (method, t), preds in all_predictions.items():
            if t != target:
                continue

            filtered = dict(preds)  # shallow copy; we'll overwrite split arrays

            # --- filter test split ---
            allowed_test = test_filtered_mrns.get(
                f"{target}", []           # key format: "target_hemoglobin_grade2plus"
            )
            (filtered["mrn_test"],
             filtered["pred_test"],
             filtered["y_test"]) = _filter_split(
                preds, "mrn_test", "pred_test", "y_test", allowed_test
            )

            # --- filter inference split ---
            allowed_inf = inf_filtered_mrns.get(
                f"{target}", []
            )
            (filtered["mrn_inference"],
             filtered["pred_inference"],
             filtered["y_inference"]) = _filter_split(
                preds, "mrn_inference", "pred_inference", "y_inference", allowed_inf
            )

            # keep train untouched (we'll ignore it in the output anyway)
            pred_dict_by_method[method] = filtered

        if not pred_dict_by_method:
            continue

        print(f"\n=== {target} ===")
        for method, d in pred_dict_by_method.items():
            print(f"  {method}: test n={len(d['mrn_test'])}, inf n={len(d['mrn_inference'])}")

        df_boot = bootstrap_aucs(
            pred_dict_by_method,
            mrn_inference_aero=mrn_inference_aero,
            n_boot=n_boot,
            base_seed=base_seed
        )
        df_boot["target"] = target
        results_all.append(df_boot)

    df_final = pd.concat(results_all, ignore_index=True)

    # Drop train splits if you don't need them
    df_final = df_final[~df_final["split"].isin(["train", "test_minus_train", "inference_minus_train"])]

    return df_final

def perform_sensitivity_analysis(
    df_prompting_tt, df_prompting_inf,
    df_tabular_tt, df_tabular_inf,
    df_nlptfidf_tt, df_nlptfidf_inf,
    df_nlpcount_tt, df_nlpcount_inf,
    df_finetune_tt, df_finetune_inf,
    save_dir,
    tt_anchored_notes_path,
    inf_anchored_notes_path,
    n_boot=1000
):
    
    # restrict df_tt_anchored_notes on or after start_test_date
    df_tt_anchored_notes = pd.read_csv(tt_anchored_notes_path)
    df_inf_anchored_notes = pd.read_csv(inf_anchored_notes_path)
    df_inf_anchored_notes['cancer_novelty'] = df_inf_anchored_notes['primary_site_desc'].map(cancer_novelty_map)
    mrn_inference_aero = df_inf_anchored_notes[df_inf_anchored_notes['cancer_novelty'] == 0]['mrn'].unique().tolist()
    df_tt_anchored_notes = df_tt_anchored_notes[df_tt_anchored_notes['treatment_date'] >= start_test_date]
    test_filtered_mrns = filter_mrns_by_grade2plus(df_tt_anchored_notes)
    inf_filtered_mrns = filter_mrns_by_grade2plus(df_inf_anchored_notes)
    
    # call load_all_raw_predictions on all methods
    all_predictions = load_all_raw_predictions(
        df_prompting_tt, df_prompting_inf,
        df_tabular_tt, df_tabular_inf,
        df_nlptfidf_tt, df_nlptfidf_inf,
        df_nlpcount_tt, df_nlpcount_inf,
        df_finetune_tt, df_finetune_inf
    )

    df_tidy = run_grade2plus_bootstrap(
        all_predictions,        # output of load_all_raw_predictions
        test_filtered_mrns,     # output of filter_mrns_by_grade2plus on test set
        inf_filtered_mrns,      # output of filter_mrns_by_grade2plus on inference set
        mrn_inference_aero,     # from mrn_inference_aero_from_anchored_notes (full set is fine)
        n_boot=n_boot,
        base_seed=12345
    )

    df_tidy.to_csv(f"{save_dir}/sensitivity_analysis_grade2plus_bootstrap.csv", index=False)

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
    parser.add_argument("tt_anchored_notes_path")
    parser.add_argument("inf_anchored_notes_path")

    args = parser.parse_args()

    perform_sensitivity_analysis(
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
        args.tt_anchored_notes_path,
        args.inf_anchored_notes_path,
        n_boot=1000
    )