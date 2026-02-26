import pandas as pd
import numpy as np
import os
import argparse
from llm_notes_classification.postproc.aggregate_methods_targets_results import (
    load_all_raw_predictions, mrn_inference_aero_from_anchored_notes
)
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
# from rpy2.robjects import numpy2ri
# from rpy2.robjects.conversion import localconverter

def delong_test_r(y_true, y_pred1, y_pred2):
    """
    Compute the DeLong test for comparing two AUCs using R's pROC package.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred1 : array-like
        Predictions from model 1 (probabilities or scores)
    y_pred2 : array-like
        Predictions from model 2 (probabilities or scores)
    
    Returns:
    --------
    p_value : float
        P-value for the test
    auc1 : float
        AUC for model 1
    auc2 : float
        AUC for model 2
    """
    
    pROC = importr('pROC')

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    # Convert to R vectors
    r_y_true = ro.FloatVector(y_true)
    r_y_pred1 = ro.FloatVector(y_pred1)
    r_y_pred2 = ro.FloatVector(y_pred2)
    
    # Create ROC objects
    roc1 = pROC.roc(r_y_true, r_y_pred1,
                levels = [0, 1],   # or ["negative", "positive"] as appropriate
                direction = "<",
                quiet = True)

    roc2 = pROC.roc(r_y_true, r_y_pred2,
                    levels = [0, 1],
                    direction = "<",
                    quiet = True)
    
    # Perform DeLong test
    test_result = pROC.roc_test(roc1, roc2, method="delong")
    
    # Extract results
    p_value = test_result.rx2('p.value')[0]
    
    return p_value

def delong_test_r_unpaired(y_true1, y_pred1, y_true2, y_pred2):
    """
    Compute DeLong test for comparing two AUCs (unpaired samples)
    using R's pROC package.
    """
    
    pROC = importr('pROC')

    # Convert to numpy arrays
    y_true1 = np.array(y_true1)
    y_pred1 = np.array(y_pred1)
    y_true2 = np.array(y_true2)
    y_pred2 = np.array(y_pred2)
    
    # Convert to R vectors
    r_y_true1 = ro.FloatVector(y_true1)
    r_y_pred1 = ro.FloatVector(y_pred1)
    r_y_true2 = ro.FloatVector(y_true2)
    r_y_pred2 = ro.FloatVector(y_pred2)
    
    # Create ROC objects
    roc1 = pROC.roc(
        r_y_true1, r_y_pred1,
        levels=[0, 1],
        direction="<",
        quiet=True
    )

    roc2 = pROC.roc(
        r_y_true2, r_y_pred2,
        levels=[0, 1],
        direction="<",
        quiet=True
    )
    
    # IMPORTANT: paired=False
    test_result = pROC.roc_test(
        roc1,
        roc2,
        method="delong",
        paired=False
    )
    
    p_value = test_result.rx2('p.value')[0]
    
    return p_value

def compute_delong_comparisons_between_methods(preds, baseline_method):
    """
    Compare all methods against a baseline method using DeLong test.
    
    Parameters:
    -----------
    preds : dict
        Dictionary from load_all_raw_predictions with structure:
        {(method, target): {"mrn_test": ..., "pred_test": ..., "y_test": ...,
                           "mrn_inference": ..., "pred_inference": ..., "y_inference": ...}}
    baseline_method : str
        One of: "prompting", "tabular", "nlp-tfidf", "nlp-count", "finetune"
    
    Returns:
    --------
    df_heldout1 : pd.DataFrame
        Results for held-out set 1 (test) with columns: method, target, p_value
    df_heldout2 : pd.DataFrame
        Results for held-out set 2 (inference) with columns: method, target, p_value
    """
    
    all_methods = ["prompting", "tabular", "nlp-tfidf", "nlp-count", "finetune"]
    
    # Get all unique targets from the predictions dict
    all_targets = set([target for (method, target) in preds.keys()])
    
    results_heldout1 = []
    results_heldout2 = []
    
    for target in all_targets:
        print(f"Comparing methods for target: {target}")
        # Get baseline predictions for this target
        baseline_key = (baseline_method, target)
        
        if baseline_key not in preds:
            print(f"Warning: Baseline {baseline_method} not found for target {target}, skipping")
            continue
        
        baseline_data = preds[baseline_key]
        
        # Process each comparison method
        for method in all_methods:
            print(f"  Comparing {method} vs {baseline_method}")
            if method == baseline_method:
                continue  # Skip comparing baseline to itself
            
            method_key = (method, target)
            
            if method_key not in preds:
                print(f"Warning: Method {method} not found for target {target}, skipping")
                continue
            
            method_data = preds[method_key]
            
            # ============ HELD-OUT SET 1 (test) ============
            # Convert MRNs to integers
            baseline_mrns_test = np.array(baseline_data["mrn_test"]).astype(int)
            method_mrns_test = np.array(method_data["mrn_test"]).astype(int)
            
            # Find common MRNs
            common_mrns_test = np.intersect1d(baseline_mrns_test, method_mrns_test)
            
            if len(common_mrns_test) > 0:
                # Get indices for common MRNs
                baseline_idx_test = np.isin(baseline_mrns_test, common_mrns_test)
                method_idx_test = np.isin(method_mrns_test, common_mrns_test)
                
                # Sort by MRN to ensure alignment
                baseline_order = np.argsort(baseline_mrns_test[baseline_idx_test])
                method_order = np.argsort(method_mrns_test[method_idx_test])
                
                y_common_test = baseline_data["y_test"][baseline_idx_test][baseline_order]
                pred_baseline_test = baseline_data["pred_test"][baseline_idx_test][baseline_order]
                pred_method_test = method_data["pred_test"][method_idx_test][method_order]
                
                # Verify alignment
                baseline_mrns_sorted = baseline_mrns_test[baseline_idx_test][baseline_order]
                method_mrns_sorted = method_mrns_test[method_idx_test][method_order]
                assert np.array_equal(baseline_mrns_sorted, method_mrns_sorted), \
                    f"MRN mismatch for {method} vs {baseline_method} on target {target} (test)"
                
                # Compute DeLong test
                p_value_test = delong_test_r(
                    y_common_test, pred_baseline_test, pred_method_test
                )
                
                results_heldout1.append({
                    "method": method,
                    "target": target,
                    "p_value": p_value_test
                })
            else:
                print(f"Warning: No common MRNs for {method} vs {baseline_method} on target {target} (test)")
            
            # ============ HELD-OUT SET 2 (inference) ============
            # Convert MRNs to integers
            baseline_mrns_inf = np.array(baseline_data["mrn_inference"]).astype(int)
            method_mrns_inf = np.array(method_data["mrn_inference"]).astype(int)
            
            # Find common MRNs
            common_mrns_inf = np.intersect1d(baseline_mrns_inf, method_mrns_inf)
            
            if len(common_mrns_inf) > 0:
                # Get indices for common MRNs
                baseline_idx_inf = np.isin(baseline_mrns_inf, common_mrns_inf)
                method_idx_inf = np.isin(method_mrns_inf, common_mrns_inf)
                
                # Sort by MRN to ensure alignment
                baseline_order = np.argsort(baseline_mrns_inf[baseline_idx_inf])
                method_order = np.argsort(method_mrns_inf[method_idx_inf])
                
                y_common_inf = baseline_data["y_inference"][baseline_idx_inf][baseline_order]
                pred_baseline_inf = baseline_data["pred_inference"][baseline_idx_inf][baseline_order]
                pred_method_inf = method_data["pred_inference"][method_idx_inf][method_order]
                
                # Verify alignment
                baseline_mrns_sorted = baseline_mrns_inf[baseline_idx_inf][baseline_order]
                method_mrns_sorted = method_mrns_inf[method_idx_inf][method_order]
                assert np.array_equal(baseline_mrns_sorted, method_mrns_sorted), \
                    f"MRN mismatch for {method} vs {baseline_method} on target {target} (inference)"
                
                # Compute DeLong test
                p_value_inf = delong_test_r(
                    y_common_inf, pred_baseline_inf, pred_method_inf
                )
                
                results_heldout2.append({
                    "method": method,
                    "target": target,
                    "p_value": p_value_inf
                })
            else:
                print(f"Warning: No common MRNs for {method} vs {baseline_method} on target {target} (inference)")
    
    df_heldout1 = pd.DataFrame(results_heldout1)
    df_heldout2 = pd.DataFrame(results_heldout2)
    
    return df_heldout1, df_heldout2

def compute_delong_comparison_across_shift(preds, mrn_inference_aero):
    """
    Compare ROCs across data shifts using unpaired DeLong test.
    
    Parameters:
    -----------
    preds : dict
        Dictionary from load_all_raw_predictions with structure:
        {(method, target): {"mrn_test": ..., "pred_test": ..., "y_test": ...,
                           "mrn_inference": ..., "pred_inference": ..., "y_inference": ...}}
    mrn_inference_aero : array-like
        List of MRNs for patients with aerodigestive cancers
    
    Returns:
    --------
    df_test_vs_inference : pd.DataFrame
        Results comparing test vs inference sets with columns: method, target, p_value
    df_aero_vs_nonaero : pd.DataFrame
        Results comparing aerodigestive vs non-aerodigestive within inference set
        with columns: method, target, p_value
    """
    
    all_methods = ["prompting", "tabular", "nlp-tfidf", "nlp-count", "finetune"]
    
    # Convert mrn_inference_aero to integer array for consistent comparison
    mrn_inference_aero = np.array(mrn_inference_aero).astype(int)
    
    # Get all unique targets from the predictions dict
    all_targets = set([target for (method, target) in preds.keys()])
    
    results_test_vs_inference = []
    results_aero_vs_nonaero = []
    
    for method in all_methods:
        for target in all_targets:
            method_key = (method, target)
            
            if method_key not in preds:
                print(f"Warning: Method {method} not found for target {target}, skipping")
                continue
            
            method_data = preds[method_key]
            
            # ============ TASK 1: Compare test vs inference ============
            # Convert MRNs to integers
            mrn_test = np.array(method_data["mrn_test"]).astype(int)
            mrn_inference = np.array(method_data["mrn_inference"]).astype(int)
            
            y_test = method_data["y_test"]
            pred_test = method_data["pred_test"]
            y_inference = method_data["y_inference"]
            pred_inference = method_data["pred_inference"]
            
            # Compute unpaired DeLong test between test and inference
            try:
                p_value_test_vs_inf = delong_test_r_unpaired(
                    y_test, pred_test,
                    y_inference, pred_inference
                )
                
                results_test_vs_inference.append({
                    "method": method,
                    "target": target,
                    "p_value": p_value_test_vs_inf
                })
            except Exception as e:
                print(f"Warning: DeLong test failed for {method} - {target} (test vs inference): {e}")
            
            # ============ TASK 2: Compare aerodigestive vs non-aerodigestive within inference ============
            # Partition inference set into aero and non-aero
            aero_mask = np.isin(mrn_inference, mrn_inference_aero)
            nonaero_mask = ~aero_mask
            
            # Extract aerodigestive subset
            mrn_aero = mrn_inference[aero_mask]
            y_aero = y_inference[aero_mask]
            pred_aero = pred_inference[aero_mask]
            
            # Extract non-aerodigestive subset
            mrn_nonaero = mrn_inference[nonaero_mask]
            y_nonaero = y_inference[nonaero_mask]
            pred_nonaero = pred_inference[nonaero_mask]
            
            # Check if both subsets have data
            if len(mrn_aero) == 0:
                print(f"Warning: No aerodigestive patients for {method} - {target}, skipping")
                continue
            
            if len(mrn_nonaero) == 0:
                print(f"Warning: No non-aerodigestive patients for {method} - {target}, skipping")
                continue
            
            # Compute unpaired DeLong test between aero and non-aero
            try:
                p_value_aero_vs_nonaero = delong_test_r_unpaired(
                    y_aero, pred_aero,
                    y_nonaero, pred_nonaero
                )
                
                results_aero_vs_nonaero.append({
                    "method": method,
                    "target": target,
                    "p_value": p_value_aero_vs_nonaero
                })
            except Exception as e:
                print(f"Warning: DeLong test failed for {method} - {target} (aero vs non-aero): {e}")
    
    df_test_vs_inference = pd.DataFrame(results_test_vs_inference)
    df_aero_vs_nonaero = pd.DataFrame(results_aero_vs_nonaero)
    
    return df_test_vs_inference, df_aero_vs_nonaero

def delong_auc_comparison(
    df_prompting_tt, df_prompting_inf,
    df_tabular_tt, df_tabular_inf,
    df_nlptfidf_tt, df_nlptfidf_inf,
    df_nlpcount_tt, df_nlpcount_inf,
    df_finetune_tt, df_finetune_inf,
    save_dir,
    inf_anchored_notes_path,
    baseline_method="tabular"
):
    
    # paired delong test

    preds = load_all_raw_predictions(
        df_prompting_tt, df_prompting_inf,
        df_tabular_tt, df_tabular_inf,
        df_nlptfidf_tt, df_nlptfidf_inf,
        df_nlpcount_tt, df_nlpcount_inf,
        df_finetune_tt, df_finetune_inf
    )

    df_test, df_inf = compute_delong_comparisons_between_methods(preds, baseline_method)
    df_test.to_csv(
        os.path.join(save_dir, "delong_comparison_test.csv"),
        index=False
    )
    df_inf.to_csv(
        os.path.join(save_dir, "delong_comparison_inference.csv"),
        index=False
    )

    mrn_inference_aero = mrn_inference_aero_from_anchored_notes(inf_anchored_notes_path)
    # unpaired delong test for shifts
    (
        df_test_vs_inference,
        df_aero_vs_nonaero,
    ) = compute_delong_comparison_across_shift(
        preds,
        mrn_inference_aero,
    )
    df_test_vs_inference.to_csv(
        os.path.join(save_dir, "delong_comparison_test_vs_inference.csv"),
        index=False
    )
    df_aero_vs_nonaero.to_csv(
        os.path.join(save_dir, "delong_comparison_aero_vs_nonaero.csv"),
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
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="tabular",
        choices=[
            "prompting",
            "tabular",
            "nlp-tfidf",
            "nlp-count",
            "finetune",
        ],
    )
    args = parser.parse_args()

    delong_auc_comparison(
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
        args.baseline_method
    )