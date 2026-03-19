import argparse
import pandas as pd
import ast
import sys
sys.path.insert(1, "/cluster/projects/gliugroup/2BLAST/data/info")
from phys_names import aliasDictionary, fellow_alias
from collections import Counter
import numpy as np
import os
import glob
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import warnings
import pickle
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import bambi as bmb
import arviz as az
from oncotrail.constants import df_physician_char_EPR, df_physician_char_EPIC
from oncotrail.constants import CANCER_COARSE_SITE_MAP as COARSE_SITE_MAP
from oncotrail.constants import CANCER_HIERARCHY as HIERARCHY

# Create a dictionary where the key is the coarse group and the value is its rank (lower rank means higher priority)
HIERARCHY_RANK = {site: rank for rank, site in enumerate(HIERARCHY)}
OTHER_ILL_DEFINED_GROUP = "Other / Ill-Defined"

def process_cancer_sites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts multiple binary cancer site columns into a single 'cancer_type' column 
    based on coarse grouping and clinical hierarchy.

    Args:
        df: The input DataFrame containing 'cancer_site_CXX' columns, 'prompting_risk', 
            'tabular_risk', and 'dictating_physician'.

    Returns:
        The DataFrame with the new 'cancer_type' column.
    """
    
    # --- Step 1: Identify all cancer site columns ---
    # Assuming all columns starting with 'C' followed by digits are cancer site indicators
    # We must exclude the 'C' code columns that are not in our mapping if they exist
    all_c_cols = [col for col in df.columns if col.startswith('C') and col[1:].isdigit()]
    
    # Filter the columns to only include those defined in our mapping
    defined_c_codes = set(COARSE_SITE_MAP.keys())
    cancer_site_cols = [col for col in all_c_cols if col in defined_c_codes]
    
    if not cancer_site_cols:
        print("Warning: No defined 'C' code columns found. Assigning 'Other / Ill-Defined' to all.")
        df['cancer_type'] = OTHER_ILL_DEFINED_GROUP
        return df

    # --- Step 2: Determine the Primary Cancer Type for each patient (row) ---
    def get_primary_cancer_type(row):
        # Identify all C-codes that are positive (value == 1) for the current patient
        positive_c_codes = [col for col in cancer_site_cols if row.get(col) == 2]

        # if positive_c_codes is empty, try to find codes with value == 1
        if not positive_c_codes:
            positive_c_codes = [col for col in cancer_site_cols if row.get(col) == 1]
        
        if not positive_c_codes:
            # --- FIX: Merge "No Primary Found" cases into "Other / Ill-Defined" ---
            return OTHER_ILL_DEFINED_GROUP

        # Map positive C-codes to their coarse groups
        coarse_groups = {COARSE_SITE_MAP.get(code, OTHER_ILL_DEFINED_GROUP) for code in positive_c_codes}
        
        # --- Step 3: Apply Hierarchy to select the single Primary Site ---
        
        # Define a function to look up the rank
        def get_rank(group_name):
            # Use the defined rank, or assign a high rank (low priority) if the group is unexpected
            # We use len(HIERARCHY) as a very low priority rank for safety
            return HIERARCHY_RANK.get(group_name, len(HIERARCHY))

        # Sort the groups by their rank. The group with the lowest rank (highest priority) is the first element.
        primary_site = sorted(list(coarse_groups), key=get_rank)[0]
        
        return primary_site

    # Apply the function across all rows to create the new 'cancer_type' column
    df['cancer_type'] = df.apply(get_primary_cancer_type, axis=1)
    
    print(f"Data processed. Created 'cancer_type' column with {df['cancer_type'].nunique()} unique groups.")
    return df

def extract_names(s):
    try:
        # Extract the line that looks like a list of names
        list_line = [line for line in s.split('\n') if line.startswith('[')][0]
        names = ast.literal_eval(list_line)

        # Apply alias mapping to each name
        result = []
        for name in names:
            alias = aliasDictionary.get(name, name)
            result.append(fellow_alias.get(alias, alias))
        return result

    except Exception:
        return []
    
def load_prompting_result(target_name, prompting_results_dir):
    target_dir = os.path.join(prompting_results_dir, target_name)
    note_dirs = [d for d in glob.glob(os.path.join(target_dir, "note_*")) if os.path.isdir(d)]
    assert len(note_dirs) == 1, f"Expected one note_ directory in {target_dir}, found: {note_dirs}"
    summary_files = glob.glob(os.path.join(note_dirs[0], "summary_*.csv"))
    assert len(summary_files) == 1, f"Expected one summary_*.csv in {note_dirs[0]}, found: {summary_files}"
    prompting_summary_df = pd.read_csv(summary_files[0])
    
    return prompting_summary_df

def load_tabular_result(target_name, tabular_results_path, held_out_set):
    df_model = pd.read_csv(tabular_results_path)
    target_row = df_model[df_model["target"] == target_name.replace("_", "-")]
    assert not target_row.empty, f"Target {target_name.replace('_', '-')} not found in dataframe"
    data = np.load(target_row["pred_file_name"].values[0], allow_pickle=True)
    if held_out_set == "test":
        mrn_var = 'mrn_test'
        pred_var = 'test_pred'
    elif held_out_set == "inference":
        mrn_var = 'mrn_inference'
        pred_var = 'inference_pred'
    mrns = data[mrn_var]
    probs = data[pred_var].ravel()

    return pd.DataFrame({'mrn': mrns, 'prob': probs})

def regress_physician_characteristics(target_list, held_out_set, anchored_notes_path, 
                                      save_dir, prompting_results_dir, tabular_results_path,
                                      raw_treatment_path):

    df_treatment = pd.read_csv(anchored_notes_path)
    df_treatment['treatment_date'] = pd.to_datetime(df_treatment['treatment_date'], utc=True)
    df_treatment['treatment_year'] = df_treatment['treatment_date'].dt.year
    df_treatment = df_treatment[['mrn', 'stats_dictated_by', 'treatment_date', 'treatment_year']].copy()   

    raw_treatment = pd.read_parquet(raw_treatment_path)
    if held_out_set == "test":
        # find all columns in raw_treatment that contain 'cancer_site'
        cancer_site_cols = [col for col in raw_treatment.columns if 'cancer_site' in col]
        raw_treatment = raw_treatment[['mrn'] + cancer_site_cols].copy()
        # rename the 'cancer_site_CXX' columns to just 'CXX'
        raw_treatment.rename(columns={col: col.replace('cancer_site_', '') for col in cancer_site_cols}, inplace=True)
        raw_treatment.drop_duplicates(subset=['mrn'], inplace=True)
        df_treatment = df_treatment.merge(raw_treatment, how='left', on='mrn')
        df_treatment = process_cancer_sites(df_treatment)
    elif held_out_set == "inference":
        raw_treatment['cancer_type'] = raw_treatment['primary_site_code'].map(COARSE_SITE_MAP).fillna(OTHER_ILL_DEFINED_GROUP)
        raw_treatment['treatment_date'] = pd.to_datetime(raw_treatment['assessment_date'], utc=True)
        # merge mrn, cancer_type, and treatment_date from raw_treatment into df_treatment
        df_treatment = df_treatment.merge(raw_treatment[['mrn', 'cancer_type', 'treatment_date']], how='left', on='mrn')

    # drop treatment_date from df_treatment
    df_treatment.drop(columns=['treatment_date'], inplace=True)

    ICC_values = []
    ICC_lower = []
    ICC_upper = []
    summary_across_targets = []

    for target_name in target_list:
        prompting_summary_df = load_prompting_result(target_name, prompting_results_dir)
        tabular_summary_df = load_tabular_result(target_name, tabular_results_path, held_out_set)

        # rename 'Probability' column in prompting_summary_df to 'prob_prompting'
        prompting_summary_df = prompting_summary_df.rename(columns={'Probability': 'prob_prompting'})
        # rename 'prob' column in tabular_summary_df to 'prob_tabular'
        tabular_summary_df = tabular_summary_df.rename(columns={'prob': 'prob_tabular'})
        # merge prompting_summary_df['mrn', 'prob_prompting'] with df_treatment on 'mrn'
        df_merged = pd.merge(prompting_summary_df[['mrn', 'prob_prompting']], df_treatment[['mrn', 'stats_dictated_by', 'cancer_type', 'treatment_year']], on='mrn', how='inner')
        # merge tabular_summary_df with df_merged on 'mrn'
        df_merged = pd.merge(df_merged, tabular_summary_df, on='mrn', how='inner')

        df_merged['dictating_physician'] = df_merged['stats_dictated_by'].apply(extract_names)
        df_merged['num_dictators'] = df_merged['dictating_physician'].apply(len)
        df_merged = df_merged.loc[df_merged['num_dictators']==1].copy()
        df_merged['dictating_physician'] = df_merged['dictating_physician'].apply(lambda x: x[0] if x else 'Unknown')
        # remove 'Unknown' dictating_physician
        df_merged = df_merged[df_merged['dictating_physician'] != 'Unknown']
        eps = 1e-6
        df_merged["logit_prompting"] = np.log((df_merged["prob_prompting"] + eps) / (1 - df_merged["prob_prompting"] + eps))
        df_merged["logit_tabular"]  = np.log((df_merged["prob_tabular"] + eps)  / (1 - df_merged["prob_tabular"]  + eps))
        df_merged['average_treatment_year'] = df_merged.groupby('dictating_physician')['treatment_year'].transform('mean')

        df_merged["logit_tabular_c"] = df_merged["logit_tabular"] - df_merged["logit_tabular"].mean()

        print(target_name)
        print(100*"#")

        # Bayesian model

        # create data matrix of physician characteristics
        df_physician_yoe = df_merged[['dictating_physician', 'average_treatment_year']].drop_duplicates()
        df_physician_yoe.rename(columns={'dictating_physician': 'med_onc'}, inplace=True)
        if held_out_set == "test":
            df_physician_char = df_physician_char_EPR.copy()
        elif held_out_set == "inference":
            df_physician_char = df_physician_char_EPIC.copy()
        df_physician_char = df_physician_char.merge(df_physician_yoe, on='med_onc')
        df_physician_char['average_YOE'] = df_physician_char['average_treatment_year'] - df_physician_char['YOG']

        # fit hierarchical model
        model = bmb.Model(
            "logit_prompting ~ logit_tabular_c + C(cancer_type) + (1|dictating_physician)",
            df_merged
        )
        res = model.fit(draws=2000, chains=4, target_accept=0.9)

        max_Rhat = az.summary(res)['r_hat'].max()
        if max_Rhat > 1.01:
            print('Check quality of samples')

        # compute Bayesian ICC
        # 1) extract posterior xarray variables
        posterior = res.posterior

        # extract as xarray DataArray
        phys_sd_name = "1|dictating_physician_sigma"
        resid_sd_name = "sigma"
        sd_phys = posterior[phys_sd_name]        # dims: (chain, draw)
        sd_resid = posterior[resid_sd_name]      # dims: (chain, draw)

        # 2) convert to variances
        var_phys = sd_phys**2     # same dims
        var_resid = sd_resid**2   # same dims

        # 3) compute ICC per posterior draw
        icc_xr = var_phys / (var_phys + var_resid)

        # 4) flatten to 1D numpy array of draws
        icc_vec = icc_xr.stack(sample=("chain", "draw")).values
        ICC_values.append(float(np.median(icc_vec)))
        ICC_lower.append(float(np.quantile(icc_vec, 0.025)))
        ICC_upper.append(float(np.quantile(icc_vec, 0.975)))

        # obtain posterior samples of the random effects for physicians we care about
        top_dictating_physicians = df_physician_char['med_onc'].unique().tolist()

        # Random effects array
        re = posterior["1|dictating_physician"]

        physicians_in_model = re.coords["dictating_physician__factor_dim"].values.tolist()

        # Select physicians present in the model
        selected_physicians = [p for p in top_dictating_physicians if p in physicians_in_model]

        selected_idx = [physicians_in_model.index(p) for p in selected_physicians]

        # Subset posterior draws
        re_subset = re[:, :, selected_idx]

        # Stack sampling dims → rows; physicians → columns
        re_df = re_subset.stack(sample=("chain", "draw")).to_pandas().T

        # Fix column names
        re_df.columns = selected_physicians

        # Flatten the MultiIndex index (chain, draw)
        re_df.index = re_df.index.map(lambda x: f"{x[0]}_{x[1]}")

        re_df = re_df.reset_index().rename(columns={'index': 'sample'})
        re_df.columns.name = None  # Remove the column name if needed

        # prepare med onc characteristics
        X = np.column_stack([
            np.ones(len(df_physician_char)),   # intercept
            df_physician_char["Canadian_Medical_Graduate"].values.astype(float),
            df_physician_char["Speaks_2nd_Language"].values.astype(float),
            df_physician_char["average_YOE"].values.astype(float)
        ])

        feature_names = ["Intercept", "Canadian_Medical_Graduate",
                        "Speaks_2nd_Language", "average_YOE"]
        
        # compute coefficients
        betas = []   # each element will be a coefficient vector

        for _, row in re_df.iterrows():
            y = row.values[1:].astype(float) # random effects for these physicians, 1d array
            # OLS closed-form solution: beta = (X^T X)^(-1) X^T y
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            betas.append(beta)

        betas = np.vstack(betas)   # shape = (samples, 4)

        # create summary data frame
        summary = pd.DataFrame({
            "Feature": feature_names,
            "Median": np.median(betas, axis=0),
            "2.5%": np.percentile(betas, 2.5, axis=0),
            "97.5%": np.percentile(betas, 97.5, axis=0),
            "Target": target_name
        })
        summary_across_targets.append(summary)

    results_df = pd.DataFrame({
        'target': target_list,
        'ICC': ICC_values,
        'ICC_lower' : ICC_lower,
        'ICC_upper': ICC_upper
    })

    results_df.to_csv(os.path.join(save_dir, f'ICC_results_{held_out_set}.csv'), index=False)
    concat_summary = pd.concat(summary_across_targets)
    concat_summary.to_csv(os.path.join(save_dir, f'characteristics_regression_coefficients_{held_out_set}.csv'), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check effect of dictating physician.")
    parser.add_argument("held_out_set", type=str, choices=["test", "inference"], help="Which held-out set to analyze.")
    parser.add_argument("target_list", type=str, help="List of target names. E.g., \"['target1', 'target2']\"")
    parser.add_argument("anchored_notes_path", type=str, help="Path to CSV file with anchored notes.")
    parser.add_argument("save_dir", type=str, help="Directory to save the results.")
    parser.add_argument("prompting_results_dir", type=str, help="Directory containing prompting results.")
    parser.add_argument("tabular_results_path", type=str, help="Path to tabular results.")
    parser.add_argument("raw_treatment_path", type=str, default="None", help="Path to raw treatment")

    args = parser.parse_args()

    # Convert string lists to actual lists
    target_list = ast.literal_eval(args.target_list)
    regress_physician_characteristics(target_list, args.held_out_set, args.anchored_notes_path, 
                                      args.save_dir, args.prompting_results_dir, args.tabular_results_path,
                                      args.raw_treatment_path)