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

# prepare medical oncologist characteristics dataframe

data = []

# Append rows as lists
data.append(["Hamzeh AdelAmin Albaba", 2006, "N", "Arabic"])
data.append(["Raymond Jang", 2006, "Y", "NA"])
data.append(["Xueyu Eric Chen", 1996, "Y", "Mandarin"])
data.append(["Anna Spreafico", 2003, "N", "Italian"])
data.append(["Frances Alice Shepherd", 1970, "Y", "French"])
data.append(["Penelope Bradbury", 1994, "N", "NA"])
data.append(["David William Hedley", 1970, "N", "NA"])
data.append(["Natasha Basant Leighl", 1994, "Y", "NA"])
data.append(["Neesha Cindy Dhani", 2001, "Y", "NA"])
data.append(["Rebecca Michelle Prince", 2005, "N", "NA"])
data.append(["Grainne Mary O'Kane", 2007, "N", "NA"])
data.append(["Aaron Richard Hansen", 2004, "N", "NA"])
data.append(["Doreen Anuli Ezeife", 2011, "Y", "NA"])
data.append(["Geoffrey Liu", 1993, "Y", "NA"])
data.append(["Jennifer Jane Knox", 1995, "Y", "NA"])
data.append(["Yanshuo Cao", 2009, "N", "Mandarin"])
data.append(["Lawson Eng", 2014, "Y", "NA"])
data.append(["Adrian Gerold Sacher", 2009, "Y", "NA"])
data.append(["Aline Fusco Fares", 2010, "N", "Portuguese"])
data.append(["Kirstin Ann Perdrizet", 2013, "Y", "NA"])
data.append(["Shari Moura", 1991, "Y", "NA"])
data.append(["Mor Tal Moskovitz", 2010, "N", "Hebrew"])
data.append(["Sally Lau", 2013, "Y", "NA"])
data.append(["Michael Herman", 2013, "Y", "NA"])
data.append(["Di Maria Jiang", 2012, "Y", "Mandarin"])
data.append(["Lillian L. Y. Siu", 1991, "Y", "Cantonese"])
data.append(["Sophie Feng", None, None, None])
data.append(["Daniel Yokom", 2011, "Y", "NA"])
data.append(["Elaine Sarah Bouttell", 1997, "Y", "NA"])
data.append(["Ibrahim Algorashi", 2016, "N", "Arabic"])
data.append(["Kelvin Young", 2010, "N", "NA"])
data.append(["Daniel Shepshelovich", 2005, "N", "Hebrew, Russian"])
data.append(["Catherine Labbe", 2009, "Y", "French"])
data.append(["Kirsty Laura Taylor", 2011, "N", "NA"])
data.append(["Charles Lim", 2012, "Y", "NA"])
data.append(["Elena Elimova", 2008, "Y", "NA"])
data.append(["Hao-Wen Sim", 2007, "N", "NA"])
data.append(["Daphne Dai", None, None, None])

df_physician_char_global = pd.DataFrame(data, columns=["Name", "YOG", "Canadian_Medical_Graduate", "Other_Languages"])
df_physician_char_global = df_physician_char_global.dropna(subset=["YOG"])
# replace value in Canadian_Medical_Graduate
df_physician_char_global['Canadian_Medical_Graduate'] = df_physician_char_global['Canadian_Medical_Graduate'].map(lambda x: True if x == "Y" else False)
df_physician_char_global['Speaks_2nd_Language'] = df_physician_char_global['Other_Languages'].map(lambda x: True if x != "NA" else False)
df_physician_char_global.rename(columns={"Name": "med_onc"}, inplace=True)

# --- 1. DEFINITION OF COARSE GROUPS ---
# This dictionary maps detailed ICD-O-3 codes (C-codes) to a coarser, clinically relevant group.
COARSE_SITE_MAP = {
    # Head & Neck (H&N)
    "C00": "Head & Neck", "C01": "Head & Neck", "C02": "Head & Neck", "C03": "Head & Neck", "C04": "Head & Neck", 
    "C05": "Head & Neck", "C06": "Head & Neck", "C07": "Head & Neck", "C08": "Head & Neck", "C09": "Head & Neck", 
    "C10": "Head & Neck", "C11": "Head & Neck", "C12": "Head & Neck", "C13": "Head & Neck", "C14": "Head & Neck", 
    "C30": "Head & Neck", "C31": "Head & Neck", "C32": "Head & Neck", 
    
    # Upper GI
    "C15": "Upper GI", "C16": "Upper GI", "C17": "Upper GI", 
    
    # Lower GI
    "C18": "Lower GI", "C19": "Lower GI", "C20": "Lower GI", "C21": "Lower GI", 
    
    # Hepato-Pancreato-Biliary (HPB)
    "C22": "HPB", "C23": "HPB", "C24": "HPB", "C25": "HPB", 
    
    # Thoracic (Lung & Chest)
    "C33": "Thoracic", "C34": "Thoracic", "C37": "Thoracic", "C38": "Thoracic", 
    
    # Musculoskeletal & Soft Tissue (Sarcomas)
    "C40": "Musculoskeletal & Soft Tissue", "C41": "Musculoskeletal & Soft Tissue", "C49": "Musculoskeletal & Soft Tissue",
    
    # Skin
    "C44": "Skin",
    
    # Breast
    "C50": "Breast",
    
    # Female Genital
    "C51": "Female Genital", "C52": "Female Genital", "C53": "Female Genital", "C54": "Female Genital", 
    "C55": "Female Genital", "C56": "Female Genital", "C57": "Female Genital", "C58": "Female Genital", 
    
    # Male Genital
    "C60": "Male Genital", "C61": "Male Genital", "C62": "Male Genital", "C63": "Male Genital", 
    
    # Urological
    "C64": "Urological", "C65": "Urological", "C66": "Urological", "C67": "Urological", "C68": "Urological", 
    
    # Nervous System
    "C69": "Nervous System", "C71": "Nervous System", "C72": "Nervous System", 
    
    # Endocrine
    "C73": "Endocrine", "C74": "Endocrine", "C75": "Endocrine", 
    
    # Other / Ill-Defined
    "C26": "Other / Ill-Defined", "C47": "Other / Ill-Defined", "C48": "Other / Ill-Defined", 
    "C76": "Other / Ill-Defined", "C80": "Other / Ill-Defined"
}

# --- 2. DEFINITION OF HIERARCHY ---
# Higher priority numbers mean the cancer site is considered more aggressive/prognostically significant 
# and should be chosen as the "Primary Site" in case of multiple primaries.
# This list is ordered from highest priority (index 0) to lowest priority.
HIERARCHY = [
    "Thoracic",                         # 1 (Highest Priority)
    "HPB",                              # 2
    "Nervous System",                   # 3
    "Upper GI",                         # 4
    "Lower GI",                         # 5
    "Head & Neck",                      # 6
    "Female Genital",                   # 7
    "Urological",                       # 8
    "Male Genital",                     # 9
    "Breast",                           # 10
    "Endocrine",                        # 11
    "Musculoskeletal & Soft Tissue",    # 12
    "Skin",                             # 13
    "Other / Ill-Defined"               # 14 (Lowest Priority)
]
# Create a dictionary where the key is the coarse group and the value is its rank (lower rank means higher priority)
HIERARCHY_RANK = {site: rank for rank, site in enumerate(HIERARCHY)}
OTHER_ILL_DEFINED_GROUP = "Other / Ill-Defined"


def process_cancer_sites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts multiple binary cancer site columns into a single 'cancer_type' column 
    based on coarse grouping and clinical hierarchy (Strategy A).

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
            # This handles the 6 cases where no indicator was set to 1.
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
        return [
            fellow_alias.get(aliasDictionary.get(name, name), aliasDictionary.get(name, name))
            for name in names
        ]

    except Exception:
        return []
    
def load_prompting_result(target_name):
    model_val = "/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification/data/prompt_engineering/test_set"
    # Find path: {model_val}/{target_name}/note_*/summary_*.csv
    target_dir = os.path.join(model_val, target_name)
    note_dirs = [d for d in glob.glob(os.path.join(target_dir, "note_*")) if os.path.isdir(d)]
    assert len(note_dirs) == 1, f"Expected one note_ directory in {target_dir}, found: {note_dirs}"
    summary_files = glob.glob(os.path.join(note_dirs[0], "summary_*.csv"))
    assert len(summary_files) == 1, f"Expected one summary_*.csv in {note_dirs[0]}, found: {summary_files}"
    prompting_summary_df = pd.read_csv(summary_files[0])
    
    return prompting_summary_df

def load_tabular_result(target_name):
    tabular_results = "/cluster/projects/gliugroup/work_dir/wayne_uy/gitrepo/2024/LLM-notes-classification/results/ML/best_result_summary_firstTreatmentOnly-medOnc-ConsultLetterClinic_deid_tabular_all_Temporal.csv"
    df_model = pd.read_csv(tabular_results)
    target_row = df_model[df_model["target"] == target_name.replace("_", "-")]
    assert not target_row.empty, f"Target {target_name.replace('_', '-')} not found in dataframe"
    data = np.load(target_row["pred_file_name"].values[0], allow_pickle=True)
    mrns = data['mrn_test']
    probs = data['test_pred'].ravel()

    return pd.DataFrame({'mrn': mrns, 'prob': probs})

def robust_mixedlm_fit(model, reml=True, full_output=True):
    """
    Fit MixedLM model, retrying with LBFGS if convergence warnings occur.
    """

    # Flag to indicate retry
    had_warning = False

    # Catch convergence warnings on first attempt
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("always", RuntimeWarning)   # catches Hessian PD issues

        result = model.fit(reml=reml, full_output=full_output)

        # Check if any warning is "bad enough" to trigger a refit
        for warn in w:
            msg = str(warn.message)
            if (
                "matrix" in msg.lower() or
                "gradient" in msg.lower() or
                "hessian" in msg.lower() or
                "positive definite" in msg.lower()
            ):
                had_warning = True
                break

    if not had_warning:
        return result   # success on first try

    print("⚠️ Convergence issues detected — retrying with method='lbfgs'...")

    # Retry with explicit LBFGS (often more stable)
    return model.fit(reml=reml, method="lbfgs", full_output=full_output)

def check_physician_effect(target_list, anchored_notes_path, save_dir, raw_treatment_path):

    df_treatment = pd.read_csv(anchored_notes_path)
    df_treatment['treatment_date'] = pd.to_datetime(df_treatment['treatment_date'])
    df_treatment['treatment_year'] = df_treatment['treatment_date'].dt.year
    df_treatment = df_treatment[['mrn', 'stats_dictated_by', 'treatment_year']].copy()   

    if raw_treatment_path != "None":
        raw_treatment = pd.read_parquet(raw_treatment_path)
        # find all columns in raw_treatment that contain 'cancer_site'
        cancer_site_cols = [col for col in raw_treatment.columns if 'cancer_site' in col]
        raw_treatment = raw_treatment[['mrn'] + cancer_site_cols].copy()
        # replace values in cancer_site_cols with 1 if > 0 else 0
        # raw_treatment[cancer_site_cols] = raw_treatment[cancer_site_cols].applymap(lambda x: 1 if x > 0 else 0)
        # rename the 'cancer_site_CXX' columns to just 'CXX'
        raw_treatment.rename(columns={col: col.replace('cancer_site_', '') for col in cancer_site_cols}, inplace=True)
        raw_treatment.drop_duplicates(subset=['mrn'], inplace=True)
        df_treatment = df_treatment.merge(raw_treatment, how='left', on='mrn')
        df_treatment = process_cancer_sites(df_treatment)

    m0R2 = []
    m1R2 = []
    deltaR2 = []
    p_values = []
    ICC_values = []
    ICC_lower = []
    ICC_upper = []
    summary_across_targets = []

    for target_name in target_list:
        prompting_summary_df = load_prompting_result(target_name)
        tabular_summary_df = load_tabular_result(target_name)
    
        # fixed effects model

        # rename 'Probability' column in prompting_summary_df to 'prob_prompting'
        prompting_summary_df = prompting_summary_df.rename(columns={'Probability': 'prob_prompting'})
        # rename 'prob' column in tabular_summary_df to 'prob_tabular'
        tabular_summary_df = tabular_summary_df.rename(columns={'prob': 'prob_tabular'})
        # merge prompting_summary_df['mrn', 'prob_prompting'] with df_treatment on 'mrn'
        if raw_treatment_path == "None":
            df_merged = pd.merge(prompting_summary_df[['mrn', 'prob_prompting']], df_treatment, on='mrn', how='inner')
        else:
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

        if raw_treatment_path == "None":
            m0 = smf.ols("logit_prompting ~ logit_tabular", data=df_merged).fit()
            m1 = smf.ols("logit_prompting ~ logit_tabular + C(dictating_physician)", data=df_merged).fit()
        else: 
            m0 = smf.ols("logit_prompting ~ logit_tabular + C(cancer_type)", data=df_merged).fit()
            m1 = smf.ols("logit_prompting ~ logit_tabular + C(cancer_type) + C(dictating_physician)", data=df_merged).fit()

        m0R2.append(m0.rsquared)
        m1R2.append(m1.rsquared)
        deltaR2.append(m1.rsquared - m0.rsquared)
        p_values.append(anova_lm(m0, m1)['Pr(>F)'][1])

        df_merged["resid_m0"] = m0.resid
        df_physician_residual = df_merged.groupby("dictating_physician")["resid_m0"].mean().reset_index()
        df_physician_residual['resid_m0'] = np.abs(df_physician_residual['resid_m0'])
        # add counts of dictating_physician in df_merged to df_physician_residual
        physician_counts = df_merged['dictating_physician'].value_counts().reset_index()
        df_physician_residual = pd.merge(df_physician_residual, physician_counts, on='dictating_physician', how='inner')
        df_physician_residual.sort_values(by='resid_m0', ascending=False).head(20).to_csv(os.path.join(save_dir, f"top20_residuals_{target_name}.csv"), index=False)
        df_physician_residual.sort_values(by='count', ascending=False).head(30).to_csv(os.path.join(save_dir, f"top30_count_{target_name}.csv"), index=False)

        df_merged["logit_tabular_c"] = df_merged["logit_tabular"] - df_merged["logit_tabular"].mean()

        print(target_name)
        print(100*"#")

        if raw_treatment_path == "None":
            raise NotImplementedError("Not implemented yet")

        # Mixed effects model

        # mixed effects model
        # if raw_treatment_path == "None":
        #     m_mixed = smf.mixedlm("logit_prompting ~ logit_tabular_c", 
        #                 data=df_merged, 
        #                 groups=df_merged["dictating_physician"], re_formula="1")
        # else:
        #     m_mixed = smf.mixedlm("logit_prompting ~ logit_tabular_c + C(cancer_type)", 
        #               data=df_merged, 
        #               groups=df_merged["dictating_physician"], re_formula="1")
        # # res_mixed = m_mixed.fit()

        # # res_mixed = m_mixed.fit(reml=True, full_output=True)

        # res_mixed = robust_mixedlm_fit(m_mixed)

        # res_mixed = m_mixed.fit(reml=True, method="lbfgs", maxiter=5000, full_output=True)

        # Catch convergence warnings
        # try:
        #     with warnings.catch_warnings(record=True) as w:
        #         warnings.filterwarnings("error", category=ConvergenceWarning)

        #         res_mixed = m_mixed.fit()

        #     # If fit succeeds with no warnings:
        #     model_failed = False

        # except ConvergenceWarning as e:
        #     print(f"[WARNING] MixedLM convergence issue for target: {target_name}")
        #     print(f"Message: {str(e)}")

        #     # Try a fallback optimizer (common for MixedLM)
        #     try:
        #         for method in ["lbfgs", "cg", "powell", "bfgs"]:
        #             try:
        #                 res_mixed = m_mixed.fit(method=method)
        #                 model_failed = False
        #                 print(method)
        #                 break
        #             except Exception:
        #                 continue
                
        #     except Exception as e2:
        #         print(f"[ERROR] Second attempt failed for {target_name}: {e2}")
        #         model_failed = True
        #         res_mixed = None

        # # ----------------------------------------------------
        # # If problematic, save the model / design matrices
        # # ----------------------------------------------------
        # if model_failed:

        #     save_path = f"{save_dir}/mixedlm_debug_{target_name}.pkl"
        #     with open(save_path, "wb") as f:
        #         pickle.dump({"model": m_mixed,
        #                     "result": res_mixed}, f)
        #     print(f"Saved problematic model to {save_path}")

        #     res_mixed = m_mixed.fit()

        # var_phys = res_mixed.cov_re.iloc[0, 0]   # variance of physician random intercepts
        # var_resid = res_mixed.scale              # residual variance
        # icc = var_phys / (var_phys + var_resid)
        # ICC_values.append(icc)

        # # # regress on physician characteristics
        # random_intercepts = res_mixed.random_effects
        # df_u = pd.DataFrame({
        #     "med_onc": list(random_intercepts.keys()),
        #     "intercept": [float(v.iloc[0]) for v in random_intercepts.values()]
        # })
        # df_u = df_u.loc[df_u['med_onc'].isin(df_physician_char['med_onc'])]
        # df_u = df_u.merge(df_physician_char[['med_onc', 'YOG', 'Canadian_Medical_Graduate', 'Speaks_2nd_Language']], on='med_onc')
        # df_physician_yoe = df_merged[['dictating_physician', 'average_treatment_year']].drop_duplicates()
        # df_physician_yoe.rename(columns={'dictating_physician': 'med_onc'}, inplace=True)
        # df_u = df_u.merge(df_physician_yoe, on='med_onc')
        # df_u['average_YOE'] = df_u['average_treatment_year'] - df_u['YOG']

        # residual_model = smf.ols("intercept ~ Canadian_Medical_Graduate + Speaks_2nd_Language + average_YOE", data=df_u).fit()

        # # save coefficient estimates
        # # 1. Get the fixed effects results table
        # results_table = residual_model.summary2().tables[1]

        # # 2. Select only the physician characteristics we care about
        # # Note: We exclude the 'const' (Intercept) term.
        # # The variable names must match exactly what was used in the OLS model.
        # # Adjust the list below if your variable names are different.
        # relevant_coefs = ['average_YOE', 'Canadian_Medical_Graduate[T.True]', 'Speaks_2nd_Language[T.True]']

        # df_results_medonc_char = results_table.loc[relevant_coefs, ['Coef.', 'Std.Err.', 't', 'P>|t|']].copy()

        # # 3. Rename columns for clarity and consistency
        # df_results_medonc_char.rename(columns={
        #     'Coef.': 'Coefficient', 
        #     'Std.Err.': 'Std_Error',
        #     'P>|t|': 'P_Value'
        # }, inplace=True)

        # # 4. Add the Target Name and convert the index (the feature names) to a column
        # df_results_medonc_char['Target'] = target_name
        # df_results_medonc_char['Feature'] = df_results_medonc_char.index
        # df_results_medonc_char.reset_index(drop=True, inplace=True)

        # df_results_medonc_char.to_csv(os.path.join(save_dir, f"physician_characteristics_effects_{target_name}.csv"), index=False)

        # Bayesian model

        # create data matrix of physician characteristics
        df_physician_yoe = df_merged[['dictating_physician', 'average_treatment_year']].drop_duplicates()
        df_physician_yoe.rename(columns={'dictating_physician': 'med_onc'}, inplace=True)
        df_physician_char = df_physician_char_global.copy()
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
        'm0R2': m0R2,
        'm1R2': m1R2,
        'deltaR2': deltaR2,
        'p_value': p_values,
        'ICC': ICC_values,
        'ICC_lower' : ICC_lower,
        'ICC_upper': ICC_upper
    })

    results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    concat_summary = pd.concat(summary_across_targets)
    concat_summary.to_csv(os.path.join(save_dir, 'stage2_coefficients.csv'), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check effect of dictating physician.")
    parser.add_argument("target_list", type=str, help="List of target names. E.g., \"['target1', 'target2']\"")
    parser.add_argument("anchored_notes_path", type=str, help="Path to CSV file with anchored notes.")
    parser.add_argument("save_dir", type=str, help="Directory to save the results.")
    parser.add_argument("--raw_treatment_path", type=str, default="None", help="path to raw treatment")

    args = parser.parse_args()

    # Convert string lists to actual lists
    target_list = ast.literal_eval(args.target_list)
    check_physician_effect(target_list, args.anchored_notes_path, args.save_dir, args.raw_treatment_path)