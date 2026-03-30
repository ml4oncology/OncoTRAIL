"""
regress_physician_characteristics.py

Fits a Bayesian hierarchical model (via Bambi) to assess how much variation
in LLM prompting risk scores is explained by the dictating physician, after
controlling for tabular risk and cancer type.  Then regresses the physician-
level random effects on physician characteristics (CMG status, bilingualism,
years of experience) using OLS on posterior samples.

Outputs
-------
ICC_results_{held_out_set}.csv
    Median + 95 % credible interval of the intraclass correlation coefficient.
characteristics_regression_coefficients_{held_out_set}.csv
    Posterior median + 95 % CI for each physician-characteristic coefficient,
    aggregated across all targets.
"""

import argparse
import ast
import glob
import os

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, "/cluster/projects/gliugroup/2BLAST/data/info")
from phys_names import aliasDictionary, fellow_alias

from oncotrail.constants import df_physician_char_EPR, df_physician_char_EPIC
from oncotrail.constants import CANCER_COARSE_SITE_MAP as COARSE_SITE_MAP
from oncotrail.constants import CANCER_HIERARCHY as HIERARCHY


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HIERARCHY_RANK: dict[str, int] = {
    site: rank for rank, site in enumerate(HIERARCHY)
}
OTHER_ILL_DEFINED_GROUP = "Other / Ill-Defined"


# ---------------------------------------------------------------------------
# Cancer-site helpers
# ---------------------------------------------------------------------------

def process_cancer_sites(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary cancer-site indicator columns into a single
    ``cancer_type`` column using the coarse grouping and clinical hierarchy.

    The function looks for columns whose names match a key in
    ``COARSE_SITE_MAP`` (e.g. ``C34``).  For each patient it selects the
    highest-priority positive code according to ``HIERARCHY_RANK``.

    Parameters
    ----------
    df:
        Input DataFrame that contains ``cancer_site_CXX``-style columns
        (already renamed to ``CXX`` before this function is called).

    Returns
    -------
    The same DataFrame with a new ``cancer_type`` column appended.
    """
    defined_c_codes = set(COARSE_SITE_MAP.keys())
    cancer_site_cols = [
        col for col in df.columns
        if col.startswith("C") and col[1:].isdigit() and col in defined_c_codes
    ]

    if not cancer_site_cols:
        print("Warning: no defined C-code columns found — assigning 'Other / Ill-Defined'.")
        df["cancer_type"] = OTHER_ILL_DEFINED_GROUP
        return df

    def get_primary_cancer_type(row: pd.Series) -> str:
        """Return the highest-priority coarse cancer group for one patient."""
        # Prefer value==2 (primary), fall back to value==1 (secondary)
        positive_codes = (
            [col for col in cancer_site_cols if row.get(col) == 2]
            or [col for col in cancer_site_cols if row.get(col) == 1]
        )
        if not positive_codes:
            return OTHER_ILL_DEFINED_GROUP

        coarse_groups = {
            COARSE_SITE_MAP.get(code, OTHER_ILL_DEFINED_GROUP)
            for code in positive_codes
        }
        return min(
            coarse_groups,
            key=lambda g: HIERARCHY_RANK.get(g, len(HIERARCHY)),
        )

    df["cancer_type"] = df.apply(get_primary_cancer_type, axis=1)
    print(
        f"Created 'cancer_type' with {df['cancer_type'].nunique()} unique groups."
    )
    return df


def extract_names(s: str) -> list[str]:
    """Parse a bracketed name list from *s* and resolve physician aliases.

    The input string is expected to contain at least one line that starts with
    ``[`` and represents a Python list literal of name strings.

    Returns an empty list if parsing fails for any reason.
    """
    try:
        list_line = next(line for line in s.split("\n") if line.startswith("["))
        names = ast.literal_eval(list_line)
        return [
            fellow_alias.get(aliasDictionary.get(n, n), aliasDictionary.get(n, n))
            for n in names
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def load_prompting_result(target_name: str, prompting_results_dir: str) -> pd.DataFrame:
    """Load the prompting summary CSV for *target_name*.

    Expects exactly one ``note_*/summary_*.csv`` file under
    ``prompting_results_dir/target_name/``.
    """
    target_dir = os.path.join(prompting_results_dir, target_name)
    note_dirs = [d for d in glob.glob(os.path.join(target_dir, "note_*")) if os.path.isdir(d)]
    assert len(note_dirs) == 1, f"Expected one note_ dir in {target_dir}, found: {note_dirs}"

    summary_files = glob.glob(os.path.join(note_dirs[0], "summary_*.csv"))
    assert len(summary_files) == 1, f"Expected one summary_*.csv in {note_dirs[0]}, found: {summary_files}"

    return pd.read_csv(summary_files[0])


def load_tabular_result(
    target_name: str,
    tabular_results_path: str,
) -> pd.DataFrame:
    """Load tabular model predictions for *target_name* from a ``.npz`` file.

    Parameters
    ----------
    held_out_set:
        Either ``"test"`` or ``"inference"``.

    Returns
    -------
    DataFrame with columns ``mrn`` and ``prob``.
    """
    df_model = pd.read_csv(tabular_results_path)
    target_row = df_model[df_model["target"] == target_name.replace("_", "-")]
    assert not target_row.empty, (
        f"Target {target_name.replace('_', '-')} not found in {tabular_results_path}"
    )

    data = np.load(target_row["pred_file_name"].values[0], allow_pickle=True)
    return pd.DataFrame({"mrn": data['mrn_test'], "prob": data['test_pred'].ravel()})


# ---------------------------------------------------------------------------
# Feature-engineering helpers
# ---------------------------------------------------------------------------

def build_merged_dataframe(
    target_name: str,
    df_treatment: pd.DataFrame,
    prompting_results_dir: str,
    tabular_results_path: str,
) -> pd.DataFrame:
    """Merge prompting + tabular predictions with treatment metadata.

    Also filters to notes with exactly one dictating physician and computes
    centred logit scores (``logit_prompting``, ``logit_tabular``,
    ``logit_tabular_c``).

    Returns a clean DataFrame ready for model fitting.
    """
    prompting_df = load_prompting_result(target_name, prompting_results_dir)
    tabular_df   = load_tabular_result(target_name, tabular_results_path)

    prompting_df = prompting_df.rename(columns={"Probability": "prob_prompting"})
    tabular_df   = tabular_df.rename(columns={"prob": "prob_tabular"})

    df = (
        prompting_df[["mrn", "prob_prompting"]]
        .merge(
            df_treatment[["mrn", "stats_dictated_by", "cancer_type", "treatment_year"]],
            on="mrn", how="inner",
        )
        .merge(tabular_df, on="mrn", how="inner")
    )

    # Resolve dictating physician; keep only notes with exactly one dictator
    df["dictating_physician"] = df["stats_dictated_by"].apply(extract_names)
    df = df[df["dictating_physician"].apply(len) == 1].copy()
    df["dictating_physician"] = df["dictating_physician"].apply(lambda x: x[0] if x else 'Unknown')
    df = df[df["dictating_physician"] != "Unknown"]

    # Logit scores
    eps = 1e-6
    df["logit_prompting"] = np.log(
        (df["prob_prompting"] + eps) / (1 - df["prob_prompting"] + eps)
    )
    df["logit_tabular"] = np.log(
        (df["prob_tabular"] + eps) / (1 - df["prob_tabular"] + eps)
    )
    # Mean-centre the tabular logit (used as a covariate in the mixed model)
    df["logit_tabular_c"] = df["logit_tabular"] - df["logit_tabular"].mean()

    # Per-physician mean treatment year (used to estimate years of experience)
    df["average_treatment_year"] = df.groupby("dictating_physician")["treatment_year"].transform("mean")

    return df


# ---------------------------------------------------------------------------
# Bayesian model fitting + ICC
# ---------------------------------------------------------------------------

def fit_bayesian_model(df: pd.DataFrame) -> az.InferenceData:
    """Fit a Bayesian linear mixed model predicting ``logit_prompting``.

    Fixed effects: mean-centred tabular logit + cancer type.
    Random effect: intercept per dictating physician.

    Returns the ArviZ InferenceData object from Bambi.
    """
    model = bmb.Model(
        "logit_prompting ~ logit_tabular_c + C(cancer_type) + (1|dictating_physician)",
        df,
    )
    res = model.fit(draws=2000, chains=4, target_accept=0.9)

    max_rhat = az.summary(res)["r_hat"].max()
    if max_rhat > 1.01:
        print(f"  Warning: max R-hat = {max_rhat:.3f} — check sample quality.")

    return res


def compute_icc(res: az.InferenceData) -> tuple[float, float, float]:
    """Compute the Bayesian ICC (physician variance / total variance).

    Parameters
    ----------
    res:
        Fitted InferenceData returned by :func:`fit_bayesian_model`.

    Returns
    -------
    Tuple of (median ICC, 2.5th percentile, 97.5th percentile).
    """
    posterior = res.posterior
    var_phys  = posterior["1|dictating_physician_sigma"] ** 2
    var_resid = posterior["sigma"] ** 2

    icc_vec = (var_phys / (var_phys + var_resid)).stack(sample=("chain", "draw")).values
    return (
        float(np.median(icc_vec)),
        float(np.quantile(icc_vec, 0.025)),
        float(np.quantile(icc_vec, 0.975)),
    )


# ---------------------------------------------------------------------------
# Regression of random effects on physician characteristics
# ---------------------------------------------------------------------------

def regress_re_on_physician_chars(
    res: az.InferenceData,
    df_physician_char: pd.DataFrame,
) -> pd.DataFrame:
    """Regress posterior physician random effects on physician characteristics.

    Uses OLS in closed form (beta = (X'X)^{-1} X'y) on each posterior sample,
    then summarises the resulting coefficient distribution.

    Parameters
    ----------
    res:
        Fitted InferenceData from :func:`fit_bayesian_model`.
    df_physician_char:
        DataFrame indexed by ``med_onc`` with columns
        ``Canadian_Medical_Graduate``, ``Speaks_2nd_Language``, ``average_YOE``.

    Returns
    -------
    DataFrame with columns Feature, Median, 2.5%, 97.5%.
    """
    posterior = res.posterior
    re = posterior["1|dictating_physician"]
    physicians_in_model = re.coords["dictating_physician__factor_dim"].values.tolist()

    target_physicians = df_physician_char["med_onc"].unique().tolist()
    selected = [p for p in target_physicians if p in physicians_in_model]
    selected_idx = [physicians_in_model.index(p) for p in selected]

    # Posterior draws: shape (n_samples, n_selected_physicians)
    re_df = (
        re[:, :, selected_idx]
        .stack(sample=("chain", "draw"))
        .to_pandas()
        .T
    )
    re_df.columns = selected
    re_df.index = re_df.index.map(lambda x: f"{x[0]}_{x[1]}")
    re_df = re_df.reset_index().rename(columns={"index": "sample"})
    re_df.columns.name = None

    # Design matrix (intercept + 3 features) for the selected physicians
    char = df_physician_char.set_index("med_onc").loc[selected]
    X = np.column_stack([
        np.ones(len(char)),
        char["Canadian_Medical_Graduate"].astype(float),
        char["Speaks_2nd_Language"].astype(float),
        char["average_YOE"].astype(float),
    ])
    feature_names = [
        "Intercept",
        "Canadian_Medical_Graduate",
        "Speaks_2nd_Language",
        "average_YOE",
    ]

    # OLS on each posterior sample: hoist X'X outside the loop
    XtX = X.T @ X
    betas = np.vstack([
        np.linalg.solve(XtX, X.T @ row.values[1:].astype(float))
        for _, row in re_df.iterrows()
    ])  # shape: (n_samples, 4)

    return pd.DataFrame({
        "Feature": feature_names,
        "Median":  np.median(betas, axis=0),
        "2.5%":    np.percentile(betas, 2.5,  axis=0),
        "97.5%":   np.percentile(betas, 97.5, axis=0),
    })


# ---------------------------------------------------------------------------
# Treatment-table loader (run once, shared across all targets)
# ---------------------------------------------------------------------------

def load_treatment_table(
    anchored_notes_path: str,
    raw_treatment_path: str,
    held_out_set: str,
) -> pd.DataFrame:
    """Build the per-note treatment table with cancer type and treatment year.

    Parameters
    ----------
    anchored_notes_path:
        CSV with columns including ``mrn``, ``stats_dictated_by``,
        ``treatment_date``.
    raw_treatment_path:
        Parquet file whose schema differs between ``test`` and ``inference``
        splits (see inline comments).
    held_out_set:
        ``"test"`` or ``"inference"``.

    Returns
    -------
    DataFrame with columns: mrn, stats_dictated_by, treatment_year, cancer_type.
    """
    df = pd.read_csv(anchored_notes_path)
    df["treatment_date"] = pd.to_datetime(df["treatment_date"], utc=True)
    df["treatment_year"] = df["treatment_date"].dt.year
    df = df[["mrn", "stats_dictated_by", "treatment_date", "treatment_year"]].copy()

    raw = pd.read_parquet(raw_treatment_path)

    if held_out_set == "test":
        cancer_site_cols = [c for c in raw.columns if "cancer_site" in c]
        raw = raw[["mrn", "treatment_date"] + cancer_site_cols].copy()
        raw.rename(columns={c: c.replace("cancer_site_", "") for c in cancer_site_cols}, inplace=True)
        raw["treatment_date"] = pd.to_datetime(raw["treatment_date"], utc=True)
        raw.drop_duplicates(subset=["mrn"], inplace=True)
        df = df.merge(raw, how="left", on=["mrn", "treatment_date"])
        df = process_cancer_sites(df)
        return df.drop(columns=["treatment_date"])
    
    elif held_out_set == "inference":
        raw["cancer_type"] = raw["primary_site_code"].map(COARSE_SITE_MAP).fillna(OTHER_ILL_DEFINED_GROUP)
        raw["treatment_date"] = pd.to_datetime(raw["assessment_date"], utc=True)
        # create a column with only the date (drops the time)
        df['treatment_date_only'] = df['treatment_date'].dt.floor('D')
        raw['treatment_date_only'] = raw['treatment_date'].dt.floor('D')

        # merge on MRN and date only
        df = df.merge(
            raw[['mrn', 'cancer_type', 'treatment_date_only']],
            how='left',
            on=['mrn', 'treatment_date_only']
        )
        return df.drop(columns=["treatment_date", "treatment_date_only"])


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def regress_physician_characteristics(
    target_list: list[str],
    held_out_set: str,
    anchored_notes_path: str,
    save_dir: str,
    prompting_results_dir: str,
    tabular_results_path: str,
    raw_treatment_path: str,
) -> None:
    """Run the full physician-characteristic regression pipeline.

    For each target in *target_list*:
      1. Merge prompting / tabular predictions with treatment metadata.
      2. Fit a Bayesian hierarchical model (Bambi).
      3. Compute the ICC from posterior samples.
      4. Regress physician random effects on physician characteristics.

    Results are written to *save_dir*.
    """
    df_treatment = load_treatment_table(anchored_notes_path, raw_treatment_path, held_out_set)

    df_physician_char_base = (
        df_physician_char_EPR.copy() if held_out_set == "test"
        else df_physician_char_EPIC.copy()
    )

    icc_values: list[float] = []
    icc_lower:  list[float] = []
    icc_upper:  list[float] = []
    summary_across_targets: list[pd.DataFrame] = []

    for target_name in target_list:
        print(f"\n{'#' * 80}\n{target_name}\n{'#' * 80}")

        # --- 1. Feature engineering ---
        df_merged = build_merged_dataframe(
            target_name, df_treatment,
            prompting_results_dir, tabular_results_path
        )
        
        # --- 2. Physician-characteristic table (adds average YOE) ---
        physician_yoe = (
            df_merged[["dictating_physician", "average_treatment_year"]]
            .drop_duplicates()
            .rename(columns={"dictating_physician": "med_onc"})
        )
        df_physician_char = df_physician_char_base.merge(physician_yoe, on="med_onc")
        df_physician_char["average_YOE"] = (
            df_physician_char["average_treatment_year"] - df_physician_char["YOG"]
        )
        
        # --- 3. Bayesian model + ICC ---
        res = fit_bayesian_model(df_merged)
        median_icc, lower_icc, upper_icc = compute_icc(res)
        icc_values.append(median_icc)
        icc_lower.append(lower_icc)
        icc_upper.append(upper_icc)

        # --- 4. Regress random effects on physician characteristics ---
        coef_summary = regress_re_on_physician_chars(res, df_physician_char)
        coef_summary["Target"] = target_name
        summary_across_targets.append(coef_summary)

    # --- Save results ---
    pd.DataFrame({
        "target":    target_list,
        "ICC":       icc_values,
        "ICC_lower": icc_lower,
        "ICC_upper": icc_upper,
    }).to_csv(os.path.join(save_dir, f"ICC_results_{held_out_set}.csv"), index=False)

    pd.concat(summary_across_targets).to_csv(
        os.path.join(save_dir, f"characteristics_regression_coefficients_{held_out_set}.csv"),
        index=False,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assess dictating-physician effects on LLM prompting risk scores."
    )
    parser.add_argument(
        "held_out_set", choices=["test", "inference"],
        help="Which held-out split to analyze.",
    )
    parser.add_argument(
        "target_list",
        help="List of target names. E.g., \"['target1', 'target2']\""
    )
    parser.add_argument("anchored_notes_path", help="CSV with anchored notes.")
    parser.add_argument("save_dir",            help="Directory to write output CSVs.")
    parser.add_argument("prompting_results_dir", help="Root dir for prompting results.")
    parser.add_argument("tabular_results_path",  help="CSV index of tabular model files.")
    parser.add_argument(
        "raw_treatment_path", default="None",
        help="Parquet file with raw treatment records.",
    )
    args = parser.parse_args()

    regress_physician_characteristics(
        target_list           = ast.literal_eval(args.target_list),
        held_out_set          = args.held_out_set,
        anchored_notes_path   = args.anchored_notes_path,
        save_dir              = args.save_dir,
        prompting_results_dir = args.prompting_results_dir,
        tabular_results_path  = args.tabular_results_path,
        raw_treatment_path    = args.raw_treatment_path,
    )