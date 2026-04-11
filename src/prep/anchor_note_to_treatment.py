"""
Anchor clinical notes to treatment dates for machine learning pipeline.

This module processes clinical notes and treatment data, anchoring notes to treatment
sessions based on configurable time windows and filtering criteria. It supports both
training and inference modes.
"""

import os
import argparse
import importlib.util

import numpy as np
import pandas as pd

from make_clinical_dataset.epr.combine import combine_meas_to_main_data
from make_clinical_dataset.epr.engineer import (
    get_change_since_prev_session,
    get_missingness_features,
    collapse_rare_categories,
    get_visit_month_feature
)
from make_clinical_dataset.epr.filter import (
    drop_samples_with_no_targets,
    drop_unused_drug_features,
    drop_highly_missing_features,
    keep_only_one_per_week
)
from make_clinical_dataset.shared.constants import (
    SYMP_COLS,
    LAB_CHANGE_COLS,
    LAB_COLS,
    SYMP_CHANGE_COLS
)
from make_clinical_dataset.epr.prep import fill_missing_data_heuristically
from oncotrail.prep.add_tabular_to_note import add_tabular_data_to_note
import logging
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load physician aliases from external configuration
spec = importlib.util.spec_from_file_location(
    "phys_names",
    "/cluster/projects/gliugroup/2BLAST/data/info/phys_names.py"
)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)
aliasDictionary = constants.aliasDictionary


# Constants
COLUMNS_TO_DROP = [
    'target_ED_note', 'target_ED_60d', 'target_ED_90d', 'drug_name', 'postal_code',
    'target_hemoglobin_min', 'target_platelet_min', 'target_neutrophil_min',
    'target_creatinine_max', 'target_alanine_aminotransferase_max',
    'target_aspartate_aminotransferase_max', 'target_total_bilirubin_max',
    'target_H_note', 'target_ED_CTAS_score', 'target_H_length_of_stay',
    'target_H_30d', 'target_H_60d', 'target_H_90d', 'target_ED2H', 'prev_ED_visit_note',
    'prev_hospitalization_note', 'postalcode', 'religion', 'preferred_language',
    'prev_hospitalization_length_of_stay', 'days_since_prev_hospitalization',
    'cisplatin', 'eGFR'
]


def load_and_prepare_treatment_data(
    treatment_data_path: str,
    mode: str,
    train_test_anchored_df_path: str = None,
    treatment_dates_path: str = None
) -> pd.DataFrame:
    """
    Load and prepare treatment data based on mode (train/inference).

    Args:
        treatment_data_path: Path to treatment parquet file
        mode: Either 'train' or 'inference'
        train_test_anchored_df_path: Path to training data (inference mode only)
        treatment_dates_path: Path to treatment dates (inference mode only)

    Returns:
        Prepared treatment dataframe
    """
    df_treat = pd.read_parquet(treatment_data_path, engine='pyarrow', use_nullable_dtypes=True)

    # Standardize date columns
    if 'treatment_date' not in df_treat.columns:
        df_treat['treatment_date'] = df_treat['assessment_date']
    else:
        df_treat['assessment_date'] = df_treat['treatment_date']

    if mode == 'inference':
        df_treat = _prepare_inference_treatment_data(
            df_treat,
            train_test_anchored_df_path,
            treatment_dates_path
        )

    # Standardize columns
    df_treat['intent'] = df_treat['intent'].str.upper()
    df_treat = df_treat.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    df_treat.rename(columns={'target_ED_30d': 'target_ED_visit'}, inplace=True)

    df_treat['treatment_date'] = pd.to_datetime(df_treat['treatment_date'])
    df_treat['assessment_date'] = pd.to_datetime(df_treat['assessment_date'])

    # Feature engineering
    if not any(col in df_treat.columns for col in ['visit_month_sin', 'visit_month_cos']):
        df_treat = get_visit_month_feature(df_treat)

    df_treat = keep_only_one_per_week(df_treat)
    df_treat = get_change_since_prev_session(df_treat)

    return df_treat


def _prepare_inference_treatment_data(
    df_treat: pd.DataFrame,
    train_test_anchored_df_path: str,
    treatment_dates_path: str
) -> pd.DataFrame:
    """
    Prepare treatment data specifically for inference mode.

    Args:
        df_treat: Raw treatment dataframe
        train_test_anchored_df_path: Path to training reference data
        treatment_dates_path: Path to treatment dates

    Returns:
        Prepared dataframe for inference
    """
    # Standardize sex column
    if 'sex' in df_treat.columns:
        df_treat.rename(columns={'sex': 'female'}, inplace=True)

    df_treat['female'] = np.where(
        df_treat['female'] == 'female', 1,
        np.where(df_treat['female'] == 'male', 0, -1)
    )

    # Align cancer site columns with training data
    df_train_test = pd.read_csv(train_test_anchored_df_path)
    cancer_site_cols = [col for col in df_train_test.columns if col.startswith('cancer_site')]

    # Create binary columns for each cancer site
    unique_sites = df_treat['primary_site_code'].unique()
    for site in unique_sites:
        col_name = f'cancer_site_{site}'
        df_treat[col_name] = (df_treat['primary_site_code'] == site).astype(int)

    # Handle "other" cancer sites
    all_cancer_cols = [col for col in df_treat.columns if col.startswith('cancer_site_')]
    other_cols = [col for col in all_cancer_cols if col not in cancer_site_cols]
    df_treat['cancer_site_other'] = df_treat[other_cols].max(axis=1)

    # Keep only relevant columns
    non_cancer_cols = [col for col in df_treat.columns if not col.startswith('cancer_site_')]
    cols_to_keep = list(set(non_cancer_cols + cancer_site_cols + ['cancer_site_other']))
    df_treat = df_treat[cols_to_keep]

    # Merge treatment dates if provided
    if treatment_dates_path:
        df_treat_dates = pd.read_parquet(treatment_dates_path, engine='pyarrow', use_nullable_dtypes=True)
        df_treat = df_treat.merge(
            df_treat_dates,
            on=['mrn', 'treatment_date', 'assessment_date'],
            how='inner'
        )

    return df_treat


def load_and_prepare_notes(
    notes_data_path: str,
    mode: str,
    config_name: str
) -> pd.DataFrame:
    """
    Load and prepare clinical notes data.

    Args:
        notes_data_path: Path to notes parquet file
        mode: Either 'train' or 'inference'
        config_name: Configuration name for note filtering

    Returns:
        Prepared notes dataframe
    """
    merged_notes = pd.read_parquet(notes_data_path)
    merged_notes['mrn'] = pd.to_numeric(merged_notes['mrn'], errors='coerce').astype(int)

    # Define procedure types based on mode
    if mode == 'inference':
        merged_notes.drop(columns=['EPIC_FLAG', 'Cosigner'], inplace=True, errors='ignore')
        proc_name = ['PROGRESS', 'CONSULT', 'H&P', 'TELEPHONE EN']
        procedure_exclude = ['TELEPHONE EN']
    else:
        proc_name = [
            'Clinic Note', 'Letter', 'History & Physical Note',
            'Consultation Note', 'Clinic Note (Non-dictated)'
        ]
        procedure_exclude = []

    # Filter notes by medical oncologists and procedure type
    config_list = [
        'mostRecentVisit-medOnc-ConsultLetterClinic',
        'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic',
        'firstVisitOnly-medOnc-ConsultLetterClinic',
        'firstTreatmentOnly-medOnc-ConsultLetterClinic'
    ]

    if any(x in config_name for x in config_list):
        med_oncs = list(set(aliasDictionary.values()))
        # if mode == 'train':
        merged_notes = merged_notes.loc[
            (merged_notes['processed_physician_name'].isin(med_oncs)) &
            (merged_notes['Observations.ProcName'].isin(proc_name))
        ].copy()

        merged_notes = _aggregate_notes_by_date(merged_notes, mode)
        merged_notes = _apply_note_configuration(merged_notes, config_name, procedure_exclude)
    else:
        raise NotImplementedError(f"Configuration '{config_name}' not implemented")

    # Convert stats columns to string
    for col in ['stats_physician', 'stats_dictated_by', 'stats_note_type']:
        merged_notes[col] = merged_notes[col].astype(str)

    return merged_notes


def _aggregate_notes_by_date(merged_notes: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Aggregate multiple notes from the same date into single entries."""
    if mode == 'train':
        merged_notes['note'] = (
            merged_notes['Observations.ProcName'] + ':\n' +
            merged_notes['clinical_notes']
        )
    elif mode == 'inference':
        merged_notes['note'] = merged_notes['clinical_notes']

    merged_notes = merged_notes.groupby(['mrn', 'processed_date']).agg(
        processed_note=('note', lambda x: '\n\n'.join(x)),
        max_epr_date=('epr_date', 'max'),
        stats_physician=('processed_physician_name', 'unique'),
        stats_dictated_by=('dictated_by', 'unique'),
        stats_note_type=('Observations.ProcName', 'unique')
    ).reset_index()

    merged_notes.rename(columns={"processed_note": "note"}, inplace=True)
    merged_notes['processed_date'] = merged_notes['processed_date'].dt.date
    merged_notes['processed_date'] = merged_notes['processed_date'].astype('<M8[ns]')

    return merged_notes


def _apply_note_configuration(
    merged_notes: pd.DataFrame,
    config_name: str,
    procedure_exclude: list
) -> pd.DataFrame:
    """Apply configuration-specific note processing."""
    if any(x in config_name for x in [
        'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic',
        'firstTreatmentOnly-medOnc-ConsultLetterClinic'
    ]):
        merged_notes = _append_first_note(merged_notes, procedure_exclude)

    if 'firstVisitOnly-medOnc-ConsultLetterClinic' in config_name:
        merged_notes = _keep_first_visit_only(merged_notes)

    return merged_notes


def _append_first_note(
    merged_notes: pd.DataFrame,
    procedure_exclude: list
) -> pd.DataFrame:
    """Append the patient's first note to all subsequent notes."""
    merged_notes.sort_values(by='processed_date', inplace=True)

    # Filter notes if needed
    if procedure_exclude:
        filtered_notes = merged_notes[
            ~merged_notes['stats_note_type'].apply(
                lambda x: any(exc in x for exc in procedure_exclude)
            )
        ]
    else:
        filtered_notes = merged_notes.copy()

    # Get first note for each patient
    first_note = filtered_notes.groupby(['mrn'])['note'].first(skipna=False).reset_index(name='first_note')
    merged_notes = merged_notes.merge(first_note, on="mrn")

    # Append first note to subsequent notes
    merged_notes["appended_note"] = np.where(
        merged_notes["note"] == merged_notes["first_note"],
        merged_notes["note"],
        merged_notes["first_note"] + "\n\n" + merged_notes["note"]
    )

    merged_notes = merged_notes[[
        'mrn', 'processed_date', 'max_epr_date', 'appended_note',
        'stats_physician', 'stats_dictated_by', 'stats_note_type'
    ]]
    merged_notes.rename(columns={'appended_note': 'note'}, inplace=True)

    return merged_notes


def _keep_first_visit_only(merged_notes: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first note for each patient."""
    merged_notes.sort_values(by='processed_date', inplace=True)
    merged_notes = merged_notes.groupby('mrn')[[
        'max_epr_date', 'processed_date', 'note',
        'stats_physician', 'stats_dictated_by', 'stats_note_type'
    ]].first(skipna=False).reset_index()

    return merged_notes


def anchor_notes_to_treatments(
    df_treat: pd.DataFrame,
    merged_notes: pd.DataFrame,
    lookback_window: int
) -> pd.DataFrame:
    """
    Anchor clinical notes to treatment dates within specified time window.

    Args:
        df_treat: Treatment dataframe
        merged_notes: Notes dataframe
        lookback_window: Days to look back for notes before treatment

    Returns:
        Dataframe with anchored notes
    """
    df_treat = combine_meas_to_main_data(
        main=df_treat,
        meas=merged_notes,
        main_date_col='treatment_date',
        meas_date_col='processed_date',
        time_window=(-lookback_window, 0),
        stats=['last']
    )

    # Remove '_LAST' suffix from column names
    df_treat.rename(
        columns=lambda x: x.replace('_LAST', '') if x.endswith('_LAST') else x,
        inplace=True
    )

    return df_treat


def filter_and_validate_data(
    df_treat: pd.DataFrame,
    mode: str,
    test_end_date: str = None
) -> pd.DataFrame:
    """
    Apply data quality filters and validation checks.

    Args:
        df_treat: Treatment dataframe with anchored notes
        mode: Either 'train' or 'inference'
        test_end_date: End date for test period (train mode only)

    Returns:
        Filtered and validated dataframe
    """
    # Drop rows without notes
    df_treat = df_treat.loc[~df_treat['note'].isna()]

    if mode == 'train':
        df_treat = _apply_training_filters(df_treat, test_end_date)

    # Prepare target columns
    cols = df_treat.columns
    keep_cols = cols[cols.str.contains('target') & ~cols.str.contains('date')].tolist()
    exclude_cols = (
        [f'target_{col}' for col in SYMP_COLS] +
        [f'target_{col}_change' for col in SYMP_COLS] +
        ['target_CTAS_score', 'target_CEDIS_complaint']
    )
    target_cols = [
        col for col in keep_cols
        if col not in exclude_cols and 'max' not in col and 'min' not in col
    ]

    # Convert target columns to appropriate types
    df_treat[target_cols] = df_treat[target_cols].replace({'False': 0, 'True': 1})
    df_treat[target_cols] = df_treat[target_cols].astype("Int64")
    df_treat.loc[:, target_cols] = df_treat.loc[:, target_cols].fillna(-1)

    # Drop samples with no targets
    df_treat = drop_samples_with_no_targets(df_treat, target_cols, missing_val=-1)

    return df_treat, target_cols


def _apply_training_filters(
    df_treat: pd.DataFrame,
    test_end_date: str
) -> pd.DataFrame:
    """Apply filters specific to training mode."""
    # Filter by test end date
    test_end_date = pd.to_datetime(test_end_date)
    df_treat = df_treat.loc[df_treat['treatment_date'] <= test_end_date].copy()

    # Remove entries with invalid EPR dates
    df_treat['max_epr_date'] = pd.to_datetime(df_treat['max_epr_date'])
    df_treat = df_treat.loc[
        (df_treat['max_epr_date'].dt.year >= 2005) &
        (df_treat['max_epr_date'].dt.year <= 2022)
    ]

    # Remove records with potential leakage (EPR date after treatment date)
    df_treat = df_treat.loc[
        pd.to_datetime(df_treat['treatment_date'], utc=True) >
        pd.to_datetime(df_treat['max_epr_date'], utc=True)
    ]

    return df_treat


def prepare_final_features(
    df_treat: pd.DataFrame,
    opis_df: pd.DataFrame,
    config_name: str,
    mode: str,
    add_tabular_to_note: bool,
    clinical_bench: bool
) -> pd.DataFrame:
    """
    Prepare final features for modeling.

    Args:
        df_treat: Treatment dataframe
        opis_df: OPIS dataframe for tabular features
        config_name: Configuration name
        mode: Either 'train' or 'inference'
        add_tabular_to_note: Whether to append tabular data to notes
        clinical_bench: Whether using clinical benchmark mode

    Returns:
        Dataframe with prepared features
    """
    # Drop drug features that were never used
    if mode == 'train':
        df_treat = drop_unused_drug_features(df_treat)

    # Fill missing data that can be filled heuristically
    df_treat = fill_missing_data_heuristically(df_treat)

    # Collapse rare morphology and cancer sites into 'Other' category
    if mode == 'train':
        cancer_site_morphology_cols = df_treat.columns[
            df_treat.columns.str.contains("cancer_site_|morphology_")
        ]
        df_treat[cancer_site_morphology_cols] = (
            df_treat[cancer_site_morphology_cols] == 2
        )
        df_treat = collapse_rare_categories(df_treat, catcols=['cancer_site', 'morphology'])

    # Create tabular data converted to note (must be done after preprocessing above)
    first_treatment_flag = 1 if 'firstTreatmentOnly' in config_name else 0
    df_treat = add_tabular_data_to_note(df_treat, opis_df, first_treatment_flag, mode)

    if add_tabular_to_note:
        df_treat['note'] = df_treat['note'] + '\n\n' + df_treat['sentencized_tabular_data']

    if clinical_bench:
        df_treat['original_note'] = df_treat['note']
        df_treat['note'] = df_treat['sentencized_tabular_data']

    # Drop highly missing features and create missingness indicators
    keep_cols = df_treat.columns[df_treat.columns.str.contains('target_')]
    df_treat = drop_highly_missing_features(df_treat, missing_thresh=80, keep_cols=keep_cols)
    df_treat = get_missingness_features(df_treat)

    return df_treat


def rename_symptom_columns(col: str, symp_cols: list) -> str:
    """
    Rename symptom columns to standardized format.

    Args:
        col: Column name
        symp_cols: List of symptom column identifiers

    Returns:
        Renamed column
    """
    for symp in symp_cols:
        if symp in col:
            if symp == "ecog":
                return col.replace(symp, "patient_ecog")
            elif symp == "lack_of_appetite":
                return col.replace(symp, "esas_appetite")
            else:
                return col.replace(symp, f"esas_{symp}")
    return col


def anchor_note_to_treatment(
    mode: str,
    notes_data_path: str,
    treatment_data_path: str,
    opis_data_path: str,
    save_dir: str,
    config_name: str,
    test_end_date: str,
    lookback_window: int,
    add_tabular_to_note: bool,
    treatment_dates_path: str = None,
    train_test_anchored_df_path: str = None,
    clinical_bench: bool = False
):
    """
    Main pipeline to anchor clinical notes to treatment dates.

    Args:
        mode: 'train' or 'inference'
        notes_data_path: Path to notes parquet file
        treatment_data_path: Path to treatment parquet file
        opis_data_path: Path to OPIS parquet file
        save_dir: Directory to save output
        config_name: Configuration name for note anchoring strategy
        test_end_date: End date for test period
        lookback_window: Days to look back for notes before treatment
        add_tabular_to_note: Whether to append tabular data to notes
        treatment_dates_path: Path to treatment dates (inference only)
        train_test_anchored_df_path: Path to training reference data (inference only)
        clinical_bench: Whether using clinical benchmark mode
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load and prepare data
    df_treat = load_and_prepare_treatment_data(
        treatment_data_path,
        mode,
        train_test_anchored_df_path,
        treatment_dates_path
    )

    merged_notes = load_and_prepare_notes(notes_data_path, mode, config_name)

    # Filter treatment data to patients with notes
    df_treat = df_treat.loc[df_treat['mrn'].isin(merged_notes['mrn'].unique())].copy()

    # Filter by test end date if in train mode
    if mode == 'train':
        test_end_date_dt = pd.to_datetime(test_end_date)
        df_treat = df_treat.loc[df_treat['treatment_date'] <= test_end_date_dt].copy()

    # Handle first treatment only configuration
    if 'firstTreatmentOnly-medOnc-ConsultLetterClinic' in config_name:
        df_treat.sort_values(by='treatment_date', inplace=True)
        df_treat = df_treat.groupby(['mrn']).first(skipna=False).reset_index()

    # Anchor notes to treatments
    df_treat = anchor_notes_to_treatments(df_treat, merged_notes, lookback_window)

    # Filter and validate
    df_treat, target_cols = filter_and_validate_data(df_treat, mode, test_end_date)

    # Prepare final features
    opis_df = pd.read_parquet(opis_data_path)
    df_treat = prepare_final_features(
        df_treat,
        opis_df,
        config_name,
        mode,
        add_tabular_to_note,
        clinical_bench
    )

    # Final cleanup and preparation
    df_treat.drop('assessment_date', axis=1, inplace=True, errors='ignore')
    df_treat = df_treat.reset_index(drop=True)

    # Create note index
    df_treat['note_index'] = pd.factorize(df_treat['note'])[0]

    # Rename symptom columns
    target_cols_renamed = [rename_symptom_columns(col, SYMP_COLS) for col in target_cols]
    df_treat = df_treat.rename(columns=dict(zip(target_cols, target_cols_renamed)))

    # Adjust female column
    df_treat['female'] = df_treat['female'].replace(-1, 0)

    # Prepare output columns
    cols = df_treat.columns
    cols_no_target = [col for col in cols if 'target' not in col]

    # Final data cleaning
    df_treat['treatment_date'] = pd.to_datetime(df_treat['treatment_date']).dt.date

    # date_cols = [col for col in df_treat.columns if 'date' in col.lower()]
    # target_cols_all = [col for col in df_treat.columns if 'target' in col.lower()]
    # cols_to_exclude = (
    #     date_cols + target_cols_all + LAB_CHANGE_COLS +
    #     SYMP_CHANGE_COLS + LAB_COLS + SYMP_COLS
    # )
    # cols_with_nan = [
    #     col for col in df_treat.columns
    #     if col not in cols_to_exclude and df_treat[col].isna().sum() > 0
    # ]
    # df_treat.dropna(subset=cols_with_nan, inplace=True)

    cols_not_imputed = ['line_of_therapy', 'height', 'weight', 'intent', 'body_surface_area']
    df_treat.dropna(subset=cols_not_imputed, inplace=True)

    # Save output
    suffix = "note_tabular" if add_tabular_to_note else "note"
    outfile = os.path.join(save_dir, f"{suffix}_anchored_{config_name}.csv")
    df_treat[cols_no_target + target_cols_renamed].to_csv(outfile, index=False)

    logger.info(f"Successfully saved output to: {outfile}")
    logger.info(f"Output shape: {df_treat.shape}")


def main():
    """Parse command line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Anchor clinical notes to treatment dates for ML pipeline"
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "inference"],
        help="Mode: 'train' or 'inference'"
    )
    parser.add_argument(
        "notes_data_path",
        type=str,
        help="Path to notes parquet file"
    )
    parser.add_argument(
        "treatment_data_path",
        type=str,
        help="Path to treatment parquet file"
    )
    parser.add_argument(
        "opis_data_path",
        type=str,
        help="Path to OPIS parquet file"
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Directory to save output"
    )
    parser.add_argument(
        "config_name",
        type=str,
        help="Configuration name for note anchoring strategy"
    )
    parser.add_argument(
        "test_end_date",
        type=str,
        help="End date for test period (YYYY-MM-DD)"
    )
    parser.add_argument(
        "lookback_window",
        type=int,
        help="Lookback window in days for notes to be anchored"
    )
    parser.add_argument(
        "add_tabular_to_note",
        type=int,
        help="Add tabular data to note? (0 or 1)"
    )
    parser.add_argument(
        "--treatment_dates_path",
        type=str,
        default=None,
        help="Path to treatment dates parquet file (inference only)"
    )
    parser.add_argument(
        "--train_test_anchored_df_path",
        type=str,
        default=None,
        help="Path to train/test anchored dataframe (inference only)"
    )
    parser.add_argument(
        "--clinical_bench",
        type=int,
        default=0,
        help="Use clinical benchmark mode? (0 or 1)"
    )

    args = parser.parse_args()

    anchor_note_to_treatment(
        mode=args.mode,
        notes_data_path=args.notes_data_path,
        treatment_data_path=args.treatment_data_path,
        opis_data_path=args.opis_data_path,
        save_dir=args.save_dir,
        config_name=args.config_name,
        test_end_date=args.test_end_date,
        lookback_window=args.lookback_window,
        add_tabular_to_note=bool(args.add_tabular_to_note),
        treatment_dates_path=args.treatment_dates_path,
        train_test_anchored_df_path=args.train_test_anchored_df_path,
        clinical_bench=bool(args.clinical_bench)
    )


if __name__ == "__main__":
    main()