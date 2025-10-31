import os
import pandas as pd
import argparse
import numpy as np
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
from make_clinical_dataset.shared.constants import SYMP_COLS
from make_clinical_dataset.epr.prep import fill_missing_data_heuristically
from llm_notes_classification.prep.add_tabular_to_note import (
    add_tabular_data_to_note
)

# refactor this to also generate inference data
# find mrns which have their first entry in EPIC for inference data

import importlib.util
spec = importlib.util.spec_from_file_location("phys_names", "/cluster/projects/gliugroup/2BLAST/data/info/phys_names.py")
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)
aliasDictionary = constants.aliasDictionary

def anchor_note_to_treatment(mode,
                            notes_data_path, 
                            treatment_data_path,  
                            opis_data_path,
                            save_dir, config_name,
                            test_end_date, 
                            lookback_window,
                            add_tabular_to_note,
                            treatment_dates_path=None,
                            clinical_bench=0):
    """
        Anchor the note to treatment date depending on specified configuration.
        mode: train or inference
        notes_data_path: file path of the notes data
        treatment_data_path: file path of the treatment data frame
        opis_data_path: data path of the opis data frame
        save_dir: directory path where processed data frame will be saved
        config_name: configuration name for how to anchor note to treatment date
        test_end_date: ending date for the test time period (and end date of the study period)
        lookback_window: lookback window for the notes to be anchored to treatment date
        add_tabular_to_note: whether to add tabular data to the note
        treatment_dates_path: file path of the treatment dates data frame (only needed for inference mode)
    """
    os.makedirs(save_dir, exist_ok=True)

    # load treatment-centered data frame
    df_treat = pd.read_parquet(f'{treatment_data_path}', engine='pyarrow', use_nullable_dtypes = True)
    # adjust sex column if mode is inference
    if mode == 'inference':
        # rename sex column to female
        if 'sex' in df_treat.columns:
            df_treat.rename(columns={'sex': 'female'}, inplace=True)
        
        # if the value of "female" column is "female", set to 1, 0 for male and -1 otherwise
        df_treat['female'] = np.where(
            df_treat['female'] == 'female', 1,
            np.where(
                df_treat['female'] == 'male', 0, -1
            )
        )
    
    # make the values of the intent column uppercase
    df_treat['intent'] = df_treat['intent'].str.upper()

    # drop these columns if they exist in df_treat
    cols_to_drop = ['target_ED_note', 'target_ED_60d', 'target_ED_90d', 'drug_name', 'postal_code',
                    'target_hemoglobin_min', 'target_platelet_min', 'target_neutrophil_min',
                    'target_creatinine_max', 'target_alanine_aminotransferase_max',
                    'target_aspartate_aminotransferase_max', 'target_total_bilirubin_max']
    df_treat = df_treat.drop(columns=cols_to_drop, errors='ignore')

    df_treat.rename(columns={'target_ED_30d': 'target_ED_visit'}, inplace=True)

    if 'treatment_date' not in df_treat.columns:
        df_treat['treatment_date'] = df_treat['assessment_date']
    else:
        df_treat['assessment_date'] = df_treat['treatment_date']

    df_treat['treatment_date'] = pd.to_datetime(df_treat['treatment_date'])
    df_treat['assessment_date'] = pd.to_datetime(df_treat['assessment_date'])

    # compute visit month features if they don't exist
    if not any(col in df_treat.columns for col in ['visit_month_sin', 'visit_month_cos']):
        df_treat = get_visit_month_feature(df_treat)

    # keep only the first treatment session of a given week
    df_treat = keep_only_one_per_week(df_treat)

    # get the change in measurement since previous assessment
    df_treat = get_change_since_prev_session(df_treat)

    # if inference mode, load treatment dates data frame
    if mode == 'inference':
        assert treatment_dates_path is not None, "treatment_dates_path must be provided in inference mode"
        df_treat_dates = pd.read_parquet(f'{treatment_dates_path}', engine='pyarrow', use_nullable_dtypes = True)
        df_treat = df_treat.merge(df_treat_dates, on=['mrn','treatment_date', 'assessment_date'], how='inner')

    # if first treatment only, select the first row for every mrn and treatment_date
    if 'firstTreatmentOnly-medOnc-ConsultLetterClinic' in config_name:
        # sort by treatment date
        df_treat.sort_values(by='treatment_date', inplace=True)
        # select the first row for every mrn and treatment_date
        df_treat = df_treat.groupby(['mrn']).first(skipna=False).reset_index()

    # load notes file
    merged_notes = pd.read_parquet(f'{notes_data_path}')
    merged_notes['mrn'] = pd.to_numeric(merged_notes['mrn'], errors='coerce')
    merged_notes['mrn'] = merged_notes['mrn'].astype(int)

    config_list = ['mostRecentVisit-medOnc-ConsultLetterClinic', 
                       'mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic', 
                       'firstVisitOnly-medOnc-ConsultLetterClinic',
                       'firstTreatmentOnly-medOnc-ConsultLetterClinic']

    if mode == 'inference':
        # find the unique mrns when EPIC_FLAG is 1
        # do this outside
        # mrns_epic = merged_notes.loc[merged_notes['EPIC_FLAG'] == 1]['mrn'].unique()
        # mrns_epr = merged_notes.loc[merged_notes['EPIC_FLAG'] == 0]['mrn'].unique()
        # # find mrns in mrns_epic but not in mrns_epr
        # mrns_to_keep = list(set(mrns_epic) - set(mrns_epr))
        # merged_notes = merged_notes.loc[merged_notes['mrn'].isin(mrns_to_keep) & (merged_notes['EPIC_FLAG'] == 1)].copy()

        # drop EPIC_FLAG, Cosigner columns
        merged_notes.drop(columns=['EPIC_FLAG', 'Cosigner'], inplace=True)
        proc_name = ['PROGRESS', 'CONSULT', 'H&P', 'TELEPHONE EN']
        procedure_exclude = ['TELEPHONE EN']

    else:
        proc_name = ['Clinic Note', 'Letter', 'History & Physical Note', 
                    'Consultation Note', 'Clinic Note (Non-dictated)']
        procedure_exclude = []

    if any(x in config_name for x in config_list):
        # only consider notes written by a medical oncologist 
        # only consider consultation, letter, clinic notes
        med_oncs = list(set(aliasDictionary.values()))
        merged_notes = merged_notes.loc[(merged_notes['processed_physician_name'].isin(med_oncs)) &\
                                       (merged_notes['Observations.ProcName'].isin(proc_name))].copy()
        
        # merge records of patient on the same day
        # take the maximum of the EPR dates

        if mode == 'train':
            merged_notes['note'] = merged_notes['Observations.ProcName'] + ':\n' + merged_notes['clinical_notes']
        elif mode == 'inference':
            merged_notes['note'] = merged_notes['clinical_notes']

        merged_notes = merged_notes.groupby(['mrn','processed_date']).agg(
                        processed_note=('note', lambda x: '\n\n'.join(x)),
                        max_epr_date=('epr_date', 'max'),
                        stats_physician=('processed_physician_name','unique'),
                        stats_dictated_by=('dictated_by','unique'),
                        stats_note_type=('Observations.ProcName','unique')).reset_index()
        merged_notes.rename(columns={"processed_note": "note"}, inplace=True)
        merged_notes['processed_date'] = merged_notes['processed_date'].dt.date
        merged_notes['processed_date'] = merged_notes['processed_date'].astype('<M8[ns]')

        if any(x in config_name for x in ['mostRecentVisit-appendFirst-medOnc-ConsultLetterClinic',
                                             'firstTreatmentOnly-medOnc-ConsultLetterClinic']):
            # get the first note
            merged_notes.sort_values(by='processed_date', inplace=True)
            if procedure_exclude:  # only apply filter if list is not empty
                filtered_notes = merged_notes[~merged_notes['stats_note_type'].isin(procedure_exclude)]
            else:
                filtered_notes = merged_notes.copy()
            first_note = filtered_notes.groupby(['mrn'])['note'].first(skipna=False).reset_index(name='first_note')
            # append the first note
            merged_notes = merged_notes.merge(first_note, on="mrn")

            merged_notes["appended_note"] = np.where(
                merged_notes["note"] == merged_notes["first_note"],
                merged_notes["note"],
                merged_notes["first_note"] + "\n\n" + merged_notes["note"]
            )
            # retain only columns of interest
            merged_notes = merged_notes[[
                'mrn', 'processed_date', 'max_epr_date', 'appended_note',
                'stats_physician', 'stats_dictated_by', 'stats_note_type'
            ]]
            merged_notes.rename(columns={'appended_note': 'note'}, inplace=True)
        
        if any(x in config_name for x in ['firstVisitOnly-medOnc-ConsultLetterClinic']):
            # keep only the first note
            merged_notes.sort_values(by='processed_date', inplace=True)
            merged_notes = merged_notes.groupby('mrn')[['max_epr_date','processed_date',
                                                      'note','stats_physician','stats_dictated_by',
                                                      'stats_note_type']].first(skipna=False).reset_index()

    else:
        raise Exception("Not implemented yet.")

    # filter the treatment-centered data frame
    df_treat = df_treat.loc[df_treat['mrn'].isin(merged_notes['mrn'].unique())].copy()
    # filter out records if treatment date is past the end date of the study period
    # note: there is no need to cap the data at the start date since the notes data set
    # only contains data starting from the start date. if no notes are anchored to treatment 
    # data, they will be dropped

    if mode == 'train':
        test_end_date = pd.to_datetime(test_end_date)
        df_treat = df_treat.loc[df_treat['treatment_date'] <= test_end_date].copy()

    # df_treat["target_ED_visit"] = df_treat["target_ED_visit"].replace(-1, pd.NA).astype("boolean")

    merged_notes["stats_physician"] = merged_notes["stats_physician"].astype(str)
    merged_notes["stats_dictated_by"] = merged_notes["stats_dictated_by"].astype(str)
    merged_notes["stats_note_type"] = merged_notes["stats_note_type"].astype(str)

    # attach notes to treatment dataframe
    df_treat = combine_meas_to_main_data(
        main=df_treat, meas=merged_notes, main_date_col='treatment_date', meas_date_col='processed_date', 
        time_window=(-lookback_window,0), stats=['last']
        )
    
    # df_treat = merge_closest_measurements(
    #     main=df_treat, meas=merged_notes, main_date_col='treatment_date', meas_date_col='processed_date',
    #     direction='backward', time_window=(-lookback_window,0), merge_individually=False, include_meas_date=True
    # )

    # look for columns with suffix _LAST in df_treat then rename to remove the suffix
    df_treat.rename(columns=lambda x: x.replace('_LAST', '') if x.endswith('_LAST') else x, inplace=True)

    # drop rows that don't have any note data
    df_treat = df_treat.loc[~df_treat['note'].isna()]
    
    if mode == 'train':
        df_treat['max_epr_date'] = pd.to_datetime(df_treat['max_epr_date'])

        # remove entries in the data with wrong EPR dates
        df_treat = df_treat.loc[(df_treat['max_epr_date'].dt.year >= 2005) & 
                                (df_treat['max_epr_date'].dt.year <= 2022)]

        # remove records with potential leakage -- EPR date is after the treatment date
        df_treat = df_treat.loc[pd.to_datetime(df_treat['treatment_date'], utc=True) > 
                                pd.to_datetime(df_treat['max_epr_date'], utc=True)]

    cols = df_treat.columns
    # drop rows where all targets are unavailable
    keep_cols = cols[cols.str.contains('target') & ~cols.str.contains('date')].tolist()
    exclude_cols = [f'target_{col}' for col in SYMP_COLS] +\
                        [f'target_{col}_change' for col in SYMP_COLS] +\
                            ['target_CTAS_score', 'target_CEDIS_complaint']
    target_cols = [col for col in keep_cols if col not in exclude_cols]
    target_cols = [col for col in target_cols if 'max' not in col and 'min' not in col]
    
    df_treat[target_cols] = df_treat[target_cols].replace({'False': 0, 'True': 1})
    df_treat[target_cols] = df_treat[target_cols].astype("Int64")
    df_treat.loc[:, target_cols].fillna(value=-1, inplace=True)

    df_treat = drop_samples_with_no_targets(df_treat, target_cols, missing_val=-1) 

    # drop drug features that were never used
    if mode == 'train':
        df_treat = drop_unused_drug_features(df_treat)

    # fill missing data that can be filled heuristically
    df_treat = fill_missing_data_heuristically(df_treat)

    # collapse rare morphology and cancer sites into 'Other' category
    if mode == 'train':
        cancer_site_morphology_cols = df_treat.columns[df_treat.columns.str.contains("cancer_site_|morphology_")]
        df_treat[cancer_site_morphology_cols] = df_treat[cancer_site_morphology_cols] == 2
        df_treat = collapse_rare_categories(df_treat, catcols=['cancer_site', 'morphology'])

    # create tabular data converted to note 
    opis_df = pd.read_parquet(f'{opis_data_path}')
    if 'firstTreatmentOnly' in config_name:
        df_treat = add_tabular_data_to_note(df_treat, opis_df, 1, mode)
    else:
        df_treat = add_tabular_data_to_note(df_treat, opis_df, 0, mode)
    if add_tabular_to_note:
        df_treat['note'] = df_treat['note'] + '\n\n' + df_treat['sentencized_tabular_data']
    if clinical_bench:        
        df_treat['original_note'] = df_treat['note']
        df_treat['note'] = df_treat['sentencized_tabular_data']

    # drop features with high missingness
    keep_cols = df_treat.columns[df_treat.columns.str.contains('target_')]
    df_treat = drop_highly_missing_features(df_treat, missing_thresh=80, keep_cols=keep_cols)

    # create missingness features
    df_treat = get_missingness_features(df_treat)

    # drop assessment_date column
    df_treat.drop('assessment_date', axis=1, inplace=True)

    # save dataframe with anchored note
    cols = df_treat.columns
    cols_no_target = [col for col in cols if 'target' not in col] + ['note_index']

    df_treat = df_treat.reset_index(drop=True)

    # create a new column that serves as an index for each unique note
    df_treat['note_index'] = pd.factorize(df_treat['note'])[0]

    # rename esas columns
    # if an element in SYMP_COLS appears as a substring in a column name in df_treat, replace it with esas_{symptom}
    def rename_column(col: str, symp_cols: list[str]) -> str:
        """Rename a single column according to SYMP_COLS rules."""
        for symp in symp_cols:
            if symp in col:
                if symp == "ecog":
                    return col.replace(symp, "patient_ecog")
                elif symp == "lack_of_appetite":
                    return col.replace(symp, "esas_appetite")
                else:
                    return col.replace(symp, f"esas_{symp}")
        return col  # no match → keep as-is

    # Apply to df_treat
    rename_map = {col: rename_column(col, SYMP_COLS) for col in df_treat.columns}
    df_treat = df_treat.rename(columns=rename_map)

    # Apply to target_cols
    target_cols = [rename_column(col, SYMP_COLS) for col in target_cols]

    # Apply to cols_no_target
    cols_no_target = [rename_column(col, SYMP_COLS) for col in cols_no_target]

    # Adjust the female column so that if -1, set to 0
    df_treat['female'] = df_treat['female'].replace(-1, 0)

    suffix = "note_tabular" if add_tabular_to_note else "note"
    outfile = f"{save_dir}/{suffix}_anchored_{config_name}.csv"

    df_treat[cols_no_target + target_cols].to_csv(outfile, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode: 'train' or 'inference'", type=str, choices=["train", "inference"])  # mode
    parser.add_argument("notes_data_path", help = "data file path", type = str) # notes data file path
    parser.add_argument("treatment_data_path", help = "file path of treatment data", type = str) # treatment data file path
    parser.add_argument("opis_data_path", help = "opis file path", type = str) # opis file path
    parser.add_argument("save_dir", help = "save directory", type = str) # save directory
    parser.add_argument("config_name", help = "configuration name", type = str) # configuration name
    parser.add_argument("test_end_date", help = "end date for test period", type = str) # test end date
    parser.add_argument("lookback_window", help = "lookback window for notes to be anchored", type = int) # lookback window
    parser.add_argument("add_tabular_to_note", help = "add tabular to note?", type = int) # add tabular to note?
    # optional argument
    parser.add_argument("--treatment_dates_path", help = "file path of treatment dates", type = str, default=None) # test end date
    parser.add_argument("--clinical_bench", help = "use clinical bench?", type = int, default=0) # use clinical bench?
    args = parser.parse_args()

    anchor_note_to_treatment(args.mode,
                             args.notes_data_path, 
                             args.treatment_data_path,
                             args.opis_data_path,
                             args.save_dir, 
                             args.config_name, 
                             args.test_end_date,
                             args.lookback_window,
                             args.add_tabular_to_note,
                             args.treatment_dates_path,
                             args.clinical_bench)
