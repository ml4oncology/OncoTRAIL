import os
import pandas as pd
import argparse
from ml_common.anchor import combine_feat_to_main_data
from ml_common.engineer import (
    get_change_since_prev_session,
    get_missingness_features,
    collapse_rare_categories
)
from ml_common.filter import (
    drop_samples_with_no_targets, 
    drop_unused_drug_features, 
    drop_highly_missing_features,
    keep_only_one_per_week
)
from ml_common.constants import SYMP_COLS

from preduce.acu.label import get_event_labels
from preduce.symp.label import (get_symptom_labels, convert_to_binary_symptom_labels)
from ml_common.prep import fill_missing_data_heuristically
from preduce.filter import indicate_immediate_events
# from preduce.prepare.prep import fill_missing_data

from llm_notes_classification.prep.label import get_ctcae_labels
from llm_notes_classification.prep.add_tabular_to_note import (
    add_tabular_data_to_note
)

# refactor this to also generate inference data
# find mrns which have their first entry in EPIC for inference data
# Question: can you just run this code as is using the patients with epic notes or are there notes that will be missing?

import importlib.util
spec = importlib.util.spec_from_file_location("phys_names", "/cluster/projects/gliugroup/2BLAST/data/info/phys_names.py")
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)
aliasDictionary = constants.aliasDictionary

def anchor_note_to_treatment(mode,
                            notes_data_path, 
                            treatment_data_path, 
                            ed_visit_data_path,
                            symptom_data_path, 
                            opis_data_path,
                            save_dir, config_name,
                            test_end_date, 
                            lookback_window,
                            add_tabular_to_note):
    """
        Anchor the note to treatment date depending on specified configuration.

        notes_data_path: file path of the notes data
        treatment_data_path: file path of the treatment data frame
        ed_visit_data_path: data path of the ED visit data frame
        symptom_data_path: data path of the symptom data frame
        opis_data_path: data path of the opis data frame
        save_dir: directory path where processed data frame will be saved
        config_name: configuration name for how to anchor note to treatment date
        test_end_date: ending date for the test time period (and end date of the study period)
        lookback_window: lookback window for the notes to be anchored to treatment date
        add_tabular_to_note: whether to add tabular data to the note
    """
    os.makedirs(save_dir, exist_ok=True)

    # load treatment-centered data frame
    df_treat = pd.read_parquet(f'{treatment_data_path}', engine='pyarrow', use_nullable_dtypes = True)
    df_treat.rename(columns={'target_ED_30d': 'target_ED_visit'}, inplace=True)

    df_treat['assessment_date'] = df_treat['treatment_date']
    df_treat['treatment_date'] = pd.to_datetime(df_treat['treatment_date'])
    # keep only the first treatment session of a given week
    df_treat = keep_only_one_per_week(df_treat)

    # get the change in measurement since previous assessment
    df_treat = get_change_since_prev_session(df_treat)
    
    # consider all events
    # process ED_visit
    # load ed target data frame
    df_target_ed = pd.read_parquet(f'{ed_visit_data_path}', engine='pyarrow', use_nullable_dtypes = True)
    df_treat = get_event_labels(df_treat, df_target_ed, event_name='ED_visit', 
                                extra_cols=['CTAS_score', 'CEDIS_complaint'])

    # exclude immediate events
    df_treat = indicate_immediate_events(df_treat, targ_cols=['target_ED_visit'], 
                                         date_cols=['target_ED_visit_date'])

    # process symptom targets
    target_pt_increases = [1, 3]
    # load symp target data frame
    df_target_symp = pd.read_parquet(f'{symptom_data_path}')
    df_treat = get_symptom_labels(df_treat, df_target_symp)
    for pt_increase in target_pt_increases:
        scoring_map = {symp: pt_increase for symp in SYMP_COLS}
        df_treat = convert_to_binary_symptom_labels(df_treat, scoring_map=scoring_map)

    # exclude immediate events
    date_cols = [f'target_{symp}_survey_date' for symp in SYMP_COLS]
    for pt in target_pt_increases:
        targ_cols = [f'target_{symp}_{pt}pt_change' for symp in SYMP_COLS]
        df_treat = indicate_immediate_events(df_treat, targ_cols, date_cols)


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
        mrns_epic = merged_notes.loc[merged_notes['EPIC_FLAG'] == 1]['mrn'].unique()
        mrns_epr = merged_notes.loc[merged_notes['EPIC_FLAG'] == 0]['mrn'].unique()
        # find mrns in mrns_epic but not in mrns_epr
        mrns_to_keep = list(set(mrns_epic) - set(mrns_epr))
        merged_notes = merged_notes.loc[merged_notes['mrn'].isin(mrns_to_keep) & (merged_notes['EPIC_FLAG'] == 1)].copy()
        
        # drop EPIC_FLAG, Cosigner columns
        merged_notes.drop(columns=['EPIC_FLAG', 'Cosigner'], inplace=True)
        procName = ['BMT Planning', 'CCRT Note', 'Code Doc', 'MEDICAL STUD', 'SubjObj', 'PROGRESS', 'CONSULT', 'H&P', 'Teleconsult']

    else:
        procName = ['Clinic Note', 'Letter', 'History & Physical Note', 'Consultation Note', 'Clinic Note (Non-dictated)']

    if any(x in config_name for x in config_list):
        # only consider notes written by a medical oncologist 
        # only consider consultation, letter, clinic notes
        medOncs = list(set(aliasDictionary.values()))
        merged_notes = merged_notes.loc[(merged_notes['processed_physician_name'].isin(medOncs)) &\
                                       (merged_notes['Observations.ProcName'].isin(procName))].copy()
        
        # merge records of patient on the same day
        # take the maximum of the EPR dates

        if mode == 'train':
            merged_notes['note'] = merged_notes['Observations.ProcName'] + ':\n' + merged_notes['clinical_notes']
        # merged_notes = merged_notes.groupby(['mrn','processed_date']).agg(
        #                 processed_note=('note', lambda x: '\n'.join(x)),
        #                 max_epr_date=('EPRDate', 'max')).reset_index()
        # add physician name and note type for statistics tracking
        merged_notes = merged_notes.groupby(['mrn','processed_date']).agg(
                        processed_note=('note', lambda x: '\n'.join(x)),
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
            first_note = merged_notes.groupby(['mrn'])['note'].first(skipna=False).reset_index(name='first_note')
            # append the first note
            merged_notes = merged_notes.merge(first_note, on="mrn")
            merged_notes['appended_note'] = merged_notes.apply(
                lambda x: x['note'] if x['note'] == x['first_note'] else '\n'.join([x['first_note'], x['note']]),
                axis=1,
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
        df_treat = df_treat.loc[df_treat['treatment_date'] <= test_end_date].copy()

    # attach notes to treatment dataframe
    df_treat = combine_feat_to_main_data(
        main=df_treat, feat=merged_notes, main_date_col='treatment_date', feat_date_col='processed_date', 
        time_window=(-lookback_window,0)
        )

    # drop rows that don't have any note data
    df_treat = df_treat.loc[~df_treat.note.isna()]
    
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
    df_treat.loc[:, target_cols].fillna(value=-1, inplace=True)

    df_treat[target_cols] = df_treat[target_cols].replace({'False': 0, 'True': 1})
    df_treat[target_cols] = df_treat[target_cols].astype(int)
    df_treat = drop_samples_with_no_targets(df_treat, target_cols, missing_val=-1) 

    # drop drug features that were never used
    df_treat = drop_unused_drug_features(df_treat)

    # fill missing data that can be filled heuristically
    df_treat = fill_missing_data_heuristically(df_treat)

    if add_tabular_to_note:
        opis_df = pd.read_parquet(f'{opis_data_path}')
        if 'firstTreatmentOnly' in config_name:
            df_treat = add_tabular_data_to_note(df_treat, opis_df, 1)
        else:
            df_treat = add_tabular_data_to_note(df_treat, opis_df, 0)

    # drop features with high missingness
    keep_cols = df_treat.columns[df_treat.columns.str.contains('target_')]
    df_treat = drop_highly_missing_features(df_treat, missing_thresh=80, keep_cols=keep_cols)

    # create missingness features
    df_treat = get_missingness_features(df_treat)

    # collapse rare morphology and cancer sites into 'Other' category
    df_treat = collapse_rare_categories(df_treat, catcols=['cancer_site', 'morphology'])

    # drop assessment_date column
    df_treat.drop('assessment_date', axis=1, inplace=True)

    # save dataframe with anchored note
    cols = df_treat.columns
    cols_no_target = [col for col in cols if 'target' not in col] + ['note_index']

    df_treat = df_treat.reset_index(drop=True)

    # create a new column that serves as an index for each unique note
    df_treat['note_index'] = pd.factorize(df_treat['note'])[0]

    if add_tabular_to_note:
        df_treat[cols_no_target + target_cols].to_csv(f"{save_dir}/note_tabular_anchored_{config_name}.csv")
    else:
        df_treat[cols_no_target + target_cols].to_csv(f"{save_dir}/note_anchored_{config_name}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode: 'train' or 'inference'", type=str, choices=["train", "inference"])  # mode
    parser.add_argument("notes_data_path", help = "data file path", type = str) # notes data file path
    parser.add_argument("treatment_data_path", help = "file path of treatment data", type = str) # treatment data file path
    parser.add_argument("ed_visit_data_path", help = "file path of ED visit data", type = str) # ed visit data file path
    parser.add_argument("symptom_data_path", help = "file path of symptom data", type = str) # file path of symptom data
    parser.add_argument("opis_data_path", help = "opis file path", type = str) # opis file path
    parser.add_argument("save_dir", help = "save directory", type = str) # save directory
    parser.add_argument("config_name", help = "configuration name", type = str) # configuration name
    parser.add_argument("test_end_date", help = "end date for test period", type = str) # test end date
    parser.add_argument("lookback_window", help = "lookback window for notes to be anchored", type = int) # lookback window
    parser.add_argument("add_tabular_to_note", help = "add tabular to note?", type = int) # add tabular to note?
    args = parser.parse_args()

    anchor_note_to_treatment(args.mode,
                             args.notes_data_path, 
                             args.treatment_data_path, 
                             args.ed_visit_data_path,
                             args.symptom_data_path, 
                             args.opis_data_path,
                             args.save_dir, 
                             args.config_name, 
                             args.test_end_date,
                             args.lookback_window,
                             args.add_tabular_to_note)
