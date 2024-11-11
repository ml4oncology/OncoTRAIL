import pandas as pd
from tqdm import tqdm
import numpy as np
from llm_notes_classification.prep.constants import (
    CTCAE_constants, map_CTCAE_lab
)

# helper functions
def less_than_thresh(x, target, grade, constants):
    if np.isnan(x):
        return -1
    else:
        if x < constants[target][grade]:
            return 1
        else:
            return 0

def compound_thresh(look_ahead_value, baseline, target, grade, constants):
    if np.isnan(look_ahead_value):
        return -1
    else:
        if target == 'AKI':
            if look_ahead_value >\
                  constants[target][grade]*np.nanmin([baseline, constants[target]['ULN']]):
                return 1
            else:
                return 0
        else:
            if look_ahead_value >\
                  constants[target][grade]*np.nanmax([baseline, constants[target]['ULN']]):
                return 1
            else:
                return 0
def get_lookahead_lab_values(df_treat,
                             df_lab, 
                             blood_lab_list,
                             date_col_trt, 
                             date_col_lab, 
                             upper_limit):

    for quantity in blood_lab_list:

        if quantity in ['hemoglobin', 'neutrophil', 'platelet']:
            func = np.nanmin
        else:
            func = np.nanmax
        
        target = []
        for mrn, main_group in tqdm(df_treat.groupby('mrn')):
            lab_group = df_lab.query('mrn == @mrn')
            
            for idx, date in main_group[date_col_trt].items():
                earliest_date = date + pd.Timedelta(days=1)
                latest_date = date + pd.Timedelta(days=upper_limit)
                
                # check if there is a treatment date within 30 days
                mask_trt = main_group[date_col_trt].between(earliest_date, latest_date)
                if mask_trt.any():
                    # get the first value
                    lab_vals_trtdf = main_group.loc[mask_trt, quantity]
                    target_val = lab_vals_trtdf.iloc[0]

                if not mask_trt.any() or np.isnan(target_val):
                    # get the values from the lab dataframe
                    mask_lab = lab_group[date_col_lab].between(earliest_date - pd.Timedelta(days=1),
                                                                latest_date)

                    if mask_lab.any():
                        # get the minimum or maximum over the time period 
                        lab_vals_labdf = lab_group.loc[mask_lab, quantity]
                        target_val = func(lab_vals_labdf)
                    else:
                        target_val = np.nan

                target.append([idx]+[target_val])

        df_target = pd.DataFrame(target, columns=['index', 'target_' + quantity + '_temp']).set_index('index')
        df_treat = df_treat.join(df_target)

        return df_treat

def get_ctcae_labels(df_treat,
                     df_lab,
                     date_col_trt,
                     date_col_lab,
                    ):
    
    CTCAE_keys = list(CTCAE_constants.keys())
    # map elements in blood_lab_list using map_CTCAE_lab
    blood_lab_list = [map_CTCAE_lab[lab] if lab in map_CTCAE_lab else lab 
                      for lab in CTCAE_keys]

    df_treat = get_lookahead_lab_values(df_treat, 
                                        df_lab,
                                        blood_lab_list,
                                        date_col_trt,
                                        date_col_lab,
                                        30)

    for grade_val in [2, 3]:
        for quantity in CTCAE_keys:
            if quantity in ['hemoglobin', 'neutrophil', 'platelet']:
                df_treat[f'target_{quantity}_grade{grade_val}plus'] =\
                    df_treat[f'target_{quantity}_temp'].apply(lambda x:\
                        less_than_thresh(x, quantity, f'grade{grade_val}plus', CTCAE_constants))
            else:
                mapped_quantity = map_CTCAE_lab[quantity]
                df_treat[f'target_{quantity}_grade{grade_val}plus'] =\
                    df_treat.apply(lambda row: compound_thresh(
                        row[f'target_{mapped_quantity}_temp'],
                        row[mapped_quantity],
                        quantity,
                        f'grade{grade_val}plus',
                        CTCAE_constants
                    ), axis=1)

    cols_to_delete = [col for col in df_treat.columns if '_temp' in col]
    df_treat.drop(cols_to_delete, axis=1, inplace=True)

    return df_treat