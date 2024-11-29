import numpy as np
import pandas as pd
from ml_common.constants import (CANCER_CODE_MAP, 
                                     UNIT_MAP)


def clean_col_name(str_name):
    if str_name == 'num_prior_ED_visits_within_5_years':
        clean_name = 'number of prior ED visits within 5 years'
    elif str_name == 'days_since_prev_ED_visit':
        clean_name = 'days since previous ED visit'
    elif '%_ideal_dose_given' in str_name:
        clean_name = str_name.replace('_',' ').replace('given ', 'planned of ')
    elif str_name == 'female':
        clean_name = 'sex assigned at birth'
    else:
        clean_name = str_name.replace('_',' ')
    
    return clean_name

def add_tabular_data_to_note(clinical_notes_df):

    # reverse the unit dictionary
    reversed_dict = {}

    # Loop through the original dictionary
    for key, values in UNIT_MAP.items():
        for value in values:
            # Add the value as a key in the new dictionary
            # Use a list to hold multiple keys if needed
            if value not in reversed_dict:
                reversed_dict[value] = key

    acute_care_use_cols = ['num_prior_ED_visits_within_5_years', 
                           'days_since_prev_ED_visit']
    cancer_cols = ([col for col in clinical_notes_df.columns 
                   if 'cancer_site' in col and 'other' not in col and 'missing' not in col] +
                  [col for col in clinical_notes_df.columns 
                   if 'morphology' in col and 'other' not in col and 'missing' not in col]
    )

    demographic_cols = ['female', 
                        'age', 
                        'height', 
                        'weight', 
                        'body_surface_area']
    
    laboratory_cols = [
        'alanine_aminotransferase',
        'alanine_aminotransferase_change',
        'albumin',
        'albumin_change',
        'alkaline_phosphatase',
        'alkaline_phosphatase_change',
        'aspartate_aminotransferase',
        'aspartate_aminotransferase_change',
        'creatinine',
        'creatinine_change',
        'glucose',
        'glucose_change',
        'hemoglobin',
        'hemoglobin_change',
        'lactate_dehydrogenase',
        'lactate_dehydrogenase_change',
        'lymphocyte',
        'lymphocyte_change',
        'neutrophil',
        'neutrophil_change',
        'platelet',
        'platelet_change',
        'potassium',
        'potassium_change'
    ]
    symptoms_cols = ([col for col in clinical_notes_df.columns 
                      if 'esas' in col and 'target' not in col and 'missing' not in col] + 
                      ['patient_ecog'] + 
                      [col for col in clinical_notes_df.columns 
                       if 'esas' in col and 'target' not in col and 'change' in col and 'missing' not in col] + 
                       ['patient_ecog_change']
    )

    treatment_cols = (['regimen', 'cycle_number', 'intent', 'line_of_therapy'] +
                       [col for col in clinical_notes_df.columns 
                        if '%_ideal_dose_given' in col and 'missing' not in col]
    )
    
    cols_tabular = (demographic_cols + 
                       acute_care_use_cols + 
                       cancer_cols + 
                       laboratory_cols + 
                       symptoms_cols + 
                       treatment_cols)
    
    cols_dict = {'Demographic': demographic_cols, 
                 'Acute care use': acute_care_use_cols, 
                 'Cancer': cancer_cols, 
                 'Laboratory': laboratory_cols, 
                 'Symptoms': symptoms_cols, 
                 'Treatment': treatment_cols}

    valid_cols_tabular = [col for col in cols_tabular
                           if col in clinical_notes_df.columns]
    df_reduced_cols = clinical_notes_df[valid_cols_tabular].copy()

    sentencized_tabular_data = []
    for _, row in df_reduced_cols.iterrows():
        # create a dataframe with colum names as 1 column
        # and values as another column

        # the idea here is to concatenate the 2 columns in a
        # column name: value format

        row_df = pd.DataFrame({'Column': row.index, 'Value': row.values})
        tabular_data_note = "Below is the patient's tabular data on the treatment date:\n\n"
        for key in cols_dict:
            col_list = cols_dict[key]
            # find column names that are in col_list
            # col_list is a list of column names that Rob wants to be 
            # included in the tabular -> notes
            row_df_temp = row_df.loc[row_df['Column'].isin(col_list)].copy()
            # find columns that are new
            # this is probably obsolete now because this function
            # is only used before we drop column names with high missingness
            unused_cols = list(set(row_df_temp['Column'].unique().tolist()) - set(col_list))
            nan_rows_df = pd.DataFrame({'Column': unused_cols, 'Value': np.nan})
            # if nan, set value to not measured
            row_df_temp = pd.concat([row_df_temp, nan_rows_df], ignore_index=True)
            row_df_temp['Value'].fillna('not measured', inplace=True)
            #row_df_temp = row_df_temp.dropna()
            if row_df_temp.shape[0] == 0: continue
            tabular_data_note = tabular_data_note + key + ':\n'
            if key != 'Cancer':
                row_df_temp['Unit'] = (row_df_temp['Column']
                       .str.replace('_change', '', regex=False)
                       .map(reversed_dict)
                       .fillna(''))
                row_df_temp.loc[row_df_temp['Value'] == 'not measured', 'Unit'] = ''
                # get clean name
                row_df_temp['clean_col_name'] = row_df_temp['Column'].apply(lambda x: clean_col_name(x))
                # adjust sex assigned at birth
                row_df_temp['Value'] = np.where(
                    (row_df_temp['clean_col_name'] == 'sex assigned at birth'),
                    np.where(row_df_temp['Value'] == True, 'female', 'male'),
                    row_df_temp['Value']
                )
                
                row_df_temp.loc[
                    row_df_temp['Column'].str.contains('%_ideal_dose'),
                    'Value'
                ] *= 100

                row_df_temp = row_df_temp[~(
                    (row_df_temp['Column'].str.contains('%_ideal_dose')) &
                    (row_df_temp['Value'] == 0.0)
                )]

                # combine to string
                row_df_temp['combined_string'] = (
                    row_df_temp['clean_col_name'] + ': ' +
                    row_df_temp['Value'].astype(str) + ' ' +
                    row_df_temp['Unit']
                )

            else:
                row_df_temp_TRUE = row_df_temp.loc[row_df_temp['Value'] == True].copy()

                row_df_temp_TRUE['code'] = (
                    row_df_temp_TRUE['Column']
                    .str.replace('cancer_site_', '')
                    .str.replace('morphology_', '')
                )
                
                row_df_temp_TRUE['mapped_code'] = row_df_temp_TRUE['code'].map(CANCER_CODE_MAP)

                row_df_temp_TRUE['name'] = row_df_temp_TRUE['Column'].apply(
                    lambda x: 'cancer site' if 'cancer_site' in x else 'morphology'
                )
                row_df_temp_TRUE = row_df_temp_TRUE.drop_duplicates(subset=['name','mapped_code'])

                # combine to string
                row_df_temp_TRUE['combined_string'] = (
                    row_df_temp_TRUE['name'] + ': ' +
                    row_df_temp_TRUE['mapped_code']
                )

                row_df_temp = row_df_temp_TRUE
            
            # Example output:
            # Demographic:
            # Sex assigned at birth: x
            # Age: x years
            # Height: x cm
            # Weight: x kg
            # Body surface area: x m^2

            # Acute care use:
            # Number of prior ed visits within 5 years: x 
            # Days since previous ed visit: x

            # Cancer:
            # Cancer site: x
            # Morphology: x
            # Morphology: x
            # ...

            # join with new line
            merged_data = row_df_temp['combined_string'].str.capitalize().str.cat(sep='\n') + '\n\n'
            # add to tabular text
            tabular_data_note = tabular_data_note + merged_data
        sentencized_tabular_data.append(tabular_data_note)

    # To-do: fix regimen names

    clinical_notes_df['sentencized_tabular_data'] = sentencized_tabular_data
    clinical_notes_df['note'] = clinical_notes_df['note'] + '\n\n' + clinical_notes_df['sentencized_tabular_data']
    clinical_notes_df.drop('sentencized_tabular_data', axis=1, inplace=True)

    return clinical_notes_df


