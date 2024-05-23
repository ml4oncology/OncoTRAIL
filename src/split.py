"""
Module to split the data
Kevin He
"""
import logging
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

logger = logging.getLogger(__name__)

def genDataSplit(df, testStartDate, splitConfig, embedding, target):
    
    if 'idx' not in df:
        df['idx'] = df.index
    
    if splitConfig == 'Temporal':
        train_eval_data, valid_data, test_data = create_train_val_test_splits(df, testStartDate)
        # delete notes in train, validation data split after certain date
        train_eval_data = train_eval_data.loc[train_eval_data['treatment_date'] < testStartDate]
        valid_data = valid_data.loc[valid_data['treatment_date'] < testStartDate]

    elif splitConfig == 'Random':
        devt_cohort, test_data = create_random_split(df, test_size=0.35)
        train_eval_data, valid_data = create_random_split(devt_cohort, test_size=0.2)

    train_data, eval_data = create_random_split(train_eval_data, test_size=0.15)

    train_idx = train_data.idx.to_list()
    eval_idx = eval_data.idx.to_list()
    valid_idx = valid_data.idx.to_list()
    test_idx = test_data.idx.to_list()

    X_train = embedding[train_idx,:]
    Y_train = target[train_idx]
    X_eval = embedding[eval_idx,:]
    Y_eval = target[eval_idx]
    X_valid = embedding[valid_idx,:]
    Y_valid = target[valid_idx]
    X_test = embedding[test_idx,:]
    Y_test = target[test_idx]

    return X_train, Y_train, X_eval, Y_eval, X_valid, Y_valid, X_test, Y_test

# Data splitting
def create_train_val_test_splits(
    data: pd.DataFrame, 
    split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the training, validation, and testing set"""
    # split data temporally based on patients first visit date
    train_data, test_data = create_temporal_cohort(data, split_date)
    # create validation set from train data (80-20 split)
    train_data, valid_data = create_random_split(train_data, test_size=0.2)

    # sanity check - make sure there are no overlap of patients in the splits
    assert(not set.intersection(set(train_data['mrn']), set(valid_data['mrn']), set(test_data['mrn'])))
    return train_data, valid_data, test_data

def create_temporal_cohort(
    df: pd.DataFrame, 
    split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the development and testing cohort by partitioning on split_date
    """
    first_date = df.groupby('mrn')['treatment_date'].min()
    first_date = df['mrn'].map(first_date)
    mask = first_date <= split_date
    dev_cohort, test_cohort = df[mask].copy(), df[~mask].copy()
    
    disp = lambda x: f"NSessions={len(x)}. NPatients={x.mrn.nunique()}"
    msg = f"Development Cohort: {disp(dev_cohort)}. Contains all patients whose first visit was on or before {split_date}"
    logger.info(msg)
    msg = f"Test Cohort: {disp(test_cohort)}. Contains all patients whose first visit was after {split_date}"
    logger.info(msg)

    return dev_cohort, test_cohort

def create_random_split(
    df: pd.DataFrame, 
    test_size: float, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data radnomly based on patient ids"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    patient_ids = df['mrn']
    train_idxs, test_idxs = next(gss.split(df, groups=patient_ids))
    train_data = df.iloc[train_idxs].copy()
    test_data = df.iloc[test_idxs].copy()
    return train_data, test_data