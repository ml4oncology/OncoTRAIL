import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import json
import math
import re
import sys
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
from functools import partial
from config import bayesopt_param, model_static_param, model_tuning_param, startTestDate
from util import save_pickle
from sklearn.preprocessing import StandardScaler
from split import genDataSplit

def main( notesPath, embeddingPath, splitConfig, hyperParamEval, modelName, setupStr, modelDir, resultsDir ):

    # save string for file
    file_save_str = f'{modelName}_{setupStr}_{splitConfig}_{hyperParamEval}'

    # algorithm dictionary
    algs = {
    'LR': LogisticRegression,
    'XGB': XGBClassifier,
    'LGBM': LGBMClassifier
    }

    # load data frame
    df = pd.read_csv(f'{notesPath}', index_col=0)
    df.reset_index(drop=True,inplace=True)

    # load embedding
    with np.load(f'{embeddingPath}') as data:
        embedding = data['embedding']
        target = data['target']

    # generate train-validation-test split
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = genDataSplit( df, startTestDate, splitConfig, embedding, target )

    # preprocess the data by scaling and centering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # define helper functions

    def convert_params(params):
        # convert necessary hyperparams to integers
        for param in ['n_estimators', 'max_depth', 'num_leaves', 'min_data_in_leaf', 'min_child_weight', 'bagging_freq']:
            if param in params: params[param] = int(params[param])
        return params

    def eval_func(alg, data, **kwargs):
        train_X, train_Y, valid_X, valid_Y = data
        kwargs = convert_params(kwargs)
        model = algs[alg](**kwargs, **model_static_param[alg])
        model.fit(train_X, train_Y)
        assert model.classes_[1] == 1 # positive class is at index 1
        pred = model.predict_proba(valid_X)[: ,1]

        if hyperParamEval == 'auroc':
            return roc_auc_score(valid_Y, pred)
        elif hyperParamEval == 'logloss':
            return -log_loss(valid_Y, pred)
    
    print('Perform hyperparameter tuning.\n')

    # hyperparameter tuning
    optim_config = bayesopt_param[modelName]
    hyperparam_config = model_tuning_param[modelName]
    data = (X_train, Y_train, X_valid, Y_valid)
    bo = BayesianOptimization(
        f=partial(eval_func, alg=modelName, data=data),
        pbounds=hyperparam_config,
        verbose=2,
        random_state=42
    )
    bo.maximize(**optim_config)
    best_param = bo.max['params']
    best_param = convert_params(best_param)
    best_params = {}
    best_params['params'] = best_param
    
    save_pickle(best_params, save_dir=f'{modelDir}', filename=f'{file_save_str}')

    print('Re-train on best parameters and evaluate the best model.\n')

    # train and evaluate best model
    model = algs[modelName](**best_params[f'params'], **model_static_param[modelName])
    model.fit(X_train, Y_train)

    def evaluate(model, X, Y):
        # check model.classes_ to confirm prediction of positive label is at index 1
        pred = model.predict_proba(X)[: ,1]
        auprc = average_precision_score(Y, pred)
        auroc = roc_auc_score(Y, pred)
        result = {'AUPRC': auprc, 'AUROC': auroc}

        return pd.DataFrame(result, index = [0])
    
    train_results = evaluate(model, X_train, Y_train)
    train_results.rename(index={0:'train'},inplace=True)
    valid_results = evaluate(model, X_valid, Y_valid)
    valid_results.rename(index={0:'valid'},inplace=True)
    test_results = evaluate(model, X_test, Y_test)
    test_results.rename(index={0:'test'},inplace=True)

    results = pd.concat([train_results, valid_results, test_results])
    results.to_csv(f'{resultsDir}/{file_save_str}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notesPath", help = "path of notes", type = str ) # path of notes
    parser.add_argument("embeddingPath", help = "path of embedding", type = str) # path of embedding
    parser.add_argument("splitConfig", help = "configuration of train-valid-test split", type = str) # configuration of train-valid-test split
    parser.add_argument("hyperParamEval", help = "function for hyperparameter evaluation", type = str) # hyperparameter evaluation function
    parser.add_argument("modelName", help = "model name", type = str) # name of machine learning model
    parser.add_argument("setupStr", help = "set up string", type = str) # name of set up string
    parser.add_argument("modelDir", help = "model directory", type = str) # directory to save model
    parser.add_argument("resultsDir", help = "results directory", type = str) # directory to save results

    args = parser.parse_args()

    main( args.notesPath, args.embeddingPath, args.splitConfig, args.hyperParamEval, args.modelName, args.setupStr, args.modelDir, args.resultsDir )