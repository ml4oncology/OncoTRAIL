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
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from functools import partial
from config import bayesopt_param, model_static_param, model_tuning_param
from util import save_pickle
from sklearn.preprocessing import StandardScaler

def main( trainDataPath, validDataPath, testDataPath, modelName, setupStr, modelDir, resultsDir ):

    # algorithm dictionary
    algs = {
    'LR': LogisticRegression,
    'XGB': XGBClassifier,
    'LGBM': LGBMClassifier
    }

    print('Load train, validation, test data.\n')

    # load train, validation, test data
    with np.load(f'{trainDataPath}') as data:
        X_train = data['embedding']
        Y_train = data['target']
    
    with np.load(f'{validDataPath}') as data:
        X_valid = data['embedding']
        Y_valid = data['target']

    with np.load(f'{testDataPath}') as data:
        X_test = data['embedding']
        Y_test = data['target']

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
        return roc_auc_score(valid_Y, pred)

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
    
    save_pickle(best_params, save_dir=f'{modelDir}', filename=f'{modelName}_{setupStr}')

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
    valid_results = evaluate(model, X_valid, Y_valid)
    test_results = evaluate(model, X_test, Y_test)

    train_results.to_csv(f'{resultsDir}/train_{modelName}_{setupStr}.csv')
    valid_results.to_csv(f'{resultsDir}/valid_{modelName}_{setupStr}.csv')
    test_results.to_csv(f'{resultsDir}/test_{modelName}_{setupStr}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainDataPath", help = "path of training data", type = str) # path of training data
    parser.add_argument("validDataPath", help = "path of validation data", type = str) # path of validation data
    parser.add_argument("testDataPath", help = "path of test data", type = str) # path of training data
    parser.add_argument("modelName", help = "model name", type = str) # name of machine learning model
    parser.add_argument("setupStr", help = "set up string", type = str) # name of set up string
    parser.add_argument("modelDir", help = "model directory", type = str) # directory to save model
    parser.add_argument("resultsDir", help = "results directory", type = str) # directory to save results

    args = parser.parse_args()

    main( args.trainDataPath, args.validDataPath, args.testDataPath, args.modelName, args.setupStr, args.modelDir, args.resultsDir )