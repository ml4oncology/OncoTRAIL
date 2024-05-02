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
from train import Trainer

def main( notesPath, embeddingPath, splitConfig, hyperParamEval, modelName, setupStr, modelDir, resultsDir ):

    # save string for file
    file_save_str = f'{modelName}_{setupStr}_{splitConfig}_{hyperParamEval}'

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

    # call trainer on predictions
    trainer = Trainer(X_train, Y_train, X_valid, Y_valid, X_test, hyperParamEval, modelDir, modelName, file_save_str)
    train_pred, val_pred, test_pred = trainer.run()

    # save data
    np.savez(  f'{resultsDir}/predData_{file_save_str}.npz', train_pred = train_pred, val_pred = val_pred, test_pred = test_pred, Y_train = Y_train, Y_valid = Y_valid, Y_test = Y_test  )

    # evaluate errors
     
    def evaluate(Y, pred):
        auprc = average_precision_score(Y, pred)
        auroc = roc_auc_score(Y, pred)
        LogLoss = log_loss(Y, pred)
        result = {'AUPRC': auprc, 'AUROC': auroc, 'LogLoss': LogLoss}

        return pd.DataFrame(result, index = [0])

    train_results = evaluate(Y_train, train_pred)
    train_results.rename(index={0:'train'},inplace=True)
    valid_results = evaluate(Y_valid, val_pred)
    valid_results.rename(index={0:'valid'},inplace=True)
    test_results = evaluate(Y_test, test_pred)
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